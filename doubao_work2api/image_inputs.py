from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from .errors import GatewayError


MAX_IMAGE_BYTES = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 20.0
MAX_REDIRECTS = 3

_DATA_URL_RE = re.compile(
    r"\Adata:(image/(?:png|jpeg|webp|gif));base64,(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)
_HOSTNAME_RE = re.compile(
    r"\A(?=.{1,253}\.?\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.?\Z",
    re.IGNORECASE,
)
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_PUBLIC_SPECIAL_USE = {
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.0.0/29"),
    ipaddress.ip_network("192.0.0.170/31"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("2001:db8::/32"),
}


class InvalidImageInputError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_image_input"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ImageDownloadError(GatewayError):
    status_code = 502
    error_type = "upstream_error"
    code = "image_download_failed"

    def __init__(self) -> None:
        super().__init__("Image download failed.")


@dataclass(frozen=True, slots=True)
class ImageInput:
    data: bytes
    mime_type: str
    filename: str


class Resolver(Protocol):
    async def __call__(self, host: str, port: int) -> list[str]:
        ...


def _invalid(message: str) -> InvalidImageInputError:
    return InvalidImageInputError(message)


def _download_failed() -> ImageDownloadError:
    return ImageDownloadError()


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except (TypeError, ValueError):
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if (
        not address.is_global
        or address.is_private
        or address.is_multicast
        or address.is_reserved
        or address.is_loopback
        or address.is_link_local
        or address.is_unspecified
    ):
        return False
    if any(address in network for network in _PUBLIC_SPECIAL_USE):
        return False
    return True


def _normalize_host(host: str) -> str:
    normalized = host.rstrip(".").lower()
    if not normalized or not _HOSTNAME_RE.fullmatch(normalized):
        raise _invalid("Image URL host is invalid.")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        raise _invalid("Image URL must use a public host.")
    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        literal = None
    if literal is not None and not _is_public_address(normalized):
        raise _invalid("Image URL must use a public host.")
    try:
        normalized.encode("idna")
    except UnicodeError as exc:
        raise _invalid("Image URL host is invalid.") from exc
    return normalized


def _split_public_url(source: str) -> tuple[str, str, int, str]:
    if not isinstance(source, str):
        raise _invalid("Image URL must be a string.")
    try:
        parts = urlsplit(source)
        port = parts.port
    except ValueError as exc:
        raise _invalid("Image URL is invalid.") from exc
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise _invalid("Only HTTP and HTTPS image URLs are supported.")
    if not parts.hostname or "@" in parts.netloc or parts.username or parts.password:
        raise _invalid("Image URL must not contain credentials.")
    host = _normalize_host(parts.hostname)
    if port is None:
        port = 443 if scheme == "https" else 80
    if port not in {80, 443}:
        raise _invalid("Image URL must use port 80 or 443.")
    return scheme, host, port, parts.path or "/"


def validate_image_source(source: str) -> str:
    """Validate an image URL or data URL without network access."""
    if not isinstance(source, str):
        raise _invalid("Image source must be a string.")
    if source[:5].lower() == "data:":
        _decode_data_url(source)
        return "data"
    _split_public_url(source)
    return "url"


def _detect_image(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "gif"
    raise _invalid("Image data is not a supported raster image.")


def _decode_data_url(source: str) -> ImageInput:
    match = _DATA_URL_RE.fullmatch(source)
    if match is None:
        raise _invalid("Only base64 PNG, JPEG, WebP, and GIF data URLs are supported.")
    declared_type = match.group(1).lower()
    encoded = match.group(2)
    max_encoded = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
    if not encoded or len(encoded) > max_encoded:
        raise _invalid("Image data exceeds the 10 MiB limit.")
    if len(encoded) % 4 or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", encoded):
        raise _invalid("Image data URL contains invalid base64.")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid("Image data URL contains invalid base64.") from exc
    if len(decoded) > MAX_IMAGE_BYTES:
        raise _invalid("Image data exceeds the 10 MiB limit.")
    mime_type, extension = _detect_image(decoded)
    if mime_type != declared_type:
        raise _invalid("Image data does not match its declared media type.")
    return ImageInput(decoded, mime_type, f"image.{extension}")


async def _default_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        results = await loop.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except (OSError, UnicodeError) as exc:
        raise _download_failed() from exc
    addresses: list[str] = []
    for result in results:
        address = str(result[4][0])
        if address not in addresses:
            addresses.append(address)
    return addresses


def _pinned_url(url: httpx.URL, address: str, port: int) -> httpx.URL:
    default_port = 443 if url.scheme == "https" else 80
    changes: dict[str, Any] = {"host": address}
    if port != default_port:
        changes["port"] = port
    return url.copy_with(**changes)


async def _resolve_public(host: str, port: int, resolver: Resolver) -> list[str]:
    try:
        addresses = await resolver(host, port)
    except ImageDownloadError:
        raise
    except Exception as exc:
        raise _download_failed() from exc
    if not addresses or not all(_is_public_address(item) for item in addresses):
        raise _download_failed()
    return addresses


async def _close_response(response: httpx.Response) -> None:
    try:
        await response.aclose()
    except Exception:
        pass


async def _read_bounded(response: httpx.Response) -> bytes:
    length = response.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError as exc:
            raise _download_failed() from exc
        if declared < 0 or declared > MAX_IMAGE_BYTES:
            raise _download_failed()
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise _download_failed()
            chunks.append(chunk)
    except ImageDownloadError:
        raise
    except Exception as exc:
        raise _download_failed() from exc
    return b"".join(chunks)


async def load_image(
    source: str,
    *,
    http_client: httpx.AsyncClient | None = None,
    resolver: Resolver | None = None,
) -> ImageInput:
    """Load one OpenAI image_url source without filesystem or logging side effects."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not isinstance(source, str):
        raise _invalid("Image source must be a string.")
    if source[:5].lower() == "data:":
        return _decode_data_url(source)
    _split_public_url(source)
    try:
        return await asyncio.wait_for(
            _load_remote_image(source, http_client=http_client, resolver=resolver),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise _download_failed() from exc


async def _load_remote_image(
    source: str,
    *,
    http_client: httpx.AsyncClient | None,
    resolver: Resolver | None,
) -> ImageInput:
    """Fetch one already-validated URL under a single total deadline."""

    _split_public_url(source)
    resolve = resolver or _default_resolver
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS),
        follow_redirects=False,
        trust_env=False,
        headers={},
        cookies=None,
    )
    try:
        current = httpx.URL(source)
        for redirect_count in range(MAX_REDIRECTS + 1):
            current_scheme, current_host, current_port, _ = _split_public_url(str(current))
            addresses = await _resolve_public(current_host, current_port, resolve)
            request = client.build_request(
                "GET",
                _pinned_url(current, addresses[0], current_port),
                headers={
                    "Host": current_host if current_port == (443 if current_scheme == "https" else 80)
                    else f"{current_host}:{current_port}",
                    "Accept": "image/png,image/jpeg,image/webp,image/gif",
                    "Accept-Encoding": "identity",
                    "User-Agent": "doubaowork2api-image-fetch/1",
                },
                cookies={},
                extensions={"sni_hostname": current_host},
            )
            for header in ("cookie", "authorization", "proxy-authorization"):
                request.headers.pop(header, None)
            response: httpx.Response | None = None
            try:
                response = await client.send(
                    request,
                    stream=True,
                    follow_redirects=False,
                    auth=None,
                )
                if response.status_code in _REDIRECT_STATUSES:
                    if redirect_count == MAX_REDIRECTS:
                        raise _download_failed()
                    location = response.headers.get("location")
                    if not location:
                        raise _download_failed()
                    try:
                        next_url = current.join(location)
                    except (httpx.InvalidURL, ValueError) as exc:
                        raise _download_failed() from exc
                    _split_public_url(str(next_url))
                    current = next_url
                    continue
                if response.status_code != 200:
                    raise _download_failed()
                content_type = response.headers.get("content-type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type not in {
                    "image/png", "image/jpeg", "image/webp", "image/gif",
                    "application/octet-stream",
                }:
                    raise _download_failed()
                data = await _read_bounded(response)
                mime_type, extension = _detect_image(data)
                if media_type != "application/octet-stream" and media_type != mime_type:
                    raise _download_failed()
                return ImageInput(data, mime_type, f"image.{extension}")
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
                raise _download_failed() from exc
            finally:
                if response is not None:
                    await _close_response(response)
    finally:
        if owned_client:
            await client.aclose()
