"""ImageX upload protocol used by the Doubao desktop image attachment flow."""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from urllib.parse import quote, urlencode, urlsplit

import httpx

from .errors import UpstreamError

if TYPE_CHECKING:
    from .accounts import Account
    from .image_inputs import ImageInput
    from .transport import DirectTransport


@dataclass(frozen=True)
class UploadedImage:
    uri: str
    identifier: str
    filename: str
    mime_type: str
    width: int = 0
    height: int = 0

    def attachment(self) -> dict[str, Any]:
        return {
            "type": 1,
            "identifier": self.identifier,
            "image": {
                "name": self.filename, "uri": self.uri,
                "image_ori": {
                    "url": "", "width": self.width, "height": self.height,
                    "format": self.mime_type.split("/")[1], "url_formats": {},
                },
            },
            "parse_state": 1, "review_state": 1,
            "upload_status": 1, "progress": 100, "src": "",
        }


class ImageUploadError(UpstreamError):
    code = "image_upload_failed"


def _allowed_upload_host(host: str, *, imagex: bool = False) -> str:
    if not isinstance(host, str) or "/" in host or "@" in host or ":" in host:
        raise ImageUploadError("Doubao returned an invalid image upload host.")
    name = host.lower()
    if imagex:
        allowed = name in {"imagex.bytedanceapi.com", "imagex.volcengineapi.com"}
    else:
        allowed = any(name.endswith("." + suffix) for suffix in ("snssdk.com", "byteimg.com", "volces.com"))
    if not allowed:
        raise ImageUploadError("Doubao returned an unsupported image upload host.")
    return "https://" + name


def _upload_result(response: httpx.Response, stage: str) -> dict[str, Any]:
    if response.status_code != 200:
        raise ImageUploadError(f"Doubao image {stage} failed (HTTP {response.status_code}).")
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ImageUploadError(f"Doubao image {stage} returned invalid JSON.") from exc
    if not isinstance(body, dict):
        raise ImageUploadError(f"Doubao image {stage} returned an invalid response.")
    metadata = body.get("ResponseMetadata") or {}
    if not isinstance(metadata, dict) or metadata.get("Error"):
        raise ImageUploadError(f"Doubao image {stage} was rejected.")
    return body


async def upload_image(
    transport: DirectTransport, account: Account, image: ImageInput,
    *, filename: str | None = None,
) -> UploadedImage:
    """Upload one image and register it for this exact account.

    The account session is used exclusively for Doubao API calls. ImageX and
    storage hosts receive only short-lived upload authorization, never cookies.
    """
    from .transport import USER_AGENT, classify_body_error, classify_status

    client = await transport._client_for(account)
    base = transport.settings.upstream_base
    query = transport._query(account)
    headers = {
        "User-Agent": USER_AGENT, "Origin": base, "Referer": base + "/chat/",
        "agw-js-conv": "str",
    }
    csrf = account.cookies.get("passport_csrf_token") or account.cookies.get("passport_csrf_token_default")
    if csrf:
        headers["x-tt-passport-csrf-token"] = csrf

    async def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await client.post(base + path + "?" + query, headers=headers, json=payload, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ImageUploadError("Doubao image API connection failed.") from exc
        classify_status(response.status_code, "Image API request failed.")
        body = _upload_result(response, "API request")
        classify_body_error(body)
        data = body.get("data")
        if not isinstance(data, dict):
            raise ImageUploadError("Doubao image API returned no resource data.")
        return data

    prepared = await api_post("/alice/resource/prepare_upload", {
        "tenant_id": "5", "scene_id": "5", "resource_type": 2,
    })
    try:
        imagex_base = _allowed_upload_host(prepared["upload_host"], imagex=True)
        credentials = prepared["upload_auth_token"]
        service_id = prepared["service_id"]
        if not all(isinstance(credentials.get(key), str) and credentials[key] for key in ("access_key", "secret_key", "session_token")):
            raise KeyError("credentials")
    except (KeyError, AttributeError, TypeError) as exc:
        raise ImageUploadError("Doubao image API returned incomplete upload credentials.") from exc
    name = filename or image.filename
    extension = "." + name.rsplit(".", 1)[-1]
    params = {
        "Action": "ApplyImageUpload", "Version": "2018-08-01",
        "ServiceId": service_id, "FileSize": str(len(image.data)),
        "FileExtension": extension, "s": uuid.uuid4().hex[:12], "device_platform": "web",
    }
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as storage:
            response = await storage.get(imagex_base + "/?" + imagex_query(params), headers=sign_imagex_request("GET", params, credentials))
            applied = _upload_result(response, "upload preparation")
            try:
                address = applied["Result"]["UploadAddress"]
                info = address["StoreInfos"][0]
                upload_base = _allowed_upload_host(address["UploadHosts"][0])
                uri = info["StoreUri"]
                session_key = address["SessionKey"]
                upload_auth = info["Auth"]
                if not all(isinstance(value, str) and value for value in (uri, session_key, upload_auth)):
                    raise KeyError("upload address")
                if uri.startswith("/") or ".." in uri.split("/") or urlsplit(uri).scheme:
                    raise KeyError("uri")
            except (KeyError, IndexError, TypeError) as exc:
                raise ImageUploadError("Doubao returned an incomplete image upload address.") from exc
            response = await storage.post(
                upload_base + "/upload/v1/" + quote(uri, safe="/"), content=image.data,
                headers={
                    "Authorization": upload_auth,
                    "Content-CRC32": f"{zlib.crc32(image.data) & 0xffffffff:08x}",
                    "Content-Type": "application/octet-stream",
                },
            )
            uploaded = _upload_result(response, "transfer")
            if str(uploaded.get("code", 2000)) not in {"0", "2000"}:
                raise ImageUploadError("Doubao image transfer was rejected.")
            params = {"Action": "CommitImageUpload", "Version": "2018-08-01", "ServiceId": service_id}
            body = json_bytes({"SessionKey": session_key})
            signed = sign_imagex_request("POST", params, credentials, body)
            signed["Content-Type"] = "application/json"
            response = await storage.post(imagex_base + "/?" + imagex_query(params), headers=signed, content=body)
            committed = _upload_result(response, "commit")
            if not isinstance(committed.get("Result"), dict):
                raise ImageUploadError("Doubao image commit returned no result.")
            results = committed["Result"].get("Results")
            if not isinstance(results, list) or not results or any(not isinstance(item, dict) or item.get("UriStatus") != 2000 for item in results):
                raise ImageUploadError("Doubao image commit did not accept the image.")
    except httpx.HTTPError as exc:
        raise ImageUploadError("Doubao image upload connection failed.") from exc
    identifier = uuid.uuid4().hex
    digest = hashlib.md5(image.data).hexdigest()
    registered = await api_post("/alice/resource/register", {
        "tenant_id": "5", "scene_id": "5",
        "resource_list": [{
            "uri": uri, "resource_type": 2,
            "meta": {"image_meta": {
                "data": {"md5": digest, "identifier": identifier},
                "parse_extra": {},
                "image_info": {"file_name": name, "format": image.mime_type.split("/")[1], "size": len(image.data)},
            }},
        }],
    })
    record = registered.get(uri)
    if not isinstance(record, dict) or not record.get("resource_id"):
        raise ImageUploadError("Doubao image registration returned no resource identifier.")
    return UploadedImage(uri=uri, identifier=identifier, filename=name, mime_type=image.mime_type)


def sign_imagex_request(
    method: str,
    params: dict[str, str],
    credentials: dict[str, str],
    body: bytes = b"",
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    """Sign the ImageX request with short-lived upload STS credentials.

    This is the standard AWS4 signing scheme used by the packaged upload SDK,
    not a browser signature. Account cookies are never part of these requests.
    """
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    day = timestamp[:8]
    scope = f"{day}/cn-north-1/imagex/aws4_request"
    headers = {
        "x-amz-date": timestamp,
        "x-amz-security-token": credentials["session_token"],
    }
    if body:
        headers["x-amz-content-sha256"] = hashlib.sha256(body).hexdigest()
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
    query = imagex_query(params)
    canonical = "\n".join([
        method.upper(), "/", query, canonical_headers, signed_headers,
        hashlib.sha256(body).hexdigest(),
    ])
    to_sign = "\n".join([
        "AWS4-HMAC-SHA256", timestamp, scope,
        hashlib.sha256(canonical.encode()).hexdigest(),
    ])
    key = ("AWS4" + credentials["secret_key"]).encode()
    for part in (day, "cn-north-1", "imagex", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={credentials['access_key']}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


def imagex_query(params: dict[str, str]) -> str:
    return urlencode(sorted(params.items()), quote_via=quote, safe="~")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
