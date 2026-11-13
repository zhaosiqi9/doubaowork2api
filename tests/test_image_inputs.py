from __future__ import annotations

import asyncio
import base64

import httpx
import pytest

from doubao_work2api.image_inputs import (
    MAX_IMAGE_BYTES,
    ImageDownloadError,
    InvalidImageInputError,
    load_image,
    validate_image_source,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"test-png"
JPEG = b"\xff\xd8\xff\xe0" + b"test-jpeg"
WEBP = b"RIFF\x04\x00\x00\x00WEBP"
GIF = b"GIF89a" + b"test-gif"


def data_url(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


async def public_resolver(host: str, port: int) -> list[str]:
    return ["93.184.216.34"]


async def test_valid_data_urls_and_magic_detection() -> None:
    png = await load_image(data_url(PNG))
    assert (png.data, png.mime_type, png.filename) == (PNG, "image/png", "image.png")
    jpeg = await load_image(data_url(JPEG, "image/jpeg"))
    assert (jpeg.data, jpeg.mime_type, jpeg.filename) == (JPEG, "image/jpeg", "image.jpg")
    webp = await load_image(data_url(WEBP, "image/webp"))
    assert (webp.data, webp.mime_type, webp.filename) == (WEBP, "image/webp", "image.webp")
    gif = await load_image(data_url(GIF, "image/gif"))
    assert (gif.data, gif.mime_type, gif.filename) == (GIF, "image/gif", "image.gif")


@pytest.mark.parametrize(
    "source",
    [
        "data:image/svg+xml;base64,PHN2Zy8+",
        "data:text/html;base64,PGh0bWw+",
        "data:image/png;base64,",
        "data:image/png;base64,AAAA",
        data_url(b"<svg/>", "image/png"),
        data_url(PNG, "image/jpeg"),
        "data:image/png;base64,PHN2Zy8+",
        "data:image/png;base64,AA=A",
        "data:image/png;base64,AAA",
        "file:///tmp/image.png",
        "ftp://example.com/image.png",
        "https://user:pass@example.com/image.png",
        "https://example.com:8443/image.png",
        "https://localhost/image.png",
        "https://127.0.0.1/image.png",
    ],
)
def test_invalid_sources_are_rejected_without_network(source: str) -> None:
    with pytest.raises(InvalidImageInputError) as caught:
        validate_image_source(source)
    assert caught.value.code == "invalid_image_input"
    assert caught.value.status_code == 400


def test_data_url_size_preflight_rejects_before_decode() -> None:
    oversized = "A" * (4 * ((MAX_IMAGE_BYTES + 2) // 3) + 4)
    with pytest.raises(InvalidImageInputError, match="10 MiB"):
        validate_image_source(f"data:image/png;base64,{oversized}")


async def test_public_url_is_pinned_to_resolved_ip_with_original_host() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=PNG,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        image = await load_image(
            "https://images.example.test/a.png?token=secret",
            http_client=client,
            resolver=public_resolver,
        )
    assert image.data == PNG
    assert len(seen) == 1
    request = seen[0]
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "images.example.test"
    assert request.extensions["sni_hostname"] == "images.example.test"
    assert "cookie" not in request.headers
    assert "authorization" not in request.headers


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["100.64.0.1"],
        ["93.184.216.34", "10.0.0.1"],
        ["224.0.0.1"],
        ["::1"],
        ["fc00::1"],
        ["ff02::1"],
        ["::ffff:127.0.0.1"],
    ],
)
async def test_private_or_mixed_dns_is_rejected_before_request(addresses) -> None:
    requests: list[httpx.Request] = []

    async def resolver(host: str, port: int) -> list[str]:
        return addresses

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: requests.append(request)),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError) as caught:
            await load_image("https://images.example.test/a.png", http_client=client, resolver=resolver)
    assert caught.value.code == "image_download_failed"
    assert caught.value.status_code == 502
    assert requests == []


async def test_redirect_to_private_address_is_rejected() -> None:
    requests: list[httpx.Request] = []
    resolution = {
        "images.example.test": ["93.184.216.34"],
        "private.example.test": ["10.0.0.8"],
    }

    async def resolver(host: str, port: int) -> list[str]:
        return resolution[host]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://private.example.test/a.png"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError):
            await load_image("https://images.example.test/a.png", http_client=client, resolver=resolver)
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"


async def test_three_redirects_are_allowed_and_fourth_is_rejected() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) <= 3:
            return httpx.Response(302, headers={"location": f"/{len(requests) + 1}"})
        return httpx.Response(200, headers={"content-type": "image/png"}, content=PNG)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        image = await load_image("https://images.example.test/0", http_client=client, resolver=public_resolver)
    assert image.data == PNG
    assert len(requests) == 4

    requests.clear()

    def too_many(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"location": "/again"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(too_many),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError):
            await load_image("https://images.example.test/0", http_client=client, resolver=public_resolver)
    assert len(requests) == 4


async def test_streamed_body_over_limit_is_rejected() -> None:
    async def chunks():
        yield PNG
        yield b"x" * (MAX_IMAGE_BYTES + 1)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=chunks(),
            )
        ),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError):
            await load_image("https://images.example.test/a.png", http_client=client, resolver=public_resolver)


async def test_declared_oversize_and_content_type_mismatch_are_rejected() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={
                    "content-type": "image/png",
                    "content-length": str(MAX_IMAGE_BYTES + 1),
                },
                content=b"",
            )
        ),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError):
            await load_image("https://images.example.test/a.png", http_client=client, resolver=public_resolver)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<html></html>",
            )
        ),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError):
            await load_image("https://images.example.test/a.png", http_client=client, resolver=public_resolver)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=PNG,
            )
        ),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError):
            await load_image("https://images.example.test/a.png", http_client=client, resolver=public_resolver)


async def test_octet_stream_uses_magic_type() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=PNG,
            )
        ),
        trust_env=False,
    ) as client:
        image = await load_image(
            "https://images.example.test/a.png",
            http_client=client,
            resolver=public_resolver,
        )
    assert image.mime_type == "image/png"


async def test_injected_client_credentials_and_redirect_cookie_are_not_sent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={
                    "location": "/next",
                    "set-cookie": "private_cookie=server-secret; Path=/",
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=PNG,
        )

    async with httpx.AsyncClient(
        auth=("default-user", "default-password"),
        headers={
            "Authorization": "Bearer private-api-key",
            "Proxy-Authorization": "Basic private-proxy",
            "Cookie": "private_cookie=client-secret",
        },
        cookies={"private_cookie": "client-secret"},
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        image = await load_image(
            "https://images.example.test/a.png",
            http_client=client,
            resolver=public_resolver,
        )
    assert image.data == PNG
    assert len(requests) == 2
    for request in requests:
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers
        assert "proxy-authorization" not in request.headers


async def test_download_errors_do_not_echo_url() -> None:
    secret = "https://images.example.test/a.png?signature=private-token"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
        trust_env=False,
    ) as client:
        with pytest.raises(ImageDownloadError) as caught:
            await load_image(secret, http_client=client, resolver=public_resolver)
    assert "private-token" not in str(caught.value)
    assert "images.example.test" not in str(caught.value)


async def test_total_download_deadline_uses_wait_for(monkeypatch) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def resolver(host: str, port: int) -> list[str]:
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ["93.184.216.34"]

    original_wait_for = asyncio.wait_for
    timeouts: list[float] = []

    async def immediate_timeout(awaitable, timeout):
        timeouts.append(timeout)
        task = asyncio.create_task(awaitable)
        await entered.wait()
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", immediate_timeout)
    try:
        with pytest.raises(ImageDownloadError):
            await load_image(
                "https://images.example.test/a.png",
                resolver=resolver,
            )
    finally:
        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)
    assert timeouts == [20.0]
    assert cancelled.is_set()


def test_chat_completion_image_url_shapes() -> None:
    assert validate_image_source(data_url(PNG)) == "data"
    assert validate_image_source("https://images.example.test/a.png") == "url"
