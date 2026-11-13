from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from doubao_work2api.accounts import Account
from doubao_work2api.errors import UpstreamAuthError
from doubao_work2api.image_inputs import ImageInput
from doubao_work2api import image_upload


URI = "tos-cn-i-test/synthetic.png"
IMAGE = ImageInput(b"\x89PNG\r\n\x1a\nsynthetic", "image/png", "image.png")


async def exercise(monkeypatch, *, commit_status=2000, upload_host="tos-hl-x.snssdk.com", api_status=200, register=True):
    seen_api = []
    seen_storage = []
    client_class = httpx.AsyncClient

    def api_handler(request):
        seen_api.append(request)
        if api_status != 200:
            return httpx.Response(api_status, text="do not echo private account data")
        if request.url.path.endswith("prepare_upload"):
            return httpx.Response(200, json={"code": 0, "data": {
                "service_id": "test-service", "upload_host": "imagex.bytedanceapi.com",
                "upload_auth_token": {"access_key": "test-access", "secret_key": "test-secret", "session_token": "test-sts"},
            }})
        return httpx.Response(200, json={"code": 0, "data": {URI: {"resource_id": "resource-test"}} if register else {}})

    def storage_handler(request):
        seen_storage.append(request)
        assert "cookie" not in request.headers
        assert "x-tt-passport-csrf-token" not in request.headers
        assert "device-private" not in str(request.url)
        if request.url.host == "imagex.bytedanceapi.com":
            assert request.headers["x-amz-security-token"] == "test-sts"
            assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
            if request.url.params["Action"] == "ApplyImageUpload":
                return httpx.Response(200, json={"Result": {"UploadAddress": {
                    "StoreInfos": [{"StoreUri": URI, "Auth": "test-upload-auth"}],
                    "UploadHosts": [upload_host], "SessionKey": "test-session",
                }}})
            assert json.loads(request.content) == {"SessionKey": "test-session"}
            return httpx.Response(200, json={"Result": {"Results": [{"Uri": URI, "UriStatus": commit_status}]}})
        assert request.content == IMAGE.data
        assert request.headers["authorization"] == "test-upload-auth"
        assert len(request.headers["content-crc32"]) == 8
        return httpx.Response(200, json={"code": 2000})

    async with client_class(transport=httpx.MockTransport(api_handler), cookies={"sessionid": "private-account-cookie"}) as api_client:
        async def client_for(account):
            return api_client
        transport = SimpleNamespace(
            settings=SimpleNamespace(upstream_base="https://www.doubao.com"),
            _client_for=client_for, _query=lambda account: "device_id=device-private",
        )
        monkeypatch.setattr(image_upload.httpx, "AsyncClient", lambda **kwargs: client_class(transport=httpx.MockTransport(storage_handler), **kwargs))
        account = Account(id="synthetic", device_id="device-private", web_id="web", cookies={"sessionid": "private-account-cookie", "passport_csrf_token": "csrf-private"})
        result = await image_upload.upload_image(transport, account, IMAGE)
    return result, seen_api, seen_storage


async def test_image_upload_uses_account_api_and_cookie_free_storage(monkeypatch):
    image, api, storage = await exercise(monkeypatch)
    assert len(api) == 2 and len(storage) == 3
    assert all(request.headers["cookie"] == "sessionid=private-account-cookie" for request in api)
    assert json.loads(api[-1].content)["resource_list"][0]["uri"] == URI
    attachment = image.attachment()
    assert attachment["type"] == 1
    assert attachment["image"]["uri"] == URI
    assert attachment["identifier"]


async def test_image_upload_rejects_commit_failure_before_registration(monkeypatch):
    with pytest.raises(image_upload.ImageUploadError, match="commit did not accept"):
        await exercise(monkeypatch, commit_status=5000)


async def test_image_upload_rejects_foreign_storage_host(monkeypatch):
    with pytest.raises(image_upload.ImageUploadError, match="unsupported image upload host"):
        await exercise(monkeypatch, upload_host="attacker.invalid")


async def test_image_upload_preserves_authentication_errors(monkeypatch):
    with pytest.raises(UpstreamAuthError) as error:
        await exercise(monkeypatch, api_status=401)
    assert "private" not in str(error.value)


async def test_image_upload_requires_registered_resource(monkeypatch):
    with pytest.raises(image_upload.ImageUploadError, match="no resource identifier"):
        await exercise(monkeypatch, register=False)
