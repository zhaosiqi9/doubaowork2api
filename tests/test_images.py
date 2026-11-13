from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from doubao_work2api import service as service_module
from doubao_work2api.accounts import Account, AccountPool
from doubao_work2api.app import create_app
from doubao_work2api.config import Settings
from doubao_work2api.errors import NoAvailableAccountError, UpstreamRateLimitError
from doubao_work2api.image_inputs import ImageInput
from doubao_work2api.image_upload import ImageUploadError, UploadedImage
from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.protocol import render_prompt, validate_request, upstream_payload
from doubao_work2api.service import ChatService


PNG = b"\x89PNG\r\n\x1a\nsynthetic-test"
SOURCE = "data:image/png;base64," + base64.b64encode(PNG).decode()


def image_part(source=SOURCE):
    return {"type": "image_url", "image_url": {"url": source}}


def request(**fields):
    return ChatCompletionRequest(**{"messages": [{"role": "user", "content": [
        {"type": "text", "text": "描述图片"}, image_part(),
    ]}], **fields})


def make_service(tmp_path, transport):
    pool = AccountPool(tmp_path / "accounts", cooldown_seconds=10)
    pool._accounts = [Account(id=key, cookies={"sessionid": "synthetic"}, device_id=key, web_id="web") for key in ("one", "two")]
    return ChatService(pool, transport, model_key="unused", app_id="app", workspace_root=tmp_path / "work")


class Transport:
    def __init__(self, failure=None):
        self.payloads = []
        self.failure = failure

    async def stream(self, account, payload):
        self.payloads.append((account.id, payload))
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        yield {"_event": "CHUNK_DELTA", "text": "synthetic image answer"}


def test_image_markers_preserve_message_and_part_order_without_urls():
    chat = request(messages=[
        {"role": "user", "content": [{"type": "text", "text": "first"}, image_part(), {"type": "text", "text": "after"}]},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": [image_part(), {"type": "text", "text": "compare"}]},
    ])
    validate_request(chat)
    prompt = render_prompt(chat.messages)
    assert "User: first\n[Attached image 1]\nafter" in prompt
    assert "User: [Attached image 2]\ncompare" in prompt
    assert "base64" not in prompt and SOURCE not in prompt


def test_payload_requires_uploaded_images_and_builds_native_attachments(tmp_path):
    chat = request()
    options = dict(model_key="person-deepseek-v4-1-flash-261017", workspace=tmp_path, device_name="TEST", app_id="app", device_id="device")
    with pytest.raises(Exception) as error:
        upstream_payload(chat, **options)
    assert error.value.code == "images_not_prepared"
    payload = upstream_payload(chat, **options, images=[UploadedImage("test/one.png", "image-one", "image-1.png", "image/png")])
    blocks = payload["messages"][0]["content_block"]
    assert blocks[0]["block_type"] == 10052
    assert blocks[0]["content"]["attachment_block"]["attachments"][0]["image"]["uri"] == "test/one.png"
    assert SOURCE not in json.dumps(payload)


@pytest.mark.parametrize("stream", [False, True])
async def test_image_service_uploads_to_selected_account_and_supports_both_modes(tmp_path, monkeypatch, stream):
    uploaded = []

    async def fake_upload(transport, account, image, *, filename=None):
        uploaded.append((account.id, image, filename))
        return UploadedImage(f"test/{account.id}.png", "identifier", filename, image.mime_type)

    monkeypatch.setattr(service_module, "upload_image", fake_upload)
    transport = Transport()
    service = make_service(tmp_path, transport)
    chat = request(stream=stream)
    if stream:
        result = [chunk async for chunk in service.stream(chat, allowed_ids={"two"})]
        assert result[-1] == "data: [DONE]\n\n"
        assert "synthetic image answer" in "".join(result)
    else:
        result = await service.complete(chat, allowed_ids={"two"})
        assert result["choices"][0]["message"]["content"] == "synthetic image answer"
        assert result["usage"]["is_estimated"] is True
        # The gateway budgets an image allowance, rather than its base64 text.
        assert result["usage"]["prompt_tokens"] >= 4096
    assert len(uploaded) == 1 and uploaded[0][0] == "two" and uploaded[0][1].data == PNG
    account_id, payload = transport.payloads[0]
    assert account_id == "two"
    assert "test/two.png" in json.dumps(payload)
    assert SOURCE not in json.dumps(payload)
    assert all(account.in_flight == 0 for account in service.pool.accounts)


async def test_account_retry_uploads_again_in_new_account_scope(tmp_path, monkeypatch):
    uploaded = []

    async def fake_upload(transport, account, image, *, filename=None):
        uploaded.append(account.id)
        return UploadedImage(f"test/{account.id}.png", account.id, filename, image.mime_type)

    monkeypatch.setattr(service_module, "upload_image", fake_upload)
    transport = Transport(UpstreamRateLimitError("synthetic rate limit"))
    await make_service(tmp_path, transport).complete(request())
    assert uploaded == ["one", "two"]
    assert "test/one.png" in json.dumps(transport.payloads[0][1])
    assert "test/two.png" in json.dumps(transport.payloads[1][1])
    assert "test/one.png" not in json.dumps(transport.payloads[1][1])


async def test_image_upload_failure_does_not_rotate_or_disable_accounts(tmp_path, monkeypatch):
    called = []

    async def broken_upload(transport, account, image, **kwargs):
        called.append(account.id)
        raise ImageUploadError("Synthetic upload failure.")

    monkeypatch.setattr(service_module, "upload_image", broken_upload)
    service = make_service(tmp_path, Transport())
    with pytest.raises(ImageUploadError):
        await service.complete(request())
    assert called == ["one"]
    assert all(account.available and account.in_flight == 0 for account in service.pool.accounts)


async def test_total_image_size_rejected_before_account_upload(tmp_path, monkeypatch):
    async def fake_load(source):
        return ImageInput(b"x" * (8 * 1024 * 1024), "image/png", "image.png")

    monkeypatch.setattr(service_module, "load_image", fake_load)
    transport = Transport()
    service = make_service(tmp_path, transport)
    with pytest.raises(Exception) as error:
        await service.complete(request(messages=[{"role": "user", "content": [image_part()] * 3}]))
    assert error.value.status_code == 400
    assert transport.payloads == []
    assert all(account.available and account.in_flight == 0 for account in service.pool.accounts)


async def test_no_available_account_never_downloads_remote_image(tmp_path, monkeypatch):
    async def forbidden_load(source):
        raise AssertionError("No image download should happen")

    monkeypatch.setattr(service_module, "load_image", forbidden_load)
    service = make_service(tmp_path, Transport())
    with pytest.raises(NoAvailableAccountError):
        await service.complete(request(), allowed_ids=set())


@pytest.mark.parametrize("change", [
    {"messages": [{"role": "user", "content": [image_part("http://127.0.0.1/private.png")]}]},
    {"messages": [{"role": "user", "content": [image_part("data:image/png;base64,not-valid")]}]},
    {"messages": [{"role": "assistant", "content": [image_part()]}]},
    {"messages": [{"role": "user", "content": [image_part()] * 9}]},
])
def test_bad_images_are_rejected_before_dispatch_including_streams(tmp_path, change):
    app = create_app(Settings(api_key="synthetic-key", accounts_dir=tmp_path / "accounts", data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={**change, "stream": True}, headers={"Authorization": "Bearer synthetic-key"})
    assert response.status_code == 400
    assert app.state.request_log.snapshot() == []
