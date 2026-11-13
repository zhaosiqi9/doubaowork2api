from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from doubao_work2api.app import create_app
from doubao_work2api.config import Settings


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("change,code", [
    ({"reasoning_effort": "ultra"}, "invalid_reasoning_effort"),
    ({"max_tokens": 8}, "unsupported_parameter"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]}]}, "unsupported_content"),
    ({"tools": [{"type": "computer", "name": "desktop"}]}, None),
])
def test_conversion_rejects_unsupported_input_before_borrowing_accounts(tmp_path, stream, change, code):
    app = create_app(Settings(api_key="test-api", accounts_dir=tmp_path / "accounts", data_dir=tmp_path / "data"))
    payload = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hello"}], "stream": stream, **change}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-api"}, json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    if code:
        assert error["code"] == code
    assert app.state.request_log.snapshot() == []


def test_public_models_only_claim_confirmed_capabilities(tmp_path):
    app = create_app(Settings(api_key="test-api", accounts_dir=tmp_path / "accounts", data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer test-api"})
    assert response.status_code == 200
    models = response.json()["data"]
    assert models[0]["context_window"] == 300_000
    assert models[0]["reasoning_efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert all(model["context_window"] is None for model in models[1:])
    assert all(model["reasoning_efforts"] == [] for model in models[1:])


def test_reasoning_validated_after_configured_default_model_is_applied(tmp_path):
    app = create_app(Settings(api_key="test-api", default_model="seed-lite-7b", accounts_dir=tmp_path / "accounts", data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-api"}, json={"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "low"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
