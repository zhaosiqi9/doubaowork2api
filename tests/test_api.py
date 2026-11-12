from __future__ import annotations

from fastapi.testclient import TestClient

from doubao_work2api.app import create_app
from doubao_work2api.config import Settings


def settings(tmp_path, *, api_key: str = "") -> Settings:
    return Settings(
        api_key=api_key,
        accounts_dir=tmp_path / "accounts",
        data_dir=tmp_path / "data",
        request_timeout_seconds=5,
    )


def test_health_and_models(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        assert client.get("/health").status_code == 200
        denied = client.get("/v1/models")
        allowed = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert [item["id"] for item in allowed.json()["data"]] == [
        "deepseek-v4.1-flash"
    ]


def test_chat_without_account_returns_clear_error(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_available_account"


def test_admin_accounts_does_not_expose_cookies(tmp_path) -> None:
    accounts = tmp_path / "accounts"
    accounts.mkdir()
    (accounts / "one.json").write_text(
        '{"id":"one","cookie":"sessionid=secret","device_id":"d","web_id":"w"}',
        encoding="utf-8",
    )
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.get(
            "/admin/accounts",
            headers={"Authorization": "Bearer secret"},
        )
    body = response.text
    assert response.status_code == 200
    assert "sessionid" not in body
    assert "secret" not in body
