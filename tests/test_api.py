from __future__ import annotations

from fastapi.testclient import TestClient

from doubao_work2api import app as app_module
from doubao_work2api.app import create_app
from doubao_work2api.config import Settings
from doubao_work2api.desktop import DesktopScanResult


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
    model_ids = [item["id"] for item in allowed.json()["data"]]
    assert model_ids[0] == "deepseek-v4.1-flash"
    assert "person-kimi-k3-1-pro-261111" in model_ids
    assert "person-glm-5-3-flash-261111" in model_ids
    assert len(model_ids) == 8


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


def test_chat_rejects_unknown_model(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "not-in-catalog",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


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


def test_web_ui_and_assets_are_served(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        index = client.get("/")
        css = client.get("/ui-assets/app.css")
        script = client.get("/ui-assets/app.js")
    assert index.status_code == 200
    assert "doubaowork2api" in index.text
    assert css.status_code == 200
    assert script.status_code == 200
    assert "Content-Security-Policy" in index.headers
    assert 'id="page-overview"' in index.text
    assert 'id="page-gateway"' in index.text
    assert 'id="page-accounts"' in index.text
    assert 'id="page-analytics"' in index.text
    assert 'id="page-settings"' in index.text
    assert 'id="page-logs"' in index.text


def test_admin_config_exposes_catalog_and_runtime(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.get(
            "/admin/config",
            headers={"Authorization": "Bearer secret"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "0.1.0"
    assert len(body["models"]) == 8
    assert body["models"][0]["upstream_key"] == (
        "person-deepseek-v4-1-flash-261017"
    )
    assert body["listen"] == "127.0.0.1:18789"


def test_import_toggle_and_clear_cooldown(tmp_path) -> None:
    headers = {"Authorization": "Bearer secret"}
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        imported = client.post(
            "/admin/accounts/import",
            headers=headers,
            json={
                "account": {
                    "id": "web-account",
                    "device_id": "device",
                    "web_id": "web",
                    "cookies": {"sessionid": "secret-cookie"},
                }
            },
        )
        assert imported.status_code == 200
        assert imported.json()["imported"] == ["web-account"]

        disabled = client.patch(
            "/admin/accounts/web-account",
            headers=headers,
            json={"enabled": False},
        )
        assert disabled.status_code == 200
        assert disabled.json()["data"]["state"] == "disabled"

        enabled = client.patch(
            "/admin/accounts/web-account",
            headers=headers,
            json={"enabled": True},
        )
        assert enabled.status_code == 200
        assert enabled.json()["data"]["state"] == "ready"

        cleared = client.post(
            "/admin/accounts/web-account/clear-cooldown",
            headers=headers,
        )
        assert cleared.status_code == 200

        accounts = client.get("/admin/accounts", headers=headers).json()["data"]
        assert len(accounts) == 1
        assert "secret-cookie" not in str(accounts)
        saved = (tmp_path / "accounts" / "web-account.json").read_text(
            encoding="utf-8"
        )
        assert "secret-cookie" in saved


def test_import_accepts_account_arrays(tmp_path) -> None:
    headers = {"Authorization": "Bearer secret"}
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.post(
            "/admin/accounts/import",
            headers=headers,
            json={
                "account": [
                    {
                        "id": "one",
                        "cookies": {"sessionid": "a"},
                    },
                    {
                        "id": "two",
                        "cookies": {"sessionid": "b"},
                    },
                ]
            },
        )
    assert response.status_code == 200
    assert response.json()["imported"] == ["one", "two"]


def test_toggle_unknown_account_returns_404(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.patch(
            "/admin/accounts/missing",
            headers={"Authorization": "Bearer secret"},
            json={"enabled": False},
        )
    assert response.status_code in {404, 405}


def test_admin_login_checks_key(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        denied = client.post("/admin/login", json={"api_key": "wrong"})
        allowed = client.post("/admin/login", json={"api_key": "secret"})
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["ok"] is True


def test_scan_desktop_imports_account_without_exposing_cookie(
    tmp_path, monkeypatch
) -> None:
    async def fake_scan(pool):
        account = await pool.upsert(
            {
                "id": "desktop-scan",
                "device_id": "device-scan",
                "web_id": "web-scan",
                "cookies": {"sessionid": "secret-cookie"},
            }
        )
        return DesktopScanResult(
            account=account,
            created=True,
            cookie_count=1,
            source="offline",
        )

    monkeypatch.setattr(app_module, "scan_desktop_account", fake_scan)
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.post(
            "/admin/accounts/scan-offline",
            headers={"Authorization": "Bearer secret"},
        )
    assert response.status_code == 200
    assert response.json()["account"]["id"] == "desktop-scan"
    assert response.json()["source"] == "offline"
    assert "secret-cookie" not in response.text
    saved = (tmp_path / "accounts" / "desktop-scan.json").read_text(
        encoding="utf-8"
    )
    assert "secret-cookie" in saved


def test_removed_online_scan_endpoint_is_not_available(tmp_path) -> None:
    with TestClient(create_app(settings(tmp_path, api_key="secret"))) as client:
        response = client.post(
            "/admin/accounts/scan-desktop",
            headers={"Authorization": "Bearer secret"},
        )
    assert response.status_code in {404, 405}
