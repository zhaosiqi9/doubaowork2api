import asyncio
import json

import httpx
import pytest

from fastapi.testclient import TestClient

from doubao_work2api.app import SESSION_COOKIE, create_app
from doubao_work2api.config import Settings


def configured(tmp_path):
    return Settings(api_key="test-api-key", admin_password="test-admin-password",
                    accounts_dir=tmp_path / "accounts", data_dir=tmp_path / "data")


def login(client):
    assert client.post("/admin/login", json={"password": "test-admin-password"}).status_code == 200


def add_account(client, account_id):
    result = client.post("/admin/accounts/import", json={"account": {
        "id": account_id, "device_id": "device-" + account_id, "web_id": "web",
        "cookies": {"sessionid": "credential-" + account_id},
    }})
    assert result.status_code == 200


def test_admin_and_api_permissions_are_separate(tmp_path):
    with TestClient(create_app(configured(tmp_path))) as client:
        assert client.get("/admin/accounts", headers={"Authorization": "Bearer test-api-key"}).status_code == 401
        assert client.post("/admin/login", json={"password": "test-api-key"}).status_code == 401
        login(client)
        assert client.get("/admin/accounts").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer test-admin-password"}).status_code == 401
        assert client.get("/v1/models", headers={"x-api-key": "test-api-key"}).status_code == 200


def test_session_cookie_logout_and_cross_site_mutation(tmp_path):
    with TestClient(create_app(configured(tmp_path))) as client:
        response = client.post("/admin/login", json={"password": "test-admin-password"})
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=strict" in response.headers["set-cookie"]
        token = client.cookies.get(SESSION_COOKIE)
        assert client.post("/admin/keys", headers={"Origin": "https://evil.example"}, json={"name": "bad"}).status_code == 403
        assert client.post("/admin/login", headers={"Origin": "null"}, json={"password": "test-admin-password"}).status_code == 403
        assert client.post("/admin/logout").status_code == 200
        assert client.get("/admin/config").status_code == 401
        client.cookies.set(SESSION_COOKIE, token, path="/admin")
        assert client.get("/admin/config").status_code == 401


def test_password_change_revokes_existing_sessions_and_survives_restart(tmp_path):
    config = configured(tmp_path)
    app = create_app(config)
    with TestClient(app) as first, TestClient(app) as second:
        login(first)
        login(second)
        result = first.post("/admin/password", json={
            "current_password": "test-admin-password", "new_password": "changed-password",
        })
        assert result.status_code == 200
        assert second.get("/admin/config").status_code == 401
    with TestClient(create_app(config)) as client:
        assert client.post("/admin/login", json={"password": "test-admin-password"}).status_code == 401
        assert client.post("/admin/login", json={"password": "changed-password"}).status_code == 200


def test_key_lifecycle_model_access_and_no_secret_in_lists(tmp_path):
    config = configured(tmp_path)
    with TestClient(create_app(config)) as client:
        login(client)
        add_account(client, "one")
        created = client.post("/admin/keys", json={
            "name": "limited", "models": ["seed-lite-7b"],
            "accounts": ["one"], "concurrency_limit": 1,
        }).json()
        secret, key_id = created["key"], created["id"]
        headers = {"Authorization": "Bearer " + secret}
        listing = client.get("/admin/keys")
        assert secret not in listing.text and "test-api-key" not in listing.text
        assert "hash" not in listing.text
        assert [item["id"] for item in client.get("/v1/models", headers=headers).json()["data"]] == ["seed-lite-7b"]
        denied = client.post("/v1/chat/completions", headers=headers, json={
            "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hello"}],
        })
        assert denied.status_code == 403
        assert client.patch("/admin/keys/" + key_id, json={"enabled": False}).status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == 401
        assert client.patch("/admin/keys/" + key_id, json={"enabled": True}).status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert client.delete("/admin/keys/" + key_id).status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == 401
    stored = (config.data_dir / "settings.json").read_text(encoding="utf-8")
    assert secret not in stored and "test-admin-password" not in stored and "test-api-key" not in stored


def test_runtime_save_applies_and_survives_restart(tmp_path):
    config = configured(tmp_path)
    app = create_app(config)
    with TestClient(app) as client:
        login(client)
        saved = client.put("/admin/config", json={
            "default_model": "seed-lite-7b", "timeout_seconds": 32, "cooldown_seconds": 21,
        })
        assert saved.status_code == 200
        assert app.state.settings.default_model == "seed-lite-7b"
        assert app.state.pool.cooldown_seconds == 21
        assert client.put("/admin/config", json={"timeout_seconds": -1}).status_code == 400
        assert client.put("/admin/config", json={"default_model": "made-up"}).status_code == 400
        assert client.put("/admin/config", json={"upstream": "https://untrusted.example"}).status_code == 400
    reloaded = create_app(config)
    assert reloaded.state.settings.default_model == "seed-lite-7b"
    assert reloaded.state.settings.request_timeout_seconds == 32


def test_account_batch_export_import_round_trip(tmp_path):
    config = configured(tmp_path)
    with TestClient(create_app(config)) as client:
        login(client)
        add_account(client, "one")
        add_account(client, "two")
        assert client.patch("/admin/accounts/one", json={"name": "First", "remark": "Note", "group": "A"}).status_code == 200
        assert client.post("/admin/accounts/batch", json={"ids": ["one"], "action": "disable"}).json()["succeeded"] == ["one"]
        assert client.post("/admin/accounts/batch", json={"ids": ["one", "two"], "action": "group", "group": "shared"}).status_code == 200
        exported = client.post("/admin/accounts/export", json={"ids": ["one"]})
        assert exported.status_code == 200 and exported.headers["cache-control"] == "no-store"
        document = exported.json()
        assert len(document["accounts"]) == 1
        assert document["accounts"][0]["cookies"]["sessionid"] == "credential-one"
        assert "credential-one" not in client.get("/admin/accounts").text
        assert client.post("/admin/accounts/batch", json={"ids": ["one"], "action": "delete"}).status_code == 200
        assert client.post("/admin/accounts/import", json=document).status_code == 200
        one = next(item for item in client.get("/admin/accounts").json()["data"] if item["id"] == "one")
        assert not one["enabled"] and one["name"] == "First" and one["group"] == "shared"


def test_key_account_binding_constrains_actual_dispatch_and_default_model(tmp_path):
    app = create_app(configured(tmp_path))
    calls = []
    class Transport:
        async def stream(self, account, payload):
            calls.append(account.id)
            yield {"_event": "CHUNK_DELTA", "text": "OK"}
    app.state.service.transport = Transport()
    with TestClient(app) as client:
        login(client)
        add_account(client, "one")
        add_account(client, "two")
        created = client.post("/admin/keys", json={"name": "bound", "accounts": ["two"]}).json()
        headers = {"x-api-key": created["key"]}
        assert client.put("/admin/config", json={"default_model": "seed-lite-7b"}).status_code == 200
        result = client.post("/v1/chat/completions", headers=headers,
                             json={"messages": [{"role": "user", "content": "hello"}]})
        assert result.status_code == 200
        assert result.json()["model"] == "seed-lite-7b"
        assert calls == ["two"]
        client.post("/admin/accounts/batch", json={"ids": ["two"], "action": "disable"})
        assert client.post("/v1/chat/completions", headers=headers,
                           json={"messages": [{"role": "user", "content": "hello"}]}).status_code == 503
        assert calls == ["two"]
        log = client.get("/admin/logs").json()["data"]
        assert log[0]["meta"]["key_id"] == created["id"]


def test_invalid_key_refs_and_no_selection_are_rejected(tmp_path):
    with TestClient(create_app(configured(tmp_path))) as client:
        login(client)
        assert client.post("/admin/keys", json={"accounts": ["missing"]}).status_code == 400
        assert client.post("/admin/keys", json={"models": ["unknown"]}).status_code == 400
        assert client.post("/admin/accounts/export", json={"ids": []}).status_code == 422
        assert client.post("/admin/accounts/batch", json={"ids": [], "action": "delete"}).status_code == 422


@pytest.mark.asyncio
async def test_key_concurrency_limit_releases_after_completion_and_failure(tmp_path):
    app = create_app(configured(tmp_path))
    entered, finish = asyncio.Event(), asyncio.Event()
    fail = False
    async def complete(payload, **options):
        entered.set()
        await finish.wait()
        if fail:
            from doubao_work2api.errors import UpstreamError
            raise UpstreamError("test failure")
        return {"ok": True}
    app.state.service.complete = complete
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        await client.post("/admin/login", json={"password": "test-admin-password"})
        key = (await client.post("/admin/keys", json={"name": "one", "concurrency_limit": 1})).json()
        body = {"messages": [{"role": "user", "content": "hello"}]}
        headers = {"x-api-key": key["key"]}
        first = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json=body))
        await asyncio.wait_for(entered.wait(), 2)
        blocked = await client.post("/v1/chat/completions", headers=headers, json=body)
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "key_concurrency_limit"
        assert (await client.put("/admin/config", json={"timeout_seconds": 30})).status_code == 409
        finish.set()
        assert (await first).status_code == 200
        fail = True
        assert (await client.post("/v1/chat/completions", headers=headers, json=body)).status_code == 502
        fail = False
        assert (await client.post("/v1/chat/completions", headers=headers, json=body)).status_code == 200


def test_stream_error_is_reported_and_key_slot_released(tmp_path):
    app = create_app(configured(tmp_path))
    async def failing_stream(payload, **options):
        from doubao_work2api.errors import UpstreamError
        yield 'data: {"choices": []}\n\n'
        raise UpstreamError("test failure")
    app.state.service.stream = failing_stream
    with TestClient(app) as client:
        login(client)
        key = client.post("/admin/keys", json={"name": "stream", "concurrency_limit": 1}).json()
        headers = {"x-api-key": key["key"]}
        body = {"stream": True, "messages": [{"role": "user", "content": "hello"}]}
        for _ in range(2):
            with client.stream("POST", "/v1/chat/completions", headers=headers, json=body) as response:
                text = response.read().decode()
                assert response.status_code == 200
                assert '"error"' in text and "[DONE]" in text


def test_deleted_bound_account_does_not_broaden_key_access(tmp_path):
    app = create_app(configured(tmp_path))
    with TestClient(app) as client:
        login(client)
        add_account(client, "one")
        add_account(client, "two")
        key = client.post("/admin/keys", json={"name": "bound", "accounts": ["one"]}).json()
        assert client.post("/admin/accounts/batch", json={"ids": ["one"], "action": "delete"}).status_code == 200
        result = client.post("/v1/chat/completions", headers={"x-api-key": key["key"]},
                             json={"messages": [{"role": "user", "content": "hello"}]})
        assert result.status_code == 503
        assert app.state.pool.find("two").in_flight == 0


def test_import_rejects_updating_busy_credentials(tmp_path):
    app = create_app(configured(tmp_path))
    with TestClient(app) as client:
        login(client)
        add_account(client, "one")
        account = app.state.pool.find("one")
        account.in_flight = 1
        response = client.post("/admin/accounts/import", json={"account": {
            "id": "one", "cookies": {"sessionid": "replacement"},
        }})
        assert response.status_code == 200
        assert response.json()["errors"] and not response.json()["imported"]
        assert account.cookies["sessionid"] == "credential-one"
        account.in_flight = 0
