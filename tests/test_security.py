from __future__ import annotations

import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from doubao_work2api import security
from doubao_work2api.config import Settings
from doubao_work2api.security import SecurityStore


PUBLIC_FIELDS = {
    "id", "name", "enabled", "models", "accounts", "concurrency_limit",
    "created_at", "last_used_at",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "private", api_key="fixture-legacy-token")


@pytest.fixture
def store(settings: Settings) -> SecurityStore:
    return SecurityStore(settings)


def test_admin_default_and_environment_are_independent_of_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DOUBAO_WORK_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("DOUBAO_WORK_API_KEY", "fixture-api-token")
    assert Settings().admin_password == "admin"
    assert Settings.from_env().admin_password == "admin"
    monkeypatch.setenv("DOUBAO_WORK_ADMIN_PASSWORD", "  fixture password  ")
    configured = Settings.from_env()
    assert configured.admin_password == "  fixture password  "
    assert configured.admin_password not in repr(configured)


def test_password_pbkdf2_and_initial_legacy_migration(store, settings) -> None:
    assert store.verify_admin("admin")
    assert not store.verify_admin(settings.api_key)
    assert not store.verify_admin("wrong")
    assert not store.verify_admin(None)
    document = json.loads(store.path.read_text(encoding="utf-8"))
    algorithm, iterations, salt, digest = document["admin_password_hash"].split("$")
    assert algorithm == "pbkdf2_sha256"
    assert int(iterations) >= 600_000
    expected = hashlib.pbkdf2_hmac("sha256", b"admin", bytes.fromhex(salt), int(iterations))
    assert expected == bytes.fromhex(digest)
    assert settings.api_key not in store.path.read_text(encoding="utf-8")
    records = store.list_keys()
    assert len(records) == 1
    assert records[0]["name"] == "默认 Key"
    assert set(records[0]) == PUBLIC_FIELDS
    assert records[0]["models"] == records[0]["accounts"] == []
    assert store.authenticate_api(settings.api_key)["id"] == records[0]["id"]


def test_password_hashes_are_salted(settings, tmp_path) -> None:
    first = SecurityStore(settings)
    second = SecurityStore(replace(settings, data_dir=tmp_path / "other"))
    first_hash = json.loads(first.path.read_text(encoding="utf-8"))["admin_password_hash"]
    second_hash = json.loads(second.path.read_text(encoding="utf-8"))["admin_password_hash"]
    assert first_hash != second_hash
    assert first.verify_admin("admin") and second.verify_admin("admin")


def test_explicit_admin_password_is_used_only_on_initialization(settings) -> None:
    configured = replace(settings, admin_password="fixture initial password")
    store = SecurityStore(configured)
    assert store.verify_admin(configured.admin_password)
    assert not store.verify_admin("admin")
    reopened = SecurityStore(settings)
    assert reopened.verify_admin(configured.admin_password)
    assert not reopened.verify_admin("admin")


def test_password_change_persists_and_invalidates_every_session(store, settings) -> None:
    tokens = [store.create_session(), store.create_session()]
    store.set_admin_password("admin", "fixture replacement password")
    assert all(not store.session_valid(token) for token in tokens)
    assert not store.verify_admin("admin")
    assert store.verify_admin("fixture replacement password")
    reopened = SecurityStore(settings)
    assert reopened.verify_admin("fixture replacement password")
    assert "fixture replacement password" not in store.path.read_text(encoding="utf-8")


@pytest.mark.parametrize("current,new", [
    ("wrong", "fixture replacement"), ("admin", "short"),
    ("admin", None), ("admin", 12345678),
])
def test_invalid_password_change_keeps_password_and_session(store, current, new) -> None:
    token = store.create_session()
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.set_admin_password(current, new)
    assert store.verify_admin("admin")
    assert store.session_valid(token)
    assert store.path.read_bytes() == before


def test_session_expiration_deletion_and_restart(store, settings, monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    token = store.create_session()
    other = store.create_session()
    assert token != other
    assert store.session_valid(token)
    assert not store.session_valid("unknown")
    assert not store.session_valid(None)
    assert not store.session_valid("")
    assert token not in store.path.read_text(encoding="utf-8")
    assert not SecurityStore(settings).session_valid(token)
    store.delete_session(other)
    store.delete_session(other)
    assert not store.session_valid(other)
    now[0] += store.SESSION_TTL_SECONDS - 0.01
    assert store.session_valid(token)
    now[0] += 0.01
    assert not store.session_valid(token)
    assert not store._sessions


def test_create_authenticate_update_disable_delete_and_reload(store, settings) -> None:
    created = store.create_key({
        "name": "  worker  ", "models": ["custom-model"],
        "accounts": ["account-a"], "concurrency_limit": 3,
    })
    assert set(created) == PUBLIC_FIELDS | {"key"}
    assert created["key"].startswith("sk-")
    assert len(created["key"]) >= 40
    assert created["name"] == "worker"
    assert created["last_used_at"] is None
    assert created["key"] not in store.path.read_text(encoding="utf-8")
    assert not store.authenticate_api("invalid")
    assert not store.authenticate_api(None)
    assert not store.authenticate_api("")
    authenticated = store.authenticate_api(created["key"])
    assert set(authenticated) == PUBLIC_FIELDS
    assert authenticated["last_used_at"] is not None
    assert authenticated["models"] == ["custom-model"]
    assert authenticated["accounts"] == ["account-a"]
    assert authenticated["concurrency_limit"] == 3
    reopened = SecurityStore(settings)
    saved = next(record for record in reopened.list_keys() if record["id"] == created["id"])
    assert saved == authenticated
    updated = reopened.update_key(created["id"], {"enabled": False, "name": "disabled"})
    assert set(updated) == PUBLIC_FIELDS
    assert not reopened.authenticate_api(created["key"])
    assert not SecurityStore(settings).authenticate_api(created["key"])
    reopened.update_key(created["id"], {"enabled": True, "models": [], "accounts": []})
    assert reopened.authenticate_api(created["key"])["models"] == []
    reopened.delete_key(created["id"])
    assert not reopened.authenticate_api(created["key"])
    assert not SecurityStore(settings).authenticate_api(created["key"])


def test_last_used_writes_are_coalesced_per_key(store, monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    created = store.create_key({"name": "coalesced"})
    writes = []
    original_save = store._save

    def counting_save(state):
        writes.append(json.loads(json.dumps(state)))
        original_save(state)

    monkeypatch.setattr(store, "_save", counting_save)
    first = store.authenticate_api(created["key"])
    second = store.authenticate_api(created["key"])
    assert len(writes) == 1
    assert second["last_used_at"] >= first["last_used_at"]
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    persisted = next(item for item in on_disk["keys"] if item["id"] == created["id"])
    assert persisted["last_used_at"] == first["last_used_at"]

    now[0] += 59.9
    store.authenticate_api(created["key"])
    assert len(writes) == 1
    now[0] += 0.1
    latest = store.authenticate_api(created["key"])
    assert len(writes) == 2
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    persisted = next(item for item in on_disk["keys"] if item["id"] == created["id"])
    assert persisted["last_used_at"] == latest["last_used_at"]


def test_restart_preserves_latest_last_used_after_a_full_save(store, settings, monkeypatch) -> None:
    now = [2000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    created = store.create_key({"name": "restart"})
    first = store.authenticate_api(created["key"])
    now[0] += 1
    second = store.authenticate_api(created["key"])
    assert second["last_used_at"] > first["last_used_at"]
    assert SecurityStore(settings).list_keys()[-1]["last_used_at"] == first["last_used_at"]

    store.update_runtime({"timeout_seconds": 9})
    reopened = SecurityStore(settings)
    saved = next(record for record in reopened.list_keys() if record["id"] == created["id"])
    assert saved["last_used_at"] == second["last_used_at"]


@pytest.mark.parametrize("operation", ["create", "update", "delete", "runtime", "password"])
def test_other_saves_flush_latest_last_used(store, settings, monkeypatch, operation) -> None:
    now = [3000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    created = store.create_key({"name": "flush"})
    first = store.authenticate_api(created["key"])
    now[0] += 1
    second = store.authenticate_api(created["key"])
    assert second["last_used_at"] > first["last_used_at"]
    other = store.create_key({"name": "other"})

    actions = {
        "create": lambda: store.create_key({"name": "another"}),
        "update": lambda: store.update_key(created["id"], {"name": "updated"}),
        "delete": lambda: store.delete_key(other["id"]),
        "runtime": lambda: store.update_runtime({"timeout_seconds": 9}),
        "password": lambda: store.set_admin_password("admin", "fixture replacement"),
    }
    actions[operation]()
    saved_keys = json.loads(store.path.read_text(encoding="utf-8"))["keys"]
    persisted = next(record for record in saved_keys if record["id"] == created["id"])
    assert persisted["last_used_at"] == second["last_used_at"]
    if operation == "delete":
        assert all(record["id"] != other["id"] for record in saved_keys)
    reopened = SecurityStore(settings)
    saved = next(record for record in reopened.list_keys() if record["id"] == created["id"])
    assert saved["last_used_at"] == second["last_used_at"]


def test_deleted_legacy_key_is_never_seeded_again(store, settings) -> None:
    store.delete_key(store.list_keys()[0]["id"])
    reopened = SecurityStore(replace(settings, api_key="different-fixture-token"))
    assert reopened.list_keys() == []
    assert reopened.authenticate_api(settings.api_key) is None
    assert reopened.authenticate_api("different-fixture-token") is None


def test_empty_legacy_key_does_not_allow_unauthenticated_access(settings) -> None:
    store = SecurityStore(replace(settings, api_key=""))
    assert store.list_keys() == []
    assert store.authenticate_api("") is None
    assert store.authenticate_api(None) is None
    assert store.verify_admin("admin")


def test_public_results_and_input_lists_are_detached(store) -> None:
    models = ["model-a", "model-a"]
    created = store.create_key({"name": "worker", "models": models})
    models.append("model-b")
    created["models"].append("model-c")
    listed = store.list_keys()
    listed[-1]["models"].append("model-d")
    authenticated = store.authenticate_api(created["key"])
    assert authenticated["models"] == ["model-a"]
    authenticated["models"].clear()
    assert store.list_keys()[-1]["models"] == ["model-a"]
    assert all(set(record) == PUBLIC_FIELDS for record in store.list_keys())


def test_runtime_defaults_partial_updates_and_restart(store, settings) -> None:
    assert store.runtime == {
        "default_model": settings.default_model,
        "timeout_seconds": settings.request_timeout_seconds,
        "cooldown_seconds": settings.cooldown_seconds,
    }
    public = store.runtime
    public["default_model"] = "mutated"
    updated = store.update_runtime({"default_model": "custom-model", "timeout_seconds": 2.5})
    assert updated["cooldown_seconds"] == settings.cooldown_seconds
    assert updated["timeout_seconds"] == 2.5
    updated["timeout_seconds"] = 3
    assert store.runtime["timeout_seconds"] == 2.5
    reopened = SecurityStore(replace(settings, default_model="different-env-model"))
    assert reopened.runtime == store.runtime
    assert store.update_runtime({"cooldown_seconds": 0})["cooldown_seconds"] == 0
    assert store.update_runtime({}) == store.runtime


@pytest.mark.parametrize("patch", [
    None, [], {"unknown": 1}, {"admin_password": "forbidden"},
    {"default_model": ""}, {"default_model": "  "}, {"default_model": 12},
    {"default_model": "a\nb"}, {"default_model": "a" * 257},
    {"timeout_seconds": True}, {"timeout_seconds": 0}, {"timeout_seconds": -1},
    {"timeout_seconds": "180"}, {"timeout_seconds": None},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
    {"timeout_seconds": float("-inf")}, {"timeout_seconds": 3601},
    {"cooldown_seconds": True}, {"cooldown_seconds": -1},
    {"cooldown_seconds": 1.5}, {"cooldown_seconds": "60"},
    {"cooldown_seconds": 86401},
    {"default_model": "valid-change", "timeout_seconds": 0},
])
def test_runtime_validation_has_no_partial_effect(store, patch) -> None:
    before = store.runtime
    persisted = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.update_runtime(patch)
    assert store.runtime == before
    assert store.path.read_bytes() == persisted


@pytest.mark.parametrize("patch", [
    None, [], {"unknown": 1}, {"key": "forbidden"}, {"secret_hash": "forbidden"},
    {"id": "forbidden"}, {"last_used_at": "forbidden"},
    {"name": ""}, {"name": " "}, {"name": "a" * 129}, {"name": False},
    {"enabled": 1}, {"enabled": "false"}, {"enabled": None},
    {"models": "model-a"}, {"models": [""]}, {"models": [None]},
    {"models": ["a" * 257]}, {"models": ["m"] * 1001},
    {"accounts": {}}, {"accounts": [1]}, {"accounts": ["  "]},
    {"concurrency_limit": True}, {"concurrency_limit": -1},
    {"concurrency_limit": 1.5}, {"concurrency_limit": "2"},
    {"concurrency_limit": 1001},
])
def test_key_validation_has_no_partial_effect(store, patch) -> None:
    before = store.list_keys()
    persisted = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.create_key(patch)
    with pytest.raises(ValueError):
        store.update_key(before[0]["id"], patch)
    assert store.list_keys() == before
    assert store.path.read_bytes() == persisted


def test_missing_name_and_missing_key_are_rejected(store) -> None:
    with pytest.raises(ValueError, match="name"):
        store.create_key({})
    with pytest.raises(KeyError):
        store.update_key("missing", {"enabled": False})
    with pytest.raises(KeyError):
        store.delete_key("missing")


@pytest.mark.parametrize("operation", ["runtime", "create", "update", "delete", "auth", "password"])
def test_failed_atomic_replace_keeps_memory_disk_and_sessions(store, settings, monkeypatch, operation) -> None:
    before_keys = store.list_keys()
    before_runtime = store.runtime
    persisted = store.path.read_bytes()
    token = store.create_session()

    def fail_replace(source, destination):
        # The complete JSON is staged in the same directory before replacing.
        assert Path(source).parent == store.path.parent
        json.loads(Path(source).read_text(encoding="utf-8"))
        assert Path(destination).read_bytes() == persisted
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(security.os, "replace", fail_replace)
    actions = {
        "runtime": lambda: store.update_runtime({"timeout_seconds": 2}),
        "create": lambda: store.create_key({"name": "new"}),
        "update": lambda: store.update_key(before_keys[0]["id"], {"enabled": False}),
        "delete": lambda: store.delete_key(before_keys[0]["id"]),
        "auth": lambda: store.authenticate_api(settings.api_key),
        "password": lambda: store.set_admin_password("admin", "fixture replacement"),
    }
    with pytest.raises(OSError, match="simulated"):
        actions[operation]()
    assert store.list_keys() == before_keys
    assert store.runtime == before_runtime
    assert store.path.read_bytes() == persisted
    assert store.verify_admin("admin")
    assert store.session_valid(token)
    assert not list(store.path.parent.glob(".settings-*.tmp"))


def test_concurrent_key_and_runtime_writes_do_not_lose_changes(store, settings) -> None:
    def create(index):
        return store.create_key({"name": f"worker-{index}"})["id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(create, range(24)))
        futures = [
            executor.submit(store.update_runtime, {"default_model": "concurrent-model"}),
            executor.submit(store.update_runtime, {"timeout_seconds": 7.5}),
            executor.submit(store.update_runtime, {"cooldown_seconds": 11}),
        ]
        for future in futures:
            future.result()
    assert len(set(ids)) == 24
    reopened = SecurityStore(settings)
    assert len(reopened.list_keys()) == 25
    assert {record["id"] for record in reopened.list_keys()} >= set(ids)
    assert reopened.runtime == {
        "default_model": "concurrent-model", "timeout_seconds": 7.5, "cooldown_seconds": 11,
    }
    assert not list(store.path.parent.glob(".settings-*.tmp"))


@pytest.mark.parametrize("payload", ["{", "null", "{}", '{"version": 99}'])
def test_invalid_settings_fail_closed_without_overwriting(settings, payload) -> None:
    settings.data_dir.mkdir(parents=True)
    path = settings.data_dir / "settings.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid security settings file"):
        SecurityStore(settings)
    assert path.read_text(encoding="utf-8") == payload


def test_invalid_hash_fails_closed(store, settings) -> None:
    state = json.loads(store.path.read_text(encoding="utf-8"))
    state["admin_password_hash"] = "pbkdf2_sha256$999999999999$invalid$invalid"
    store.path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid security settings file"):
        SecurityStore(settings)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions are not available on Windows")
def test_private_permissions_for_directory_and_atomic_replacements(store, settings) -> None:
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    store.update_runtime({"timeout_seconds": 2})
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    store.path.chmod(0o666)
    store.path.parent.chmod(0o777)
    SecurityStore(settings)
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
