from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from doubao_work2api.accounts import AccountPool, parse_account_document
from doubao_work2api.errors import NoAvailableAccountError


def test_parse_cookie_string_account() -> None:
    account = parse_account_document(
        {
            "id": "a1",
            "device_id": "device-1",
            "web_id": "web-1",
            "cookie": "sessionid=abc; passport_csrf_token=csrf",
        }
    )
    assert account.id == "a1"
    assert account.cookies["sessionid"] == "abc"
    assert account.cookies["passport_csrf_token"] == "csrf"
    assert account.device_id == "device-1"


def test_parse_nested_cookie_array() -> None:
    account = parse_account_document(
        {
            "name": "nested",
            "identity": {"device_id": "d", "web_id": "w"},
            "cookies": [
                {"name": "sessionid", "value": "abc"},
                {"name": "other", "value": "x"},
            ],
        }
    )
    assert account.id.startswith("account-")
    assert account.name == "nested"
    assert account.cookies == {"sessionid": "abc", "other": "x"}
    assert account.device_id == "d"
    assert account.web_id == "w"


def test_parse_account_rejects_missing_cookies() -> None:
    with pytest.raises(ValueError, match="no cookies"):
        parse_account_document({"id": "empty"})


def test_pool_loads_json_files_and_ignores_bad_files(tmp_path: Path) -> None:
    (tmp_path / "one.json").write_text(
        json.dumps(
            {
                "id": "one",
                "device_id": "d",
                "web_id": "w",
                "cookie": "sessionid=x",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    pool = AccountPool(tmp_path)
    assert pool.load() == 1
    assert pool.accounts[0].id == "one"


def document(account_id: str = "one", **changes) -> dict:
    return {"id": account_id, "cookies": {"sessionid": f"fake-{account_id}"},
            "device_id": "shared-device", "web_id": "web", **changes}


async def test_import_preserves_omitted_metadata_and_identity(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    original = await pool.upsert(document(
        name="First", remark="Keep", group="Team", enabled=False, device_name="Custom"
    ))
    original_path = original.source_path
    updated = await pool.upsert({"account": {
        "cookies": {"sessionid": "fake-one", "rotating": "changed"},
        "device_id": "new-device",
    }})
    assert updated is original
    assert updated.id == "one"
    assert updated.source_path == original_path
    assert (updated.name, updated.remark, updated.group) == ("First", "Keep", "Team")
    assert updated.enabled is False
    assert updated.device_name == "Custom"
    assert updated.web_id == "web"
    assert updated.device_id == "new-device"
    reloaded = AccountPool(tmp_path)
    assert reloaded.load() == 1
    assert reloaded.accounts[0].to_document() == updated.to_document()


async def test_import_explicit_fields_replace_and_id_wins(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    first = await pool.upsert(document(name="First", enabled=False, device_name="Custom"))
    await pool.upsert(document("two"))
    updated = await pool.upsert(document(
        cookies={"sessionid": "fake-two"}, name="", remark="New", group="G",
        enabled=True, device_name="Replacement",
    ))
    assert updated is first
    assert len(pool.accounts) == 2
    assert (updated.name, updated.remark, updated.group) == ("", "New", "G")
    assert updated.enabled is True
    assert updated.device_name == "Replacement"


async def test_identity_dedup_preserves_existing_id_even_when_supplied_id_changes(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    original = await pool.upsert(document())
    updated = await pool.upsert(document("other", cookies={"sessionid": "fake-one"}))
    assert updated is original
    assert updated.id == "one"
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_shared_devices_and_missing_sessions_do_not_merge(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    for cookies in (
        {"sessionid": "fake-a"}, {"sessionid": "fake-b"},
        {"tracking": "fake-a"}, {"tracking": "fake-b"},
        {"sessionid": "", "tracking": "fake-c"},
    ):
        await pool.upsert({"device_id": "same", "name": "Same name", "cookies": cookies})
    assert len(pool.accounts) == 5
    assert len({account.id for account in pool.accounts}) == 5


async def test_generated_identity_survives_cookie_rotation_and_session_alias(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    first = await pool.upsert({"cookies": {"sessionid": "fake", "csrf": "old"}})
    second = await pool.upsert({"cookies": {"sessionid_ss": "fake", "csrf": "new"}})
    assert second is first
    assert parse_account_document({"cookies": {"sessionid": "fake"}}).id == first.id


async def test_paths_are_collision_safe_and_legacy_paths_stay(tmp_path: Path) -> None:
    ids = ["a/b", "ab", "AB", "a" * 80 + "b", "a" * 80 + "c", "///", "CON"]
    pool = AccountPool(tmp_path)
    for account_id in ids:
        await pool.upsert(document(account_id))
    paths = [account.source_path for account in pool.accounts]
    assert len({str(path).lower() for path in paths}) == len(ids)
    assert all(path is not None and path.exists() for path in paths)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(document("legacy")), encoding="utf-8")
    assert pool.load() == len(ids) + 1
    updated = await pool.upsert(document("legacy", remark="new"))
    assert updated.source_path == legacy
    assert json.loads(legacy.read_text(encoding="utf-8"))["remark"] == "new"


async def test_metadata_enabled_and_errors_persist_without_public_credentials(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    account = await pool.upsert(document())
    pool.mark_cooldown(account, "synthetic reason", error_code="429")
    error_at = account.last_error_at
    assert error_at > 0
    updated = await pool.update_metadata("one", {
        "name": "Name", "remark": "Remark", "group": "Group", "enabled": False,
    })
    assert updated is account
    assert (account.last_error, account.error_code, account.last_error_at) == (
        "synthetic reason", "429", error_at,
    )
    snapshot = pool.snapshot()[0]
    for key in ("name", "remark", "group", "error_code", "last_error_at"):
        assert snapshot[key] == account.to_document()[key]
    assert "cookies" not in snapshot
    reloaded = AccountPool(tmp_path)
    assert reloaded.load() == 1
    assert reloaded.accounts[0].to_document() == account.to_document()
    await pool.clear_cooldown("one")
    assert account.error_code == account.last_error == ""
    assert account.last_error_at == 0
    assert json.loads(account.source_path.read_text(encoding="utf-8"))["last_error_at"] == 0


async def test_set_enabled_preserves_auth_error_fields(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    account = await pool.upsert(document())
    account.last_error, account.error_code, account.last_error_at = "expired", "401", 123.0
    await pool.set_enabled("one", False)
    saved = json.loads(account.source_path.read_text(encoding="utf-8"))
    assert saved["last_error"] == "expired"
    assert saved["error_code"] == "401"
    assert saved["last_error_at"] == 123.0
    assert account.enabled is False


@pytest.mark.parametrize("patch", [{"cookies": {}}, {"enabled": "false"}, {"name": None}])
async def test_metadata_rejects_invalid_patches(tmp_path: Path, patch: dict) -> None:
    pool = AccountPool(tmp_path)
    account = await pool.upsert(document())
    before = account.to_document()
    with pytest.raises(ValueError):
        await pool.update_metadata("one", patch)
    assert account.to_document() == before
    with pytest.raises(KeyError):
        await pool.update_metadata("missing", {"name": "New"})


async def test_export_is_ordered_detached_and_rejects_missing_ids(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    await pool.upsert(document())
    await pool.upsert(document("two"))
    exported = pool.export_accounts(["two", "one", "two"])
    assert [item["id"] for item in exported] == ["two", "one"]
    exported[0]["cookies"]["sessionid"] = "changed"
    assert pool.find("two").cookies["sessionid"] == "fake-two"
    assert pool.export_accounts([]) == []
    with pytest.raises(KeyError):
        pool.export_accounts(["one", "missing"])


async def test_delete_validates_all_ids_and_busy_before_removing_any(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    one = await pool.upsert(document())
    two = await pool.upsert(document("two"))
    with pytest.raises(KeyError):
        await pool.delete_many(["one", "missing"])
    held = await pool.acquire(allowed_ids={"two"})
    with pytest.raises(ValueError, match="busy"):
        await pool.delete_many(["one", "two"])
    assert one.source_path.exists() and two.source_path.exists()
    await pool.release(held)
    async with one.lock:
        with pytest.raises(ValueError, match="busy"):
            await pool.delete_many(["one"])
    assert await pool.delete_many(["two", "one", "two"]) == ["two", "one"]
    assert pool.accounts == []
    assert not list(tmp_path.iterdir())
    assert await pool.delete_many([]) == []
    assert AccountPool(tmp_path).load() == 0


async def test_delete_rolls_back_files_on_staging_failure(tmp_path: Path, monkeypatch) -> None:
    from doubao_work2api import accounts as module

    pool = AccountPool(tmp_path)
    one = await pool.upsert(document())
    two = await pool.upsert(document("two"))
    original_replace = module.os.replace

    def fail_second(source, target):
        if Path(source) == two.source_path:
            raise OSError("synthetic rename failure")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        await pool.delete_many(["one", "two"])
    assert [account.id for account in pool.accounts] == ["one", "two"]
    assert one.source_path.exists() and two.source_path.exists()
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("operation", ["insert", "upsert", "metadata", "enabled", "cooldown", "clear", "success"])
async def test_failed_atomic_replace_preserves_memory_and_disk(
    tmp_path: Path, monkeypatch, operation: str,
) -> None:
    from doubao_work2api import accounts as module

    pool = AccountPool(tmp_path)
    account = await pool.upsert(document())
    pool.mark_cooldown(account, "old error", "old-code")
    before = account.to_document()
    disk_before = account.source_path.read_bytes()

    def fail_replace(source, target):
        # Nothing in the live object changes while persistence is pending.
        assert account.to_document() == before
        assert len(pool.accounts) == 1
        raise PermissionError("synthetic write failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="synthetic"):
        if operation == "insert":
            await pool.upsert(document("new"))
        elif operation == "upsert":
            await pool.upsert(document(cookies={"sessionid": "new"}, name="changed"))
        elif operation == "metadata":
            await pool.update_metadata("one", {"name": "changed", "enabled": False})
        elif operation == "enabled":
            await pool.set_enabled("one", False)
        elif operation == "cooldown":
            pool.mark_cooldown(account, "new error", "new-code")
        elif operation == "clear":
            await pool.clear_cooldown("one")
        else:
            pool.mark_success(account)
    assert account.to_document() == before
    assert account.source_path.read_bytes() == disk_before
    assert len(list(tmp_path.iterdir())) == 1


async def test_acquire_filters_and_keeps_round_robin(tmp_path: Path) -> None:
    pool = AccountPool(tmp_path)
    for account_id in ("one", "two", "three"):
        await pool.upsert(document(account_id))
    for allowed in (set(), {"missing"}):
        with pytest.raises(NoAvailableAccountError):
            await pool.acquire(allowed_ids=allowed)
    acquired = []
    for _ in range(4):
        account = await pool.acquire(allowed_ids={"one", "three"})
        acquired.append(account.id)
        await pool.release(account)
    assert acquired == ["one", "three", "one", "three"]
    await pool.set_enabled("one", False)
    with pytest.raises(NoAvailableAccountError, match="disabled"):
        await pool.acquire(allowed_ids={"one"})
    held = await pool.acquire(allowed_ids={"three"})
    with pytest.raises(NoAvailableAccountError, match="busy"):
        await pool.acquire(allowed_ids={"three"})
    await pool.release(held)
    pool.mark_cooldown(held, "synthetic cooldown")
    with pytest.raises(NoAvailableAccountError, match="cooling"):
        await pool.acquire(allowed_ids={"three"})
    assert (await pool.acquire()).id == "two"


async def test_desktop_scan_uses_fake_identity_and_preserves_metadata(tmp_path: Path, monkeypatch) -> None:
    from doubao_work2api import desktop

    pool = AccountPool(tmp_path)
    original = await pool.upsert(document(
        enabled=False, name="Keep", remark="R", group="G", device_name="Custom",
    ))
    monkeypatch.setattr(desktop, "scan_offline_account", lambda: SimpleNamespace(
        cookies={"sessionid": "fake-one", "csrf": "refreshed"},
        device_id="changed-device", web_id="changed-web",
    ))
    result = await desktop.scan_desktop_account(pool)
    assert result.account is original
    assert result.created is False
    assert original.enabled is False
    assert (original.name, original.remark, original.group, original.device_name) == ("Keep", "R", "G", "Custom")
    monkeypatch.setattr(desktop, "scan_offline_account", lambda: SimpleNamespace(
        cookies={"sessionid": "fake-new"}, device_id="changed-device", web_id="changed-web",
    ))
    result = await desktop.scan_desktop_account(pool)
    assert result.created is True
    assert len(pool.accounts) == 2
