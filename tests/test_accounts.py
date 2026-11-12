from __future__ import annotations

import json
from pathlib import Path

import pytest

from doubao_work2api.accounts import AccountPool, parse_account_document


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
    assert account.id == "nested"
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
