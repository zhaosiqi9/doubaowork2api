from __future__ import annotations

from doubao_work2api.accounts import Account
from doubao_work2api.config import Settings
from doubao_work2api.transport import DirectTransport, classify_body_error


def test_query_contains_required_desktop_identity() -> None:
    transport = DirectTransport(Settings())
    query = transport._query(
        Account(
            id="one",
            cookies={"sessionid": "x"},
            device_id="device-1",
            web_id="web-1",
        )
    )
    assert "device_id=device-1" in query
    assert "web_id=web-1" in query
    assert "fp=verify_device-1" in query
    assert "doubao_device_platform=desktop" in query
    assert "web_tab_id=" in query


def test_classify_body_error_ignores_success_codes() -> None:
    classify_body_error({"code": 0})
    classify_body_error({"code": 200})
    classify_body_error({})
