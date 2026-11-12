from __future__ import annotations

from doubao_work2api.logs import RequestLog


def test_request_log_keeps_metadata_not_prompt() -> None:
    log = RequestLog(capacity=2)
    entry = log.start(model="deepseek-v4.1-flash", stream=False)
    log.account(entry, "account-1")
    log.success(entry)
    snapshot = log.snapshot()
    assert snapshot[0]["status"] == "success"
    assert snapshot[0]["account_id"] == "account-1"
    assert "prompt" not in snapshot[0]
    assert "content" not in snapshot[0]


def test_request_log_is_bounded() -> None:
    log = RequestLog(capacity=2)
    for index in range(4):
        entry = log.start(model=f"m{index}", stream=False)
        log.failure(entry, "failure")
    snapshot = log.snapshot()
    assert len(snapshot) == 2
    assert [item["model"] for item in snapshot] == ["m3", "m2"]
