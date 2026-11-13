import json
import time

import pytest

from doubao_work2api.models import ChatMessage
from doubao_work2api.response_store import InvalidReplayToken, ResponseNotFound, ResponseStore, response_owner


def test_cache_is_key_and_policy_isolated_and_copies_objects(tmp_path):
    store = ResponseStore(tmp_path)
    owner = response_owner({"id": "one", "models": [], "accounts": []})
    changed = response_owner({"id": "one", "models": ["deepseek-v4.1-flash"], "accounts": []})
    response = {"id": "resp_one", "output": [{"id": "msg_one", "type": "message"}]}
    assert store.put(owner, response, [ChatMessage(role="user", content="private prompt")], [])
    response["output"].clear()
    assert store.find_item(owner, "msg_one")["type"] == "message"
    assert store.get(owner, "resp_one")["history"][0]["content"] == "private prompt"
    with pytest.raises(ResponseNotFound):
        store.get(changed, "resp_one")
    with pytest.raises(ResponseNotFound):
        store.find_item("another-key", "msg_one")
    assert not any(path.suffix == ".json" for path in tmp_path.iterdir())
    store.delete(owner, "resp_one")
    with pytest.raises(ResponseNotFound):
        store.get(owner, "resp_one")


def test_cache_ttl_and_eviction(tmp_path):
    store = ResponseStore(tmp_path, ttl=0.01, max_entries=1)
    store.put("one", {"id": "resp_one"}, [], [])
    store.put("one", {"id": "resp_two"}, [], [])
    with pytest.raises(ResponseNotFound):
        store.get("one", "resp_one")
    record = store._records["resp_two"]
    store._records["resp_two"] = (time.time() - 10, record[1], record[2])
    with pytest.raises(ResponseNotFound):
        store.get("one", "resp_two")


def test_encrypted_compaction_replay_survives_restart_and_enforces_owner_model(tmp_path):
    store = ResponseStore(tmp_path)
    text = "private conversation summary"
    token = store.encode("owner", "model", [ChatMessage(role="assistant", content=text)], kind="compaction")
    assert text not in token
    item = {"type": "compaction", "encrypted_content": token}
    second = ResponseStore(tmp_path)
    assert second.decode("owner", "model", item)[0].content == text
    for owner, model in [("other", "model"), ("owner", "other-model")]:
        with pytest.raises(InvalidReplayToken):
            second.decode(owner, model, item)
    with pytest.raises(InvalidReplayToken):
        second.decode("owner", "model", {**item, "encrypted_content": token[:-3] + "bad"})
    with pytest.raises(InvalidReplayToken):
        second.decode("owner", "model", {**item, "type": "reasoning"})


def test_sealed_reasoning_preserves_whitespace_and_does_not_mutate_response(tmp_path):
    store = ResponseStore(tmp_path)
    response = {"model": "model", "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": " a\n b "}]}]}
    sealed = store.seal_reasoning("owner", response)
    assert "encrypted_content" not in response["output"][0]
    replay = store.decode("owner", "model", sealed["output"][0])
    assert replay[0].reasoning_content == " a\n b "


def test_oversized_cache_record_not_stored(tmp_path):
    store = ResponseStore(tmp_path, max_bytes=100)
    assert not store.put("one", {"id": "resp_big", "content": "x" * 200}, [], [])
    assert store._total == 0
