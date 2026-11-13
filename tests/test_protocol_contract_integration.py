import json

import pytest

from doubao_work2api.accounts import Account, AccountPool
from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.protocol import UpstreamTextParser, extract_text_fragments, render_prompt, validate_request
from doubao_work2api.service import ChatService


class LiveLikeTransport:
    def __init__(self, text):
        self.text = text
        self.closed = False
        self.read_count = 0

    async def stream(self, account, payload):
        try:
            for character in self.text:
                self.read_count += 1
                yield {"_event": "CHUNK_DELTA", "text": character}
        finally:
            self.closed = True


def chat_service(tmp_path, transport):
    pool = AccountPool(tmp_path / "accounts")
    pool._accounts = [Account(id="synthetic", cookies={"sessionid": "synthetic"}, device_id="test", web_id="test")]
    return ChatService(pool, transport, model_key="unused", app_id="test", workspace_root=tmp_path / "work")


@pytest.mark.parametrize("stream", [False, True])
async def test_output_budget_cuts_delivery_and_closes_upstream_immediately(tmp_path, stream):
    transport = LiveLikeTransport("abcdefghi")
    chat = chat_service(tmp_path, transport)
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}], max_tokens=3, stream=stream)
    if stream:
        chunks = [chunk async for chunk in chat.stream(request)]
        records = [json.loads(chunk[6:]) for chunk in chunks if "[DONE]" not in chunk]
        text = "".join(record["choices"][0]["delta"].get("content", "") for record in records)
        assert text == "abc"
        assert records[-1]["choices"][0]["finish_reason"] == "length"
    else:
        result = await chat.complete(request)
        assert result["choices"][0]["message"]["content"] == "abc"
        assert result["choices"][0]["finish_reason"] == "length"
    assert transport.closed and transport.read_count == 3
    assert chat.pool.accounts[0].in_flight == 0
    assert not chat.pool.accounts[0].lock.locked()


async def test_json_schema_is_enforced_in_shared_pipeline(tmp_path):
    transport = LiveLikeTransport('{"value":"wrong type"}')
    chat = chat_service(tmp_path, transport)
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}], response_format={"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}})
    with pytest.raises(Exception) as error:
        await chat.complete(request)
    assert error.value.code == "schema_validation_failed"
    assert chat.pool.accounts[0].available


def test_reasoning_replay_is_not_dropped_and_fields_are_not_duplicated():
    raw = " \noriginal reasoning  "
    request = ChatCompletionRequest(messages=[{"role": "assistant", "content": "answer", "reasoning_content": raw, "reasoning": raw}, {"role": "user", "content": "continue"}])
    normalized = validate_request(request)
    prompt = render_prompt(normalized.messages)
    assert prompt.count(raw) == 1
    assert "historical_reasoning" in prompt


def test_multiple_upstream_reasoning_shapes_are_supported_without_mirroring_duplicates():
    assert list(extract_text_fragments({"choices": [{"delta": {"reasoning_content": "a", "reasoning": "a", "content": "b"}}]})) == [("reasoning", "a"), ("content", "b")]
    assert list(extract_text_fragments({"_event": "CHUNK_DELTA", "reasoning_details": [{"type": "reasoning.text", "text": "think"}], "text": "answer"})) == [("reasoning", "think"), ("content", "answer")]
    assert list(extract_text_fragments({"choices": [{"delta": {"reasoning_content": [{"encrypted_content": "opaque"}], "reasoning": "visible"}}]})) == [("reasoning", "visible")]


def test_legacy_functions_request_converts_to_tool_contract():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}], functions=[{"name": "lookup", "parameters": {"type": "object"}}], function_call={"name": "lookup"})
    normalized = validate_request(request)
    assert normalized.tools[0]["function"]["name"] == "lookup"
    assert normalized.tool_choice["function"]["name"] == "lookup"
    assert validate_request(normalized).tools == normalized.tools


def test_orphan_legacy_function_choice_is_rejected():
    with pytest.raises(Exception) as error:
        validate_request(ChatCompletionRequest(messages=[{"role": "user", "content": "test"}], function_call="auto"))
    assert error.value.code == "invalid_tools"


def test_real_container_updates_do_not_swallow_answer():
    parser = UpstreamTextParser()
    def block(value):
        return {"_event": "STREAM_CHUNK", "patch_op": [{"patch_value": {"content_block": [value]}}]}
    container = block({"block_type": 10040, "block_id": "thinking", "content": {"thinking_block": {"finish_title": "done"}}, "control_info": {"collapse_block_id": "elapsed"}})
    assert list(parser.feed(container)) == []
    assert list(parser.feed(container)) == []
    thought = block({"block_type": 10000, "block_id": "thought", "parent_id": "thinking", "content": {"text_block": {"text": "visible thought"}}})
    assert list(parser.feed(thought)) == [("reasoning", "visible thought")]
    assert list(parser.feed({"_event": "CHUNK_DELTA", "block_id": "thought", "text": "."})) == [("reasoning", ".")]
    answer = block({"block_type": 10000, "block_id": "answer", "content": {"text_block": {"text": "SHAPE_OK"}}, "control_info": {"collapse_block_id": "elapsed"}})
    assert list(parser.feed(answer)) == [("content", "SHAPE_OK")]
    assert list(parser.feed(container)) == []
    assert list(parser.feed({"_event": "CHUNK_DELTA", "text": "!"})) == [("content", "!")]


async def test_real_container_shape_reaches_json_contract(tmp_path):
    class Blocks(LiveLikeTransport):
        async def stream(self, account, payload):
            yield {"_event": "STREAM_CHUNK", "content": {"content_block": [{"block_type": 10040, "block_id": "thinking"}]}}
            yield {"_event": "STREAM_CHUNK", "content": {"content_block": [{"block_type": 10000, "block_id": "answer", "content": {"text_block": {"text": '{"ok":true}'}}}]}}
    result = await chat_service(tmp_path, Blocks("")).complete(ChatCompletionRequest(messages=[{"role": "user", "content": "test"}], response_format={"type": "json_object"}))
    assert json.loads(result["choices"][0]["message"]["content"]) == {"ok": True}


async def test_sse_verification_error_disables_account_with_actual_code(tmp_path, monkeypatch):
    import httpx
    from doubao_work2api.config import Settings
    from doubao_work2api.errors import UpstreamChallengeError
    from doubao_work2api.transport import DirectTransport
    transport = DirectTransport(Settings(upstream_base="https://upstream.test"))
    def create(account):
        return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, text='event: STREAM_ERROR\ndata: {"error_code":710022004}\n\n')))
    monkeypatch.setattr(transport, "_create_client", create)
    service = chat_service(tmp_path, transport)
    try:
        with pytest.raises(UpstreamChallengeError):
            await service.complete(ChatCompletionRequest(messages=[{"role": "user", "content": "synthetic"}]))
        account = service.pool.accounts[0]
        assert not account.enabled
        assert account.error_code == "710022004"
        assert account.in_flight == 0 and not account.lock.locked()
        reloaded = AccountPool(tmp_path / "accounts")
        reloaded.load()
        assert not reloaded.accounts[0].enabled
        assert reloaded.accounts[0].error_code == "710022004"
    finally:
        await transport.close()
