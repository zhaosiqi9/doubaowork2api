from __future__ import annotations

import json

import pytest

from doubao_work2api.accounts import Account, AccountPool
from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.service import ChatService
from doubao_work2api.errors import GatewayError, UpstreamError


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
ANSWER = json.dumps({"type": "tool_calls", "tool_calls": [{"name": "lookup", "arguments": '{"city":"北京"}'}], "content": None}, ensure_ascii=False)


class Transport:
    def __init__(self, answer=ANSWER, fail=False):
        self.answer = answer
        self.payloads = []
        self.fail = fail

    async def stream(self, account, payload):
        self.payloads.append(payload)
        for start in range(0, len(self.answer), 7):
            yield {"_event": "CHUNK_DELTA", "text": self.answer[start:start + 7]}
        if self.fail:
            raise UpstreamError("interrupted")


def service(tmp_path, transport):
    pool = AccountPool(tmp_path / "accounts")
    pool._accounts = [Account(id="synthetic", cookies={"sessionid": "not-real"}, device_id="device", web_id="web")]
    return ChatService(pool, transport, model_key="unused", app_id="app", workspace_root=tmp_path / "work")


def request(**updates):
    return ChatCompletionRequest(**{"messages": [{"role": "user", "content": "查询北京"}], "tools": TOOLS, "tool_choice": "required", **updates})


@pytest.mark.asyncio
async def test_nonstream_returns_validated_tool_calls_and_prompt(tmp_path):
    transport = Transport()
    result = await service(tmp_path, transport).complete(request(reasoning_effort="low"))
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert result["usage"]["completion_tokens"] > 1
    call = choice["message"]["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert json.loads(call["function"]["arguments"]) == {"city": "北京"}
    payload = transport.payloads[0]
    assert payload["option"]["model_config"]["reasoning_effort"] == 3
    prompt = payload["messages"][0]["content_block"][0]["content"]["text_block"]["text"]
    assert "lookup" in prompt and "TOOL CALLING PROTOCOL" in prompt


@pytest.mark.asyncio
async def test_stream_buffers_split_envelope_without_leaking_json(tmp_path):
    chunks = [chunk async for chunk in service(tmp_path, Transport()).stream(request(stream=True))]
    assert chunks[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks[:-1]]
    choices = [item["choices"][0] for item in payloads]
    assert choices[-1]["finish_reason"] == "tool_calls"
    calls = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
    assert len(calls) == 1 and calls[0]["index"] == 0
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "北京"}
    assert not any(choice["delta"].get("content") for choice in choices)


@pytest.mark.asyncio
async def test_tool_roundtrip_preserves_history_and_can_return_text(tmp_path):
    first = await service(tmp_path, Transport()).complete(request())
    message = first["choices"][0]["message"]
    call = message["tool_calls"][0]
    transport = Transport("北京晴天")
    followup = request(tool_choice="auto", messages=[
        {"role": "user", "content": "查询北京"}, message,
        {"role": "tool", "tool_call_id": call["id"], "name": "lookup", "content": '{"weather":"晴天"}'},
    ])
    result = await service(tmp_path, transport).complete(followup)
    assert result["choices"][0]["message"]["content"] == "北京晴天"
    prompt = transport.payloads[0]["messages"][0]["content_block"][0]["content"]["text_block"]["text"]
    assert call["id"] in prompt and "lookup" in prompt and "晴天" in prompt


@pytest.mark.asyncio
async def test_tool_choice_none_preserves_literal_envelope_as_text(tmp_path):
    result = await service(tmp_path, Transport()).complete(request(tool_choice="none"))
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["message"]["content"] == ANSWER


@pytest.mark.asyncio
async def test_invalid_tool_output_does_not_disable_or_cooldown_account(tmp_path):
    chat = service(tmp_path, Transport(ANSWER.replace('"lookup"', '"unknown"')))
    with pytest.raises(GatewayError) as error:
        await chat.complete(request())
    assert error.value.status_code == 502
    assert chat.pool.accounts[0].available
    assert chat.pool.accounts[0].in_flight == 0


@pytest.mark.asyncio
async def test_buffered_tool_request_does_not_retry_partial_upstream_output(tmp_path):
    transport = Transport(fail=True)
    with pytest.raises(UpstreamError, match="interrupted"):
        await service(tmp_path, transport).complete(request())
    assert len(transport.payloads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_without_answer_is_error_in_both_modes(tmp_path, stream):
    class ReasoningOnly(Transport):
        async def stream(self, account, payload):
            yield {"_event": "STREAM_CHUNK", "content": {"content_block": [{"block_type": 10040}]}}
            yield {"_event": "CHUNK_DELTA", "text": "reasoning only", "_thinking": True}

    chat = service(tmp_path, ReasoningOnly())
    message = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=stream)
    with pytest.raises(UpstreamError) as error:
        if stream:
            _ = [chunk async for chunk in chat.stream(message)]
        else:
            await chat.complete(message)
    assert error.value.code == "empty_response"
