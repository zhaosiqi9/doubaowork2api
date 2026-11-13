from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from doubao_work2api.errors import GatewayError
from doubao_work2api.models import ChatMessage
from doubao_work2api.responses import (
    ResponsesConversionError,
    assemble_response_stream,
    chat_stream_to_responses,
    chat_to_response,
    response_output_to_messages,
    responses_to_chat,
)


def sse(value: dict | str) -> str:
    if isinstance(value, str):
        return f"data: {value}\n\n"
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def chat_chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    choices: list[dict] | None = None,
) -> dict:
    if choices is None:
        delta: dict = {}
        if content is not None:
            delta["content"] = content
        if reasoning is not None:
            delta["reasoning_content"] = reasoning
            delta["reasoning"] = reasoning
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        choices = [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    value = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "deepseek-v4.1-flash",
        "choices": choices,
    }
    if usage is not None:
        value["usage"] = usage
    return value


async def collect(
    chunks: list[str],
    payload: dict | None = None,
) -> tuple[list[str], list[dict]]:
    async def source() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    events = [
        event
        async for event in chat_stream_to_responses(
            source(),
            payload or {"model": "deepseek-v4.1-flash", "input": "hello"},
        )
    ]
    parsed: list[dict] = []
    for event in events:
        assert event.startswith("event: ")
        name_line, data_line, _ = event.split("\n", 2)
        assert data_line.startswith("data: ")
        data = json.loads(data_line[6:])
        assert name_line == f"event: {data['type']}"
        parsed.append(data)
    return events, parsed


def test_responses_request_maps_official_flat_schema() -> None:
    request = responses_to_chat(
        {
            "model": "deepseek-v4.1-flash",
            "instructions": "Be precise.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Look"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/a.png",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "reasoning": {"effort": "high"},
            "text": {
                "verbosity": "medium",
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
            "max_output_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "lookup"},
            "parallel_tool_calls": False,
            "store": False,
            "metadata": {"trace": "t1"},
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "truncation": "auto",
        },
        "fallback-model",
    )
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == "Be precise."
    assert request.messages[1].content[0] == {"type": "text", "text": "Look"}
    assert request.messages[1].content[1]["image_url"] == {
        "url": "https://example.com/a.png",
        "detail": "high",
    }
    assert request.reasoning_effort == "high"
    assert request.response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    assert request.verbosity == "medium"
    assert request.max_completion_tokens == 64
    assert request.tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert request.tool_choice == {
        "type": "function",
        "function": {"name": "lookup"},
    }
    assert request.parallel_tool_calls is False
    assert request.stream is True
    assert request.store is False
    assert request.metadata == {"trace": "t1"}
    assert request.include == ["reasoning.encrypted_content"]
    assert request.truncation == "auto"


def test_previous_messages_and_reasoning_replay() -> None:
    previous = [
        ChatMessage(role="user", content="first"),
        ChatMessage(
            role="assistant",
            content=None,
            reasoning_content="summary",
        ),
    ]
    request = responses_to_chat(
        {
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "replayed"}],
                    "content": [{"type": "reasoning_text", "text": "raw"}],
                },
                {"role": "user", "content": "next"},
            ],
            "previous_response_id": "resp_old",
        },
        "deepseek-v4.1-flash",
        previous_messages=previous,
    )
    assert request.messages[:2] == previous
    replayed = request.messages[2]
    assert replayed.reasoning_content == "raw"
    assert replayed.reasoning_summary == "replayed"
    assert request.messages[-1].content == "next"
    assert all(
        message.role != "system" for message in request.messages
    )


def test_previous_response_id_without_lookup_is_a_400() -> None:
    with pytest.raises(ResponsesConversionError) as caught:
        responses_to_chat(
            {"input": "next", "previous_response_id": "resp_old"},
            "deepseek-v4.1-flash",
        )
    assert caught.value.status_code == 400
    assert caught.value.code == "previous_response_not_resolved"


def test_unknown_include_and_bad_truncation_are_clear_400s() -> None:
    with pytest.raises(ResponsesConversionError) as include:
        responses_to_chat(
            {"input": "x", "include": ["message.output_text.logprobs"]},
            "deepseek-v4.1-flash",
        )
    assert include.value.status_code == 400
    assert include.value.code == "unsupported_include"

    with pytest.raises(ResponsesConversionError) as truncation:
        responses_to_chat(
            {"input": "x", "truncation": "invalid"},
            "deepseek-v4.1-flash",
        )
    assert truncation.value.status_code == 400
    assert truncation.value.code == "invalid_truncation"


def test_compaction_receives_the_whole_item_and_inserts_messages() -> None:
    seen: list[dict] = []

    def decode(item: dict) -> list[ChatMessage]:
        seen.append(item)
        return [
            ChatMessage(role="assistant", content="compacted"),
            ChatMessage(role="user", content="resume"),
        ]

    item = {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": "token",
    }
    request = responses_to_chat(
        {"input": [item, {"role": "user", "content": "go"}]},
        "deepseek-v4.1-flash",
        decode_compaction=decode,
    )
    assert seen == [item]
    assert [message.content for message in request.messages] == [
        "compacted",
        "resume",
        "go",
    ]


def test_unknown_compaction_and_encrypted_reasoning_are_explicit_400s() -> None:
    with pytest.raises(ResponsesConversionError) as compaction:
        responses_to_chat(
            {
                "input": [
                    {
                        "type": "compaction",
                        "encrypted_content": "opaque",
                    }
                ]
            },
            "deepseek-v4.1-flash",
        )
    assert compaction.value.code == "compaction_decoder_required"

    with pytest.raises(ResponsesConversionError) as reasoning:
        responses_to_chat(
            {
                "input": [
                    {
                        "type": "reasoning",
                        "encrypted_content": "third-party-token",
                    }
                ]
            },
            "deepseek-v4.1-flash",
        )
    assert reasoning.value.status_code == 400
    assert reasoning.value.code == "encrypted_reasoning_unresolved"


def test_known_reasoning_token_can_use_a_decode_hook() -> None:
    seen: list[dict] = []

    def decode(item: dict) -> list[ChatMessage]:
        seen.append(item)
        return [ChatMessage(role="assistant", reasoning_content="decoded")]

    request = responses_to_chat(
        {
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_own",
                    "encrypted_content": "own-token",
                }
            ]
        },
        "deepseek-v4.1-flash",
        decode_reasoning=decode,
    )
    assert seen[0]["id"] == "rs_own"
    assert request.messages[0].reasoning_content == "decoded"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"input": "x", "background": True}, "unsupported_parameter"),
        (
            {"input": "x", "tools": [{"type": "web_search"}]},
            "unsupported_tool",
        ),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "file_id": "file_1",
                            }
                        ],
                    }
                ]
            },
            "unsupported_image_reference",
        ),
        (
            {"input": [{"type": "item_reference", "id": "msg_1"}]},
            "item_reference_unresolved",
        ),
        (
            {"input": [{"type": "unknown", "value": 1}]},
            "unsupported_input_item",
        ),
    ],
)
def test_unsupported_request_shapes_are_preflight_400s(
    payload: dict,
    code: str,
) -> None:
    with pytest.raises(ResponsesConversionError) as caught:
        responses_to_chat(payload, "deepseek-v4.1-flash")
    assert caught.value.status_code == 400
    assert caught.value.code == code


def test_function_call_and_output_round_trip_in_history() -> None:
    request = responses_to_chat(
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"city":"北京"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "晴"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/result.png",
                        },
                    ],
                },
                {"role": "user", "content": "continue"},
            ]
        },
        "deepseek-v4.1-flash",
    )
    assistant, tool, user = request.messages
    assert assistant.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city":"北京"}'},
        }
    ]
    assert tool.role == "tool"
    assert tool.tool_call_id == "call_1"
    assert tool.content[0] == {"type": "text", "text": "晴"}
    assert tool.content[1]["image_url"]["url"].endswith("result.png")
    assert user.content == "continue"


def test_chat_to_response_keeps_reasoning_once_and_tool_arguments() -> None:
    response = chat_to_response(
        {
            "id": "chatcmpl-1",
            "created": 1700000000,
            "model": "deepseek-v4.1-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "same",
                        "reasoning": "same",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q":"x"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        },
        {
            "model": "deepseek-v4.1-flash",
            "input": "x",
            "tools": [{"type": "function", "name": "lookup"}],
        },
        response_id="resp_test",
    )
    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["store"] is True
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 4
    assert response["usage"]["total_tokens"] == 14
    reasoning = [
        item for item in response["output"] if item["type"] == "reasoning"
    ]
    assert len(reasoning) == 1
    assert reasoning[0]["summary"][0]["text"] == "same"
    function_call = response["output"][-1]
    assert function_call["type"] == "function_call"
    assert function_call["arguments"] == '{"q":"x"}'
    assert function_call["call_id"] == "call_1"

    replay = response_output_to_messages(response)
    assert replay[0].reasoning_content == "same"
    assert replay[1].tool_calls[0]["function"]["arguments"] == '{"q":"x"}'


def test_chat_to_response_marks_length_incomplete_even_without_content() -> None:
    response = chat_to_response(
        {
            "created": 1700000000,
            "model": "deepseek-v4.1-flash",
            "choices": [
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length",
                }
            ],
        },
        {"model": "deepseek-v4.1-flash", "input": "long"},
    )
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["error"] is None
    assert response["output"] == []


def test_usage_estimation_uses_utf8_byte_counts() -> None:
    response = chat_to_response(
        {
            "created": 1700000000,
            "model": "deepseek-v4.1-flash",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "好"},
                    "finish_reason": "stop",
                }
            ],
        },
        {"model": "deepseek-v4.1-flash", "input": "豆包"},
    )
    assert response["usage"]["input_tokens"] == 8
    assert response["usage"]["output_tokens"] == 3
    assert response["usage"]["total_tokens"] == 11
    assert response["usage"]["estimated"] is True


@pytest.mark.asyncio
async def test_stream_text_is_incremental_and_has_stable_event_order() -> None:
    chunks = [
        sse(chat_chunk(content="你")),
        sse(chat_chunk(content="好")),
        sse(
            chat_chunk(
                finish_reason="stop",
                choices=[
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            )
        ),
        sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
            }
        ),
        sse("[DONE]"),
    ]
    _, events = await collect(chunks)
    types = [event["type"] for event in events]
    assert types[:4] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert types[4:6] == [
        "response.output_text.delta",
        "response.output_text.delta",
    ]
    assert types[-1] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "你好"
    assert events[-1]["response"]["usage"]["input_tokens"] == 7
    assert [event["sequence_number"] for event in events] == list(
        range(len(events))
    )
    assert events[0]["response"]["id"].startswith("resp_")
    final = assemble_response_stream(
        [
            f"event: {event['type']}\n"
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            for event in events
        ]
    )
    assert final == events[-1]["response"]


@pytest.mark.asyncio
async def test_stream_reasoning_mirror_is_not_duplicated() -> None:
    chunks = [
        sse(chat_chunk(reasoning="think")),
        sse(chat_chunk(content="answer")),
        sse(chat_chunk(finish_reason="stop")),
        sse("[DONE]"),
    ]
    _, events = await collect(chunks)
    reasoning_deltas = [
        event["delta"]
        for event in events
        if event["type"] == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas == ["think"]
    final = events[-1]["response"]
    assert final["output"][0]["type"] == "reasoning"
    assert final["output"][0]["summary"][0]["text"] == "think"
    assert final["output"][1]["type"] == "message"
    assert final["output"][1]["content"][0]["text"] == "answer"


@pytest.mark.asyncio
async def test_stream_tool_arguments_can_arrive_in_multiple_deltas() -> None:
    chunks = [
        sse(
            chat_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"ci',
                        },
                    }
                ]
            )
        ),
        sse(
            chat_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": 'ty":"北京"}'},
                    }
                ]
            )
        ),
        sse(chat_chunk(finish_reason="tool_calls")),
        sse("[DONE]"),
    ]
    _, events = await collect(chunks)
    deltas = [
        event["delta"]
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    ]
    assert deltas == ['{"ci', 'ty":"北京"}']
    done = next(
        event
        for event in events
        if event["type"] == "response.function_call_arguments.done"
    )
    assert done["arguments"] == '{"city":"北京"}'
    final = events[-1]["response"]
    assert final["output"][0]["type"] == "function_call"
    assert final["output"][0]["arguments"] == '{"city":"北京"}'
    assert final["output"][0]["call_id"] == "call_1"


@pytest.mark.asyncio
async def test_stream_error_terminates_without_completed_event() -> None:
    chunks = [
        sse(chat_chunk(content="partial")),
        sse(
            {
                "error": {
                    "message": "upstream failed",
                    "type": "upstream_error",
                    "code": "upstream_failure",
                }
            }
        ),
        sse(chat_chunk(content="must-not-appear")),
    ]
    _, events = await collect(chunks)
    types = [event["type"] for event in events]
    assert "response.completed" not in types
    assert "error" in types
    assert types[-1] == "response.failed"
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["code"] == "upstream_failure"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "partial"


@pytest.mark.asyncio
async def test_stream_gateway_error_preserves_code_without_completed_event() -> None:
    class RateLimited(GatewayError):
        status_code = 429
        error_type = "rate_limit_error"
        code = "upstream_rate_limited"

    async def source() -> AsyncIterator[str]:
        yield sse(chat_chunk(content="partial"))
        raise RateLimited("slow down")

    events = [
        json.loads(event.split("\n", 2)[1][6:])
        async for event in chat_stream_to_responses(
            source(),
            {"model": "deepseek-v4.1-flash", "input": "hello"},
        )
    ]
    assert "response.completed" not in [event["type"] for event in events]
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"] == {
        "code": "upstream_rate_limited",
        "message": "slow down",
        "param": None,
        "type": "rate_limit_error",
    }


@pytest.mark.asyncio
async def test_stream_unexpected_error_does_not_echo_internal_details() -> None:
    secret = "https://internal.example/token=super-secret"

    async def source() -> AsyncIterator[str]:
        if False:
            yield ""
        raise RuntimeError(secret)

    events = [
        json.loads(event.split("\n", 2)[1][6:])
        async for event in chat_stream_to_responses(
            source(),
            {"model": "deepseek-v4.1-flash", "input": "hello"},
        )
    ]
    assert "response.completed" not in [event["type"] for event in events]
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "stream_error"
    assert events[-1]["response"]["error"]["message"] == (
        "Chat stream failed unexpectedly"
    )
    assert secret not in json.dumps(events, ensure_ascii=False)


@pytest.mark.asyncio
async def test_stream_length_with_empty_content_is_incomplete() -> None:
    chunks = [
        sse(chat_chunk(finish_reason="length")),
        sse("[DONE]"),
    ]
    _, events = await collect(chunks)
    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["status"] == "incomplete"
    assert events[-1]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    assert events[-1]["response"]["output"] == []


def test_response_output_replay_keeps_reasoning_and_arguments() -> None:
    response = {
        "object": "response",
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "summary"}],
                "content": [{"type": "reasoning_text", "text": "raw"}],
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "checking",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            },
        ],
    }
    messages = response_output_to_messages(response)
    assert messages[0].reasoning_content == "raw"
    assert messages[1].content == "checking"
    assert messages[2].tool_calls[0]["function"]["arguments"] == '{"q":"x"}'


def test_response_output_replay_preserves_string_message_content() -> None:
    messages = response_output_to_messages(
        {
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": "plain replay",
                }
            ]
        }
    )
    assert messages[0].content == "plain replay"


def test_compaction_item_shape_is_accepted_by_input_conversion() -> None:
    def decode(item: dict) -> list[ChatMessage]:
        assert item["object"] == "response.compaction" if "object" in item else True
        return [ChatMessage(role="assistant", content="checkpoint")]

    request = responses_to_chat(
        {
            "input": [
                {
                    "id": "cmp_1",
                    "type": "compaction",
                    "encrypted_content": "token",
                }
            ]
        },
        "deepseek-v4.1-flash",
        decode_compaction=decode,
    )
    assert request.messages == [ChatMessage(role="assistant", content="checkpoint")]
