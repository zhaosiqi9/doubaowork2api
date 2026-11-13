from __future__ import annotations

import copy
import json

import pytest

from doubao_work2api.history import (
    normalize_messages,
    render_reasoning_history,
)
from doubao_work2api.models import ChatMessage
from doubao_work2api.tool_bridge import ToolBridgeRequestError


def call(
    call_id: str, name: str = "lookup", arguments: object = "{}"
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def tool_result(call_id: str, content: str) -> ChatMessage:
    return ChatMessage(role="tool", tool_call_id=call_id, content=content)


def test_normalize_merges_adjacent_assistant_calls_and_reorders_results() -> None:
    messages = [
        ChatMessage(role="user", content="Check both."),
        ChatMessage(role="assistant", content="One.", tool_calls=[call("a")]),
        ChatMessage(role="assistant", content="Two.", tool_calls=[call("b")]),
        tool_result("b", "second"),
        tool_result("a", "first"),
        ChatMessage(role="assistant", content="Done."),
    ]

    normalized = normalize_messages(messages)

    assert [message.role for message in normalized] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert normalized[1].content == "One.\n\nTwo."
    assert [item["id"] for item in normalized[1].tool_calls] == ["a", "b"]
    assert [item.tool_call_id for item in normalized[2:4]] == ["a", "b"]
    assert [item.content for item in normalized[2:4]] == ["first", "second"]
    assert [item.name for item in normalized[2:4]] == ["lookup", "lookup"]


def test_normalize_does_not_mutate_input_messages_or_nested_values() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content="Checking.",
            reasoning_content="  keep  ",
            reasoning_details=[{"type": "summary", "text": "why"}],
            tool_calls=[call("a", arguments='{"city": "Shanghai"}')],
        ),
        tool_result("a", "sunny"),
    ]
    before = copy.deepcopy(messages)

    normalized = normalize_messages(messages)
    normalized[0].content = "changed"
    normalized[0].tool_calls[0]["function"]["arguments"] = "{}"
    normalized[1].content = "changed"

    assert [message.model_dump() for message in messages] == [
        message.model_dump() for message in before
    ]
    assert messages[0].tool_calls[0]["function"]["arguments"] == (
        '{"city": "Shanghai"}'
    )


def test_normalize_keeps_valid_argument_string_byte_for_byte() -> None:
    arguments = '{ "z": 1, "a": [2, 3] }'
    normalized = normalize_messages(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[call("a", arguments=arguments)],
            ),
            tool_result("a", "ok"),
        ]
    )
    assert normalized[0].tool_calls[0]["function"]["arguments"] == arguments


def test_normalize_converts_object_arguments_without_reencoding_strings() -> None:
    normalized = normalize_messages(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[call("a", arguments={"city": "上海", "n": 1})],
            ),
            tool_result("a", "ok"),
        ]
    )
    assert normalized[0].tool_calls[0]["function"]["arguments"] == (
        '{"city":"上海","n":1}'
    )


def test_normalize_fills_missing_tool_call_id_only_for_unique_pending() -> None:
    normalized = normalize_messages(
        [
            ChatMessage(role="assistant", tool_calls=[call("only")]),
            ChatMessage(role="tool", content="result"),
        ]
    )
    assert normalized[1].tool_call_id == "only"


def test_normalize_rejects_ambiguous_missing_id_or_duplicate_result() -> None:
    ambiguous = [
        ChatMessage(role="assistant", tool_calls=[call("a"), call("b")]),
        ChatMessage(role="tool", content="result"),
    ]
    with pytest.raises(ToolBridgeRequestError) as raised:
        normalize_messages(ambiguous)
    assert raised.value.status_code == 400
    assert raised.value.code == "invalid_tool_history"

    duplicate = [
        ChatMessage(role="assistant", tool_calls=[call("a")]),
        tool_result("a", "first"),
        tool_result("a", "second"),
    ]
    with pytest.raises(ToolBridgeRequestError) as raised:
        normalize_messages(duplicate)
    assert raised.value.code == "invalid_tool_history"


def test_normalize_rejects_orphans_unknown_ids_and_id_conflicts() -> None:
    with pytest.raises(ToolBridgeRequestError):
        normalize_messages([tool_result("orphan", "result")])

    with pytest.raises(ToolBridgeRequestError):
        normalize_messages(
            [
                ChatMessage(role="assistant", tool_calls=[call("a")]),
                tool_result("other", "result"),
            ]
        )

    with pytest.raises(ToolBridgeRequestError):
        normalize_messages(
            [
                ChatMessage(role="assistant", tool_calls=[call("same")]),
                ChatMessage(role="assistant", tool_calls=[call("same")]),
            ]
        )


def test_normalize_never_crosses_a_user_turn() -> None:
    messages = [
        ChatMessage(role="assistant", tool_calls=[call("a")]),
        ChatMessage(role="user", content="Next question."),
        tool_result("a", "late result"),
    ]
    with pytest.raises(ToolBridgeRequestError) as raised:
        normalize_messages(messages)
    assert raised.value.code == "invalid_tool_history"


def test_normalize_does_not_invent_missing_results() -> None:
    normalized = normalize_messages(
        [ChatMessage(role="assistant", tool_calls=[call("a")])]
    )
    assert len(normalized) == 1
    assert len(normalized[0].tool_calls) == 1


def test_normalize_preserves_reasoning_fields_and_tool_text() -> None:
    reasoning_details = [{"type": "summary", "text": "  "}]
    normalized = normalize_messages(
        [
            ChatMessage(
                role="assistant",
                content=None,
                reasoning_content="",
                reasoning="full chain",
                reasoning_details=reasoning_details,
                tool_calls=[call("a")],
            ),
            tool_result("a", '{"raw": "  text  "}'),
        ]
    )
    assert normalized[0].reasoning_content == ""
    assert normalized[0].reasoning == "full chain"
    assert normalized[0].reasoning_details == reasoning_details
    assert normalized[1].content == '{"raw": "  text  "}'


def test_normalize_legacy_function_call_and_function_role() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            function_call={"name": "lookup", "arguments": '{"key": "A"}'},
        ),
        ChatMessage.model_construct(
            role="function", name="lookup", content="legacy result"
        ),
    ]

    normalized = normalize_messages(messages)

    assert normalized[0].role == "assistant"
    assert "function_call" not in (normalized[0].__pydantic_extra__ or {})
    generated_id = normalized[0].tool_calls[0]["id"]
    assert generated_id.startswith("call_legacy_")
    assert normalized[1].role == "tool"
    assert normalized[1].tool_call_id == generated_id
    assert normalized[1].name == "lookup"
    assert normalized[1].content == "legacy result"


def test_normalize_legacy_function_role_with_explicit_id() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            tool_calls=[call("a")],
        ),
        ChatMessage.model_construct(
            role="function",
            function_call_id="a",
            name="lookup",
            content="result",
        ),
    ]
    normalized = normalize_messages(messages)
    assert normalized[1].role == "tool"
    assert normalized[1].tool_call_id == "a"


def test_normalize_rejects_legacy_call_conflicts() -> None:
    with pytest.raises(ToolBridgeRequestError):
        normalize_messages(
            [
                ChatMessage(
                    role="assistant",
                    tool_calls=[call("same")],
                    function_call={
                        "id": "same",
                        "name": "lookup",
                        "arguments": "{}",
                    },
                )
            ]
        )


def test_normalize_keeps_non_tool_text_and_roles_unchanged() -> None:
    messages = [
        ChatMessage(role="system", content="System."),
        ChatMessage(role="user", content="Hello."),
        ChatMessage(role="assistant", content="Hi."),
    ]
    normalized = normalize_messages(messages)
    assert [message.model_dump() for message in normalized] == [
        message.model_dump() for message in messages
    ]


def test_render_reasoning_history_is_explicit_and_preserves_whitespace() -> None:
    message = ChatMessage(
        role="assistant",
        content="Visible answer.",
        reasoning_content="  ",
        reasoning_details=[{"type": "summary", "text": "why"}],
    )
    rendered = render_reasoning_history(message)
    assert rendered is not None
    assert rendered.startswith("<historical_reasoning>\n")
    assert "not user content" in rendered
    assert "  " in rendered
    assert "why" in rendered
    assert rendered.endswith("\n</historical_reasoning>")
    assert "Visible answer." not in rendered


def test_render_reasoning_history_supports_text_lists_and_dedupes() -> None:
    message = ChatMessage(
        role="assistant",
        reasoning_content=[
            {"type": "text", "text": "step one"},
            {"type": "text", "text": "step two"},
        ],
        reasoning="step one\nstep two",
    )
    rendered = render_reasoning_history(message)
    assert rendered is not None
    assert rendered.count("step one") == 1
    assert "step two" in rendered


def test_render_reasoning_history_none_for_non_reasoning_or_non_assistant() -> None:
    assert render_reasoning_history(
        ChatMessage(role="assistant", content="answer")
    ) is None
    assert render_reasoning_history(
        ChatMessage(role="user", content="question", reasoning_content="why")
    ) is None


def test_reasoning_details_dict_text_is_preserved() -> None:
    message = ChatMessage(
        role="assistant",
        reasoning_details=[
            {"type": "reasoning.text", "text": "keep  this"},
            {"type": "summary", "summary": "short"},
        ],
    )
    rendered = render_reasoning_history(message)
    assert rendered is not None
    assert "keep  this" in rendered
    assert "short" in rendered


def test_normalize_rejects_non_chat_messages() -> None:
    with pytest.raises(ToolBridgeRequestError):
        normalize_messages([{"role": "user", "content": "x"}])  # type: ignore[list-item]
