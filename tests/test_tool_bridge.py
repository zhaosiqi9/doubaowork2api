from __future__ import annotations

import json

import pytest

from doubao_work2api.errors import GatewayError
from doubao_work2api.models import ChatMessage
from doubao_work2api.tool_bridge import (
    ToolBridgeRequestError,
    ToolBridgeResponseError,
    parse_tool_response,
    render_tool_history,
    render_tool_instructions,
    validate_tools,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up current weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clock",
            "description": "Read a clock.",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
            },
        },
    },
]


def envelope(*calls: tuple[str, object], content: str | None = None) -> str:
    return json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [
                {"name": name, "arguments": json.dumps(arguments)}
                for name, arguments in calls
            ],
            "content": content,
        }
    )


def test_validate_tools_accepts_none_and_multiple_tools() -> None:
    validate_tools(None)
    validate_tools(TOOLS, "auto", True)
    validate_tools(TOOLS, "none", False)
    validate_tools(TOOLS, "auto", None)


@pytest.mark.parametrize(
    ("tools", "choice", "parallel", "code"),
    [
        ([{"type": "function"}], None, True, "invalid_tools"),
        ([TOOLS[0], TOOLS[0]], None, True, "duplicate_tool"),
        ([{**TOOLS[0], "function": {**TOOLS[0]["function"], "name": "bad name"}}], None, True, "invalid_tool_name"),
        (None, "required", True, "tool_choice_without_tools"),
        (TOOLS, {"type": "function", "function": {"name": "missing"}}, True, "undeclared_tool_choice"),
        (TOOLS, "auto", "yes", "invalid_parallel_tool_calls"),
    ],
)
def test_validate_tools_rejects_malformed_requests(
    tools: object, choice: object, parallel: object, code: str
) -> None:
    with pytest.raises(ToolBridgeRequestError) as raised:
        validate_tools(tools, choice, parallel)  # type: ignore[arg-type]
    assert raised.value.status_code == 400
    assert raised.value.code == code


def test_instructions_are_empty_without_tools_and_include_strict_envelope() -> None:
    assert render_tool_instructions(None) == ""
    instructions = render_tool_instructions(TOOLS)
    assert '"type":"tool_calls"' in instructions
    assert '"name":"weather"' in instructions
    assert "ordinary text" in instructions
    assert "never invent a tool name" in instructions


def test_none_instructions_forbid_envelope() -> None:
    instructions = render_tool_instructions(TOOLS, "none")
    assert "Do not call any tool" in instructions
    assert "answer with ordinary text" in instructions


def test_parse_multiple_calls_and_keep_final_text() -> None:
    text = envelope(
        ("weather", {"city": "Shanghai"}),
        ("clock", {"zone": "Asia/Shanghai"}),
        content="Checking both.",
    )
    content, calls = parse_tool_response(text, TOOLS)
    assert content == "Checking both."
    assert [call["function"]["name"] for call in calls] == ["weather", "clock"]
    assert calls[0]["type"] == "function"
    assert isinstance(calls[0]["function"]["arguments"], str)
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Shanghai"}
    assert calls[0]["id"] != calls[1]["id"]


def test_parse_accepts_complete_text_after_chunking() -> None:
    text = envelope(("weather", {"city": "Beijing"}))
    chunks = [text[:7], text[7:29], text[29:]]
    content, calls = parse_tool_response("".join(chunks), TOOLS, "required")
    assert content is None
    assert calls[0]["function"]["name"] == "weather"


def test_parse_json_fence_and_rejects_surrounding_prose() -> None:
    text = envelope(("clock", {"zone": "UTC"}))
    fenced = f"```json\n{text}\n```"
    assert parse_tool_response(fenced, TOOLS) == (fenced, [])
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(fenced, TOOLS, "required")
    assert raised.value.code == "tool_call_required"
    assert parse_tool_response(f"Here is the call:\n{text}", TOOLS) == (
        f"Here is the call:\n{text}",
        [],
    )


def test_parse_does_not_mistake_ordinary_json_or_code_for_a_call() -> None:
    ordinary = '{"answer": 42}'
    assert parse_tool_response(ordinary, TOOLS) == (ordinary, [])
    code = "```python\nprint('hello')\n```"
    assert parse_tool_response(code, TOOLS) == (code, [])
    prose_with_fake = 'I would call {"type":"tool_calls"} but I will not.'
    assert parse_tool_response(prose_with_fake, TOOLS) == (prose_with_fake, [])


def test_required_and_named_choice_are_enforced() -> None:
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response("plain answer", TOOLS, "required")
    assert raised.value.code == "tool_call_required"

    wrong = envelope(("clock", {"zone": "UTC"}))
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(
            wrong,
            TOOLS,
            {"type": "function", "function": {"name": "weather"}},
        )
    assert raised.value.code == "tool_choice_violation"


def test_none_does_not_parse_envelopes() -> None:
    text = envelope(("weather", {"city": "Shanghai"}))
    assert parse_tool_response(text, TOOLS, "none") == (text, [])


def test_rejects_undeclared_and_malformed_calls() -> None:
    undeclared = envelope(("delete_everything", {}))
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(undeclared, TOOLS)
    assert raised.value.code == "undeclared_tool"

    malformed = json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [{"name": "weather", "arguments": "{not json"}],
        }
    )
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(malformed, TOOLS)
    assert raised.value.code == "invalid_tool_arguments"

    non_object = envelope(("weather", ["Shanghai"]))
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(non_object, TOOLS)
    assert raised.value.code == "invalid_tool_arguments"


def test_invalid_model_tool_name_is_a_502_response_error() -> None:
    malformed = json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [{"name": "bad name", "arguments": "{}"}],
        }
    )
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(malformed, TOOLS)
    assert raised.value.status_code == 502
    assert raised.value.code == "invalid_tool_response"


def test_rejects_extra_fields_and_parallel_calls_when_disabled() -> None:
    raw = json.loads(envelope(("weather", {"city": "Shanghai"})))
    raw["extra"] = True
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(json.dumps(raw), TOOLS)
    assert raised.value.code == "invalid_tool_response"

    two_calls = envelope(
        ("weather", {"city": "Shanghai"}),
        ("clock", {"zone": "UTC"}),
    )
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(two_calls, TOOLS, "auto", False)
    assert raised.value.code == "parallel_tool_calls_disabled"


def test_rejects_non_finite_json_numbers_as_response_errors() -> None:
    malformed = (
        '{"type":"tool_calls","tool_calls":'
        '[{"name":"weather","arguments":"{\\"value\\":NaN}"}]}'
    )
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(malformed, TOOLS)
    assert raised.value.status_code == 502
    assert raised.value.code == "invalid_tool_arguments"


def test_rejects_oversized_response_before_parsing() -> None:
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response("x" * 8_388_609, TOOLS)
    assert raised.value.status_code == 502
    assert raised.value.code == "response_too_large"


def test_no_tools_leave_large_or_unusual_text_unchanged() -> None:
    text = "x" * 8_388_609
    assert parse_tool_response(text, None) == (text, [])


def test_render_assistant_and_tool_history() -> None:
    assistant = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"city":"Shanghai"}',
                },
            }
        ],
    )
    history = render_tool_history(assistant)
    assert history is not None
    assert json.loads(history)["tool_calls"][0]["id"] == "call_1"
    assert json.loads(history)["tool_calls"][0]["function"]["arguments"] == '{"city":"Shanghai"}'

    result = ChatMessage(
        role="tool",
        tool_call_id="call_1",
        name="weather",
        content="sunny",
    )
    tool_history = render_tool_history(result)
    assert tool_history is not None
    assert json.loads(tool_history) == {
        "type": "tool_result",
        "tool_call_id": "call_1",
        "name": "weather",
        "content": "sunny",
    }
    assert render_tool_history(ChatMessage(role="user", content="hello")) is None


def test_render_history_rejects_malformed_tool_data() -> None:
    with pytest.raises(ToolBridgeRequestError) as raised:
        render_tool_history(ChatMessage(role="tool", content="missing id"))
    assert raised.value.code == "invalid_tool_history"


def test_render_history_rejects_excessive_json_depth_as_request_error() -> None:
    arguments = "{}"
    for _ in range(70):
        arguments = json.dumps({"nested": json.loads(arguments)})
    with pytest.raises(ToolBridgeRequestError) as raised:
        render_tool_history(
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_deep",
                        "type": "function",
                        "function": {"name": "weather", "arguments": arguments},
                    }
                ],
            )
        )
    assert raised.value.status_code == 400

    history = render_tool_history(
        ChatMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": {}},
                }
            ],
        )
    )
    assert history is not None
    assert json.loads(history)["tool_calls"][0]["function"]["arguments"] == "{}"

    with pytest.raises(ToolBridgeRequestError) as raised:
        render_tool_history(
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"value":NaN}',
                        },
                    }
                ],
            )
        )
    assert raised.value.code == "invalid_tool_history"


def test_errors_are_gateway_errors_with_openai_shape() -> None:
    request_error = ToolBridgeRequestError("bad tools")
    response_error = ToolBridgeResponseError("bad output")
    assert isinstance(request_error, GatewayError)
    assert request_error.status_code == 400
    assert response_error.status_code == 502
    assert request_error.as_openai()["error"]["code"] == "invalid_tool_request"


def test_parse_uses_fresh_uuid_call_ids_per_response() -> None:
    text = envelope(("weather", {"city": "Shanghai"}))
    first = parse_tool_response(text, TOOLS, parallel_tool_calls=None)[1]
    second = parse_tool_response(text, TOOLS, parallel_tool_calls=None)[1]
    assert first[0]["id"] != second[0]["id"]
    assert first[0]["id"].startswith("call_")


def test_parse_preserves_valid_argument_string_formatting() -> None:
    arguments = '{ "city": "Shanghai", "unit": "c" }'
    text = json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [{"name": "weather", "arguments": arguments}],
            "content": None,
        }
    )
    content, calls = parse_tool_response(text, TOOLS)
    assert content is None
    assert calls[0]["function"]["arguments"] == arguments


def test_parse_accepts_object_arguments_and_missing_empty_arguments() -> None:
    raw = {
        "type": "tool_calls",
        "tool_calls": [
            {"name": "weather", "arguments": {"city": "Beijing"}},
            {"name": "clock"},
        ],
        "content": "Working.",
    }
    content, calls = parse_tool_response(json.dumps(raw), TOOLS)
    assert content == "Working."
    assert calls[0]["function"]["arguments"] == '{"city":"Beijing"}'
    assert calls[1]["function"]["arguments"] == "{}"


def test_parse_unwraps_multiple_json_string_layers() -> None:
    inner = envelope(("weather", {"city": "Shanghai"}))
    text = json.dumps(json.dumps(inner))
    content, calls = parse_tool_response(text, TOOLS)
    assert content is None
    assert calls[0]["function"]["name"] == "weather"


def test_parse_repairs_only_trailing_commas_in_explicit_envelope() -> None:
    text = (
        '{"type":"tool_calls","tool_calls":'
        '[{"name":"weather","arguments":"{\\"city\\":\\"Shanghai\\"}",},],}'
    )
    content, calls = parse_tool_response(text, TOOLS)
    assert content is None
    assert calls[0]["function"]["name"] == "weather"

    ordinary = '{"answer": 42,}'
    assert parse_tool_response(ordinary, TOOLS) == (ordinary, [])


def test_parse_does_not_treat_nested_ordinary_json_as_tool_calls() -> None:
    ordinary = json.dumps({"answer": envelope(("weather", {"city": "Shanghai"}))})
    assert parse_tool_response(ordinary, TOOLS) == (ordinary, [])


def test_parse_rejects_python_literals_and_broken_argument_json() -> None:
    python_literal = (
        '{"type":"tool_calls","tool_calls":'
        "[{'name':'weather','arguments':{'city':'Shanghai'}}]}"
    )
    assert parse_tool_response(python_literal, TOOLS) == (
        python_literal,
        [],
    )

    broken = json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [
                {"name": "weather", "arguments": '{"city": }'}
            ],
        }
    )
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(broken, TOOLS)
    assert raised.value.code == "invalid_tool_arguments"


def test_parse_still_rejects_undeclared_functions_after_repair() -> None:
    text = (
        '{"type":"tool_calls","tool_calls":'
        '[{"name":"missing","arguments":{},},],}'
    )
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(text, TOOLS)
    assert raised.value.status_code == 502
    assert raised.value.code == "undeclared_tool"


def test_parse_required_semantics_survive_failed_repair() -> None:
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(
            '{"type":"tool_calls","tool_calls": [', TOOLS, "required"
        )
    assert raised.value.code == "invalid_tool_response"


def test_parse_does_not_execute_python_literal_arguments() -> None:
    raw = {
        "type": "tool_calls",
        "tool_calls": [
            {"name": "weather", "arguments": "{'city': 'Shanghai'}"}
        ],
    }
    with pytest.raises(ToolBridgeResponseError) as raised:
        parse_tool_response(json.dumps(raw), TOOLS)
    assert raised.value.code == "invalid_tool_arguments"


def test_render_history_keeps_argument_text_and_accepts_objects() -> None:
    arguments = '{ "city": "Shanghai", "unit": "c" }'
    assistant = ChatMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": arguments},
            }
        ],
    )
    history = render_tool_history(assistant)
    assert history is not None
    assert (
        json.loads(history)["tool_calls"][0]["function"]["arguments"]
        == arguments
    )

    object_arguments = ChatMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {"city": "Shanghai"},
                },
            }
        ],
    )
    history = render_tool_history(object_arguments)
    assert history is not None
    assert json.loads(history)["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"Shanghai"}'
    )


def test_render_history_missing_empty_arguments_become_object_string() -> None:
    assistant = ChatMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather"},
            }
        ],
    )
    history = render_tool_history(assistant)
    assert history is not None
    assert json.loads(history)["tool_calls"][0]["function"]["arguments"] == "{}"
