from __future__ import annotations

import json

import pytest

from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.request_contract import (
    ContractRequestError,
    ContractResponseError,
    enforce_contract,
    estimate_tokens,
    get_output_limit,
    limit_output,
    render_contract_instructions,
    validate_contract,
    validate_tool_arguments,
)


def make_request(**overrides) -> ChatCompletionRequest:
    values = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "Return data"}],
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def json_schema(
    schema: dict,
    *,
    strict: bool = False,
    name: str = "result",
) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": strict,
        },
    }


def test_json_object_accepts_fenced_object_and_serializes_stably() -> None:
    request = make_request(response_format={"type": "json_object"})
    text = '```json\n{"z":1,"name":"豆包"}\n```'
    assert enforce_contract(text, request) == '{"name":"豆包","z":1}'


def test_json_object_requires_top_level_object() -> None:
    request = make_request(response_format={"type": "json_object"})
    with pytest.raises(ContractResponseError) as caught:
        enforce_contract("[1, 2, 3]", request)
    assert caught.value.status_code == 502
    assert caught.value.code == "json_object_required"


def test_nested_schema_and_local_defs_reference_validate() -> None:
    schema = {
        "$defs": {
            "item": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/item"},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    request = make_request(
        response_format=json_schema(schema, strict=True)
    )
    validate_contract(request)
    text = '{"items":[{"id":1},{"id":2}]}'
    assert json.loads(enforce_contract(text, request)) == {
        "items": [{"id": 1}, {"id": 2}]
    }
    instructions = render_contract_instructions(request)
    assert "#/$defs/item" in instructions
    assert "strict" in instructions.lower()


@pytest.mark.parametrize(
    "ref",
    [
        "https://example.test/schema.json",
        "file:///tmp/schema.json",
        "#/$defs/item",
        "#/$defs/missing",
    ],
)
def test_schema_reference_policy(ref: str) -> None:
    schema = {
        "$defs": {"item": {"type": "integer"}},
        "type": "object",
        "properties": {"value": {"$ref": ref}},
    }
    request = make_request(response_format=json_schema(schema))
    if ref == "#/$defs/item":
        validate_contract(request)
        assert enforce_contract('{"value":1}', request) == '{"value":1}'
        return
    with pytest.raises(ContractRequestError) as caught:
        validate_contract(request)
    expected = (
        "invalid_json_schema"
        if ref == "#/$defs/missing"
        else "external_schema_reference"
    )
    assert caught.value.status_code == 400
    assert caught.value.code == expected


def test_dynamic_ref_must_be_local() -> None:
    schema = {
        "$dynamicAnchor": "node",
        "type": "object",
        "properties": {
            "child": {"$dynamicRef": "https://example.test/node"}
        },
    }
    request = make_request(response_format=json_schema(schema))
    with pytest.raises(ContractRequestError) as caught:
        validate_contract(request)
    assert caught.value.code == "external_schema_reference"


@pytest.mark.parametrize(
    "response_format",
    [None, {"type": "text"}],
)
def test_null_and_text_formats_are_compatible(response_format) -> None:
    request = make_request(response_format=response_format)
    validate_contract(request)
    assert render_contract_instructions(request) == ""
    assert enforce_contract("plain text", request) == "plain text"


def test_invalid_response_format_is_a_400_without_content_echo() -> None:
    secret = "do-not-echo-this-user-content"
    request = make_request(
        messages=[{"role": "user", "content": secret}],
        response_format={"type": "json_schema", "json_schema": {"name": "x"}},
    )
    with pytest.raises(ContractRequestError) as caught:
        validate_contract(request)
    assert caught.value.status_code == 400
    assert secret not in caught.value.message


def test_json_fence_requires_a_complete_fence() -> None:
    request = make_request(response_format={"type": "json_object"})
    with pytest.raises(ContractResponseError) as caught:
        enforce_contract("```json\n{\"ok\":true}", request)
    assert caught.value.code == "invalid_json_response"


@pytest.mark.parametrize("value", [0, -1])
def test_output_limit_rejects_non_positive_integers(value: int) -> None:
    request = make_request(max_tokens=value)
    with pytest.raises(ContractRequestError) as caught:
        get_output_limit(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_output_limit"


@pytest.mark.parametrize("name", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", ["10", True, 1.5])
def test_output_limit_rejects_uncoerced_non_integers(
    name: str, value
) -> None:
    request = make_request()
    request.__dict__[name] = value
    request.__pydantic_fields_set__.add(name)
    with pytest.raises(ContractRequestError) as caught:
        get_output_limit(request)
    assert caught.value.code == "invalid_output_limit"


def test_output_limit_aliases_must_agree() -> None:
    request = make_request(max_tokens=10, max_completion_tokens=11)
    with pytest.raises(ContractRequestError) as caught:
        get_output_limit(request)
    assert caught.value.code == "conflicting_output_limit"
    matching = make_request(max_tokens=10, max_completion_tokens=10)
    assert get_output_limit(matching) == 10
    preferred = make_request(max_completion_tokens=12)
    assert get_output_limit(preferred) == 12


def test_estimate_tokens_is_utf8_byte_upper_bound() -> None:
    assert estimate_tokens("abc") == 3
    assert estimate_tokens("豆包") == 6
    assert estimate_tokens("😀") == 4
    assert estimate_tokens("") == 0
    text = "a豆😀"
    assert estimate_tokens(text) == sum(
        estimate_tokens(character) for character in text
    )


def test_limit_output_uses_earliest_complete_stop() -> None:
    request = make_request(stop=["END", "STOP"])
    assert limit_output("before STOP and END", request) == (
        "before ",
        "stop",
    )
    assert limit_output("no marker", request) == ("no marker", "stop")


def test_limit_output_supports_split_stop_text() -> None:
    request = make_request(stop=["</end>"])
    chunks = ["answer</en", "d>ignored"]
    assert limit_output("".join(chunks), request) == ("answer", "stop")


@pytest.mark.parametrize(
    "stop",
    ["", [], ["a", "b", "c", "d", "e"], ["ok", ""], [1]],
)
def test_stop_validation_rejects_bad_values(stop) -> None:
    request = make_request()
    request.stop = stop
    request.model_fields_set.add("stop")
    with pytest.raises(ContractRequestError) as caught:
        validate_contract(request)
    assert caught.value.code == "invalid_stop"


def test_limit_output_does_not_split_unicode() -> None:
    request = make_request(max_tokens=4)
    text, reason = limit_output("😀😀", request)
    assert text == "😀"
    assert reason == "length"
    assert estimate_tokens(text) <= 4


def test_limit_output_stop_takes_precedence_over_budget() -> None:
    request = make_request(max_tokens=3, stop="STOP")
    text, reason = limit_output("abcSTOPtail", request)
    assert text == "abc"
    assert reason == "stop"


def test_limit_output_can_make_json_incomplete_for_parent_buffering() -> None:
    request = make_request(
        max_tokens=8,
        response_format={"type": "json_object"},
    )
    text, reason = limit_output('{"message":"too long"}', request)
    assert reason == "length"
    assert text == '{"messag'
    with pytest.raises(ContractResponseError):
        enforce_contract(text, request)


def test_tool_strict_schema_validation_and_non_strict_object_check() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "strict_lookup",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["city", "count"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "loose_lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
    ]
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "strict_lookup",
                "arguments": '{"city":"上海","count":2}',
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "loose_lookup",
                "arguments": '{"city":"上海","extra":true}',
            },
        },
    ]
    validate_contract(make_request(tools=tools))
    validate_tool_arguments(calls, tools)

    calls[0]["function"]["arguments"] = '{"city":"上海"}'
    with pytest.raises(ContractResponseError) as caught:
        validate_tool_arguments(calls, tools)
    assert caught.value.status_code == 502
    assert caught.value.code == "invalid_tool_arguments"

    calls[0]["function"]["arguments"] = '{"city":"上海","count":2}'
    calls[1]["function"]["arguments"] = '["not", "an", "object"]'
    with pytest.raises(ContractResponseError) as caught:
        validate_tool_arguments(calls, tools)
    assert caught.value.code == "invalid_tool_arguments"


def test_tool_schema_and_json_schema_can_coexist_without_content() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    request = make_request(
        response_format=json_schema(
            {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            }
        ),
        tools=tools,
    )
    validate_contract(request)
    validate_tool_arguments(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"key":"alpha"}',
                },
            }
        ],
        tools,
    )


def test_errors_have_openai_shapes_and_do_not_echo_payload() -> None:
    secret = "secret-model-output-123"
    request = make_request(
        response_format=json_schema(
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        )
    )
    with pytest.raises(ContractResponseError) as caught:
        enforce_contract(json.dumps({"value": secret}), request)
    body = caught.value.as_openai()
    assert body["error"]["type"] == "upstream_error"
    assert body["error"]["code"] == "schema_validation_failed"
    assert secret not in body["error"]["message"]

    bad_request = make_request(
        response_format={"type": "json_schema", "json_schema": {}}
    )
    with pytest.raises(ContractRequestError) as caught:
        validate_contract(bad_request)
    assert caught.value.as_openai()["error"]["type"] == "invalid_request_error"
    assert caught.value.status_code == 400
