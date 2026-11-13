from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from doubao_work2api.accounts import Account, AccountPool
from doubao_work2api.errors import GatewayError, UnknownModelError
from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.protocol import (
    RequestValidationError,
    UnsupportedContentError,
    extract_text_fragments,
    phase_markers,
    render_prompt,
    resolve_reasoning_effort,
    upstream_payload,
    validate_request,
)
from doubao_work2api.service import ChatService

DEEPSEEK_KEY = "person-deepseek-v4-1-flash-261017"


def make_request(**overrides) -> ChatCompletionRequest:
    values = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def make_payload(request: ChatCompletionRequest, tmp_path, **overrides):
    values = {
        "model_key": DEEPSEEK_KEY,
        "workspace": tmp_path,
        "device_name": "LINUX",
        "app_id": "1044603",
        "device_id": "device-1",
    }
    values.update(overrides)
    return upstream_payload(request, **values)


def test_render_prompt_separates_system_instruction() -> None:
    request = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
    )
    assert render_prompt(request.messages) == (
        "System instructions:\nBe concise.\n\nConversation:\nUser: Hello"
    )


def test_upstream_payload_uses_model_key_and_new_conversation(tmp_path) -> None:
    request = ChatCompletionRequest(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    payload = upstream_payload(
        request,
        model_key="person-deepseek-v4-1-flash-261017",
        workspace=tmp_path,
        device_name="LINUX",
        app_id="1044603",
        device_id="device-1",
    )
    assert payload["option"]["need_create_conversation"] is True
    assert payload["option"]["conversation_init_ext"]["model_item_key"] == (
        "person-deepseek-v4-1-flash-261017"
    )
    assert payload["option"]["model_config"]["reasoning_effort"] == 7
    assert payload["option"]["aggregate_params"]["model_item_key"] == (
        "person-deepseek-v4-1-flash-261017"
    )
    assert payload["option"]["general_task_param"]["client_option"]["os"] == (
        "Linux"
    )
    assert "device-1" in payload["option"]["conversation_init_ext"][
        "agent_task_param"
    ]


def test_extract_stream_chunk_and_phase_markers() -> None:
    event = {
        "_event": "STREAM_CHUNK",
        "patch_op": [
            {
                "patch_value": {
                    "content_block": [
                        {
                            "block_type": 10040,
                            "content": {"text_block": {"text": "thinking"}},
                        }
                    ]
                }
            }
        ],
    }
    assert phase_markers(event) == ["thinking"]
    assert list(extract_text_fragments(event)) == []

    content = {
        "_event": "STREAM_CHUNK",
        "patch_op": [
            {
                "patch_value": {
                    "content_block": [
                        {
                            "block_type": 10000,
                            "content": {"text_block": {"text": "hello"}},
                        }
                    ]
                }
            }
        ],
    }
    assert list(extract_text_fragments(content)) == [("content", "hello")]


def test_extract_chunk_delta_respects_thinking() -> None:
    event = {"_event": "CHUNK_DELTA", "text": "abc", "_thinking": True}
    assert list(extract_text_fragments(event)) == [("reasoning", "abc")]


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("low", 3),
        ("medium", 4),
        ("high", 5),
        ("xhigh", 6),
        ("max", 7),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
    ],
)
def test_reasoning_effort_maps_for_verified_model(
    effort: str | int, expected: int
) -> None:
    request = make_request(reasoning_effort=effort)
    validate_request(request)
    assert resolve_reasoning_effort(request) == expected


def test_reasoning_effort_defaults_to_existing_max(tmp_path) -> None:
    request = make_request()
    validate_request(request)
    assert resolve_reasoning_effort(request) == 7
    payload = make_payload(request, tmp_path)
    option = payload["option"]
    assert option["conversation_init_ext"]["reasoning_effort"] == "7"
    assert option["model_config"]["reasoning_effort"] == 7
    assert option["aggregate_params"]["reasoning_effort"] == "7"


@pytest.mark.parametrize("effort", ["minimal", 2, 8, True, "LOW "])
def test_invalid_reasoning_effort_is_rejected(effort) -> None:
    request = make_request(reasoning_effort=effort)
    with pytest.raises(GatewayError) as caught:
        validate_request(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_reasoning_effort"


def test_model_specific_reasoning_levels_are_checked() -> None:
    request = make_request(
        model="person-glm-5-3-261111",
        reasoning_effort="medium",
    )
    with pytest.raises(RequestValidationError) as caught:
        validate_request(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_reasoning_effort"


def test_model_uses_its_own_official_window_and_default(tmp_path) -> None:
    request = make_request(model="seed-lite-7b")
    payload = make_payload(
        request,
        tmp_path,
        model_key="seed-lite-7b",
    )
    model_config = payload["option"]["model_config"]
    assert model_config["reasoning_effort"] == 5
    assert model_config["model_extra_params"] == {"memory_profile": "256k", "total_window_size": "256000"}


def test_verified_model_gets_memory_params_in_all_effort_locations(tmp_path) -> None:
    request = make_request(reasoning_effort="xhigh")
    payload = make_payload(request, tmp_path)
    option = payload["option"]
    assert option["conversation_init_ext"]["reasoning_effort"] == "6"
    assert option["model_config"]["reasoning_effort"] == 6
    assert option["aggregate_params"]["reasoning_effort"] == "6"
    assert option["model_config"]["model_extra_params"] == {
        "memory_profile": "300k_230",
        "total_window_size": "300000",
    }


def test_upstream_payload_validates_direct_service_use(tmp_path) -> None:
    request = make_request(reasoning_effort="minimal")
    with pytest.raises(RequestValidationError):
        make_payload(request, tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}],
        [{"type": "video_url", "video_url": {"url": "https://example.test/a.mp4"}}],
        [{"type": "unknown_part", "value": 1}],
    ],
)
def test_non_text_content_is_rejected(content) -> None:
    request = make_request(
        messages=[{"role": "user", "content": content}]
    )
    with pytest.raises(UnsupportedContentError) as caught:
        validate_request(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "unsupported_content"


def test_text_and_input_text_content_are_preserved() -> None:
    request = make_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "input_text", "text": "two"},
                ],
            }
        ]
    )
    validate_request(request)
    assert render_prompt(request.messages) == "User: one\ntwo"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature", 0.2),
        ("top_p", 0.9),
    ],
)
def test_unsupported_sampling_and_limit_parameters_are_rejected(
    name: str, value
) -> None:
    request = make_request(**{name: value})
    with pytest.raises(RequestValidationError) as caught:
        validate_request(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "unsupported_parameter"
    assert name in caught.value.message


@pytest.mark.parametrize(
    "name",
    [
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
    ],
)
def test_explicit_null_unsupported_parameters_are_ignored(name: str) -> None:
    request = make_request(**{name: None})
    validate_request(request)


def test_unknown_model_is_rejected_without_fallback() -> None:
    request = make_request(model="not-in-catalog")
    with pytest.raises(UnknownModelError) as caught:
        validate_request(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "unknown_model"


def test_model_can_be_applied_after_parsing_without_model_field(tmp_path) -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hi"}],
        reasoning_effort="high",
    )
    validate_request(request)
    assert resolve_reasoning_effort(request) == 5
    payload = make_payload(request, tmp_path)
    assert payload["option"]["model_config"]["reasoning_effort"] == 5


def test_tool_history_is_not_lost_without_tools(tmp_path, monkeypatch) -> None:
    bridge = SimpleNamespace(
        validate_tools=lambda *args, **kwargs: None,
        render_tool_history=lambda message: (
            f"Tool history: {message.content}"
            if message.role == "tool"
            else "Assistant called lookup"
            if getattr(message, "tool_calls", None)
            else None
        ),
        render_tool_instructions=lambda *args, **kwargs: "",
    )
    monkeypatch.setitem(sys.modules, "doubao_work2api.tool_bridge", bridge)
    request = make_request(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "result",
            },
        ]
    )
    prompt = render_prompt(request.messages)
    assert '"type":"tool_calls"' in prompt
    assert '"id":"call-1"' in prompt
    assert '"type":"tool_result"' in prompt
    assert '"tool_call_id":"call-1"' in prompt
    payload = make_payload(request, tmp_path)
    assert '"type":"tool_calls"' in payload["messages"][0]["content_block"][0][
        "content"
    ]["text_block"]["text"]


def test_tool_result_content_is_not_duplicated_in_prompt() -> None:
    request = make_request(
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "lookup",
                "content": "sunny",
            }
        ]
    )
    prompt = render_prompt(request.messages)
    assert prompt.count("sunny") == 1
    assert '"type":"tool_result"' in prompt
    assert '"tool_call_id":"call-1"' in prompt


def test_assistant_text_and_tool_calls_are_both_kept_once() -> None:
    request = make_request(
        messages=[
            {
                "role": "assistant",
                "content": "Checking now.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ]
    )
    prompt = render_prompt(request.messages)
    assert prompt.count("Checking now.") == 1
    assert prompt.count('"type":"tool_calls"') == 1
    assert '"id":"call-1"' in prompt


def test_invalid_tool_history_is_rejected_by_validate_request() -> None:
    request = make_request(
        messages=[{"role": "tool", "content": "missing call id"}]
    )
    with pytest.raises(GatewayError) as caught:
        validate_request(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_tool_history"


@pytest.mark.asyncio
async def test_invalid_tool_history_is_rejected_before_account_acquire(
    tmp_path,
) -> None:
    class Transport:
        calls = 0

        async def stream(self, account, payload):
            self.calls += 1
            if False:
                yield {}

    pool = AccountPool(tmp_path / "accounts")
    pool._accounts = [
        Account(
            id="synthetic",
            cookies={"sessionid": "not-real"},
            device_id="device",
            web_id="web",
        )
    ]
    transport = Transport()
    service = ChatService(
        pool,
        transport,  # type: ignore[arg-type]
        model_key="unused",
        app_id="app",
        workspace_root=tmp_path / "work",
    )
    request = make_request(
        messages=[{"role": "tool", "content": "missing call id"}]
    )
    with pytest.raises(GatewayError) as caught:
        await service.complete(request)
    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_tool_history"
    assert pool.accounts[0].in_flight == 0
    assert transport.calls == 0


def test_tool_instructions_are_independent_system_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "doubao_work2api.tool_bridge", raising=False)
    request = make_request(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )
    payload = make_payload(request, tmp_path)
    prompt = payload["messages"][0]["content_block"][0]["content"]["text_block"][
        "text"
    ]
    assert prompt.startswith("System instructions:\nTOOL CALLING PROTOCOL\n")
    assert '"name":"lookup"' in prompt
    assert prompt.endswith("User: Hi")


def test_tools_do_not_get_rejected_as_extra_fields(monkeypatch) -> None:
    bridge = SimpleNamespace(
        validate_tools=lambda *args, **kwargs: None,
        render_tool_history=lambda message: None,
        render_tool_instructions=lambda *args, **kwargs: "",
    )
    monkeypatch.setitem(sys.modules, "doubao_work2api.tool_bridge", bridge)
    request = make_request(
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        parallel_tool_calls=False,
    )
    validate_request(request)
