from __future__ import annotations

from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.protocol import (
    extract_text_fragments,
    phase_markers,
    render_prompt,
    upstream_payload,
)


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
