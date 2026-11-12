from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import ChatCompletionRequest, ChatMessage


def text_content(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if not message.content:
        return ""
    parts: list[str] = []
    for item in message.content:
        if item.get("type") in {"text", "input_text"} and isinstance(
            item.get("text"), str
        ):
            parts.append(item["text"])
    return "\n".join(parts)


def render_prompt(messages: list[ChatMessage]) -> str:
    systems: list[str] = []
    dialogue: list[str] = []
    for message in messages:
        content = text_content(message)
        if not content:
            continue
        if message.role in {"system", "developer"}:
            systems.append(content)
            continue
        label = {
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }.get(message.role, message.role)
        dialogue.append(f"{label}: {content}")
    body = "\n\n".join(dialogue)
    if systems:
        return (
            "System instructions:\n"
            + "\n\n".join(systems)
            + "\n\nConversation:\n"
            + body
        )
    return body


def upstream_payload(
    request: ChatCompletionRequest,
    *,
    model_key: str,
    workspace: Path,
    device_name: str,
    app_id: str,
    device_id: str,
    reasoning_effort: int = 7,
) -> dict[str, Any]:
    prompt = render_prompt(request.messages)
    conversation_id = (request.conversation_id or "").strip()
    need_create = not conversation_id or conversation_id == "0"
    local_message_id = str(uuid.uuid4())
    workspace_text = str(workspace)
    task_param = {
        "runtime_type": 2,
        "sandbox_auth_type": 1,
        "device_name": device_name,
        "folder_name": "",
        "local_app_id": app_id,
        "local_device_id": device_id,
        "workspace": workspace_text,
    }
    general_task_param = {
        "action": 0,
        "thread_local_message_id": [local_message_id],
        "client_option": {
            "enable_sandbox": True,
            "os": "Linux",
            "shared_folder_path": [workspace_text],
            "workspace": workspace_text,
            "sandbox_auth_type": 1,
        },
        "runtime_type": 2,
        "agent_task_param": task_param,
        "server_do_compare": True,
    }
    task_json = json.dumps(
        task_param, ensure_ascii=False, separators=(",", ":")
    )
    general_json = json.dumps(
        general_task_param, ensure_ascii=False, separators=(",", ":")
    )
    return {
        "client_meta": {
            "local_conversation_id": (
                f"local_{uuid.uuid4().hex[:16]}" if need_create else ""
            ),
            "conversation_id": "" if need_create else conversation_id,
            "bot_id": "7338286299411103781",
            "last_section_id": "",
            "last_message_index": None,
        },
        "messages": [
            {
                "local_message_id": local_message_id,
                "content_block": [
                    {
                        "block_type": 10000,
                        "content": {
                            "text_block": {
                                "text": prompt,
                                "icon_url": "",
                                "icon_url_dark": "",
                                "summary": "",
                            },
                            "pc_event_block": "",
                        },
                        "block_id": str(uuid.uuid4()),
                        "parent_id": "",
                        "meta_info": [],
                        "extra_info": {},
                        "append_fields": [],
                    }
                ],
                "message_status": 0,
            }
        ],
        "option": {
            "send_message_scene": "",
            "create_time_ms": int(time.time() * 1000),
            "collect_id": "",
            "is_audio": False,
            "answer_with_suggest": False,
            "agent_mode": 1,
            "tts_switch": False,
            "need_deep_think": 10001,
            "click_clear_context": False,
            "from_suggest": False,
            "is_regen": False,
            "is_replace": False,
            "is_from_click_option": False,
            "is_from_click_softlink": False,
            "disable_sse_cache": False,
            "select_text_action": "",
            "is_select_text": False,
            "resend_for_regen": False,
            "scene_type": 0,
            "unique_key": str(uuid.uuid4()),
            "start_seq": 0,
            "need_create_conversation": need_create,
            "conversation_init_option": {
                "project_id": "",
                "need_ack_conversation": True,
            },
            "conversation_init_ext": {
                "agent_task_param": task_json,
                "model_item_key": model_key,
                "reasoning_effort": str(reasoning_effort),
                "mode_id": "3",
            },
            "regen_query_id": [],
            "edit_query_id": [],
            "regen_instruction": "",
            "no_replace_for_regen": False,
            "message_from": 0,
            "shared_app_name": "",
            "shared_app_id": "",
            "sse_recv_event_options": {"support_chunk_delta": True},
            "support_lazy_fetch_stream": True,
            "is_ai_playground": False,
            "is_old_user": True,
            "general_task_param": general_task_param,
            "recovery_option": {
                "is_recovery": False,
                "req_create_time_sec": int(time.time()),
                "append_sse_event_scene": 0,
            },
            "message_storage_type": 0,
            "related_deleted_message_ids": {},
            "connector_info_list": [],
            "model_config": {
                "model_item_key": model_key,
                "model_extra_params": {
                    "memory_profile": "300k_230",
                    "total_window_size": "300000",
                },
                "reasoning_effort": reasoning_effort,
            },
            "aggregate_params": {
                "mention_skill_list": "[]",
                "mention_plugin_list": "[]",
                "mention_ext": "[{}]",
                "conversation_mode": "",
                "mode_id": "3",
                "model_item_key": model_key,
                "agent_mode": "1",
                "reasoning_effort": str(reasoning_effort),
                "provider_id": "",
            },
        },
        "user_context": [],
        "ext": {
            "general_task_param": general_json,
            "use_deep_think": "10001",
            "agent_mode": "1",
            "conversation_init_option": json.dumps(
                {"project_id": "", "need_ack_conversation": True},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "sub_conv_firstmet_type": "1" if need_create else "0",
            "collection_id": "",
            "is_finish": "1",
            "commerce_credit_config_enable": "0",
        },
    }


def completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def chunk_payload(
    *,
    request_id: str,
    model: str,
    content: str = "",
    reasoning: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def completion_payload(
    *,
    request_id: str,
    model: str,
    content: str,
    reasoning: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_data(payload: dict[str, Any] | str) -> str:
    body = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return f"data: {body}\n\n"


def extract_text_fragments(payload: Any) -> Iterable[tuple[str, str]]:
    if not isinstance(payload, dict):
        return
    event_name = str(payload.get("_event") or "")
    if event_name == "CHUNK_DELTA" and isinstance(payload.get("text"), str):
        kind = "reasoning" if payload.get("_thinking") else "content"
        yield kind, payload["text"]
        return
    if event_name not in {"STREAM_MSG_NOTIFY", "STREAM_CHUNK", ""}:
        return
    for block in _iter_content_blocks(payload):
        if block.get("block_type") != 10000:
            continue
        content = block.get("content")
        if not isinstance(content, dict):
            continue
        text_block = content.get("text_block")
        if isinstance(text_block, dict) and isinstance(
            text_block.get("text"), str
        ):
            kind = "reasoning" if payload.get("_thinking") else "content"
            yield kind, text_block["text"]
    for patch in payload.get("patch_op", []):
        if not isinstance(patch, dict):
            continue
        value = patch.get("patch_value")
        if not isinstance(value, dict):
            continue
        raw = value.get("content")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                yield "content", parsed["text"]


def _iter_content_blocks(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for patch in payload.get("patch_op", []):
        if not isinstance(patch, dict):
            continue
        value = patch.get("patch_value")
        if not isinstance(value, dict):
            continue
        blocks = value.get("content_block")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    yield block
    content = payload.get("content")
    if isinstance(content, dict):
        blocks = content.get("content_block")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    yield block


def phase_markers(payload: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    containers = [payload]
    content = payload.get("content")
    if isinstance(content, dict):
        containers.append(content)
    for patch in payload.get("patch_op", []):
        if isinstance(patch, dict) and isinstance(
            patch.get("patch_value"), dict
        ):
            containers.append(patch["patch_value"])
    for container in containers:
        blocks = container.get("content_block")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict) and block.get("block_type") == 10040:
                markers.append("thinking")
    return markers
