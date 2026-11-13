from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .catalog import (
    REASONING_EFFORT_BY_LEVEL,
    REASONING_LEVELS,
    ModelSpec,
    find_model,
)
from .errors import GatewayError
from .models import ChatCompletionRequest, ChatMessage
from .image_inputs import validate_image_source
from .image_upload import UploadedImage
from .history import normalize_messages, render_reasoning_history
from .request_contract import validate_contract, render_contract_instructions


class RequestValidationError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "unsupported_parameter"


class UnsupportedContentError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "unsupported_content"


_UNSUPPORTED_PARAMETERS = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logprobs",
    "top_logprobs",
    "n",
)
_REASONING_INTEGER_RANGE = range(3, 8)
_MAX_PROMPT_CONTEXT_ITEMS = 100_000


def _request_field(request: ChatCompletionRequest, name: str) -> Any:
    extras = getattr(request, "__pydantic_extra__", None)
    if isinstance(extras, dict) and name in extras:
        return extras[name]
    return getattr(request, name, None)


def _is_explicit(request: ChatCompletionRequest, name: str) -> bool:
    if name in getattr(request, "model_fields_set", set()):
        return _request_field(request, name) is not None
    extras = getattr(request, "__pydantic_extra__", None)
    return (
        isinstance(extras, dict)
        and name in extras
        and extras[name] is not None
    )


def _request_tools(request: ChatCompletionRequest) -> Any:
    tools = _request_field(request, "tools")
    return tools if tools is not None else []


def _tool_bridge():
    try:
        from . import tool_bridge
    except ImportError as exc:
        raise RequestValidationError(
            "Tool support is not available in this gateway build.",
            code="unsupported_parameter",
        ) from exc
    return tool_bridge


def _validate_tools(request: ChatCompletionRequest) -> None:
    tools = _request_tools(request)
    tool_choice = _request_field(request, "tool_choice")
    parallel_tool_calls = _parallel_tool_calls(request)
    if tools or tool_choice is not None or _has_tool_history(request.messages):
        _tool_bridge().validate_tools(
            tools, tool_choice, parallel_tool_calls
        )


def _parallel_tool_calls(request: ChatCompletionRequest) -> bool:
    value = _request_field(request, "parallel_tool_calls")
    return True if value is None else value


def _has_tool_history(messages: list[ChatMessage]) -> bool:
    for message in messages:
        if message.role == "tool" or _request_field(message, "tool_calls"):
            return True
    return False


def _validate_content(request: ChatCompletionRequest) -> None:
    count = 0
    image_count = 0
    for message in request.messages:
        content = message.content
        if content is None:
            continue
        if isinstance(content, str):
            count += 1
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise UnsupportedContentError(
                        "Message content parts must be objects."
                    )
                if part.get("type") == "image_url":
                    if message.role not in {"user", "tool"}:
                        raise UnsupportedContentError("Images are supported only in user messages and tool results.")
                    image = part.get("image_url")
                    if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                        raise UnsupportedContentError("image_url must contain a string url.")
                    if image.get("detail", "auto") not in {"auto", "low", "high"}:
                        raise UnsupportedContentError("image_url.detail must be auto, low, or high.")
                    validate_image_source(image["url"])
                    image_count += 1
                    if image_count > 8:
                        raise UnsupportedContentError("At most 8 images may be included in one request.")
                    count += 1
                    continue
                if part.get("type") not in {"text", "input_text"}:
                    kind = part.get("type")
                    raise UnsupportedContentError(
                        f"Unsupported message content type: {kind!r}. "
                        "Only text, input_text, and user/tool image_url content are supported."
                    )
                if not isinstance(part.get("text"), str):
                    raise UnsupportedContentError(
                        "Text content parts must include a string 'text' value."
                    )
                count += 1
        else:
            raise UnsupportedContentError(
                "Message content must be a string or a list of content parts."
            )
        if count > _MAX_PROMPT_CONTEXT_ITEMS:
            raise RequestValidationError(
                "Request contains too many message content parts."
            )


def _resolve_model(request: ChatCompletionRequest) -> ModelSpec | None:
    return find_model(request.normalized_model)


def resolve_reasoning_effort(request: ChatCompletionRequest) -> int:
    spec = _resolve_model(request)
    if spec is None:
        if _is_explicit(request, "reasoning_effort"):
            value = _request_field(request, "reasoning_effort")
            if isinstance(value, int) and not isinstance(value, bool):
                if value in _REASONING_INTEGER_RANGE:
                    return value
            raise RequestValidationError(
                "reasoning_effort requires a recognized model.",
                code="invalid_reasoning_effort",
            )
        return 7

    explicit = _is_explicit(request, "reasoning_effort")
    value = _request_field(request, "reasoning_effort")
    if not explicit:
        if spec.default_reasoning_effort:
            return REASONING_EFFORT_BY_LEVEL[spec.default_reasoning_effort]
        return 7

    if isinstance(value, str):
        if value in spec.reasoning_levels:
            return REASONING_EFFORT_BY_LEVEL[value]
    elif isinstance(value, int) and not isinstance(value, bool):
        if value in _REASONING_INTEGER_RANGE:
            level = REASONING_LEVELS[value - 3]
            if level in spec.reasoning_levels:
                return value

    if not spec.reasoning_levels:
        raise RequestValidationError(
            f"Model {spec.id!r} does not support reasoning_effort.",
            code="unsupported_parameter",
        )
    raise RequestValidationError(
        f"Unsupported reasoning_effort for model {spec.id!r}. "
        f"Supported levels are {', '.join(spec.reasoning_levels)} "
        "or their corresponding upstream integers.",
        code="invalid_reasoning_effort",
    )


def validate_request(request: ChatCompletionRequest) -> ChatCompletionRequest:
    legacy_functions = getattr(request, "functions", None)
    if legacy_functions is not None:
        if request.tools or not isinstance(legacy_functions, list):
            raise RequestValidationError("Use either functions or tools with a list of function definitions.", code="invalid_tools")
        choice = getattr(request, "function_call", None)
        if isinstance(choice, dict):
            choice = {"type": "function", "function": {"name": choice.get("name")}}
        request = request.model_copy(update={"tools": [{"type": "function", "function": definition} for definition in legacy_functions], "tool_choice": choice})
        # Mark the legacy fields consumed so repeated validation is idempotent.
        request.__pydantic_extra__.pop("functions", None)
        request.__pydantic_extra__.pop("function_call", None)
    elif getattr(request, "function_call", None) is not None:
        raise RequestValidationError("Legacy function_call requires functions; use tool_choice with modern tools.", code="invalid_tools")
    request = request.model_copy(update={"messages": normalize_messages(request.messages)})
    model = request.model
    if not isinstance(model, str) or not model.strip():
        raise RequestValidationError(
            "model must be a non-empty string.",
            code="invalid_model",
        )
    spec = find_model(model)
    if spec is None:
        from .errors import UnknownModelError

        raise UnknownModelError(model)

    for name in _UNSUPPORTED_PARAMETERS:
        if _is_explicit(request, name):
            raise RequestValidationError(
                f"{name} is not supported by the Doubao Work upstream.",
                code="unsupported_parameter",
            )

    _validate_content(request)
    validate_contract(request)
    _validate_tools(request)
    for message in request.messages:
        render_tool_history(message)
    resolve_reasoning_effort(request)
    if request.stream_options is not None and (
        set(request.stream_options) - {"include_usage"}
        or ("include_usage" in request.stream_options and type(request.stream_options["include_usage"]) is not bool)
    ):
        raise RequestValidationError("stream_options supports only a boolean include_usage.")
    from .context import apply_context_budget
    return apply_context_budget(request)


def render_tool_history(message: ChatMessage) -> str | None:
    if message.role != "tool" and not _request_field(message, "tool_calls"):
        return None
    bridge = _tool_bridge()
    render = getattr(bridge, "render_tool_history", None)
    if not callable(render):
        raise RequestValidationError(
            "Tool history support is not available in this gateway build.",
            code="unsupported_parameter",
        )
    return render(message)


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
    image_index = 0
    for message in messages:
        rendered_history = render_tool_history(message)
        historical_reasoning = render_reasoning_history(message)
        content = text_content(message)
        if isinstance(message.content, list) and any(part.get("type") == "image_url" for part in message.content):
            parts = []
            for part in message.content:
                if part.get("type") == "image_url":
                    image_index += 1
                    parts.append(f"[Attached image {image_index}]")
                elif part.get("type") in {"text", "input_text"}:
                    parts.append(part["text"])
            content = "\n".join(parts)
            if message.role == "tool":
                rendered_history = render_tool_history(message.model_copy(update={"content": [{"type": "text", "text": content}]}))
        if message.role == "tool":
            content = rendered_history or ""
        elif rendered_history is not None:
            content = "\n\n".join(item for item in (content, rendered_history) if item)
        if historical_reasoning is not None:
            content = "\n\n".join(item for item in (historical_reasoning, content) if item)
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
    reasoning_effort: int | None = None,
    images: list[UploadedImage] | None = None,
) -> dict[str, Any]:
    request = validate_request(request)
    spec = find_model(request.normalized_model)
    if spec is None:
        raise RequestValidationError(
            "model must be resolved before building an upstream payload.",
            code="invalid_model",
        )
    prompt = render_prompt(request.messages)
    tools = _request_tools(request)
    if tools:
        instructions = _tool_bridge().render_tool_instructions(
            tools,
            _request_field(request, "tool_choice"),
            _parallel_tool_calls(request),
        )
        if instructions:
            prompt = f"System instructions:\n{instructions}\n\n{prompt}"
    contract_instructions = render_contract_instructions(request)
    if contract_instructions:
        prompt = f"System instructions:\n{contract_instructions}\n\n{prompt}"
    resolved_effort = (
        resolve_reasoning_effort(request)
        if reasoning_effort is None
        else reasoning_effort
    )
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
    payload = {
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
                "reasoning_effort": str(resolved_effort),
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
                "reasoning_effort": resolved_effort,
            },
            "aggregate_params": {
                "mention_skill_list": "[]",
                "mention_plugin_list": "[]",
                "mention_ext": "[{}]",
                "conversation_mode": "",
                "mode_id": "3",
                "model_item_key": model_key,
                "agent_mode": "1",
                "reasoning_effort": str(resolved_effort),
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
    if spec.has_verified_memory_profile:
        payload["option"]["model_config"]["model_extra_params"] = {
            "memory_profile": spec.memory_profile,
            "total_window_size": str(spec.context_window),
        }
    image_count = sum(1 for message in request.messages if isinstance(message.content, list) for part in message.content if part.get("type") == "image_url")
    if image_count != len(images or []):
        raise RequestValidationError("Image inputs must be uploaded before building the upstream request.", code="images_not_prepared")
    if images:
        payload["messages"][0]["content_block"].insert(0, {
            "block_type": 10052, "block_id": str(uuid.uuid4()), "parent_id": "",
            "meta_info": [], "extra_info": {}, "append_fields": [],
            "content": {
                "attachment_block": {"attachments": [image.attachment() for image in images]},
                "pc_event_block": "",
            },
        })
    return payload


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
        delta["reasoning"] = reasoning
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
        message["reasoning"] = reasoning
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


def _reasoning_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_reasoning_text(part) for part in value)
    if isinstance(value, dict):
        for key in ("text", "reasoning", "content"):
            text = _reasoning_text(value.get(key))
            if text:
                return text
    return ""


def _reasoning_field(payload: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking", "reasoning_details"):
        text = _reasoning_text(payload.get(key))
        if text:
            return text
    return ""


def _text_block_kind(block: dict[str, Any], reasoning_ids: set[str]) -> str:
    # collapse_block_id is a visual grouping (often the elapsed-time block),
    # not proof that text belongs to the thinking container.
    control = block.get("control_info") or {}
    parent = block.get("parent_id") or block.get("parent_block_id") or (control.get("parent_block_id") if isinstance(control, dict) else None)
    return "reasoning" if block.get("block_id") in reasoning_ids or parent in reasoning_ids else "content"


class UpstreamTextParser:
    """Track explicit block ancestry, never toggle on a container update."""

    def __init__(self) -> None:
        self.reasoning_ids: set[str] = set()
        self.text_kinds: dict[str, str] = {}
        self.last_text_kind = "content"

    def feed(self, payload: dict[str, Any]) -> Iterable[tuple[str, str]]:
        blocks = list(_iter_content_blocks(payload))
        for block in blocks:
            identity = block.get("block_id")
            if block.get("block_type") == 10040 and isinstance(identity, str):
                self.reasoning_ids.add(identity)
        for block in blocks:
            if block.get("block_type") != 10000:
                continue
            kind = _text_block_kind(block, self.reasoning_ids)
            identity = block.get("block_id")
            if isinstance(identity, str):
                self.text_kinds[identity] = kind
            self.last_text_kind = kind
        if payload.get("_event") == "CHUNK_DELTA" and "_thinking" not in payload:
            kind = self.text_kinds.get(payload.get("block_id"), self.last_text_kind)
            payload = {**payload, "_thinking": kind == "reasoning"}
        yield from extract_text_fragments(payload, reasoning_ids=self.reasoning_ids)


def extract_text_fragments(payload: Any, *, reasoning_ids: set[str] | None = None) -> Iterable[tuple[str, str]]:
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("choices"), list):
        for choice in payload["choices"]:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            reason = _reasoning_field(delta)
            if isinstance(reason, str) and reason:
                yield "reasoning", reason
            if isinstance(delta.get("content"), str) and delta["content"]:
                yield "content", delta["content"]
        return
    direct_reasoning = _reasoning_field(payload)
    if isinstance(direct_reasoning, str) and direct_reasoning:
        yield "reasoning", direct_reasoning
    event_name = str(payload.get("_event") or "")
    if event_name == "CHUNK_DELTA" and isinstance(payload.get("text"), str):
        kind = "reasoning" if payload.get("_thinking") else "content"
        if kind != "reasoning" or payload["text"] != direct_reasoning:
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
            kind = _text_block_kind(block, reasoning_ids or set())
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
                kind = "reasoning" if parsed.get("type") in {"reasoning", "thinking"} else "content"
                yield kind, parsed["text"]


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
