from __future__ import annotations

import copy
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from .errors import GatewayError
from .models import ChatCompletionRequest, ChatMessage


class ResponsesConversionError(GatewayError):
    """A request or response shape error in the Responses conversion layer."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_responses_request"


_TERMINAL_RESPONSE_EVENTS = {
    "response.completed",
    "response.incomplete",
    "response.failed",
}
_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}
_IMAGE_DETAILS = {"auto", "low", "high", "original"}
_ROLES = {"system", "developer", "user", "assistant"}
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ALLOWED_TOP_LEVEL = {
    "model",
    "input",
    "instructions",
    "previous_response_id",
    "reasoning",
    "text",
    "max_output_tokens",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "store",
    "metadata",
    "stream",
    "background",
    "user",
    "include",
    "truncation",
}
_KNOWN_UNSUPPORTED_TOP_LEVEL = {
    "access_programs",
    "conversation",
    "context_management",
    "image",
    "max_tool_calls",
    "moderation",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "stream_options",
    "temperature",
    "top_logprobs",
    "top_p",
}


def _fail(message: str, code: str = "invalid_responses_request") -> None:
    raise ResponsesConversionError(message, code=code)


def _as_dict(value: Any, label: str, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object.", code)
    return value


def _as_list(value: Any, label: str, code: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array.", code)
    return value


def _string(value: Any, label: str, code: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _fail(f"{label} must be a non-empty string.", code)
    return value


def _bool(value: Any, label: str, code: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be a boolean.", code)
    return value


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _response_id(value: str | None) -> str:
    if value is None:
        return _new_id("resp")
    if not isinstance(value, str) or not value.strip():
        _fail("response_id must be a non-empty string.", "invalid_response_id")
    return value if value.startswith("resp_") else f"resp_{value}"


def _validate_payload_shape(payload: dict[str, Any]) -> None:
    unknown = set(payload) - _ALLOWED_TOP_LEVEL - _KNOWN_UNSUPPORTED_TOP_LEVEL
    if unknown:
        _fail(
            "Unsupported Responses parameter(s): "
            + ", ".join(sorted(unknown))
            + ".",
            "unsupported_parameter",
        )
    for name in _KNOWN_UNSUPPORTED_TOP_LEVEL:
        if payload.get(name) is not None:
            _fail(
                f"{name} is not implemented by this Responses adapter.",
                "unsupported_parameter",
            )
    if payload.get("background") is not None:
        if not isinstance(payload["background"], bool):
            _fail("background must be a boolean.", "invalid_background")
        if payload["background"]:
            _fail(
                "background responses are not implemented.",
                "unsupported_parameter",
            )


def _validate_store(value: Any) -> bool:
    if value is None:
        return True
    return _bool(value, "store", "invalid_store")


def _validate_include(value: Any) -> list[str]:
    if value is None:
        return []
    includes = _as_list(value, "include", "invalid_include")
    result: list[str] = []
    for index, item in enumerate(includes):
        if not isinstance(item, str):
            _fail(f"include[{index}] must be a string.", "invalid_include")
        if item != "reasoning.encrypted_content":
            _fail(
                f"include value {item!r} is not implemented.",
                "unsupported_include",
            )
        result.append(item)
    return result


def _validate_truncation(value: Any) -> str:
    if value is None:
        return "disabled"
    if not isinstance(value, str) or value not in {"disabled", "auto"}:
        _fail(
            "truncation must be disabled or auto.",
            "invalid_truncation",
        )
    return value


def _validate_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    metadata = _as_dict(value, "metadata", "invalid_metadata")
    if len(metadata) > 16:
        _fail("metadata cannot contain more than 16 entries.", "invalid_metadata")
    result: dict[str, Any] = {}
    for key, item in metadata.items():
        if not isinstance(key, str) or len(key) > 64:
            _fail(
                "metadata keys must be strings no longer than 64 characters.",
                "invalid_metadata",
            )
        if isinstance(item, str):
            if len(item) > 512:
                _fail(
                    "metadata string values cannot exceed 512 characters.",
                    "invalid_metadata",
                )
        elif not isinstance(item, (bool, int, float)):
            _fail(
                "metadata values must be strings, booleans, or numbers.",
                "invalid_metadata",
            )
        result[key] = item
    try:
        _json_text(result)
    except (TypeError, ValueError, RecursionError):
        _fail("metadata must be JSON-safe.", "invalid_metadata")
    return result


def _coerce_previous_messages(
    previous_messages: list[ChatMessage] | None,
    previous_response_id: Any,
) -> list[ChatMessage]:
    if previous_response_id is not None:
        _string(
            previous_response_id,
            "previous_response_id",
            "invalid_previous_response_id",
        )
        if previous_messages is None:
            _fail(
                "previous_response_id must be resolved by the parent before "
                "calling responses_to_chat.",
                "previous_response_not_resolved",
            )
    if previous_messages is None:
        return []
    if not isinstance(previous_messages, (list, tuple)):
        _fail(
            "previous_messages must be an array of ChatMessage objects.",
            "invalid_previous_messages",
        )
    result: list[ChatMessage] = []
    for index, message in enumerate(previous_messages):
        if isinstance(message, ChatMessage):
            result.append(message.model_copy(deep=True))
            continue
        if not isinstance(message, Mapping):
            _fail(
                f"previous_messages[{index}] must be a ChatMessage or object.",
                "invalid_previous_messages",
            )
        try:
            result.append(ChatMessage.model_validate(dict(message)))
        except ValidationError as exc:
            raise ResponsesConversionError(
                f"previous_messages[{index}] is not a valid ChatMessage.",
                code="invalid_previous_messages",
            ) from exc
    return result


def _normalize_reasoning_request(value: Any) -> str | None:
    if value is None:
        return None
    reasoning = _as_dict(value, "reasoning", "invalid_reasoning")
    allowed = {"effort", "summary", "generate_summary"}
    unknown = set(reasoning) - allowed
    if unknown:
        _fail(
            "Unsupported reasoning parameter(s): "
            + ", ".join(sorted(unknown))
            + ".",
            "unsupported_parameter",
        )
    for name in ("summary", "generate_summary"):
        if reasoning.get(name) is not None:
            _fail(
                "reasoning summary generation is not supported by the "
                "Chat conversion.",
                "unsupported_parameter",
            )
    effort = reasoning.get("effort")
    if effort is None:
        return None
    if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
        _fail(
            "reasoning.effort must be none, minimal, low, medium, high, "
            "xhigh, or max.",
            "invalid_reasoning_effort",
        )
    return effort


def _normalize_text_format(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    text = _as_dict(value, "text", "invalid_text")
    unknown = set(text) - {"format", "verbosity"}
    if unknown:
        _fail(
            "Unsupported text parameter(s): " + ", ".join(sorted(unknown)) + ".",
            "unsupported_parameter",
        )
    verbosity = text.get("verbosity")
    if verbosity is not None and (
        not isinstance(verbosity, str)
        or verbosity not in {"low", "medium", "high"}
    ):
        _fail(
            "text.verbosity must be low, medium, or high.",
            "invalid_text_verbosity",
        )
    format_value = text.get("format")
    if format_value is None:
        return None
    format_object = _as_dict(format_value, "text.format", "invalid_text_format")
    kind = format_object.get("type")
    if kind == "text":
        if set(format_object) != {"type"}:
            _fail(
                "A text response format may only contain type.",
                "invalid_text_format",
            )
        return None
    if kind == "json_object":
        if set(format_object) != {"type"}:
            _fail(
                "A json_object response format may only contain type.",
                "invalid_text_format",
            )
        return {"type": "json_object"}
    if kind == "json_schema":
        allowed = {"type", "name", "schema", "description", "strict"}
        if set(format_object) - allowed:
            _fail(
                "text.format contains unsupported json_schema fields.",
                "invalid_text_format",
            )
        name = _string(
            format_object.get("name"),
            "text.format.name",
            "invalid_text_format",
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            _fail(
                "text.format.name must contain only letters, numbers, "
                "underscores, or dashes and be at most 64 characters.",
                "invalid_text_format",
            )
        schema = _as_dict(
            format_object.get("schema"),
            "text.format.schema",
            "invalid_text_format",
        )
        strict = format_object.get("strict")
        if strict is not None:
            strict = _bool(strict, "text.format.strict", "invalid_text_format")
        description = format_object.get("description")
        if description is not None and not isinstance(description, str):
            _fail(
                "text.format.description must be a string.",
                "invalid_text_format",
            )
        nested: dict[str, Any] = {
            "name": name,
            "schema": copy.deepcopy(schema),
        }
        if description is not None:
            nested["description"] = description
        if strict is not None:
            nested["strict"] = strict
        return {"type": "json_schema", "json_schema": nested}
    _fail(
        f"Unsupported text.format type: {kind!r}.",
        "unsupported_parameter",
    )


def _text_verbosity(value: Any) -> str | None:
    if value is None:
        return None
    text = _as_dict(value, "text", "invalid_text")
    verbosity = text.get("verbosity")
    if verbosity is None:
        return None
    return verbosity


def _convert_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    tools = _as_list(value, "tools", "invalid_tools")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(tools):
        tool = _as_dict(raw, f"tools[{index}]", "invalid_tools")
        if tool.get("type") != "function":
            kind = tool.get("type")
            _fail(
                f"tools[{index}] uses unsupported tool type {kind!r}; only "
                "flat function tools are implemented.",
                "unsupported_tool",
            )
        allowed = {"type", "name", "description", "parameters", "strict"}
        if set(tool) - allowed:
            _fail(
                f"tools[{index}] contains unsupported fields.",
                "invalid_tools",
            )
        name = _string(tool.get("name"), f"tools[{index}].name", "invalid_tools")
        if not _TOOL_NAME.fullmatch(name):
            _fail(
                f"tools[{index}].name is invalid.",
                "invalid_tool_name",
            )
        if name in names:
            _fail(f"Duplicate tool name: {name}.", "duplicate_tool")
        names.add(name)
        nested: dict[str, Any] = {"name": name}
        description = tool.get("description")
        if description is not None:
            if not isinstance(description, str):
                _fail(
                    f"tools[{index}].description must be a string.",
                    "invalid_tools",
                )
            nested["description"] = description
        parameters = tool.get("parameters")
        if parameters is not None:
            nested["parameters"] = copy.deepcopy(
                _as_dict(
                    parameters,
                    f"tools[{index}].parameters",
                    "invalid_tools",
                )
            )
        strict = tool.get("strict")
        if strict is not None:
            nested["strict"] = _bool(
                strict,
                f"tools[{index}].strict",
                "invalid_tools",
            )
        result.append({"type": "function", "function": nested})
    return result


def _convert_tool_choice(value: Any) -> str | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            _fail(
                "tool_choice must be auto, none, required, or a function "
                "object.",
                "invalid_tool_choice",
            )
        return value
    choice = _as_dict(value, "tool_choice", "invalid_tool_choice")
    if choice.get("type") != "function":
        _fail(
            "Only function tool_choice objects are implemented.",
            "unsupported_tool_choice",
        )
    if set(choice) != {"type", "name"}:
        _fail(
            "A Responses function tool_choice must contain only type and name.",
            "invalid_tool_choice",
        )
    name = _string(
        choice.get("name"),
        "tool_choice.name",
        "invalid_tool_choice",
    )
    if not _TOOL_NAME.fullmatch(name):
        _fail("tool_choice.name is invalid.", "invalid_tool_choice")
    return {"type": "function", "function": {"name": name}}


def _image_part(part: dict[str, Any], index: int, *, allow_file: bool) -> dict[str, Any]:
    image_url = part.get("image_url")
    file_id = part.get("file_id")
    if not isinstance(image_url, str) or not image_url:
        if file_id is not None:
            _fail(
                f"content[{index}].file_id requires parent file resolution; "
                "only image_url is available in this adapter.",
                "unsupported_image_reference",
            )
        _fail(
            f"content[{index}].image_url must be a non-empty string.",
            "invalid_image_input",
        )
    detail = part.get("detail", "auto")
    if detail is None:
        detail = "auto"
    if not isinstance(detail, str) or detail not in _IMAGE_DETAILS:
        _fail(
            f"content[{index}].detail must be auto, low, high, or original.",
            "invalid_image_input",
        )
    image: dict[str, Any] = {"url": image_url}
    if detail == "original":
        image["detail"] = "high"
        image["detail_original"] = True
    else:
        image["detail"] = detail
    if file_id is not None and not isinstance(file_id, str):
        _fail(
            f"content[{index}].file_id must be a string or null.",
            "invalid_image_input",
        )
    return {"type": "image_url", "image_url": image}


def _convert_content_parts(
    parts: list[Any],
    *,
    role: str,
    label: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(parts):
        part = _as_dict(raw, f"{label}[{index}]", "invalid_content")
        kind = part.get("type")
        if kind in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                _fail(
                    f"{label}[{index}].text must be a string.",
                    "invalid_content",
                )
            result.append({"type": "text", "text": text})
            continue
        if kind == "input_image":
            if role != "user":
                _fail(
                    "input_image is supported only in user messages.",
                    "unsupported_content",
                )
            result.append(_image_part(part, index, allow_file=False))
            continue
        _fail(
            f"Unsupported content type {kind!r} in {label}.",
            "unsupported_content",
        )
    return result


def _convert_message_item(item: dict[str, Any]) -> ChatMessage:
    role = item.get("role")
    if role not in _ROLES:
        _fail(
            "message.role must be user, assistant, system, or developer.",
            "invalid_message",
        )
    content = item.get("content")
    if isinstance(content, str):
        converted: str | list[dict[str, Any]] = content
    elif isinstance(content, list):
        converted = _convert_content_parts(
            content,
            role=role,
            label="message.content",
        )
    else:
        _fail(
            "message.content must be a string or an array of content parts.",
            "invalid_message",
        )
    extras: dict[str, Any] = {}
    if item.get("phase") is not None:
        if not isinstance(item["phase"], str):
            _fail("message.phase must be a string.", "invalid_message")
        extras["phase"] = item["phase"]
    if item.get("id") is not None:
        extras["response_item_id"] = item["id"]
    return ChatMessage(role=role, content=converted, **extras)


def _replay_output_message_item(item: dict[str, Any]) -> ChatMessage:
    role = item.get("role")
    if role not in _ROLES:
        _fail(
            "message.role must be user, assistant, system, or developer.",
            "invalid_message",
        )
    content = item.get("content")
    if not isinstance(content, str):
        _fail("message.content must be a string.", "invalid_message")
    extras: dict[str, Any] = {}
    if item.get("id") is not None:
        extras["response_item_id"] = item["id"]
    return ChatMessage(
        role=role,
        content=[{"type": "text", "text": content}],
        **extras,
    )


def _chat_tool_call(item: dict[str, Any], index: int) -> dict[str, Any]:
    call_id = _string(
        item.get("call_id"),
        f"function_call[{index}].call_id",
        "invalid_function_call",
    )
    name = _string(
        item.get("name"),
        f"function_call[{index}].name",
        "invalid_function_call",
    )
    if not _TOOL_NAME.fullmatch(name):
        _fail(
            f"function_call[{index}].name is invalid.",
            "invalid_function_call",
        )
    arguments = item.get("arguments")
    if not isinstance(arguments, str):
        _fail(
            f"function_call[{index}].arguments must be a JSON string.",
            "invalid_function_call",
        )
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _append_tool_call(
    messages: list[ChatMessage],
    call: dict[str, Any],
    item: dict[str, Any],
) -> None:
    if (
        messages
        and messages[-1].role == "assistant"
        and messages[-1].content is None
        and isinstance(getattr(messages[-1], "tool_calls", None), list)
    ):
        messages[-1].tool_calls.append(call)
        ids = getattr(messages[-1], "response_item_ids", None)
        if isinstance(ids, list) and item.get("id") is not None:
            ids.append(item["id"])
        return
    extras: dict[str, Any] = {}
    if item.get("id") is not None:
        extras["response_item_ids"] = [item["id"]]
    messages.append(
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[call],
            **extras,
        )
    )


def _function_output_content(value: Any, index: int) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        _fail(
            f"function_call_output[{index}].output must be a string or array.",
            "invalid_function_output",
        )
    result: list[dict[str, Any]] = []
    for part_index, raw in enumerate(value):
        part = _as_dict(
            raw,
            f"function_call_output[{index}].output[{part_index}]",
            "invalid_function_output",
        )
        kind = part.get("type")
        if kind in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                _fail(
                    "Function output text parts require a string text value.",
                    "invalid_function_output",
                )
            result.append({"type": "text", "text": text})
        elif kind == "input_image":
            result.append(_image_part(part, part_index, allow_file=False))
        else:
            _fail(
                f"Unsupported function output content type {kind!r}.",
                "unsupported_content",
            )
    return result


def _extract_reasoning_parts(
    value: Any,
    *,
    label: str,
    expected_type: str,
) -> list[str]:
    if value is None:
        return []
    parts = _as_list(value, label, "invalid_reasoning_item")
    result: list[str] = []
    for index, raw in enumerate(parts):
        part = _as_dict(raw, f"{label}[{index}]", "invalid_reasoning_item")
        if part.get("type") != expected_type:
            _fail(
                f"{label}[{index}].type must be {expected_type}.",
                "invalid_reasoning_item",
            )
        text = part.get("text")
        if not isinstance(text, str):
            _fail(
                f"{label}[{index}].text must be a string.",
                "invalid_reasoning_item",
            )
        result.append(text)
    return result


def _decoded_messages(
    result: Any,
    *,
    label: str,
    code: str,
) -> list[ChatMessage]:
    if not isinstance(result, (list, tuple)):
        _fail(f"{label} must return a list of ChatMessage objects.", code)
    messages: list[ChatMessage] = []
    for index, message in enumerate(result):
        if isinstance(message, ChatMessage):
            messages.append(message.model_copy(deep=True))
        elif isinstance(message, Mapping):
            try:
                messages.append(ChatMessage.model_validate(dict(message)))
            except ValidationError as exc:
                raise ResponsesConversionError(
                    f"{label} returned an invalid ChatMessage at index {index}.",
                    code=code,
                ) from exc
        else:
            _fail(
                f"{label} returned a non-ChatMessage at index {index}.",
                code,
            )
    return messages


def _convert_reasoning_item(
    item: dict[str, Any],
    decode_reasoning: Callable[[dict[str, Any]], list[ChatMessage]] | None,
) -> list[ChatMessage]:
    summary = _extract_reasoning_parts(
        item.get("summary"),
        label="reasoning.summary",
        expected_type="summary_text",
    )
    content = _extract_reasoning_parts(
        item.get("content"),
        label="reasoning.content",
        expected_type="reasoning_text",
    )
    encrypted = item.get("encrypted_content")
    if encrypted is not None:
        encrypted = _string(
            encrypted,
            "reasoning.encrypted_content",
            "invalid_reasoning_item",
        )
    if encrypted is not None and decode_reasoning is not None:
        try:
            decoded = decode_reasoning(copy.deepcopy(item))
        except Exception as exc:
            if isinstance(exc, GatewayError) and exc.status_code == 400:
                raise
            raise ResponsesConversionError(
                "reasoning.encrypted_content could not be decoded.",
                code="encrypted_reasoning_unresolved",
            ) from exc
        messages = _decoded_messages(
            decoded,
            label="decode_reasoning",
            code="invalid_reasoning_decoding",
        )
        return messages
    if encrypted is not None and not summary and not content:
        _fail(
            "reasoning.encrypted_content is opaque to this adapter. "
            "Provide a decode_reasoning hook or omit the unknown token.",
            "encrypted_reasoning_unresolved",
        )
    if not summary and not content:
        _fail(
            "reasoning must contain summary, content, or encrypted_content.",
            "invalid_reasoning_item",
        )
    extras: dict[str, Any] = {}
    if item.get("id") is not None:
        extras["response_item_id"] = item["id"]
    if content:
        extras["reasoning_content"] = "\n".join(content)
    elif summary:
        extras["reasoning_content"] = "\n".join(summary)
    if summary:
        extras["reasoning_summary"] = "\n".join(summary)
    if encrypted is not None:
        extras["encrypted_content"] = encrypted
        extras["opaque_reasoning"] = {
            "id": item.get("id"),
            "type": "reasoning",
            "encrypted_content": encrypted,
        }
    return [ChatMessage(role="assistant", content=None, **extras)]


def _convert_compaction_item(
    item: dict[str, Any],
    decode_compaction: Callable[[dict[str, Any]], list[ChatMessage]] | None,
) -> list[ChatMessage]:
    encrypted = item.get("encrypted_content")
    _string(
        encrypted,
        "compaction.encrypted_content",
        "invalid_compaction",
    )
    if decode_compaction is None:
        _fail(
            "compaction items require decode_compaction supplied by the parent.",
            "compaction_decoder_required",
        )
    try:
        decoded = decode_compaction(copy.deepcopy(item))
    except Exception as exc:
        if isinstance(exc, GatewayError) and exc.status_code == 400:
            raise
        raise ResponsesConversionError(
            "compaction.encrypted_content could not be decoded.",
            code="invalid_compaction",
        ) from exc
    return _decoded_messages(
        decoded,
        label="decode_compaction",
        code="invalid_compaction",
    )


def _convert_input(
    value: Any,
    *,
    decode_compaction: Callable[[dict[str, Any]], list[ChatMessage]] | None,
    decode_reasoning: Callable[[dict[str, Any]], list[ChatMessage]] | None,
) -> list[ChatMessage]:
    if value is None:
        return []
    if isinstance(value, str):
        return [ChatMessage(role="user", content=value)]
    if not isinstance(value, list):
        _fail(
            "input must be a string or an array of input items.",
            "invalid_input",
        )
    messages: list[ChatMessage] = []
    for index, raw in enumerate(value):
        item = _as_dict(raw, f"input[{index}]", "invalid_input")
        kind = item.get("type")
        if kind == "item_reference":
            _fail(
                f"input[{index}] is an item_reference. The parent must expand "
                "it before calling responses_to_chat.",
                "item_reference_unresolved",
            )
        if kind == "compaction":
            messages.extend(
                _convert_compaction_item(item, decode_compaction)
            )
            continue
        if kind == "reasoning":
            messages.extend(
                _convert_reasoning_item(item, decode_reasoning)
            )
            continue
        if kind == "function_call":
            call = _chat_tool_call(item, index)
            _append_tool_call(messages, call, item)
            continue
        if kind == "function_call_output":
            call_id = _string(
                item.get("call_id"),
                f"input[{index}].call_id",
                "invalid_function_output",
            )
            output = _function_output_content(item.get("output"), index)
            name = item.get("name")
            if name is not None:
                name = _string(
                    name,
                    f"input[{index}].name",
                    "invalid_function_output",
                )
            extras: dict[str, Any] = {"tool_call_id": call_id}
            if name is not None:
                extras["name"] = name
            if item.get("id") is not None:
                extras["response_item_id"] = item["id"]
            messages.append(
                ChatMessage(role="tool", content=output, **extras)
            )
            continue
        if kind in {None, "message"} or "role" in item:
            if item.get("object") == "response.output_item" and isinstance(
                item.get("content"), str
            ):
                messages.append(_replay_output_message_item(item))
            else:
                messages.append(_convert_message_item(item))
            continue
        _fail(
            f"Unsupported input item type: {kind!r}.",
            "unsupported_input_item",
        )
    return messages


def responses_to_chat(
    payload: dict[str, Any],
    default_model: str,
    previous_messages: list[ChatMessage] | None = None,
    decode_compaction: Callable[[dict[str, Any]], list[ChatMessage]] | None = None,
    *,
    decode_reasoning: Callable[[dict[str, Any]], list[ChatMessage]] | None = None,
) -> ChatCompletionRequest:
    """Convert a Responses request into the gateway's Chat request model."""

    if not isinstance(payload, dict):
        _fail("payload must be an object.", "invalid_payload")
    _validate_payload_shape(payload)

    model_value = payload.get("model")
    model = default_model if model_value is None else model_value
    model = _string(model, "model", "invalid_model")

    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            _fail("instructions must be a string or null.", "invalid_instructions")
    previous_id = payload.get("previous_response_id")
    previous = _coerce_previous_messages(previous_messages, previous_id)

    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))
    messages.extend(previous)
    messages.extend(
        _convert_input(
            payload.get("input"),
            decode_compaction=decode_compaction,
            decode_reasoning=decode_reasoning,
        )
    )
    if payload.get("input") is None and not previous:
        _fail(
            "input is required when previous_response_id and "
            "previous_messages are absent.",
            "empty_input",
        )
    if not messages:
        _fail(
            "input or previous_messages must contain at least one message.",
            "empty_input",
        )

    stream_value = payload.get("stream")
    stream = False if stream_value is None else _bool(
        stream_value, "stream", "invalid_stream"
    )
    parallel_value = payload.get("parallel_tool_calls")
    parallel_tool_calls = (
        True
        if parallel_value is None
        else _bool(
            parallel_value,
            "parallel_tool_calls",
            "invalid_parallel_tool_calls",
        )
    )
    store = _validate_store(payload.get("store"))
    metadata = _validate_metadata(payload.get("metadata"))
    includes = _validate_include(payload.get("include"))
    truncation = _validate_truncation(payload.get("truncation"))
    reasoning_effort = _normalize_reasoning_request(payload.get("reasoning"))
    response_format = _normalize_text_format(payload.get("text"))
    verbosity = _text_verbosity(payload.get("text"))
    tools = _convert_tools(payload.get("tools"))
    tool_choice = _convert_tool_choice(payload.get("tool_choice"))

    max_output_tokens = payload.get("max_output_tokens")
    if max_output_tokens is not None:
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens < 16
        ):
            _fail(
                "max_output_tokens must be an integer greater than or equal to 16.",
                "invalid_max_output_tokens",
            )

    fields: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "store": store,
        "metadata": metadata,
        "parallel_tool_calls": parallel_tool_calls,
        "include": includes,
        "truncation": truncation,
    }
    if reasoning_effort is not None:
        fields["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        fields["response_format"] = response_format
    if verbosity is not None:
        fields["verbosity"] = verbosity
    if max_output_tokens is not None:
        fields["max_completion_tokens"] = max_output_tokens
    if tools is not None:
        fields["tools"] = tools
    if tool_choice is not None:
        fields["tool_choice"] = tool_choice
    user = payload.get("user")
    if user is not None:
        fields["user"] = _string(user, "user", "invalid_user")
    return ChatCompletionRequest(**fields)


def _select_reasoning_text(value: Mapping[str, Any]) -> str:
    primary = value.get("reasoning_content")
    if isinstance(primary, str) and primary:
        return primary
    mirror = value.get("reasoning")
    if isinstance(mirror, str) and mirror:
        return mirror
    return ""


def _chat_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif item.get("type") == "refusal":
            refusal = item.get("refusal")
            if isinstance(refusal, str):
                parts.append(refusal)
    return "\n".join(parts)


def _chat_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        call = _as_dict(raw, f"chat.tool_calls[{index}]", "invalid_chat_response")
        function = _as_dict(
            call.get("function"),
            f"chat.tool_calls[{index}].function",
            "invalid_chat_response",
        )
        name = _string(
            function.get("name"),
            f"chat.tool_calls[{index}].function.name",
            "invalid_chat_response",
        )
        arguments = function.get("arguments", "")
        if isinstance(arguments, (dict, list)):
            arguments = _json_text(arguments)
        elif not isinstance(arguments, str):
            _fail(
                "Chat tool arguments must be a JSON string.",
                "invalid_chat_response",
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = _new_id("call")
        result.append(
            {
                "id": call_id,
                "name": name,
                "arguments": arguments,
            }
        )
    return result


def _normalize_error(
    value: Any,
    *,
    default_code: str = "upstream_error",
    default_message: str = "The upstream Chat request failed.",
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        message = value.get("message")
        if not isinstance(message, str) or not message:
            message = default_message
        code = value.get("code")
        if not isinstance(code, str) or not code:
            code = default_code
        error_type = value.get("type")
        if not isinstance(error_type, str) or not error_type:
            error_type = "upstream_error"
        param = value.get("param")
        if param is not None and not isinstance(param, str):
            param = None
        return {
            "code": code,
            "message": message,
            "param": param,
            "type": error_type,
        }
    if isinstance(value, str) and value:
        return {
            "code": default_code,
            "message": value,
            "param": None,
            "type": "upstream_error",
        }
    return {
        "code": default_code,
        "message": default_message,
        "param": None,
        "type": "upstream_error",
    }


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.encode("utf-8"))


def _usage_from_chat(
    usage: Any,
    *,
    payload: dict[str, Any],
    output_text: str,
    reasoning_text: str,
) -> dict[str, Any]:
    source = usage if isinstance(usage, dict) else {}
    estimated = bool(source.get("estimated", True))
    input_tokens = source.get("input_tokens", source.get("prompt_tokens"))
    output_tokens = source.get(
        "output_tokens",
        source.get("completion_tokens"),
    )
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
        input_tokens = _estimate_tokens(_json_text(payload.get("input", "")))
        estimated = True
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
        output_tokens = _estimate_tokens(output_text + reasoning_text)
        estimated = True
    input_tokens = max(0, input_tokens)
    output_tokens = max(0, output_tokens)
    total_tokens = source.get("total_tokens")
    if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
        total_tokens = input_tokens + output_tokens
        estimated = True
    total_tokens = max(0, total_tokens)

    input_details = source.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    cached_tokens = input_details.get(
        "cached_tokens",
        source.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if isinstance(source.get("prompt_tokens_details"), dict)
        else 0,
    )
    if not isinstance(cached_tokens, int) or isinstance(cached_tokens, bool):
        cached_tokens = 0

    output_details = source.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}
    reasoning_tokens = output_details.get(
        "reasoning_tokens",
        source.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        if isinstance(source.get("completion_tokens_details"), dict)
        else 0,
    )
    if not isinstance(reasoning_tokens, int) or isinstance(reasoning_tokens, bool):
        reasoning_tokens = _estimate_tokens(reasoning_text)
        estimated = True
    reasoning_tokens = max(0, reasoning_tokens)

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": max(0, cached_tokens),
            "estimated": estimated,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
            "estimated": estimated,
        },
        "total_tokens": total_tokens,
        "estimated": estimated,
    }


def _response_text_config(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        return {"format": copy.deepcopy(text["format"])}
    return {"format": {"type": "text"}}


def _response_reasoning_config(payload: dict[str, Any]) -> dict[str, Any]:
    reasoning = payload.get("reasoning")
    effort = None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    return {"effort": effort, "summary": None}


def build_response(
    payload: dict[str, Any],
    *,
    output: list[dict[str, Any]],
    response_id: str | None = None,
    model: str | None = None,
    created_at: int | None = None,
    status: Literal["in_progress", "completed", "incomplete", "failed"] = "completed",
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """Build a complete Responses response object.

    ``store`` defaults to the request value and to ``True`` when omitted, as
    required by the Responses API. The parent owns the actual persistence.
    """

    if not isinstance(payload, dict):
        _fail("payload must be an object.", "invalid_payload")
    if status not in {"in_progress", "completed", "incomplete", "failed"}:
        _fail("Invalid response status.", "invalid_response_status")
    store = _validate_store(payload.get("store"))
    created = int(created_at if created_at is not None else time.time())
    response_model = model or payload.get("model") or ""
    if not isinstance(response_model, str):
        response_model = str(response_model)
    instructions = payload.get("instructions")
    result: dict[str, Any] = {
        "id": _response_id(response_id),
        "object": "response",
        "created_at": created,
        "status": status,
        "error": error,
        "incomplete_details": (
            {"reason": incomplete_reason}
            if status == "incomplete" and incomplete_reason is not None
            else None
        ),
        "instructions": instructions,
        "max_output_tokens": payload.get("max_output_tokens"),
        "model": response_model,
        "output": output,
        "parallel_tool_calls": (
            True
            if payload.get("parallel_tool_calls") is None
            else bool(payload.get("parallel_tool_calls"))
        ),
        "previous_response_id": payload.get("previous_response_id"),
        "reasoning": _response_reasoning_config(payload),
        "store": store,
        "text": _response_text_config(payload),
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": copy.deepcopy(payload.get("tools", [])),
        "usage": usage,
        "metadata": _validate_metadata(payload.get("metadata")),
        "user": payload.get("user"),
        "include": _validate_include(payload.get("include")),
        "truncation": _validate_truncation(payload.get("truncation")),
    }
    if status != "in_progress":
        result["completed_at"] = int(time.time())
    return result


def _reasoning_item(
    text: str,
    *,
    item_id: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": item_id or _new_id("rs"),
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
        "status": status,
    }


def _message_item(
    text: str,
    *,
    item_id: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": item_id or _new_id("msg"),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _function_call_item(
    *,
    call_id: str,
    name: str,
    arguments: str,
    item_id: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": item_id or _new_id("fc"),
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def chat_to_response(
    chat: dict[str, Any],
    payload: dict[str, Any],
    response_id: str | None = None,
) -> dict[str, Any]:
    """Convert one non-streaming Chat completion into a Responses object."""

    if not isinstance(chat, dict):
        _fail("chat must be an object.", "invalid_chat_response")
    if not isinstance(payload, dict):
        _fail("payload must be an object.", "invalid_payload")
    choices = chat.get("choices")
    if not isinstance(choices, list):
        choices = []
    message: dict[str, Any] = {}
    finish_reason: str | None = None
    if choices:
        choice = choices[0]
        if isinstance(choice, dict):
            raw_message = choice.get("message")
            if isinstance(raw_message, dict):
                message = raw_message
            raw_finish = choice.get("finish_reason")
            if isinstance(raw_finish, str):
                finish_reason = raw_finish

    reasoning = _select_reasoning_text(message)
    content = _chat_content_text(message.get("content"))
    calls = _chat_tool_calls(message.get("tool_calls"))
    chat_error = chat.get("error")
    error = None
    if chat_error is not None:
        error = _normalize_error(chat_error)

    output: list[dict[str, Any]] = []
    status: Literal["completed", "incomplete", "failed"] = "completed"
    incomplete_reason: str | None = None
    if error is not None:
        status = "failed"
    elif finish_reason == "length":
        status = "incomplete"
        incomplete_reason = "max_output_tokens"
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_reason = "content_filter"
    elif not reasoning and not content and not calls:
        status = "incomplete"
        incomplete_reason = "max_output_tokens"

    item_status = "completed" if status == "completed" else "incomplete"
    if reasoning:
        output.append(_reasoning_item(reasoning, status=item_status))
    if content:
        output.append(_message_item(content, status=item_status))
    for call in calls:
        output.append(
            _function_call_item(
                call_id=call["id"],
                name=call["name"],
                arguments=call["arguments"],
                status=item_status,
            )
        )

    usage = _usage_from_chat(
        chat.get("usage"),
        payload=payload,
        output_text=content,
        reasoning_text=reasoning,
    )
    model = chat.get("model")
    if not isinstance(model, str) or not model:
        model = payload.get("model")
    created = chat.get("created")
    if not isinstance(created, int) or isinstance(created, bool):
        created = int(time.time())
    response = build_response(
        payload,
        output=output,
        response_id=response_id,
        model=model if isinstance(model, str) else None,
        created_at=created,
        status=status,
        usage=usage,
        error=error,
        incomplete_reason=incomplete_reason,
    )
    if response["usage"] is None:
        response["usage"] = _usage_from_chat(
            None,
            payload=payload,
            output_text=content,
            reasoning_text=reasoning,
        )
    return response


def response_output_to_messages(response: dict[str, Any]) -> list[ChatMessage]:
    """Convert stored Responses output items into replayable Chat messages."""

    if not isinstance(response, dict):
        _fail("response must be an object.", "invalid_response")
    output = response.get("output")
    if not isinstance(output, list):
        _fail("response.output must be an array.", "invalid_response")
    messages: list[ChatMessage] = []
    for index, raw in enumerate(output):
        item = _as_dict(raw, f"response.output[{index}]", "invalid_response")
        kind = item.get("type")
        if kind == "reasoning":
            summary = _extract_reasoning_parts(
                item.get("summary"),
                label=f"response.output[{index}].summary",
                expected_type="summary_text",
            )
            content = _extract_reasoning_parts(
                item.get("content"),
                label=f"response.output[{index}].content",
                expected_type="reasoning_text",
            )
            extras: dict[str, Any] = {}
            if item.get("id") is not None:
                extras["response_item_id"] = item["id"]
            reasoning_text = "\n".join(content or summary)
            if reasoning_text:
                extras["reasoning_content"] = reasoning_text
            if summary:
                extras["reasoning_summary"] = "\n".join(summary)
            encrypted = item.get("encrypted_content")
            if encrypted is not None:
                extras["encrypted_content"] = encrypted
                extras["opaque_reasoning"] = copy.deepcopy(item)
            messages.append(
                ChatMessage(role="assistant", content=None, **extras)
            )
            continue
        if kind == "message":
            raw_content = item.get("content")
            if isinstance(raw_content, str):
                text: str | list[dict[str, Any]] = raw_content
            else:
                text = _chat_content_text(raw_content)
            extras = {}
            if item.get("id") is not None:
                extras["response_item_id"] = item["id"]
            annotations = []
            if isinstance(raw_content, list):
                for part in raw_content:
                    if isinstance(part, dict) and isinstance(
                        part.get("annotations"), list
                    ):
                        annotations.extend(part["annotations"])
            if annotations:
                extras["annotations"] = annotations
            messages.append(
                ChatMessage(role="assistant", content=text, **extras)
            )
            continue
        if kind == "function_call":
            call_id = _string(
                item.get("call_id"),
                f"response.output[{index}].call_id",
                "invalid_response",
            )
            name = _string(
                item.get("name"),
                f"response.output[{index}].name",
                "invalid_response",
            )
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                _fail(
                    f"response.output[{index}].arguments must be a string.",
                    "invalid_response",
                )
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            if (
                messages
                and messages[-1].role == "assistant"
                and messages[-1].content is None
                and isinstance(getattr(messages[-1], "tool_calls", None), list)
            ):
                messages[-1].tool_calls.append(call)
            else:
                extras = {}
                if item.get("id") is not None:
                    extras["response_item_ids"] = [item["id"]]
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[call],
                        **extras,
                    )
                )
            continue
        if kind == "function_call_output":
            call_id = _string(
                item.get("call_id"),
                f"response.output[{index}].call_id",
                "invalid_response",
            )
            content = _function_output_content(item.get("output"), index)
            extras = {"tool_call_id": call_id}
            if item.get("name") is not None:
                extras["name"] = item["name"]
            messages.append(
                ChatMessage(role="tool", content=content, **extras)
            )
            continue
        _fail(
            f"Unsupported response output item type: {kind!r}.",
            "unsupported_output_item",
        )
    return messages


class _SSEDecoder:
    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not isinstance(text, str):
            _fail("SSE chunks must be strings.", "invalid_stream")
        self.buffer = (self.buffer + text).replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        events: list[tuple[str, str]] = []
        while "\n\n" in self.buffer:
            block, self.buffer = self.buffer.split("\n\n", 1)
            parsed = self._parse(block)
            if parsed is not None:
                events.append(parsed)
        return events

    def flush(self) -> list[tuple[str, str]]:
        if not self.buffer.strip():
            self.buffer = ""
            return []
        block, self.buffer = self.buffer, ""
        parsed = self._parse(block)
        return [parsed] if parsed is not None else []

    @staticmethod
    def _parse(block: str) -> tuple[str, str] | None:
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].lstrip(" ")
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if not data_lines:
            return None
        return event_name, "\n".join(data_lines)


def _sse_event(name: str, data: dict[str, Any], sequence: int) -> str:
    body = dict(data)
    body["type"] = name
    body["sequence_number"] = sequence
    return (
        f"event: {name}\n"
        f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def _stream_delta_text(delta: dict[str, Any], key: str) -> str:
    if key == "reasoning":
        value = delta.get("reasoning_content")
        if isinstance(value, str) and value:
            return value
        value = delta.get("reasoning")
        return value if isinstance(value, str) else ""
    value = delta.get("content")
    if isinstance(value, str):
        return value
    return _chat_content_text(value)


def _stream_tool_delta(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            function = {}
        index = raw.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            index = 0
        call_id = raw.get("id")
        if not isinstance(call_id, str):
            call_id = ""
        name = function.get("name")
        if not isinstance(name, str):
            name = ""
        arguments = function.get("arguments", "")
        if isinstance(arguments, (dict, list)):
            arguments = _json_text(arguments)
        elif not isinstance(arguments, str):
            arguments = ""
        result.append(
            {
                "index": index,
                "id": call_id,
                "name": name,
                "arguments": arguments,
            }
        )
    return result


@dataclass
class _OpenStreamItem:
    kind: str
    output_index: int
    item_id: str
    text_parts: list[str] = field(default_factory=list)
    call_id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "".join(self.text_parts)

    def arguments(self) -> str:
        return "".join(self.arguments_parts)


def _finalize_stream_item(
    state: _OpenStreamItem,
    status: str,
) -> list[tuple[str, dict[str, Any]]]:
    if state.kind == "reasoning":
        text = state.text()
        part = {"type": "summary_text", "text": text}
        item = _reasoning_item(text, item_id=state.item_id, status=status)
        return [
            (
                "response.reasoning_summary_text.done",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "summary_index": 0,
                    "text": text,
                },
            ),
            (
                "response.reasoning_summary_part.done",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "summary_index": 0,
                    "part": part,
                },
            ),
            (
                "response.output_item.done",
                {"output_index": state.output_index, "item": item},
            ),
        ]
    if state.kind == "message":
        text = state.text()
        part = {
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }
        item = _message_item(text, item_id=state.item_id, status=status)
        return [
            (
                "response.output_text.done",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
            ),
            (
                "response.content_part.done",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "part": part,
                },
            ),
            (
                "response.output_item.done",
                {"output_index": state.output_index, "item": item},
            ),
        ]
    arguments = state.arguments()
    item = _function_call_item(
        call_id=state.call_id,
        name=state.name,
        arguments=arguments,
        item_id=state.item_id,
        status=status,
    )
    return [
        (
            "response.function_call_arguments.done",
            {
                "item_id": state.item_id,
                "output_index": state.output_index,
                "arguments": arguments,
            },
        ),
        (
            "response.output_item.done",
            {"output_index": state.output_index, "item": item},
        ),
    ]


async def chat_stream_to_responses(
    chunks: AsyncIterator[str],
    payload: dict[str, Any],
    response_id: str | None = None,
) -> AsyncIterator[str]:
    """Translate Chat completion SSE chunks into Responses SSE events."""

    if not isinstance(payload, dict):
        _fail("payload must be an object.", "invalid_payload")
    _validate_payload_shape(payload)
    created_at = int(time.time())
    response_id_value = _response_id(response_id)
    initial = build_response(
        payload,
        output=[],
        response_id=response_id_value,
        created_at=created_at,
        status="in_progress",
    )
    sequence = 0
    yield _sse_event(
        "response.created",
        {"response": copy.deepcopy(initial)},
        sequence,
    )
    sequence += 1
    yield _sse_event(
        "response.in_progress",
        {"response": copy.deepcopy(initial)},
        sequence,
    )
    sequence += 1

    decoder = _SSEDecoder()
    states: dict[int, _OpenStreamItem] = {}
    tool_states: dict[int, _OpenStreamItem] = {}
    next_output_index = 0
    reasoning_state: _OpenStreamItem | None = None
    message_state: _OpenStreamItem | None = None
    finish_reason: str | None = None
    usage: Any = None
    saw_done = False
    error: dict[str, Any] | None = None

    def new_state(kind: str, item_id: str) -> _OpenStreamItem:
        nonlocal next_output_index
        state = _OpenStreamItem(
            kind=kind,
            output_index=next_output_index,
            item_id=item_id,
        )
        states[next_output_index] = state
        next_output_index += 1
        return state

    def error_from_payload(
        event_name: str,
        value: Any,
    ) -> dict[str, Any] | None:
        if event_name == "error":
            return _normalize_error(value)
        if isinstance(value, dict):
            raw_error = value.get("error")
            if raw_error is not None:
                return _normalize_error(raw_error)
            if value.get("type") == "error":
                return _normalize_error(value)
            if value.get("object") == "error":
                return _normalize_error(value)
        return None

    def finalize_items(item_status: str) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []
        for state in sorted(states.values(), key=lambda item: item.output_index):
            for name, data in _finalize_stream_item(state, item_status):
                output_items.append({"__event__": name, "data": data})
        return output_items

    try:
        async for chunk in chunks:
            for event_name, data in decoder.feed(chunk):
                if data.strip() == "[DONE]":
                    saw_done = True
                    break
                try:
                    value = json.loads(data)
                except json.JSONDecodeError:
                    error = _normalize_error(
                        "Malformed SSE JSON from the Chat stream.",
                        default_code="invalid_stream_event",
                    )
                    break
                parsed_error = error_from_payload(event_name, value)
                if parsed_error is not None:
                    error = parsed_error
                    break
                if not isinstance(value, dict):
                    continue
                raw_usage = value.get("usage")
                if raw_usage is not None:
                    usage = raw_usage
                choices = value.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                raw_finish = choice.get("finish_reason")
                if isinstance(raw_finish, str):
                    finish_reason = raw_finish
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = {}

                reasoning_delta = _stream_delta_text(delta, "reasoning")
                if reasoning_delta:
                    if reasoning_state is None:
                        reasoning_state = new_state("reasoning", _new_id("rs"))
                        yield _sse_event(
                            "response.output_item.added",
                            {
                                "output_index": reasoning_state.output_index,
                                "item": {
                                    "id": reasoning_state.item_id,
                                    "type": "reasoning",
                                    "summary": [],
                                    "status": "in_progress",
                                },
                            },
                            sequence,
                        )
                        sequence += 1
                        yield _sse_event(
                            "response.reasoning_summary_part.added",
                            {
                                "item_id": reasoning_state.item_id,
                                "output_index": reasoning_state.output_index,
                                "summary_index": 0,
                                "part": {"type": "summary_text", "text": ""},
                            },
                            sequence,
                        )
                    reasoning_state.text_parts.append(reasoning_delta)
                    sequence += 1
                    yield _sse_event(
                        "response.reasoning_summary_text.delta",
                        {
                            "item_id": reasoning_state.item_id,
                            "output_index": reasoning_state.output_index,
                            "summary_index": 0,
                            "delta": reasoning_delta,
                        },
                        sequence,
                    )

                content_delta = _stream_delta_text(delta, "content")
                if content_delta:
                    if message_state is None:
                        message_state = new_state("message", _new_id("msg"))
                        yield _sse_event(
                            "response.output_item.added",
                            {
                                "output_index": message_state.output_index,
                                "item": {
                                    "id": message_state.item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                },
                            },
                            sequence,
                        )
                        sequence += 1
                        yield _sse_event(
                            "response.content_part.added",
                            {
                                "item_id": message_state.item_id,
                                "output_index": message_state.output_index,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                    "logprobs": [],
                                },
                            },
                            sequence,
                        )
                        sequence += 1
                    message_state.text_parts.append(content_delta)
                    yield _sse_event(
                        "response.output_text.delta",
                        {
                            "item_id": message_state.item_id,
                            "output_index": message_state.output_index,
                            "content_index": 0,
                            "delta": content_delta,
                            "logprobs": [],
                        },
                        sequence,
                    )
                    sequence += 1

                for call_delta in _stream_tool_delta(delta.get("tool_calls")):
                    tool_index = call_delta["index"]
                    state = tool_states.get(tool_index)
                    if state is None:
                        state = new_state("function_call", _new_id("fc"))
                        state.call_id = call_delta["id"] or _new_id("call")
                        state.name = call_delta["name"]
                        tool_states[tool_index] = state
                        yield _sse_event(
                            "response.output_item.added",
                            {
                                "output_index": state.output_index,
                                "item": {
                                    "id": state.item_id,
                                    "type": "function_call",
                                    "call_id": state.call_id,
                                    "name": state.name,
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            },
                            sequence,
                        )
                        sequence += 1
                    else:
                        if call_delta["id"]:
                            state.call_id = call_delta["id"]
                        if call_delta["name"]:
                            state.name = call_delta["name"]
                    argument_delta = call_delta["arguments"]
                    if argument_delta:
                        state.arguments_parts.append(argument_delta)
                        yield _sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "item_id": state.item_id,
                                "output_index": state.output_index,
                                "delta": argument_delta,
                            },
                            sequence,
                        )
            if saw_done or error is not None:
                break
        if not saw_done and error is None:
            for event_name, data in decoder.flush():
                if data.strip() == "[DONE]":
                    saw_done = True
                    break
                try:
                    value = json.loads(data)
                except json.JSONDecodeError:
                    error = _normalize_error(
                        "Malformed SSE JSON from the Chat stream.",
                        default_code="invalid_stream_event",
                    )
                    break
                parsed_error = error_from_payload(event_name, value)
                if parsed_error is not None:
                    error = parsed_error
                    break
    except Exception as exc:
        if isinstance(exc, GatewayError):
            error = _normalize_error(
                exc.as_openai()["error"],
                default_code="stream_error",
            )
        else:
            error = _normalize_error(
                None,
                default_code="stream_error",
                default_message="Chat stream failed unexpectedly",
            )

    if error is not None:
        item_status = "incomplete"
        finalized = finalize_items(item_status)
        for event in finalized:
            yield _sse_event(event["__event__"], event["data"], sequence)
            sequence += 1
        yield _sse_event(
            "error",
            {
                "code": error["code"],
                "message": error["message"],
                "param": error["param"],
            },
            sequence,
        )
        output_items = [
            event["data"]["item"]
            for event in finalized
            if event["__event__"] == "response.output_item.done"
        ]
        failed = build_response(
            payload,
            output=output_items,
            response_id=response_id_value,
            model=payload.get("model"),
            created_at=created_at,
            status="failed",
            usage=_usage_from_chat(
                usage,
                payload=payload,
                output_text="".join(
                    state.text() for state in states.values()
                    if state.kind == "message"
                ),
                reasoning_text="".join(
                    state.text() for state in states.values()
                    if state.kind == "reasoning"
                ),
            ),
            error=error,
        )
        yield _sse_event(
            "response.failed",
            {"response": failed},
            sequence,
        )
        sequence += 1
        return

    if finish_reason == "length":
        status = "incomplete"
        incomplete_reason = "max_output_tokens"
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_reason = "content_filter"
    elif not states and finish_reason != "length":
        status = "failed"
        incomplete_reason = None
    else:
        status = "completed"
        incomplete_reason = None

    if status == "completed" and not states and not saw_done:
        status = "failed"
    if status == "completed" and not states:
        status = "failed"

    item_status = "completed" if status == "completed" else "incomplete"
    finalized = finalize_items(item_status)
    for event in finalized:
        yield _sse_event(event["__event__"], event["data"], sequence)
        sequence += 1
    output_items = [
        event["data"]["item"]
        for event in finalized
        if event["__event__"] == "response.output_item.done"
    ]
    output_text = "".join(
        state.text() for state in states.values() if state.kind == "message"
    )
    reasoning_text = "".join(
        state.text() for state in states.values() if state.kind == "reasoning"
    )
    response_usage = _usage_from_chat(
        usage,
        payload=payload,
        output_text=output_text,
        reasoning_text=reasoning_text,
    )
    response_error = None
    if status == "failed":
        response_error = {
            "code": "empty_response",
            "message": "Chat stream contained no output.",
            "param": None,
            "type": "upstream_error",
        }
    response = build_response(
        payload,
        output=output_items,
        response_id=response_id_value,
        model=payload.get("model"),
        created_at=created_at,
        status=status,
        usage=response_usage,
        error=response_error,
        incomplete_reason=incomplete_reason,
    )
    terminal_event = {
        "completed": "response.completed",
        "incomplete": "response.incomplete",
        "failed": "response.failed",
    }[status]
    yield _sse_event(terminal_event, {"response": response}, sequence)


def assemble_response_stream(events: Iterable[str]) -> dict[str, Any]:
    """Return the complete response carried by a terminal Responses SSE event."""

    decoder = _SSEDecoder()
    final: dict[str, Any] | None = None
    for event_name, data in decoder.feed("".join(events)):
        if data.strip() == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ResponsesConversionError(
                "Malformed Responses SSE JSON.",
                code="invalid_stream_event",
            ) from exc
        if not isinstance(value, dict):
            continue
        kind = value.get("type", event_name)
        if kind in _TERMINAL_RESPONSE_EVENTS and isinstance(
            value.get("response"), dict
        ):
            final = value["response"]
    if final is None:
        raise ResponsesConversionError(
            "The Responses stream did not contain a terminal response.",
            code="missing_terminal_response",
        )
    return final
