from __future__ import annotations

import copy
import json
from typing import Any

from .models import ChatMessage
from .tool_bridge import (
    ToolBridgeRequestError,
    _check_loaded_json,
    _loads_limited,
    _NAME_PATTERN,
)


REASONING_HISTORY_START = "<historical_reasoning>"
REASONING_HISTORY_END = "</historical_reasoning>"
_REASONING_HISTORY_HEADER = "Historical assistant reasoning (not user content):"
_MISSING = object()


def normalize_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return a normalized copy of a chat history.

    Tool calls and results are paired by call id.  The function deliberately
    does not invent a result for a pending call, and it rejects history that
    cannot be paired without guessing.
    """
    if not isinstance(messages, list):
        raise _invalid("messages must be a list.")

    copies: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ChatMessage):
            raise _invalid(f"messages[{index}] must be a ChatMessage.")
        try:
            copies.append(message.model_copy(deep=True))
        except Exception as exc:
            raise _invalid(f"messages[{index}] cannot be copied.") from exc

    allocator = _LegacyIdAllocator(_raw_call_ids(copies))
    for message in copies:
        _convert_legacy_message(message, allocator)

    _normalize_message_calls(copies)
    _validate_content_preserved(copies)
    copies = _merge_adjacent_tool_call_assistants(copies)
    return _pair_and_order_tool_results(copies)


def render_reasoning_history(message: ChatMessage) -> str | None:
    """Render assistant reasoning as a separate historical block.

    Reasoning is never returned as message content.  Whitespace-only values
    are kept so callers can distinguish an explicitly empty reasoning field
    from a missing one.
    """
    if not isinstance(message, ChatMessage) or message.role != "assistant":
        return None

    extras = _extras(message)
    sources: list[str] = []
    for field_name in ("reasoning_content", "reasoning"):
        value = _field(message, extras, field_name)
        if value is None:
            continue
        text = _reasoning_text(value)
        if text is not None:
            sources.append(text)

    details = _field(message, extras, "reasoning_details")
    if details is not None:
        text = _reasoning_text(details)
        if text is not None:
            sources.append(text)

    if not sources:
        return None

    # Do not repeat identical fields, but keep every distinct value.
    unique: list[str] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    body = "\n\n".join(unique)
    if REASONING_HISTORY_END in body:
        raise _invalid("Reasoning history contains a reserved closing marker.")
    return (
        f"{REASONING_HISTORY_START}\n"
        f"{_REASONING_HISTORY_HEADER}\n"
        f"{body}\n"
        f"{REASONING_HISTORY_END}"
    )


def _invalid(message: str) -> ToolBridgeRequestError:
    return ToolBridgeRequestError(message, code="invalid_tool_history")


def _extras(message: ChatMessage) -> dict[str, Any]:
    extras = getattr(message, "__pydantic_extra__", None)
    if isinstance(extras, dict):
        return extras
    extras = {}
    message.__pydantic_extra__ = extras
    return extras


def _field(message: ChatMessage, extras: dict[str, Any], name: str) -> Any:
    if name in extras:
        return extras[name]
    return getattr(message, name, None)


def _set_extra(message: ChatMessage, name: str, value: Any) -> None:
    _extras(message)[name] = value


def _delete_extra(message: ChatMessage, name: str) -> None:
    _extras(message).pop(name, None)


def _raw_call_ids(messages: list[ChatMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.role != "assistant":
            continue
        raw_calls = _field(message, _extras(message), "tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                ids.add(call["id"])
    return ids


class _LegacyIdAllocator:
    def __init__(self, used: set[str]) -> None:
        self._used = set(used)
        self._next = 1

    def new_id(self) -> str:
        while True:
            candidate = f"call_legacy_{self._next}"
            self._next += 1
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate


def _convert_legacy_message(
    message: ChatMessage, allocator: _LegacyIdAllocator
) -> None:
    extras = _extras(message)
    role = getattr(message, "role", None)

    if role == "function":
        # Legacy function results are represented as tool results in the
        # modern chat schema.  Preserve the function name when it is present.
        message.role = "tool"
        if message.name is None:
            function = extras.get("function")
            if isinstance(function, dict) and isinstance(
                function.get("name"), str
            ):
                message.name = function["name"]
        if "tool_call_id" not in extras and "function_call_id" in extras:
            _set_extra(message, "tool_call_id", extras["function_call_id"])
        return

    legacy = _field(message, extras, "function_call")
    if legacy is None:
        return
    if role != "assistant":
        raise _invalid("Only assistant messages may contain function_call.")
    if not isinstance(legacy, dict):
        raise _invalid("assistant.function_call must be an object.")
    if set(legacy) - {"name", "arguments", "id"}:
        raise _invalid("assistant.function_call contains unsupported fields.")

    name = legacy.get("name")
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise _invalid("assistant.function_call.name is invalid.")
    arguments = _history_arguments(legacy.get("arguments", _MISSING))
    call_id = legacy.get("id")
    if call_id is None:
        call_id = allocator.new_id()
    elif not isinstance(call_id, str) or not call_id.strip():
        raise _invalid("assistant.function_call.id must be non-empty.")

    raw_calls = _field(message, extras, "tool_calls")
    if raw_calls is None:
        calls: list[Any] = []
    elif isinstance(raw_calls, list):
        calls = list(raw_calls)
    else:
        raise _invalid("Assistant tool_calls must be a list.")
    calls.append(
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
    )
    _set_extra(message, "tool_calls", calls)
    _delete_extra(message, "function_call")


def _normalize_message_calls(messages: list[ChatMessage]) -> None:
    for index, message in enumerate(messages):
        extras = _extras(message)
        raw_calls = _field(message, extras, "tool_calls")
        if raw_calls is None:
            continue
        if message.role != "assistant":
            raise _invalid(
                f"messages[{index}] contains tool_calls but is not assistant."
            )
        if not isinstance(raw_calls, list):
            raise _invalid("Assistant tool_calls must be a list.")
        _set_extra(
            message,
            "tool_calls",
            [
                _normalize_call(call, index, call_index)
                for call_index, call in enumerate(raw_calls)
            ],
        )


def _validate_content_preserved(messages: list[ChatMessage]) -> None:
    for index, message in enumerate(messages):
        if message.content is None or message.content == "":
            continue
        if not isinstance(message.content, (str, list)):
            raise _invalid(f"messages[{index}].content is malformed.")


def _normalize_call(
    raw: Any, message_index: int, call_index: int
) -> dict[str, Any]:
    label = f"messages[{message_index}].tool_calls[{call_index}]"
    if not isinstance(raw, dict) or set(raw) != {"id", "type", "function"}:
        raise _invalid(f"{label} must contain id, type, and function.")
    call_id = raw.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise _invalid(f"{label}.id must be non-empty.")
    function = raw.get("function")
    if (
        raw.get("type") != "function"
        or not isinstance(function, dict)
        or "name" not in function
        or not set(function) <= {"name", "arguments"}
    ):
        raise _invalid(f"{label} is malformed.")
    name = function.get("name")
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise _invalid(f"{label}.function.name is invalid.")
    arguments = _history_arguments(function.get("arguments", _MISSING))
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _history_arguments(value: Any) -> str:
    if value is _MISSING or value is None:
        return "{}"
    if isinstance(value, str):
        if not value.strip():
            return "{}"
        try:
            parsed = _loads_limited(value)
        except (ValueError, RecursionError) as exc:
            raise _invalid("Tool call arguments must be valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise _invalid("Tool call arguments must encode a JSON object.")
        # A valid string is kept byte-for-byte, including whitespace and key
        # order.  This is important when replaying provider-generated history.
        return value
    if isinstance(value, dict):
        try:
            _check_loaded_json(value)
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise _invalid(
                "Tool call arguments must be JSON-safe object data."
            ) from exc
    raise _invalid("Tool call arguments must be a JSON string or object.")


def _merge_adjacent_tool_call_assistants(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    merged: list[ChatMessage] = []
    for message in messages:
        if (
            merged
            and _has_calls(merged[-1])
            and _has_calls(message)
            and merged[-1].role == "assistant"
            and message.role == "assistant"
        ):
            merged[-1] = _merge_assistant(merged[-1], message)
        else:
            merged.append(message)
    return merged


def _has_calls(message: ChatMessage) -> bool:
    calls = _field(message, _extras(message), "tool_calls")
    return isinstance(calls, list) and bool(calls)


def _merge_assistant(
    first: ChatMessage, second: ChatMessage
) -> ChatMessage:
    merged = first.model_copy(deep=True)
    first_extras = _extras(first)
    second_extras = _extras(second)
    merged_extras = _extras(merged)

    merged_calls = copy.deepcopy(first_extras.get("tool_calls") or [])
    merged_calls.extend(copy.deepcopy(second_extras.get("tool_calls") or []))
    _set_extra(merged, "tool_calls", merged_calls)

    merged.content = _merge_content(first.content, second.content)
    merged.name = _merge_name(first.name, second.name)
    _merge_extra_text(merged, first, second, "reasoning_content")
    _merge_extra_text(merged, first, second, "reasoning")
    _merge_reasoning_details(merged, first, second)

    handled = {
        "tool_calls",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
    }
    for key, value in second_extras.items():
        if key in handled:
            continue
        if key not in merged_extras:
            merged_extras[key] = copy.deepcopy(value)
        elif merged_extras[key] != value:
            raise _invalid(
                f"Cannot merge conflicting assistant field {key!r}."
            )
    return merged


def _merge_content(first: Any, second: Any) -> Any:
    if first is None:
        return copy.deepcopy(second)
    if second is None:
        return copy.deepcopy(first)
    if isinstance(first, str) and isinstance(second, str):
        if not first:
            return second
        if not second:
            return first
        return f"{first}\n\n{second}"
    if isinstance(first, list) and isinstance(second, list):
        return copy.deepcopy(first) + copy.deepcopy(second)
    raise _invalid("Cannot merge assistant content with incompatible types.")


def _merge_name(first: str | None, second: str | None) -> str | None:
    if first is None:
        return second
    if second is None or first == second:
        return first
    raise _invalid("Cannot merge adjacent assistant messages with different names.")


def _merge_extra_text(
    merged: ChatMessage,
    first: ChatMessage,
    second: ChatMessage,
    field_name: str,
) -> None:
    first_value = _field(first, _extras(first), field_name)
    second_value = _field(second, _extras(second), field_name)
    if first_value is None:
        if second_value is not None:
            _set_extra(merged, field_name, copy.deepcopy(second_value))
        return
    if second_value is None:
        return
    if isinstance(first_value, str) and isinstance(second_value, str):
        if not first_value:
            _set_extra(merged, field_name, second_value)
        elif second_value:
            _set_extra(
                merged, field_name, f"{first_value}\n\n{second_value}"
            )
        return
    if isinstance(first_value, list) and isinstance(second_value, list):
        _set_extra(
            merged,
            field_name,
            copy.deepcopy(first_value) + copy.deepcopy(second_value),
        )
        return
    raise _invalid(f"Cannot merge assistant {field_name} values.")


def _merge_reasoning_details(
    merged: ChatMessage, first: ChatMessage, second: ChatMessage
) -> None:
    first_value = _field(first, _extras(first), "reasoning_details")
    second_value = _field(second, _extras(second), "reasoning_details")
    if first_value is None:
        if second_value is not None:
            _set_extra(
                merged, "reasoning_details", copy.deepcopy(second_value)
            )
        return
    if second_value is None:
        return
    if isinstance(first_value, list) and isinstance(second_value, list):
        _set_extra(
            merged,
            "reasoning_details",
            copy.deepcopy(first_value) + copy.deepcopy(second_value),
        )
        return
    if first_value == second_value:
        return
    raise _invalid("Cannot merge assistant reasoning_details values.")


def _pair_and_order_tool_results(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    result: list[ChatMessage] = []
    segment: list[ChatMessage] = []
    seen_ids: set[str] = set()
    for message in messages:
        if message.role == "user":
            if segment:
                result.extend(_normalize_segment(segment, seen_ids))
                segment = []
            result.append(message)
        else:
            segment.append(message)
    if segment:
        result.extend(_normalize_segment(segment, seen_ids))
    return result


def _normalize_segment(
    segment: list[ChatMessage],
    seen_ids: set[str],
) -> list[ChatMessage]:
    calls_by_id: dict[str, dict[str, Any]] = {}
    calls_by_message: dict[int, list[dict[str, Any]]] = {}
    tool_messages: list[ChatMessage] = []

    for index, message in enumerate(segment):
        if message.role == "assistant":
            calls = _field(message, _extras(message), "tool_calls")
            if isinstance(calls, list) and calls:
                calls_by_message[index] = calls
                for call in calls:
                    call_id = call["id"]
                    if call_id in seen_ids:
                        raise _invalid(f"Duplicate tool call id: {call_id}.")
                    seen_ids.add(call_id)
                    calls_by_id[call_id] = call
        elif message.role == "tool":
            tool_messages.append(message)

    if not tool_messages:
        return segment

    results: dict[str, ChatMessage] = {}
    deferred: list[ChatMessage] = []
    for message in tool_messages:
        raw_id = _field(message, _extras(message), "tool_call_id")
        if raw_id is None or (
            isinstance(raw_id, str) and not raw_id.strip()
        ):
            deferred.append(message)
            continue
        if not isinstance(raw_id, str):
            raise _invalid("tool_call_id must be a string.")
        if raw_id not in calls_by_id:
            raise _invalid(
                f"Tool result references unknown call id: {raw_id}."
            )
        if raw_id in results:
            raise _invalid(f"Duplicate tool result for call id: {raw_id}.")
        results[raw_id] = message

    pending = [call_id for call_id in calls_by_id if call_id not in results]
    if deferred:
        if len(deferred) != 1 or len(pending) != 1:
            raise _invalid(
                "A tool result without tool_call_id is ambiguous."
            )
        message = deferred[0]
        call_id = pending[0]
        _set_extra(message, "tool_call_id", call_id)
        results[call_id] = message

    if not results:
        # Keep the segment structure intact rather than silently dropping
        # tool text when no calls or results could be paired.
        raise _invalid("Tool results cannot be paired with assistant calls.")

    ordered: list[ChatMessage] = []
    for index, message in enumerate(segment):
        if message.role == "tool":
            continue
        ordered.append(message)
        for call in calls_by_message.get(index, []):
            result = results.get(call["id"])
            if result is not None:
                if (
                    result.name is None
                    and "name" not in _extras(result)
                ):
                    result.name = call["function"]["name"]
                ordered.append(result)
    return ordered


def _reasoning_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            part
            for item in value
            if (part := _reasoning_text(item)) is not None
        ]
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("text", "content", "summary", "reasoning", "value"):
            if key in value:
                text = _reasoning_text(value[key])
                if text is not None:
                    return text
        return None
    return None
