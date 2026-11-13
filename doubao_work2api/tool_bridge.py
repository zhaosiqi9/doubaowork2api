from __future__ import annotations

import copy
import json
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import GatewayError
from .models import ChatMessage


class ToolBridgeRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_tool_request"


class ToolBridgeResponseError(GatewayError):
    status_code = 502
    error_type = "upstream_error"
    code = "invalid_tool_response"


class _ToolFunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=64)
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class _ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"]
    function: _ToolFunctionDefinition


class _FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class _ToolEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_calls"]
    tool_calls: list[_FunctionCall]
    content: str | None = None


_ENVELOPE_START = '{"type":"tool_calls","tool_calls":'
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MARKDOWN_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n.*?\r?\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_MAX_TOOLS = 128
_MAX_HISTORY_CALLS = 128
_MAX_PARALLEL_CALLS = 16
_MAX_ARGUMENT_BYTES = 1_048_576
_MAX_ARGUMENT_DEPTH = 64
_MAX_ENVELOPE_BYTES = 8_388_608
_MISSING = object()


def validate_tools(
    tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = True,
) -> None:
    """Validate OpenAI function-tool fields without mutating the caller's data."""
    normalized = _normalize_tools(tools)
    choice = _normalize_tool_choice(tool_choice)
    _normalize_parallel_tool_calls(parallel_tool_calls)
    _validate_choice_against_tools(choice, normalized)


def render_tool_instructions(
    tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = True,
) -> str:
    """Render the strict text protocol used when function tools are declared."""
    normalized = _normalize_tools(tools)
    choice = _normalize_tool_choice(tool_choice)
    parallel = _normalize_parallel_tool_calls(parallel_tool_calls)
    _validate_choice_against_tools(choice, normalized)
    if not normalized:
        return ""

    names = [tool["function"]["name"] for tool in normalized]
    if choice[0] == "none":
        return (
            "TOOL CALLING PROTOCOL\n"
            f"The available tool names are: {_json(names)}.\n"
            "Tool choice is none for this turn. Do not call any tool and do "
            "not emit a tool-call envelope; answer with ordinary text."
        )

    mode = _render_choice(choice)
    if choice[0] == "function":
        parallel_rule = (
            "Parallel tool calls are allowed, but this choice requires exactly "
            "one call to the selected function."
        )
    elif not parallel:
        parallel_rule = (
            "Parallel tool calls are disabled: return at most one call."
        )
    else:
        parallel_rule = (
            f"Parallel tool calls are allowed, with at most "
            f"{_MAX_PARALLEL_CALLS} calls in one response."
        )

    lines = [
        "TOOL CALLING PROTOCOL",
        "The tools below are the only tools you may call. Tool execution is "
        "performed by the caller; never claim that a tool has already run.",
        mode,
        parallel_rule,
        "To call tools, return exactly one JSON object and no Markdown fence, "
        "commentary, or text outside it. Use this exact shape:",
        _ENVELOPE_START
        + '[{"name":"<declared function name>","arguments":"<JSON object string>"}]'
        + ',"content":null}',
        "Each arguments value must be a JSON-encoded object string, not a JSON "
        "object. If there is no call, return ordinary text and do not emit the "
        "envelope. Never invent a tool name or call a tool not listed below.",
        "This rule is absolute: never invent a tool name.",
        "Declared functions:",
        _json(normalized),
    ]
    return "\n".join(lines)


def render_tool_history(message: ChatMessage) -> str | None:
    """Render assistant tool calls or a tool result for the text-only upstream."""
    if not isinstance(message, ChatMessage):
        raise ToolBridgeRequestError(
            "message must be a ChatMessage.", code="invalid_tool_history"
        )

    extras = getattr(message, "__pydantic_extra__", None) or {}
    raw_calls = _message_field(message, extras, "tool_calls")
    if message.role == "tool":
        call_id = _message_field(message, extras, "tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ToolBridgeRequestError(
                "A tool message must have a non-empty tool_call_id.",
                code="invalid_tool_history",
            )
        name = message.name
        if name is not None:
            _validate_name(name, "tool history name")
        return _json(
            {
                "type": "tool_result",
                "tool_call_id": call_id,
                "name": name,
                "content": message.content,
            }
        )

    if raw_calls is None:
        return None
    if message.role != "assistant":
        raise ToolBridgeRequestError(
            "Only assistant messages may contain tool_calls.",
            code="invalid_tool_history",
        )
    calls = _validate_history_calls(raw_calls)
    return _json({"type": "tool_calls", "tool_calls": calls})


def parse_tool_response(
    text: Any,
    tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = True,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse an upstream answer into visible text and validated OpenAI tool calls."""
    normalized = _normalize_tools(tools)
    choice = _normalize_tool_choice(tool_choice)
    parallel = _normalize_parallel_tool_calls(parallel_tool_calls)
    _validate_choice_against_tools(choice, normalized)
    if not isinstance(text, str):
        raise ToolBridgeResponseError(
            "The model response must be text.", code="invalid_tool_response"
        )
    if not normalized or choice[0] == "none":
        return text, []
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolBridgeResponseError(
            "The model response is not valid UTF-8 text.",
            code="invalid_tool_response",
        ) from exc
    if len(encoded) > _MAX_ENVELOPE_BYTES:
        raise ToolBridgeResponseError(
            "The model response exceeds the maximum tool-envelope size.",
            code="response_too_large",
        )

    candidate = text.strip()
    if _MARKDOWN_FENCE.fullmatch(candidate):
        if choice[0] in {"required", "function"}:
            raise ToolBridgeResponseError(
                "The model returned a fenced tool-call envelope; fenced "
                "envelopes are treated as examples, not calls.",
                code="tool_call_required",
            )
        return text, []
    if not (candidate.startswith("{") or candidate.startswith('"')):
        if choice[0] in {"required", "function"}:
            raise ToolBridgeResponseError(
                "The model did not return the required tool-call envelope.",
                code="tool_call_required",
            )
        return text, []

    raw = _decode_tool_envelope(candidate)
    if raw is None:
        if choice[0] in {"required", "function"}:
            raise ToolBridgeResponseError(
                "The model returned malformed tool-call JSON.",
                code="invalid_tool_response",
            )
        return text, []
    if not isinstance(raw, dict) or raw.get("type") != "tool_calls":
        if choice[0] in {"required", "function"}:
            raise ToolBridgeResponseError(
                "The model did not return the required tool-call envelope.",
                code="tool_call_required",
            )
        return text, []

    envelope = _parse_envelope(raw)
    calls = _validate_model_calls(envelope.tool_calls)
    if not calls:
        raise ToolBridgeResponseError(
            "A tool-call envelope must contain at least one call.",
            code="invalid_tool_response",
        )
    if len(calls) > _MAX_PARALLEL_CALLS:
        raise ToolBridgeResponseError(
            f"The model returned more than {_MAX_PARALLEL_CALLS} tool calls.",
            code="too_many_tool_calls",
        )
    if not parallel and len(calls) > 1:
        raise ToolBridgeResponseError(
            "The model returned multiple tool calls while parallel calls are disabled.",
            code="parallel_tool_calls_disabled",
        )
    if choice[0] == "function":
        if len(calls) != 1 or calls[0].name != choice[1]:
            raise ToolBridgeResponseError(
                f"The model must call only {choice[1]!r} for this request.",
                code="tool_choice_violation",
            )

    by_name = {tool["function"]["name"]: tool for tool in normalized}
    result: list[dict[str, Any]] = []
    for call in calls:
        definition = by_name.get(call.name)
        if definition is None:
            raise ToolBridgeResponseError(
                f"The model called undeclared function {call.name!r}.",
                code="undeclared_tool",
            )
        _decode_arguments(call.arguments)
        result.append(
            {
                "id": _new_call_id(),
                "type": "function",
                "function": {
                    "name": call.name,
                    # Preserve the provider's valid JSON text verbatim.
                    "arguments": call.arguments,
                },
            }
        )
    return envelope.content, result


def _normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ToolBridgeRequestError(
            "tools must be a list or null.", code="invalid_tools"
        )
    if len(tools) > _MAX_TOOLS:
        raise ToolBridgeRequestError(
            f"At most {_MAX_TOOLS} tools may be declared.",
            code="too_many_tools",
        )
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(tools):
        if not isinstance(item, dict):
            raise ToolBridgeRequestError(
                f"tools[{index}] must be an object.", code="invalid_tools"
            )
        try:
            tool = _ToolDefinition.model_validate(item)
        except ValidationError as exc:
            raise ToolBridgeRequestError(
                f"tools[{index}] is not a valid function tool.",
                code="invalid_tools",
            ) from exc
        try:
            _json(item)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ToolBridgeRequestError(
                f"tools[{index}] contains a non-JSON-safe value.",
                code="invalid_tools",
            ) from exc
        name = tool.function.name
        if not _NAME_PATTERN.fullmatch(name):
            raise ToolBridgeRequestError(
                f"tools[{index}].function.name is invalid.",
                code="invalid_tool_name",
            )
        if name in names:
            raise ToolBridgeRequestError(
                f"Duplicate tool name: {name}.",
                code="duplicate_tool",
            )
        names.add(name)
        normalized.append(copy.deepcopy(item))
    return normalized


def _normalize_tool_choice(
    tool_choice: Any,
) -> tuple[str, str | None]:
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            raise ToolBridgeRequestError(
                "tool_choice must be auto, none, required, or a function object.",
                code="invalid_tool_choice",
            )
        return tool_choice, None
    if not isinstance(tool_choice, dict):
        raise ToolBridgeRequestError(
            "tool_choice must be a string or an object.",
            code="invalid_tool_choice",
        )
    if set(tool_choice) != {"type", "function"}:
        raise ToolBridgeRequestError(
            "A function tool_choice must contain only type and function.",
            code="invalid_tool_choice",
        )
    function = tool_choice.get("function")
    if (
        tool_choice.get("type") != "function"
        or not isinstance(function, dict)
        or set(function) != {"name"}
        or not isinstance(function.get("name"), str)
    ):
        raise ToolBridgeRequestError(
            "tool_choice.function must contain exactly one string name.",
            code="invalid_tool_choice",
        )
    name = function["name"]
    _validate_name(name, "tool_choice.function.name")
    return "function", name


def _normalize_parallel_tool_calls(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ToolBridgeRequestError(
            "parallel_tool_calls must be a boolean.",
            code="invalid_parallel_tool_calls",
        )
    return value


def _validate_choice_against_tools(
    choice: tuple[str, str | None], tools: list[dict[str, Any]]
) -> None:
    if choice[0] in {"required", "function"} and not tools:
        raise ToolBridgeRequestError(
            "tool_choice requires at least one declared tool.",
            code="tool_choice_without_tools",
        )
    if choice[0] == "function":
        names = {tool["function"]["name"] for tool in tools}
        if choice[1] not in names:
            raise ToolBridgeRequestError(
                f"tool_choice references undeclared function {choice[1]!r}.",
                code="undeclared_tool_choice",
            )


def _validate_name(name: Any, label: str) -> None:
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ToolBridgeRequestError(
            f"{label} is invalid.", code="invalid_tool_name"
        )


def _render_choice(choice: tuple[str, str | None]) -> str:
    if choice[0] == "auto":
        return (
            "Tool choice is auto: call one or more tools only when they are "
            "needed; otherwise answer normally."
        )
    if choice[0] == "required":
        return "Tool choice is required: you must return at least one tool call."
    return (
        "Tool choice is fixed: you must call only "
        f"{_json(choice[1])} exactly once."
    )


def _validate_history_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list) or not raw_calls:
        raise ToolBridgeRequestError(
            "Assistant tool_calls must be a non-empty list.",
            code="invalid_tool_history",
        )
    if len(raw_calls) > _MAX_HISTORY_CALLS:
        raise ToolBridgeRequestError(
            "Assistant tool_calls cannot exceed "
            f"{_MAX_HISTORY_CALLS} entries.",
            code="invalid_tool_history",
        )
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict) or set(raw) != {"id", "type", "function"}:
            raise ToolBridgeRequestError(
                f"tool_calls[{index}] must contain id, type, and function.",
                code="invalid_tool_history",
            )
        call_id = raw.get("id")
        function = raw.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or raw.get("type") != "function"
            or not isinstance(function, dict)
            or not {"name"} <= set(function)
            or not set(function) <= {"name", "arguments"}
        ):
            raise ToolBridgeRequestError(
                f"tool_calls[{index}] is malformed.",
                code="invalid_tool_history",
            )
        if call_id in ids:
            raise ToolBridgeRequestError(
                f"Duplicate tool call id: {call_id}.",
                code="invalid_tool_history",
            )
        ids.add(call_id)
        name = function.get("name")
        _validate_name(name, f"tool_calls[{index}].function.name")
        arguments = _normalize_history_arguments(
            function.get("arguments", _MISSING)
        )
        result.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return result


def _normalize_history_arguments(value: Any) -> str:
    if value is _MISSING or value is None:
        return "{}"
    if isinstance(value, str):
        if not value.strip():
            return "{}"
        try:
            parsed = _loads_limited(value)
        except (ValueError, RecursionError) as exc:
            raise ToolBridgeRequestError(
                "Tool call arguments must be valid JSON.",
                code="invalid_tool_history",
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolBridgeRequestError(
                "Tool call arguments must encode a JSON object.",
                code="invalid_tool_history",
            )
        # Keep the exact provider string, including key order and whitespace.
        return value
    if isinstance(value, dict):
        try:
            _check_loaded_json(value)
            return _json(value)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ToolBridgeRequestError(
                "Tool call arguments must be JSON-safe object data.",
                code="invalid_tool_history",
            ) from exc
    raise ToolBridgeRequestError(
        "Tool call arguments must be a JSON string or object.",
        code="invalid_tool_history",
    )


def _parse_envelope(raw: dict[str, Any]) -> _ToolEnvelope:
    allowed = {"type", "tool_calls", "content"}
    if set(raw) - allowed:
        raise ToolBridgeResponseError(
            "The tool-call envelope contains unknown fields.",
            code="invalid_tool_response",
        )
    calls = raw.get("tool_calls")
    if not isinstance(calls, list):
        raise ToolBridgeResponseError(
            "The tool-call envelope must contain a tool_calls list.",
            code="invalid_tool_response",
        )
    if len(calls) > _MAX_PARALLEL_CALLS:
        raise ToolBridgeResponseError(
            f"The model returned more than {_MAX_PARALLEL_CALLS} tool calls.",
            code="too_many_tool_calls",
        )
    for call in calls:
        if (
            not isinstance(call, dict)
            or "name" not in call
            or not set(call) <= {"name", "arguments"}
        ):
            raise ToolBridgeResponseError(
                "Each tool call must contain exactly name and arguments.",
                code="invalid_tool_response",
            )
        if not isinstance(call.get("name"), str):
            raise ToolBridgeResponseError(
                "Each tool call name must be a string.",
                code="invalid_tool_response",
            )
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            try:
                _check_loaded_json(arguments)
                call["arguments"] = _json(arguments)
            except (TypeError, ValueError, RecursionError) as exc:
                raise ToolBridgeResponseError(
                    "Tool call arguments must be JSON-safe object data.",
                    code="invalid_tool_response",
                ) from exc
        elif arguments is None and "arguments" in call:
            call["arguments"] = "{}"
        elif not isinstance(arguments, str):
            raise ToolBridgeResponseError(
                "Each tool call arguments value must be a string or object.",
                code="invalid_tool_response",
            )
    try:
        return _ToolEnvelope.model_validate(raw)
    except ValidationError as exc:
        raise ToolBridgeResponseError(
            "The tool-call envelope is malformed.",
            code="invalid_tool_response",
        ) from exc


def _validate_model_calls(calls: list[_FunctionCall]) -> list[_FunctionCall]:
    for call in calls:
        if not _NAME_PATTERN.fullmatch(call.name):
            raise ToolBridgeResponseError(
                f"The model returned an invalid tool name: {call.name!r}.",
                code="invalid_tool_response",
            )
        try:
            parsed = _loads_limited(call.arguments)
        except (ValueError, RecursionError) as exc:
            raise ToolBridgeResponseError(
                f"Arguments for {call.name!r} are not valid JSON.",
                code="invalid_tool_arguments",
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolBridgeResponseError(
                f"Arguments for {call.name!r} must encode a JSON object.",
                code="invalid_tool_arguments",
            )
    return calls


def _decode_tool_envelope(candidate: str) -> dict[str, Any] | None:
    """Decode only explicit tool-call envelopes with narrow repair support."""
    value: Any = candidate
    layers = 0
    while isinstance(value, str):
        if layers >= 4:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        try:
            decoded = json.loads(
                stripped, parse_constant=_reject_json_constant
            )
        except (ValueError, RecursionError):
            if not stripped.startswith("{"):
                return None
            decoded = _repair_explicit_envelope(stripped)
            if decoded is None:
                return None
        if isinstance(decoded, str) and decoded.strip() == stripped:
            return None
        value = decoded
        layers += 1

    if not isinstance(value, dict) or value.get("type") != "tool_calls":
        return None
    return _normalize_response_envelope(value)


def _repair_explicit_envelope(value: str) -> dict[str, Any] | None:
    if '"type"' not in value or '"tool_calls"' not in value:
        return None
    repaired = re.sub(
        r",(?=\s*[}\]])",
        "",
        value,
    )
    if repaired == value:
        return None
    try:
        decoded = json.loads(
            repaired, parse_constant=_reject_json_constant
        )
    except (ValueError, RecursionError):
        return None
    if not isinstance(decoded, dict) or decoded.get("type") != "tool_calls":
        return None
    return decoded


def _normalize_response_envelope(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(raw)
    calls = normalized.get("tool_calls")
    if not isinstance(calls, list):
        return normalized
    for call in calls:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            try:
                _check_loaded_json(arguments)
                call["arguments"] = _json(arguments)
            except (TypeError, ValueError, RecursionError):
                continue
        elif arguments is None and "arguments" in call:
            call["arguments"] = "{}"
        elif "arguments" not in call:
            call["arguments"] = "{}"
    return normalized


def _decode_arguments(value: str) -> dict[str, Any]:
    try:
        parsed = _loads_limited(value)
    except (ValueError, RecursionError) as exc:
        raise ToolBridgeResponseError(
            "Tool arguments are not valid JSON.",
            code="invalid_tool_arguments",
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolBridgeResponseError(
            "Tool arguments must encode a JSON object.",
            code="invalid_tool_arguments",
        )
    return parsed


def _loads_limited(value: str) -> Any:
    if not value.strip():
        raise ValueError("empty JSON")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("invalid UTF-8") from exc
    if len(encoded) > _MAX_ARGUMENT_BYTES:
        raise ValueError("JSON arguments are too large")
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    _check_loaded_json(parsed)
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _check_json_depth(value: Any, depth: int = 0) -> None:
    if depth > _MAX_ARGUMENT_DEPTH:
        raise ValueError(
            "JSON value exceeds the maximum nesting depth"
        )
    if isinstance(value, dict):
        for item in value.values():
            _check_json_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_json_depth(item, depth + 1)


def _check_loaded_json(value: Any) -> None:
    _check_json_depth(value)
    try:
        _json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("JSON value cannot be serialized safely") from exc


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def _message_field(
    message: ChatMessage, extras: dict[str, Any], name: str
) -> Any:
    return extras[name] if name in extras else getattr(message, name, None)


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
