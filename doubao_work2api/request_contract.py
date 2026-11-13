from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource, Unresolvable
from referencing.jsonschema import DRAFT202012

from .errors import GatewayError
from .models import ChatCompletionRequest


class ContractRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request_contract"


class ContractResponseError(GatewayError):
    status_code = 502
    error_type = "upstream_error"
    code = "invalid_response_contract"


_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_LOCAL_REF = re.compile(r"\A#(?:/.*|[A-Za-z_][A-Za-z0-9._~-]*)?\Z", re.DOTALL)
_MAX_STOP_SEQUENCES = 4


@dataclass(frozen=True)
class _ResponseContract:
    kind: Literal["text", "json_object", "json_schema"]
    schema: dict[str, Any] | None = None
    strict: bool = False


def validate_contract(request: ChatCompletionRequest) -> None:
    """Validate response-format, output-limit, stop, and tool schemas."""
    _contract(request)
    get_output_limit(request)
    _stop_sequences(request)
    verbosity = getattr(request, "verbosity", None)
    if verbosity is not None and verbosity not in {"low", "medium", "high"}:
        raise ContractRequestError("verbosity must be low, medium, or high.")
    tools = _request_field(request, "tools")
    if tools is not None:
        _tool_definitions(tools)


def render_contract_instructions(request: ChatCompletionRequest) -> str:
    """Return deterministic text instructions for the requested response contract."""
    contract = _contract(request)
    if contract.kind == "text":
        verbosity = getattr(request, "verbosity", None)
        return f"OUTPUT STYLE\nUse {verbosity} verbosity in the answer." if verbosity else ""
    if contract.kind == "json_object":
        return (
            "OUTPUT CONTRACT\n"
            "Return exactly one JSON object as the entire answer. Do not use a "
            "Markdown fence, commentary, or text outside the object."
        )
    schema_text = json.dumps(
        contract.schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    strict_note = (
        "The schema is strict: include every required property, use no "
        "undeclared properties, and use null only where the schema permits it."
        if contract.strict
        else "The schema is advisory at generation time but will be validated."
    )
    return (
        "OUTPUT CONTRACT\n"
        "Return exactly one JSON value as the entire answer. Do not use a "
        "Markdown fence, commentary, or text outside the value. The value must "
        f"validate against this JSON Schema: {schema_text}\n{strict_note}"
    )


def enforce_contract(text: str, request: ChatCompletionRequest) -> str:
    """Validate complete model text against response_format without rewriting values."""
    if not isinstance(text, str):
        raise ContractResponseError(
            "The model response must be text.",
            code="invalid_response_contract",
        )
    contract = _contract(request)
    if contract.kind == "text":
        return text

    candidate = _json_candidate(text)
    parsed = _loads_json(candidate)
    if contract.kind == "json_object" and not isinstance(parsed, dict):
        raise ContractResponseError(
            "The model response is not a top-level JSON object.",
            code="json_object_required",
        )
    if contract.schema is not None:
        _validate_instance(parsed, contract.schema)
    return _json(parsed)


def validate_tool_arguments(
    tool_calls: Any,
    tools: Any,
) -> None:
    """Validate model tool calls; strict schemas are checked in full."""
    definitions = _tool_definitions(tools if tools is not None else [])
    if tool_calls is None:
        return
    if not isinstance(tool_calls, list):
        raise ContractResponseError(
            "The model tool_calls value must be a list.",
            code="invalid_tool_arguments",
        )
    by_name = {definition["name"]: definition for definition in definitions}
    for index, call in enumerate(tool_calls):
        name, raw_arguments = _tool_call_parts(call, index)
        definition = by_name.get(name)
        if definition is None:
            raise ContractResponseError(
                f"Tool call {index} references an undeclared function.",
                code="undeclared_tool",
            )
        arguments = _loads_json(raw_arguments)
        if not isinstance(arguments, dict):
            raise ContractResponseError(
                f"Tool call {index} arguments must be a JSON object.",
                code="invalid_tool_arguments",
            )
        if definition["strict"]:
            _validate_instance(
                arguments,
                definition["schema"],
                tool_index=index,
            )


def get_output_limit(request: ChatCompletionRequest) -> int | None:
    """Return the shared positive output budget, rejecting conflicting aliases."""
    max_tokens = _positive_output_limit(request, "max_tokens")
    max_completion_tokens = _positive_output_limit(
        request, "max_completion_tokens"
    )
    if (
        max_tokens is not None
        and max_completion_tokens is not None
        and max_tokens != max_completion_tokens
    ):
        raise ContractRequestError(
            "max_tokens and max_completion_tokens must agree when both are set.",
            code="conflicting_output_limit",
        )
    return (
        max_completion_tokens
        if max_completion_tokens is not None
        else max_tokens
    )


def estimate_tokens(text: str) -> int:
    """Conservative local estimate: one token per valid UTF-8 byte."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("text is not valid UTF-8") from exc


def limit_output(
    text: str,
    request: ChatCompletionRequest,
) -> tuple[str, str]:
    """Apply stop sequences first, then a byte-budget-safe content truncation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _contract(request)
    sequences = _stop_sequences(request)
    stop_position = _first_stop(text, sequences)
    if stop_position is not None:
        return text[:stop_position], "stop"

    limit = get_output_limit(request)
    if limit is None or estimate_tokens(text) <= limit:
        return text, "stop"
    return _truncate_utf8(text, limit), "length"


def _request_field(request: ChatCompletionRequest, name: str) -> Any:
    extras = getattr(request, "__pydantic_extra__", None)
    if isinstance(extras, dict) and name in extras:
        return extras[name]
    return getattr(request, name, None)


def _is_explicit(request: ChatCompletionRequest, name: str) -> bool:
    extras = getattr(request, "__pydantic_extra__", None)
    if isinstance(extras, dict) and name in extras:
        return extras[name] is not None
    fields_set = getattr(request, "model_fields_set", set())
    if name in fields_set:
        return getattr(request, name, None) is not None
    model_extra = getattr(request, "model_extra", None)
    return (
        isinstance(model_extra, dict)
        and name in model_extra
        and model_extra[name] is not None
    )


def _positive_output_limit(
    request: ChatCompletionRequest, name: str
) -> int | None:
    if not _is_explicit(request, name):
        return None
    value = _request_field(request, name)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ContractRequestError(
            f"{name} must be a positive integer.",
            code="invalid_output_limit",
        )
    return value


def _stop_sequences(request: ChatCompletionRequest) -> tuple[str, ...]:
    value = _request_field(request, "stop")
    if value is None:
        return ()
    if isinstance(value, str):
        if not value:
            raise ContractRequestError(
                "stop must not be empty.",
                code="invalid_stop",
            )
        return (value,)
    if not isinstance(value, list):
        raise ContractRequestError(
            "stop must be a string or a list of strings.",
            code="invalid_stop",
        )
    if not value or len(value) > _MAX_STOP_SEQUENCES:
        raise ContractRequestError(
            f"stop must contain between 1 and {_MAX_STOP_SEQUENCES} strings.",
            code="invalid_stop",
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise ContractRequestError(
            "Every stop sequence must be a non-empty string.",
            code="invalid_stop",
        )
    return tuple(value)


def _first_stop(text: str, sequences: tuple[str, ...]) -> int | None:
    positions = [
        position
        for sequence in sequences
        if (position := text.find(sequence)) >= 0
    ]
    return min(positions) if positions else None


def _truncate_utf8(text: str, limit: int) -> str:
    used = 0
    end = 0
    for index, character in enumerate(text):
        width = len(character.encode("utf-8"))
        if used + width > limit:
            break
        used += width
        end = index + 1
    return text[:end]


def _contract(request: ChatCompletionRequest) -> _ResponseContract:
    raw = _request_field(request, "response_format")
    if raw is None:
        return _ResponseContract("text")
    if not isinstance(raw, dict):
        raise ContractRequestError(
            "response_format must be an object or null.",
            code="invalid_response_format",
        )
    kind = raw.get("type")
    if kind == "text":
        _require_keys(raw, {"type"}, "response_format")
        return _ResponseContract("text")
    if kind == "json_object":
        _require_keys(raw, {"type"}, "response_format")
        return _ResponseContract("json_object")
    if kind != "json_schema":
        raise ContractRequestError(
            "response_format.type must be text, json_object, or json_schema.",
            code="invalid_response_format",
        )

    _require_keys(raw, {"type", "json_schema"}, "response_format")
    definition = raw.get("json_schema")
    if not isinstance(definition, dict):
        raise ContractRequestError(
            "response_format.json_schema must be an object.",
            code="invalid_response_format",
        )
    _require_keys(
        definition,
        {"name", "schema"},
        "response_format.json_schema",
        optional={"description", "strict"},
    )
    name = definition.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ContractRequestError(
            "response_format.json_schema.name must be a non-empty string.",
            code="invalid_response_format",
        )
    strict = definition.get("strict", False)
    if not isinstance(strict, bool):
        raise ContractRequestError(
            "response_format.json_schema.strict must be a boolean.",
            code="invalid_response_format",
        )
    schema = definition.get("schema")
    if not isinstance(schema, dict):
        raise ContractRequestError(
            "response_format.json_schema.schema must be an object.",
            code="invalid_response_format",
        )
    _validate_schema_definition(schema)
    return _ResponseContract("json_schema", schema, strict)


def _require_keys(
    value: dict[str, Any],
    required: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    if set(value) - allowed:
        raise ContractRequestError(
            f"{label} contains unsupported fields.",
            code="invalid_response_format",
        )
    missing = required - set(value)
    if missing:
        raise ContractRequestError(
            f"{label} is missing required fields.",
            code="invalid_response_format",
        )


def _validate_schema_definition(schema: dict[str, Any]) -> None:
    _reject_non_local_references(schema)
    validator_class = validator_for(schema, default=Draft202012Validator)
    try:
        validator_class.check_schema(schema)
        _check_references(schema)
    except SchemaError as exc:
        path = _json_path(exc.absolute_path)
        raise ContractRequestError(
            f"Invalid JSON Schema at {path}.",
            code="invalid_json_schema",
        ) from exc
    except Unresolvable as exc:
        raise ContractRequestError(
            "A local JSON Schema reference does not resolve.",
            code="invalid_json_schema",
        ) from exc


def _check_references(schema: dict[str, Any]) -> None:
    """Resolve every local reference without fetching external resources."""
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    resolver = Registry().with_resource("urn:request-contract", resource).resolver(
        "urn:request-contract"
    )
    try:
        for reference in _iter_references(schema):
            resolver.lookup(reference)
    except Unresolvable as exc:
        raise ContractRequestError(
            "A local JSON Schema reference does not resolve.",
            code="invalid_json_schema",
        ) from exc


def _iter_references(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"$ref", "$dynamicRef"}:
                yield item
            yield from _iter_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_references(item)


def _reject_non_local_references(value: Any, path: tuple[Any, ...] = ()) -> None:
    if len(path) > 64:
        raise ContractRequestError("JSON Schema exceeds the maximum nesting depth.", code="invalid_json_schema")
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = (*path, key)
            if key in {"$ref", "$dynamicRef"}:
                if not isinstance(item, str):
                    raise ContractRequestError(
                        "JSON Schema references must be strings.",
                        code="invalid_json_schema",
                    )
                if not _LOCAL_REF.fullmatch(item):
                    raise ContractRequestError(
                        "Only local JSON Schema references beginning with '#' "
                        "are allowed.",
                        code="external_schema_reference",
                    )
            _reject_non_local_references(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_local_references(item, (*path, index))


def _json_candidate(text: str) -> str:
    stripped = text.strip()
    fence = _JSON_FENCE.fullmatch(text)
    if fence is not None:
        return fence.group("body").strip()
    return stripped


def _loads_json(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ContractResponseError(
            "The model response is not valid JSON.",
            code="invalid_json_response",
        ) from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractResponseError(
            "The model response cannot be serialized as JSON.",
            code="invalid_json_response",
        ) from exc


def _validate_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    tool_index: int | None = None,
) -> None:
    validator_class = validator_for(schema, default=Draft202012Validator)
    def deny_resource(uri):
        raise NoSuchResource(ref=uri)
    validator = validator_class(schema, registry=Registry(retrieve=deny_resource))
    try:
        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: (
                list(error.absolute_path),
                list(error.absolute_schema_path),
            ),
        )
    except (Unresolvable, RecursionError) as exc:
        raise ContractResponseError(
            "The requested JSON Schema contains an unresolved reference.",
            code="invalid_json_schema",
        ) from exc
    if not errors:
        return
    error = errors[0]
    instance_path = _json_path(error.absolute_path)
    schema_path = _json_path(error.absolute_schema_path)
    if tool_index is None:
        message = (
            f"The model response does not match the requested JSON Schema "
            f"at {instance_path} (schema {schema_path})."
        )
        code = "schema_validation_failed"
    else:
        message = (
            f"Tool call {tool_index} arguments do not match the strict "
            f"schema at {instance_path} (schema {schema_path})."
        )
        code = "invalid_tool_arguments"
    raise ContractResponseError(message, code=code)


def _json_path(parts: Any) -> str:
    rendered = "$"
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            escaped = str(part).replace("~", "~0").replace("/", "~1")
            rendered += f"/{escaped}"
    return rendered


def _tool_definitions(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise ContractRequestError(
            "tools must be a list or null.",
            code="invalid_tools",
        )
    definitions: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(tools):
        if not isinstance(item, dict):
            raise ContractRequestError(
                f"tools[{index}] must be an object.",
                code="invalid_tools",
            )
        function = item.get("function")
        if item.get("type") != "function" or not isinstance(function, dict):
            raise ContractRequestError(
                f"tools[{index}] must be a function definition.",
                code="invalid_tools",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ContractRequestError(
                f"tools[{index}].function.name must be a non-empty string.",
                code="invalid_tool_name",
            )
        if name in names:
            raise ContractRequestError(
                f"Duplicate tool name: {name}.",
                code="duplicate_tool",
            )
        names.add(name)
        strict = function.get("strict", False)
        if not isinstance(strict, bool):
            raise ContractRequestError(
                f"tools[{index}].function.strict must be a boolean.",
                code="invalid_tools",
            )
        schema = function.get("parameters")
        if schema is None:
            schema = {"type": "object"}
        if not isinstance(schema, dict):
            raise ContractRequestError(
                f"tools[{index}].function.parameters must be an object.",
                code="invalid_tools",
            )
        _validate_schema_definition(schema)
        definitions.append(
            {"name": name, "schema": schema, "strict": strict}
        )
    return definitions


def _tool_call_parts(call: Any, index: int) -> tuple[str, str]:
    if not isinstance(call, dict):
        raise ContractResponseError(
            f"Tool call {index} must be an object.",
            code="invalid_tool_response",
        )
    function = call.get("function")
    if (
        call.get("type") != "function"
        or not isinstance(function, dict)
        or not isinstance(function.get("name"), str)
        or not isinstance(function.get("arguments"), str)
    ):
        raise ContractResponseError(
            f"Tool call {index} is malformed.",
            code="invalid_tool_response",
        )
    return function["name"], function["arguments"]
