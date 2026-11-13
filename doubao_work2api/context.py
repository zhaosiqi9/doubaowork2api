"""Conservative gateway context budgets; never presented as upstream billing."""
from __future__ import annotations

import json
from typing import Any

from .catalog import find_model
from .errors import GatewayError
from .models import ChatCompletionRequest, ChatMessage
from .request_contract import estimate_tokens, get_output_limit, render_contract_instructions


class ContextLengthError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "context_length_exceeded"


def conversation_groups(messages: list[ChatMessage]) -> tuple[list[ChatMessage], list[list[ChatMessage]]]:
    """Keep complete user turns, including their function calls and results."""
    systems: list[ChatMessage] = []
    groups: list[list[ChatMessage]] = []
    for message in messages:
        if message.role in {"system", "developer"}:
            systems.append(message)
        else:
            if message.role == "user" or not groups:
                groups.append([])
            groups[-1].append(message)
    return systems, groups


def estimate_request_tokens(request: ChatCompletionRequest) -> int:
    from .protocol import render_prompt
    from .tool_bridge import render_tool_instructions

    text = render_prompt(request.messages)
    if request.tools:
        text += render_tool_instructions(request.tools, request.tool_choice, request.parallel_tool_calls)
    text += render_contract_instructions(request)
    images = sum(1 for message in request.messages if isinstance(message.content, list) for part in message.content if part.get("type") == "image_url")
    # Image encodings never count as text. The fixed allowance is a budget
    # estimate; the upstream does not expose an exact image tokenizer.
    return estimate_tokens(text) + images * 4096 + len(request.messages) * 8


def apply_context_budget(request: ChatCompletionRequest) -> ChatCompletionRequest:
    spec = find_model(request.normalized_model)
    window = spec.context_window if spec else None
    estimated = estimate_request_tokens(request)
    output = get_output_limit(request) or 4096
    # The live channel reports a server system prompt around 6K tokens.
    # Reserve more, rather than claiming all of the 300K belongs to the user.
    reserve = 8192
    old = getattr(request, "gateway_context", {})
    previous_dropped = old.get("dropped_turns", 0) if isinstance(old, dict) else 0
    previous_dropped = previous_dropped if type(previous_dropped) is int and previous_dropped >= 0 else 0
    metadata: dict[str, Any] = {"estimated_input_tokens": estimated, "reserved_output_tokens": output, "server_reserve": reserve, "window": window, "dropped_turns": previous_dropped, "is_estimated": True}
    if window is None:
        return request.model_copy(update={"gateway_context": metadata})
    available = window - reserve - output
    if estimated <= available:
        return request.model_copy(update={"gateway_context": metadata})
    if request.truncation != "auto":
        raise ContextLengthError("Estimated context exceeds the verified model window. Use truncation='auto' or /v1/responses/compact.")
    systems, groups = conversation_groups(request.messages)
    dropped = 0
    while len(groups) > 1:
        groups.pop(0)
        dropped += 1
        candidate = request.model_copy(update={"messages": systems + [message for group in groups for message in group]})
        estimated = estimate_request_tokens(candidate)
        if estimated <= available:
            metadata.update(estimated_input_tokens=estimated, dropped_turns=previous_dropped + dropped)
            return candidate.model_copy(update={"gateway_context": metadata})
    raise ContextLengthError("The system instructions and latest complete turn exceed the verified model window; they cannot be safely truncated.")
