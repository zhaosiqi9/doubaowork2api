import pytest

from doubao_work2api.context import ContextLengthError, apply_context_budget, conversation_groups
from doubao_work2api.models import ChatCompletionRequest


def test_auto_truncation_keeps_system_and_whole_tool_turns():
    messages = [
        {"role": "system", "content": "Keep this instruction"},
        {"role": "user", "content": "x" * 295000},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_old", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
        {"role": "user", "content": "latest question"},
    ]
    request = ChatCompletionRequest(messages=messages, truncation="auto")
    bounded = apply_context_budget(request)
    assert [message.content for message in bounded.messages] == ["Keep this instruction", "latest question"]
    assert len(request.messages) == 5
    assert bounded.gateway_context["dropped_turns"] == 1


def test_default_rejects_overflow_and_does_not_silently_drop():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "x" * 300000}])
    with pytest.raises(ContextLengthError):
        apply_context_budget(request)


def test_latest_turn_and_system_are_never_cut_in_half():
    request = ChatCompletionRequest(messages=[{"role": "system", "content": "x" * 295000}, {"role": "user", "content": "latest"}], truncation="auto")
    with pytest.raises(ContextLengthError):
        apply_context_budget(request)


def test_unknown_model_window_is_not_invented():
    request = ChatCompletionRequest(model="future-unverified-model", messages=[{"role": "user", "content": "hello"}])
    assert apply_context_budget(request).gateway_context["window"] is None


def test_output_budget_is_reserved_before_dispatch():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "x" * 10000}], max_completion_tokens=290000)
    with pytest.raises(ContextLengthError):
        apply_context_budget(request)
