from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.output_budget import OutputBudget


def request(**fields):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "test"}], **fields)


def test_stop_sequence_is_not_leaked_across_fragments():
    budget = OutputBudget(request(stop="END"))
    assert budget.feed("content", "hello E") == "hello"
    assert budget.feed("content", "N") == " "
    assert budget.feed("content", "Dignored") == ""
    assert budget.finished and budget.finish_reason == "stop"
    assert budget.flush() == ""


def test_unicode_budget_never_splits_codepoint():
    budget = OutputBudget(request(max_completion_tokens=4))
    assert budget.feed("content", "你ab") == "你a"
    assert budget.finished and budget.finish_reason == "length"


def test_reasoning_and_text_share_completion_budget():
    budget = OutputBudget(request(max_completion_tokens=5))
    assert budget.feed("reasoning", "abc") == "abc"
    assert budget.feed("content", "def") == "de"
    assert budget.finished


def test_partial_stop_prefix_at_eof_is_flushed():
    budget = OutputBudget(request(stop="STOP"))
    assert budget.feed("content", "hello ST") + budget.flush() == "hello ST"
    assert not budget.finished
