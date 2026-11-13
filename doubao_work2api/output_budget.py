"""Incremental Unicode-safe output bounds and cross-chunk stop matching."""
from __future__ import annotations

from .models import ChatCompletionRequest
from .request_contract import estimate_tokens, get_output_limit


class OutputBudget:
    def __init__(self, request: ChatCompletionRequest) -> None:
        self.remaining = get_output_limit(request)
        self.stops = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
        self.pending = ""
        self.finished = False
        self.finish_reason = "stop"

    def _take(self, text: str) -> str:
        if self.remaining is None:
            return text
        if estimate_tokens(text) < self.remaining:
            self.remaining -= estimate_tokens(text)
            return text
        parts = []
        for character in text:
            cost = estimate_tokens(character)
            if cost > self.remaining:
                break
            parts.append(character)
            self.remaining -= cost
            if self.remaining == 0:
                break
        self.finished = True
        self.finish_reason = "length"
        return "".join(parts)

    def feed(self, kind: str, text: str) -> str:
        if self.finished:
            return ""
        if kind == "reasoning":
            return self._take(text)
        self.pending += text
        positions = [self.pending.find(stop) for stop in self.stops if stop in self.pending]
        if positions:
            output = self._take(self.pending[:min(positions)])
            self.pending = ""
            if not self.finished:
                self.finished = True
                self.finish_reason = "stop"
            return output
        keep = max((len(stop) - 1 for stop in self.stops), default=0)
        if keep:
            output, self.pending = self.pending[:-keep], self.pending[-keep:]
        else:
            output, self.pending = self.pending, ""
        return self._take(output)

    def flush(self) -> str:
        if self.finished:
            self.pending = ""
            return ""
        output, self.pending = self.pending, ""
        return self._take(output)
