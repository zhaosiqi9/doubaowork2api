from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestLogEntry:
    id: str
    started_at: float
    model: str
    stream: bool
    status: str = "running"
    account_id: str = ""
    error: str = ""
    finished_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        duration = (
            (self.finished_at or time.time()) - self.started_at
            if self.started_at
            else 0.0
        )
        return {
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": max(0, round(duration * 1000)),
            "model": self.model,
            "stream": self.stream,
            "status": self.status,
            "account_id": self.account_id,
            "error": self.error,
            "meta": self.meta,
        }


class RequestLog:
    def __init__(self, capacity: int = 200) -> None:
        self._entries: deque[RequestLogEntry] = deque(maxlen=capacity)

    def start(
        self,
        *,
        model: str,
        stream: bool,
        meta: dict[str, Any] | None = None,
    ) -> RequestLogEntry:
        entry = RequestLogEntry(
            id=f"req_{uuid.uuid4().hex[:12]}",
            started_at=time.time(),
            model=model,
            stream=stream,
            meta=meta or {},
        )
        self._entries.appendleft(entry)
        return entry

    def account(self, entry: RequestLogEntry, account_id: str) -> None:
        entry.account_id = account_id

    def success(self, entry: RequestLogEntry) -> None:
        entry.status = "success"
        entry.finished_at = time.time()

    def failure(self, entry: RequestLogEntry, error: str) -> None:
        entry.status = "error"
        entry.error = error[:500]
        entry.finished_at = time.time()

    def snapshot(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            entry.public_dict()
            for entry in list(self._entries)[: max(1, min(limit, 200))]
        ]
