from __future__ import annotations

import asyncio
import socket
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .accounts import Account, AccountPool
from .errors import (
    NoAvailableAccountError,
    UpstreamAuthError,
    UpstreamChallengeError,
    UpstreamError,
    UpstreamRateLimitError,
)
from .logs import RequestLog, RequestLogEntry
from .models import ChatCompletionRequest
from .protocol import (
    chunk_payload,
    completion_id,
    completion_payload,
    extract_text_fragments,
    phase_markers,
    sse_data,
    upstream_payload,
)
from .transport import DirectTransport


class ChatService:
    def __init__(
        self,
        pool: AccountPool,
        transport: DirectTransport,
        *,
        model_key: str,
        app_id: str,
        workspace_root: Path,
        default_model: str = "deepseek-v4.1-flash",
        request_log: RequestLog | None = None,
    ) -> None:
        self.pool = pool
        self.transport = transport
        self.model_key = model_key
        self.app_id = app_id
        self.workspace_root = workspace_root
        self.default_model = default_model
        self.request_log = request_log

    async def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        log_entry = self._start_log(request)
        content: list[str] = []
        reasoning: list[str] = []
        try:
            async for kind, text in self._fragments_with_retry(
                request, log_entry
            ):
                if kind == "content":
                    content.append(text)
                else:
                    reasoning.append(text)
            answer = "".join(content)
            if not answer:
                raise UpstreamError("Doubao returned an empty response.")
            prompt_text = "".join(
                str(message.content or "") for message in request.messages
            )
            result = completion_payload(
                request_id=completion_id(),
                model=request.normalized_model,
                content=answer,
                reasoning="".join(reasoning),
                prompt_tokens=max(1, len(prompt_text) // 3),
                completion_tokens=max(1, len(answer) // 3),
            )
            self._log_success(log_entry)
            return result
        except Exception as exc:
            self._log_failure(log_entry, str(exc))
            raise
        finally:
            self._log_interrupted(log_entry)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        log_entry = self._start_log(request)
        request_id = completion_id()
        try:
            yield sse_data(
                chunk_payload(
                    request_id=request_id,
                    model=request.normalized_model,
                    role="assistant",
                )
            )
            async for kind, text in self._fragments_with_retry(
                request, log_entry
            ):
                field = "content" if kind == "content" else "reasoning"
                yield sse_data(
                    chunk_payload(
                        request_id=request_id,
                        model=request.normalized_model,
                        **{field: text},
                    )
                )
            yield sse_data(
                chunk_payload(
                    request_id=request_id,
                    model=request.normalized_model,
                    finish_reason="stop",
                )
            )
            yield "data: [DONE]\n\n"
            self._log_success(log_entry)
        except Exception as exc:
            self._log_failure(log_entry, str(exc))
            raise
        finally:
            self._log_interrupted(log_entry)

    async def _fragments_with_retry(
        self,
        request: ChatCompletionRequest,
        log_entry: RequestLogEntry | None,
    ) -> AsyncIterator[tuple[str, str]]:
        attempted: set[str] = set()
        last_error: Exception | None = None
        for _ in range(max(1, len(self.pool.accounts))):
            try:
                account = await self.pool.acquire()
            except NoAvailableAccountError:
                if last_error:
                    raise last_error
                raise
            if account.id in attempted:
                await self.pool.release(account)
                break
            attempted.add(account.id)
            if self.request_log is not None and log_entry is not None:
                self.request_log.account(log_entry, account.id)
            emitted = False
            try:
                async for fragment in self._fragments(account, request):
                    emitted = True
                    yield fragment
                if not emitted:
                    raise UpstreamError(
                        "Doubao returned an empty response.", code="empty_response"
                    )
                self.pool.mark_success(account)
                return
            except UpstreamAuthError as exc:
                if emitted:
                    raise
                try:
                    await self.pool.set_enabled(account.id, False)
                except Exception:
                    account.enabled = False
                    account.last_error = exc.message
                last_error = exc
            except UpstreamChallengeError as exc:
                if emitted:
                    raise
                try:
                    await self.pool.set_enabled(account.id, False)
                except Exception:
                    account.enabled = False
                    account.last_error = exc.message
                last_error = exc
            except UpstreamRateLimitError as exc:
                if emitted:
                    raise
                self.pool.mark_cooldown(account, exc.message)
                last_error = exc
            except (UpstreamError, OSError) as exc:
                if emitted:
                    raise
                self.pool.mark_cooldown(account, str(exc))
                last_error = exc
            finally:
                await self.pool.release(account)
        if last_error:
            raise last_error
        raise NoAvailableAccountError("No usable Doubao account.")

    async def _fragments(
        self, account: Account, request: ChatCompletionRequest
    ) -> AsyncIterator[tuple[str, str]]:
        async with account.lock:
            workspace = self._workspace(account)
            payload = upstream_payload(
                request,
                model_key=self.model_key,
                workspace=workspace,
                device_name=account.device_name or socket.gethostname().upper(),
                app_id=self.app_id,
                device_id=account.device_id,
            )
            thinking = False
            async for event in self.transport.stream(account, payload):
                for _ in phase_markers(event):
                    thinking = not thinking
                if (
                    event.get("_event") == "CHUNK_DELTA"
                    and isinstance(event.get("text"), str)
                ):
                    event["_thinking"] = thinking
                for kind, text in extract_text_fragments(event):
                    if text:
                        yield kind, text

    def _workspace(self, account: Account) -> Path:
        safe_id = "".join(
            character
            for character in account.id
            if character.isalnum() or character in {"-", "_"}
        )[:40] or "account"
        workspace = (
            self.workspace_root
            / time.strftime("%Y-%m-%d")
            / f"{safe_id}-{uuid.uuid4().hex[:8]}"
        )
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _start_log(
        self, request: ChatCompletionRequest
    ) -> RequestLogEntry | None:
        if self.request_log is None:
            return None
        return self.request_log.start(
            model=request.normalized_model,
            stream=request.stream,
            meta={"messages": len(request.messages)},
        )

    def _log_success(self, entry: RequestLogEntry | None) -> None:
        if self.request_log is not None and entry is not None:
            self.request_log.success(entry)

    def _log_failure(
        self, entry: RequestLogEntry | None, error: str
    ) -> None:
        if self.request_log is not None and entry is not None:
            self.request_log.failure(entry, error)

    def _log_interrupted(self, entry: RequestLogEntry | None) -> None:
        if (
            self.request_log is not None
            and entry is not None
            and entry.status == "running"
        ):
            self.request_log.failure(entry, "request ended before completion")
