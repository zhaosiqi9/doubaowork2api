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
    ) -> None:
        self.pool = pool
        self.transport = transport
        self.model_key = model_key
        self.app_id = app_id
        self.workspace_root = workspace_root
        self.default_model = default_model

    async def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        content: list[str] = []
        reasoning: list[str] = []
        async for kind, text in self._fragments_with_retry(request):
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
        return completion_payload(
            request_id=completion_id(),
            model=request.normalized_model,
            content=answer,
            reasoning="".join(reasoning),
            prompt_tokens=max(1, len(prompt_text) // 3),
            completion_tokens=max(1, len(answer) // 3),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        request_id = completion_id()
        yield sse_data(
            chunk_payload(
                request_id=request_id,
                model=request.normalized_model,
                role="assistant",
            )
        )
        async for kind, text in self._fragments_with_retry(request):
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

    async def _fragments_with_retry(
        self, request: ChatCompletionRequest
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
                account.enabled = False
                account.last_error = exc.message
                last_error = exc
            except UpstreamChallengeError as exc:
                if emitted:
                    raise
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
