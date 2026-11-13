from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .accounts import Account, AccountPool
from .catalog import resolve_upstream_key
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
    validate_request,
)
from .transport import DirectTransport
from .tool_bridge import parse_tool_response
from .image_inputs import load_image, ImageInput
from .image_upload import upload_image, ImageUploadError


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

    async def complete(
        self, request: ChatCompletionRequest, *,
        allowed_ids: set[str] | None = None, key_id: str = "",
    ) -> dict[str, Any]:
        validate_request(request)
        log_entry = self._start_log(request, key_id)
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        try:
            async for kind, text in self._converted_fragments(
                request, log_entry, allowed_ids
            ):
                if kind == "content":
                    content.append(text)
                elif kind == "tool_calls":
                    tool_calls.extend(text)
                else:
                    reasoning.append(text)
            answer = "".join(content)
            if not answer and not tool_calls:
                raise UpstreamError("Doubao returned an empty response.")
            prompt_text = json.dumps({
                "messages": [
                    {**message.model_dump(exclude_none=True), "content": [
                        {"type": "image_url"} if part.get("type") == "image_url" else part
                        for part in message.content
                    ]} if isinstance(message.content, list) else message.model_dump(exclude_none=True)
                    for message in request.messages
                ],
                "tools": request.tools,
            }, ensure_ascii=False)
            output_text = answer + "".join(reasoning)
            if tool_calls:
                output_text += json.dumps(tool_calls, ensure_ascii=False)
            result = completion_payload(
                request_id=completion_id(),
                model=request.normalized_model,
                content=answer,
                reasoning="".join(reasoning),
                prompt_tokens=max(1, len(prompt_text) // 3),
                completion_tokens=max(1, len(output_text) // 3),
            )
            if tool_calls:
                result["choices"][0]["message"]["content"] = answer or None
                result["choices"][0]["message"]["tool_calls"] = tool_calls
                result["choices"][0]["finish_reason"] = "tool_calls"
            self._log_success(log_entry)
            return result
        except Exception as exc:
            self._log_failure(log_entry, str(exc))
            raise
        finally:
            self._log_interrupted(log_entry)

    async def stream(
        self, request: ChatCompletionRequest, *,
        allowed_ids: set[str] | None = None, key_id: str = "",
    ) -> AsyncIterator[str]:
        validate_request(request)
        log_entry = self._start_log(request, key_id)
        request_id = completion_id()
        has_tools = False
        try:
            yield sse_data(
                chunk_payload(
                    request_id=request_id,
                    model=request.normalized_model,
                    role="assistant",
                )
            )
            async for kind, text in self._converted_fragments(
                request, log_entry, allowed_ids
            ):
                if kind == "tool_calls":
                    has_tools = True
                    chunk = chunk_payload(request_id=request_id, model=request.normalized_model)
                    chunk["choices"][0]["delta"] = {
                        "tool_calls": [dict(call, index=index) for index, call in enumerate(text)]
                    }
                    yield sse_data(chunk)
                    continue
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
                    finish_reason="tool_calls" if has_tools else "stop",
                )
            )
            yield "data: [DONE]\n\n"
            self._log_success(log_entry)
        except Exception as exc:
            self._log_failure(log_entry, str(exc))
            raise
        finally:
            self._log_interrupted(log_entry)

    async def _converted_fragments(
        self,
        request: ChatCompletionRequest,
        log_entry: RequestLogEntry | None,
        allowed_ids: set[str] | None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Use one response conversion path for JSON and SSE clients.

        Tool responses must be complete before validation. Ordinary text remains
        incremental; tool-enabled requests buffer only their answer text.
        """
        tools = getattr(request, "tools", None)
        tool_choice = getattr(request, "tool_choice", None)
        bridge_enabled = bool(tools) and tool_choice != "none"
        buffered: list[str] = []
        has_answer = False
        async for kind, text in self._fragments_with_retry(request, log_entry, allowed_ids):
            if kind == "content" and text:
                has_answer = True
            if bridge_enabled and kind == "content":
                buffered.append(text)
            else:
                yield kind, text
        if bridge_enabled:
            content, calls = parse_tool_response(
                "".join(buffered), tools, tool_choice,
                getattr(request, "parallel_tool_calls", True),
            )
            if content:
                yield "content", content
            if calls:
                yield "tool_calls", calls
            has_answer = bool(content or calls)
        if not has_answer:
            raise UpstreamError("Doubao returned an empty response.", code="empty_response")

    async def _fragments_with_retry(
        self,
        request: ChatCompletionRequest,
        log_entry: RequestLogEntry | None,
        allowed_ids: set[str] | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        sources = [
            part["image_url"]["url"] for message in request.messages if isinstance(message.content, list)
            for part in message.content if part.get("type") == "image_url"
        ]
        if sources and not any(account.available and (allowed_ids is None or account.id in allowed_ids) for account in self.pool.accounts):
            raise NoAvailableAccountError("No usable Doubao account.")
        images: list[ImageInput] = []
        total_bytes = 0
        for source in sources:
            image = await load_image(source)
            total_bytes += len(image.data)
            if total_bytes > 20 * 1024 * 1024:
                from .protocol import UnsupportedContentError
                raise UnsupportedContentError("Total image data exceeds the 20 MiB request limit.")
            images.append(image)
        attempted: set[str] = set()
        last_error: Exception | None = None
        for _ in range(max(1, len(self.pool.accounts))):
            try:
                account = await self.pool.acquire(allowed_ids=allowed_ids)
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
                async for fragment in self._fragments(account, request, images):
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
                account.last_error = exc.message
                account.error_code = exc.code
                account.last_error_at = time.time()
                try:
                    await self.pool.set_enabled(account.id, False)
                except Exception:
                    account.enabled = False
                    account.last_error = exc.message
                last_error = exc
            except UpstreamChallengeError as exc:
                if emitted:
                    raise
                account.last_error = exc.message
                account.error_code = exc.code
                account.last_error_at = time.time()
                try:
                    await self.pool.set_enabled(account.id, False)
                except Exception:
                    account.enabled = False
                    account.last_error = exc.message
                last_error = exc
            except UpstreamRateLimitError as exc:
                if emitted:
                    raise
                self.pool.mark_cooldown(account, exc.message, error_code=exc.code)
                last_error = exc
            except ImageUploadError:
                # A storage upload failure does not invalidate this account and
                # must not fan out the same private images across the pool.
                raise
            except (UpstreamError, OSError) as exc:
                if emitted:
                    raise
                self.pool.mark_cooldown(account, str(exc), error_code=getattr(exc, "code", "network_error"))
                last_error = exc
            finally:
                await self.pool.release(account)
        if last_error:
            raise last_error
        raise NoAvailableAccountError("No usable Doubao account.")

    async def _fragments(
        self, account: Account, request: ChatCompletionRequest,
        images: list[ImageInput] | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        async with account.lock:
            uploaded = [
                await upload_image(self.transport, account, image, filename=f"image-{index}.{image.filename.rsplit('.', 1)[-1]}")
                for index, image in enumerate(images or [], 1)
            ]
            workspace = self._workspace(account)
            payload = upstream_payload(
                request,
                model_key=resolve_upstream_key(request.normalized_model),
                workspace=workspace,
                device_name=account.device_name or socket.gethostname().upper(),
                app_id=self.app_id,
                device_id=account.device_id,
                images=uploaded,
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
        self, request: ChatCompletionRequest, key_id: str = ""
    ) -> RequestLogEntry | None:
        if self.request_log is None:
            return None
        return self.request_log.start(
            model=request.normalized_model,
            stream=request.stream,
            meta={"messages": len(request.messages), "key_id": key_id},
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
