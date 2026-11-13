from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import httpx

from .accounts import Account
from .config import Settings
from .errors import (
    UpstreamAuthError,
    UpstreamChallengeError,
    UpstreamError,
    UpstreamRateLimitError,
)

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36 SamanthaDoubaoWork/2.36.11"
)


def classify_status(status: int, body: str) -> None:
    if status == 401:
        raise UpstreamAuthError(
            "Doubao login is invalid or expired; import a fresh account.",
            code="401",
        )
    if status in {403, 412}:
        raise UpstreamChallengeError(
            f"Doubao rejected the request (HTTP {status}); the account may "
            "require manual verification.",
            code=str(status),
        )
    if status == 429:
        raise UpstreamRateLimitError(
            "Doubao rate limit reached.", code="429"
        )
    if status >= 500:
        raise UpstreamError(
            f"Doubao upstream returned HTTP {status}.", code=str(status)
        )
    if status >= 400:
        detail = body[:500].strip() or "no response body"
        raise UpstreamError(
            f"Doubao upstream returned HTTP {status}: {detail}",
            code=str(status),
        )


def classify_body_error(payload: dict[str, Any]) -> None:
    code = str(payload.get("code") or payload.get("status_code") or "")
    if code in {"0", "200", ""}:
        return
    if code in {"710012001"}:
        raise UpstreamAuthError(
            "Doubao session was revoked; import a fresh account.", code=code
        )
    if code in {"710022002", "710022004", "712010702"}:
        raise UpstreamChallengeError(
            f"Doubao requires account verification ({code}).", code=code
        )
    if code in {"710022005", "710022006"}:
        raise UpstreamRateLimitError(
            f"Doubao account is rate limited ({code}).", code=code
        )
    message = payload.get("message") or payload.get("msg") or "upstream error"
    raise UpstreamError(f"Doubao error {code}: {message}", code=code)


class DirectTransport:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Never let httpx log full URLs containing device_id / web_id.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self._clients: dict[str, tuple[str, httpx.AsyncClient]] = {}
        self._clients_lock = asyncio.Lock()

    def _create_client(self, account: Account) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.settings.request_timeout_seconds, connect=20
            ),
            follow_redirects=True,
            cookies=account.cookies,
        )

    async def _client_for(self, account: Account) -> httpx.AsyncClient:
        # Hash the imported credentials, not the client's cookie jar: responses
        # may refresh cookies, while an import must discard the old jar/pool.
        revision = hashlib.sha256(json.dumps(
            [account.cookies, account.device_id, account.web_id, account.device_name,
             repr(self.settings)],
            sort_keys=True,
        ).encode("utf-8")).hexdigest()
        async with self._clients_lock:
            cached = self._clients.get(account.id)
            if cached is not None:
                previous_revision, client = cached
                if previous_revision == revision and not client.is_closed:
                    return client
                await client.aclose()
                del self._clients[account.id]
            client = self._create_client(account)
            self._clients[account.id] = (revision, client)
            return client

    async def configure(self, settings: Settings) -> None:
        async with self._clients_lock:
            await self._close_clients()
            self.settings = settings

    async def forget_accounts(self, account_ids: list[str]) -> None:
        async with self._clients_lock:
            for account_id in account_ids:
                cached = self._clients.pop(account_id, None)
                if cached is not None:
                    await cached[1].aclose()

    async def _close_clients(self) -> None:
        # Attempt every close even if one client fails. On failure, leave the
        # cache available for retry and do not apply a new configuration.
        results = await asyncio.gather(
            *(client.aclose() for _, client in self._clients.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        self._clients.clear()

    def _query(self, account: Account) -> str:
        device_id = account.device_id
        params = {
            "aid": self.settings.app_id,
            "real_aid": self.settings.app_id,
            "device_id": device_id,
            "tea_uuid": device_id,
            "web_id": account.web_id,
            "device_platform": "web",
            "doubao_device_platform": "desktop",
            "language": "zh",
            "region": "CN",
            "sys_region": "CN",
            "pkg_type": "release_version",
            "version_code": "20800",
            "pc_version": self.settings.pc_version,
            "doubao_pc_version": self.settings.pc_version,
            "runtime": "web",
            "runtime_version": self.settings.runtime_version,
            "samantha_web": "1",
            "web_platform": "desktop",
            "use-olympus-account": "1",
            "client_platform": "pc_client",
            "chromium_version": "147.0.7727.149",
            "channel": "win",
            "fp": f"verify_{device_id}",
            "web_tab_id": str(uuid.uuid4()),
            "tz_name": "Asia/Shanghai",
        }
        return urlencode(sorted(params.items()))

    async def stream(
        self, account: Account, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        client = await self._client_for(account)
        csrf = (
            account.cookies.get("passport_csrf_token")
            or account.cookies.get("passport_csrf_token_default")
            or ""
        )
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Origin": self.settings.upstream_base,
            "Referer": f"{self.settings.upstream_base}/chat/",
            "User-Agent": USER_AGENT,
            "agw-js-conv": "str",
            "last-event-id": "undefined",
            "x-lgw-req-sdk-type": "3",
            "x-net-sdk-domain-dispatch": "1",
            "x-request-id": str(uuid.uuid4()),
            "x-ttnet-bypass-decompression": "1",
            "x-flow-trace": "04-" + uuid.uuid4().hex,
        }
        if csrf:
            headers["x-tt-passport-csrf-token"] = csrf
        url = (
            f"{self.settings.upstream_base}/chat/completion?"
            + self._query(account)
        )
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "replace")
                classify_status(response.status_code, body)
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                body = (await response.aread()).decode("utf-8", "replace")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    classify_body_error(parsed)
                raise UpstreamError(
                    "Doubao returned a non-SSE response "
                    f"({content_type or 'unknown'})."
                )
            event_name = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw in {"[DONE]", "{}"}:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                parsed["_event"] = event_name
                classify_body_error(parsed)
                yield parsed

    async def close(self) -> None:
        async with self._clients_lock:
            await self._close_clients()
