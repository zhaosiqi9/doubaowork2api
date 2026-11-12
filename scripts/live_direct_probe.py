from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

from doubao_work2api.accounts import Account
from doubao_work2api.config import Settings
from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.protocol import (
    extract_text_fragments,
    phase_markers,
    upstream_payload,
)
from doubao_work2api.transport import DirectTransport


CDP_URL = "http://127.0.0.1:9222"


async def read_identity() -> tuple[dict[str, str], str, str]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0]
        page = next(
            (
                item
                for item in context.pages
                if "doubaowork-chat" in item.url
            ),
            context.pages[0],
        )
        identity = await page.evaluate(
            """() => {
                const tea = JSON.parse(
                    localStorage.getItem('__tea_cache_tokens_1044603') || '{}'
                );
                return {
                    device_id: tea.user_unique_id || '',
                    web_id: tea.web_id || ''
                };
            }"""
        )
        cookies = {
            item["name"]: item["value"]
            for item in await context.cookies("https://www.doubao.com")
        }
        return cookies, str(identity.get("device_id") or ""), str(
            identity.get("web_id") or ""
        )


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    prompt = "只回复 OK"
    cookies, device_id, web_id = await read_identity()
    if not cookies or not device_id:
        raise SystemExit("DoubaoWork is not logged in or CDP identity is missing.")
    settings = Settings(
        request_timeout_seconds=180,
        data_dir=Path("data"),
        accounts_dir=Path("accounts"),
    )
    account = Account(
        id="live-probe",
        cookies=cookies,
        device_id=device_id,
        web_id=web_id,
    )
    request = ChatCompletionRequest(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": prompt}],
    )
    workspace = Path("data/probe")
    workspace.mkdir(parents=True, exist_ok=True)
    payload = upstream_payload(
        request,
        model_key=settings.model_key,
        workspace=workspace,
        device_name="LINUX-PROBE",
        app_id=settings.app_id,
        device_id=device_id,
    )
    transport = DirectTransport(settings)
    event_counts: dict[str, int] = {}
    content: list[str] = []
    reasoning: list[str] = []
    thinking = False
    try:
        async for event in transport.stream(account, payload):
            name = str(event.get("_event") or "unknown")
            event_counts[name] = event_counts.get(name, 0) + 1
            for _ in phase_markers(event):
                thinking = not thinking
            if event.get("_event") == "CHUNK_DELTA" and isinstance(
                event.get("text"), str
            ):
                event["_thinking"] = thinking
            for kind, text in extract_text_fragments(event):
                if kind == "content":
                    content.append(text)
                else:
                    reasoning.append(text)
    finally:
        await transport.close()
    print(
        json.dumps(
            {
                "events": event_counts,
                "content": "".join(content),
                "reasoning_chars": len("".join(reasoning)),
                "device_id_prefix": device_id[:6] + "...",
                "web_id_prefix": web_id[:6] + "...",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
