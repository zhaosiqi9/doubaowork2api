from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def export_account(cdp_url: str, output: Path, name: str) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(cdp_url)
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
        cookies = await context.cookies("https://www.doubao.com")
        if not cookies:
            raise SystemExit("No Doubao cookies found; sign in first.")
        device_id = str(identity.get("device_id") or "")
        web_id = str(identity.get("web_id") or "")
        if not device_id:
            raise SystemExit(
                "Missing localStorage __tea_cache_tokens_1044603.user_unique_id."
            )
        document = {
            "id": name,
            "device_id": device_id,
            "web_id": web_id,
            "device_name": "DOUBAO-LINUX",
            "cookies": {
                str(item["name"]): str(item["value"])
                for item in cookies
                if item.get("name")
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if os.name != "nt":
            output.chmod(0o600)
        print(
            json.dumps(
                {
                    "written": str(output),
                    "cookie_count": len(document["cookies"]),
                    "device_id_prefix": device_id[:6] + "...",
                    "web_id_prefix": web_id[:6] + "...",
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdp", default="http://127.0.0.1:9222")
    parser.add_argument("--name", default="account-1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("accounts/account-1.json"),
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(export_account(args.cdp, args.output, args.name))


if __name__ == "__main__":
    main()
