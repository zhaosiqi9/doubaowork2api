from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .accounts import Account, AccountPool


DEFAULT_CDP_URL = "http://127.0.0.1:9222"


class DesktopScanError(RuntimeError):
    pass


@dataclass(slots=True)
class DesktopScanResult:
    account: Account
    created: bool
    cookie_count: int


def _validate_cdp_url(cdp_url: str) -> str:
    parsed = urlparse(cdp_url)
    if parsed.scheme not in {"http", "https"}:
        raise DesktopScanError("CDP 地址必须是 http(s) URL。")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise DesktopScanError("只允许扫描本机 CDP（127.0.0.1/localhost）。")
    if not parsed.port:
        raise DesktopScanError("CDP 地址缺少端口。")
    return cdp_url.rstrip("/")


def _cdp_available(cdp_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{cdp_url}/json/version",
            timeout=2,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("webSocketDebuggerUrl"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _account_id(cookies: dict[str, str]) -> str:
    digest = hashlib.sha256(
        json.dumps(cookies, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"desktop-{digest}"


async def _read_cdp_identity(cdp_url: str) -> tuple[dict[str, str], str, str]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise DesktopScanError(
            "缺少 Playwright，无法读取桌面客户端。请安装项目开发依赖后再试。"
        ) from exc

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise DesktopScanError(
                "无法连接桌面客户端调试端口。请确认 DoubaoWork 已登录，"
                "并使用 --remote-debugging-port=9222 启动。"
            ) from exc
        context = browser.contexts[0]
        pages = context.pages
        page = next(
            (
                item
                for item in pages
                if "doubaowork-chat" in item.url
            ),
            pages[0] if pages else None,
        )
        if page is None:
            raise DesktopScanError("桌面客户端没有可读取的页面。")
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
            if item.get("name") and item.get("value") is not None
        }

    device_id = str(identity.get("device_id") or "").strip()
    web_id = str(identity.get("web_id") or "").strip()
    if not cookies:
        raise DesktopScanError("没有读取到豆包 Cookie，请先在客户端完成登录。")
    if not device_id:
        raise DesktopScanError(
            "没有读取到 device_id，请确认客户端已登录并打开聊天页面。"
        )
    return cookies, device_id, web_id


async def scan_desktop_account(
    pool: AccountPool,
    *,
    cdp_url: str = DEFAULT_CDP_URL,
) -> DesktopScanResult:
    cdp_url = _validate_cdp_url(cdp_url)
    if not _cdp_available(cdp_url):
        raise DesktopScanError(
            "未检测到本机调试端口 9222。请先关闭 DoubaoWork，再使用 "
            "--remote-debugging-port=9222 启动并登录客户端。"
        )
    cookies, device_id, web_id = await _read_cdp_identity(cdp_url)
    account_id = _account_id(cookies)
    existing = next(
        (
            account
            for account in pool.accounts
            if account.device_id == device_id
            and account.cookies.get("sessionid") == cookies.get("sessionid")
        ),
        None,
    )
    document: dict[str, Any] = {
        "id": existing.id if existing else account_id,
        "device_id": device_id,
        "web_id": web_id,
        "device_name": "DOUBAO-LINUX",
        "cookies": cookies,
    }
    account = await pool.upsert(document)
    return DesktopScanResult(
        account=account,
        created=existing is None,
        cookie_count=len(cookies),
    )
