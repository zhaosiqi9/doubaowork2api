from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .accounts import Account, AccountPool
from .offline import OfflineScanError, scan_offline_account


class DesktopScanError(RuntimeError):
    pass


@dataclass(slots=True)
class DesktopScanResult:
    account: Account
    created: bool
    cookie_count: int
    source: str = "offline"


def _account_id(cookies: dict[str, str]) -> str:
    digest = hashlib.sha256(
        json.dumps(cookies, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"desktop-{digest}"


async def scan_desktop_account(
    pool: AccountPool,
) -> DesktopScanResult:
    try:
        identity = await asyncio.to_thread(scan_offline_account)
    except (OfflineScanError, OSError) as exc:
        raise DesktopScanError(str(exc)) from exc
    cookies = identity.cookies
    device_id = identity.device_id
    web_id = identity.web_id
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
        source="offline",
    )
