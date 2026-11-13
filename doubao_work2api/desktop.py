from __future__ import annotations

import asyncio
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
    previous_ids = {account.id for account in pool.accounts}
    document: dict[str, Any] = {
        "device_id": device_id,
        "web_id": web_id,
        "cookies": cookies,
    }
    account = await pool.upsert(document)
    return DesktopScanResult(
        account=account,
        created=account.id not in previous_ids,
        cookie_count=len(cookies),
        source="offline",
    )
