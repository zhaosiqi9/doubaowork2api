from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import NoAvailableAccountError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Account:
    id: str
    cookies: dict[str, str]
    device_id: str
    web_id: str
    device_name: str = "DOUBAO-LINUX"
    enabled: bool = True
    cooldown_until: float = 0.0
    last_error: str = ""
    in_flight: int = 0
    source_path: Path | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def available(self) -> bool:
        return self.enabled and self.cooldown_until <= time.time()

    def cooldown(self, seconds: int, reason: str) -> None:
        self.cooldown_until = max(self.cooldown_until, time.time() + seconds)
        self.last_error = reason

    def clear_error(self) -> None:
        self.cooldown_until = 0.0
        self.last_error = ""

    def public_dict(self) -> dict[str, Any]:
        remaining = max(0, int(self.cooldown_until - time.time()))
        if not self.enabled:
            state = "disabled"
        elif remaining:
            state = "cooling"
        elif self.in_flight:
            state = "busy"
        else:
            state = "ready"
        return {
            "id": self.id,
            "enabled": self.enabled,
            "available": self.available,
            "state": state,
            "cooldown_remaining_seconds": remaining,
            "last_error": self.last_error,
            "device_id": self.device_id[:6] + "..." if self.device_id else "",
            "web_id": self.web_id[:6] + "..." if self.web_id else "",
            "in_flight": self.in_flight,
        }

    def to_document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "web_id": self.web_id,
            "device_name": self.device_name,
            "enabled": self.enabled,
            "cookies": dict(self.cookies),
        }


def _cookie_dict(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            str(key).strip(): str(item).strip()
            for key, item in value.items()
            if str(key).strip()
        }
    if isinstance(value, str):
        result: dict[str, str] = {}
        for part in value.split(";"):
            key, separator, item = part.partition("=")
            if separator and key.strip():
                result[key.strip()] = item.strip()
        return result
    if isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            cookie_value = item.get("value")
            if name and cookie_value is not None:
                result[name] = str(cookie_value)
        return result
    return {}


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def parse_account_document(document: Any) -> Account:
    if not isinstance(document, dict):
        raise ValueError("account file must contain a JSON object")
    source = document.get("account")
    if isinstance(source, dict):
        document = {**document, **source}
    cookies = _cookie_dict(
        document.get("cookies")
        or document.get("cookie")
        or document.get("cookie_string")
    )
    device_id = str(
        document.get("device_id")
        or document.get("tea_uuid")
        or document.get("user_unique_id")
        or ""
    ).strip()
    web_id = str(document.get("web_id") or "").strip()
    identity = document.get("identity")
    if isinstance(identity, dict):
        device_id = device_id or str(
            identity.get("device_id")
            or identity.get("user_unique_id")
            or ""
        ).strip()
        web_id = web_id or str(identity.get("web_id") or "").strip()
    account_id = str(
        document.get("id")
        or document.get("account_id")
        or document.get("name")
        or device_id
        or ""
    ).strip()
    if not cookies:
        raise ValueError("account has no cookies")
    if not account_id:
        digest = hashlib.sha256(
            json.dumps(cookies, sort_keys=True).encode("utf-8")
        ).hexdigest()
        account_id = f"account-{digest[:8]}"
    return Account(
        id=account_id,
        cookies=cookies,
        device_id=device_id,
        web_id=web_id,
        device_name=str(
            document.get("device_name") or "DOUBAO-LINUX"
        ).strip(),
        enabled=_as_bool(document.get("enabled"), True),
    )


class AccountPool:
    def __init__(
        self,
        accounts_dir: Path,
        *,
        cooldown_seconds: int = 60,
    ) -> None:
        self.accounts_dir = accounts_dir
        self.cooldown_seconds = cooldown_seconds
        self._accounts: list[Account] = []
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def accounts(self) -> list[Account]:
        return list(self._accounts)

    def load(self) -> int:
        self.accounts_dir.mkdir(parents=True, exist_ok=True)
        loaded: list[Account] = []
        seen: set[str] = set()
        for path in sorted(self.accounts_dir.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                account = parse_account_document(document)
                account.source_path = path
            except Exception as exc:
                logger.warning("Skipping account file %s: %s", path.name, exc)
                continue
            if account.id in seen:
                logger.warning("Skipping duplicate account id %s", account.id)
                continue
            seen.add(account.id)
            loaded.append(account)
        self._accounts = loaded
        self._index = 0
        logger.info("Loaded %d Doubao account(s)", len(loaded))
        return len(loaded)

    def find(self, account_id: str) -> Account | None:
        return next(
            (account for account in self._accounts if account.id == account_id),
            None,
        )

    async def acquire(self) -> Account:
        async with self._lock:
            if not self._accounts:
                raise NoAvailableAccountError(
                    "No Doubao accounts are configured. Put account JSON files "
                    "under the accounts directory."
                )
            candidates = [
                account
                for account in self._accounts
                if account.available and account.in_flight == 0
            ]
            if not candidates:
                enabled = [account for account in self._accounts if account.enabled]
                if not enabled:
                    raise NoAvailableAccountError(
                        "All configured Doubao accounts are disabled."
                    )
                wait = min(
                    max(0, int(account.cooldown_until - time.time()))
                    for account in enabled
                )
                raise NoAvailableAccountError(
                    f"All Doubao accounts are busy or cooling down; "
                    f"retry in about {wait}s."
                )
            for offset in range(len(self._accounts)):
                index = (self._index + offset) % len(self._accounts)
                account = self._accounts[index]
                if account in candidates:
                    self._index = (index + 1) % len(self._accounts)
                    account.in_flight += 1
                    return account
        raise NoAvailableAccountError("Could not acquire a Doubao account.")

    async def release(self, account: Account) -> None:
        async with self._lock:
            account.in_flight = max(0, account.in_flight - 1)

    def mark_cooldown(self, account: Account, reason: str) -> None:
        account.cooldown(self.cooldown_seconds, reason)

    def mark_success(self, account: Account) -> None:
        account.clear_error()

    async def upsert(self, document: Any) -> Account:
        incoming = parse_account_document(document)
        async with self._lock:
            existing = self.find(incoming.id)
            if existing is None:
                existing = incoming
                existing.source_path = self._account_path(existing.id)
                self._accounts.append(existing)
            else:
                existing.cookies = incoming.cookies
                existing.device_id = incoming.device_id
                existing.web_id = incoming.web_id
                existing.device_name = incoming.device_name
                existing.enabled = incoming.enabled
                existing.cooldown_until = 0.0
                existing.last_error = ""
            if existing.source_path is None:
                existing.source_path = self._account_path(existing.id)
            self._write(existing)
            return existing

    async def set_enabled(self, account_id: str, enabled: bool) -> Account:
        async with self._lock:
            account = self.find(account_id)
            if account is None:
                raise KeyError(account_id)
            previous = account.enabled
            account.enabled = enabled
            if not enabled:
                account.cooldown_until = 0.0
            try:
                if account.source_path is None:
                    account.source_path = self._account_path(account.id)
                self._write(account)
            except Exception:
                account.enabled = previous
                raise
            return account

    async def clear_cooldown(self, account_id: str) -> Account:
        async with self._lock:
            account = self.find(account_id)
            if account is None:
                raise KeyError(account_id)
            account.cooldown_until = 0.0
            account.last_error = ""
            return account

    def snapshot(self) -> list[dict[str, Any]]:
        return [account.public_dict() for account in self._accounts]

    def _account_path(self, account_id: str) -> Path:
        safe = "".join(
            character
            for character in account_id
            if character.isalnum() or character in {"-", "_"}
        )[:60]
        if not safe:
            safe = "account-" + hashlib.sha256(
                account_id.encode("utf-8")
            ).hexdigest()[:8]
        return self.accounts_dir / f"{safe}.json"

    def _write(self, account: Account) -> None:
        assert account.source_path is not None
        path = account.source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            account.to_document(),
            ensure_ascii=False,
            indent=2,
        )
        try:
            path.write_text(payload, encoding="utf-8")
        except PermissionError:
            # Windows sandbox can reject replace() across some mapped paths.
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
