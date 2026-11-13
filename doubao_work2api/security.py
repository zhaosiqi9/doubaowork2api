"""Local security state; callers enforce model/account catalog permissions.

API key secrets are returned only by create_key(). A concurrency limit of zero
means unlimited; empty model/account lists grant access to every entry.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import secrets
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


_PASSWORD_ITERATIONS = 600_000
_PUBLIC_KEY_FIELDS = (
    "id", "name", "enabled", "models", "accounts", "concurrency_limit",
    "created_at", "last_used_at",
)
_KEY_FIELDS = frozenset({"name", "enabled", "models", "accounts", "concurrency_limit"})
_RUNTIME_FIELDS = frozenset({"default_model", "timeout_seconds", "cooldown_seconds"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${_PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    if not isinstance(password, str):
        return False
    algorithm, iterations, salt, expected = encoded.split("$")
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
    )
    return hmac.compare_digest(digest, bytes.fromhex(expected))


def _secret_hash(secret: str) -> str:
    # Generated API keys contain 256 bits of entropy; no slow password KDF needed.
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _document(value: Any, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("document must be an object")
    if set(value) - fields:
        raise ValueError("document contains unknown fields")
    return copy.deepcopy(value)


def _text(value: Any, field: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a nonempty string of at most {maximum} characters")
    value = value.strip()
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


def _runtime_patch(document: Any) -> dict[str, Any]:
    result = _document(document, _RUNTIME_FIELDS)
    if "default_model" in result:
        result["default_model"] = _text(result["default_model"], "default_model")
    if "timeout_seconds" in result:
        value = result["timeout_seconds"]
        if (
            type(value) not in (int, float)
            or not 0 < value <= 3600
            or not math.isfinite(value)
        ):
            raise ValueError("timeout_seconds must be a finite number greater than 0 and at most 3600")
    if "cooldown_seconds" in result:
        value = result["cooldown_seconds"]
        if type(value) is not int or not 0 <= value <= 86400:
            raise ValueError("cooldown_seconds must be an integer between 0 and 86400")
    return result


def _key_patch(document: Any) -> dict[str, Any]:
    result = _document(document, _KEY_FIELDS)
    if "name" in result:
        result["name"] = _text(result["name"], "name", 128)
    if "enabled" in result and type(result["enabled"]) is not bool:
        raise ValueError("enabled must be a boolean")
    if "concurrency_limit" in result:
        value = result["concurrency_limit"]
        if type(value) is not int or not 0 <= value <= 1000:
            raise ValueError("concurrency_limit must be an integer between 0 and 1000")
    for field in ("models", "accounts"):
        if field not in result:
            continue
        values = result[field]
        if not isinstance(values, list) or len(values) > 1000:
            raise ValueError(f"{field} must be a list of at most 1000 strings")
        result[field] = list(dict.fromkeys(_text(value, field) for value in values))
    return result


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {field: copy.deepcopy(record[field]) for field in _PUBLIC_KEY_FIELDS}


class SecurityStore:
    SESSION_TTL_SECONDS = 12 * 60 * 60
    LAST_USED_WRITE_INTERVAL_SECONDS = 60.0

    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.data_dir) / "settings.json"
        self._lock = threading.RLock()
        self._sessions: dict[str, float] = {}
        self._last_used_write_at: dict[str, float] = {}
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self.path.parent.chmod(0o700)
        with self._lock:
            if self.path.exists():
                if os.name != "nt":
                    self.path.chmod(0o600)
                # Never silently reset credentials or revive deleted keys on bad JSON.
                try:
                    state = json.loads(self.path.read_text(encoding="utf-8"))
                    self._validate_state(state)
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                    raise ValueError("invalid security settings file") from None
                self._state = state
            else:
                state = {
                    "version": 1,
                    "runtime": _runtime_patch({
                        "default_model": settings.default_model,
                        "timeout_seconds": settings.request_timeout_seconds,
                        "cooldown_seconds": settings.cooldown_seconds,
                    }),
                    "admin_password_hash": _password_hash(settings.admin_password),
                    "keys": [],
                }
                if settings.api_key:
                    state["keys"].append(self._new_record("默认 Key", settings.api_key))
                self._save(state)
                self._state = state

    @staticmethod
    def _validate_state(state: Any) -> None:
        if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
            raise ValueError("unsupported settings format")
        if set(state) != {"version", "runtime", "admin_password_hash", "keys"}:
            raise ValueError("invalid settings fields")
        runtime = _runtime_patch(state["runtime"])
        if set(runtime) != _RUNTIME_FIELDS or runtime != state["runtime"]:
            raise ValueError("invalid runtime settings")
        algorithm, iterations, salt, digest = state["admin_password_hash"].split("$")
        if (
            algorithm != "pbkdf2_sha256"
            or iterations != str(_PASSWORD_ITERATIONS)
            or len(bytes.fromhex(salt)) != 16
            or len(bytes.fromhex(digest)) != 32
        ):
            raise ValueError("invalid password hash")
        if not isinstance(state["keys"], list):
            raise ValueError("invalid keys")
        seen: set[str] = set()
        for record in state["keys"]:
            if not isinstance(record, dict) or set(record) != set(_PUBLIC_KEY_FIELDS) | {"secret_hash"}:
                raise ValueError("invalid key fields")
            key_id = _text(record["id"], "id")
            if key_id in seen:
                raise ValueError("duplicate key id")
            seen.add(key_id)
            if _key_patch({field: record[field] for field in _KEY_FIELDS}) != {
                field: record[field] for field in _KEY_FIELDS
            }:
                raise ValueError("invalid key metadata")
            if len(bytes.fromhex(record["secret_hash"])) != 32:
                raise ValueError("invalid key hash")
            datetime.fromisoformat(record["created_at"])
            if record["last_used_at"] is not None:
                datetime.fromisoformat(record["last_used_at"])

    def _save(self, state: dict[str, Any]) -> None:
        """Write a complete private file, then replace the old file atomically.

        Called under _lock. Callers publish in-memory changes only after success.
        """
        payload = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".settings-", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # mkstemp creates mode 0600 on POSIX, preserved by replace().
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _new_record(name: str, secret: str) -> dict[str, Any]:
        return {
            "id": secrets.token_hex(16),
            "name": name,
            "enabled": True,
            "models": [],
            "accounts": [],
            "concurrency_limit": 0,
            "created_at": _timestamp(),
            "last_used_at": None,
            "secret_hash": _secret_hash(secret),
        }

    def _mark_last_used_persisted(self, state: dict[str, Any]) -> None:
        """Record when each live last_used_at was included in a full save."""
        now = time.monotonic()
        active_ids = {record["id"] for record in state["keys"]}
        self._last_used_write_at = {
            key_id: stamp
            for key_id, stamp in self._last_used_write_at.items()
            if key_id in active_ids
        }
        for record in state["keys"]:
            key_id = record["id"]
            if record["last_used_at"] is None:
                self._last_used_write_at.pop(key_id, None)
            else:
                self._last_used_write_at[key_id] = now

    def _commit(self, state: dict[str, Any]) -> None:
        self._save(state)
        self._state = state
        self._mark_last_used_persisted(state)

    @property
    def runtime(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state["runtime"])

    def update_runtime(self, document: dict[str, Any]) -> dict[str, Any]:
        patch = _runtime_patch(document)
        with self._lock:
            state = copy.deepcopy(self._state)
            state["runtime"].update(patch)
            self._commit(state)
            return copy.deepcopy(state["runtime"])

    def verify_admin(self, password: str) -> bool:
        with self._lock:
            return _verify_password(password, self._state["admin_password_hash"])

    def _expire_sessions(self) -> None:
        now = time.monotonic()
        self._sessions = {
            token: deadline for token, deadline in self._sessions.items()
            if deadline > now
        }

    def create_session(self) -> str:
        """Issue a session after the caller has verified the admin password."""
        with self._lock:
            self._expire_sessions()
            token = secrets.token_urlsafe(32)
            self._sessions[_secret_hash(token)] = time.monotonic() + self.SESSION_TTL_SECONDS
            return token

    def session_valid(self, token: str) -> bool:
        if not isinstance(token, str) or not token:
            return False
        with self._lock:
            self._expire_sessions()
            return _secret_hash(token) in self._sessions

    def delete_session(self, token: str) -> None:
        if isinstance(token, str):
            with self._lock:
                self._sessions.pop(_secret_hash(token), None)

    def set_admin_password(self, current: str, new: str) -> None:
        with self._lock:
            if not self.verify_admin(current):
                raise ValueError("current password is incorrect")
            if not isinstance(new, str) or len(new) < 8:
                raise ValueError("new password must contain at least 8 characters")
            state = copy.deepcopy(self._state)
            state["admin_password_hash"] = _password_hash(new)
            self._commit(state)
            self._sessions.clear()

    def list_keys(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_public(record) for record in self._state["keys"]]

    def create_key(self, document: dict[str, Any]) -> dict[str, Any]:
        patch = _key_patch(document)
        if "name" not in patch:
            raise ValueError("name is required")
        with self._lock:
            secret = "sk-" + secrets.token_urlsafe(32)
            record = self._new_record(patch["name"], secret)
            record.update(patch)
            state = copy.deepcopy(self._state)
            state["keys"].append(record)
            self._commit(state)
            return {**_public(record), "key": secret}

    def update_key(self, key_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        patch = _key_patch(patch)
        with self._lock:
            state = copy.deepcopy(self._state)
            for record in state["keys"]:
                if record["id"] == key_id:
                    record.update(patch)
                    self._commit(state)
                    return _public(record)
            raise KeyError(key_id)

    def delete_key(self, key_id: str) -> None:
        with self._lock:
            state = copy.deepcopy(self._state)
            remaining = [record for record in state["keys"] if record["id"] != key_id]
            if len(remaining) == len(state["keys"]):
                raise KeyError(key_id)
            state["keys"] = remaining
            self._commit(state)

    def authenticate_api(self, secret: str) -> dict[str, Any] | None:
        if not isinstance(secret, str) or not secret:
            return None
        digest = _secret_hash(secret)
        with self._lock:
            for index, record in enumerate(self._state["keys"]):
                if hmac.compare_digest(digest, record["secret_hash"]) and record["enabled"]:
                    now = time.monotonic()
                    last_used = _timestamp()
                    key_id = record["id"]
                    last_write = self._last_used_write_at.get(key_id)
                    if (
                        last_write is None
                        or now - last_write >= self.LAST_USED_WRITE_INTERVAL_SECONDS
                    ):
                        state = copy.deepcopy(self._state)
                        updated = state["keys"][index]
                        updated["last_used_at"] = last_used
                        self._commit(state)
                        return _public(updated)
                    record["last_used_at"] = last_used
                    return _public(record)
            return None
