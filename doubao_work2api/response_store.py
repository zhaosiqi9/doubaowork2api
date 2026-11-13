"""Bounded, API-key-isolated response state and authenticated replay envelopes."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from .errors import GatewayError
from .models import ChatMessage


class ResponseNotFound(GatewayError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "response_not_found"


class InvalidReplayToken(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_replay_token"


def response_owner(key: dict[str, Any]) -> str:
    policy = json.dumps({"id": key["id"], "models": sorted(key["models"]), "accounts": sorted(key["accounts"])}, sort_keys=True)
    return hashlib.sha256(policy.encode()).hexdigest()


class ResponseStore:
    """Response bodies remain in memory; only the replay encryption key is saved.

    `store=False` skips this cache. Encrypted compaction/reasoning items are
    independently replayable until their expiry, bound to the issuing key policy.
    """

    def __init__(self, data_dir: Path, *, ttl: float = 3600, max_entries: int = 128, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._records: OrderedDict[str, tuple[float, str, bytes]] = OrderedDict()
        self._total = 0
        self._lock = threading.RLock()
        directory = Path(data_dir)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / "response-replay.key"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(Fernet.generate_key())
                stream.flush()
                os.fsync(stream.fileno())
        if os.name != "nt":
            path.chmod(0o600)
        self._cipher = Fernet(path.read_bytes())

    def _drop(self, response_id: str) -> None:
        record = self._records.pop(response_id)
        self._total -= len(record[2])

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for response_id, record in list(self._records.items()):
            if record[0] < cutoff:
                self._drop(response_id)

    def put(self, owner: str, response: dict[str, Any], history: list[ChatMessage], input_items: list[dict[str, Any]]) -> bool:
        body = json.dumps({
            "response": response,
            "history": [message.model_dump(exclude_none=True) for message in history],
            "input_items": input_items,
        }, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if len(body) > self.max_bytes:
            return False
        response_id = response["id"]
        with self._lock:
            self._expire()
            if response_id in self._records:
                self._drop(response_id)
            while self._records and (len(self._records) >= self.max_entries or self._total + len(body) > self.max_bytes):
                self._drop(next(iter(self._records)))
            self._records[response_id] = (time.time(), owner, body)
            self._total += len(body)
        return True

    def get(self, owner: str, response_id: str) -> dict[str, Any]:
        with self._lock:
            self._expire()
            record = self._records.get(response_id)
            if record is None or record[1] != owner:
                raise ResponseNotFound("Response is unavailable, expired, or belongs to another API Key.")
            self._records.move_to_end(response_id)
            return json.loads(record[2])

    def delete(self, owner: str, response_id: str) -> None:
        with self._lock:
            self.get(owner, response_id)
            self._drop(response_id)

    def find_item(self, owner: str, item_id: str) -> dict[str, Any]:
        with self._lock:
            self._expire()
            for _, record in reversed(self._records.items()):
                if record[1] != owner:
                    continue
                for item in json.loads(record[2])["response"].get("output", []):
                    if item.get("id") == item_id:
                        return item
        raise ResponseNotFound("Referenced response item is unavailable or belongs to another API Key.")

    def encode(self, owner: str, model: str, messages: list[ChatMessage], *, kind: str) -> str:
        payload = {
            "version": 1, "kind": kind, "owner": owner, "model": model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 32 * 1024 * 1024:
            raise InvalidReplayToken("Replay context exceeds the supported size.")
        return self._cipher.encrypt(raw).decode("ascii")

    def decode(self, owner: str, model: str, item: dict[str, Any]) -> list[ChatMessage]:
        token = item.get("encrypted_content")
        kind = item.get("type")
        if not isinstance(token, str) or len(token) > 44 * 1024 * 1024 or kind not in {"compaction", "reasoning"}:
            raise InvalidReplayToken("Unsupported replay envelope.")
        try:
            raw = self._cipher.decrypt(token.encode("ascii"), ttl=24 * 3600)
            value = json.loads(raw)
            if value["version"] != 1 or value["kind"] != kind or value["owner"] != owner or value["model"] != model:
                raise ValueError
            messages = [ChatMessage.model_validate(message) for message in value["messages"]]
            if not messages:
                raise ValueError
            return messages
        except (InvalidToken, UnicodeError, ValueError, TypeError, KeyError) as exc:
            raise InvalidReplayToken("Replay envelope is invalid, expired, or belongs to another API Key/model.") from exc

    def seal_reasoning(self, owner: str, response: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(response)
        for item in result.get("output", []):
            if item.get("type") != "reasoning":
                continue
            if item.get("encrypted_content"):
                continue
            texts = [part.get("text", "") for part in item.get("summary", []) if part.get("type") == "summary_text"]
            content = [part.get("text", "") for part in item.get("content", []) if part.get("type") == "reasoning_text"]
            text = "".join(content or texts)
            if text:
                item["encrypted_content"] = self.encode(owner, result["model"], [ChatMessage(role="assistant", content=None, reasoning_content=text)], kind="reasoning")
        return result
