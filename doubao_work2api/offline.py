from __future__ import annotations

import base64
import ctypes
import json
import os
import re
import shutil
import sqlite3
import tempfile
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OfflineScanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OfflineIdentity:
    cookies: dict[str, str]
    device_id: str
    web_id: str


def default_user_data_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if not local_app_data:
        raise OfflineScanError("当前系统没有可用的 LOCALAPPDATA 目录。")
    return Path(local_app_data) / "DoubaoWork" / "User Data"


def _profile_dir(user_data_dir: Path) -> Path:
    default_profile = user_data_dir / "Default"
    if (default_profile / "Network" / "Cookies").is_file():
        return default_profile
    profiles = sorted(
        path
        for path in user_data_dir.iterdir()
        if path.is_dir() and (path / "Network" / "Cookies").is_file()
    )
    if not profiles:
        raise OfflineScanError(
            "没有找到豆包工作电脑版的配置目录，请先登录客户端。"
        )
    return profiles[0]


def _copy_database(source: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="doubaowork-cookies-")
    destination = Path(temporary.name) / "Cookies"
    try:
        shutil.copy2(source, destination)
        # Chromium may keep recent cookie writes in the WAL file. Copy it
        # together with the database so a cleanly closed client is readable.
        wal = source.with_name(source.name + "-wal")
        if wal.is_file():
            shutil.copy2(wal, destination.with_name(destination.name + "-wal"))
    except OSError as exc:
        temporary.cleanup()
        raise OfflineScanError(
            "豆包工作电脑版正在运行并占用了 Cookie 数据库。"
            "请完全退出客户端后重试离线扫描。"
        ) from exc
    return temporary, destination


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_unprotect(data: bytes) -> bytes:
    if os.name != "nt":
        raise OfflineScanError("离线 Cookie 解密仅支持 Windows。")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    input_buffer = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(
        len(data),
        ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_char)),
    )
    output_blob = _DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(output_blob),
    ):
        raise OfflineScanError("无法使用当前 Windows 用户配置解密 Cookie。")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output_blob.pbData)


def _cookie_key(user_data_dir: Path) -> bytes:
    local_state = user_data_dir / "Local State"
    try:
        payload = json.loads(local_state.read_text(encoding="utf-8"))
        encrypted = base64.b64decode(
            payload["os_crypt"]["encrypted_key"], validate=True
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise OfflineScanError(
            "无法读取豆包工作电脑版的 Cookie 加密密钥。"
        ) from exc
    if not encrypted.startswith(b"DPAPI"):
        raise OfflineScanError("当前 Cookie 加密格式不受支持。")
    key = _dpapi_unprotect(encrypted[5:])
    if len(key) != 32:
        raise OfflineScanError("Cookie 加密密钥长度无效。")
    return key


def _strip_app_prefix(plaintext: bytes) -> bytes:
    """Strip Doubao's 32-byte app-bound prefix without corrupting normal cookies."""
    if len(plaintext) <= 32:
        return plaintext
    prefix = plaintext[:32]
    value = plaintext[32:]
    prefix_is_binary = any(byte < 32 or byte > 126 for byte in prefix)
    value_is_text = bool(value) and all(32 <= byte <= 126 for byte in value)
    return value if prefix_is_binary and value_is_text else plaintext


def _decrypt_cookie(encrypted: bytes, key: bytes) -> str:
    if not encrypted:
        return ""
    if encrypted.startswith(b"v10") or encrypted.startswith(b"v11"):
        try:
            from cryptography.exceptions import InvalidTag
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise OfflineScanError(
                "离线解密需要安装 cryptography 依赖。"
            ) from exc
        nonce = encrypted[3:15]
        ciphertext = encrypted[15:]
        if len(nonce) != 12 or len(ciphertext) < 16:
            raise OfflineScanError("Chromium Cookie 密文无效。")
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        except (InvalidTag, ValueError) as exc:
            raise OfflineScanError("Cookie 解密校验失败。") from exc
        # DoubaoWork prefixes values with a 32-byte app-bound digest. The
        # actual cookie value follows it, while ordinary Chromium cookies
        # do not carry this prefix.
        plaintext = _strip_app_prefix(plaintext)
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OfflineScanError("解密后的 Cookie 不是有效 UTF-8。") from exc
    try:
        return _dpapi_unprotect(encrypted).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OfflineScanError("解密后的 Cookie 不是有效 UTF-8。") from exc


def _read_cookies(profile: Path, key: bytes) -> dict[str, str]:
    temporary, database = _copy_database(profile / "Network" / "Cookies")
    try:
        connection = sqlite3.connect(str(database))
        try:
            connection.execute("PRAGMA query_only = ON")
            rows = connection.execute(
                "SELECT host_key, name, value, encrypted_value "
                "FROM cookies WHERE host_key = 'doubao.com' "
                "OR host_key LIKE '%.doubao.com'"
            ).fetchall()
        except sqlite3.Error as exc:
            raise OfflineScanError("无法读取豆包 Cookie 数据库。") from exc
        finally:
            connection.close()
    finally:
        temporary.cleanup()
    result: dict[str, str] = {}
    errors: list[str] = []
    for _, name, value, encrypted in rows:
        if not name:
            continue
        if value:
            result[str(name)] = str(value)
            continue
        try:
            result[str(name)] = _decrypt_cookie(bytes(encrypted or b""), key)
        except OfflineScanError as exc:
            errors.append(f"{name}: {exc}")
    if not result:
        detail = f" ({errors[0]})" if errors else ""
        raise OfflineScanError(f"没有读取到可用的豆包 Cookie{detail}。")
    return result


def _iter_leveldb_files(profile: Path) -> list[Path]:
    storage = profile / "Local Storage" / "leveldb"
    if not storage.is_dir():
        return []
    return [
        path
        for path in sorted(storage.iterdir())
        if path.is_file() and path.suffix in {".ldb", ".log"}
    ]


def _read_leveldb_bytes(profile: Path) -> bytes:
    chunks: list[bytes] = []
    for path in _iter_leveldb_files(profile):
        try:
            chunks.append(path.read_bytes())
        except OSError:
            continue
    return b"\n".join(chunks)


def _json_objects(data: bytes) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    text = data.decode("utf-8", "ignore")
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = start + max(consumed, 1)
    return objects


def _read_identity(profile: Path) -> tuple[str, str]:
    data = _read_leveldb_bytes(profile)
    if not data:
        raise OfflineScanError("没有找到豆包工作电脑版的本地存储数据。")
    objects = _json_objects(data)
    preferred = [
        item
        for item in objects
        if item.get("user_unique_id") and item.get("web_id")
    ]
    fallback = [
        item
        for item in objects
        if item.get("web_id") and item.get("timestamp")
    ]
    candidates = preferred or fallback
    latest = max(candidates, key=_timestamp) if candidates else {}
    web_id = str(latest.get("web_id") or "").strip()
    device_id = str(latest.get("user_unique_id") or "").strip()
    if not device_id:
        for item in reversed(objects):
            if item.get("user_unique_id"):
                device_id = str(item["user_unique_id"]).strip()
                break
    if not device_id:
        # Older Tea SDK entries use a compact LevelDB/protobuf value layout.
        device_id = _compact_user_unique_id(data)
    if not web_id or not device_id:
        compact_device, compact_web = _compact_identity(data)
        device_id = device_id or compact_device
        web_id = web_id or compact_web
    if not candidates and not (device_id and web_id):
        raise OfflineScanError(
            "没有找到豆包工作电脑版的登录身份，请先登录客户端。"
        )
    if not web_id:
        raise OfflineScanError("没有找到豆包工作电脑版的 web_id。")
    if not device_id:
        raise OfflineScanError("没有找到豆包工作电脑版的 device_id。")
    return device_id, web_id


def _timestamp(item: dict[str, Any]) -> int:
    try:
        return int(item.get("timestamp") or 0)
    except (TypeError, ValueError):
        return 0


def _compact_user_unique_id(data: bytes) -> str:
    marker = b'"user_unique'
    start = 0
    while True:
        index = data.find(marker, start)
        if index < 0:
            return ""
        tail = data[index + len(marker) : index + len(marker) + 32]
        digits = re.search(rb"(\d{10,})", tail)
        if digits:
            return digits.group(1).decode("ascii")
        start = index + 1


def _compact_identity(data: bytes) -> tuple[str, str]:
    """Read Tea SDK identity fields from partially binary LevelDB records."""
    web_pattern = re.compile(rb'"web_id"\s*:\s*"([^"]+)"')
    timestamp_pattern = re.compile(rb'"timestamp"\s*:\s*(\d+)')
    best: tuple[int, str, str] | None = None
    fallback_web = ""
    fallback_device = ""
    for match in web_pattern.finditer(data):
        web_id = match.group(1).decode("ascii", "ignore").strip()
        if not web_id:
            continue
        fallback_web = web_id
        window = data[match.start() : match.start() + 512]
        device_match = re.search(rb'"user_unique.{0,48}?(\d{10,})', window, re.DOTALL)
        if not device_match:
            continue
        device_id = device_match.group(1).decode("ascii", "ignore")
        fallback_device = device_id
        timestamp_match = timestamp_pattern.search(window)
        timestamp = int(timestamp_match.group(1)) if timestamp_match else 0
        candidate = (timestamp, device_id, web_id)
        if best is None or candidate[0] >= best[0]:
            best = candidate
    if best is not None:
        return best[1], best[2]
    if not fallback_device:
        fallback_device = _compact_user_unique_id(data)
    return fallback_device, fallback_web


def scan_offline_account(user_data_dir: Path | None = None) -> OfflineIdentity:
    if os.name != "nt":
        raise OfflineScanError(
            "离线扫描仅支持 Windows，其他平台请导入账号 JSON。"
        )
    root = (user_data_dir or default_user_data_dir()).expanduser()
    if not root.is_dir():
        raise OfflineScanError(
            "没有找到豆包工作电脑版的数据目录，请先登录客户端。"
        )
    profile = _profile_dir(root)
    key = _cookie_key(root)
    cookies = _read_cookies(profile, key)
    device_id, web_id = _read_identity(profile)
    return OfflineIdentity(
        cookies=cookies,
        device_id=device_id,
        web_id=web_id,
    )
