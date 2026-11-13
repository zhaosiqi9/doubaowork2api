from __future__ import annotations

import os
from pathlib import Path

import pytest

from doubao_work2api.offline import (
    OfflineScanError,
    _compact_identity,
    _compact_user_unique_id,
    _decrypt_cookie,
    _json_objects,
    _profile_dir,
    _strip_app_prefix,
)


def test_json_objects_reads_concatenated_leveldb_json() -> None:
    payload = (
        b'prefix{"web_id":"first","user_unique_id":"111","timestamp":1}'
        b'middle{"web_id":"second","user_unique_id":"222","timestamp":2}'
    )
    assert _json_objects(payload) == [
        {"web_id": "first", "user_unique_id": "111", "timestamp": 1},
        {"web_id": "second", "user_unique_id": "222", "timestamp": 2},
    ]


def test_compact_identity_fallback_reads_binary_tail() -> None:
    payload = (
        b'{"web_id":"7707296447172183562","user_unique'
        b'\t\'\xf0F1026322651175628","timestamp":1}'
    )
    assert _compact_user_unique_id(payload) == "1026322651175628"


def test_compact_identity_reads_truncated_leveldb_record() -> None:
    payload = (
        b'\\x00\\x01{"web_id":"7707295574780872243","user_unique'
        b'\\t\\\'\\xf0F1026322651175628","timestamp":1794494595889,'
        b'"other":"ignored"}'
    )
    assert _compact_identity(payload) == (
        "1026322651175628",
        "7707295574780872243",
    )


def test_strip_app_prefix_only_removes_binary_digest() -> None:
    normal = b"ordinary-cookie-value-without-app-prefix"
    assert _strip_app_prefix(normal) == normal
    assert _strip_app_prefix(b"\x01" * 32 + b"real-cookie") == b"real-cookie"


def test_decrypt_cookie_rejects_invalid_ciphertext() -> None:
    with pytest.raises(OfflineScanError, match="密文无效"):
        _decrypt_cookie(b"v10short", b"0" * 32)


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is Windows-only")
def test_real_windows_offline_scan() -> None:
    from doubao_work2api.offline import scan_offline_account

    try:
        identity = scan_offline_account()
    except OfflineScanError as exc:
        pytest.skip(f"本机没有可读取的豆包登录数据：{exc}")
    assert identity.cookies
    assert identity.device_id
    assert identity.web_id


def test_profile_dir_uses_default_profile(tmp_path: Path) -> None:
    network = tmp_path / "Default" / "Network"
    network.mkdir(parents=True)
    (network / "Cookies").write_bytes(b"")
    assert _profile_dir(tmp_path) == tmp_path / "Default"


def test_profile_dir_finds_non_default_profile(tmp_path: Path) -> None:
    network = tmp_path / "Profile 1" / "Network"
    network.mkdir(parents=True)
    (network / "Cookies").write_bytes(b"")
    assert _profile_dir(tmp_path) == tmp_path / "Profile 1"


def test_profile_dir_rejects_missing_profile(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="配置目录"):
        _profile_dir(tmp_path)
