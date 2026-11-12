from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 18789
    api_key: str = "sk-local-change-me"
    data_dir: Path = Path("data")
    accounts_dir: Path = Path("accounts")
    upstream_base: str = "https://www.doubao.com"
    request_timeout_seconds: float = 180.0
    cooldown_seconds: int = 60
    default_model: str = "deepseek-v4.1-flash"
    app_id: str = "1044603"
    pc_version: str = "2.36.11"
    runtime_version: str = "3.45.3"
    model_key: str = "person-deepseek-v4-1-flash-261017"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("DOUBAO_WORK_HOST", "127.0.0.1").strip(),
            port=_env_int("DOUBAO_WORK_PORT", 18789),
            api_key=os.getenv("DOUBAO_WORK_API_KEY", "sk-local-change-me").strip(),
            data_dir=Path(
                os.getenv("DOUBAO_WORK_DATA_DIR", "data")
            ).expanduser().resolve(),
            accounts_dir=Path(
                os.getenv("DOUBAO_WORK_ACCOUNTS_DIR", "accounts")
            ).expanduser().resolve(),
            upstream_base=os.getenv(
                "DOUBAO_WORK_UPSTREAM", "https://www.doubao.com"
            ).rstrip("/"),
            request_timeout_seconds=float(
                os.getenv("DOUBAO_WORK_TIMEOUT", "180")
            ),
            cooldown_seconds=_env_int("DOUBAO_WORK_COOLDOWN_SECONDS", 60),
            default_model=os.getenv(
                "DOUBAO_WORK_DEFAULT_MODEL", "deepseek-v4.1-flash"
            ).strip(),
            app_id=os.getenv("DOUBAO_WORK_APP_ID", "1044603").strip(),
            pc_version=os.getenv(
                "DOUBAO_WORK_PC_VERSION", "2.36.11"
            ).strip(),
            runtime_version=os.getenv(
                "DOUBAO_WORK_RUNTIME_VERSION", "3.45.3"
            ).strip(),
            model_key=os.getenv(
                "DOUBAO_WORK_MODEL_KEY",
                "person-deepseek-v4-1-flash-261017",
            ).strip(),
        )
