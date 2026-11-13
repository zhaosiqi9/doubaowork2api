from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import UnknownModelError


REASONING_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
REASONING_EFFORT_BY_LEVEL: dict[str, int] = {
    "low": 3,
    "medium": 4,
    "high": 5,
    "xhigh": 6,
    "max": 7,
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    name: str
    upstream_key: str
    mode: str = "work"
    price: str = ""
    description: str = ""
    context_window: int | None = None
    reasoning_levels: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    memory_profile: str | None = None
    image_input: bool | None = None

    def public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "upstream_key": self.upstream_key,
            "mode": self.mode,
            "price": self.price,
            "description": self.description,
            "context_window": self.context_window,
            "reasoning_efforts": list(self.reasoning_levels),
            "image_input": self.image_input,
        }
        if self.default_reasoning_effort is not None:
            result["default_reasoning_effort"] = self.default_reasoning_effort
        return result

    @property
    def has_verified_memory_profile(self) -> bool:
        return self.memory_profile is not None and self.context_window is not None


# 公开模型名保持稳定；上游键取自豆包工作电脑版的模型菜单。
# 公开别名不会静默回退到其他模型。
MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="deepseek-v4.1-flash",
        name="DeepSeek V4.1 Flash",
        upstream_key="person-deepseek-v4-1-flash-261017",
        price="0.01x",
        description="默认模型，豆包工作对话入口。",
        context_window=300_000,
        reasoning_levels=REASONING_LEVELS,
        default_reasoning_effort="max",
        memory_profile="300k_230",
        image_input=True,
    ),
    ModelSpec(
        id="seed-lite-7b",
        name="豆包 2.1 Lite",
        upstream_key="seed-lite-7b",
        price="免费",
        description="轻量快速模型。",
        context_window=256_000,
        reasoning_levels=("low", "medium", "high"),
        default_reasoning_effort="high",
        memory_profile="256k",
    ),
    ModelSpec(
        id="person-kimi-k3-1-pro-261111",
        name="Kimi K3.1 Pro",
        upstream_key="person-kimi-k3-1-pro-261111",
        price="0.95x",
        description="长上下文推理。",
        context_window=300_000,
        reasoning_levels=REASONING_LEVELS,
        default_reasoning_effort="max",
        memory_profile="300k_230",
    ),
    ModelSpec(
        id="person-kimi-k3-1-pro-highspeed-261111",
        name="Kimi K3.1 Pro 高速",
        upstream_key="person-kimi-k3-1-pro-highspeed-261111",
        price="1.44x",
        description="高吞吐版本。",
        context_window=300_000,
        reasoning_levels=REASONING_LEVELS,
        default_reasoning_effort="max",
        memory_profile="300k_230",
    ),
    ModelSpec(
        id="person-kimi-k3-261111",
        name="Kimi K3",
        upstream_key="person-kimi-k3-261111",
        price="1.48x",
        description="Kimi K3 标准版。",
        context_window=300_000,
        reasoning_levels=REASONING_LEVELS,
        default_reasoning_effort="max",
        memory_profile="300k_230",
    ),
    ModelSpec(
        id="person-glm-5-3-261111",
        name="GLM 5.3",
        upstream_key="person-glm-5-3-261111",
        price="0.80x",
        description="通用文本模型。",
        context_window=300_000,
        reasoning_levels=("low", "high", "max"),
        default_reasoning_effort="max",
        memory_profile="300k_230",
    ),
    ModelSpec(
        id="person-glm-5-3-flash-261111",
        name="GLM 5.3 Flash",
        upstream_key="person-glm-5-3-flash-261111",
        price="0.08x",
        description="低倍率快速模型。",
        context_window=300_000,
        reasoning_levels=("low", "high", "max"),
        default_reasoning_effort="max",
        memory_profile="300k_230",
    ),
    ModelSpec(
        id="person-deepseek-v4-pro-ga-260813",
        name="DeepSeek V4 Pro",
        upstream_key="person-deepseek-v4-pro-ga-260813",
        price="0.46x",
        description="DeepSeek V4 Pro 工作模式。",
        context_window=300_000,
        reasoning_levels=REASONING_LEVELS,
        default_reasoning_effort="max",
        memory_profile="300k_230",
    ),
)

_BY_ID = {item.id: item for item in MODEL_CATALOG}
_ALIASES = {
    "deepseek-v4-1-flash": "deepseek-v4.1-flash",
    "deepseek-v4.1-flash-261017": "deepseek-v4.1-flash",
    "person-deepseek-v4-1-flash-261017": "deepseek-v4.1-flash",
    "kimi-k3.1-pro": "person-kimi-k3-1-pro-261111",
    "kimi-k3.1-pro-highspeed": "person-kimi-k3-1-pro-highspeed-261111",
    "kimi-k3": "person-kimi-k3-261111",
    "glm-5.3": "person-glm-5-3-261111",
    "glm-5.3-flash": "person-glm-5-3-flash-261111",
    "deepseek-v4-pro": "person-deepseek-v4-pro-ga-260813",
}


def normalize_model(model: str) -> str:
    value = (model or "").strip().lower()
    return _ALIASES.get(value, value)


def find_model(model: str) -> ModelSpec | None:
    return _BY_ID.get(normalize_model(model))


def resolve_upstream_key(model: str) -> str:
    spec = find_model(model)
    if spec is None:
        raise UnknownModelError(model)
    return spec.upstream_key
