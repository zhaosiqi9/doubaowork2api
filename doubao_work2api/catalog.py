from __future__ import annotations

from dataclasses import dataclass

from .errors import UnknownModelError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    name: str
    upstream_key: str
    mode: str = "work"
    price: str = ""
    description: str = ""

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "upstream_key": self.upstream_key,
            "mode": self.mode,
            "price": self.price,
            "description": self.description,
        }


# 公开模型名保持稳定；上游键取自豆包工作电脑版的模型菜单。
# 公开别名不会静默回退到其他模型。
MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="deepseek-v4.1-flash",
        name="DeepSeek V4.1 Flash",
        upstream_key="person-deepseek-v4-1-flash-261017",
        price="0.01x",
        description="默认模型，豆包工作对话入口。",
    ),
    ModelSpec(
        id="seed-lite-7b",
        name="豆包 2.1 Lite",
        upstream_key="seed-lite-7b",
        price="免费",
        description="轻量快速模型。",
    ),
    ModelSpec(
        id="person-kimi-k3-1-pro-261111",
        name="Kimi K3.1 Pro",
        upstream_key="person-kimi-k3-1-pro-261111",
        price="0.95x",
        description="长上下文推理。",
    ),
    ModelSpec(
        id="person-kimi-k3-1-pro-highspeed-261111",
        name="Kimi K3.1 Pro 高速",
        upstream_key="person-kimi-k3-1-pro-highspeed-261111",
        price="1.44x",
        description="高吞吐版本。",
    ),
    ModelSpec(
        id="person-kimi-k3-261111",
        name="Kimi K3",
        upstream_key="person-kimi-k3-261111",
        price="1.48x",
        description="Kimi K3 标准版。",
    ),
    ModelSpec(
        id="person-glm-5-3-261111",
        name="GLM 5.3",
        upstream_key="person-glm-5-3-261111",
        price="0.80x",
        description="通用文本模型。",
    ),
    ModelSpec(
        id="person-glm-5-3-flash-261111",
        name="GLM 5.3 Flash",
        upstream_key="person-glm-5-3-flash-261111",
        price="0.08x",
        description="低倍率快速模型。",
    ),
    ModelSpec(
        id="person-deepseek-v4-pro-ga-260813",
        name="DeepSeek V4 Pro",
        upstream_key="person-deepseek-v4-pro-ga-260813",
        price="0.46x",
        description="DeepSeek V4 Pro 工作模式。",
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
