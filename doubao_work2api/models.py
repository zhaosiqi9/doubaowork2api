from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "deepseek-v4.1-flash"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    user: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None

    @property
    def normalized_model(self) -> str:
        aliases = {
            "deepseek-v4-1-flash": "deepseek-v4.1-flash",
            "deepseek-v4.1-flash-261017": "deepseek-v4.1-flash",
            "person-deepseek-v4-1-flash-261017": "deepseek-v4.1-flash",
        }
        return aliases.get(self.model.lower(), self.model)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "doubao"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
