"""Pydantic models for API request/response (strict schema)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator
from pydantic.networks import AnyHttpUrl


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=200)


class Recommendation(BaseModel):
    name: str = Field(..., min_length=1, max_length=512)
    url: str = Field(..., min_length=8, max_length=2048)
    test_type: str = Field(..., max_length=256)

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not (v.startswith("https://") or v.startswith("http://")):
            raise ValueError("url must be http(s)")
        TypeAdapter(AnyHttpUrl).validate_python(v)
        return v


class ChatResponse(BaseModel):
    reply: str = Field(..., max_length=16000)
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def at_most_ten(cls, v: list[Recommendation]) -> list[Recommendation]:
        if len(v) > 10:
            raise ValueError("at most 10 recommendations")
        return v


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
