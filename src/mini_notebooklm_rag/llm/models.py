"""Typed LLM request and response structures."""

from __future__ import annotations

from dataclasses import dataclass


class OpenAIClientError(RuntimeError):
    """Raised when OpenAI generation fails."""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    response_id: str | None = None
    token_usage: TokenUsage = TokenUsage()
    raw_finish_reason: str | None = None
