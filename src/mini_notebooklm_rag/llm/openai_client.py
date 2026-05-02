"""OpenAI Responses API wrapper."""

from __future__ import annotations

from typing import Any

from mini_notebooklm_rag.llm.models import LLMResponse, OpenAIClientError, TokenUsage


class OpenAIClient:
    """Small non-streaming wrapper around the official OpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        default_model: str,
        client: Any | None = None,
    ):
        if not api_key:
            raise OpenAIClientError("OpenAI API key is required.")
        self.default_model = default_model
        self._client = client or self._build_client(api_key)

    def generate(
        self,
        instructions: str,
        input_text: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text through the non-streaming Responses API."""
        selected_model = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "instructions": instructions,
            "input": input_text,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens

        try:
            response = self._client.responses.create(**kwargs)
        except Exception as exc:
            raise OpenAIClientError("OpenAI request failed.") from exc

        text = str(getattr(response, "output_text", "") or "")
        if not text:
            text = _extract_output_text(response)

        return LLMResponse(
            text=text.strip(),
            model=str(getattr(response, "model", selected_model)),
            response_id=getattr(response, "id", None),
            token_usage=_extract_usage(response),
            raw_finish_reason=getattr(response, "finish_reason", None),
        )

    def _build_client(self, api_key: str) -> Any:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise OpenAIClientError("The openai package is not installed.") from exc
        return OpenAI(api_key=api_key)


def _extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    input_tokens = _get_usage_value(usage, "input_tokens")
    output_tokens = _get_usage_value(usage, "output_tokens")
    total_tokens = _get_usage_value(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _get_usage_value(usage: Any, key: str) -> int | None:
    value = getattr(usage, key, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(key)
    return int(value) if value is not None else None


def _extract_output_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)
