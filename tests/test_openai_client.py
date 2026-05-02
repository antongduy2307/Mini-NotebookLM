from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mini_notebooklm_rag.llm.models import OpenAIClientError
from mini_notebooklm_rag.llm.openai_client import OpenAIClient


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class FakeResponse:
    output_text: str
    model: str = "gpt-test"
    id: str = "resp_123"
    usage: FakeUsage = field(default_factory=lambda: FakeUsage(10, 5, 15))


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(output_text="Answer [S1]")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_openai_client_wraps_responses_api_and_usage() -> None:
    fake_client = FakeClient()
    client = OpenAIClient(api_key="secret", default_model="gpt-default", client=fake_client)

    response = client.generate("instructions", "input", model="gpt-custom", max_output_tokens=50)

    assert response.text == "Answer [S1]"
    assert response.model == "gpt-test"
    assert response.response_id == "resp_123"
    assert response.token_usage.input_tokens == 10
    assert fake_client.responses.calls[0]["model"] == "gpt-custom"
    assert fake_client.responses.calls[0]["instructions"] == "instructions"
    assert fake_client.responses.calls[0]["input"] == "input"
    assert fake_client.responses.calls[0]["max_output_tokens"] == 50


def test_openai_client_requires_api_key() -> None:
    with pytest.raises(OpenAIClientError, match="API key is required"):
        OpenAIClient(api_key="", default_model="gpt-default", client=FakeClient())


def test_openai_client_sanitizes_sdk_exception() -> None:
    class BrokenResponses:
        def create(self, **kwargs):
            raise RuntimeError("raw secret-bearing SDK error")

    class BrokenClient:
        responses = BrokenResponses()

    client = OpenAIClient(api_key="secret", default_model="gpt-default", client=BrokenClient())

    with pytest.raises(OpenAIClientError) as exc:
        client.generate("instructions", "input")

    assert str(exc.value) == "OpenAI request failed."
