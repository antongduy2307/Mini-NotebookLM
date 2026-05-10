from __future__ import annotations

from dataclasses import dataclass

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.learning.service import LearningService
from mini_notebooklm_rag.llm.models import LLMResponse, TokenUsage
from mini_notebooklm_rag.retrieval.models import (
    DenseCandidate,
    EmbeddingInfo,
    IndexStatus,
    RetrievalResponse,
    RetrievalTrace,
    RetrievedChunk,
    SparseCandidate,
)


def test_missing_api_key_does_not_call_llm() -> None:
    llm = FakeLLM('{"items":[],"warnings":[]}')
    service = LearningService(_settings(), FakeRetrievalService(), llm_client=None)

    result = service.generate_quiz("workspace-1", ["doc-1"], "retrieval", api_key="")

    assert result.status == "skipped"
    assert "OpenAI API key is required" in result.message
    assert llm.calls == 0


def test_no_selected_docs_does_not_call_llm() -> None:
    llm = FakeLLM('{"items":[],"warnings":[]}')
    service = LearningService(_settings(), FakeRetrievalService(), llm_client=llm)

    result = service.generate_quiz("workspace-1", [], "retrieval", api_key="test-key")

    assert result.status == "skipped"
    assert "Select at least one" in result.message
    assert llm.calls == 0


def test_missing_or_stale_index_does_not_call_llm() -> None:
    llm = FakeLLM('{"items":[],"warnings":[]}')
    retrieval = FakeRetrievalService(index_status=IndexStatus("stale", "Index is stale.", 1))
    service = LearningService(_settings(), retrieval, llm_client=llm)

    result = service.generate_quiz("workspace-1", ["doc-1"], "retrieval", api_key="test-key")

    assert result.status == "skipped"
    assert result.message == "Index is stale."
    assert llm.calls == 0


def test_empty_retrieval_does_not_call_llm() -> None:
    llm = FakeLLM('{"items":[],"warnings":[]}')
    retrieval = FakeRetrievalService(retrieval_response=_retrieval_response([]))
    service = LearningService(_settings(), retrieval, llm_client=llm)

    result = service.generate_quiz("workspace-1", ["doc-1"], "retrieval", api_key="test-key")

    assert result.status == "failed"
    assert "Not enough retrieved context" in result.message
    assert llm.calls == 0


def test_mocked_quiz_generation_returns_validated_quiz_set() -> None:
    llm = FakeLLM(
        '{"items":[{"question":"What does hybrid retrieval combine?",'
        '"options":["Dense and sparse signals","Only PDFs","Only summaries","Only evals"],'
        '"correct_index":0,"explanation":"It combines dense and sparse signals. [S1]",'
        '"source_markers":["[S1]"],"difficulty":"easy","topic":"Retrieval"}],'
        '"warnings":[]}',
        token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
    )
    service = LearningService(_settings(), FakeRetrievalService(), llm_client=llm)

    result = service.generate_quiz("workspace-1", ["doc-1"], "retrieval", api_key="")

    assert result.status == "generated"
    assert result.quiz_set is not None
    assert result.quiz_set.items[0].correct_index == 0
    assert result.quiz_set.items[0].source_markers == ["[S1]"]
    assert result.token_usage.total_tokens == 30
    assert llm.calls == 1


def test_mocked_flashcard_generation_returns_validated_flashcard_set() -> None:
    llm = FakeLLM(
        '{"cards":[{"front":"What is hybrid retrieval?",'
        '"back":"It combines dense and sparse retrieval signals. [S1]",'
        '"hint":"Two retrieval styles","topic":"Retrieval","source_markers":["[S1]"]}],'
        '"warnings":[]}'
    )
    service = LearningService(_settings(), FakeRetrievalService(), llm_client=llm)

    result = service.generate_flashcards(
        "workspace-1",
        ["doc-1"],
        "retrieval",
        api_key="",
    )

    assert result.status == "generated"
    assert result.flashcard_set is not None
    assert result.flashcard_set.cards[0].front == "What is hybrid retrieval?"
    assert llm.calls == 1


def test_malformed_llm_json_returns_failed_result() -> None:
    llm = FakeLLM("not json")
    service = LearningService(_settings(), FakeRetrievalService(), llm_client=llm)

    result = service.generate_quiz("workspace-1", ["doc-1"], "retrieval", api_key="")

    assert result.status == "failed"
    assert "not valid JSON" in result.message


@dataclass
class FakeLLM:
    text: str
    token_usage: TokenUsage = TokenUsage()
    calls: int = 0

    def generate(self, instructions, input_text, model=None, max_output_tokens=None):
        self.calls += 1
        return LLMResponse(
            text=self.text,
            model=model or "fake-model",
            token_usage=self.token_usage,
        )


class FakeRetrievalService:
    embedding_info = EmbeddingInfo(
        model_name="fake-embedding",
        requested_device="cpu",
        selected_device="cpu",
        dimension=3,
        normalized=True,
    )

    def __init__(
        self,
        index_status: IndexStatus | None = None,
        retrieval_response: RetrievalResponse | None = None,
    ):
        self._index_status = index_status or IndexStatus(
            status="current",
            message="Index is current.",
            chunk_count=1,
            indexed_chunk_count=1,
        )
        self._retrieval_response = retrieval_response or _retrieval_response([_chunk()])

    def index_status(self, workspace_id: str) -> IndexStatus:
        return self._index_status

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        selected_document_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> RetrievalResponse:
        return self._retrieval_response


def _settings() -> Settings:
    return Settings(_env_file=None, openai_model="fake-model")


def _retrieval_response(results: list[RetrievedChunk]) -> RetrievalResponse:
    return RetrievalResponse(
        results=results,
        trace=RetrievalTrace(
            original_query="retrieval",
            selected_document_ids=["doc-1"],
            embedding_model="fake-embedding",
            embedding_device="cpu",
            top_k=6,
            dense_weight=0.65,
            sparse_weight=0.35,
            dense_candidates=[DenseCandidate("chunk-1", "doc-1", 0.9, 1)] if results else [],
            sparse_candidates=[SparseCandidate("chunk-1", "doc-1", 1.2, 1)] if results else [],
            fused_results=results,
            warnings=[],
        ),
        warnings=[],
    )


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="sample.md",
        text="Hybrid retrieval combines dense and sparse retrieval signals.",
        source_type="markdown",
        page_start=None,
        page_end=None,
        heading_path=["Retrieval"],
        dense_score=0.9,
        sparse_score=0.8,
        fused_score=0.85,
        rank=1,
        citation="sample.md > Retrieval",
    )
