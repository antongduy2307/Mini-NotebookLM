from __future__ import annotations

from mini_notebooklm_rag.chat.service import ChatService
from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.llm.models import LLMResponse, TokenUsage
from mini_notebooklm_rag.qa.prompts import NOT_FOUND_MESSAGE
from mini_notebooklm_rag.qa.service import QAService
from mini_notebooklm_rag.retrieval.models import (
    IndexStatus,
    RetrievalResponse,
    RetrievalTrace,
    RetrievedChunk,
)
from mini_notebooklm_rag.storage.repositories import WorkspaceRepository
from mini_notebooklm_rag.storage.sqlite import initialize_database


class FakeDocuments:
    def __init__(self, counts: dict[str, int]):
        self.counts = counts

    def count_chunks_for_documents(
        self, workspace_id: str, document_ids: list[str]
    ) -> dict[str, int]:
        return {document_id: self.counts.get(document_id, 0) for document_id in document_ids}


class FakeRetrievalService:
    def __init__(
        self,
        status: str = "current",
        results: list[RetrievedChunk] | None = None,
        counts: dict[str, int] | None = None,
    ):
        self.status = status
        self.results = results if results is not None else [_retrieved_chunk()]
        self.documents = FakeDocuments(counts or {"doc1": 1})
        self.retrieve_calls = 0

    def index_status(self, workspace_id: str) -> IndexStatus:
        return IndexStatus(
            status=self.status,
            message="FAISS index is current."
            if self.status == "current"
            else "FAISS index is stale.",
            chunk_count=1,
        )

    def retrieve(self, **kwargs) -> RetrievalResponse:
        self.retrieve_calls += 1
        return _retrieval_response(kwargs["query"], self.results)


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []

    def generate(self, instructions, input_text, model=None, max_output_tokens=None):
        self.calls.append(
            {
                "instructions": instructions,
                "input_text": input_text,
                "model": model,
            }
        )
        text = self.responses.pop(0)
        return LLMResponse(
            text=text,
            model=model or "fake-model",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, app_storage_dir=str(tmp_path / "storage"))


def _workspace_and_chat(tmp_path):
    settings = _settings(tmp_path)
    db_path = tmp_path / "storage" / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    chat_service = ChatService(settings)
    session = chat_service.create_session(workspace.id, ["doc1"])
    return settings, workspace, chat_service, session


def _retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk1",
        document_id="doc1",
        filename="notes.md",
        text="alpha answer source",
        source_type="markdown",
        page_start=None,
        page_end=None,
        heading_path=["Intro"],
        dense_score=1.0,
        sparse_score=1.0,
        fused_score=1.0,
        rank=1,
        citation="notes.md > Intro",
    )


def _retrieval_response(query: str, results: list[RetrievedChunk]) -> RetrievalResponse:
    trace = RetrievalTrace(
        original_query=query,
        selected_document_ids=["doc1"],
        embedding_model="fake-embedding",
        embedding_device="cpu",
        top_k=3,
        dense_weight=0.65,
        sparse_weight=0.35,
        dense_candidates=[],
        sparse_candidates=[],
        fused_results=results,
        warnings=[],
    )
    return RetrievalResponse(results=results, trace=trace, warnings=[])


def test_qa_service_returns_actionable_message_without_openai_on_stale_index(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    llm = FakeLLM(["should not be used"])
    service = QAService(
        settings,
        chat_service,
        FakeRetrievalService(status="stale"),
        llm_client=llm,
    )

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What is alpha?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.answer == "FAISS index is stale."
    assert llm.calls == []


def test_qa_service_grounded_not_found_shortcuts_without_openai(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    retrieval = FakeRetrievalService(results=[])
    llm = FakeLLM(["should not be used"])
    service = QAService(settings, chat_service, retrieval, llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="Missing?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.answer == NOT_FOUND_MESSAGE
    assert retrieval.retrieve_calls == 1
    assert llm.calls == []


def test_qa_service_generates_grounded_answer_and_compact_source_map(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    llm = FakeLLM(["Alpha is covered [S1]."])
    service = QAService(settings, chat_service, FakeRetrievalService(), llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What is alpha?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.answer == "Alpha is covered [S1]."
    assert result.sources[0].source_id == "S1"
    assert result.assistant_message.source_map is not None
    assert "text" not in result.assistant_message.source_map[0]
    assert result.assistant_message.total_tokens == 15


def test_qa_service_warns_on_unknown_source_marker(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    service = QAService(
        settings,
        chat_service,
        FakeRetrievalService(),
        llm_client=FakeLLM(["Bad marker [S9]."]),
    )

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What is alpha?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert "Answer referenced unknown source marker [S9]." in result.warnings


def test_qa_service_outside_knowledge_prompt_separates_sections(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    llm = FakeLLM(["From your documents:\nAlpha [S1]\n\nOutside the selected documents:\nExtra"])
    service = QAService(settings, chat_service, FakeRetrievalService(), llm_client=llm)

    service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What is alpha?",
        selected_document_ids=["doc1"],
        api_key="key",
        allow_outside_knowledge=True,
    )

    assert "From your documents:" in llm.calls[0]["instructions"]
    assert "Outside the selected documents:" in llm.calls[0]["instructions"]


def test_qa_service_missing_selected_document_prevents_openai(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    llm = FakeLLM(["should not be used"])
    retrieval = FakeRetrievalService(counts={"doc1": 0})
    service = QAService(settings, chat_service, retrieval, llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What is alpha?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.answer == "Selected documents are missing or have no chunks."
    assert llm.calls == []
