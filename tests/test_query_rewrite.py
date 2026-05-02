from __future__ import annotations

from mini_notebooklm_rag.chat.models import NewChatMessage
from mini_notebooklm_rag.qa.service import QAService
from tests.test_qa_service import FakeLLM, FakeRetrievalService, _workspace_and_chat


def test_query_rewrite_skips_when_disabled(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    chat_service.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="assistant",
            content="Previous answer",
        )
    )
    llm = FakeLLM(["Answer [S1]."])
    service = QAService(settings, chat_service, FakeRetrievalService(), llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What about it?",
        selected_document_ids=["doc1"],
        api_key="key",
        enable_query_rewrite=False,
    )

    assert result.rewrite_skipped_reason == "query rewrite disabled"
    assert len(llm.calls) == 1


def test_query_rewrite_skips_without_history(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    llm = FakeLLM(["Answer [S1]."])
    service = QAService(settings, chat_service, FakeRetrievalService(), llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What about it?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.rewrite_skipped_reason == "no current-session history"
    assert len(llm.calls) == 1


def test_query_rewrite_uses_current_session_history(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    other_session = chat_service.create_session(workspace.id, ["doc1"])
    chat_service.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=other_session.id,
            role="assistant",
            content="Other session secret",
        )
    )
    chat_service.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="assistant",
            content="Current session context",
        )
    )
    llm = FakeLLM(['{"action":"rewrite","query":"rewritten query"}', "Answer [S1]."])
    service = QAService(settings, chat_service, FakeRetrievalService(), llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What about it?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.rewritten_query == "rewritten query"
    assert "Current session context" in llm.calls[0]["input_text"]
    assert "Other session secret" not in llm.calls[0]["input_text"]
    assert len(llm.calls) == 2


def test_query_rewrite_clarification_persists_without_retrieval(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    chat_service.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="assistant",
            content="Previous answer",
        )
    )
    retrieval = FakeRetrievalService()
    llm = FakeLLM(['{"action":"clarify","question":"Which topic do you mean?"}'])
    service = QAService(settings, chat_service, retrieval, llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What about it?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.clarification_question == "Which topic do you mean?"
    assert retrieval.retrieve_calls == 0
    assert len(llm.calls) == 1


def test_query_rewrite_bad_json_falls_back_to_original_query(tmp_path) -> None:
    settings, workspace, chat_service, session = _workspace_and_chat(tmp_path)
    chat_service.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="assistant",
            content="Previous answer",
        )
    )
    llm = FakeLLM(["not json", "Answer [S1]."])
    service = QAService(settings, chat_service, FakeRetrievalService(), llm_client=llm)

    result = service.answer_question(
        workspace_id=workspace.id,
        session_id=session.id,
        question="What about it?",
        selected_document_ids=["doc1"],
        api_key="key",
    )

    assert result.rewritten_query is None
    assert result.rewrite_skipped_reason == "rewrite JSON parse failed"
    assert any(
        "Query rewrite response was not valid JSON" in warning for warning in result.warnings
    )
