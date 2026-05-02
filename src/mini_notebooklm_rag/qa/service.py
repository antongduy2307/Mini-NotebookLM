"""Grounded QA orchestration over Phase 02 retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from mini_notebooklm_rag.chat.models import ChatMessage, ChatSession, NewChatMessage
from mini_notebooklm_rag.chat.service import ChatService
from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.llm.models import LLMResponse, TokenUsage
from mini_notebooklm_rag.llm.openai_client import OpenAIClient
from mini_notebooklm_rag.qa.prompts import (
    NOT_FOUND_MESSAGE,
    build_grounded_qa_prompt,
    build_outside_knowledge_prompt,
    build_query_rewrite_prompt,
)
from mini_notebooklm_rag.qa.source_mapping import (
    SourceReference,
    build_source_references,
    compact_source_map,
    find_unknown_source_markers,
    has_source_marker,
)
from mini_notebooklm_rag.retrieval.models import RetrievalResponse
from mini_notebooklm_rag.retrieval.service import MAX_SELECTED_DOCUMENTS, RetrievalService


class LLMClientProtocol(Protocol):
    def generate(
        self,
        instructions: str,
        input_text: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


class QAServiceError(RuntimeError):
    """Raised when QA orchestration cannot proceed."""


@dataclass(frozen=True)
class RewriteResult:
    query: str
    rewritten_query: str | None = None
    clarification_question: str | None = None
    skipped_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QAResult:
    user_message: ChatMessage
    assistant_message: ChatMessage
    answer: str
    sources: list[SourceReference]
    retrieval_response: RetrievalResponse | None
    original_query: str
    rewritten_query: str | None
    rewrite_skipped_reason: str | None
    clarification_question: str | None
    prompt_metadata: dict
    retrieval_metadata: dict
    model_name: str | None
    token_usage: TokenUsage
    warnings: list[str]


class QAService:
    """Coordinate chat persistence, query rewrite, retrieval, and answer generation."""

    def __init__(
        self,
        settings: Settings,
        chat_service: ChatService,
        retrieval_service: RetrievalService,
        llm_client: LLMClientProtocol | None = None,
    ):
        self.settings = settings
        self.chat_service = chat_service
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client

    def answer_question(
        self,
        workspace_id: str,
        session_id: str,
        question: str,
        selected_document_ids: list[str],
        api_key: str,
        model: str | None = None,
        rewrite_model: str | None = None,
        allow_outside_knowledge: bool | None = None,
        enable_query_rewrite: bool | None = None,
        top_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> QAResult:
        question = question.strip()
        if not question:
            raise QAServiceError("Enter a question.")
        session = self._require_session(session_id, workspace_id)
        selected_document_ids = list(dict.fromkeys(selected_document_ids))
        allow_outside = (
            self.settings.allow_outside_knowledge
            if allow_outside_knowledge is None
            else allow_outside_knowledge
        )
        rewrite_enabled = (
            self.settings.enable_query_rewrite
            if enable_query_rewrite is None
            else enable_query_rewrite
        )
        model = model or self.settings.openai_model
        rewrite_model = rewrite_model or self.settings.openai_query_rewrite_model
        top_k = top_k or self.settings.retrieval_top_k
        dense_weight = self.settings.dense_weight if dense_weight is None else dense_weight
        sparse_weight = self.settings.sparse_weight if sparse_weight is None else sparse_weight

        user_message = self.chat_service.add_message(
            NewChatMessage(
                workspace_id=workspace_id,
                session_id=session_id,
                role="user",
                content=question,
                selected_document_ids=selected_document_ids,
                original_query=question,
            )
        )
        self.chat_service.maybe_title_from_question(session, question)
        self.chat_service.update_session_documents(session_id, selected_document_ids)

        preflight_warning = self._preflight_warning(workspace_id, selected_document_ids)
        if preflight_warning is not None:
            return self._persist_direct_answer(
                workspace_id=workspace_id,
                session_id=session_id,
                user_message=user_message,
                answer=preflight_warning,
                selected_document_ids=selected_document_ids,
                original_query=question,
                rewritten_query=None,
                answer_mode="actionable",
                warnings=[preflight_warning],
            )

        if not api_key:
            warning = "OpenAI API key is required. Set OPENAI_API_KEY or enter a temporary key."
            return self._persist_direct_answer(
                workspace_id=workspace_id,
                session_id=session_id,
                user_message=user_message,
                answer=warning,
                selected_document_ids=selected_document_ids,
                original_query=question,
                rewritten_query=None,
                answer_mode="actionable",
                warnings=[warning],
            )

        client = self.llm_client or OpenAIClient(api_key=api_key, default_model=model)
        history = [
            message
            for message in self.chat_service.list_messages(session_id)
            if message.id != user_message.id
        ]
        rewrite_result = self._rewrite_query(
            client=client,
            question=question,
            history=history,
            selected_document_ids=selected_document_ids,
            enabled=rewrite_enabled,
            rewrite_model=rewrite_model,
        )
        if rewrite_result.clarification_question:
            return self._persist_direct_answer(
                workspace_id=workspace_id,
                session_id=session_id,
                user_message=user_message,
                answer=rewrite_result.clarification_question,
                selected_document_ids=selected_document_ids,
                original_query=question,
                rewritten_query=None,
                answer_mode="clarification",
                warnings=rewrite_result.warnings,
                prompt_metadata={"rewrite_action": "clarify"},
            )

        retrieval_query = rewrite_result.rewritten_query or rewrite_result.query
        retrieval_response = self.retrieval_service.retrieve(
            workspace_id=workspace_id,
            query=retrieval_query,
            selected_document_ids=selected_document_ids,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        retrieval_metadata = _compact_retrieval_metadata(retrieval_response)
        if not retrieval_response.results and not allow_outside:
            return self._persist_direct_answer(
                workspace_id=workspace_id,
                session_id=session_id,
                user_message=user_message,
                answer=NOT_FOUND_MESSAGE,
                selected_document_ids=selected_document_ids,
                original_query=question,
                rewritten_query=rewrite_result.rewritten_query,
                answer_mode="grounded",
                warnings=[*rewrite_result.warnings, *retrieval_response.warnings],
                retrieval_metadata=retrieval_metadata,
            )

        sources = build_source_references(retrieval_response.results)
        prompt = (
            build_outside_knowledge_prompt(question, sources)
            if allow_outside
            else build_grounded_qa_prompt(question, sources)
        )
        llm_response = client.generate(
            instructions=prompt.instructions,
            input_text=prompt.input_text,
            model=model,
        )
        answer = llm_response.text.strip() or NOT_FOUND_MESSAGE
        warnings = [
            *rewrite_result.warnings,
            *retrieval_response.warnings,
            *_citation_warnings(answer, sources, allow_outside),
        ]
        source_map = compact_source_map(sources)
        assistant_message = self.chat_service.add_message(
            NewChatMessage(
                workspace_id=workspace_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                selected_document_ids=selected_document_ids,
                original_query=question,
                rewritten_query=rewrite_result.rewritten_query,
                answer_mode="outside_knowledge" if allow_outside else "grounded",
                source_map=source_map,
                retrieval_metadata=retrieval_metadata,
                prompt_metadata=prompt.metadata,
                model_name=llm_response.model,
                input_tokens=llm_response.token_usage.input_tokens,
                output_tokens=llm_response.token_usage.output_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
            )
        )
        return QAResult(
            user_message=user_message,
            assistant_message=assistant_message,
            answer=answer,
            sources=sources,
            retrieval_response=retrieval_response,
            original_query=question,
            rewritten_query=rewrite_result.rewritten_query,
            rewrite_skipped_reason=rewrite_result.skipped_reason,
            clarification_question=None,
            prompt_metadata=prompt.metadata,
            retrieval_metadata=retrieval_metadata,
            model_name=llm_response.model,
            token_usage=llm_response.token_usage,
            warnings=warnings,
        )

    def _require_session(self, session_id: str, workspace_id: str) -> ChatSession:
        session = self.chat_service.get_session(session_id)
        if session is None or session.workspace_id != workspace_id:
            raise QAServiceError("Select or create a chat session.")
        return session

    def _preflight_warning(
        self,
        workspace_id: str,
        selected_document_ids: list[str],
    ) -> str | None:
        if not selected_document_ids:
            return "Select at least one document before asking a question."
        if len(selected_document_ids) > MAX_SELECTED_DOCUMENTS:
            return f"Select at most {MAX_SELECTED_DOCUMENTS} documents."
        chunk_counts = self.retrieval_service.documents.count_chunks_for_documents(
            workspace_id,
            selected_document_ids,
        )
        if any(count == 0 for count in chunk_counts.values()):
            return "Selected documents are missing or have no chunks."
        status = self.retrieval_service.index_status(workspace_id)
        if status.status != "current":
            return status.message
        return None

    def _rewrite_query(
        self,
        client: LLMClientProtocol,
        question: str,
        history: list[ChatMessage],
        selected_document_ids: list[str],
        enabled: bool,
        rewrite_model: str,
    ) -> RewriteResult:
        if not enabled:
            return RewriteResult(query=question, skipped_reason="query rewrite disabled")
        if not history:
            return RewriteResult(query=question, skipped_reason="no current-session history")
        if _looks_standalone(question):
            return RewriteResult(query=question, skipped_reason="query appears standalone")

        prompt = build_query_rewrite_prompt(question, history, selected_document_ids)
        response = client.generate(
            instructions=prompt.instructions,
            input_text=prompt.input_text,
            model=rewrite_model,
        )
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return RewriteResult(
                query=question,
                skipped_reason="rewrite JSON parse failed",
                warnings=["Query rewrite response was not valid JSON; using original query."],
            )

        action = payload.get("action")
        if action == "clarify" and payload.get("question"):
            return RewriteResult(query=question, clarification_question=str(payload["question"]))
        if action == "rewrite" and payload.get("query"):
            rewritten = str(payload["query"]).strip()
            return RewriteResult(query=question, rewritten_query=rewritten or question)
        return RewriteResult(
            query=question,
            skipped_reason="rewrite response missing action/query",
            warnings=["Query rewrite response was incomplete; using original query."],
        )

    def _persist_direct_answer(
        self,
        workspace_id: str,
        session_id: str,
        user_message: ChatMessage,
        answer: str,
        selected_document_ids: list[str],
        original_query: str,
        rewritten_query: str | None,
        answer_mode: str,
        warnings: list[str],
        retrieval_metadata: dict | None = None,
        prompt_metadata: dict | None = None,
    ) -> QAResult:
        assistant_message = self.chat_service.add_message(
            NewChatMessage(
                workspace_id=workspace_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                selected_document_ids=selected_document_ids,
                original_query=original_query,
                rewritten_query=rewritten_query,
                answer_mode=answer_mode,
                retrieval_metadata=retrieval_metadata,
                prompt_metadata=prompt_metadata,
            )
        )
        return QAResult(
            user_message=user_message,
            assistant_message=assistant_message,
            answer=answer,
            sources=[],
            retrieval_response=None,
            original_query=original_query,
            rewritten_query=rewritten_query,
            rewrite_skipped_reason=None,
            clarification_question=answer if answer_mode == "clarification" else None,
            prompt_metadata=prompt_metadata or {},
            retrieval_metadata=retrieval_metadata or {},
            model_name=None,
            token_usage=TokenUsage(),
            warnings=warnings,
        )


def _looks_standalone(question: str) -> bool:
    tokens = {token.strip(".,?!;:").casefold() for token in question.split()}
    follow_up_markers = {
        "it",
        "they",
        "that",
        "this",
        "those",
        "he",
        "she",
        "above",
        "earlier",
        "same",
        "compare",
    }
    return len(question.split()) >= 6 and not tokens.intersection(follow_up_markers)


def _citation_warnings(
    answer: str,
    sources: list[SourceReference],
    allow_outside_knowledge: bool,
) -> list[str]:
    warnings = [
        f"Answer referenced unknown source marker {marker}."
        for marker in find_unknown_source_markers(answer, sources)
    ]
    if (
        not allow_outside_knowledge
        and answer.strip() != NOT_FOUND_MESSAGE
        and not has_source_marker(answer)
    ):
        warnings.append("Grounded answer does not contain a source marker.")
    return warnings


def _compact_retrieval_metadata(response: RetrievalResponse) -> dict:
    return {
        "warnings": response.warnings,
        "trace": {
            "original_query": response.trace.original_query,
            "selected_document_ids": response.trace.selected_document_ids,
            "embedding_model": response.trace.embedding_model,
            "embedding_device": response.trace.embedding_device,
            "top_k": response.trace.top_k,
            "dense_weight": response.trace.dense_weight,
            "sparse_weight": response.trace.sparse_weight,
            "dense_candidates": [
                candidate.__dict__ for candidate in response.trace.dense_candidates
            ],
            "sparse_candidates": [
                candidate.__dict__ for candidate in response.trace.sparse_candidates
            ],
            "fused_result_ids": [result.chunk_id for result in response.trace.fused_results],
            "warnings": response.trace.warnings,
        },
    }
