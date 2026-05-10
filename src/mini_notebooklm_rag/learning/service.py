"""Learning artifact generation service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.learning.models import (
    ARTIFACT_FLASHCARDS,
    ARTIFACT_QUIZ,
    LEARNING_MODE_QUERY,
    LEARNING_PROMPT_VERSION,
    FlashcardSet,
    LearningResult,
    QuizSet,
)
from mini_notebooklm_rag.learning.parsing import LearningParseError, parse_learning_json
from mini_notebooklm_rag.learning.prompts import build_flashcard_prompt, build_quiz_prompt
from mini_notebooklm_rag.learning.validation import (
    validate_flashcard_payload,
    validate_quiz_payload,
)
from mini_notebooklm_rag.llm.models import LLMResponse, TokenUsage
from mini_notebooklm_rag.llm.openai_client import OpenAIClient, OpenAIClientError
from mini_notebooklm_rag.qa.source_mapping import build_source_references
from mini_notebooklm_rag.retrieval.service import MAX_SELECTED_DOCUMENTS, RetrievalService


class LLMClientProtocol(Protocol):
    """Minimal client shape used by learning generation."""

    def generate(
        self,
        instructions: str,
        input_text: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


class LearningService:
    """Generate grounded quiz and flashcard artifacts from retrieved context."""

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        llm_client: LLMClientProtocol | None = None,
    ):
        self.settings = settings
        self.retrieval_service = retrieval_service
        self._llm_client = llm_client

    def generate_quiz(
        self,
        workspace_id: str,
        selected_document_ids: list[str],
        topic_or_query: str,
        api_key: str,
        item_count: int = 5,
        model_name: str | None = None,
        top_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> LearningResult:
        """Generate a query-focused quiz from selected documents."""
        return self._generate(
            artifact_type=ARTIFACT_QUIZ,
            workspace_id=workspace_id,
            selected_document_ids=selected_document_ids,
            topic_or_query=topic_or_query,
            api_key=api_key,
            requested_count=item_count,
            model_name=model_name,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def generate_flashcards(
        self,
        workspace_id: str,
        selected_document_ids: list[str],
        topic_or_query: str,
        api_key: str,
        card_count: int = 10,
        model_name: str | None = None,
        top_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> LearningResult:
        """Generate query-focused flashcards from selected documents."""
        return self._generate(
            artifact_type=ARTIFACT_FLASHCARDS,
            workspace_id=workspace_id,
            selected_document_ids=selected_document_ids,
            topic_or_query=topic_or_query,
            api_key=api_key,
            requested_count=card_count,
            model_name=model_name,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def _generate(
        self,
        artifact_type: str,
        workspace_id: str,
        selected_document_ids: list[str],
        topic_or_query: str,
        api_key: str,
        requested_count: int,
        model_name: str | None,
        top_k: int | None,
        dense_weight: float | None,
        sparse_weight: float | None,
    ) -> LearningResult:
        topic = topic_or_query.strip()
        effective_model = model_name or self.settings.openai_model
        preflight = self._preflight(
            artifact_type,
            workspace_id,
            selected_document_ids,
            topic,
            api_key,
            effective_model,
        )
        if preflight is not None:
            return preflight

        retrieval_response = self.retrieval_service.retrieve(
            workspace_id=workspace_id,
            query=topic,
            selected_document_ids=list(dict.fromkeys(selected_document_ids)),
            top_k=top_k or self.settings.retrieval_top_k,
            dense_weight=self.settings.dense_weight if dense_weight is None else dense_weight,
            sparse_weight=self.settings.sparse_weight if sparse_weight is None else sparse_weight,
        )
        if not retrieval_response.results:
            warnings = list(retrieval_response.warnings)
            return LearningResult(
                status="failed",
                message="Not enough retrieved context to generate learning artifacts.",
                artifact_type=artifact_type,
                retrieval_response=retrieval_response,
                warnings=warnings,
                model_name=effective_model,
            )

        source_references = build_source_references(retrieval_response.results)
        prompt = (
            build_quiz_prompt(topic, source_references, requested_count)
            if artifact_type == ARTIFACT_QUIZ
            else build_flashcard_prompt(topic, source_references, requested_count)
        )
        try:
            llm_response = self._client(api_key, effective_model).generate(
                instructions=prompt.instructions,
                input_text=prompt.input_text,
                model=effective_model,
            )
            parsed = parse_learning_json(llm_response.text)
        except (LearningParseError, OpenAIClientError) as exc:
            return LearningResult(
                status="failed",
                message=f"Learning artifact generation failed: {exc}",
                artifact_type=artifact_type,
                retrieval_response=retrieval_response,
                warnings=list(retrieval_response.warnings),
                model_name=effective_model,
            )

        warnings = [*retrieval_response.warnings, *parsed.warnings]
        created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        if artifact_type == ARTIFACT_QUIZ:
            validated = validate_quiz_payload(parsed.data, source_references, requested_count)
            warnings.extend(validated.warnings)
            if not validated.items:
                return LearningResult(
                    status="failed",
                    message="No valid grounded quiz items were generated.",
                    artifact_type=artifact_type,
                    retrieval_response=retrieval_response,
                    warnings=warnings,
                    token_usage=llm_response.token_usage,
                    model_name=llm_response.model,
                )
            quiz_set = QuizSet(
                id=uuid.uuid4().hex,
                workspace_id=workspace_id,
                selected_document_ids=list(dict.fromkeys(selected_document_ids)),
                mode=LEARNING_MODE_QUERY,
                topic_or_query=topic,
                model_name=llm_response.model,
                prompt_version=LEARNING_PROMPT_VERSION,
                items=validated.items,
                source_map=source_references,
                warnings=warnings,
                token_usage=llm_response.token_usage,
                created_at=created_at,
            )
            return LearningResult(
                status="generated",
                message="Generated quiz.",
                artifact_type=artifact_type,
                quiz_set=quiz_set,
                retrieval_response=retrieval_response,
                warnings=warnings,
                token_usage=llm_response.token_usage,
                model_name=llm_response.model,
            )

        validated = validate_flashcard_payload(parsed.data, source_references, requested_count)
        warnings.extend(validated.warnings)
        if not validated.cards:
            return LearningResult(
                status="failed",
                message="No valid grounded flashcards were generated.",
                artifact_type=artifact_type,
                retrieval_response=retrieval_response,
                warnings=warnings,
                token_usage=llm_response.token_usage,
                model_name=llm_response.model,
            )
        flashcard_set = FlashcardSet(
            id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            selected_document_ids=list(dict.fromkeys(selected_document_ids)),
            mode=LEARNING_MODE_QUERY,
            topic_or_query=topic,
            model_name=llm_response.model,
            prompt_version=LEARNING_PROMPT_VERSION,
            cards=validated.cards,
            source_map=source_references,
            warnings=warnings,
            token_usage=llm_response.token_usage,
            created_at=created_at,
        )
        return LearningResult(
            status="generated",
            message="Generated flashcards.",
            artifact_type=artifact_type,
            flashcard_set=flashcard_set,
            retrieval_response=retrieval_response,
            warnings=warnings,
            token_usage=llm_response.token_usage,
            model_name=llm_response.model,
        )

    def _preflight(
        self,
        artifact_type: str,
        workspace_id: str,
        selected_document_ids: list[str],
        topic_or_query: str,
        api_key: str,
        model_name: str,
    ) -> LearningResult | None:
        if not topic_or_query:
            return _direct_result(
                artifact_type,
                "Enter a topic or question for Learning Tools.",
                model_name,
            )
        if not selected_document_ids:
            return _direct_result(
                artifact_type,
                "Select at least one source document before generating learning artifacts.",
                model_name,
            )
        if len(selected_document_ids) > MAX_SELECTED_DOCUMENTS:
            return _direct_result(
                artifact_type,
                f"Select at most {MAX_SELECTED_DOCUMENTS} source documents.",
                model_name,
            )
        status = self.retrieval_service.index_status(workspace_id)
        if status.status != "current":
            return _direct_result(artifact_type, status.message, model_name, list(status.warnings))
        if not api_key and self._llm_client is None:
            return _direct_result(
                artifact_type,
                "OpenAI API key is required. Set OPENAI_API_KEY or enter a temporary key.",
                model_name,
            )
        return None

    def _client(self, api_key: str, model_name: str) -> LLMClientProtocol:
        if self._llm_client is not None:
            return self._llm_client
        return OpenAIClient(api_key=api_key, default_model=model_name)


def _direct_result(
    artifact_type: str,
    message: str,
    model_name: str,
    warnings: list[str] | None = None,
) -> LearningResult:
    return LearningResult(
        status="skipped",
        message=message,
        artifact_type=artifact_type,
        warnings=warnings or [message],
        model_name=model_name,
        token_usage=TokenUsage(),
    )
