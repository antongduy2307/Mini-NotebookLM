"""Typed models for quiz and flashcard generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mini_notebooklm_rag.llm.models import TokenUsage
from mini_notebooklm_rag.qa.source_mapping import SourceReference
from mini_notebooklm_rag.retrieval.models import RetrievalResponse

ARTIFACT_QUIZ = "quiz"
ARTIFACT_FLASHCARDS = "flashcards"
LEARNING_MODE_QUERY = "query"
LEARNING_PROMPT_VERSION = "learning-query-v1"

LearningStatus = Literal["generated", "skipped", "failed"]
ArtifactType = Literal["quiz", "flashcards"]
LearningMode = Literal["query"]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class QuizItem:
    """Validated multiple-choice quiz item."""

    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source_markers: list[str]
    difficulty: Difficulty | None = None
    topic: str | None = None


@dataclass(frozen=True)
class QuizSet:
    """Generated quiz artifact kept in Streamlit session state."""

    id: str
    workspace_id: str
    selected_document_ids: list[str]
    mode: LearningMode
    topic_or_query: str
    model_name: str
    prompt_version: str
    items: list[QuizItem]
    source_map: list[SourceReference]
    warnings: list[str]
    token_usage: TokenUsage
    created_at: str


@dataclass(frozen=True)
class Flashcard:
    """Validated flashcard."""

    front: str
    back: str
    source_markers: list[str]
    hint: str | None = None
    topic: str | None = None


@dataclass(frozen=True)
class FlashcardSet:
    """Generated flashcard artifact kept in Streamlit session state."""

    id: str
    workspace_id: str
    selected_document_ids: list[str]
    mode: LearningMode
    topic_or_query: str
    model_name: str
    prompt_version: str
    cards: list[Flashcard]
    source_map: list[SourceReference]
    warnings: list[str]
    token_usage: TokenUsage
    created_at: str


@dataclass(frozen=True)
class LearningResult:
    """Service result for quiz and flashcard generation."""

    status: LearningStatus
    message: str
    artifact_type: ArtifactType
    quiz_set: QuizSet | None = None
    flashcard_set: FlashcardSet | None = None
    retrieval_response: RetrievalResponse | None = None
    warnings: list[str] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model_name: str | None = None
