"""JSON export helpers for learning artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime

from mini_notebooklm_rag.learning.models import FlashcardSet, QuizSet
from mini_notebooklm_rag.utils.filenames import sanitize_filename

EXPORT_FORMAT_VERSION = 1


def quiz_set_to_json_dict(quiz_set: QuizSet) -> dict:
    """Return a structured, compact quiz export dictionary."""
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "artifact_type": "quiz",
        "metadata": _metadata(quiz_set),
        "items": [asdict(item) for item in quiz_set.items],
        "source_map": _compact_sources(quiz_set.source_map),
        "warnings": quiz_set.warnings,
        "token_usage": asdict(quiz_set.token_usage),
    }


def flashcard_set_to_json_dict(flashcard_set: FlashcardSet) -> dict:
    """Return a structured, compact flashcard export dictionary."""
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "artifact_type": "flashcards",
        "metadata": _metadata(flashcard_set),
        "cards": [asdict(card) for card in flashcard_set.cards],
        "source_map": _compact_sources(flashcard_set.source_map),
        "warnings": flashcard_set.warnings,
        "token_usage": asdict(flashcard_set.token_usage),
    }


def quiz_set_to_json_string(quiz_set: QuizSet) -> str:
    """Return pretty JSON for a quiz set."""
    return json.dumps(quiz_set_to_json_dict(quiz_set), indent=2, sort_keys=True)


def flashcard_set_to_json_string(flashcard_set: FlashcardSet) -> str:
    """Return pretty JSON for a flashcard set."""
    return json.dumps(flashcard_set_to_json_dict(flashcard_set), indent=2, sort_keys=True)


def artifact_export_filename(workspace_name: str, artifact_type: str, extension: str) -> str:
    """Return a safe suggested export filename."""
    clean_type = sanitize_filename(_flatten_name(artifact_type)).lower() or "artifact"
    clean_workspace = sanitize_filename(_flatten_name(workspace_name)).lower() or "workspace"
    clean_extension = extension.lower().lstrip(".")
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return f"{clean_type}_{clean_workspace}_{stamp}.{clean_extension}"


def _metadata(artifact) -> dict:
    return {
        "id": artifact.id,
        "workspace_id": artifact.workspace_id,
        "selected_document_ids": artifact.selected_document_ids,
        "mode": artifact.mode,
        "topic_or_query": artifact.topic_or_query,
        "model_name": artifact.model_name,
        "prompt_version": artifact.prompt_version,
        "created_at": artifact.created_at,
    }


def _compact_sources(source_map) -> list[dict]:
    return [source.to_compact_dict() for source in source_map]


def _flatten_name(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_")
