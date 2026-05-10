"""Prompt builders for learning artifact generation."""

from __future__ import annotations

from dataclasses import dataclass

from mini_notebooklm_rag.learning.models import LEARNING_PROMPT_VERSION
from mini_notebooklm_rag.qa.source_mapping import SourceReference

SOURCE_MAX_CHARS = 1_800


@dataclass(frozen=True)
class LearningPrompt:
    """Prompt text and compact metadata."""

    instructions: str
    input_text: str
    metadata: dict


def build_quiz_prompt(
    topic_or_query: str,
    source_references: list[SourceReference],
    item_count: int,
) -> LearningPrompt:
    """Build a JSON-only grounded quiz prompt."""
    instructions = "\n".join(
        [
            "You generate grounded multiple-choice quiz items from provided sources.",
            "Use only the provided source text. Do not use outside knowledge.",
            "Return strict JSON only. Do not use Markdown fences.",
            "Use only source markers that appear in the provided sources.",
            "Avoid duplicate or near-duplicate questions.",
            "Avoid ambiguous trick questions and 'all of the above' options.",
            "Each item must have exactly 4 options and one correct_index from 0 to 3.",
            "Each explanation must be concise and include source markers.",
            'If the context is insufficient, return {"items": [], "warnings": ["..."]}.',
        ]
    )
    input_text = "\n\n".join(
        [
            f"Topic or query: {topic_or_query}",
            f"Requested quiz item count: {item_count}",
            "Expected JSON shape:",
            _quiz_schema_example(),
            "Sources:",
            _source_blocks(source_references),
        ]
    )
    return LearningPrompt(
        instructions=instructions,
        input_text=input_text,
        metadata=_metadata("quiz", item_count, source_references),
    )


def build_flashcard_prompt(
    topic_or_query: str,
    source_references: list[SourceReference],
    card_count: int,
) -> LearningPrompt:
    """Build a JSON-only grounded flashcard prompt."""
    instructions = "\n".join(
        [
            "You generate grounded study flashcards from provided sources.",
            "Use only the provided source text. Do not use outside knowledge.",
            "Return strict JSON only. Do not use Markdown fences.",
            "Use only source markers that appear in the provided sources.",
            "Avoid duplicate or near-duplicate cards.",
            "Keep fronts focused and backs concise but complete.",
            "Each back must include source markers.",
            'If the context is insufficient, return {"cards": [], "warnings": ["..."]}.',
        ]
    )
    input_text = "\n\n".join(
        [
            f"Topic or query: {topic_or_query}",
            f"Requested flashcard count: {card_count}",
            "Expected JSON shape:",
            _flashcard_schema_example(),
            "Sources:",
            _source_blocks(source_references),
        ]
    )
    return LearningPrompt(
        instructions=instructions,
        input_text=input_text,
        metadata=_metadata("flashcards", card_count, source_references),
    )


def _source_blocks(source_references: list[SourceReference]) -> str:
    return "\n\n".join(
        source.to_prompt_block(max_chars=SOURCE_MAX_CHARS) for source in source_references
    )


def _metadata(
    artifact_type: str,
    requested_count: int,
    source_references: list[SourceReference],
) -> dict:
    return {
        "artifact_type": artifact_type,
        "prompt_version": LEARNING_PROMPT_VERSION,
        "requested_count": requested_count,
        "source_count": len(source_references),
        "source_character_count": sum(len(source.text) for source in source_references),
        "mode": "query",
    }


def _quiz_schema_example() -> str:
    return (
        '{"items":[{"question":"...","options":["...","...","...","..."],'
        '"correct_index":0,"explanation":"... [S1]","source_markers":["[S1]"],'
        '"difficulty":"medium","topic":"..."}],"warnings":[]}'
    )


def _flashcard_schema_example() -> str:
    return (
        '{"cards":[{"front":"...","back":"... [S1]","hint":"...",'
        '"topic":"...","source_markers":["[S1]"]}],"warnings":[]}'
    )
