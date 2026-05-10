"""Markdown export helpers for learning artifacts."""

from __future__ import annotations

from mini_notebooklm_rag.learning.models import FlashcardSet, QuizSet


def quiz_set_to_markdown(quiz_set: QuizSet) -> str:
    """Return a readable Markdown quiz export."""
    lines = [
        "# Quiz",
        "",
        *_metadata_lines(quiz_set),
        "",
        "## Questions",
        "",
    ]
    for index, item in enumerate(quiz_set.items, start=1):
        lines.extend(
            [
                f"### {index}. {item.question}",
                "",
                *[
                    f"- {'ABCD'[option_index]}. {option}"
                    for option_index, option in enumerate(item.options)
                ],
                "",
                f"**Answer:** {'ABCD'[item.correct_index]}",
                "",
                f"**Explanation:** {item.explanation}",
                "",
                f"**Sources:** {', '.join(item.source_markers)}",
                "",
            ]
        )
    lines.extend(_warnings_lines(quiz_set.warnings))
    lines.extend(_source_lines(quiz_set.source_map))
    return "\n".join(lines).rstrip() + "\n"


def flashcard_set_to_markdown(flashcard_set: FlashcardSet) -> str:
    """Return a readable Markdown flashcard export."""
    lines = [
        "# Flashcards",
        "",
        *_metadata_lines(flashcard_set),
        "",
        "## Cards",
        "",
    ]
    for index, card in enumerate(flashcard_set.cards, start=1):
        lines.extend(
            [
                f"### {index}. {card.front}",
                "",
                f"**Back:** {card.back}",
                "",
            ]
        )
        if card.hint:
            lines.extend([f"**Hint:** {card.hint}", ""])
        lines.extend([f"**Sources:** {', '.join(card.source_markers)}", ""])
    lines.extend(_warnings_lines(flashcard_set.warnings))
    lines.extend(_source_lines(flashcard_set.source_map))
    return "\n".join(lines).rstrip() + "\n"


def _metadata_lines(artifact) -> list[str]:
    return [
        "## Metadata",
        "",
        f"- Workspace ID: `{artifact.workspace_id}`",
        f"- Selected documents: {', '.join(artifact.selected_document_ids)}",
        f"- Mode: `{artifact.mode}`",
        f"- Topic/query: {artifact.topic_or_query}",
        f"- Model: `{artifact.model_name}`",
        f"- Prompt version: `{artifact.prompt_version}`",
        f"- Created at: `{artifact.created_at}`",
    ]


def _warnings_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["## Warnings", "", *[f"- {warning}" for warning in warnings], ""]


def _source_lines(source_map) -> list[str]:
    lines = ["## Sources", ""]
    for source in source_map:
        lines.append(f"- [{source.source_id}] {source.citation}")
        lines.append(f"  - Document ID: `{source.document_id}`")
        lines.append(f"  - Chunk ID: `{source.chunk_id}`")
    return lines
