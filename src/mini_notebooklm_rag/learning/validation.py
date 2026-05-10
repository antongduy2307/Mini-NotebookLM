"""Validation and normalization for learning artifacts."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

from mini_notebooklm_rag.learning.models import Flashcard, QuizItem
from mini_notebooklm_rag.qa.source_mapping import SourceReference


@dataclass(frozen=True)
class QuizValidationResult:
    """Validated quiz items and warnings."""

    items: list[QuizItem]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FlashcardValidationResult:
    """Validated flashcards and warnings."""

    cards: list[Flashcard]
    warnings: list[str] = field(default_factory=list)


def validate_quiz_payload(
    payload: dict[str, Any],
    source_references: list[SourceReference],
    requested_count: int,
) -> QuizValidationResult:
    """Validate quiz JSON payload into typed items."""
    warnings = _payload_warnings(payload)
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return QuizValidationResult(
            items=[],
            warnings=[*warnings, "Quiz payload missing items list."],
        )

    known_markers = _known_markers(source_references)
    items: list[QuizItem] = []
    seen_questions: set[str] = set()
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            warnings.append(f"Quiz item {index} was not an object and was rejected.")
            continue
        item, item_warnings = _validate_quiz_item(raw_item, index, known_markers)
        warnings.extend(item_warnings)
        if item is None:
            continue
        normalized = _normalize_for_dedupe(item.question)
        if normalized in seen_questions:
            warnings.append(f"Duplicate quiz question rejected: {item.question}")
            continue
        seen_questions.add(normalized)
        items.append(item)

    if not items:
        warnings.append("No valid grounded quiz items remained after validation.")
    elif len(items) < requested_count:
        warnings.append(
            f"Only {len(items)} valid quiz item(s) remained from {requested_count} requested."
        )
    return QuizValidationResult(items=items, warnings=warnings)


def validate_flashcard_payload(
    payload: dict[str, Any],
    source_references: list[SourceReference],
    requested_count: int,
) -> FlashcardValidationResult:
    """Validate flashcard JSON payload into typed cards."""
    warnings = _payload_warnings(payload)
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        return FlashcardValidationResult(
            cards=[],
            warnings=[*warnings, "Flashcard payload missing cards list."],
        )

    known_markers = _known_markers(source_references)
    cards: list[Flashcard] = []
    seen_fronts: set[str] = set()
    for index, raw_card in enumerate(raw_cards, start=1):
        if not isinstance(raw_card, dict):
            warnings.append(f"Flashcard {index} was not an object and was rejected.")
            continue
        card, card_warnings = _validate_flashcard(raw_card, index, known_markers)
        warnings.extend(card_warnings)
        if card is None:
            continue
        normalized = _normalize_for_dedupe(card.front)
        if normalized in seen_fronts:
            warnings.append(f"Duplicate flashcard rejected: {card.front}")
            continue
        seen_fronts.add(normalized)
        cards.append(card)

    if not cards:
        warnings.append("No valid grounded flashcards remained after validation.")
    elif len(cards) < requested_count:
        warnings.append(
            f"Only {len(cards)} valid flashcard(s) remained from {requested_count} requested."
        )
    return FlashcardValidationResult(cards=cards, warnings=warnings)


def _validate_quiz_item(
    raw_item: dict[str, Any],
    index: int,
    known_markers: set[str],
) -> tuple[QuizItem | None, list[str]]:
    warnings: list[str] = []
    question = _clean_string(raw_item.get("question"))
    options = raw_item.get("options")
    explanation = _clean_string(raw_item.get("explanation"))
    correct_index = raw_item.get("correct_index")

    if not question:
        return None, [f"Quiz item {index} rejected because question is empty."]
    if not isinstance(options, list) or len(options) != 4:
        return None, [f"Quiz item {index} rejected because it does not have exactly 4 options."]
    clean_options = [_clean_string(option) for option in options]
    if any(not option for option in clean_options):
        return None, [f"Quiz item {index} rejected because one or more options are empty."]
    if not isinstance(correct_index, int) or not 0 <= correct_index <= 3:
        return None, [f"Quiz item {index} rejected because correct_index is invalid."]
    if not explanation:
        return None, [f"Quiz item {index} rejected because explanation is empty."]

    markers, marker_warnings = _valid_markers(raw_item.get("source_markers"), known_markers)
    warnings.extend(f"Quiz item {index}: {warning}" for warning in marker_warnings)
    if not markers:
        return None, [*warnings, f"Quiz item {index} rejected because it has no valid sources."]

    difficulty = _clean_string(raw_item.get("difficulty")).lower()
    if difficulty and difficulty not in {"easy", "medium", "hard"}:
        warnings.append(f"Quiz item {index}: unsupported difficulty ignored.")
        difficulty = ""

    topic = _clean_string(raw_item.get("topic")) or None
    return (
        QuizItem(
            question=question,
            options=clean_options,
            correct_index=correct_index,
            explanation=explanation,
            source_markers=markers,
            difficulty=difficulty or None,
            topic=topic,
        ),
        warnings,
    )


def _validate_flashcard(
    raw_card: dict[str, Any],
    index: int,
    known_markers: set[str],
) -> tuple[Flashcard | None, list[str]]:
    warnings: list[str] = []
    front = _clean_string(raw_card.get("front"))
    back = _clean_string(raw_card.get("back"))
    if not front:
        return None, [f"Flashcard {index} rejected because front is empty."]
    if not back:
        return None, [f"Flashcard {index} rejected because back is empty."]

    markers, marker_warnings = _valid_markers(raw_card.get("source_markers"), known_markers)
    warnings.extend(f"Flashcard {index}: {warning}" for warning in marker_warnings)
    if not markers:
        return None, [*warnings, f"Flashcard {index} rejected because it has no valid sources."]

    return (
        Flashcard(
            front=front,
            back=back,
            source_markers=markers,
            hint=_clean_string(raw_card.get("hint")) or None,
            topic=_clean_string(raw_card.get("topic")) or None,
        ),
        warnings,
    )


def _valid_markers(raw_markers: Any, known_markers: set[str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if not isinstance(raw_markers, list):
        return [], ["source_markers missing or not a list."]

    valid: list[str] = []
    for marker in raw_markers:
        normalized = _normalize_marker(marker)
        if normalized is None:
            warnings.append("invalid source marker ignored.")
            continue
        if normalized not in known_markers:
            warnings.append(f"unknown source marker {normalized} ignored.")
            continue
        if normalized not in valid:
            valid.append(normalized)
    return valid, warnings


def _known_markers(source_references: list[SourceReference]) -> set[str]:
    return {f"[{source.source_id}]" for source in source_references}


def _normalize_marker(value: Any) -> str | None:
    text = _clean_string(value)
    if not text:
        return None
    match = re.fullmatch(r"\[?S(\d+)\]?", text, flags=re.I)
    if not match:
        return None
    return f"[S{int(match.group(1))}]"


def _payload_warnings(payload: dict[str, Any]) -> list[str]:
    raw_warnings = payload.get("warnings", [])
    if not isinstance(raw_warnings, list):
        return ["Payload warnings field was not a list and was ignored."]
    return [_clean_string(warning) for warning in raw_warnings if _clean_string(warning)]


def _clean_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_for_dedupe(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = normalized.strip(string.punctuation + string.whitespace)
    return re.sub(r"\s+", " ", normalized)
