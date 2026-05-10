from __future__ import annotations

from mini_notebooklm_rag.learning.validation import (
    validate_flashcard_payload,
    validate_quiz_payload,
)
from mini_notebooklm_rag.qa.source_mapping import SourceReference


def test_quiz_requires_exactly_four_options_and_valid_correct_index() -> None:
    result = validate_quiz_payload(
        {
            "items": [
                {
                    "question": "Q?",
                    "options": ["A", "B", "C"],
                    "correct_index": 0,
                    "explanation": "E [S1]",
                    "source_markers": ["[S1]"],
                },
                {
                    "question": "Q2?",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 9,
                    "explanation": "E [S1]",
                    "source_markers": ["[S1]"],
                },
            ]
        },
        _sources(),
        requested_count=2,
    )

    assert result.items == []
    assert any("exactly 4 options" in warning for warning in result.warnings)
    assert any("correct_index is invalid" in warning for warning in result.warnings)


def test_quiz_rejects_empty_fields_unknown_markers_and_deduplicates() -> None:
    result = validate_quiz_payload(
        {
            "items": [
                {
                    "question": "What is retrieval?",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "E [S1]",
                    "source_markers": ["[S1]", "[S9]"],
                },
                {
                    "question": "What is retrieval?",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 1,
                    "explanation": "E [S1]",
                    "source_markers": ["[S1]"],
                },
                {
                    "question": "",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "E [S1]",
                    "source_markers": ["[S1]"],
                },
            ]
        },
        _sources(),
        requested_count=3,
    )

    assert len(result.items) == 1
    assert result.items[0].source_markers == ["[S1]"]
    assert any("unknown source marker [S9]" in warning for warning in result.warnings)
    assert any("Duplicate quiz question rejected" in warning for warning in result.warnings)
    assert any("question is empty" in warning for warning in result.warnings)


def test_ungrounded_quiz_item_is_rejected() -> None:
    result = validate_quiz_payload(
        {
            "items": [
                {
                    "question": "Q?",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "E",
                    "source_markers": ["[S9]"],
                }
            ]
        },
        _sources(),
        requested_count=1,
    )

    assert result.items == []
    assert any("no valid sources" in warning for warning in result.warnings)


def test_flashcard_front_back_validation_markers_and_deduplication() -> None:
    result = validate_flashcard_payload(
        {
            "cards": [
                {
                    "front": "Hybrid retrieval",
                    "back": "Combines dense and sparse retrieval. [S1]",
                    "source_markers": ["[S1]", "[S9]"],
                },
                {
                    "front": " hybrid   retrieval ",
                    "back": "Duplicate. [S1]",
                    "source_markers": ["[S1]"],
                },
                {"front": "", "back": "Missing front", "source_markers": ["[S1]"]},
                {"front": "No back", "back": "", "source_markers": ["[S1]"]},
            ]
        },
        _sources(),
        requested_count=4,
    )

    assert len(result.cards) == 1
    assert result.cards[0].source_markers == ["[S1]"]
    assert any("unknown source marker [S9]" in warning for warning in result.warnings)
    assert any("Duplicate flashcard rejected" in warning for warning in result.warnings)
    assert any("front is empty" in warning for warning in result.warnings)
    assert any("back is empty" in warning for warning in result.warnings)


def _sources() -> list[SourceReference]:
    return [
        SourceReference(
            source_id="S1",
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="sample.md",
            citation="sample.md > Intro",
            text="Hybrid retrieval combines dense and sparse signals.",
            source_type="markdown",
            page_start=None,
            page_end=None,
            heading_path=["Intro"],
            dense_score=0.9,
            sparse_score=0.8,
            fused_score=0.85,
        )
    ]
