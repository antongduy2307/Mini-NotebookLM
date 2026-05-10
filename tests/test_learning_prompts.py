from __future__ import annotations

from mini_notebooklm_rag.learning.prompts import build_flashcard_prompt, build_quiz_prompt
from mini_notebooklm_rag.qa.source_mapping import SourceReference


def test_quiz_prompt_requires_json_only_and_no_outside_knowledge() -> None:
    prompt = build_quiz_prompt("hybrid retrieval", _sources(), item_count=5)

    assert "Return strict JSON only" in prompt.instructions
    assert "Do not use outside knowledge" in prompt.instructions
    assert "[S1]" in prompt.input_text
    assert "hybrid retrieval" in prompt.input_text
    assert prompt.metadata["artifact_type"] == "quiz"


def test_flashcard_prompt_requires_json_only_and_no_outside_knowledge() -> None:
    prompt = build_flashcard_prompt("citations", _sources(), card_count=10)

    assert "Return strict JSON only" in prompt.instructions
    assert "Do not use outside knowledge" in prompt.instructions
    assert "[S1]" in prompt.input_text
    assert prompt.metadata["artifact_type"] == "flashcards"


def _sources() -> list[SourceReference]:
    return [
        SourceReference(
            source_id="S1",
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="sample.md",
            citation="sample.md > Intro",
            text="Citations map answers to source chunks.",
            source_type="markdown",
            page_start=None,
            page_end=None,
            heading_path=["Intro"],
            dense_score=0.9,
            sparse_score=0.8,
            fused_score=0.85,
        )
    ]
