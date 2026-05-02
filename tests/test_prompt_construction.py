from __future__ import annotations

from mini_notebooklm_rag.qa.prompts import (
    NOT_FOUND_MESSAGE,
    build_grounded_qa_prompt,
    build_outside_knowledge_prompt,
    build_query_rewrite_prompt,
)
from mini_notebooklm_rag.qa.source_mapping import SourceReference


def _source() -> SourceReference:
    return SourceReference(
        source_id="S1",
        chunk_id="chunk1",
        document_id="doc1",
        filename="notes.md",
        citation="notes.md > Intro",
        text="This is source text.",
        source_type="markdown",
        page_start=None,
        page_end=None,
        heading_path=["Intro"],
        dense_score=1.0,
        sparse_score=0.5,
        fused_score=0.8,
    )


def test_grounded_prompt_includes_sources_and_not_found_instruction() -> None:
    prompt = build_grounded_qa_prompt("What is covered?", [_source()])

    assert "only the provided sources" in prompt.instructions
    assert NOT_FOUND_MESSAGE in prompt.instructions
    assert "[S1]" in prompt.input_text
    assert "This is source text." in prompt.input_text
    assert prompt.metadata["source_count"] == 1


def test_outside_knowledge_prompt_requires_section_separation() -> None:
    prompt = build_outside_knowledge_prompt("What else?", [_source()])

    assert "From your documents:" in prompt.instructions
    assert "Outside the selected documents:" in prompt.instructions
    assert "must not use source IDs" in prompt.instructions


def test_query_rewrite_prompt_uses_current_history_only() -> None:
    prompt = build_query_rewrite_prompt("What about it?", [], ["doc1"])

    assert "Selected document IDs: doc1" in prompt.input_text
    assert "No prior messages." in prompt.input_text
    assert "Return only JSON" in prompt.instructions
