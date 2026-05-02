from __future__ import annotations

from mini_notebooklm_rag.ingestion.chunker import approximate_token_count, chunk_document
from mini_notebooklm_rag.ingestion.models import ParsedDocument, SourceBlock


def test_chunker_preserves_pdf_page_range_and_overlap() -> None:
    parsed = ParsedDocument(
        source_type="pdf",
        filename="sample.pdf",
        blocks=[
            SourceBlock(text="alpha beta gamma delta epsilon", page_start=1, page_end=1),
            SourceBlock(text="zeta eta theta iota kappa", page_start=2, page_end=2),
        ],
    )

    chunks = chunk_document(parsed, chunk_size_tokens=4, chunk_overlap_tokens=1)

    assert len(chunks) > 1
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.content_hash for chunk in chunks)


def test_chunker_preserves_markdown_heading_path() -> None:
    parsed = ParsedDocument(
        source_type="markdown",
        filename="notes.md",
        blocks=[SourceBlock(text="alpha beta gamma", heading_path=["Intro"])],
    )

    chunks = chunk_document(parsed, chunk_size_tokens=20, chunk_overlap_tokens=0)

    assert chunks[0].heading_path == ["Intro"]
    assert chunks[0].approximate_token_count == approximate_token_count("alpha beta gamma")
