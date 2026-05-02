from __future__ import annotations

from mini_notebooklm_rag.retrieval.citations import format_citation


def test_pdf_single_page_citation() -> None:
    assert format_citation("paper.pdf", "pdf", page_start=5, page_end=5) == "paper.pdf, p. 5"


def test_pdf_page_range_citation() -> None:
    assert format_citation("paper.pdf", "pdf", page_start=5, page_end=6) == "paper.pdf, pp. 5-6"


def test_markdown_heading_citation() -> None:
    assert (
        format_citation("notes.md", "markdown", heading_path=["Parent", "Child"])
        == "notes.md > Parent > Child"
    )


def test_markdown_document_start_citation() -> None:
    assert format_citation("notes.md", "markdown") == "notes.md > document start"
