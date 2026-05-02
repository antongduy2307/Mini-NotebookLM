"""PDF text extraction for normal text PDFs."""

from __future__ import annotations

from pathlib import Path

import fitz

from mini_notebooklm_rag.ingestion.models import ParsedDocument, SourceBlock


class PdfParserError(RuntimeError):
    """Raised when a PDF cannot be parsed for Phase 01 ingestion."""


def parse_pdf(path: Path, filename: str) -> ParsedDocument:
    """Extract page text from a normal text PDF with 1-indexed page metadata."""
    blocks: list[SourceBlock] = []
    warnings: list[str] = []

    try:
        with fitz.open(path) as document:
            page_count = document.page_count
            for page_index in range(page_count):
                page = document.load_page(page_index)
                text = page.get_text("text").strip()
                page_number = page_index + 1
                if not text:
                    warnings.append(f"Page {page_number} has no extractable text.")
                    continue
                blocks.append(
                    SourceBlock(
                        text=text,
                        page_start=page_number,
                        page_end=page_number,
                    )
                )
    except Exception as exc:  # PyMuPDF exposes several exception types.
        raise PdfParserError(f"Could not parse PDF: {filename}") from exc

    if not blocks:
        raise PdfParserError(f"PDF has no extractable text: {filename}")

    if page_count > 100:
        warnings.append("PDF is over the 100-page MVP target; ingestion continued.")

    return ParsedDocument(
        source_type="pdf",
        filename=filename,
        blocks=blocks,
        page_count=page_count,
        warnings=tuple(warnings),
    )
