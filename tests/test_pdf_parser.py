from __future__ import annotations

import fitz
import pytest

from mini_notebooklm_rag.ingestion.pdf_parser import PdfParserError, parse_pdf


def _write_pdf(path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_pdf_parser_extracts_page_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(pdf_path, ["First page text", "Second page text"])

    parsed = parse_pdf(pdf_path, "sample.pdf")

    assert parsed.source_type == "pdf"
    assert parsed.page_count == 2
    assert [block.page_start for block in parsed.blocks] == [1, 2]
    assert parsed.blocks[0].text.startswith("First page")


def test_pdf_parser_rejects_pdf_with_no_extractable_text(tmp_path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    _write_pdf(pdf_path, [""])

    with pytest.raises(PdfParserError):
        parse_pdf(pdf_path, "empty.pdf")
