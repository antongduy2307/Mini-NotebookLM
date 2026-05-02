"""Heading-aware Markdown parsing."""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from mini_notebooklm_rag.ingestion.models import ParsedDocument, SourceBlock

DOCUMENT_START = "document start"


class MarkdownParserError(RuntimeError):
    """Raised when Markdown cannot be parsed for Phase 01 ingestion."""


def parse_markdown(path: Path, filename: str) -> ParsedDocument:
    """Parse Markdown into text blocks with nearest heading hierarchy."""
    try:
        markdown = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MarkdownParserError(f"Markdown must be UTF-8: {filename}") from exc

    parser = MarkdownIt("commonmark")
    tokens = parser.parse(markdown)

    blocks: list[SourceBlock] = []
    heading_stack: list[str] = []
    pending_heading_level: int | None = None
    capture_inline = False

    for token in tokens:
        if token.type == "heading_open":
            pending_heading_level = int(token.tag[1:])
            continue

        if token.type == "inline":
            text = token.content.strip()
            if not text:
                continue
            if pending_heading_level is not None:
                heading_stack = heading_stack[: pending_heading_level - 1]
                heading_stack.append(text)
                pending_heading_level = None
                continue
            if capture_inline or token.map is not None:
                blocks.append(
                    SourceBlock(
                        text=text,
                        heading_path=heading_stack[:] if heading_stack else [DOCUMENT_START],
                    )
                )
            continue

        if token.type in {"paragraph_open", "list_item_open", "blockquote_open"}:
            capture_inline = True
            continue

        if token.type in {"paragraph_close", "list_item_close", "blockquote_close"}:
            capture_inline = False
            continue

        if token.type in {"fence", "code_block"}:
            text = token.content.strip()
            if text:
                blocks.append(
                    SourceBlock(
                        text=text,
                        heading_path=heading_stack[:] if heading_stack else [DOCUMENT_START],
                    )
                )

    if not blocks and markdown.strip():
        blocks.append(
            SourceBlock(
                text=markdown.strip(),
                heading_path=heading_stack[:] if heading_stack else [DOCUMENT_START],
            )
        )

    return ParsedDocument(
        source_type="markdown",
        filename=filename,
        blocks=blocks,
        page_count=None,
    )
