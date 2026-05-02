"""Dependency-free approximate chunking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from mini_notebooklm_rag.ingestion.models import DocumentChunk, ParsedDocument, SourceBlock
from mini_notebooklm_rag.utils.hashing import sha256_text

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TextUnit:
    text: str
    approx_tokens: int
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None


def approximate_token_count(text: str) -> int:
    """Estimate token count with a replaceable dependency-free heuristic."""
    total = 0
    for match in _TOKEN_RE.finditer(text):
        total += max(1, math.ceil(len(match.group(0)) / 4))
    return total


def chunk_document(
    parsed: ParsedDocument,
    chunk_size_tokens: int = 700,
    chunk_overlap_tokens: int = 120,
) -> list[DocumentChunk]:
    """Chunk parsed source blocks while preserving citation metadata."""
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than zero.")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative.")

    units = _build_units(parsed.blocks)
    chunks: list[DocumentChunk] = []
    current: list[TextUnit] = []
    current_tokens = 0

    for unit in units:
        if current and current_tokens + unit.approx_tokens > chunk_size_tokens:
            chunks.append(_make_chunk(parsed, len(chunks), current))
            current = _overlap_units(current, chunk_overlap_tokens)
            current_tokens = sum(item.approx_tokens for item in current)

        current.append(unit)
        current_tokens += unit.approx_tokens

    if current:
        chunks.append(_make_chunk(parsed, len(chunks), current))

    return chunks


def _build_units(blocks: list[SourceBlock]) -> list[TextUnit]:
    units: list[TextUnit] = []
    for block in blocks:
        for match in _TOKEN_RE.finditer(block.text):
            text = match.group(0)
            units.append(
                TextUnit(
                    text=text,
                    approx_tokens=max(1, math.ceil(len(text) / 4)),
                    page_start=block.page_start,
                    page_end=block.page_end,
                    heading_path=block.heading_path,
                )
            )
    return units


def _overlap_units(units: list[TextUnit], overlap_tokens: int) -> list[TextUnit]:
    if overlap_tokens <= 0:
        return []

    selected: list[TextUnit] = []
    total = 0
    for unit in reversed(units):
        selected.append(unit)
        total += unit.approx_tokens
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def _make_chunk(parsed: ParsedDocument, chunk_index: int, units: list[TextUnit]) -> DocumentChunk:
    text = " ".join(unit.text for unit in units).strip()
    page_values = [unit.page_start for unit in units if unit.page_start is not None]
    page_end_values = [unit.page_end for unit in units if unit.page_end is not None]
    heading_path = next((unit.heading_path for unit in units if unit.heading_path), None)
    hash_basis = "|".join(
        [
            parsed.source_type,
            parsed.filename,
            str(min(page_values) if page_values else ""),
            str(max(page_end_values) if page_end_values else ""),
            "/".join(heading_path or []),
            text,
        ]
    )
    return DocumentChunk(
        chunk_index=chunk_index,
        source_type=parsed.source_type,
        filename=parsed.filename,
        text=text,
        page_start=min(page_values) if page_values else None,
        page_end=max(page_end_values) if page_end_values else None,
        heading_path=heading_path,
        approximate_token_count=sum(unit.approx_tokens for unit in units),
        content_hash=sha256_text(hash_basis),
    )
