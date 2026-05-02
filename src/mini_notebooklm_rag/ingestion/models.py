"""Typed ingestion data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceBlock:
    text: str
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] | None = None


@dataclass(frozen=True)
class ParsedDocument:
    source_type: str
    filename: str
    blocks: list[SourceBlock]
    page_count: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    chunk_index: int
    source_type: str
    filename: str
    text: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    approximate_token_count: int
    content_hash: str
