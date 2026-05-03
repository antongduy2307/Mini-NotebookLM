"""Chunk grouping for direct and map-reduce document summaries."""

from __future__ import annotations

from mini_notebooklm_rag.storage.repositories import ChunkRecord, DocumentRecord
from mini_notebooklm_rag.summary.models import (
    SummaryChunkGroup,
    SummaryConfig,
    SummaryPlan,
)


def build_summary_plan(
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    config: SummaryConfig | None = None,
) -> SummaryPlan:
    """Create a bounded summary plan from stored document chunks."""
    summary_config = config or SummaryConfig()
    warnings: list[str] = []
    source_chunks = chunks
    is_partial = False

    if len(source_chunks) > summary_config.max_chunks:
        source_chunks = source_chunks[: summary_config.max_chunks]
        is_partial = True
        warnings.append(
            f"Summary used the first {summary_config.max_chunks} chunks only; "
            "the document was truncated for cost control."
        )

    source_character_count = sum(len(chunk.text) for chunk in source_chunks)
    if source_character_count <= summary_config.direct_max_chars:
        return SummaryPlan(
            groups=(_make_group("Document", source_chunks),),
            use_map_reduce=False,
            source_chunk_count=len(source_chunks),
            source_character_count=source_character_count,
            is_partial=is_partial,
            warnings=warnings,
        )

    groups = _build_groups(document, source_chunks, summary_config.map_group_max_chars)
    if len(groups) > summary_config.max_groups:
        groups = groups[: summary_config.max_groups]
        is_partial = True
        warnings.append(
            f"Summary used the first {summary_config.max_groups} chunk groups only; "
            "later content was omitted for cost control."
        )

    grouped_character_count = sum(group.source_character_count for group in groups)
    return SummaryPlan(
        groups=tuple(groups),
        use_map_reduce=True,
        source_chunk_count=sum(len(group.chunks) for group in groups),
        source_character_count=grouped_character_count,
        is_partial=is_partial,
        warnings=warnings,
    )


def _build_groups(
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    max_group_chars: int,
) -> list[SummaryChunkGroup]:
    if document.source_type == "markdown":
        return _build_markdown_groups(chunks, max_group_chars)
    return _build_ordered_groups(chunks, max_group_chars)


def _build_markdown_groups(
    chunks: list[ChunkRecord],
    max_group_chars: int,
) -> list[SummaryChunkGroup]:
    groups: list[SummaryChunkGroup] = []
    current: list[ChunkRecord] = []
    current_heading: list[str] | None = None
    current_chars = 0

    for chunk in chunks:
        heading = chunk.heading_path or ["document start"]
        should_flush = current and (
            heading != current_heading or current_chars + len(chunk.text) > max_group_chars
        )
        if should_flush:
            groups.append(_make_group(_heading_label(current_heading), current))
            current = []
            current_chars = 0
        current.append(chunk)
        current_heading = heading
        current_chars += len(chunk.text)

    if current:
        groups.append(_make_group(_heading_label(current_heading), current))
    return groups


def _build_ordered_groups(
    chunks: list[ChunkRecord],
    max_group_chars: int,
) -> list[SummaryChunkGroup]:
    groups: list[SummaryChunkGroup] = []
    current: list[ChunkRecord] = []
    current_chars = 0

    for chunk in chunks:
        if current and current_chars + len(chunk.text) > max_group_chars:
            groups.append(_make_group(_page_label(current), current))
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += len(chunk.text)

    if current:
        groups.append(_make_group(_page_label(current), current))
    return groups


def _make_group(label: str, chunks: list[ChunkRecord]) -> SummaryChunkGroup:
    text = "\n\n".join(_format_chunk_for_prompt(chunk) for chunk in chunks)
    page_values = [
        page for chunk in chunks for page in (chunk.page_start, chunk.page_end) if page is not None
    ]
    heading_path = chunks[0].heading_path if chunks and chunks[0].heading_path else None
    return SummaryChunkGroup(
        label=label,
        chunks=tuple(chunks),
        text=text,
        source_character_count=sum(len(chunk.text) for chunk in chunks),
        page_start=min(page_values) if page_values else None,
        page_end=max(page_values) if page_values else None,
        heading_path=heading_path,
    )


def _format_chunk_for_prompt(chunk: ChunkRecord) -> str:
    if chunk.source_type == "pdf":
        page_hint = _page_label([chunk])
        return f"[{page_hint}]\n{chunk.text}"
    heading = _heading_label(chunk.heading_path)
    return f"[{heading}]\n{chunk.text}"


def _heading_label(heading_path: list[str] | None) -> str:
    if not heading_path:
        return "document start"
    return " > ".join(heading_path)


def _page_label(chunks: list[ChunkRecord]) -> str:
    pages = [
        page for chunk in chunks for page in (chunk.page_start, chunk.page_end) if page is not None
    ]
    if not pages:
        return "document pages unavailable"
    page_start = min(pages)
    page_end = max(pages)
    if page_start == page_end:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"
