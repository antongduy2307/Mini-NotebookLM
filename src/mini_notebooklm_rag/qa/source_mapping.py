"""Source ID assignment and citation marker validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from mini_notebooklm_rag.retrieval.models import RetrievedChunk

SOURCE_MARKER_RE = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    citation: str
    text: str
    source_type: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    dense_score: float
    sparse_score: float
    fused_score: float

    def to_prompt_block(self, max_chars: int | None = None) -> str:
        text = self.text
        if max_chars is not None and len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return "\n".join(
            [
                f"[{self.source_id}]",
                f"Citation: {self.citation}",
                f"Document ID: {self.document_id}",
                f"Chunk ID: {self.chunk_id}",
                "Content:",
                text,
            ]
        )

    def to_compact_dict(self) -> dict:
        data = asdict(self)
        data.pop("text", None)
        return data


def build_source_references(results: list[RetrievedChunk]) -> list[SourceReference]:
    """Assign S1/S2 source IDs in retrieval result order."""
    return [
        SourceReference(
            source_id=f"S{index + 1}",
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            citation=result.citation,
            text=result.text,
            source_type=result.source_type,
            page_start=result.page_start,
            page_end=result.page_end,
            heading_path=result.heading_path,
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
            fused_score=result.fused_score,
        )
        for index, result in enumerate(results)
    ]


def compact_source_map(source_references: list[SourceReference]) -> list[dict]:
    """Return source metadata without chunk text for SQLite persistence."""
    return [source.to_compact_dict() for source in source_references]


def find_unknown_source_markers(answer: str, source_references: list[SourceReference]) -> list[str]:
    """Return markers in answer text that are not present in the source map."""
    known = {f"[{source.source_id}]" for source in source_references}
    markers = {match.group(0) for match in SOURCE_MARKER_RE.finditer(answer)}
    return sorted(markers - known)


def has_source_marker(answer: str) -> bool:
    return SOURCE_MARKER_RE.search(answer) is not None
