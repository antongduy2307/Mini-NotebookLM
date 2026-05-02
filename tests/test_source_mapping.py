from __future__ import annotations

from mini_notebooklm_rag.qa.source_mapping import (
    build_source_references,
    compact_source_map,
    find_unknown_source_markers,
    has_source_marker,
)
from mini_notebooklm_rag.retrieval.models import RetrievedChunk


def _result(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        filename="notes.md",
        text=f"text {chunk_id}",
        source_type="markdown",
        page_start=None,
        page_end=None,
        heading_path=["Intro"],
        dense_score=1.0,
        sparse_score=0.5,
        fused_score=0.8,
        rank=rank,
        citation="notes.md > Intro",
    )


def test_source_mapping_assigns_stable_ids_and_compacts_text() -> None:
    sources = build_source_references([_result("c1", 1), _result("c2", 2)])
    compact = compact_source_map(sources)

    assert [source.source_id for source in sources] == ["S1", "S2"]
    assert compact[0]["chunk_id"] == "c1"
    assert "text" not in compact[0]


def test_source_marker_validation_detects_unknown_markers() -> None:
    sources = build_source_references([_result("c1", 1)])

    assert has_source_marker("Answer [S1]")
    assert find_unknown_source_markers("Answer [S1] and [S9]", sources) == ["[S9]"]
