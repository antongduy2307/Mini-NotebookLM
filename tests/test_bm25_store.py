from __future__ import annotations

from mini_notebooklm_rag.retrieval.bm25_store import BM25Store, tokenize
from mini_notebooklm_rag.storage.repositories import ChunkRecord


def _chunk(chunk_id: str, document_id: str, text: str, index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        workspace_id="workspace",
        document_id=document_id,
        chunk_index=index,
        source_type="markdown",
        filename=f"{document_id}.md",
        text=text,
        page_start=None,
        page_end=None,
        heading_path=["Intro"],
        approximate_token_count=5,
        content_hash=f"hash-{chunk_id}",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_tokenize_is_deterministic_and_english_first() -> None:
    assert tokenize("Apple, banana-42!") == ["apple", "banana", "42"]


def test_bm25_search_filters_selected_documents() -> None:
    store = BM25Store.from_chunks(
        [
            _chunk("c1", "doc1", "zebra apple", 0),
            _chunk("c2", "doc2", "zebra banana", 0),
        ]
    )

    results = store.search("zebra", top_k=5, selected_document_ids={"doc2"})

    assert [result.chunk_id for result in results] == ["c2"]


def test_bm25_returns_empty_for_unmatched_query() -> None:
    store = BM25Store.from_chunks([_chunk("c1", "doc1", "zebra apple")])

    assert store.search("missing", top_k=5) == []
