from __future__ import annotations

import pytest

from mini_notebooklm_rag.retrieval.hybrid import (
    HybridRetrievalError,
    fuse_results,
    normalize_scores,
)
from mini_notebooklm_rag.retrieval.models import DenseCandidate, SparseCandidate
from mini_notebooklm_rag.storage.repositories import ChunkRecord


def _chunk(chunk_id: str, document_id: str, text: str, index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        workspace_id="workspace",
        document_id=document_id,
        chunk_index=index,
        source_type="pdf",
        filename="paper.pdf",
        text=text,
        page_start=1,
        page_end=1,
        heading_path=None,
        approximate_token_count=5,
        content_hash=f"hash-{chunk_id}",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_normalize_scores_handles_equal_values() -> None:
    assert normalize_scores({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}


def test_fuse_results_normalizes_weights_and_missing_scores() -> None:
    results = fuse_results(
        chunks=[_chunk("c1", "doc1", "alpha", 0), _chunk("c2", "doc1", "beta", 1)],
        dense_candidates=[DenseCandidate("c1", "doc1", 0.9, 1)],
        sparse_candidates=[SparseCandidate("c2", "doc1", 5.0, 1)],
        top_k=2,
        dense_weight=2.0,
        sparse_weight=2.0,
    )

    assert [result.chunk_id for result in results] == ["c1", "c2"]
    assert results[0].dense_score == 1.0
    assert results[0].sparse_score == 0.0
    assert results[1].dense_score == 0.0
    assert results[1].sparse_score == 1.0


def test_fuse_results_rejects_zero_total_weight() -> None:
    with pytest.raises(HybridRetrievalError, match="cannot both be zero"):
        fuse_results([], [], [], top_k=1, dense_weight=0.0, sparse_weight=0.0)
