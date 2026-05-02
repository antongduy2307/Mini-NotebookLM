"""Hybrid dense/sparse score fusion."""

from __future__ import annotations

from mini_notebooklm_rag.retrieval.citations import format_citation
from mini_notebooklm_rag.retrieval.models import DenseCandidate, RetrievedChunk, SparseCandidate
from mini_notebooklm_rag.storage.repositories import ChunkRecord


class HybridRetrievalError(ValueError):
    """Raised when hybrid retrieval parameters are invalid."""


def normalize_scores(scores_by_id: dict[str, float]) -> dict[str, float]:
    """Normalize non-empty score mappings to [0, 1]."""
    if not scores_by_id:
        return {}
    values = list(scores_by_id.values())
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {key: 1.0 for key in scores_by_id}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores_by_id.items()}


def fuse_results(
    chunks: list[ChunkRecord],
    dense_candidates: list[DenseCandidate],
    sparse_candidates: list[SparseCandidate],
    top_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[RetrievedChunk]:
    """Fuse dense and sparse candidates into final ranked retrieval results."""
    if top_k <= 0:
        raise HybridRetrievalError("top_k must be greater than zero.")
    if dense_weight < 0 or sparse_weight < 0:
        raise HybridRetrievalError("dense_weight and sparse_weight cannot be negative.")
    weight_sum = dense_weight + sparse_weight
    if weight_sum == 0:
        raise HybridRetrievalError("dense_weight and sparse_weight cannot both be zero.")

    dense_weight = dense_weight / weight_sum
    sparse_weight = sparse_weight / weight_sum
    dense_raw = {candidate.chunk_id: candidate.score for candidate in dense_candidates}
    sparse_raw = {candidate.chunk_id: candidate.score for candidate in sparse_candidates}
    dense_norm = normalize_scores(dense_raw)
    sparse_norm = normalize_scores(sparse_raw)
    candidate_ids = set(dense_raw) | set(sparse_raw)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    fused: list[tuple[ChunkRecord, float, float, float]] = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        dense_score = dense_norm.get(chunk_id, 0.0)
        sparse_score = sparse_norm.get(chunk_id, 0.0)
        fused_score = dense_weight * dense_score + sparse_weight * sparse_score
        fused.append((chunk, dense_score, sparse_score, fused_score))

    fused.sort(
        key=lambda item: (
            -item[3],
            -item[1],
            -item[2],
            item[0].document_id,
            item[0].chunk_index,
        )
    )
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            text=chunk.text,
            source_type=chunk.source_type,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            heading_path=chunk.heading_path,
            dense_score=dense_score,
            sparse_score=sparse_score,
            fused_score=fused_score,
            rank=index + 1,
            citation=format_citation(
                filename=chunk.filename,
                source_type=chunk.source_type,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading_path=chunk.heading_path,
            ),
        )
        for index, (chunk, dense_score, sparse_score, fused_score) in enumerate(fused[:top_k])
    ]
