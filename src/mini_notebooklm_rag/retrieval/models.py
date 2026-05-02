"""Typed retrieval data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingInfo:
    model_name: str
    requested_device: str
    selected_device: str
    dimension: int | None
    normalized: bool


@dataclass(frozen=True)
class DenseCandidate:
    chunk_id: str
    document_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class SparseCandidate:
    chunk_id: str
    document_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    source_type: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    dense_score: float
    sparse_score: float
    fused_score: float
    rank: int
    citation: str


@dataclass(frozen=True)
class RetrievalTrace:
    original_query: str
    selected_document_ids: list[str]
    embedding_model: str
    embedding_device: str
    top_k: int
    dense_weight: float
    sparse_weight: float
    dense_candidates: list[DenseCandidate]
    sparse_candidates: list[SparseCandidate]
    fused_results: list[RetrievedChunk]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary without logging it automatically."""
        return asdict(self)


@dataclass(frozen=True)
class FaissPosition:
    vector_index: int
    chunk_id: str
    document_id: str


@dataclass(frozen=True)
class FaissMetadata:
    workspace_id: str
    embedding_model: str
    embedding_dimension: int
    normalized: bool
    built_at: str
    chunk_count: int
    chunk_fingerprint: str
    positions: list[FaissPosition]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "normalized": self.normalized,
            "built_at": self.built_at,
            "chunk_count": self.chunk_count,
            "chunk_fingerprint": self.chunk_fingerprint,
            "positions": [asdict(position) for position in self.positions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaissMetadata:
        return cls(
            workspace_id=str(data["workspace_id"]),
            embedding_model=str(data["embedding_model"]),
            embedding_dimension=int(data["embedding_dimension"]),
            normalized=bool(data["normalized"]),
            built_at=str(data["built_at"]),
            chunk_count=int(data["chunk_count"]),
            chunk_fingerprint=str(data["chunk_fingerprint"]),
            positions=[
                FaissPosition(
                    vector_index=int(position["vector_index"]),
                    chunk_id=str(position["chunk_id"]),
                    document_id=str(position["document_id"]),
                )
                for position in data.get("positions", [])
            ],
        )


@dataclass(frozen=True)
class IndexStatus:
    status: str
    message: str
    chunk_count: int
    indexed_chunk_count: int = 0
    metadata: FaissMetadata | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResponse:
    results: list[RetrievedChunk]
    trace: RetrievalTrace
    warnings: list[str]
