"""FAISS workspace index build, load, search, and stale checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import faiss
import numpy as np

from mini_notebooklm_rag.retrieval.embeddings import EmbeddingModel
from mini_notebooklm_rag.retrieval.models import (
    DenseCandidate,
    FaissMetadata,
    FaissPosition,
    IndexStatus,
)
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.repositories import ChunkRecord
from mini_notebooklm_rag.utils.hashing import sha256_text


class FaissIndexError(RuntimeError):
    """Raised when a workspace FAISS index cannot be used."""


def compute_chunk_fingerprint(chunks: list[ChunkRecord]) -> str:
    """Hash ordered chunk IDs and content hashes for stale-index checks."""
    basis = "\n".join(f"{chunk.id}:{chunk.content_hash}" for chunk in chunks)
    return sha256_text(basis)


class FaissStore:
    """Manage one FAISS index and metadata file per workspace."""

    def __init__(self, paths: StoragePaths, embedding_model: EmbeddingModel):
        self.paths = paths
        self.embedding_model = embedding_model

    def build(self, workspace_id: str, chunks: list[ChunkRecord]) -> IndexStatus:
        """Build and persist a workspace FAISS index from chunk records."""
        index_path = self.paths.faiss_index_path(workspace_id)
        metadata_path = self.paths.faiss_metadata_path(workspace_id)

        if not chunks:
            self.paths.remove_file_if_exists(index_path)
            self.paths.remove_file_if_exists(metadata_path)
            return IndexStatus(
                status="empty",
                message="Workspace has no chunks to index.",
                chunk_count=0,
            )

        vectors = self.embedding_model.encode([chunk.text for chunk in chunks])
        if vectors.shape[0] != len(chunks):
            raise FaissIndexError("Embedding count does not match chunk count.")
        if vectors.shape[1] == 0:
            raise FaissIndexError("Embedding model returned zero-dimensional vectors.")

        vectors = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        self.paths.indexes_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        metadata = FaissMetadata(
            workspace_id=workspace_id,
            embedding_model=self.embedding_model.model_name,
            embedding_dimension=int(vectors.shape[1]),
            normalized=self.embedding_model.normalized,
            built_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
            chunk_count=len(chunks),
            chunk_fingerprint=compute_chunk_fingerprint(chunks),
            positions=[
                FaissPosition(
                    vector_index=index_position,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                )
                for index_position, chunk in enumerate(chunks)
            ],
        )
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        return IndexStatus(
            status="current",
            message=f"Built FAISS index for {len(chunks)} chunks.",
            chunk_count=len(chunks),
            indexed_chunk_count=len(chunks),
            metadata=metadata,
        )

    def status(self, workspace_id: str, chunks: list[ChunkRecord]) -> IndexStatus:
        """Return missing/current/stale/empty status for the workspace index."""
        if not chunks:
            return IndexStatus(
                status="empty",
                message="Workspace has no chunks to index.",
                chunk_count=0,
            )

        index_path = self.paths.faiss_index_path(workspace_id)
        metadata_path = self.paths.faiss_metadata_path(workspace_id)
        if not index_path.exists() or not metadata_path.exists():
            return IndexStatus(
                status="missing",
                message="FAISS index has not been built for this workspace.",
                chunk_count=len(chunks),
            )

        try:
            metadata = self.load_metadata(workspace_id)
        except Exception as exc:
            return IndexStatus(
                status="stale",
                message=f"FAISS metadata could not be read: {exc}",
                chunk_count=len(chunks),
                warnings=("Rebuild the workspace index.",),
            )

        expected_fingerprint = compute_chunk_fingerprint(chunks)
        warnings: list[str] = []
        if metadata.workspace_id != workspace_id:
            warnings.append("metadata workspace does not match")
        if metadata.embedding_model != self.embedding_model.model_name:
            warnings.append("embedding model changed")
        if metadata.normalized != self.embedding_model.normalized:
            warnings.append("embedding normalization changed")
        if metadata.chunk_fingerprint != expected_fingerprint:
            warnings.append("workspace chunks changed")

        if warnings:
            return IndexStatus(
                status="stale",
                message="FAISS index is stale; rebuild before retrieval.",
                chunk_count=len(chunks),
                indexed_chunk_count=metadata.chunk_count,
                metadata=metadata,
                warnings=tuple(warnings),
            )

        return IndexStatus(
            status="current",
            message="FAISS index is current.",
            chunk_count=len(chunks),
            indexed_chunk_count=metadata.chunk_count,
            metadata=metadata,
        )

    def search(
        self,
        workspace_id: str,
        query: str,
        top_k: int,
        selected_document_ids: set[str] | None = None,
    ) -> list[DenseCandidate]:
        """Search the workspace FAISS index and filter selected documents."""
        index = self._load_index(workspace_id)
        metadata = self.load_metadata(workspace_id)
        if index.ntotal == 0:
            return []

        selected_document_ids = selected_document_ids or None
        search_k = min(index.ntotal, max(top_k * 5, top_k + 20))
        candidates = self._search_index(index, metadata, query, search_k, selected_document_ids)
        if len(candidates) < top_k and search_k < index.ntotal:
            candidates = self._search_index(
                index,
                metadata,
                query,
                index.ntotal,
                selected_document_ids,
            )
        return candidates[:top_k]

    def load_metadata(self, workspace_id: str) -> FaissMetadata:
        metadata_path = self.paths.faiss_metadata_path(workspace_id)
        return FaissMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))

    def _load_index(self, workspace_id: str):
        index_path = self.paths.faiss_index_path(workspace_id)
        if not index_path.exists():
            raise FaissIndexError("FAISS index is missing. Build the workspace index first.")
        return faiss.read_index(str(index_path))

    def _search_index(
        self,
        index,
        metadata: FaissMetadata,
        query: str,
        search_k: int,
        selected_document_ids: set[str] | None,
    ) -> list[DenseCandidate]:
        query_vector = self.embedding_model.encode([query])
        scores, positions = index.search(query_vector, search_k)
        positions_by_index = {position.vector_index: position for position in metadata.positions}

        candidates: list[DenseCandidate] = []
        for score, vector_index in zip(scores[0], positions[0], strict=False):
            if vector_index < 0:
                continue
            position = positions_by_index.get(int(vector_index))
            if position is None:
                continue
            if selected_document_ids and position.document_id not in selected_document_ids:
                continue
            candidates.append(
                DenseCandidate(
                    chunk_id=position.chunk_id,
                    document_id=position.document_id,
                    score=float(score),
                    rank=len(candidates) + 1,
                )
            )
        return candidates
