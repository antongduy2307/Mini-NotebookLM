"""Retrieval orchestration service for Phase 02."""

from __future__ import annotations

from pathlib import Path

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.retrieval.bm25_store import BM25Store
from mini_notebooklm_rag.retrieval.embeddings import EmbeddingModel
from mini_notebooklm_rag.retrieval.faiss_store import FaissIndexError, FaissStore
from mini_notebooklm_rag.retrieval.hybrid import fuse_results
from mini_notebooklm_rag.retrieval.models import (
    EmbeddingInfo,
    IndexStatus,
    RetrievalResponse,
    RetrievalTrace,
)
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.repositories import ChunkRecord, DocumentRepository
from mini_notebooklm_rag.storage.sqlite import initialize_database

MAX_SELECTED_DOCUMENTS = 3


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot run with the requested inputs."""


class RetrievalService:
    """Coordinate dense, sparse, and hybrid retrieval without UI/SQL coupling."""

    def __init__(
        self,
        settings: Settings,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.settings = settings
        self.paths = StoragePaths(Path(settings.app_storage_dir))
        self.paths.ensure_root()
        initialize_database(self.paths.db_path)
        self.documents = DocumentRepository(self.paths.db_path)
        self.embedding_model = embedding_model or EmbeddingModel(
            model_name=settings.embedding_model_name,
            requested_device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        self.faiss_store = FaissStore(self.paths, self.embedding_model)

    @property
    def embedding_info(self) -> EmbeddingInfo:
        return self.embedding_model.info

    def index_status(self, workspace_id: str) -> IndexStatus:
        chunks = self.documents.list_chunks_for_workspace(workspace_id)
        return self.faiss_store.status(workspace_id, chunks)

    def rebuild_index(self, workspace_id: str) -> IndexStatus:
        chunks = self.documents.list_chunks_for_workspace(workspace_id)
        return self.faiss_store.build(workspace_id, chunks)

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        selected_document_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> RetrievalResponse:
        query = query.strip()
        _validate_retrieval_params(top_k, dense_weight, sparse_weight)
        if not query:
            raise RetrievalError("Enter a retrieval query.")
        if len(selected_document_ids) > MAX_SELECTED_DOCUMENTS:
            raise RetrievalError("Select at most 3 documents for retrieval.")
        if not selected_document_ids:
            return self._empty_response(
                query=query,
                selected_document_ids=[],
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                warnings=["Select at least one document before running retrieval."],
            )

        selected_set = set(selected_document_ids)
        workspace_chunks = self.documents.list_chunks_for_workspace(workspace_id)
        selected_chunks = [chunk for chunk in workspace_chunks if chunk.document_id in selected_set]
        if not selected_chunks:
            return self._empty_response(
                query=query,
                selected_document_ids=selected_document_ids,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                warnings=["Selected documents have no chunks."],
            )

        status = self.faiss_store.status(workspace_id, workspace_chunks)
        if status.status != "current":
            return self._empty_response(
                query=query,
                selected_document_ids=selected_document_ids,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                warnings=[status.message, *status.warnings],
            )

        candidate_k = min(len(workspace_chunks), max(top_k * 5, top_k + 20))
        try:
            dense_candidates = self.faiss_store.search(
                workspace_id=workspace_id,
                query=query,
                top_k=candidate_k,
                selected_document_ids=selected_set,
            )
        except FaissIndexError as exc:
            return self._empty_response(
                query=query,
                selected_document_ids=selected_document_ids,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                warnings=[str(exc)],
            )

        sparse_candidates = BM25Store.from_chunks(workspace_chunks).search(
            query=query,
            top_k=candidate_k,
            selected_document_ids=selected_set,
        )
        candidate_ids = {candidate.chunk_id for candidate in dense_candidates} | {
            candidate.chunk_id for candidate in sparse_candidates
        }
        candidate_chunks = self._chunks_for_fusion(selected_chunks, candidate_ids)
        fused_results = fuse_results(
            chunks=candidate_chunks,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        warnings: list[str] = []
        if not fused_results:
            warnings.append("No retrieval results matched the selected documents.")

        trace = RetrievalTrace(
            original_query=query,
            selected_document_ids=selected_document_ids,
            embedding_model=self.embedding_model.model_name,
            embedding_device=self.embedding_model.selected_device,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            fused_results=fused_results,
            warnings=warnings,
        )
        return RetrievalResponse(results=fused_results, trace=trace, warnings=warnings)

    def _chunks_for_fusion(
        self,
        selected_chunks: list[ChunkRecord],
        candidate_ids: set[str],
    ) -> list[ChunkRecord]:
        if not candidate_ids:
            return []
        return [chunk for chunk in selected_chunks if chunk.id in candidate_ids]

    def _empty_response(
        self,
        query: str,
        selected_document_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        warnings: list[str],
    ) -> RetrievalResponse:
        trace = RetrievalTrace(
            original_query=query,
            selected_document_ids=selected_document_ids,
            embedding_model=self.embedding_model.model_name,
            embedding_device=self.embedding_model.selected_device,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            dense_candidates=[],
            sparse_candidates=[],
            fused_results=[],
            warnings=warnings,
        )
        return RetrievalResponse(results=[], trace=trace, warnings=warnings)


def _validate_retrieval_params(top_k: int, dense_weight: float, sparse_weight: float) -> None:
    if top_k <= 0:
        raise RetrievalError("top_k must be greater than zero.")
    if dense_weight < 0 or sparse_weight < 0:
        raise RetrievalError("dense_weight and sparse_weight cannot be negative.")
    if dense_weight + sparse_weight == 0:
        raise RetrievalError("dense_weight and sparse_weight cannot both be zero.")
