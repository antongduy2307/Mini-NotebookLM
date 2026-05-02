from __future__ import annotations

import numpy as np
import pytest

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.retrieval.embeddings import EmbeddingModel
from mini_notebooklm_rag.retrieval.service import (
    MAX_SELECTED_DOCUMENTS,
    RetrievalError,
    RetrievalService,
)
from mini_notebooklm_rag.storage.repositories import (
    DocumentRepository,
    NewChunkRecord,
    NewDocumentRecord,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import initialize_database


class FakeModel:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "alpha" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in lower:
                vectors.append([0.0, 1.0, 0.0])
            elif "gamma" in lower:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 1.0, 0.0])
        return np.array(vectors, dtype="float32")


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, app_storage_dir=str(tmp_path / "storage"))


def _embedding_model() -> EmbeddingModel:
    return EmbeddingModel(
        model_name="fake-model",
        requested_device="cpu",
        model_factory=lambda _name, _device: FakeModel(),
    )


def _insert_document_with_chunk(
    repository: DocumentRepository,
    workspace_id: str,
    document_id: str,
    text: str,
) -> None:
    repository.insert_with_chunks(
        NewDocumentRecord(
            id=document_id,
            workspace_id=workspace_id,
            display_name=f"{document_id}.md",
            stored_filename=f"{document_id}__notes.md",
            relative_path=f"workspaces/{workspace_id}/documents/{document_id}__notes.md",
            source_type="markdown",
            content_hash=f"hash-{document_id}",
            size_bytes=len(text),
            page_count=None,
        ),
        [
            NewChunkRecord(
                id=f"chunk-{document_id}",
                workspace_id=workspace_id,
                document_id=document_id,
                chunk_index=0,
                source_type="markdown",
                filename=f"{document_id}.md",
                text=text,
                page_start=None,
                page_end=None,
                heading_path=["Intro"],
                approximate_token_count=5,
                content_hash=f"chunk-hash-{document_id}",
            )
        ],
    )


def _seed_workspace(tmp_path):
    settings = _settings(tmp_path)
    db_path = tmp_path / "storage" / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    documents = DocumentRepository(db_path)
    _insert_document_with_chunk(documents, workspace.id, "doc1", "alpha topic")
    _insert_document_with_chunk(documents, workspace.id, "doc2", "beta topic")
    return settings, workspace, documents


def test_retrieval_service_reports_missing_index_without_crashing(tmp_path) -> None:
    settings, workspace, _documents = _seed_workspace(tmp_path)
    service = RetrievalService(settings, embedding_model=_embedding_model())

    response = service.retrieve(
        workspace_id=workspace.id,
        query="alpha",
        selected_document_ids=["doc1"],
        top_k=3,
        dense_weight=0.65,
        sparse_weight=0.35,
    )

    assert response.results == []
    assert "FAISS index has not been built" in response.warnings[0]


def test_retrieval_service_rebuilds_and_retrieves_selected_document(tmp_path) -> None:
    settings, workspace, _documents = _seed_workspace(tmp_path)
    service = RetrievalService(settings, embedding_model=_embedding_model())

    build_status = service.rebuild_index(workspace.id)
    response = service.retrieve(
        workspace_id=workspace.id,
        query="beta",
        selected_document_ids=["doc2"],
        top_k=3,
        dense_weight=0.65,
        sparse_weight=0.35,
    )

    assert build_status.status == "current"
    assert [result.document_id for result in response.results] == ["doc2"]
    assert response.results[0].citation == "doc2.md > Intro"
    assert response.trace.embedding_model == "fake-model"
    assert response.trace.embedding_device == "cpu"


def test_retrieval_service_enforces_selected_document_limit(tmp_path) -> None:
    settings, workspace, _documents = _seed_workspace(tmp_path)
    service = RetrievalService(settings, embedding_model=_embedding_model())

    with pytest.raises(RetrievalError, match=f"at most {MAX_SELECTED_DOCUMENTS}"):
        service.retrieve(
            workspace_id=workspace.id,
            query="alpha",
            selected_document_ids=["a", "b", "c", "d"],
            top_k=3,
            dense_weight=0.65,
            sparse_weight=0.35,
        )


def test_retrieval_service_detects_stale_index_after_chunk_change(tmp_path) -> None:
    settings, workspace, documents = _seed_workspace(tmp_path)
    service = RetrievalService(settings, embedding_model=_embedding_model())
    service.rebuild_index(workspace.id)

    _insert_document_with_chunk(documents, workspace.id, "doc3", "gamma topic")
    status = service.index_status(workspace.id)

    assert status.status == "stale"


def test_retrieval_service_rejects_zero_weights(tmp_path) -> None:
    settings, workspace, _documents = _seed_workspace(tmp_path)
    service = RetrievalService(settings, embedding_model=_embedding_model())

    with pytest.raises(RetrievalError, match="cannot both be zero"):
        service.retrieve(
            workspace_id=workspace.id,
            query="alpha",
            selected_document_ids=["doc1"],
            top_k=3,
            dense_weight=0.0,
            sparse_weight=0.0,
        )
