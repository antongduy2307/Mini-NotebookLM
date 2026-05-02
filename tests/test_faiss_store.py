from __future__ import annotations

import numpy as np

from mini_notebooklm_rag.retrieval.embeddings import EmbeddingModel
from mini_notebooklm_rag.retrieval.faiss_store import FaissStore
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.repositories import ChunkRecord


class FakeModel:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "alpha" in lower:
                vectors.append([1.0, 0.0])
            elif "beta" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 1.0])
        return np.array(vectors, dtype="float32")


def _embedding_model() -> EmbeddingModel:
    return EmbeddingModel(
        model_name="fake-model",
        requested_device="cpu",
        model_factory=lambda _name, _device: FakeModel(),
    )


def _chunk(chunk_id: str, document_id: str, text: str, index: int) -> ChunkRecord:
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


def test_faiss_store_build_save_load_and_search(tmp_path) -> None:
    paths = StoragePaths(tmp_path / "storage")
    store = FaissStore(paths, _embedding_model())
    chunks = [
        _chunk("c1", "doc1", "alpha topic", 0),
        _chunk("c2", "doc2", "beta topic", 0),
    ]

    status = store.build("workspace", chunks)
    metadata = store.load_metadata("workspace")
    results = store.search(
        workspace_id="workspace",
        query="beta question",
        top_k=1,
        selected_document_ids={"doc2"},
    )

    assert status.status == "current"
    assert paths.faiss_index_path("workspace").is_file()
    assert paths.faiss_metadata_path("workspace").is_file()
    assert metadata.chunk_count == 2
    assert [position.chunk_id for position in metadata.positions] == ["c1", "c2"]
    assert [result.chunk_id for result in results] == ["c2"]


def test_faiss_store_detects_stale_index_when_chunks_change(tmp_path) -> None:
    paths = StoragePaths(tmp_path / "storage")
    store = FaissStore(paths, _embedding_model())
    chunks = [_chunk("c1", "doc1", "alpha topic", 0)]

    store.build("workspace", chunks)
    changed = [_chunk("c1", "doc1", "alpha topic changed", 0)]
    changed = [
        ChunkRecord(
            **{
                **changed[0].__dict__,
                "content_hash": "changed-hash",
            }
        )
    ]
    status = store.status("workspace", changed)

    assert status.status == "stale"
    assert "workspace chunks changed" in status.warnings


def test_faiss_store_empty_workspace_removes_index_files(tmp_path) -> None:
    paths = StoragePaths(tmp_path / "storage")
    store = FaissStore(paths, _embedding_model())
    chunks = [_chunk("c1", "doc1", "alpha topic", 0)]
    store.build("workspace", chunks)

    status = store.build("workspace", [])

    assert status.status == "empty"
    assert not paths.faiss_index_path("workspace").exists()
    assert not paths.faiss_metadata_path("workspace").exists()
