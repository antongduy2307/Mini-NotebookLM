from __future__ import annotations

import pytest

from mini_notebooklm_rag.storage.repositories import (
    DocumentRepository,
    DuplicateDocumentError,
    NewChunkRecord,
    NewDocumentRecord,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import connect, initialize_database


def test_document_insert_list_chunks_and_delete_cascades(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    repository = DocumentRepository(db_path)

    document = NewDocumentRecord(
        id="doc1",
        workspace_id=workspace.id,
        display_name="notes.md",
        stored_filename="doc1__notes.md",
        relative_path="workspaces/ws/documents/doc1__notes.md",
        source_type="markdown",
        content_hash="hash1",
        size_bytes=10,
        page_count=None,
    )
    chunk = NewChunkRecord(
        id="chunk1",
        workspace_id=workspace.id,
        document_id="doc1",
        chunk_index=0,
        source_type="markdown",
        filename="notes.md",
        text="hello",
        page_start=None,
        page_end=None,
        heading_path=["Intro"],
        approximate_token_count=2,
        content_hash="chunkhash",
    )

    inserted = repository.insert_with_chunks(document, [chunk])

    assert inserted.chunk_count == 1
    assert repository.list_for_workspace(workspace.id) == [inserted]
    assert repository.list_chunks("doc1")[0].heading_path == ["Intro"]

    repository.delete("doc1")

    assert repository.get("doc1") is None
    assert repository.list_chunks("doc1") == []


def test_duplicate_document_hash_is_rejected_per_workspace(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    repository = DocumentRepository(db_path)

    base = NewDocumentRecord(
        id="doc1",
        workspace_id=workspace.id,
        display_name="a.md",
        stored_filename="doc1__a.md",
        relative_path="workspaces/ws/documents/doc1__a.md",
        source_type="markdown",
        content_hash="samehash",
        size_bytes=10,
        page_count=None,
    )
    repository.insert_with_chunks(base, [])

    duplicate = NewDocumentRecord(
        id="doc2",
        workspace_id=workspace.id,
        display_name="b.md",
        stored_filename="doc2__b.md",
        relative_path="workspaces/ws/documents/doc2__b.md",
        source_type="markdown",
        content_hash="samehash",
        size_bytes=10,
        page_count=None,
    )

    with pytest.raises(DuplicateDocumentError):
        repository.insert_with_chunks(duplicate, [])


def test_workspace_delete_cascades_documents_and_chunks(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    workspace_repository = WorkspaceRepository(db_path)
    workspace = workspace_repository.create("Research")
    document_repository = DocumentRepository(db_path)
    document_repository.insert_with_chunks(
        NewDocumentRecord(
            id="doc1",
            workspace_id=workspace.id,
            display_name="notes.md",
            stored_filename="doc1__notes.md",
            relative_path="workspaces/ws/documents/doc1__notes.md",
            source_type="markdown",
            content_hash="hash1",
            size_bytes=10,
            page_count=None,
        ),
        [
            NewChunkRecord(
                id="chunk1",
                workspace_id=workspace.id,
                document_id="doc1",
                chunk_index=0,
                source_type="markdown",
                filename="notes.md",
                text="hello",
                page_start=None,
                page_end=None,
                heading_path=["Intro"],
                approximate_token_count=2,
                content_hash="chunkhash",
            )
        ],
    )

    workspace_repository.delete(workspace.id)

    with connect(db_path) as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert document_count == 0
    assert chunk_count == 0
