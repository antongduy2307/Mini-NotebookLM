from __future__ import annotations

from mini_notebooklm_rag.chat.models import NewChatMessage
from mini_notebooklm_rag.chat.repositories import ChatRepository
from mini_notebooklm_rag.storage.repositories import (
    DocumentRepository,
    NewChunkRecord,
    NewDocumentRecord,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import connect, initialize_database


def test_chat_repository_session_message_lifecycle(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    repository = ChatRepository(db_path)

    session = repository.create_session(workspace.id, ["doc1"], "New chat")
    message = repository.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="assistant",
            content="Answer [S1]",
            selected_document_ids=["doc1"],
            source_map=[{"source_id": "S1", "citation": "notes.md > Intro"}],
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
    )

    assert repository.list_sessions(workspace.id)[0].selected_document_ids == ["doc1"]
    assert repository.list_messages(session.id)[0] == message
    assert repository.list_messages(session.id)[0].source_map[0]["source_id"] == "S1"

    repository.delete_session(session.id)

    assert repository.list_messages(session.id) == []


def test_workspace_delete_cascades_chat(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    workspace_repository = WorkspaceRepository(db_path)
    workspace = workspace_repository.create("Research")
    repository = ChatRepository(db_path)
    session = repository.create_session(workspace.id, [], "Chat")
    repository.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="user",
            content="Question",
        )
    )

    workspace_repository.delete(workspace.id)

    assert repository.list_sessions(workspace.id) == []
    with connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
    assert count == 0


def test_document_delete_does_not_delete_historical_chat(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    documents = DocumentRepository(db_path)
    documents.insert_with_chunks(
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
    repository = ChatRepository(db_path)
    session = repository.create_session(workspace.id, ["doc1"], "Chat")
    repository.add_message(
        NewChatMessage(
            workspace_id=workspace.id,
            session_id=session.id,
            role="assistant",
            content="Historical answer [S1]",
            selected_document_ids=["doc1"],
        )
    )

    documents.delete("doc1")

    assert len(repository.list_messages(session.id)) == 1
