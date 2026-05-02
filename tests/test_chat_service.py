from __future__ import annotations

from mini_notebooklm_rag.chat.service import ChatService
from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.storage.repositories import WorkspaceRepository
from mini_notebooklm_rag.storage.sqlite import initialize_database


def test_chat_service_titles_from_first_question(tmp_path) -> None:
    settings = Settings(_env_file=None, app_storage_dir=str(tmp_path / "storage"))
    db_path = tmp_path / "storage" / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    service = ChatService(settings)
    session = service.create_session(workspace.id, ["doc1"])

    service.maybe_title_from_question(session, "What does the document say about retrieval?")

    updated = service.get_session(session.id)
    assert updated is not None
    assert updated.title == "What does the document say about retrieval?"
