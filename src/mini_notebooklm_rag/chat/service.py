"""Chat lifecycle service."""

from __future__ import annotations

from pathlib import Path

from mini_notebooklm_rag.chat.models import ChatMessage, ChatSession, NewChatMessage
from mini_notebooklm_rag.chat.repositories import ChatRepository
from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.sqlite import initialize_database


class ChatService:
    """Coordinate chat session persistence."""

    def __init__(self, settings: Settings):
        self.paths = StoragePaths(Path(settings.app_storage_dir))
        self.paths.ensure_root()
        initialize_database(self.paths.db_path)
        self.repository = ChatRepository(self.paths.db_path)

    def create_session(
        self,
        workspace_id: str,
        selected_document_ids: list[str],
        title: str = "New chat",
    ) -> ChatSession:
        return self.repository.create_session(workspace_id, selected_document_ids, title)

    def list_sessions(self, workspace_id: str) -> list[ChatSession]:
        return self.repository.list_sessions(workspace_id)

    def get_session(self, session_id: str) -> ChatSession | None:
        return self.repository.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self.repository.delete_session(session_id)

    def list_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        return self.repository.list_messages(session_id, limit)

    def add_message(self, message: NewChatMessage) -> ChatMessage:
        return self.repository.add_message(message)

    def update_session_documents(self, session_id: str, selected_document_ids: list[str]) -> None:
        self.repository.update_session_documents(session_id, selected_document_ids)

    def maybe_title_from_question(self, session: ChatSession, question: str) -> None:
        if session.title != "New chat":
            return
        title = _title_from_question(question)
        self.repository.update_session_title(session.id, title)


def _title_from_question(question: str, max_length: int = 60) -> str:
    title = " ".join(question.strip().split())
    if not title:
        return "New chat"
    if len(title) <= max_length:
        return title
    return title[: max_length - 3].rstrip() + "..."
