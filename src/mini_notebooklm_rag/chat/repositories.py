"""SQLite repository for workspace chat sessions and messages."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from mini_notebooklm_rag.chat.models import ChatMessage, ChatSession, NewChatMessage
from mini_notebooklm_rag.storage.repositories import utc_now
from mini_notebooklm_rag.storage.sqlite import connect


class ChatRepositoryError(RuntimeError):
    """Raised for chat persistence failures."""


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _json_loads(value: str | None, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    return json.loads(value)


def _session_from_row(row: sqlite3.Row) -> ChatSession:
    return ChatSession(
        id=row["id"],
        workspace_id=row["workspace_id"],
        title=row["title"],
        selected_document_ids=_json_loads(row["selected_document_ids"], []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        id=row["id"],
        workspace_id=row["workspace_id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        selected_document_ids=_json_loads(row["selected_document_ids"], None),
        original_query=row["original_query"],
        rewritten_query=row["rewritten_query"],
        answer_mode=row["answer_mode"],
        source_map=_json_loads(row["source_map"], None),
        retrieval_metadata=_json_loads(row["retrieval_metadata"], None),
        prompt_metadata=_json_loads(row["prompt_metadata"], None),
        model_name=row["model_name"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        created_at=row["created_at"],
    )


class ChatRepository:
    """CRUD operations for chat sessions and messages."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_session(
        self,
        workspace_id: str,
        selected_document_ids: list[str],
        title: str,
    ) -> ChatSession:
        now = utc_now()
        session_id = uuid.uuid4().hex
        clean_title = title.strip() or "New chat"
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    id, workspace_id, title, selected_document_ids, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    workspace_id,
                    clean_title,
                    json.dumps(selected_document_ids),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return _session_from_row(row)

    def list_sessions(self, workspace_id: str) -> list[ChatSession]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM chat_sessions
                WHERE workspace_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (workspace_id,),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def get_session(self, session_id: str) -> ChatSession | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row else None

    def update_session_documents(self, session_id: str, selected_document_ids: list[str]) -> None:
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE chat_sessions
                SET selected_document_ids = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(selected_document_ids), now, session_id),
            )

    def update_session_title(self, session_id: str, title: str) -> None:
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

    def add_message(self, message: NewChatMessage) -> ChatMessage:
        now = utc_now()
        message_id = uuid.uuid4().hex
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    id, workspace_id, session_id, role, content, selected_document_ids,
                    original_query, rewritten_query, answer_mode, source_map,
                    retrieval_metadata, prompt_metadata, model_name, input_tokens,
                    output_tokens, total_tokens, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    message.workspace_id,
                    message.session_id,
                    message.role,
                    message.content,
                    _json_dumps(message.selected_document_ids),
                    message.original_query,
                    message.rewritten_query,
                    message.answer_mode,
                    _json_dumps(message.source_map),
                    _json_dumps(message.retrieval_metadata),
                    _json_dumps(message.prompt_metadata),
                    message.model_name,
                    message.input_tokens,
                    message.output_tokens,
                    message.total_tokens,
                    now,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, message.session_id),
            )
            row = connection.execute(
                "SELECT * FROM chat_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return _message_from_row(row)

    def list_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        sql = """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """
        params: tuple = (session_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (session_id, limit)
        with connect(self.db_path) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_message_from_row(row) for row in rows]
