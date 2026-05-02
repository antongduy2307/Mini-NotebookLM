"""Repository layer for Phase 01 SQLite metadata."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mini_notebooklm_rag.storage.sqlite import connect
from mini_notebooklm_rag.utils.filenames import normalize_workspace_name


class RepositoryError(RuntimeError):
    """Base repository error."""


class DuplicateWorkspaceError(RepositoryError):
    """Raised when a workspace name already exists."""


class DuplicateDocumentError(RepositoryError):
    """Raised when a document hash already exists in a workspace."""


def utc_now() -> str:
    """Return a UTC ISO timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    name_normalized: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    workspace_id: str
    display_name: str
    stored_filename: str
    relative_path: str
    source_type: str
    content_hash: str
    size_bytes: int
    page_count: int | None
    chunk_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class NewDocumentRecord:
    id: str
    workspace_id: str
    display_name: str
    stored_filename: str
    relative_path: str
    source_type: str
    content_hash: str
    size_bytes: int
    page_count: int | None


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    workspace_id: str
    document_id: str
    chunk_index: int
    source_type: str
    filename: str
    text: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    approximate_token_count: int
    content_hash: str
    created_at: str


@dataclass(frozen=True)
class NewChunkRecord:
    id: str
    workspace_id: str
    document_id: str
    chunk_index: int
    source_type: str
    filename: str
    text: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    approximate_token_count: int
    content_hash: str


def _workspace_from_row(row: sqlite3.Row) -> Workspace:
    return Workspace(
        id=row["id"],
        name=row["name"],
        name_normalized=row["name_normalized"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        display_name=row["display_name"],
        stored_filename=row["stored_filename"],
        relative_path=row["relative_path"],
        source_type=row["source_type"],
        content_hash=row["content_hash"],
        size_bytes=row["size_bytes"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
    heading_path = json.loads(row["heading_path"]) if row["heading_path"] else None
    return ChunkRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        source_type=row["source_type"],
        filename=row["filename"],
        text=row["text"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        heading_path=heading_path,
        approximate_token_count=row["approximate_token_count"],
        content_hash=row["content_hash"],
        created_at=row["created_at"],
    )


class WorkspaceRepository:
    """CRUD operations for workspaces."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create(self, name: str) -> Workspace:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Workspace name is required.")

        now = utc_now()
        workspace = Workspace(
            id=uuid.uuid4().hex,
            name=clean_name,
            name_normalized=normalize_workspace_name(clean_name),
            created_at=now,
            updated_at=now,
        )
        try:
            with connect(self.db_path) as connection:
                connection.execute(
                    """
                    INSERT INTO workspaces (id, name, name_normalized, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        workspace.id,
                        workspace.name,
                        workspace.name_normalized,
                        workspace.created_at,
                        workspace.updated_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateWorkspaceError(f"Workspace already exists: {clean_name}") from exc
        return workspace

    def list(self) -> list[Workspace]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM workspaces ORDER BY name_normalized ASC"
            ).fetchall()
        return [_workspace_from_row(row) for row in rows]

    def get(self, workspace_id: str) -> Workspace | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
            ).fetchone()
        return _workspace_from_row(row) if row else None

    def delete(self, workspace_id: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))


class DocumentRepository:
    """CRUD operations for documents and chunks."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def find_by_hash(self, workspace_id: str, content_hash: str) -> DocumentRecord | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE workspace_id = ? AND content_hash = ?
                """,
                (workspace_id, content_hash),
            ).fetchone()
        return _document_from_row(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _document_from_row(row) if row else None

    def list_for_workspace(self, workspace_id: str) -> list[DocumentRecord]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE workspace_id = ?
                ORDER BY created_at DESC, display_name ASC
                """,
                (workspace_id,),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def insert_with_chunks(
        self,
        document: NewDocumentRecord,
        chunks: list[NewChunkRecord],
    ) -> DocumentRecord:
        now = utc_now()
        try:
            with connect(self.db_path) as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, workspace_id, display_name, stored_filename, relative_path,
                        source_type, content_hash, size_bytes, page_count, chunk_count,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.workspace_id,
                        document.display_name,
                        document.stored_filename,
                        document.relative_path,
                        document.source_type,
                        document.content_hash,
                        document.size_bytes,
                        document.page_count,
                        len(chunks),
                        now,
                        now,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        id, workspace_id, document_id, chunk_index, source_type, filename,
                        text, page_start, page_end, heading_path, approximate_token_count,
                        content_hash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.workspace_id,
                            chunk.document_id,
                            chunk.chunk_index,
                            chunk.source_type,
                            chunk.filename,
                            chunk.text,
                            chunk.page_start,
                            chunk.page_end,
                            json.dumps(chunk.heading_path) if chunk.heading_path else None,
                            chunk.approximate_token_count,
                            chunk.content_hash,
                            now,
                        )
                        for chunk in chunks
                    ],
                )
                row = connection.execute(
                    "SELECT * FROM documents WHERE id = ?", (document.id,)
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise DuplicateDocumentError(
                f"Document already exists in workspace: {document.display_name}"
            ) from exc
        return _document_from_row(row)

    def list_chunks(self, document_id: str) -> list[ChunkRecord]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def delete(self, document_id: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
