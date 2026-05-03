"""SQLite initialization and connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workspaces_created_at
ON workspaces(created_at);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'markdown')),
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    UNIQUE (workspace_id, content_hash),
    UNIQUE (workspace_id, stored_filename)
);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_id
ON documents(workspace_id);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_hash
ON documents(workspace_id, content_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'markdown')),
    filename TEXT NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    heading_path TEXT,
    approximate_token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id
ON chunks(workspace_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_workspace_document
ON chunks(workspace_id, document_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title TEXT NOT NULL,
    selected_document_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_workspace_updated
ON chat_sessions(workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    selected_document_ids TEXT,
    original_query TEXT,
    rewritten_query TEXT,
    answer_mode TEXT,
    source_map TEXT,
    retrieval_metadata TEXT,
    prompt_metadata TEXT,
    model_name TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
ON chat_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace_created
ON chat_messages(workspace_id, created_at);

CREATE TABLE IF NOT EXISTS document_summaries (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_content_hash TEXT NOT NULL,
    summary_mode TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    source_chunk_count INTEGER NOT NULL,
    source_character_count INTEGER NOT NULL,
    is_partial INTEGER NOT NULL DEFAULT 0,
    warnings TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE (
        document_id,
        document_content_hash,
        summary_mode,
        model_name,
        prompt_version,
        config_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_document_summaries_workspace_document
ON document_summaries(workspace_id, document_id);

CREATE INDEX IF NOT EXISTS idx_document_summaries_cache_key
ON document_summaries(
    document_id,
    document_content_hash,
    summary_mode,
    model_name,
    prompt_version,
    config_hash
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with row mapping and foreign keys enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: Path) -> None:
    """Create application tables and indexes if they do not exist."""
    with connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
