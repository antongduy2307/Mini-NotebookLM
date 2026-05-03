"""SQLite repository for per-document summary cache rows."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from mini_notebooklm_rag.storage.repositories import utc_now
from mini_notebooklm_rag.storage.sqlite import connect
from mini_notebooklm_rag.summary.models import (
    CachedSummary,
    NewCachedSummary,
    SummaryCacheKey,
)


def _summary_from_row(row: sqlite3.Row) -> CachedSummary:
    return CachedSummary(
        id=row["id"],
        workspace_id=row["workspace_id"],
        document_id=row["document_id"],
        document_content_hash=row["document_content_hash"],
        summary_mode=row["summary_mode"],
        model_name=row["model_name"],
        prompt_version=row["prompt_version"],
        config_hash=row["config_hash"],
        config_json=row["config_json"],
        summary_text=row["summary_text"],
        source_chunk_count=row["source_chunk_count"],
        source_character_count=row["source_character_count"],
        is_partial=bool(row["is_partial"]),
        warnings=json.loads(row["warnings"]),
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SummaryRepository:
    """Cache CRUD operations for document summaries."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_by_cache_key(self, key: SummaryCacheKey) -> CachedSummary | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM document_summaries
                WHERE document_id = ?
                  AND document_content_hash = ?
                  AND summary_mode = ?
                  AND model_name = ?
                  AND prompt_version = ?
                  AND config_hash = ?
                """,
                (
                    key.document_id,
                    key.document_content_hash,
                    key.summary_mode,
                    key.model_name,
                    key.prompt_version,
                    key.config_hash,
                ),
            ).fetchone()
        return _summary_from_row(row) if row else None

    def latest_for_document(
        self,
        document_id: str,
        summary_mode: str,
    ) -> CachedSummary | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM document_summaries
                WHERE document_id = ? AND summary_mode = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (document_id, summary_mode),
            ).fetchone()
        return _summary_from_row(row) if row else None

    def upsert(self, summary: NewCachedSummary) -> CachedSummary:
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO document_summaries (
                    id, workspace_id, document_id, document_content_hash, summary_mode,
                    model_name, prompt_version, config_hash, config_json, summary_text,
                    source_chunk_count, source_character_count, is_partial, warnings,
                    input_tokens, output_tokens, total_tokens, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    document_id,
                    document_content_hash,
                    summary_mode,
                    model_name,
                    prompt_version,
                    config_hash
                )
                DO UPDATE SET
                    summary_text = excluded.summary_text,
                    source_chunk_count = excluded.source_chunk_count,
                    source_character_count = excluded.source_character_count,
                    is_partial = excluded.is_partial,
                    warnings = excluded.warnings,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    total_tokens = excluded.total_tokens,
                    updated_at = excluded.updated_at
                """,
                (
                    summary.id,
                    summary.workspace_id,
                    summary.document_id,
                    summary.document_content_hash,
                    summary.summary_mode,
                    summary.model_name,
                    summary.prompt_version,
                    summary.config_hash,
                    summary.config_json,
                    summary.summary_text,
                    summary.source_chunk_count,
                    summary.source_character_count,
                    int(summary.is_partial),
                    json.dumps(summary.warnings),
                    summary.token_usage.input_tokens,
                    summary.token_usage.output_tokens,
                    summary.token_usage.total_tokens,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM document_summaries
                WHERE document_id = ?
                  AND document_content_hash = ?
                  AND summary_mode = ?
                  AND model_name = ?
                  AND prompt_version = ?
                  AND config_hash = ?
                """,
                (
                    summary.document_id,
                    summary.document_content_hash,
                    summary.summary_mode,
                    summary.model_name,
                    summary.prompt_version,
                    summary.config_hash,
                ),
            ).fetchone()
        return _summary_from_row(row)

    def delete_for_document(self, document_id: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                "DELETE FROM document_summaries WHERE document_id = ?",
                (document_id,),
            )


def new_summary_id() -> str:
    """Generate a summary row ID."""
    return uuid.uuid4().hex
