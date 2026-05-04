"""SQLite repositories for retrieval evaluation cases and runs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from mini_notebooklm_rag.evaluation.models import (
    CompactRetrievedResult,
    EvalAggregateMetrics,
    EvalCase,
    EvalCaseMetrics,
    EvalRunConfig,
    EvalRunItemResult,
    EvalRunRecord,
    NewEvalCase,
)
from mini_notebooklm_rag.storage.repositories import utc_now
from mini_notebooklm_rag.storage.sqlite import connect


class EvalRepositoryError(RuntimeError):
    """Raised when evaluation repository operations fail."""


class EvalValidationError(ValueError):
    """Raised when an eval case is invalid."""


def new_eval_id() -> str:
    """Generate a stable local ID."""
    return uuid.uuid4().hex


def validate_eval_case(case: NewEvalCase) -> None:
    """Validate an eval case before persistence/import."""
    if not case.question.strip():
        raise EvalValidationError("Question is required.")
    if not case.expected_filename.strip():
        raise EvalValidationError("Expected filename is required.")
    if not case.selected_document_ids:
        raise EvalValidationError("Select at least one document.")
    if len(case.selected_document_ids) > 3:
        raise EvalValidationError("Select at most 3 documents.")
    if case.expected_page is not None and case.expected_page <= 0:
        raise EvalValidationError("Expected page must be positive.")
    has_start = case.expected_page_start is not None
    has_end = case.expected_page_end is not None
    if has_start != has_end:
        raise EvalValidationError("Expected page start and end must be provided together.")
    if has_start and has_end:
        if case.expected_page_start <= 0 or case.expected_page_end <= 0:
            raise EvalValidationError("Expected page range values must be positive.")
        if case.expected_page_start > case.expected_page_end:
            raise EvalValidationError("Expected page start cannot exceed expected page end.")


class EvaluationRepository:
    """CRUD for eval cases, runs, and run items."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_case(self, case: NewEvalCase) -> EvalCase:
        validate_eval_case(case)
        now = utc_now()
        case_id = case.id or new_eval_id()
        if self.get_case(case_id) is not None:
            case_id = new_eval_id()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO eval_cases (
                    id, workspace_id, question, selected_document_ids,
                    expected_filename, expected_page, expected_page_start,
                    expected_page_end, expected_answer, notes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    case.workspace_id,
                    case.question.strip(),
                    json.dumps(case.selected_document_ids),
                    case.expected_filename.strip(),
                    case.expected_page,
                    case.expected_page_start,
                    case.expected_page_end,
                    case.expected_answer,
                    case.notes,
                    now,
                    now,
                ),
            )
        created = self.get_case(case_id)
        if created is None:
            raise EvalRepositoryError("Eval case insert did not return a row.")
        return created

    def update_case(self, case_id: str, case: NewEvalCase) -> EvalCase:
        validate_eval_case(case)
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE eval_cases
                SET question = ?,
                    selected_document_ids = ?,
                    expected_filename = ?,
                    expected_page = ?,
                    expected_page_start = ?,
                    expected_page_end = ?,
                    expected_answer = ?,
                    notes = ?,
                    updated_at = ?
                WHERE id = ? AND workspace_id = ?
                """,
                (
                    case.question.strip(),
                    json.dumps(case.selected_document_ids),
                    case.expected_filename.strip(),
                    case.expected_page,
                    case.expected_page_start,
                    case.expected_page_end,
                    case.expected_answer,
                    case.notes,
                    now,
                    case_id,
                    case.workspace_id,
                ),
            )
        updated = self.get_case(case_id)
        if updated is None:
            raise EvalRepositoryError("Eval case was not found for update.")
        return updated

    def get_case(self, case_id: str) -> EvalCase | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM eval_cases WHERE id = ?",
                (case_id,),
            ).fetchone()
        return _case_from_row(row) if row else None

    def list_cases(self, workspace_id: str) -> list[EvalCase]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM eval_cases
                WHERE workspace_id = ?
                ORDER BY updated_at DESC, question ASC
                """,
                (workspace_id,),
            ).fetchall()
        return [_case_from_row(row) for row in rows]

    def delete_case(self, case_id: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM eval_cases WHERE id = ?", (case_id,))

    def get_cases_by_ids(self, workspace_id: str, case_ids: list[str]) -> list[EvalCase]:
        if not case_ids:
            return []
        placeholders = ",".join("?" for _ in case_ids)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM eval_cases
                WHERE workspace_id = ? AND id IN ({placeholders})
                """,
                (workspace_id, *case_ids),
            ).fetchall()
        by_id = {_case_from_row(row).id: _case_from_row(row) for row in rows}
        return [by_id[case_id] for case_id in case_ids if case_id in by_id]

    def create_run(
        self,
        workspace_id: str,
        status: str,
        config: EvalRunConfig,
        metrics: EvalAggregateMetrics,
        items: list[EvalRunItemResult],
        warnings: list[str],
        mlflow_run_id: str | None = None,
    ) -> EvalRunRecord:
        run_id = items[0].run_id if items else new_eval_id()
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO eval_runs (
                    id, workspace_id, status, top_k, dense_weight, sparse_weight,
                    embedding_model, embedding_device, eval_case_count,
                    filename_hit_count, filename_hit_rate, page_evaluable_count,
                    page_hit_count, page_hit_rate, page_range_evaluable_count,
                    page_range_hit_count, page_range_hit_rate, mean_reciprocal_rank,
                    mlflow_run_id, warnings, created_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    workspace_id,
                    status,
                    config.top_k,
                    config.dense_weight,
                    config.sparse_weight,
                    config.embedding_model,
                    config.embedding_device,
                    metrics.eval_case_count,
                    metrics.filename_hit_count,
                    metrics.filename_hit_rate,
                    metrics.page_evaluable_count,
                    metrics.page_hit_count,
                    metrics.page_hit_rate,
                    metrics.page_range_evaluable_count,
                    metrics.page_range_hit_count,
                    metrics.page_range_hit_rate,
                    metrics.mean_reciprocal_rank,
                    mlflow_run_id,
                    json.dumps(warnings),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO eval_run_items (
                    id, run_id, workspace_id, case_id, question,
                    selected_document_ids, expected_filename, expected_page,
                    expected_page_start, expected_page_end, filename_hit, page_hit,
                    page_range_hit, filename_hit_rank, page_hit_rank,
                    page_range_hit_rank, reciprocal_rank, retrieved_results,
                    retrieval_trace, warnings, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_item_to_row(item, now) for item in items],
            )
        run = self.get_run(run_id)
        if run is None:
            raise EvalRepositoryError("Eval run insert did not return a row.")
        return run

    def get_run(self, run_id: str) -> EvalRunRecord | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM eval_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return _run_from_row(row) if row else None

    def list_runs(self, workspace_id: str) -> list[EvalRunRecord]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM eval_runs
                WHERE workspace_id = ?
                ORDER BY created_at DESC
                """,
                (workspace_id,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def list_run_items(self, run_id: str) -> list[EvalRunItemResult]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM eval_run_items
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [_item_from_row(row) for row in rows]


def _case_from_row(row: sqlite3.Row) -> EvalCase:
    return EvalCase(
        id=row["id"],
        workspace_id=row["workspace_id"],
        question=row["question"],
        selected_document_ids=json.loads(row["selected_document_ids"]),
        expected_filename=row["expected_filename"],
        expected_page=row["expected_page"],
        expected_page_start=row["expected_page_start"],
        expected_page_end=row["expected_page_end"],
        expected_answer=row["expected_answer"],
        notes=row["notes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _run_from_row(row: sqlite3.Row) -> EvalRunRecord:
    metrics = EvalAggregateMetrics(
        eval_case_count=row["eval_case_count"],
        filename_hit_count=row["filename_hit_count"],
        filename_hit_rate=row["filename_hit_rate"],
        page_evaluable_count=row["page_evaluable_count"],
        page_hit_count=row["page_hit_count"],
        page_hit_rate=row["page_hit_rate"],
        page_range_evaluable_count=row["page_range_evaluable_count"],
        page_range_hit_count=row["page_range_hit_count"],
        page_range_hit_rate=row["page_range_hit_rate"],
        mean_reciprocal_rank=row["mean_reciprocal_rank"],
    )
    return EvalRunRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        status=row["status"],
        config=EvalRunConfig(
            top_k=row["top_k"],
            dense_weight=row["dense_weight"],
            sparse_weight=row["sparse_weight"],
            embedding_model=row["embedding_model"],
            embedding_device=row["embedding_device"],
        ),
        metrics=metrics,
        mlflow_run_id=row["mlflow_run_id"],
        warnings=json.loads(row["warnings"]),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def _item_to_row(item: EvalRunItemResult, created_at: str) -> tuple:
    return (
        item.id,
        item.run_id,
        item.workspace_id,
        item.case_id,
        item.question,
        json.dumps(item.selected_document_ids),
        item.expected_filename,
        item.expected_page,
        item.expected_page_start,
        item.expected_page_end,
        int(item.metrics.filename_hit),
        _nullable_bool(item.metrics.page_hit),
        _nullable_bool(item.metrics.page_range_hit),
        item.metrics.filename_hit_rank,
        item.metrics.page_hit_rank,
        item.metrics.page_range_hit_rank,
        item.metrics.reciprocal_rank,
        json.dumps([result.to_dict() for result in item.retrieved_results]),
        json.dumps(item.retrieval_trace),
        json.dumps(item.warnings),
        created_at,
    )


def _item_from_row(row: sqlite3.Row) -> EvalRunItemResult:
    return EvalRunItemResult(
        id=row["id"],
        run_id=row["run_id"],
        workspace_id=row["workspace_id"],
        case_id=row["case_id"],
        question=row["question"],
        selected_document_ids=json.loads(row["selected_document_ids"]),
        expected_filename=row["expected_filename"],
        expected_page=row["expected_page"],
        expected_page_start=row["expected_page_start"],
        expected_page_end=row["expected_page_end"],
        metrics=EvalCaseMetrics(
            filename_hit=bool(row["filename_hit"]),
            page_hit=_optional_bool(row["page_hit"]),
            page_range_hit=_optional_bool(row["page_range_hit"]),
            filename_hit_rank=row["filename_hit_rank"],
            page_hit_rank=row["page_hit_rank"],
            page_range_hit_rank=row["page_range_hit_rank"],
            reciprocal_rank=row["reciprocal_rank"],
        ),
        retrieved_results=[
            CompactRetrievedResult(**result) for result in json.loads(row["retrieved_results"])
        ],
        retrieval_trace=json.loads(row["retrieval_trace"]),
        warnings=json.loads(row["warnings"]),
        created_at=row["created_at"],
    )


def _nullable_bool(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)
