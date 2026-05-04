"""JSON import/export helpers for workspace evaluation cases."""

from __future__ import annotations

import json
from typing import Any

from mini_notebooklm_rag.evaluation.models import (
    EVAL_FORMAT_VERSION,
    EvalCase,
    ImportValidationError,
    NewEvalCase,
)
from mini_notebooklm_rag.evaluation.repositories import EvalValidationError, validate_eval_case


def export_cases_payload(workspace_id: str, cases: list[EvalCase]) -> dict[str, Any]:
    """Return a portable JSON payload for workspace eval cases."""
    return {
        "format_version": EVAL_FORMAT_VERSION,
        "workspace_id": workspace_id,
        "cases": [_case_to_export(case) for case in cases],
    }


def export_cases_json(workspace_id: str, cases: list[EvalCase]) -> str:
    """Serialize eval cases to stable, human-readable JSON."""
    return json.dumps(export_cases_payload(workspace_id, cases), indent=2, sort_keys=True)


def parse_import_payload(
    raw_json: str,
    workspace_id: str,
) -> tuple[list[NewEvalCase], list[ImportValidationError]]:
    """Parse append-only import cases bound to the current workspace."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return [], [ImportValidationError(index=-1, message=f"Invalid JSON: {exc}")]

    if not isinstance(payload, dict):
        return [], [ImportValidationError(index=-1, message="Import payload must be an object.")]
    if payload.get("format_version") != EVAL_FORMAT_VERSION:
        return [], [
            ImportValidationError(
                index=-1,
                message=f"Unsupported format_version: {payload.get('format_version')}",
            )
        ]
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return [], [ImportValidationError(index=-1, message="Import payload must contain cases.")]

    imported: list[NewEvalCase] = []
    errors: list[ImportValidationError] = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            errors.append(ImportValidationError(index=index, message="Case must be an object."))
            continue
        try:
            new_case = _case_from_import(item, workspace_id)
            validate_eval_case(new_case)
            imported.append(new_case)
        except (TypeError, ValueError, EvalValidationError) as exc:
            errors.append(ImportValidationError(index=index, message=str(exc)))
    return imported, errors


def _case_to_export(case: EvalCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "question": case.question,
        "selected_document_ids": case.selected_document_ids,
        "expected_filename": case.expected_filename,
        "expected_page": case.expected_page,
        "expected_page_start": case.expected_page_start,
        "expected_page_end": case.expected_page_end,
        "expected_answer": case.expected_answer,
        "notes": case.notes,
    }


def _case_from_import(data: dict[str, Any], workspace_id: str) -> NewEvalCase:
    selected_document_ids = data.get("selected_document_ids", [])
    if not isinstance(selected_document_ids, list) or not all(
        isinstance(document_id, str) for document_id in selected_document_ids
    ):
        raise ValueError("selected_document_ids must be a list of strings.")
    return NewEvalCase(
        id=data.get("id") if isinstance(data.get("id"), str) else None,
        workspace_id=workspace_id,
        question=_required_str(data, "question"),
        selected_document_ids=selected_document_ids,
        expected_filename=_required_str(data, "expected_filename"),
        expected_page=_optional_int(data, "expected_page"),
        expected_page_start=_optional_int(data, "expected_page_start"),
        expected_page_end=_optional_int(data, "expected_page_end"),
        expected_answer=data.get("expected_answer")
        if isinstance(data.get("expected_answer"), str)
        else None,
        notes=data.get("notes") if isinstance(data.get("notes"), str) else "",
    )


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} is required.")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null.")
    return value
