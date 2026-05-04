from __future__ import annotations

import json

from mini_notebooklm_rag.evaluation.import_export import (
    export_cases_json,
    parse_import_payload,
)
from mini_notebooklm_rag.evaluation.models import EVAL_FORMAT_VERSION, EvalCase


def test_export_and_import_cases_bind_to_current_workspace() -> None:
    exported_case = EvalCase(
        id="case1",
        workspace_id="old-workspace",
        question="Where is alpha?",
        selected_document_ids=["doc1"],
        expected_filename="paper.pdf",
        expected_page=5,
        expected_page_start=None,
        expected_page_end=None,
        expected_answer="alpha",
        notes="note",
        created_at="now",
        updated_at="now",
    )

    raw_json = export_cases_json("old-workspace", [exported_case])
    imported, errors = parse_import_payload(raw_json, "new-workspace")

    assert errors == []
    assert imported[0].workspace_id == "new-workspace"
    assert imported[0].id == "case1"
    assert imported[0].selected_document_ids == ["doc1"]


def test_import_reports_per_case_validation_errors() -> None:
    payload = {
        "format_version": EVAL_FORMAT_VERSION,
        "workspace_id": "source",
        "cases": [
            {
                "question": "",
                "selected_document_ids": [],
                "expected_filename": "",
            },
            {
                "id": "valid",
                "question": "Question?",
                "selected_document_ids": ["doc"],
                "expected_filename": "notes.md",
            },
        ],
    }

    imported, errors = parse_import_payload(json.dumps(payload), "target")

    assert len(imported) == 1
    assert imported[0].id == "valid"
    assert len(errors) == 1
    assert errors[0].index == 0


def test_import_rejects_unknown_format_version() -> None:
    payload = {"format_version": 999, "cases": []}

    imported, errors = parse_import_payload(json.dumps(payload), "target")

    assert imported == []
    assert "Unsupported format_version" in errors[0].message
