from __future__ import annotations

from mini_notebooklm_rag.evaluation.metrics import aggregate_metrics, evaluate_case
from mini_notebooklm_rag.evaluation.models import (
    CompactRetrievedResult,
    EvalRunItemResult,
    NewEvalCase,
)


def _result(
    rank: int,
    filename: str,
    page_start: int | None = None,
    page_end: int | None = None,
) -> CompactRetrievedResult:
    return CompactRetrievedResult(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        document_id=f"doc-{rank}",
        filename=filename,
        citation=filename,
        source_type="pdf" if page_start is not None else "markdown",
        page_start=page_start,
        page_end=page_end,
        heading_path=None,
        dense_score=1.0,
        sparse_score=0.0,
        fused_score=1.0 / rank,
    )


def test_evaluate_case_matches_filename_page_and_page_range_case_insensitively() -> None:
    eval_case = NewEvalCase(
        workspace_id="workspace",
        question="Where is alpha?",
        selected_document_ids=["doc1"],
        expected_filename="Paper.PDF",
        expected_page=5,
        expected_page_start=4,
        expected_page_end=6,
    )
    results = [
        _result(1, "other.pdf", 5, 5),
        _result(2, "paper.pdf", 3, 5),
    ]

    metrics = evaluate_case(eval_case, results)

    assert metrics.filename_hit is True
    assert metrics.filename_hit_rank == 2
    assert metrics.page_hit is True
    assert metrics.page_hit_rank == 2
    assert metrics.page_range_hit is True
    assert metrics.page_range_hit_rank == 2
    assert metrics.reciprocal_rank == 0.5


def test_evaluate_case_marks_page_metrics_not_applicable_when_absent() -> None:
    eval_case = NewEvalCase(
        workspace_id="workspace",
        question="Where is alpha?",
        selected_document_ids=["doc1"],
        expected_filename="notes.md",
    )

    metrics = evaluate_case(eval_case, [_result(1, "notes.md")])

    assert metrics.filename_hit is True
    assert metrics.page_hit is None
    assert metrics.page_range_hit is None


def test_aggregate_metrics_skips_not_applicable_page_denominators() -> None:
    page_item = EvalRunItemResult(
        id="item1",
        run_id="run",
        workspace_id="workspace",
        case_id="case1",
        question="q1",
        selected_document_ids=["doc1"],
        expected_filename="paper.pdf",
        expected_page=3,
        expected_page_start=None,
        expected_page_end=None,
        metrics=evaluate_case(
            NewEvalCase(
                workspace_id="workspace",
                question="q1",
                selected_document_ids=["doc1"],
                expected_filename="paper.pdf",
                expected_page=3,
            ),
            [_result(1, "paper.pdf", 3, 4)],
        ),
        retrieved_results=[],
        retrieval_trace={},
    )
    markdown_item = EvalRunItemResult(
        id="item2",
        run_id="run",
        workspace_id="workspace",
        case_id="case2",
        question="q2",
        selected_document_ids=["doc2"],
        expected_filename="notes.md",
        expected_page=None,
        expected_page_start=None,
        expected_page_end=None,
        metrics=evaluate_case(
            NewEvalCase(
                workspace_id="workspace",
                question="q2",
                selected_document_ids=["doc2"],
                expected_filename="notes.md",
            ),
            [_result(1, "missing.md")],
        ),
        retrieved_results=[],
        retrieval_trace={},
    )

    aggregate = aggregate_metrics([page_item, markdown_item])

    assert aggregate.eval_case_count == 2
    assert aggregate.filename_hit_count == 1
    assert aggregate.filename_hit_rate == 0.5
    assert aggregate.page_evaluable_count == 1
    assert aggregate.page_hit_rate == 1.0
    assert aggregate.page_range_hit_rate is None
