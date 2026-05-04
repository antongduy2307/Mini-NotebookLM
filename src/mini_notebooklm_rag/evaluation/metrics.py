"""Retrieval evaluation metric helpers."""

from __future__ import annotations

from collections.abc import Sequence

from mini_notebooklm_rag.evaluation.models import (
    CompactRetrievedResult,
    EvalAggregateMetrics,
    EvalCase,
    EvalCaseMetrics,
    EvalRunItemResult,
)


def filenames_match(left: str, right: str) -> bool:
    """Compare filenames case-insensitively while preserving original values elsewhere."""
    return left.strip().casefold() == right.strip().casefold()


def ranges_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int,
    right_end: int,
) -> bool:
    """Return true when two 1-indexed page ranges overlap."""
    if left_start is None or left_end is None:
        return False
    return max(left_start, right_start) <= min(left_end, right_end)


def evaluate_case(
    eval_case: EvalCase,
    results: Sequence[CompactRetrievedResult],
) -> EvalCaseMetrics:
    """Compute retrieval-only metrics for one eval case."""
    filename_rank: int | None = None
    page_rank: int | None = None
    page_range_rank: int | None = None

    for result in results:
        if not filenames_match(result.filename, eval_case.expected_filename):
            continue
        if filename_rank is None:
            filename_rank = result.rank
        if eval_case.expected_page is not None and page_rank is None:
            if ranges_overlap(
                result.page_start,
                result.page_end,
                eval_case.expected_page,
                eval_case.expected_page,
            ):
                page_rank = result.rank
        if (
            eval_case.expected_page_start is not None
            and eval_case.expected_page_end is not None
            and page_range_rank is None
            and ranges_overlap(
                result.page_start,
                result.page_end,
                eval_case.expected_page_start,
                eval_case.expected_page_end,
            )
        ):
            page_range_rank = result.rank

    reciprocal_rank = 1 / filename_rank if filename_rank else 0.0
    return EvalCaseMetrics(
        filename_hit=filename_rank is not None,
        page_hit=None if eval_case.expected_page is None else page_rank is not None,
        page_range_hit=None
        if eval_case.expected_page_start is None or eval_case.expected_page_end is None
        else page_range_rank is not None,
        filename_hit_rank=filename_rank,
        page_hit_rank=page_rank,
        page_range_hit_rank=page_range_rank,
        reciprocal_rank=reciprocal_rank,
    )


def aggregate_metrics(items: Sequence[EvalRunItemResult]) -> EvalAggregateMetrics:
    """Aggregate per-case metrics into batch metrics."""
    case_count = len(items)
    filename_hit_count = sum(1 for item in items if item.metrics.filename_hit)
    page_items = [item for item in items if item.metrics.page_hit is not None]
    page_range_items = [item for item in items if item.metrics.page_range_hit is not None]
    page_hit_count = sum(1 for item in page_items if item.metrics.page_hit)
    page_range_hit_count = sum(1 for item in page_range_items if item.metrics.page_range_hit)
    reciprocal_ranks = [
        item.metrics.reciprocal_rank for item in items if item.metrics.reciprocal_rank is not None
    ]
    return EvalAggregateMetrics(
        eval_case_count=case_count,
        filename_hit_count=filename_hit_count,
        filename_hit_rate=filename_hit_count / case_count if case_count else 0.0,
        page_evaluable_count=len(page_items),
        page_hit_count=page_hit_count,
        page_hit_rate=page_hit_count / len(page_items) if page_items else None,
        page_range_evaluable_count=len(page_range_items),
        page_range_hit_count=page_range_hit_count,
        page_range_hit_rate=page_range_hit_count / len(page_range_items)
        if page_range_items
        else None,
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else None,
    )
