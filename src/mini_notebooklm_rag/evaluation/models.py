"""Data models for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EVAL_FORMAT_VERSION = 1
EvalRunStatus = Literal["completed", "failed"]
MLflowStatus = Literal["disabled", "missing", "logged", "failed"]


@dataclass(frozen=True)
class EvalCase:
    id: str
    workspace_id: str
    question: str
    selected_document_ids: list[str]
    expected_filename: str
    expected_page: int | None
    expected_page_start: int | None
    expected_page_end: int | None
    expected_answer: str | None
    notes: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class NewEvalCase:
    workspace_id: str
    question: str
    selected_document_ids: list[str]
    expected_filename: str
    expected_page: int | None = None
    expected_page_start: int | None = None
    expected_page_end: int | None = None
    expected_answer: str | None = None
    notes: str = ""
    id: str | None = None


@dataclass(frozen=True)
class EvalRunConfig:
    top_k: int
    dense_weight: float
    sparse_weight: float
    embedding_model: str
    embedding_device: str


@dataclass(frozen=True)
class CompactRetrievedResult:
    rank: int
    chunk_id: str
    document_id: str
    filename: str
    citation: str
    source_type: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    dense_score: float
    sparse_score: float
    fused_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "citation": self.citation,
            "source_type": self.source_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": self.heading_path,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fused_score": self.fused_score,
        }


@dataclass(frozen=True)
class EvalCaseMetrics:
    filename_hit: bool
    page_hit: bool | None
    page_range_hit: bool | None
    filename_hit_rank: int | None
    page_hit_rank: int | None
    page_range_hit_rank: int | None
    reciprocal_rank: float | None


@dataclass(frozen=True)
class EvalRunItemResult:
    id: str
    run_id: str
    workspace_id: str
    case_id: str
    question: str
    selected_document_ids: list[str]
    expected_filename: str
    expected_page: int | None
    expected_page_start: int | None
    expected_page_end: int | None
    metrics: EvalCaseMetrics
    retrieved_results: list[CompactRetrievedResult]
    retrieval_trace: dict[str, object]
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass(frozen=True)
class EvalAggregateMetrics:
    eval_case_count: int
    filename_hit_count: int
    filename_hit_rate: float
    page_evaluable_count: int
    page_hit_count: int
    page_hit_rate: float | None
    page_range_evaluable_count: int
    page_range_hit_count: int
    page_range_hit_rate: float | None
    mean_reciprocal_rank: float | None


@dataclass(frozen=True)
class EvalRunRecord:
    id: str
    workspace_id: str
    status: EvalRunStatus
    config: EvalRunConfig
    metrics: EvalAggregateMetrics
    mlflow_run_id: str | None
    warnings: list[str]
    created_at: str
    completed_at: str


@dataclass(frozen=True)
class EvalRunResult:
    run: EvalRunRecord
    items: list[EvalRunItemResult]
    mlflow_status: MLflowLogResult


@dataclass(frozen=True)
class ImportValidationError:
    index: int
    message: str


@dataclass(frozen=True)
class EvalImportResult:
    imported: list[EvalCase]
    errors: list[ImportValidationError]


@dataclass(frozen=True)
class MLflowLogResult:
    status: MLflowStatus
    message: str
    run_id: str | None = None
