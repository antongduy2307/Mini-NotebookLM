"""Retrieval evaluation runner."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.evaluation.metrics import aggregate_metrics, evaluate_case
from mini_notebooklm_rag.evaluation.mlflow_logger import MLflowEvalLogger
from mini_notebooklm_rag.evaluation.models import (
    CompactRetrievedResult,
    EvalCase,
    EvalRunConfig,
    EvalRunItemResult,
    EvalRunRecord,
    EvalRunResult,
    MLflowLogResult,
)
from mini_notebooklm_rag.evaluation.repositories import EvaluationRepository, new_eval_id
from mini_notebooklm_rag.retrieval.models import EmbeddingInfo, RetrievalResponse, RetrievedChunk
from mini_notebooklm_rag.retrieval.service import RetrievalService
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.sqlite import initialize_database


class RetrievalServiceProtocol(Protocol):
    """Retrieval service shape used by evaluation."""

    @property
    def embedding_info(self) -> EmbeddingInfo:
        """Return embedding model/device info."""

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        selected_document_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> RetrievalResponse:
        """Run retrieval."""


class EvaluationRunner:
    """Run retrieval-only eval batches and persist compact results."""

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalServiceProtocol | None = None,
        mlflow_logger: MLflowEvalLogger | None = None,
    ):
        paths = StoragePaths(Path(settings.app_storage_dir))
        paths.ensure_root()
        initialize_database(paths.db_path)
        self.settings = settings
        self.repository = EvaluationRepository(paths.db_path)
        self.retrieval_service = retrieval_service or RetrievalService(settings)
        self.mlflow_logger = mlflow_logger or MLflowEvalLogger(settings.mlflow_tracking_uri)

    def run_batch(
        self,
        workspace_id: str,
        case_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> EvalRunResult:
        """Run a retrieval evaluation batch."""
        cases = self.repository.get_cases_by_ids(workspace_id, case_ids)
        warnings: list[str] = []
        if len(cases) != len(case_ids):
            warnings.append("Some selected evaluation cases were not found.")
        run_id = new_eval_id()
        embedding_info = self.retrieval_service.embedding_info
        config = EvalRunConfig(
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            embedding_model=embedding_info.model_name,
            embedding_device=embedding_info.selected_device,
        )
        items = [self._run_case(run_id, workspace_id, eval_case, config) for eval_case in cases]
        metrics = aggregate_metrics(items)
        local_run = EvalRunRecord(
            id=run_id,
            workspace_id=workspace_id,
            status="completed",
            config=config,
            metrics=metrics,
            mlflow_run_id=None,
            warnings=warnings,
            created_at="",
            completed_at="",
        )
        provisional_result = EvalRunResult(
            run=local_run,
            items=items,
            mlflow_status=MLflowLogResult(status="disabled", message="MLflow logging disabled."),
        )
        mlflow_status = self.mlflow_logger.log_eval_run(provisional_result, cases)
        run_record = self.repository.create_run(
            workspace_id=workspace_id,
            status="completed",
            config=config,
            metrics=metrics,
            items=items,
            warnings=warnings,
            mlflow_run_id=mlflow_status.run_id,
        )
        stored_items = self.repository.list_run_items(run_record.id)
        return EvalRunResult(run=run_record, items=stored_items, mlflow_status=mlflow_status)

    def _run_case(
        self,
        run_id: str,
        workspace_id: str,
        eval_case: EvalCase,
        config: EvalRunConfig,
    ) -> EvalRunItemResult:
        warnings: list[str] = []
        compact_results: list[CompactRetrievedResult] = []
        compact_trace: dict[str, object] = {}
        try:
            response = self.retrieval_service.retrieve(
                workspace_id=workspace_id,
                query=eval_case.question,
                selected_document_ids=eval_case.selected_document_ids,
                top_k=config.top_k,
                dense_weight=config.dense_weight,
                sparse_weight=config.sparse_weight,
            )
            compact_results = [_compact_result(result) for result in response.results]
            compact_trace = _compact_trace(response)
            warnings.extend(response.warnings)
        except Exception as exc:
            warnings.append(f"Evaluation retrieval failed: {type(exc).__name__}: {exc}")
            compact_trace = {
                "original_query": eval_case.question,
                "selected_document_ids": eval_case.selected_document_ids,
                "warnings": warnings,
            }

        metrics = evaluate_case(eval_case, compact_results)
        return EvalRunItemResult(
            id=new_eval_id(),
            run_id=run_id,
            workspace_id=workspace_id,
            case_id=eval_case.id,
            question=eval_case.question,
            selected_document_ids=eval_case.selected_document_ids,
            expected_filename=eval_case.expected_filename,
            expected_page=eval_case.expected_page,
            expected_page_start=eval_case.expected_page_start,
            expected_page_end=eval_case.expected_page_end,
            metrics=metrics,
            retrieved_results=compact_results,
            retrieval_trace=compact_trace,
            warnings=warnings,
        )


def _compact_result(result: RetrievedChunk) -> CompactRetrievedResult:
    return CompactRetrievedResult(
        rank=result.rank,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        filename=result.filename,
        citation=result.citation,
        source_type=result.source_type,
        page_start=result.page_start,
        page_end=result.page_end,
        heading_path=result.heading_path,
        dense_score=result.dense_score,
        sparse_score=result.sparse_score,
        fused_score=result.fused_score,
    )


def _compact_trace(response: RetrievalResponse) -> dict[str, object]:
    return {
        "original_query": response.trace.original_query,
        "selected_document_ids": response.trace.selected_document_ids,
        "embedding_model": response.trace.embedding_model,
        "embedding_device": response.trace.embedding_device,
        "top_k": response.trace.top_k,
        "dense_weight": response.trace.dense_weight,
        "sparse_weight": response.trace.sparse_weight,
        "dense_candidates": [candidate.__dict__ for candidate in response.trace.dense_candidates],
        "sparse_candidates": [candidate.__dict__ for candidate in response.trace.sparse_candidates],
        "fused_results": [_compact_result(result).to_dict() for result in response.results],
        "warnings": response.trace.warnings,
    }
