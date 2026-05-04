"""Optional MLflow logging for retrieval evaluation batches."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from mini_notebooklm_rag.evaluation.import_export import export_cases_json
from mini_notebooklm_rag.evaluation.models import EvalCase, EvalRunResult, MLflowLogResult


class MLflowEvalLogger:
    """Lazy optional MLflow boundary."""

    def __init__(self, tracking_uri: str, mlflow_module: Any | None = None):
        self.tracking_uri = tracking_uri.strip()
        self._mlflow_module = mlflow_module

    def log_eval_run(
        self,
        run_result: EvalRunResult,
        cases: list[EvalCase],
    ) -> MLflowLogResult:
        """Log a completed eval batch when MLflow is configured and available."""
        if not self.tracking_uri:
            return MLflowLogResult(status="disabled", message="MLflow logging disabled.")

        mlflow = self._load_mlflow()
        if mlflow is None:
            return MLflowLogResult(
                status="missing",
                message=(
                    "MLflow is configured but the optional mlflow package is not installed. "
                    "Install with uv sync --extra observability."
                ),
            )

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = Path(temp_dir)
                _write_artifacts(artifact_dir, run_result, cases)
                with mlflow.start_run(run_name=f"eval-{run_result.run.id}") as active_run:
                    mlflow.log_params(_params(run_result))
                    mlflow.log_metrics(_metrics(run_result))
                    mlflow.log_artifacts(str(artifact_dir))
                    run_id = getattr(getattr(active_run, "info", None), "run_id", None)
            return MLflowLogResult(
                status="logged",
                message="Logged eval run to MLflow.",
                run_id=run_id,
            )
        except Exception as exc:
            return MLflowLogResult(
                status="failed",
                message=f"MLflow logging failed: {type(exc).__name__}",
            )

    def _load_mlflow(self):
        if self._mlflow_module is not None:
            return self._mlflow_module
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            return None
        return mlflow


def _params(run_result: EvalRunResult) -> dict[str, object]:
    config = run_result.run.config
    return {
        "embedding_model": config.embedding_model,
        "embedding_device": config.embedding_device,
        "top_k": config.top_k,
        "dense_weight": config.dense_weight,
        "sparse_weight": config.sparse_weight,
        "workspace_id": run_result.run.workspace_id,
        "eval_case_count": run_result.run.metrics.eval_case_count,
    }


def _metrics(run_result: EvalRunResult) -> dict[str, float]:
    metrics = run_result.run.metrics
    values = {
        "filename_hit_rate_at_k": metrics.filename_hit_rate,
    }
    if metrics.page_hit_rate is not None:
        values["page_hit_rate_at_k"] = metrics.page_hit_rate
    if metrics.page_range_hit_rate is not None:
        values["page_range_hit_rate_at_k"] = metrics.page_range_hit_rate
    if metrics.mean_reciprocal_rank is not None:
        values["mean_reciprocal_rank"] = metrics.mean_reciprocal_rank
    return values


def _write_artifacts(
    artifact_dir: Path,
    run_result: EvalRunResult,
    cases: list[EvalCase],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "eval_cases.json").write_text(
        export_cases_json(run_result.run.workspace_id, cases),
        encoding="utf-8",
    )
    (artifact_dir / "eval_results.json").write_text(
        json.dumps(_results_payload(run_result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "retrieval_config.json").write_text(
        json.dumps(_params(run_result), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _results_payload(run_result: EvalRunResult) -> dict[str, object]:
    return {
        "run_id": run_result.run.id,
        "metrics": {
            "filename_hit_rate": run_result.run.metrics.filename_hit_rate,
            "page_hit_rate": run_result.run.metrics.page_hit_rate,
            "page_range_hit_rate": run_result.run.metrics.page_range_hit_rate,
            "mean_reciprocal_rank": run_result.run.metrics.mean_reciprocal_rank,
        },
        "items": [
            {
                "case_id": item.case_id,
                "question": item.question,
                "expected_filename": item.expected_filename,
                "expected_page": item.expected_page,
                "expected_page_start": item.expected_page_start,
                "expected_page_end": item.expected_page_end,
                "filename_hit": item.metrics.filename_hit,
                "page_hit": item.metrics.page_hit,
                "page_range_hit": item.metrics.page_range_hit,
                "filename_hit_rank": item.metrics.filename_hit_rank,
                "page_hit_rank": item.metrics.page_hit_rank,
                "page_range_hit_rank": item.metrics.page_range_hit_rank,
                "retrieved_results": [result.to_dict() for result in item.retrieved_results],
                "warnings": item.warnings,
            }
            for item in run_result.items
        ],
    }
