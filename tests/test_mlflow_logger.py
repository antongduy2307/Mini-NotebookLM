from __future__ import annotations

import builtins
import json
from pathlib import Path

from mini_notebooklm_rag.evaluation.mlflow_logger import MLflowEvalLogger
from mini_notebooklm_rag.evaluation.models import (
    EvalAggregateMetrics,
    EvalCase,
    EvalRunConfig,
    EvalRunRecord,
    EvalRunResult,
    MLflowLogResult,
)


def _run_result() -> EvalRunResult:
    return EvalRunResult(
        run=EvalRunRecord(
            id="run1",
            workspace_id="workspace",
            status="completed",
            config=EvalRunConfig(
                top_k=6,
                dense_weight=0.65,
                sparse_weight=0.35,
                embedding_model="fake-model",
                embedding_device="cpu",
            ),
            metrics=EvalAggregateMetrics(
                eval_case_count=1,
                filename_hit_count=1,
                filename_hit_rate=1.0,
                page_evaluable_count=0,
                page_hit_count=0,
                page_hit_rate=None,
                page_range_evaluable_count=0,
                page_range_hit_count=0,
                page_range_hit_rate=None,
                mean_reciprocal_rank=1.0,
            ),
            mlflow_run_id=None,
            warnings=[],
            created_at="now",
            completed_at="now",
        ),
        items=[],
        mlflow_status=MLflowLogResult(status="disabled", message="disabled"),
    )


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="case1",
            workspace_id="workspace",
            question="Where is alpha?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
            expected_page=None,
            expected_page_start=None,
            expected_page_end=None,
            expected_answer=None,
            notes="",
            created_at="now",
            updated_at="now",
        )
    ]


def test_mlflow_logger_disabled_when_tracking_uri_empty() -> None:
    result = MLflowEvalLogger("").log_eval_run(_run_result(), _cases())

    assert result.status == "disabled"


def test_mlflow_logger_missing_package_is_actionable(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = MLflowEvalLogger("file:./mlruns").log_eval_run(_run_result(), _cases())

    assert result.status == "missing"
    assert "uv sync --extra observability" in result.message


def test_mlflow_logger_logs_compact_artifacts_with_fake_module() -> None:
    class ActiveRun:
        class Info:
            run_id = "mlflow-run"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeMLflow:
        def __init__(self):
            self.tracking_uri = None
            self.params = None
            self.metrics = None
            self.artifact_dir = None

        def set_tracking_uri(self, tracking_uri):
            self.tracking_uri = tracking_uri

        def start_run(self, run_name):
            assert run_name == "eval-run1"
            return ActiveRun()

        def log_params(self, params):
            self.params = params

        def log_metrics(self, metrics):
            self.metrics = metrics

        def log_artifacts(self, artifact_dir):
            self.artifact_dir = artifact_dir
            artifact_path = Path(artifact_dir)
            results = json.loads((artifact_path / "eval_results.json").read_text())
            assert "items" in results
            assert "full chunk text" not in json.dumps(results)
            assert (artifact_path / "eval_cases.json").exists()
            assert (artifact_path / "retrieval_config.json").exists()

    fake_mlflow = FakeMLflow()
    result = MLflowEvalLogger(
        "file:./mlruns",
        mlflow_module=fake_mlflow,
    ).log_eval_run(_run_result(), _cases())

    assert result.status == "logged"
    assert result.run_id == "mlflow-run"
    assert fake_mlflow.tracking_uri == "file:./mlruns"
    assert fake_mlflow.params["embedding_model"] == "fake-model"
    assert fake_mlflow.metrics["filename_hit_rate_at_k"] == 1.0
    assert fake_mlflow.artifact_dir is not None
