from __future__ import annotations

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.evaluation.models import MLflowLogResult, NewEvalCase
from mini_notebooklm_rag.evaluation.repositories import EvaluationRepository
from mini_notebooklm_rag.evaluation.runner import EvaluationRunner
from mini_notebooklm_rag.retrieval.models import (
    EmbeddingInfo,
    RetrievalResponse,
    RetrievalTrace,
    RetrievedChunk,
)
from mini_notebooklm_rag.storage.repositories import WorkspaceRepository
from mini_notebooklm_rag.storage.sqlite import initialize_database


class FakeRetrievalService:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    @property
    def embedding_info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            model_name="fake-model",
            requested_device="cpu",
            selected_device="cpu",
            dimension=3,
            normalized=True,
        )

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        selected_document_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> RetrievalResponse:
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "query": query,
                "selected_document_ids": selected_document_ids,
                "top_k": top_k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
            }
        )
        if self.fail:
            raise RuntimeError("FAISS index is missing or stale")
        chunk = RetrievedChunk(
            chunk_id="chunk1",
            document_id=selected_document_ids[0],
            filename="paper.pdf",
            text="This full chunk text must not be persisted in eval results.",
            source_type="pdf",
            page_start=2,
            page_end=3,
            heading_path=None,
            dense_score=0.9,
            sparse_score=0.1,
            fused_score=0.8,
            rank=1,
            citation="paper.pdf, pp. 2-3",
        )
        trace = RetrievalTrace(
            original_query=query,
            selected_document_ids=selected_document_ids,
            embedding_model="fake-model",
            embedding_device="cpu",
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            dense_candidates=[],
            sparse_candidates=[],
            fused_results=[chunk],
            warnings=[],
        )
        return RetrievalResponse(results=[chunk], trace=trace, warnings=["trace warning"])


class FakeMLflowLogger:
    def __init__(self):
        self.calls = 0

    def log_eval_run(self, run_result, cases):
        self.calls += 1
        assert cases
        return MLflowLogResult(status="disabled", message="MLflow logging disabled.")


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, app_storage_dir=str(tmp_path / "storage"))


def _seed_case(tmp_path):
    settings = _settings(tmp_path)
    db_path = tmp_path / "storage" / "app.db"
    initialize_database(db_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    repository = EvaluationRepository(db_path)
    eval_case = repository.create_case(
        NewEvalCase(
            workspace_id=workspace.id,
            question="Where is alpha?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
            expected_page=2,
        )
    )
    return settings, workspace, eval_case


def test_runner_uses_retrieval_service_and_stores_compact_results(tmp_path) -> None:
    settings, workspace, eval_case = _seed_case(tmp_path)
    retrieval = FakeRetrievalService()
    mlflow = FakeMLflowLogger()
    runner = EvaluationRunner(settings, retrieval_service=retrieval, mlflow_logger=mlflow)

    result = runner.run_batch(
        workspace_id=workspace.id,
        case_ids=[eval_case.id],
        top_k=6,
        dense_weight=0.65,
        sparse_weight=0.35,
    )

    assert retrieval.calls[0]["query"] == "Where is alpha?"
    assert mlflow.calls == 1
    assert result.run.metrics.filename_hit_rate == 1.0
    assert result.run.metrics.page_hit_rate == 1.0
    assert result.items[0].retrieved_results[0].filename == "paper.pdf"
    assert "full chunk text" not in str(result.items[0].retrieved_results[0].to_dict())
    assert "full chunk text" not in str(result.items[0].retrieval_trace)


def test_runner_treats_recoverable_retrieval_failure_as_miss(tmp_path) -> None:
    settings, workspace, eval_case = _seed_case(tmp_path)
    runner = EvaluationRunner(
        settings,
        retrieval_service=FakeRetrievalService(fail=True),
        mlflow_logger=FakeMLflowLogger(),
    )

    result = runner.run_batch(
        workspace_id=workspace.id,
        case_ids=[eval_case.id],
        top_k=6,
        dense_weight=0.65,
        sparse_weight=0.35,
    )

    assert result.run.metrics.filename_hit_rate == 0.0
    assert "Evaluation retrieval failed" in result.items[0].warnings[0]
