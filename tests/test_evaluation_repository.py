from __future__ import annotations

from mini_notebooklm_rag.evaluation.models import EvalRunConfig, NewEvalCase
from mini_notebooklm_rag.evaluation.repositories import EvaluationRepository
from mini_notebooklm_rag.storage.repositories import (
    DocumentRepository,
    NewDocumentRecord,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import initialize_database


def _db(tmp_path):
    db_path = tmp_path / "storage" / "app.db"
    initialize_database(db_path)
    return db_path


def test_eval_case_crud_and_append_import_id_conflict(tmp_path) -> None:
    db_path = _db(tmp_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    repository = EvaluationRepository(db_path)

    first = repository.create_case(
        NewEvalCase(
            id="imported",
            workspace_id=workspace.id,
            question="Where is alpha?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
            expected_page=1,
        )
    )
    second = repository.create_case(
        NewEvalCase(
            id="imported",
            workspace_id=workspace.id,
            question="Where is beta?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
        )
    )

    assert first.id == "imported"
    assert second.id != first.id
    assert len(repository.list_cases(workspace.id)) == 2

    updated = repository.update_case(
        first.id,
        NewEvalCase(
            workspace_id=workspace.id,
            question="Updated?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
            expected_page_start=1,
            expected_page_end=2,
        ),
    )
    assert updated.question == "Updated?"
    assert updated.expected_page_start == 1

    repository.delete_case(second.id)
    assert repository.get_case(second.id) is None


def test_document_delete_does_not_delete_eval_case_but_workspace_delete_cascades(
    tmp_path,
) -> None:
    db_path = _db(tmp_path)
    workspace_repository = WorkspaceRepository(db_path)
    workspace = workspace_repository.create("Research")
    document_repository = DocumentRepository(db_path)
    document_repository.insert_with_chunks(
        NewDocumentRecord(
            id="doc1",
            workspace_id=workspace.id,
            display_name="paper.pdf",
            stored_filename="doc1__paper.pdf",
            relative_path="workspaces/ws/documents/doc1__paper.pdf",
            source_type="pdf",
            content_hash="hash",
            size_bytes=10,
            page_count=1,
        ),
        [],
    )
    repository = EvaluationRepository(db_path)
    eval_case = repository.create_case(
        NewEvalCase(
            workspace_id=workspace.id,
            question="Where is alpha?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
        )
    )

    document_repository.delete("doc1")
    assert repository.get_case(eval_case.id) is not None

    workspace_repository.delete(workspace.id)
    assert repository.list_cases(workspace.id) == []


def test_eval_run_persistence_with_items(tmp_path) -> None:
    from mini_notebooklm_rag.evaluation.metrics import aggregate_metrics, evaluate_case
    from mini_notebooklm_rag.evaluation.models import (
        CompactRetrievedResult,
        EvalRunItemResult,
    )

    db_path = _db(tmp_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    repository = EvaluationRepository(db_path)
    eval_case = repository.create_case(
        NewEvalCase(
            workspace_id=workspace.id,
            question="Where is alpha?",
            selected_document_ids=["doc1"],
            expected_filename="paper.pdf",
        )
    )
    result = CompactRetrievedResult(
        rank=1,
        chunk_id="chunk1",
        document_id="doc1",
        filename="paper.pdf",
        citation="paper.pdf, p. 1",
        source_type="pdf",
        page_start=1,
        page_end=1,
        heading_path=None,
        dense_score=1.0,
        sparse_score=0.0,
        fused_score=1.0,
    )
    item = EvalRunItemResult(
        id="item1",
        run_id="run1",
        workspace_id=workspace.id,
        case_id=eval_case.id,
        question=eval_case.question,
        selected_document_ids=eval_case.selected_document_ids,
        expected_filename=eval_case.expected_filename,
        expected_page=None,
        expected_page_start=None,
        expected_page_end=None,
        metrics=evaluate_case(eval_case, [result]),
        retrieved_results=[result],
        retrieval_trace={"fused_results": [result.to_dict()]},
    )
    config = EvalRunConfig(
        top_k=6,
        dense_weight=0.65,
        sparse_weight=0.35,
        embedding_model="fake-model",
        embedding_device="cpu",
    )

    run = repository.create_run(
        workspace_id=workspace.id,
        status="completed",
        config=config,
        metrics=aggregate_metrics([item]),
        items=[item],
        warnings=[],
    )
    stored_items = repository.list_run_items(run.id)

    assert run.id == "run1"
    assert stored_items[0].retrieved_results[0].filename == "paper.pdf"
