"""Retrieval evaluation Streamlit panel."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.evaluation.import_export import export_cases_json, parse_import_payload
from mini_notebooklm_rag.evaluation.mlflow_logger import MLflowEvalLogger
from mini_notebooklm_rag.evaluation.models import EvalRunResult, NewEvalCase
from mini_notebooklm_rag.evaluation.repositories import EvaluationRepository, EvalValidationError
from mini_notebooklm_rag.evaluation.runner import EvaluationRunner
from mini_notebooklm_rag.retrieval.service import MAX_SELECTED_DOCUMENTS, RetrievalService
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.ui.shared import _zero_to_none


def render_evaluation_panel(
    repository: EvaluationRepository,
    retrieval_service: RetrievalService | None,
    workspace: Workspace,
    documents: list[DocumentRecord],
    settings,
) -> None:
    st.subheader("Evaluation")
    st.caption("Create retrieval eval cases and run local hit@k evaluation batches.")
    if settings.mlflow_tracking_uri:
        st.info("MLflow logging is configured; eval runs will try optional MLflow logging.")
    else:
        st.info("MLflow logging disabled.")

    cases = repository.list_cases(workspace.id)
    with st.expander("Create eval case"):
        _render_eval_case_create(repository, workspace, documents)
    cases = repository.list_cases(workspace.id)
    with st.expander("Import / export eval cases"):
        _render_eval_case_import_export(repository, workspace, cases)
    with st.expander("Edit eval cases"):
        _render_eval_case_editor(repository, workspace, documents, cases)
    with st.expander("Run retrieval evaluation", expanded=True):
        _render_eval_run_controls(repository, retrieval_service, workspace, cases, settings)


def _render_eval_case_create(
    repository: EvaluationRepository,
    workspace: Workspace,
    documents: list[DocumentRecord],
) -> None:
    document_options = {
        f"{document.display_name} ({document.id[:8]})": document.id for document in documents
    }
    with st.form(f"create_eval_case_{workspace.id}", clear_on_submit=True):
        question = st.text_area("Question", key=f"eval_create_question_{workspace.id}")
        selected_labels = st.multiselect(
            "Selected documents",
            list(document_options),
            key=f"eval_create_documents_{workspace.id}",
            help=f"Select up to {MAX_SELECTED_DOCUMENTS} documents.",
        )
        selected_document_ids = [document_options[label] for label in selected_labels]
        expected_filename = st.text_input(
            "Expected filename",
            key=f"eval_create_filename_{workspace.id}",
        )
        expected_page = st.number_input(
            "Expected page (0 for none)",
            min_value=0,
            value=0,
            step=1,
            key=f"eval_create_page_{workspace.id}",
        )
        col_start, col_end = st.columns(2)
        expected_page_start = col_start.number_input(
            "Expected page start (0 for none)",
            min_value=0,
            value=0,
            step=1,
            key=f"eval_create_page_start_{workspace.id}",
        )
        expected_page_end = col_end.number_input(
            "Expected page end (0 for none)",
            min_value=0,
            value=0,
            step=1,
            key=f"eval_create_page_end_{workspace.id}",
        )
        expected_answer = st.text_area(
            "Expected answer (stored for future, unused in Phase 05)",
            key=f"eval_create_answer_{workspace.id}",
        )
        notes = st.text_area("Notes", key=f"eval_create_notes_{workspace.id}")
        if st.form_submit_button("Create eval case"):
            try:
                repository.create_case(
                    NewEvalCase(
                        workspace_id=workspace.id,
                        question=question,
                        selected_document_ids=selected_document_ids,
                        expected_filename=expected_filename,
                        expected_page=_zero_to_none(expected_page),
                        expected_page_start=_zero_to_none(expected_page_start),
                        expected_page_end=_zero_to_none(expected_page_end),
                        expected_answer=expected_answer or None,
                        notes=notes,
                    )
                )
                st.success("Created eval case.")
                st.rerun()
            except EvalValidationError as exc:
                st.error(str(exc))


def _render_eval_case_import_export(
    repository: EvaluationRepository,
    workspace: Workspace,
    cases,
) -> None:
    st.download_button(
        "Export eval cases JSON",
        data=export_cases_json(workspace.id, cases),
        file_name=f"{workspace.name}_eval_cases.json",
        mime="application/json",
        disabled=not cases,
    )
    uploaded = st.file_uploader(
        "Import eval cases JSON",
        type=["json"],
        key=f"eval_import_{workspace.id}",
    )
    if uploaded is not None and st.button(
        "Import eval cases",
        key=f"eval_import_btn_{workspace.id}",
    ):
        new_cases, errors = parse_import_payload(uploaded.getvalue().decode("utf-8"), workspace.id)
        imported_count = 0
        for new_case in new_cases:
            try:
                repository.create_case(new_case)
                imported_count += 1
            except EvalValidationError as exc:
                st.warning(str(exc))
        if imported_count:
            st.success(f"Imported {imported_count} eval cases.")
        for error in errors:
            st.warning(f"Case {error.index}: {error.message}")
        st.rerun()


def _render_eval_case_editor(
    repository: EvaluationRepository,
    workspace: Workspace,
    documents: list[DocumentRecord],
    cases,
) -> None:
    if not cases:
        st.caption("No eval cases yet.")
        return

    with st.expander("Current eval cases"):
        for eval_case in cases:
            st.write(
                {
                    "question": eval_case.question,
                    "expected_filename": eval_case.expected_filename,
                    "expected_page": eval_case.expected_page,
                    "expected_page_start": eval_case.expected_page_start,
                    "expected_page_end": eval_case.expected_page_end,
                    "selected_document_ids": eval_case.selected_document_ids,
                }
            )

    labels = {f"{case.question[:60]} ({case.id[:8]})": case for case in cases}
    selected_label = st.selectbox(
        "Edit eval case",
        list(labels),
        key=f"eval_edit_select_{workspace.id}",
    )
    eval_case = labels[selected_label]
    document_options = {
        f"{document.display_name} ({document.id[:8]})": document.id for document in documents
    }
    default_labels = [
        label
        for label, document_id in document_options.items()
        if document_id in set(eval_case.selected_document_ids)
    ]
    with st.form(f"edit_eval_case_{eval_case.id}"):
        question = st.text_area(
            "Question",
            value=eval_case.question,
            key=f"eval_edit_q_{eval_case.id}",
        )
        selected_labels = st.multiselect(
            "Selected documents",
            list(document_options),
            default=default_labels,
            key=f"eval_edit_docs_{eval_case.id}",
        )
        expected_filename = st.text_input(
            "Expected filename",
            value=eval_case.expected_filename,
            key=f"eval_edit_filename_{eval_case.id}",
        )
        expected_page = st.number_input(
            "Expected page (0 for none)",
            min_value=0,
            value=eval_case.expected_page or 0,
            step=1,
            key=f"eval_edit_page_{eval_case.id}",
        )
        col_start, col_end = st.columns(2)
        expected_page_start = col_start.number_input(
            "Expected page start (0 for none)",
            min_value=0,
            value=eval_case.expected_page_start or 0,
            step=1,
            key=f"eval_edit_page_start_{eval_case.id}",
        )
        expected_page_end = col_end.number_input(
            "Expected page end (0 for none)",
            min_value=0,
            value=eval_case.expected_page_end or 0,
            step=1,
            key=f"eval_edit_page_end_{eval_case.id}",
        )
        expected_answer = st.text_area(
            "Expected answer (stored for future, unused in Phase 05)",
            value=eval_case.expected_answer or "",
            key=f"eval_edit_answer_{eval_case.id}",
        )
        notes = st.text_area("Notes", value=eval_case.notes, key=f"eval_edit_notes_{eval_case.id}")
        if st.form_submit_button("Save eval case"):
            try:
                repository.update_case(
                    eval_case.id,
                    NewEvalCase(
                        workspace_id=workspace.id,
                        question=question,
                        selected_document_ids=[
                            document_options[label] for label in selected_labels
                        ],
                        expected_filename=expected_filename,
                        expected_page=_zero_to_none(expected_page),
                        expected_page_start=_zero_to_none(expected_page_start),
                        expected_page_end=_zero_to_none(expected_page_end),
                        expected_answer=expected_answer or None,
                        notes=notes,
                    ),
                )
                st.success("Updated eval case.")
                st.rerun()
            except EvalValidationError as exc:
                st.error(str(exc))

    confirm = st.checkbox(
        f"Confirm delete eval case '{eval_case.question[:40]}'",
        key=f"eval_delete_confirm_{eval_case.id}",
    )
    if st.button("Delete eval case", disabled=not confirm, key=f"eval_delete_{eval_case.id}"):
        repository.delete_case(eval_case.id)
        st.success("Deleted eval case.")
        st.rerun()


def _render_eval_run_controls(
    repository: EvaluationRepository,
    retrieval_service: RetrievalService | None,
    workspace: Workspace,
    cases,
    settings,
) -> None:
    if not cases:
        return
    case_options = {f"{case.question[:70]} ({case.id[:8]})": case.id for case in cases}
    selected_case_labels = st.multiselect(
        "Cases to run",
        list(case_options),
        default=list(case_options),
        key=f"eval_run_cases_{workspace.id}",
    )
    selected_case_ids = [case_options[label] for label in selected_case_labels]
    with st.expander("Advanced eval retrieval settings"):
        top_k = st.number_input("Eval Top K", min_value=1, max_value=20, value=6, step=1)
        dense_weight = st.slider(
            "Eval dense weight",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
        )
        sparse_weight = st.slider(
            "Eval sparse weight",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )
    if retrieval_service is None:
        st.error(
            "Retrieval service is unavailable; fix embedding device configuration before eval."
        )
        return

    if st.button("Run eval batch", disabled=not selected_case_ids):
        runner = EvaluationRunner(
            settings,
            retrieval_service=retrieval_service,
            mlflow_logger=MLflowEvalLogger(settings.mlflow_tracking_uri),
        )
        result = runner.run_batch(
            workspace_id=workspace.id,
            case_ids=selected_case_ids,
            top_k=int(top_k),
            dense_weight=float(dense_weight),
            sparse_weight=float(sparse_weight),
        )
        st.session_state[f"last_eval_result_{workspace.id}"] = result
        st.rerun()

    result = st.session_state.get(f"last_eval_result_{workspace.id}")
    if result is not None:
        _render_eval_result(result)


def _render_eval_result(result: EvalRunResult) -> None:
    metrics = result.run.metrics
    st.success(f"Eval run completed: {result.run.id}")
    st.info(result.mlflow_status.message)
    st.write(
        {
            "filename_hit_rate@k": metrics.filename_hit_rate,
            "page_hit_rate@k": metrics.page_hit_rate
            if metrics.page_hit_rate is not None
            else "N/A",
            "page_range_hit_rate@k": metrics.page_range_hit_rate
            if metrics.page_range_hit_rate is not None
            else "N/A",
            "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
            "case_count": metrics.eval_case_count,
        }
    )
    with st.expander("Eval run configuration"):
        st.write(
            {
                "embedding_model": result.run.config.embedding_model,
                "embedding_device": result.run.config.embedding_device,
            }
        )
    for item in result.items:
        with st.expander(f"{item.question[:80]} ({item.case_id[:8]})"):
            st.write(
                {
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
                    "warnings": item.warnings,
                }
            )
            st.write("Retrieved results")
            for retrieved in item.retrieved_results:
                st.write(retrieved.to_dict())
            with st.expander("Compact retrieval trace"):
                st.write(item.retrieval_trace)
