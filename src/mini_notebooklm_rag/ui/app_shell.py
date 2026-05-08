"""Top-level Streamlit application shell."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mini_notebooklm_rag.chat.service import ChatService
from mini_notebooklm_rag.config import get_settings
from mini_notebooklm_rag.evaluation.repositories import EvaluationRepository
from mini_notebooklm_rag.ingestion.service import IngestionService, WorkspaceService
from mini_notebooklm_rag.qa.service import QAService
from mini_notebooklm_rag.retrieval.embeddings import EmbeddingDeviceError
from mini_notebooklm_rag.retrieval.models import IndexStatus
from mini_notebooklm_rag.retrieval.service import RetrievalService
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.summary.service import SummaryService
from mini_notebooklm_rag.ui.chat_panel import render_chat_panel
from mini_notebooklm_rag.ui.document_panel import render_document_panel
from mini_notebooklm_rag.ui.evaluation_panel import render_evaluation_panel
from mini_notebooklm_rag.ui.retrieval_panel import render_index_controls, render_retrieval_panel
from mini_notebooklm_rag.ui.shared import (
    EMBEDDING_DEVICE_OPTIONS,
    _embedding_device_index,
    document_by_id,
    render_api_key_input,
    render_shared_source_selector,
    settings_for_embedding_device,
)
from mini_notebooklm_rag.ui.summary_panel import render_summary_panel
from mini_notebooklm_rag.ui.workspace_panel import render_workspace_panel


def render() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="mini-notebooklm-rag", page_icon="MNR", layout="wide")

    settings = get_settings()
    with st.sidebar:
        st.markdown("# mini-notebooklm-rag")
        st.caption("Local sources, grounded chat, and studio tools.")
        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
        with st.expander("Model/API settings"):
            api_key = render_api_key_input(settings)
            requested_embedding_device = st.selectbox(
                "Embedding device",
                EMBEDDING_DEVICE_OPTIONS,
                index=_embedding_device_index(settings.embedding_device),
                help="auto uses CUDA when available and falls back to CPU. FAISS remains CPU-only.",
            )
    runtime_settings = settings_for_embedding_device(settings, requested_embedding_device)

    workspace_service = WorkspaceService(runtime_settings)
    ingestion_service = IngestionService(runtime_settings)
    chat_service = ChatService(runtime_settings)
    summary_service = SummaryService(runtime_settings)
    evaluation_repository = EvaluationRepository(
        StoragePaths(Path(runtime_settings.app_storage_dir)).db_path
    )
    retrieval_service: RetrievalService | None
    retrieval_error: str | None = None
    try:
        retrieval_service = RetrievalService(runtime_settings)
    except EmbeddingDeviceError as exc:
        retrieval_service = None
        retrieval_error = str(exc)

    workspaces = workspace_service.list_workspaces()
    documents: list[DocumentRecord] = []
    selected_document_ids: list[str] = []
    index_status: IndexStatus | None = None
    with st.sidebar:
        st.divider()
        selected_workspace = render_workspace_panel(workspace_service, workspaces)
        if selected_workspace is None:
            st.info("Create or select a workspace to add sources.")
        else:
            st.divider()
            documents = render_document_panel(
                ingestion_service,
                workspace_service,
                summary_service,
                selected_workspace,
                api_key,
                runtime_settings.auto_summary,
                runtime_settings.openai_model,
            )
            st.divider()
            st.caption("Shared source selection")
            selected_document_ids = render_shared_source_selector(selected_workspace, documents)
            st.divider()
            if retrieval_service is None:
                st.error(retrieval_error or "Retrieval service is unavailable.")
            else:
                render_index_controls(retrieval_service, selected_workspace)
                index_status = retrieval_service.index_status(selected_workspace.id)
            st.divider()
            _render_sidebar_status(
                selected_workspace,
                selected_document_ids,
                documents,
                retrieval_service,
                index_status,
                api_key,
                runtime_settings.mlflow_tracking_uri,
            )

    if selected_workspace is None:
        st.warning("Create or select a workspace from the Sources sidebar.")
        return

    chat_column, studio_column = st.columns([0.70, 0.30], gap="large")
    with chat_column:
        if retrieval_service is not None:
            qa_service = QAService(runtime_settings, chat_service, retrieval_service)
            render_chat_panel(
                qa_service,
                chat_service,
                selected_workspace,
                documents,
                selected_document_ids,
                index_status,
                api_key,
                runtime_settings,
            )
        else:
            st.error(retrieval_error or "Retrieval service is unavailable.")

    with studio_column:
        st.subheader("Studio")
        summary_tab, retrieval_tab, evaluation_tab, dev_tab = st.tabs(
            ["Summary", "Retrieval Debug", "Evaluation", "Dev Info"]
        )
        with summary_tab:
            render_summary_panel(
                summary_service,
                selected_workspace,
                documents,
                selected_document_ids,
                api_key,
            )
        with retrieval_tab:
            if retrieval_service is None:
                st.error(retrieval_error or "Retrieval service is unavailable.")
            else:
                render_retrieval_panel(
                    retrieval_service,
                    selected_workspace,
                    documents,
                    selected_document_ids,
                )
        with evaluation_tab:
            render_evaluation_panel(
                evaluation_repository,
                retrieval_service,
                selected_workspace,
                documents,
                runtime_settings,
            )
        with dev_tab:
            _render_dev_info(
                selected_workspace,
                selected_document_ids,
                documents,
                retrieval_service,
                index_status,
                api_key,
                runtime_settings.mlflow_tracking_uri,
            )


def _render_sidebar_status(
    workspace: Workspace,
    selected_document_ids: list[str],
    documents: list[DocumentRecord],
    retrieval_service: RetrievalService | None,
    index_status: IndexStatus | None,
    api_key: str,
    mlflow_tracking_uri: str,
) -> None:
    api_status = "configured" if api_key else "missing"
    index_label = index_status.status if index_status is not None else "unavailable"
    embedding_device = (
        retrieval_service.embedding_info.selected_device
        if retrieval_service is not None
        else "unavailable"
    )

    with st.expander("Workspace and runtime status", expanded=True):
        st.write(f"Workspace: **{workspace.name}**")
        st.write(f"Sources selected: **{len(selected_document_ids)}**")
        st.write(f"Embedding device: **{embedding_device}**")
        st.write(f"Index: **{index_label}**")
        st.write(f"API key: **{api_status}**")
        st.write(f"MLflow: **{'configured' if mlflow_tracking_uri else 'disabled'}**")

        if selected_document_ids:
            document_lookup = document_by_id(documents)
            selected_names = [
                document_lookup[document_id].display_name
                for document_id in selected_document_ids
                if document_id in document_lookup
            ]
            st.caption("Selected: " + ", ".join(selected_names))


def _render_dev_info(
    workspace: Workspace,
    selected_document_ids: list[str],
    documents: list[DocumentRecord],
    retrieval_service: RetrievalService | None,
    index_status: IndexStatus | None,
    api_key: str,
    mlflow_tracking_uri: str,
) -> None:
    st.caption("Compact app status and future placeholders.")
    document_lookup = document_by_id(documents)
    selected_sources = [
        {
            "id": document_id,
            "display_name": document_lookup[document_id].display_name,
        }
        for document_id in selected_document_ids
        if document_id in document_lookup
    ]
    st.write(
        {
            "workspace": {"id": workspace.id, "name": workspace.name},
            "selected_sources": selected_sources,
            "api_key": "configured" if api_key else "missing",
            "mlflow": "configured" if mlflow_tracking_uri else "disabled",
        }
    )
    if retrieval_service is not None:
        embedding_info = retrieval_service.embedding_info
        st.write(
            {
                "embedding_model": embedding_info.model_name,
                "requested_device": embedding_info.requested_device,
                "selected_device": embedding_info.selected_device,
                "embedding_dimension": embedding_info.dimension,
                "normalized_embeddings": embedding_info.normalized,
            }
        )
    if index_status is not None:
        st.write(
            {
                "index_status": index_status.status,
                "index_message": index_status.message,
                "workspace_chunks": index_status.chunk_count,
                "indexed_chunks": index_status.indexed_chunk_count,
                "warnings": list(index_status.warnings),
            }
        )
    st.info("Learning tools coming next: Quiz, Flashcards, Export.")
