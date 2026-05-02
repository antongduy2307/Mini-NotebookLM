"""Phase 02 Streamlit UI for ingestion and retrieval debugging."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.config import get_settings
from mini_notebooklm_rag.ingestion.service import IngestionError, IngestionService, WorkspaceService
from mini_notebooklm_rag.retrieval.embeddings import EmbeddingDeviceError
from mini_notebooklm_rag.retrieval.service import (
    MAX_SELECTED_DOCUMENTS,
    RetrievalError,
    RetrievalService,
)
from mini_notebooklm_rag.storage.repositories import (
    DocumentRecord,
    DuplicateWorkspaceError,
    Workspace,
)


def render() -> None:
    """Render workspace and document ingestion UI."""
    st.set_page_config(page_title="mini-notebooklm-rag", page_icon="MNR", layout="wide")

    settings = get_settings()
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)
    retrieval_service: RetrievalService | None
    try:
        retrieval_service = RetrievalService(settings)
    except EmbeddingDeviceError as exc:
        retrieval_service = None
        st.error(str(exc))

    st.title("mini-notebooklm-rag")
    st.caption("Phase 02: local ingestion plus retrieval debugging foundation")
    st.info(
        "Create a workspace, upload PDF or Markdown documents, build a local retrieval index, "
        "and inspect retrieved chunks. Answer generation and chat are intentionally deferred."
    )

    workspaces = workspace_service.list_workspaces()
    selected_workspace = _render_workspace_panel(workspace_service, workspaces)

    st.divider()
    if selected_workspace is None:
        st.warning("Create or select a workspace to upload documents.")
    else:
        documents = _render_document_panel(
            ingestion_service,
            workspace_service,
            selected_workspace,
        )
        st.divider()
        if retrieval_service is not None:
            _render_retrieval_panel(retrieval_service, selected_workspace, documents)

    st.divider()
    st.subheader("Answer generation and chat")
    st.warning(
        "Answer generation, chat, summaries, evaluation, MLflow, and OpenAI calls are Phase 03+."
    )


def _render_workspace_panel(
    workspace_service: WorkspaceService,
    workspaces: list[Workspace],
) -> Workspace | None:
    st.subheader("Workspace")

    selected_id = st.session_state.get("selected_workspace_id")
    if workspaces and selected_id not in {workspace.id for workspace in workspaces}:
        selected_id = workspaces[0].id
        st.session_state["selected_workspace_id"] = selected_id

    workspace_by_label = {
        f"{workspace.name} ({workspace.id[:8]})": workspace for workspace in workspaces
    }
    labels = list(workspace_by_label)
    if labels:
        current_index = next(
            (
                index
                for index, label in enumerate(labels)
                if workspace_by_label[label].id == selected_id
            ),
            0,
        )
        selected_label = st.selectbox("Select workspace", labels, index=current_index)
        selected_workspace = workspace_by_label[selected_label]
        st.session_state["selected_workspace_id"] = selected_workspace.id
    else:
        selected_workspace = None

    with st.form("create_workspace", clear_on_submit=True):
        name = st.text_input("Create workspace", placeholder="Research notes")
        submitted = st.form_submit_button("Create")
        if submitted:
            try:
                workspace = workspace_service.create_workspace(name)
                st.session_state["selected_workspace_id"] = workspace.id
                st.success(f"Created workspace: {workspace.name}")
                st.rerun()
            except DuplicateWorkspaceError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))

    if selected_workspace is not None:
        st.write(f"Selected workspace ID: `{selected_workspace.id}`")
        confirm_key = f"confirm_delete_workspace_{selected_workspace.id}"
        confirm_delete = st.checkbox(
            f"Confirm delete workspace '{selected_workspace.name}'",
            key=confirm_key,
        )
        if st.button("Delete selected workspace", disabled=not confirm_delete):
            workspace_service.delete_workspace(selected_workspace.id)
            st.session_state.pop("selected_workspace_id", None)
            st.success(f"Deleted workspace: {selected_workspace.name}")
            st.rerun()

    return selected_workspace


def _render_document_panel(
    ingestion_service: IngestionService,
    workspace_service: WorkspaceService,
    workspace: Workspace,
) -> list[DocumentRecord]:
    refreshed_workspace = workspace_service.get_workspace(workspace.id)
    if refreshed_workspace is None:
        st.warning("Selected workspace no longer exists.")
        return []

    st.subheader("Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF or Markdown",
        type=["pdf", "md", "markdown"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None and st.button("Ingest uploaded document"):
        try:
            result = ingestion_service.ingest_upload(
                uploaded_file.getvalue(),
                uploaded_file.name,
                refreshed_workspace.id,
            )
            if result.status == "duplicate":
                st.warning(result.message)
            else:
                st.success(result.message)
                for warning in result.warnings:
                    st.warning(warning)
            st.rerun()
        except IngestionError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Document ingestion failed: {exc}")

    documents = ingestion_service.list_documents(refreshed_workspace.id)
    if not documents:
        st.caption("No documents uploaded yet.")
        return []

    for document in documents:
        with st.container(border=True):
            st.write(f"**{document.display_name}**")
            st.write(
                {
                    "type": document.source_type,
                    "chunks": document.chunk_count,
                    "size_bytes": document.size_bytes,
                    "created_at": document.created_at,
                }
            )
            confirm_key = f"confirm_delete_document_{document.id}"
            confirm_delete = st.checkbox(
                f"Confirm delete '{document.display_name}'",
                key=confirm_key,
            )
            if st.button(
                "Delete document",
                key=f"delete_document_{document.id}",
                disabled=not confirm_delete,
            ):
                ingestion_service.delete_document(document.id)
                st.success(f"Deleted document: {document.display_name}")
                st.rerun()

    return documents


def _render_retrieval_panel(
    retrieval_service: RetrievalService,
    workspace: Workspace,
    documents: list[DocumentRecord],
) -> None:
    st.subheader("Retrieval debug")
    st.caption("Build a local FAISS index and inspect hybrid dense/BM25 retrieval results.")
    st.warning("This is not a chat interface. Answer generation is planned for Phase 03.")

    embedding_info = retrieval_service.embedding_info
    st.write(
        {
            "embedding_model": embedding_info.model_name,
            "requested_device": embedding_info.requested_device,
            "selected_device": embedding_info.selected_device,
            "dimension": embedding_info.dimension,
            "normalized": embedding_info.normalized,
        }
    )

    status = retrieval_service.index_status(workspace.id)
    status_method = st.success if status.status == "current" else st.warning
    if status.status == "empty":
        status_method = st.info
    status_method(status.message)
    if status.warnings:
        for warning in status.warnings:
            st.warning(warning)
    st.write(
        {
            "index_status": status.status,
            "workspace_chunks": status.chunk_count,
            "indexed_chunks": status.indexed_chunk_count,
        }
    )

    if st.button("Build or rebuild workspace index"):
        with st.spinner("Building FAISS index from workspace chunks..."):
            try:
                build_status = retrieval_service.rebuild_index(workspace.id)
                st.success(build_status.message)
                st.rerun()
            except Exception as exc:
                st.error(f"Index build failed: {exc}")

    if not documents:
        st.caption("Upload documents before running retrieval.")
        return

    document_options = {
        f"{document.display_name} ({document.id[:8]})": document.id for document in documents
    }
    selected_labels = st.multiselect(
        "Select documents for retrieval",
        list(document_options),
        help=f"Select up to {MAX_SELECTED_DOCUMENTS} documents.",
    )
    selected_document_ids = [document_options[label] for label in selected_labels]
    if len(selected_document_ids) > MAX_SELECTED_DOCUMENTS:
        st.error(f"Select at most {MAX_SELECTED_DOCUMENTS} documents.")
        return

    query = st.text_input("Retrieval query")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=6, step=1)
    dense_weight = st.slider("Dense weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    sparse_weight = st.slider(
        "Sparse weight",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    if st.button("Run retrieval"):
        try:
            response = retrieval_service.retrieve(
                workspace_id=workspace.id,
                query=query,
                selected_document_ids=selected_document_ids,
                top_k=int(top_k),
                dense_weight=float(dense_weight),
                sparse_weight=float(sparse_weight),
            )
        except RetrievalError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            return

        for warning in response.warnings:
            st.warning(warning)
        if not response.results:
            st.caption("No retrieval results to display.")
            return

        for result in response.results:
            with st.container(border=True):
                st.write(f"**#{result.rank} {result.citation}**")
                st.write(
                    {
                        "dense_score": round(result.dense_score, 4),
                        "sparse_score": round(result.sparse_score, 4),
                        "fused_score": round(result.fused_score, 4),
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                    }
                )
                with st.expander("Source chunk"):
                    st.write(result.text)


render()
