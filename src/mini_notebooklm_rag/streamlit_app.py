"""Phase 01 Streamlit UI for local workspace and document ingestion."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.config import get_settings
from mini_notebooklm_rag.ingestion.service import IngestionError, IngestionService, WorkspaceService
from mini_notebooklm_rag.storage.repositories import DuplicateWorkspaceError, Workspace


def render() -> None:
    """Render workspace and document ingestion UI."""
    settings = get_settings()
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)

    st.set_page_config(page_title="mini-notebooklm-rag", page_icon="MNR", layout="wide")

    st.title("mini-notebooklm-rag")
    st.caption("Phase 01: workspace and document ingestion foundation")
    st.info(
        "Create a workspace, upload PDF or Markdown documents, and inspect stored chunk counts. "
        "Retrieval and chat are intentionally deferred to later phases."
    )

    workspaces = workspace_service.list_workspaces()
    selected_workspace = _render_workspace_panel(workspace_service, workspaces)

    st.divider()
    if selected_workspace is None:
        st.warning("Create or select a workspace to upload documents.")
    else:
        _render_document_panel(ingestion_service, workspace_service, selected_workspace)

    st.divider()
    st.subheader("Retrieval and chat")
    st.warning("Retrieval, chat, summaries, evaluation, and OpenAI calls are not implemented yet.")


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
) -> None:
    refreshed_workspace = workspace_service.get_workspace(workspace.id)
    if refreshed_workspace is None:
        st.warning("Selected workspace no longer exists.")
        return

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
        return

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


render()
