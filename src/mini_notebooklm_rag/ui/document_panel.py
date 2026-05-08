"""Document ingestion Streamlit panel."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.ingestion.service import IngestionError, IngestionService, WorkspaceService
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.summary.service import SummaryService


def render_document_panel(
    ingestion_service: IngestionService,
    workspace_service: WorkspaceService,
    summary_service: SummaryService,
    workspace: Workspace,
    api_key: str,
    auto_summary: bool,
    summary_model: str,
) -> list[DocumentRecord]:
    refreshed_workspace = workspace_service.get_workspace(workspace.id)
    if refreshed_workspace is None:
        st.warning("Selected workspace no longer exists.")
        return []

    st.caption("Documents")
    auto_summary_status_key = f"auto_summary_status_{workspace.id}"
    if auto_summary_status_key in st.session_state:
        status = st.session_state.pop(auto_summary_status_key)
        if status["level"] == "success":
            st.success(status["message"])
        elif status["level"] == "warning":
            st.warning(status["message"])
        else:
            st.error(status["message"])

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
                if auto_summary and result.document is not None:
                    summary_result = summary_service.generate_for_document(
                        result.document.id,
                        api_key=api_key,
                        model_name=summary_model,
                    )
                    st.session_state[auto_summary_status_key] = {
                        "level": "success"
                        if summary_result.status in {"cached", "generated"}
                        else "warning"
                        if summary_result.status == "skipped"
                        else "error",
                        "message": (
                            f"Auto-summary for {result.document.display_name}: "
                            f"{summary_result.message}"
                        ),
                    }
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
        with st.expander(f"{document.display_name} ({document.chunk_count} chunks)"):
            st.write(
                {
                    "type": document.source_type,
                    "chunks": document.chunk_count,
                    "size_bytes": document.size_bytes,
                    "created_at": document.created_at,
                    "document_id": document.id,
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
