"""Streamlit UI for ingestion, retrieval debugging, and grounded chat."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.chat.service import ChatService
from mini_notebooklm_rag.config import get_settings
from mini_notebooklm_rag.ingestion.service import IngestionError, IngestionService, WorkspaceService
from mini_notebooklm_rag.qa.service import QAResult, QAService, QAServiceError
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

EMBEDDING_DEVICE_OPTIONS = ["auto", "cuda", "cpu"]


def render() -> None:
    """Render workspace and document ingestion UI."""
    st.set_page_config(page_title="mini-notebooklm-rag", page_icon="MNR", layout="wide")

    settings = get_settings()
    st.title("mini-notebooklm-rag")
    st.caption("Phase 03: grounded QA and workspace chat over local retrieval")
    st.info(
        "Create a workspace, upload PDF or Markdown documents, build a local retrieval index, "
        "inspect retrieved chunks, and ask grounded questions with citations."
    )

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
    retrieval_service: RetrievalService | None
    try:
        retrieval_service = RetrievalService(runtime_settings)
    except EmbeddingDeviceError as exc:
        retrieval_service = None
        st.error(str(exc))

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
            qa_service = QAService(runtime_settings, chat_service, retrieval_service)
            _render_chat_panel(qa_service, chat_service, selected_workspace, documents)

    st.divider()
    st.subheader("Later phases")
    st.warning("Summaries, evaluation, MLflow, and deployment are Phase 04+.")


def settings_for_embedding_device(settings, requested_device: str):
    """Return a settings copy with the UI-selected embedding device."""
    return settings.model_copy(update={"embedding_device": requested_device})


def _embedding_device_index(requested_device: str) -> int:
    requested = requested_device.strip().lower()
    if requested in EMBEDDING_DEVICE_OPTIONS:
        return EMBEDDING_DEVICE_OPTIONS.index(requested)
    return EMBEDDING_DEVICE_OPTIONS.index("auto")


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
    st.warning("This panel is for retrieval debugging. Use Chat below for grounded answers.")

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


def _render_chat_panel(
    qa_service: QAService,
    chat_service: ChatService,
    workspace: Workspace,
    documents: list[DocumentRecord],
) -> None:
    st.subheader("Chat")
    st.caption("Ask grounded questions over selected documents with source citations.")

    settings = get_settings()
    temporary_key = st.text_input(
        "Temporary OpenAI API key",
        type="password",
        key="temporary_openai_api_key",
        help="Stored only in Streamlit session state. It is not written to SQLite or files.",
    )
    api_key = temporary_key or settings.openai_api_key
    if temporary_key:
        st.caption("API key source: temporary session input")
    elif settings.openai_api_key:
        st.caption("API key source: .env")
    else:
        st.caption("API key source: not configured")

    model = st.text_input("Generation model", value=settings.openai_model)
    rewrite_model = st.text_input("Query rewrite model", value=settings.openai_query_rewrite_model)
    allow_outside = st.toggle(
        "Allow outside knowledge",
        value=settings.allow_outside_knowledge,
        help="When enabled, answers separate document-grounded content from outside knowledge.",
    )
    enable_rewrite = st.toggle(
        "Enable query rewrite",
        value=settings.enable_query_rewrite,
        help="Uses only the current chat session history.",
    )

    if not documents:
        st.caption("Upload documents before starting a chat.")
        return

    document_options = {
        f"{document.display_name} ({document.id[:8]})": document.id for document in documents
    }
    selected_labels = st.multiselect(
        "Chat documents",
        list(document_options),
        help=f"Select up to {MAX_SELECTED_DOCUMENTS} documents for this chat.",
        key=f"chat_documents_{workspace.id}",
    )
    selected_document_ids = [document_options[label] for label in selected_labels]
    if len(selected_document_ids) > MAX_SELECTED_DOCUMENTS:
        st.error(f"Select at most {MAX_SELECTED_DOCUMENTS} documents.")
        return

    sessions = chat_service.list_sessions(workspace.id)
    session_labels = {f"{session.title} ({session.id[:8]})": session for session in sessions}
    selected_session_id = st.session_state.get(f"selected_chat_session_{workspace.id}")
    if session_labels:
        labels = list(session_labels)
        current_index = next(
            (
                index
                for index, label in enumerate(labels)
                if session_labels[label].id == selected_session_id
            ),
            0,
        )
        selected_label = st.selectbox("Chat history", labels, index=current_index)
        session = session_labels[selected_label]
        st.session_state[f"selected_chat_session_{workspace.id}"] = session.id
    else:
        session = None
        st.caption("No chat sessions yet.")

    col_new, col_delete = st.columns(2)
    if col_new.button("New chat"):
        session = chat_service.create_session(
            workspace_id=workspace.id,
            selected_document_ids=selected_document_ids,
            title="New chat",
        )
        st.session_state[f"selected_chat_session_{workspace.id}"] = session.id
        st.rerun()

    if session is not None and col_delete.button("Delete selected chat"):
        chat_service.delete_session(session.id)
        st.session_state.pop(f"selected_chat_session_{workspace.id}", None)
        st.rerun()

    if session is None:
        st.info("Create a new chat to ask questions.")
        return

    messages = chat_service.list_messages(session.id)
    for message in messages:
        with st.chat_message(
            message.role if message.role in {"user", "assistant"} else "assistant"
        ):
            st.write(message.content)
            if message.role == "assistant" and message.source_map:
                with st.expander("Sources"):
                    for source in message.source_map:
                        st.write(f"[{source['source_id']}] {source['citation']}")

    result = st.session_state.get(f"last_qa_result_{session.id}")
    if result is not None:
        _render_qa_debug(result)

    question = st.chat_input("Ask a question about the selected documents")
    if not question:
        return

    try:
        result = qa_service.answer_question(
            workspace_id=workspace.id,
            session_id=session.id,
            question=question,
            selected_document_ids=selected_document_ids,
            api_key=api_key,
            model=model,
            rewrite_model=rewrite_model,
            allow_outside_knowledge=allow_outside,
            enable_query_rewrite=enable_rewrite,
        )
        st.session_state[f"last_qa_result_{session.id}"] = result
        st.rerun()
    except QAServiceError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Chat request failed: {exc}")


def _render_qa_debug(result: QAResult) -> None:
    with st.expander("Latest answer sources and debug"):
        st.write("Sources")
        for source in result.sources:
            st.write(f"[{source.source_id}] {source.citation}")
            with st.expander(f"Chunk {source.chunk_id}"):
                st.write(source.text)
                st.write(
                    {
                        "dense_score": round(source.dense_score, 4),
                        "sparse_score": round(source.sparse_score, 4),
                        "fused_score": round(source.fused_score, 4),
                    }
                )
        st.write(
            {
                "original_query": result.original_query,
                "rewritten_query": result.rewritten_query,
                "rewrite_skipped_reason": result.rewrite_skipped_reason,
                "clarification_question": result.clarification_question,
                "model_name": result.model_name,
                "token_usage": {
                    "input_tokens": result.token_usage.input_tokens,
                    "output_tokens": result.token_usage.output_tokens,
                    "total_tokens": result.token_usage.total_tokens,
                },
                "warnings": result.warnings,
                "prompt_metadata": result.prompt_metadata,
                "retrieval_metadata": result.retrieval_metadata,
            }
        )


if __name__ == "__main__":
    render()
