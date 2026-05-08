"""Grounded chat Streamlit panel."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.chat.service import ChatService
from mini_notebooklm_rag.qa.service import QAResult, QAService, QAServiceError
from mini_notebooklm_rag.retrieval.models import IndexStatus
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.ui.shared import document_by_id


def render_chat_panel(
    qa_service: QAService,
    chat_service: ChatService,
    workspace: Workspace,
    documents: list[DocumentRecord],
    selected_document_ids: list[str],
    index_status: IndexStatus | None,
    api_key: str,
    settings,
) -> None:
    st.subheader("Chat")
    st.caption("Ask grounded questions over selected documents with source citations.")

    with st.expander("Chat settings"):
        model = st.text_input("Generation model", value=settings.openai_model)
        rewrite_model = st.text_input(
            "Query rewrite model",
            value=settings.openai_query_rewrite_model,
        )
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

    if not selected_document_ids:
        st.info("Select up to 3 source documents in the Sources sidebar before chatting.")
        return

    document_lookup = document_by_id(documents)
    selected_names = [
        document_lookup[document_id].display_name
        for document_id in selected_document_ids
        if document_id in document_lookup
    ]
    st.write("Selected sources: " + ", ".join(selected_names))
    if index_status is not None and index_status.status != "current":
        st.warning(index_status.message)
        for warning in index_status.warnings:
            st.warning(warning)
        st.caption("Build or rebuild the workspace index from the Sources sidebar before chatting.")
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
