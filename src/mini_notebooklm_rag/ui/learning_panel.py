"""Learning Tools Streamlit panel for quiz and flashcard generation."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.export import (
    artifact_export_filename,
    flashcard_set_to_json_string,
    flashcard_set_to_markdown,
    quiz_set_to_json_string,
    quiz_set_to_markdown,
)
from mini_notebooklm_rag.learning.models import ARTIFACT_FLASHCARDS, ARTIFACT_QUIZ, LearningResult
from mini_notebooklm_rag.learning.service import LearningService
from mini_notebooklm_rag.retrieval.models import IndexStatus
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.ui.shared import document_by_id


def render_learning_panel(
    learning_service: LearningService,
    workspace: Workspace,
    documents: list[DocumentRecord],
    selected_document_ids: list[str],
    index_status: IndexStatus | None,
    api_key: str,
) -> None:
    """Render query-focused quiz and flashcard generation."""
    st.subheader("Learning Tools")
    st.caption("Generate study artifacts from selected sources. Chat stays in the center panel.")

    _render_preflight_status(selected_document_ids, index_status, api_key)
    if not documents:
        st.caption("Upload documents before generating learning artifacts.")
        return
    if not selected_document_ids:
        st.info("Select up to 3 source documents in the Sources sidebar first.")
        return

    document_lookup = document_by_id(documents)
    selected_names = [
        document_lookup[document_id].display_name
        for document_id in selected_document_ids
        if document_id in document_lookup
    ]
    st.write("Selected sources: " + ", ".join(selected_names))

    artifact_label = st.radio(
        "Artifact type",
        ["Quiz", "Flashcards"],
        horizontal=True,
        key=f"learning_artifact_type_{workspace.id}",
    )
    artifact_type = ARTIFACT_QUIZ if artifact_label == "Quiz" else ARTIFACT_FLASHCARDS
    topic_or_query = st.text_input(
        "Topic or question",
        key=f"learning_topic_or_query_{workspace.id}",
        placeholder="Example: hybrid retrieval and citations",
    )
    if artifact_type == ARTIFACT_QUIZ:
        requested_count = st.number_input("Quiz item count", min_value=1, max_value=20, value=5)
    else:
        requested_count = st.number_input("Flashcard count", min_value=1, max_value=50, value=10)

    with st.expander("Advanced learning settings"):
        model_name = st.text_input(
            "Generation model",
            value=learning_service.settings.openai_model,
            key=f"learning_model_{workspace.id}",
        )
        top_k = st.number_input(
            "Retrieval top K",
            min_value=1,
            max_value=20,
            value=learning_service.settings.retrieval_top_k,
            step=1,
            key=f"learning_top_k_{workspace.id}",
        )
        dense_weight = st.slider(
            "Dense weight",
            min_value=0.0,
            max_value=1.0,
            value=float(learning_service.settings.dense_weight),
            step=0.05,
            key=f"learning_dense_weight_{workspace.id}",
        )
        sparse_weight = st.slider(
            "Sparse weight",
            min_value=0.0,
            max_value=1.0,
            value=float(learning_service.settings.sparse_weight),
            step=0.05,
            key=f"learning_sparse_weight_{workspace.id}",
        )

    if st.button("Generate learning artifact", key=f"generate_learning_{workspace.id}"):
        result = _generate(
            learning_service,
            artifact_type,
            workspace.id,
            selected_document_ids,
            topic_or_query,
            api_key,
            int(requested_count),
            model_name,
            int(top_k),
            float(dense_weight),
            float(sparse_weight),
        )
        st.session_state[_result_key(workspace.id, artifact_type)] = result

    result = st.session_state.get(_result_key(workspace.id, artifact_type))
    if result is not None:
        _render_result(workspace, result)


def _generate(
    learning_service: LearningService,
    artifact_type: str,
    workspace_id: str,
    selected_document_ids: list[str],
    topic_or_query: str,
    api_key: str,
    requested_count: int,
    model_name: str,
    top_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> LearningResult:
    with st.spinner("Generating grounded learning artifact..."):
        if artifact_type == ARTIFACT_QUIZ:
            return learning_service.generate_quiz(
                workspace_id=workspace_id,
                selected_document_ids=selected_document_ids,
                topic_or_query=topic_or_query,
                api_key=api_key,
                item_count=requested_count,
                model_name=model_name,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        return learning_service.generate_flashcards(
            workspace_id=workspace_id,
            selected_document_ids=selected_document_ids,
            topic_or_query=topic_or_query,
            api_key=api_key,
            card_count=requested_count,
            model_name=model_name,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )


def _render_preflight_status(
    selected_document_ids: list[str],
    index_status: IndexStatus | None,
    api_key: str,
) -> None:
    index_label = index_status.status if index_status is not None else "unavailable"
    st.write(
        {
            "selected_sources": len(selected_document_ids),
            "index_status": index_label,
            "api_key": "configured" if api_key else "missing",
        }
    )
    if index_status is not None and index_status.status != "current":
        st.warning(index_status.message)


def _render_result(workspace: Workspace, result: LearningResult) -> None:
    if result.status == "generated":
        st.success(result.message)
    elif result.status == "skipped":
        st.warning(result.message)
    else:
        st.error(result.message)

    for warning in result.warnings:
        st.warning(warning)

    if result.quiz_set is not None:
        _render_quiz(workspace, result.quiz_set)
    if result.flashcard_set is not None:
        _render_flashcards(workspace, result.flashcard_set)

    with st.expander("Learning generation metadata"):
        st.write(
            {
                "model_name": result.model_name,
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "total_tokens": result.token_usage.total_tokens,
            }
        )


def _render_quiz(workspace: Workspace, quiz_set) -> None:
    st.write(f"Topic/query: **{quiz_set.topic_or_query}**")
    for index, item in enumerate(quiz_set.items, start=1):
        with st.container(border=True):
            st.write(f"**{index}. {item.question}**")
            for option_index, option in enumerate(item.options):
                prefix = "ABCD"[option_index]
                marker = " (correct)" if option_index == item.correct_index else ""
                st.write(f"{prefix}. {option}{marker}")
            st.caption(f"Sources: {', '.join(item.source_markers)}")
            with st.expander("Explanation"):
                st.write(item.explanation)
    _render_sources(quiz_set.source_map)
    _render_downloads(
        workspace,
        "quiz",
        quiz_set_to_markdown(quiz_set),
        quiz_set_to_json_string(quiz_set),
    )


def _render_flashcards(workspace: Workspace, flashcard_set) -> None:
    st.write(f"Topic/query: **{flashcard_set.topic_or_query}**")
    for index, card in enumerate(flashcard_set.cards, start=1):
        with st.container(border=True):
            st.write(f"**{index}. {card.front}**")
            st.write(card.back)
            if card.hint:
                st.caption(f"Hint: {card.hint}")
            st.caption(f"Sources: {', '.join(card.source_markers)}")
    _render_sources(flashcard_set.source_map)
    _render_downloads(
        workspace,
        "flashcards",
        flashcard_set_to_markdown(flashcard_set),
        flashcard_set_to_json_string(flashcard_set),
    )


def _render_sources(source_map) -> None:
    with st.expander("Sources and chunks"):
        for source in source_map:
            st.write(f"[{source.source_id}] {source.citation}")
            with st.expander(f"Chunk {source.chunk_id}"):
                st.write(source.text)
                st.write(
                    {
                        "document_id": source.document_id,
                        "filename": source.filename,
                        "dense_score": round(source.dense_score, 4),
                        "sparse_score": round(source.sparse_score, 4),
                        "fused_score": round(source.fused_score, 4),
                    }
                )


def _render_downloads(
    workspace: Workspace,
    artifact_type: str,
    markdown_text: str,
    json_text: str,
) -> None:
    col_md, col_json = st.columns(2)
    col_md.download_button(
        "Download Markdown",
        data=markdown_text,
        file_name=artifact_export_filename(workspace.name, artifact_type, "md"),
        mime="text/markdown",
    )
    col_json.download_button(
        "Download JSON",
        data=json_text,
        file_name=artifact_export_filename(workspace.name, artifact_type, "json"),
        mime="application/json",
    )


def _result_key(workspace_id: str, artifact_type: str) -> str:
    return f"last_learning_{artifact_type}_set_{workspace_id}"
