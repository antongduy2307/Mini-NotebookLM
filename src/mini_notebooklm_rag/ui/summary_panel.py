"""Document summary Streamlit panel."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.summary import SUMMARY_MODE_OVERVIEW
from mini_notebooklm_rag.summary.service import SummaryService


def render_summary_panel(
    summary_service: SummaryService,
    workspace: Workspace,
    documents: list[DocumentRecord],
    selected_document_ids: list[str],
    api_key: str,
) -> None:
    st.subheader("Document summary")
    st.caption("Generate or view cached per-document overview summaries.")

    if not documents:
        st.caption("Upload documents before generating summaries.")
        return

    document_options = {
        f"{document.display_name} ({document.id[:8]})": document for document in documents
    }
    labels = list(document_options)
    selected_source = next(
        (
            document
            for document in documents
            if selected_document_ids and document.id == selected_document_ids[0]
        ),
        None,
    )
    default_index = next(
        (
            index
            for index, label in enumerate(labels)
            if selected_source is not None and document_options[label].id == selected_source.id
        ),
        0,
    )
    selected_label = st.selectbox(
        "Summary document",
        labels,
        index=default_index,
        key=f"summary_document_{workspace.id}",
    )
    document = document_options[selected_label]
    summary_mode = st.selectbox(
        "Summary mode",
        [SUMMARY_MODE_OVERVIEW],
        key=f"summary_mode_{workspace.id}",
        help="Only overview summaries are implemented in Phase 04.",
    )
    model_name = st.text_input(
        "Summary model",
        value=summary_service.settings.openai_model,
        key=f"summary_model_{workspace.id}",
    )

    cached = summary_service.get_cached_summary(
        document.id,
        summary_mode=summary_mode,
        model_name=model_name,
    )
    if cached is None:
        st.info("Summary status: no cached summary.")
    else:
        st.success("Summary status: cached.")
        _render_cached_summary(cached)

    generate_col, regenerate_col = st.columns(2)
    if generate_col.button("Generate summary", key=f"generate_summary_{document.id}"):
        result = summary_service.generate_for_document(
            document.id,
            api_key=api_key,
            model_name=model_name,
            summary_mode=summary_mode,
            regenerate=False,
        )
        _render_summary_result(result)
        if result.status in {"cached", "generated"}:
            st.rerun()

    if regenerate_col.button("Regenerate summary", key=f"regenerate_summary_{document.id}"):
        result = summary_service.generate_for_document(
            document.id,
            api_key=api_key,
            model_name=model_name,
            summary_mode=summary_mode,
            regenerate=True,
        )
        _render_summary_result(result)
        if result.status == "generated":
            st.rerun()


def _render_summary_result(result) -> None:
    if result.status == "generated":
        st.success(result.message)
    elif result.status == "cached":
        st.info(result.message)
    elif result.status == "skipped":
        st.warning(result.message)
    else:
        st.error(result.message)

    if result.summary is not None:
        _render_cached_summary(result.summary)


def _render_cached_summary(summary) -> None:
    st.markdown(summary.summary_text)
    with st.expander("Summary metadata"):
        st.write(
            {
                "model_name": summary.model_name,
                "summary_mode": summary.summary_mode,
                "prompt_version": summary.prompt_version,
                "source_chunk_count": summary.source_chunk_count,
                "source_character_count": summary.source_character_count,
                "is_partial": summary.is_partial,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "total_tokens": summary.total_tokens,
                "updated_at": summary.updated_at,
            }
        )
    for warning in summary.warnings:
        st.warning(warning)
