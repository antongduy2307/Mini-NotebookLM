"""Retrieval debug Streamlit panel."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.retrieval.service import (
    RetrievalError,
    RetrievalService,
)
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace
from mini_notebooklm_rag.ui.shared import document_by_id


def render_index_controls(
    retrieval_service: RetrievalService,
    workspace: Workspace,
) -> None:
    """Render compact FAISS index status and rebuild controls."""
    st.caption("Index")
    status = retrieval_service.index_status(workspace.id)
    status_method = st.success if status.status == "current" else st.warning
    if status.status == "empty":
        status_method = st.info
    status_method(status.message)
    if status.warnings:
        for warning in status.warnings:
            st.warning(warning)
    st.caption(f"{status.indexed_chunk_count}/{status.chunk_count} chunks indexed.")

    if st.button("Build or rebuild workspace index"):
        with st.spinner("Building FAISS index from workspace chunks..."):
            try:
                build_status = retrieval_service.rebuild_index(workspace.id)
                st.success(build_status.message)
                st.rerun()
            except Exception as exc:
                st.error(f"Index build failed: {exc}")


def render_retrieval_panel(
    retrieval_service: RetrievalService,
    workspace: Workspace,
    documents: list[DocumentRecord],
    selected_document_ids: list[str],
) -> None:
    st.subheader("Retrieval Debug")
    st.caption("Inspect hybrid dense/BM25 retrieval results.")

    with st.expander("Embedding and index details"):
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
        st.write(
            {
                "index_status": status.status,
                "workspace_chunks": status.chunk_count,
                "indexed_chunks": status.indexed_chunk_count,
            }
        )
        for warning in status.warnings:
            st.warning(warning)

    if not documents:
        st.caption("Upload documents before running retrieval.")
        return

    if not selected_document_ids:
        st.info("Select up to 3 source documents in the Sources sidebar before retrieval.")
        return

    document_lookup = document_by_id(documents)
    selected_names = [
        document_lookup[document_id].display_name
        for document_id in selected_document_ids
        if document_id in document_lookup
    ]
    st.write("Selected sources: " + ", ".join(selected_names))

    query = st.text_input("Retrieval query")
    with st.expander("Advanced retrieval settings"):
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=6, step=1)
        dense_weight = st.slider(
            "Dense weight",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
        )
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
                st.caption(f"Fused score: {result.fused_score:.4f}")
                with st.expander("Scores and IDs"):
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
