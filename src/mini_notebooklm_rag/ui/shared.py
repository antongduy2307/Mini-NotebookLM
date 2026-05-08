"""Shared Streamlit UI helpers."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.retrieval.service import MAX_SELECTED_DOCUMENTS
from mini_notebooklm_rag.storage.repositories import DocumentRecord, Workspace

EMBEDDING_DEVICE_OPTIONS = ["auto", "cuda", "cpu"]


def settings_for_embedding_device(settings, requested_device: str):
    """Return a settings copy with the UI-selected embedding device."""
    return settings.model_copy(update={"embedding_device": requested_device})


def render_api_key_input(settings) -> str:
    temporary_key = st.text_input(
        "Temporary OpenAI API key",
        type="password",
        key="temporary_openai_api_key",
        help=(
            "Used for chat and summaries in this Streamlit session only. "
            "It is not written to SQLite or files."
        ),
    )
    api_key = temporary_key or settings.openai_api_key
    if temporary_key:
        st.caption("OpenAI API key source: temporary session input")
    elif settings.openai_api_key:
        st.caption("OpenAI API key source: .env")
    else:
        st.caption("OpenAI API key source: not configured")
    return api_key


def _embedding_device_index(requested_device: str) -> int:
    requested = requested_device.strip().lower()
    if requested in EMBEDDING_DEVICE_OPTIONS:
        return EMBEDDING_DEVICE_OPTIONS.index(requested)
    return EMBEDDING_DEVICE_OPTIONS.index("auto")


def _zero_to_none(value: int) -> int | None:
    return None if int(value) == 0 else int(value)


def document_label(document: DocumentRecord) -> str:
    """Return a stable user-facing label for document selectors."""
    return f"{document.display_name} ({document.id[:8]})"


def document_options(documents: list[DocumentRecord]) -> dict[str, str]:
    """Map document labels to IDs for Streamlit selectors."""
    return {document_label(document): document.id for document in documents}


def document_by_id(documents: list[DocumentRecord]) -> dict[str, DocumentRecord]:
    """Map document IDs to records."""
    return {document.id: document for document in documents}


def selected_source_state_key(workspace_id: str) -> str:
    """Return the shared selected-source session key for a workspace."""
    return f"selected_source_document_ids_{workspace_id}"


def normalize_selected_document_ids(
    selected_document_ids: list[str],
    documents: list[DocumentRecord],
    max_selected: int = MAX_SELECTED_DOCUMENTS,
) -> list[str]:
    """Filter deleted/unknown document IDs and enforce the shared max selection."""
    valid_ids = {document.id for document in documents}
    normalized: list[str] = []
    for document_id in selected_document_ids:
        if document_id in valid_ids and document_id not in normalized:
            normalized.append(document_id)
        if len(normalized) == max_selected:
            break
    return normalized


def selected_source_labels(
    selected_document_ids: list[str],
    documents: list[DocumentRecord],
) -> list[str]:
    """Return selector labels for selected document IDs."""
    id_to_label = {document.id: document_label(document) for document in documents}
    return [
        id_to_label[document_id]
        for document_id in selected_document_ids
        if document_id in id_to_label
    ]


def render_shared_source_selector(
    workspace: Workspace,
    documents: list[DocumentRecord],
) -> list[str]:
    """Render and persist the shared source selector used by chat and retrieval."""
    key = selected_source_state_key(workspace.id)
    current_ids = normalize_selected_document_ids(st.session_state.get(key, []), documents)
    st.session_state[key] = current_ids

    if not documents:
        st.caption("Upload documents to select sources.")
        return []

    options = document_options(documents)
    label_key = f"{key}_labels"
    if label_key in st.session_state:
        valid_labels = set(options)
        st.session_state[label_key] = [
            label for label in st.session_state[label_key] if label in valid_labels
        ][:MAX_SELECTED_DOCUMENTS]
    selected_labels = st.multiselect(
        "Selected sources",
        list(options),
        default=selected_source_labels(current_ids, documents),
        key=label_key,
        help=f"Select up to {MAX_SELECTED_DOCUMENTS} documents for chat and retrieval.",
    )
    selected_ids = [options[label] for label in selected_labels]
    if len(selected_ids) > MAX_SELECTED_DOCUMENTS:
        st.error(f"Select at most {MAX_SELECTED_DOCUMENTS} source documents.")
        selected_ids = selected_ids[:MAX_SELECTED_DOCUMENTS]

    selected_ids = normalize_selected_document_ids(selected_ids, documents)
    st.session_state[key] = selected_ids
    if selected_ids:
        st.caption(f"{len(selected_ids)} source document(s) selected.")
    return selected_ids
