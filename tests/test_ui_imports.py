from __future__ import annotations

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.storage.repositories import DocumentRecord
from mini_notebooklm_rag.streamlit_app import render as streamlit_render
from mini_notebooklm_rag.ui import app_shell
from mini_notebooklm_rag.ui.chat_panel import render_chat_panel
from mini_notebooklm_rag.ui.document_panel import render_document_panel
from mini_notebooklm_rag.ui.evaluation_panel import render_evaluation_panel
from mini_notebooklm_rag.ui.retrieval_panel import render_index_controls, render_retrieval_panel
from mini_notebooklm_rag.ui.shared import (
    document_label,
    document_options,
    normalize_selected_document_ids,
    selected_source_state_key,
    settings_for_embedding_device,
)
from mini_notebooklm_rag.ui.summary_panel import render_summary_panel
from mini_notebooklm_rag.ui.workspace_panel import render_workspace_panel


def test_streamlit_entrypoint_delegates_to_ui_app_shell() -> None:
    assert streamlit_render is app_shell.render


def test_ui_panel_modules_import_render_functions() -> None:
    assert callable(render_workspace_panel)
    assert callable(render_document_panel)
    assert callable(render_summary_panel)
    assert callable(render_index_controls)
    assert callable(render_retrieval_panel)
    assert callable(render_chat_panel)
    assert callable(render_evaluation_panel)


def test_ui_shared_embedding_device_settings_copy() -> None:
    settings = Settings(_env_file=None, embedding_device="auto")

    runtime_settings = settings_for_embedding_device(settings, "cpu")

    assert settings.embedding_device == "auto"
    assert runtime_settings.embedding_device == "cpu"


def test_ui_shared_document_helpers_filter_deleted_and_limit_selection() -> None:
    documents = [_document("doc-a", "A.md"), _document("doc-b", "B.md"), _document("doc-c", "C.md")]

    selected = normalize_selected_document_ids(
        ["doc-b", "missing", "doc-a", "doc-a", "doc-c", "extra"],
        documents,
        max_selected=2,
    )

    assert selected == ["doc-b", "doc-a"]
    assert selected_source_state_key("workspace-1") == "selected_source_document_ids_workspace-1"
    assert document_label(documents[0]) == "A.md (doc-a)"
    assert document_options(documents) == {
        "A.md (doc-a)": "doc-a",
        "B.md (doc-b)": "doc-b",
        "C.md (doc-c)": "doc-c",
    }


def _document(document_id: str, name: str) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        workspace_id="workspace-1",
        display_name=name,
        stored_filename=name,
        relative_path=f"workspaces/workspace-1/documents/{name}",
        source_type="markdown",
        content_hash=f"hash-{document_id}",
        size_bytes=10,
        page_count=None,
        chunk_count=1,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
