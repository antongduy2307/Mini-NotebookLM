"""Minimal Phase 00 Streamlit shell.

This module intentionally contains no RAG, storage, indexing, OpenAI, or secret
persistence behavior.
"""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.config import get_settings


def render() -> None:
    """Render the scaffold-only application shell."""
    settings = get_settings()

    st.set_page_config(page_title="mini-notebooklm-rag", page_icon="MNR", layout="wide")

    st.title("mini-notebooklm-rag")
    st.caption("Phase 00 scaffold only")

    st.info(
        "This shell verifies the project can start with `uv run app`. "
        "RAG features are intentionally not implemented in Phase 00."
    )

    st.subheader("Planned modules")
    st.table(
        [
            {"Module": "config", "Phase 00 status": "typed settings scaffold"},
            {"Module": "ui", "Phase 00 status": "minimal Streamlit shell"},
            {"Module": "storage", "Phase 00 status": "planned for later phases"},
            {"Module": "ingestion", "Phase 00 status": "deferred"},
            {"Module": "retrieval", "Phase 00 status": "deferred"},
            {"Module": "llm", "Phase 00 status": "deferred"},
            {"Module": "evaluation", "Phase 00 status": "deferred"},
        ]
    )

    st.subheader("Configured defaults")
    st.write(
        {
            "storage_dir": settings.app_storage_dir,
            "auto_summary": settings.auto_summary,
            "query_rewrite": settings.enable_query_rewrite,
            "outside_knowledge": settings.allow_outside_knowledge,
        }
    )

    st.warning(
        "No API keys are requested or displayed in this scaffold. "
        "OpenAI calls and local key persistence are deferred to later phases."
    )


render()
