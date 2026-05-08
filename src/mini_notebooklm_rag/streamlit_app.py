"""Compatibility entrypoint for the Streamlit UI."""

from __future__ import annotations

from mini_notebooklm_rag.ui.app_shell import render
from mini_notebooklm_rag.ui.shared import _embedding_device_index, settings_for_embedding_device

__all__ = ["_embedding_device_index", "render", "settings_for_embedding_device"]


if __name__ == "__main__":
    render()
