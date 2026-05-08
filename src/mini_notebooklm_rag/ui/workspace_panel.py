"""Workspace Streamlit panel."""

from __future__ import annotations

import streamlit as st

from mini_notebooklm_rag.ingestion.service import WorkspaceService
from mini_notebooklm_rag.storage.repositories import DuplicateWorkspaceError, Workspace


def render_workspace_panel(
    workspace_service: WorkspaceService,
    workspaces: list[Workspace],
) -> Workspace | None:
    st.subheader("Sources")
    st.caption("Workspace")

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
        with st.expander("Workspace details"):
            st.write(f"ID: `{selected_workspace.id}`")
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
