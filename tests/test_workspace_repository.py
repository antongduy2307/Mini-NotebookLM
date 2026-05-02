from __future__ import annotations

import pytest

from mini_notebooklm_rag.storage.repositories import (
    DuplicateWorkspaceError,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import initialize_database


def test_workspace_create_list_get_delete(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    repository = WorkspaceRepository(db_path)

    workspace = repository.create("Research")

    assert repository.get(workspace.id) == workspace
    assert repository.list() == [workspace]

    repository.delete(workspace.id)

    assert repository.get(workspace.id) is None
    assert repository.list() == []


def test_workspace_names_are_unique_case_insensitively(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    initialize_database(db_path)
    repository = WorkspaceRepository(db_path)

    repository.create("Research Notes")

    with pytest.raises(DuplicateWorkspaceError):
        repository.create(" research   notes ")
