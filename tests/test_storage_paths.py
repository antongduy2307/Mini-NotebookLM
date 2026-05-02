from __future__ import annotations

import pytest

from mini_notebooklm_rag.storage.paths import PathSafetyError, StoragePaths


def test_storage_paths_create_reserved_workspace_dirs(tmp_path) -> None:
    paths = StoragePaths(tmp_path / "storage")

    paths.create_workspace_dirs("abc123")

    for subdir in ("documents", "indexes", "summaries", "eval", "logs"):
        assert (tmp_path / "storage" / "workspaces" / "abc123" / subdir).is_dir()


def test_storage_paths_reject_path_escape(tmp_path) -> None:
    paths = StoragePaths(tmp_path / "storage")

    with pytest.raises(PathSafetyError):
        paths.ensure_inside(tmp_path / "outside.txt")


def test_storage_paths_relative_roundtrip(tmp_path) -> None:
    paths = StoragePaths(tmp_path / "storage")
    document_path = paths.stored_document_path("abc123", "doc.md")

    relative = paths.relative_to_root(document_path)

    assert relative == "workspaces/abc123/documents/doc.md"
    assert paths.resolve_relative(relative) == document_path
