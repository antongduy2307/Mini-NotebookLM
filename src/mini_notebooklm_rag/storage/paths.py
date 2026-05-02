"""Path helpers with storage-root containment checks."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a path escapes the configured storage root."""


@dataclass(frozen=True)
class StoragePaths:
    """Resolve and validate local runtime storage paths."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def db_path(self) -> Path:
        return self.root / "app.db"

    @property
    def workspaces_dir(self) -> Path:
        return self.root / "workspaces"

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.ensure_inside(self.workspaces_dir / workspace_id)

    def workspace_subdir(self, workspace_id: str, name: str) -> Path:
        return self.ensure_inside(self.workspace_dir(workspace_id) / name)

    def documents_dir(self, workspace_id: str) -> Path:
        return self.workspace_subdir(workspace_id, "documents")

    def indexes_dir(self, workspace_id: str) -> Path:
        return self.workspace_subdir(workspace_id, "indexes")

    def faiss_index_path(self, workspace_id: str) -> Path:
        return self.ensure_inside(self.indexes_dir(workspace_id) / "faiss.index")

    def faiss_metadata_path(self, workspace_id: str) -> Path:
        return self.ensure_inside(self.indexes_dir(workspace_id) / "faiss_meta.json")

    def stored_document_path(self, workspace_id: str, stored_filename: str) -> Path:
        return self.ensure_inside(self.documents_dir(workspace_id) / stored_filename)

    def create_workspace_dirs(self, workspace_id: str) -> None:
        for subdir in ("documents", "indexes", "summaries", "eval", "logs"):
            self.workspace_subdir(workspace_id, subdir).mkdir(parents=True, exist_ok=True)

    def relative_to_root(self, path: Path) -> str:
        return str(self.ensure_inside(path).relative_to(self.root)).replace("\\", "/")

    def resolve_relative(self, relative_path: str) -> Path:
        return self.ensure_inside(self.root / relative_path)

    def ensure_inside(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if resolved == self.root or self.root in resolved.parents:
            return resolved
        raise PathSafetyError(f"Path escapes storage root: {resolved}")

    def remove_file_if_exists(self, path: Path) -> None:
        safe_path = self.ensure_inside(path)
        if safe_path.exists() and safe_path.is_file():
            safe_path.unlink()

    def remove_tree_if_exists(self, path: Path) -> None:
        safe_path = self.ensure_inside(path)
        if safe_path.exists():
            shutil.rmtree(safe_path)
