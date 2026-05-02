"""Workspace and document ingestion services."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.ingestion.chunker import chunk_document
from mini_notebooklm_rag.ingestion.markdown_parser import parse_markdown
from mini_notebooklm_rag.ingestion.pdf_parser import parse_pdf
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.repositories import (
    DocumentRecord,
    DocumentRepository,
    NewChunkRecord,
    NewDocumentRecord,
    Workspace,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import initialize_database
from mini_notebooklm_rag.utils.filenames import sanitize_filename
from mini_notebooklm_rag.utils.hashing import sha256_bytes


class IngestionError(RuntimeError):
    """Raised when document ingestion fails."""


@dataclass(frozen=True)
class IngestionResult:
    status: str
    message: str
    document: DocumentRecord | None = None
    duplicate_document: DocumentRecord | None = None
    warnings: tuple[str, ...] = ()


class WorkspaceService:
    """Coordinate workspace metadata and filesystem directories."""

    def __init__(self, settings: Settings):
        self.paths = StoragePaths(Path(settings.app_storage_dir))
        self.paths.ensure_root()
        initialize_database(self.paths.db_path)
        self.repository = WorkspaceRepository(self.paths.db_path)

    def create_workspace(self, name: str) -> Workspace:
        workspace = self.repository.create(name)
        try:
            self.paths.create_workspace_dirs(workspace.id)
        except Exception:
            self.repository.delete(workspace.id)
            raise
        return workspace

    def list_workspaces(self) -> list[Workspace]:
        return self.repository.list()

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        return self.repository.get(workspace_id)

    def delete_workspace(self, workspace_id: str) -> None:
        self.repository.delete(workspace_id)
        self.paths.remove_tree_if_exists(self.paths.workspace_dir(workspace_id))


class IngestionService:
    """Ingest uploaded PDF and Markdown files into a workspace."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.paths = StoragePaths(Path(settings.app_storage_dir))
        self.paths.ensure_root()
        initialize_database(self.paths.db_path)
        self.workspaces = WorkspaceRepository(self.paths.db_path)
        self.documents = DocumentRepository(self.paths.db_path)

    def ingest_upload(
        self,
        uploaded_bytes: bytes,
        filename: str,
        workspace_id: str,
    ) -> IngestionResult:
        workspace = self.workspaces.get(workspace_id)
        if workspace is None:
            raise IngestionError("Select an existing workspace before uploading documents.")

        source_type = _source_type_for_filename(filename)
        content_hash = sha256_bytes(uploaded_bytes)
        duplicate = self.documents.find_by_hash(workspace_id, content_hash)
        if duplicate:
            return IngestionResult(
                status="duplicate",
                message=f"Skipped duplicate upload: {duplicate.display_name}",
                duplicate_document=duplicate,
            )

        document_id = uuid.uuid4().hex
        display_name = Path(filename).name
        safe_name = sanitize_filename(display_name)
        stored_filename = f"{document_id}__{safe_name}"
        stored_path = self.paths.stored_document_path(workspace_id, stored_filename)
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(uploaded_bytes)

        try:
            parsed = (
                parse_pdf(stored_path, display_name)
                if source_type == "pdf"
                else parse_markdown(stored_path, display_name)
            )
            chunks = chunk_document(
                parsed,
                chunk_size_tokens=self.settings.chunk_size_tokens,
                chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            )
            if not chunks:
                raise IngestionError(f"No ingestible text found in {display_name}.")

            document = NewDocumentRecord(
                id=document_id,
                workspace_id=workspace_id,
                display_name=display_name,
                stored_filename=stored_filename,
                relative_path=self.paths.relative_to_root(stored_path),
                source_type=source_type,
                content_hash=content_hash,
                size_bytes=len(uploaded_bytes),
                page_count=parsed.page_count,
            )
            chunk_records = [
                NewChunkRecord(
                    id=uuid.uuid4().hex,
                    workspace_id=workspace_id,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    source_type=chunk.source_type,
                    filename=chunk.filename,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading_path=chunk.heading_path,
                    approximate_token_count=chunk.approximate_token_count,
                    content_hash=chunk.content_hash,
                )
                for chunk in chunks
            ]
            inserted = self.documents.insert_with_chunks(document, chunk_records)
        except Exception:
            self.paths.remove_file_if_exists(stored_path)
            raise

        return IngestionResult(
            status="created",
            message=f"Indexed {display_name} into {inserted.chunk_count} chunks.",
            document=inserted,
            warnings=parsed.warnings,
        )

    def list_documents(self, workspace_id: str) -> list[DocumentRecord]:
        return self.documents.list_for_workspace(workspace_id)

    def delete_document(self, document_id: str) -> None:
        document = self.documents.get(document_id)
        if document is None:
            return

        stored_path = self.paths.resolve_relative(document.relative_path)
        self.documents.delete(document_id)
        self.paths.remove_file_if_exists(stored_path)
        # Phase 02 will rebuild FAISS indexes after document deletion.


def _source_type_for_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    raise IngestionError("Only PDF and Markdown files are supported in Phase 01.")
