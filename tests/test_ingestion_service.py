from __future__ import annotations

import fitz
import pytest

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.ingestion.pdf_parser import PdfParserError
from mini_notebooklm_rag.ingestion.service import IngestionError, IngestionService, WorkspaceService


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        app_storage_dir=str(tmp_path / "storage"),
        chunk_size_tokens=20,
        chunk_overlap_tokens=2,
    )


def _pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_ingestion_service_ingests_markdown_and_skips_duplicate(tmp_path) -> None:
    settings = _settings(tmp_path)
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)
    workspace = workspace_service.create_workspace("Research")
    markdown = b"# Intro\n\nThis is a document."

    first = ingestion_service.ingest_upload(markdown, "notes.md", workspace.id)
    second = ingestion_service.ingest_upload(markdown, "notes.md", workspace.id)

    assert first.status == "created"
    assert first.document is not None
    assert first.document.chunk_count == 1
    assert second.status == "duplicate"
    assert second.duplicate_document == first.document
    assert (tmp_path / "storage" / first.document.relative_path).is_file()


def test_ingestion_service_ingests_pdf(tmp_path) -> None:
    settings = _settings(tmp_path)
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)
    workspace = workspace_service.create_workspace("Research")

    result = ingestion_service.ingest_upload(_pdf_bytes("PDF text"), "sample.pdf", workspace.id)

    assert result.status == "created"
    assert result.document is not None
    assert result.document.source_type == "pdf"
    assert result.document.page_count == 1


def test_ingestion_service_cleans_up_copied_file_on_parse_failure(tmp_path) -> None:
    settings = _settings(tmp_path)
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)
    workspace = workspace_service.create_workspace("Research")

    with pytest.raises(PdfParserError):
        ingestion_service.ingest_upload(b"not a pdf", "broken.pdf", workspace.id)

    documents_dir = tmp_path / "storage" / "workspaces" / workspace.id / "documents"
    assert list(documents_dir.iterdir()) == []
    assert ingestion_service.list_documents(workspace.id) == []


def test_ingestion_service_rejects_unsupported_extension(tmp_path) -> None:
    settings = _settings(tmp_path)
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)
    workspace = workspace_service.create_workspace("Research")

    with pytest.raises(IngestionError):
        ingestion_service.ingest_upload(b"hello", "notes.txt", workspace.id)


def test_ingestion_service_deletes_document_file_and_metadata(tmp_path) -> None:
    settings = _settings(tmp_path)
    workspace_service = WorkspaceService(settings)
    ingestion_service = IngestionService(settings)
    workspace = workspace_service.create_workspace("Research")
    result = ingestion_service.ingest_upload(b"# Intro\n\nDelete me.", "notes.md", workspace.id)
    assert result.document is not None
    stored_path = tmp_path / "storage" / result.document.relative_path

    ingestion_service.delete_document(result.document.id)

    assert not stored_path.exists()
    assert ingestion_service.list_documents(workspace.id) == []


def test_workspace_service_deletes_workspace_directory(tmp_path) -> None:
    settings = _settings(tmp_path)
    workspace_service = WorkspaceService(settings)
    workspace = workspace_service.create_workspace("Research")
    workspace_dir = tmp_path / "storage" / "workspaces" / workspace.id

    workspace_service.delete_workspace(workspace.id)

    assert not workspace_dir.exists()
    assert workspace_service.list_workspaces() == []
