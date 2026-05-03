from __future__ import annotations

import time

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.llm.models import LLMResponse, TokenUsage
from mini_notebooklm_rag.storage.repositories import (
    DocumentRepository,
    NewChunkRecord,
    NewDocumentRecord,
    WorkspaceRepository,
)
from mini_notebooklm_rag.storage.sqlite import initialize_database
from mini_notebooklm_rag.summary.grouping import build_summary_plan
from mini_notebooklm_rag.summary.models import (
    SUMMARY_MODE_OVERVIEW,
    SUMMARY_PROMPT_VERSION,
    NewCachedSummary,
    SummaryConfig,
)
from mini_notebooklm_rag.summary.prompts import (
    build_direct_overview_prompt,
    build_map_summary_prompt,
    build_reduce_summary_prompt,
)
from mini_notebooklm_rag.summary.repositories import SummaryRepository, new_summary_id
from mini_notebooklm_rag.summary.service import SummaryService


class FakeLLM:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [
            "Overview\nGenerated\n\nKey points\n- A\n\nUseful details\n- B\n\n"
            "Limitations or caveats in the document\n- C"
        ]
        self.calls: list[dict] = []

    def generate(self, instructions, input_text, model=None, max_output_tokens=None):
        self.calls.append(
            {
                "instructions": instructions,
                "input_text": input_text,
                "model": model,
                "max_output_tokens": max_output_tokens,
            }
        )
        return LLMResponse(
            text=self.responses.pop(0),
            model=model or "fake-model",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, app_storage_dir=str(tmp_path / "storage"))


def _repo_paths(tmp_path):
    db_path = tmp_path / "storage" / "app.db"
    initialize_database(db_path)
    return db_path


def _document_with_chunks(
    tmp_path,
    source_type: str = "markdown",
    texts: list[str] | None = None,
    headings: list[list[str] | None] | None = None,
):
    db_path = _repo_paths(tmp_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    document_id = "doc1"
    texts = texts or ["# Intro\n\nAlpha source text."]
    headings = headings or [["Intro"] for _ in texts]
    document = NewDocumentRecord(
        id=document_id,
        workspace_id=workspace.id,
        display_name="notes.md" if source_type == "markdown" else "paper.pdf",
        stored_filename="doc1__notes.md" if source_type == "markdown" else "doc1__paper.pdf",
        relative_path="workspaces/ws/documents/doc1__notes.md",
        source_type=source_type,
        content_hash="document-hash",
        size_bytes=sum(len(text) for text in texts),
        page_count=2 if source_type == "pdf" else None,
    )
    chunks = [
        NewChunkRecord(
            id=f"chunk{index}",
            workspace_id=workspace.id,
            document_id=document_id,
            chunk_index=index,
            source_type=source_type,
            filename=document.display_name,
            text=text,
            page_start=index + 1 if source_type == "pdf" else None,
            page_end=index + 1 if source_type == "pdf" else None,
            heading_path=headings[index] if source_type == "markdown" else None,
            approximate_token_count=max(1, len(text.split())),
            content_hash=f"chunk-hash-{index}",
        )
        for index, text in enumerate(texts)
    ]
    inserted = DocumentRepository(db_path).insert_with_chunks(document, chunks)
    return db_path, workspace, inserted, DocumentRepository(db_path).list_chunks(inserted.id)


def test_summary_repository_cache_key_and_upsert(tmp_path) -> None:
    db_path, workspace, document, _chunks = _document_with_chunks(tmp_path)
    repository = SummaryRepository(db_path)
    config = SummaryConfig()
    summary = NewCachedSummary(
        id=new_summary_id(),
        workspace_id=workspace.id,
        document_id=document.id,
        document_content_hash=document.content_hash,
        summary_mode=SUMMARY_MODE_OVERVIEW,
        model_name="fake-model",
        prompt_version=SUMMARY_PROMPT_VERSION,
        config_hash=config.hash(),
        config_json=config.to_json(),
        summary_text="first",
        source_chunk_count=1,
        source_character_count=10,
        is_partial=False,
        warnings=[],
        token_usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
    )

    first = repository.upsert(summary)
    second = repository.upsert(
        NewCachedSummary(
            **{
                **summary.__dict__,
                "id": new_summary_id(),
                "summary_text": "second",
                "token_usage": TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            }
        )
    )

    assert first.id == second.id
    assert second.summary_text == "second"
    assert second.total_tokens == 5


def test_summary_cache_hit_avoids_llm_call(tmp_path) -> None:
    settings = _settings(tmp_path)
    _db_path, _workspace, document, _chunks = _document_with_chunks(tmp_path)
    llm = FakeLLM()
    service = SummaryService(settings, llm_client=llm)

    first = service.generate_for_document(document.id, api_key="key", model_name="fake-model")
    second = service.generate_for_document(document.id, api_key="key", model_name="fake-model")

    assert first.status == "generated"
    assert second.status == "cached"
    assert second.summary is not None
    assert len(llm.calls) == 1


def test_summary_regenerate_bypasses_cache_and_updates_tokens(tmp_path) -> None:
    settings = _settings(tmp_path)
    _db_path, _workspace, document, _chunks = _document_with_chunks(tmp_path)
    llm = FakeLLM(["first summary", "second summary"])
    service = SummaryService(settings, llm_client=llm)

    first = service.generate_for_document(document.id, api_key="key", model_name="fake-model")
    time.sleep(1.1)
    second = service.generate_for_document(
        document.id,
        api_key="key",
        model_name="fake-model",
        regenerate=True,
    )

    assert first.summary is not None
    assert second.summary is not None
    assert second.summary.summary_text == "second summary"
    assert second.summary.updated_at > first.summary.updated_at
    assert len(llm.calls) == 2


def test_summary_service_skips_without_api_key_or_chunks(tmp_path) -> None:
    settings = _settings(tmp_path)
    db_path = _repo_paths(tmp_path)
    workspace = WorkspaceRepository(db_path).create("Research")
    document = DocumentRepository(db_path).insert_with_chunks(
        NewDocumentRecord(
            id="doc1",
            workspace_id=workspace.id,
            display_name="empty.md",
            stored_filename="doc1__empty.md",
            relative_path="workspaces/ws/documents/doc1__empty.md",
            source_type="markdown",
            content_hash="hash",
            size_bytes=0,
            page_count=None,
        ),
        [],
    )
    service = SummaryService(settings)

    no_chunks = service.generate_for_document(document.id, api_key="")
    _db_path, _workspace, document_with_chunks, _chunks = _document_with_chunks(tmp_path / "other")
    no_key = SummaryService(_settings(tmp_path / "other")).generate_for_document(
        document_with_chunks.id,
        api_key="",
    )

    assert no_chunks.status == "skipped"
    assert "no chunks" in no_chunks.message
    assert no_key.status == "skipped"
    assert "no OpenAI API key" in no_key.message


def test_grouping_direct_map_reduce_partial_markdown_and_pdf(tmp_path) -> None:
    _db_path, _workspace, markdown_doc, markdown_chunks = _document_with_chunks(
        tmp_path / "md",
        texts=["Alpha " * 20, "Beta " * 20, "Gamma " * 20],
        headings=[["Intro"], ["Intro"], ["Details"]],
    )
    direct = build_summary_plan(
        markdown_doc,
        markdown_chunks,
        SummaryConfig(direct_max_chars=10_000, map_group_max_chars=50),
    )
    mapped = build_summary_plan(
        markdown_doc,
        markdown_chunks,
        SummaryConfig(direct_max_chars=20, map_group_max_chars=90, max_groups=1),
    )

    _pdf_db, _pdf_workspace, pdf_doc, pdf_chunks = _document_with_chunks(
        tmp_path / "pdf",
        source_type="pdf",
        texts=["Page one " * 20, "Page two " * 20],
    )
    pdf_plan = build_summary_plan(
        pdf_doc,
        pdf_chunks,
        SummaryConfig(direct_max_chars=20, map_group_max_chars=90),
    )

    assert direct.use_map_reduce is False
    assert mapped.use_map_reduce is True
    assert mapped.is_partial is True
    assert mapped.groups[0].heading_path == ["Intro"]
    assert pdf_plan.groups[0].page_start == 1


def test_prompt_construction_includes_required_rules(tmp_path) -> None:
    _db_path, _workspace, document, chunks = _document_with_chunks(tmp_path)
    plan = build_summary_plan(document, chunks, SummaryConfig())

    direct = build_direct_overview_prompt(document, plan)
    mapped = build_map_summary_prompt(document, plan.groups[0], 1, 1, False)
    reduced = build_reduce_summary_prompt(document, ["partial"], plan, False)

    assert "Do not use outside knowledge" in direct.instructions
    assert "Overview" in direct.instructions
    assert "Source excerpts" in direct.input_text
    assert mapped.metadata["prompt_type"] == "map"
    assert reduced.metadata["prompt_type"] == "reduce"


def test_summary_rows_cascade_on_document_and_workspace_delete(tmp_path) -> None:
    settings = _settings(tmp_path)
    db_path, workspace, document, _chunks = _document_with_chunks(tmp_path)
    service = SummaryService(settings, llm_client=FakeLLM())
    generated = service.generate_for_document(document.id, api_key="key", model_name="fake-model")
    assert generated.summary is not None

    DocumentRepository(db_path).delete(document.id)
    assert (
        SummaryRepository(db_path).latest_for_document(document.id, SUMMARY_MODE_OVERVIEW) is None
    )

    db_path, workspace, document, _chunks = _document_with_chunks(tmp_path / "workspace")
    service = SummaryService(_settings(tmp_path / "workspace"), llm_client=FakeLLM())
    generated = service.generate_for_document(document.id, api_key="key", model_name="fake-model")
    assert generated.summary is not None

    WorkspaceRepository(db_path).delete(workspace.id)
    assert (
        SummaryRepository(db_path).latest_for_document(document.id, SUMMARY_MODE_OVERVIEW) is None
    )
