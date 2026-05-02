# mini-notebooklm-rag

Local-first NotebookLM-like RAG application for portfolio and learning purposes.

Current status: Phase 01 ingestion foundation. The repository contains packaging, typed configuration, local workspace persistence, PDF/Markdown ingestion, approximate chunking, baseline tests, and planning documents. It does not implement retrieval or generation features yet.

## Phase 01 Scope

Included:

- `uv` project metadata.
- `src/` package layout.
- `uv run app` console command.
- Streamlit UI for workspace and document ingestion.
- Typed settings through `pydantic-settings`.
- `.env.example` with planned settings and no real secrets.
- SQLite metadata for workspaces, documents, and chunks.
- Local workspace directories and stored original documents.
- PDF text extraction with page metadata.
- Markdown parsing with heading metadata.
- Approximate dependency-free chunking.
- Duplicate upload detection by SHA-256 hash per workspace.
- Tests for storage, parsers, chunking, repositories, and ingestion cleanup.

Not included:

- FAISS or BM25.
- Embeddings.
- OpenAI API calls.
- API key persistence.
- Chat, summaries, evaluation UI, or MLflow.

## Setup

```bash
uv sync
```

## Run

```bash
uv run app
```

The app command starts the Streamlit workspace/document ingestion UI. It may create `storage/app.db` and workspace/document runtime files. It must not create FAISS indexes, summary artifacts, eval artifacts, log artifacts, or secret files.

## Validate

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Configuration

Copy `.env.example` to `.env` for local overrides when needed. Do not commit `.env`, `.env.local`, `.local/`, or runtime storage files.

API key handling is not implemented in Phase 01. Future saved keys must remain outside SQLite, include an owner name, and be stored only in a Git-ignored local secrets file.

## Planning Documents

- `docs/PROJECT_PLAN.md`
- `docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md`
