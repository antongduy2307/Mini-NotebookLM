# mini-notebooklm-rag

Local-first NotebookLM-like RAG application for portfolio and learning purposes.

Current status: Phase 02 retrieval foundation. The repository contains packaging, typed configuration, local workspace persistence, PDF/Markdown ingestion, approximate chunking, local FAISS/BM25 retrieval, baseline tests, and planning documents. It does not implement answer generation or chat features yet.

## Phase 02 Scope

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
- Local embedding wrapper through `sentence-transformers`.
- CUDA-first `auto` embedding device selection with CPU fallback.
- One FAISS dense vector index per workspace.
- In-memory BM25 rebuilt from SQLite chunks.
- Weighted hybrid retrieval over up to 3 selected documents.
- PDF and Markdown citation formatting.
- Streamlit retrieval debug panel.
- Tests for storage, parsers, chunking, repositories, and ingestion cleanup.
- Tests for embedding device behavior, fake embeddings, FAISS, BM25, fusion, citations, and retrieval service behavior.

Not included:

- OpenAI API calls.
- API key persistence.
- Answer generation.
- Chat, summaries, evaluation UI, or MLflow.
- LangChain or LlamaIndex.

## Setup

```bash
uv sync
```

## Run

```bash
uv run app
```

The app command starts the Streamlit workspace/document ingestion and retrieval debug UI. It may create `storage/app.db`, workspace/document runtime files, and FAISS index files when the user clicks the rebuild control. It must not create summary artifacts, eval artifacts, log artifacts, or secret files.

Phase 02 adds a retrieval debug panel. Building a workspace index creates runtime files under:

```text
storage/workspaces/<workspace_id>/indexes/faiss.index
storage/workspaces/<workspace_id>/indexes/faiss_meta.json
```

These files are local runtime artifacts and remain ignored by Git.

The first real embedding model use may download the configured local model through `sentence-transformers`. Tests use fake embeddings and do not download a model.

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
- `docs/phases/PHASE_01_INGESTION_PLAN.md`
- `docs/output_prompt/PHASE_02_RETRIEVAL_PLAN_REVIEW.md`
