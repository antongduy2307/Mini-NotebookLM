# mini-notebooklm-rag

Local-first NotebookLM-like RAG application for portfolio and learning purposes.

Current status: Phase 00 scaffold only. The repository contains packaging, typed configuration, a minimal Streamlit shell, baseline tests, and planning documents. It does not implement RAG features yet.

## Phase 00 Scope

Included:

- `uv` project metadata.
- `src/` package layout.
- `uv run app` console command.
- Minimal scaffold-only Streamlit UI.
- Typed settings through `pydantic-settings`.
- `.env.example` with planned settings and no real secrets.
- Baseline scaffold tests.
- Runtime storage placeholder.

Not included:

- PDF or Markdown parsing.
- SQLite schema.
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

The app command starts the scaffold-only Streamlit shell. It should not create a SQLite database, FAISS index, workspace files, or secret files.

## Validate

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Configuration

Copy `.env.example` to `.env` for local overrides when needed. Do not commit `.env`, `.env.local`, `.local/`, or runtime storage files.

API key handling is not implemented in Phase 00. Future saved keys must remain outside SQLite, include an owner name, and be stored only in a Git-ignored local secrets file.

## Planning Documents

- `docs/PROJECT_PLAN.md`
- `docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md`
