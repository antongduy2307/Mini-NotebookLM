# mini-notebooklm-rag

Local-first NotebookLM-like RAG application for portfolio and learning purposes.

Current status: Phase 03 grounded QA and chat. The repository contains packaging, typed configuration, local workspace persistence, PDF/Markdown ingestion, approximate chunking, local FAISS/BM25 retrieval, OpenAI-backed grounded QA, workspace chat sessions, baseline tests, and planning/review documents.

## Phase 03 Scope

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
- OpenAI Responses API wrapper for non-streaming answer generation.
- Grounded-only QA by default with `[S#]` inline source markers.
- Optional outside-knowledge mode with separated document and outside-document sections.
- Query rewriting using only the current chat session history.
- SQLite chat sessions and chat messages with compact source/retrieval/prompt metadata.
- Streamlit chat panel with temporary API key input, chat history, source list, and dev/debug details.
- Tests for storage, parsers, chunking, repositories, and ingestion cleanup.
- Tests for embedding device behavior, fake embeddings, FAISS, BM25, fusion, citations, and retrieval service behavior.
- Tests for OpenAI wrapper mocking, prompt construction, source mapping, grounded shortcuts, query rewriting, and chat persistence.

Not included:

- API key persistence.
- Streaming responses.
- Saved local API key manager or keyring integration.
- Summaries, evaluation UI, or MLflow.
- LangChain or LlamaIndex.

## Setup

```bash
uv sync
```

## Run

```bash
uv run app
```

The app command starts the Streamlit workspace/document ingestion, retrieval debug UI, and chat UI. It may create `storage/app.db`, workspace/document runtime files, FAISS index files when the user clicks the rebuild control, and chat records in SQLite. It must not create summary artifacts, eval artifacts, log artifacts, or secret files.

Building a workspace index creates runtime files under:

```text
storage/workspaces/<workspace_id>/indexes/faiss.index
storage/workspaces/<workspace_id>/indexes/faiss_meta.json
```

These files are local runtime artifacts and remain ignored by Git.

The first real embedding model use may download the configured local model through `sentence-transformers`. Tests use fake embeddings and do not download a model.

Phase 03 chat uses the OpenAI Responses API only when an API key is available from `.env` or temporary Streamlit UI input. Temporary UI keys live only in `st.session_state` and are not written to SQLite or local files.

## Validate

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Configuration

Copy `.env.example` to `.env` for local overrides when needed. Do not commit `.env`, `.env.local`, `.local/`, or runtime storage files.

API keys may be supplied through `.env` or the temporary Streamlit password input. Phase 03 does not implement saved local API keys. Future saved keys must remain outside SQLite, include an owner name, and be stored only in a Git-ignored local secrets file after a later approved phase.

### Embedding Device

`EMBEDDING_DEVICE` controls only the local `sentence-transformers` embedding model:

- `EMBEDDING_DEVICE=auto` prefers CUDA when PyTorch reports CUDA as available, otherwise CPU.
- `EMBEDDING_DEVICE=cuda` requires CUDA-enabled PyTorch and a compatible NVIDIA driver.
- `EMBEDDING_DEVICE=cpu` forces CPU embeddings.

FAISS remains CPU-only through `faiss-cpu`. This project does not require the full CUDA Toolkit.
The lockfile is configured for the PyTorch CUDA 12.6 wheel source on supported environments.
Check the active environment with:

```bash
uv run python scripts/check_cuda.py
```

If CUDA is unavailable, first verify the NVIDIA driver with `nvidia-smi`, then rerun `uv sync`.
Do not install NVIDIA system drivers or CUDA Toolkit as part of this project setup.

## Planning Documents

- `docs/PROJECT_PLAN.md`
- `docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md`
- `docs/phases/PHASE_01_INGESTION_PLAN.md`
- `docs/output_prompt/PHASE_02_RETRIEVAL_PLAN_REVIEW.md`
- `docs/output_prompt/PHASE_02_IMPLEMENTATION_REVIEW_DIGEST.md`
- `docs/output_prompt/PHASE_03_QA_CHAT_PLAN_REVIEW.md`
