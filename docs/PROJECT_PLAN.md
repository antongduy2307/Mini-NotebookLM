# mini-notebooklm-rag Project Plan

Status: planning draft for user and reviewer approval.

This document is an implementation-oriented plan for a local-first NotebookLM-like RAG application. It is not an implementation record. No application code should be created until the user approves the relevant phase plan.

## 1. Objective

Build `mini-notebooklm-rag`, a portfolio and learning project that demonstrates a practical local-first RAG system:

- Local Streamlit application launched with `uv run app`.
- Persistent local workspaces and document storage.
- SQLite metadata database.
- One FAISS dense vector index per workspace.
- BM25 sparse retrieval rebuilt from SQLite chunks when needed.
- Hybrid retrieval over selected documents.
- Local open-source embeddings.
- OpenAI API generation with configurable model selection.
- Grounded citations, source inspection, summaries, and retrieval evaluation.

The package name will be `mini_notebooklm_rag`.

## 2. MVP Product Scope

The MVP should let a user:

1. Create, select, and delete workspaces.
2. Upload PDF and Markdown documents into a workspace.
3. Store original uploaded files locally.
4. Detect duplicate uploads by file hash and skip duplicate indexing.
5. Extract text and metadata from supported documents.
6. Chunk documents for retrieval quality.
7. Build dense and sparse retrieval indexes.
8. Select up to 3 documents for a chat session.
9. Ask grounded questions over selected documents.
10. Receive answers with inline source references like `[S1]`.
11. Expand source chunks and inspect retrieval traces in a dev panel.
12. Generate and cache per-document summaries.
13. Create, import, export, and run workspace-specific retrieval evaluation cases.
14. Optionally log evaluation batches to MLflow when configured.

## 3. Explicit Non-Goals for MVP

The MVP will not include:

- OCR or scanned PDF support.
- DOCX, PPTX, image, spreadsheet, HTML, or URL ingestion.
- Multi-user auth.
- Cloud deployment.
- Advanced PDF rendering or visual highlight overlays.
- Cross-document summary synthesis.
- Answer-quality LLM judging.
- Workspace rename.
- Document rename.
- LangChain or LlamaIndex as the primary pipeline.

These can be considered after the local MVP is stable.

## 4. Supported Documents

Initial support:

- PDF: normal text PDFs only, English first, target size around 100 pages.
- Markdown: heading-aware parsing, English first.

PDF citations must support page and page ranges:

- `filename, p. 5`
- `filename, pp. 5-6`

Markdown citations use filename plus nearest heading:

- `filename > Heading`
- `filename > Parent > Child`
- `filename > document start` when the chunk appears before the first heading.

## 5. Main User Workflow

1. User starts the app:

   ```bash
   uv run app
   ```

2. User creates or selects a workspace.
3. User uploads PDF or Markdown files.
4. App stores the original files under the workspace.
5. App computes a file hash and skips duplicate indexing in the same workspace.
6. App extracts text and citation metadata.
7. App chunks extracted text using configurable chunking settings.
8. App stores metadata and chunks in SQLite.
9. App computes local embeddings and builds or updates the workspace FAISS index.
10. App rebuilds BM25 from SQLite chunks as needed.
11. Optional summary generation runs after upload/indexing when `AUTO_SUMMARY=true`.
12. User starts or resumes a chat session and selects up to 3 documents.
13. App optionally rewrites the query using only the current chat session history.
14. App retrieves chunks using hybrid FAISS + BM25.
15. App asks the configured OpenAI model to answer with grounded citations.
16. UI displays answer, compact source list, expandable source chunks, and dev traces.

## 6. Proposed Repository Structure

This structure is planned for future implementation after approval. This first run only creates planning documents.

```text
mini-notebooklm-rag/
  pyproject.toml
  README.md
  .gitignore
  .env.example
  docs/
    PROJECT_PLAN.md
    phases/
      PHASE_00_REPO_SCAFFOLD_PLAN.md
      PHASE_01_INGESTION_PLAN.md
      PHASE_02_RETRIEVAL_PLAN.md
      PHASE_03_QA_CHAT_PLAN.md
      PHASE_04_SUMMARY_PLAN.md
      PHASE_05_EVALUATION_MLFLOW_PLAN.md
  src/
    mini_notebooklm_rag/
      __init__.py
      app.py
      streamlit_app.py
      config.py
      logging_config.py
      ui/
      storage/
      ingestion/
      retrieval/
      llm/
      evaluation/
      utils/
  tests/
  storage/
    .gitkeep
```

Decision: use `src/` layout.

Reason: it prevents accidentally importing from the project root during tests and is a common packaging layout for Python applications.

Status: provisional, requires user approval in Phase 00.

## 7. Phase Breakdown

### Phase 00: Repository Scaffold

Create the basic package, dependency metadata, typed config examples, README, initial tests directory, Streamlit app shell, and ignored local storage placeholders.

Phase 00 must include a minimal scaffold-only Streamlit shell so `uv run app` works immediately. The shell must not implement ingestion, retrieval, QA, summary, evaluation, OpenAI calls, SQLite schema, FAISS, or BM25 behavior.

Phase 00 must separate the console launcher from the Streamlit page:

- `src/mini_notebooklm_rag/app.py`: console entrypoint for `uv run app`.
- `src/mini_notebooklm_rag/streamlit_app.py`: minimal Streamlit UI shell.

Phase 00 must also fix the `.gitignore` rules so `docs/` is not ignored and planning docs are trackable by Git.

Detailed plan: `docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md`.

No application features should be implemented before Phase 00 is approved.

### Phase 01: Workspace and Document Ingestion

Implement:

- Workspace create/select/delete.
- SQLite schema and repository layer.
- Local file storage layout.
- PDF parsing with page metadata.
- Markdown parsing with heading metadata.
- File hashing and duplicate detection.
- Configurable chunking.
- Document deletion cleanup.

Acceptance criteria:

- Upload PDF and Markdown files.
- Store original files.
- Persist workspace, document, and chunk metadata.
- Skip duplicate indexing by file hash.
- Delete document and related artifacts.
- Unit tests cover parsing, chunking, duplicate detection, and deletion cleanup.

### Phase 02: Embeddings and Hybrid Retrieval

Implement:

- Local embedding model loader through `sentence-transformers` or equivalent.
- Embedding device config: `auto`, `cuda`, `cpu`.
- One FAISS index per workspace.
- BM25 rebuilt from SQLite chunks.
- Hybrid retrieval with configurable `top_k`, dense weight, and sparse weight.
- Document filter for selected documents.
- Citation formatting utilities.

Acceptance criteria:

- Embedding model loads on CPU and prefers CUDA when available.
- FAISS index builds per workspace.
- BM25 rebuilds from SQLite chunks.
- Hybrid retrieval returns ranked chunks with trace metadata.
- Retrieval respects selected documents.
- Citation formatting works for PDFs and Markdown.

### Phase 03: OpenAI QA, Chat, and Query Rewrite

Implement:

- OpenAI generation client.
- Configurable generation model.
- Grounded-only answering by default.
- Optional outside-knowledge mode with explicit separation.
- Chat sessions and chat history per workspace.
- Query rewriting using only the current chat session.
- Ambiguity handling that asks a clarifying question when needed.
- Chat UI and dev panel.

Acceptance criteria:

- User can ask questions over selected documents.
- Answers include inline citations.
- Unsupported answers in grounded-only mode refuse with the required message.
- Outside knowledge mode separates document-grounded content from model knowledge.
- New chat and chat history work.
- Dev panel shows original query, rewritten query, selected documents, scores, and chunks.

### Phase 04: Summary Generation and Cache

Implement:

- Per-document summaries after indexing when `AUTO_SUMMARY=true`.
- Map-reduce summary for long documents.
- Heading-aware Markdown summary when possible.
- Summary cache keyed by document hash, model, and summary config.

Acceptance criteria:

- Summary generation can be enabled or disabled.
- Long documents use map-reduce.
- Markdown summaries preserve heading context when available.
- Cached summaries prevent repeated API calls for unchanged inputs/config.

### Phase 05: Evaluation and MLflow

Implement:

- Workspace-specific retrieval evaluation cases.
- UI create/edit for evaluation cases.
- JSON import/export.
- Metrics: hit@k by filename, page, and page range.
- Retrieval trace display for eval runs.
- Optional MLflow logging when `MLFLOW_TRACKING_URI` is configured.

Acceptance criteria:

- Eval cases can be created and stored per workspace.
- Eval JSON can be imported and exported.
- Eval batches compute retrieval metrics.
- MLflow logs metrics/config/artifacts only when configured.
- App still works without MLflow.

### Phase 06: Packaging, Docker, and Portfolio Polish

Implement after MVP:

- Dockerfile for CPU mode.
- Deployment notes.
- Optional local MLflow server documentation.
- Demo data and screenshots.
- Portfolio-focused README sections.

GPU Docker support remains optional/future.

## 8. Architecture Overview

Planned module boundaries:

- `ui`: Streamlit pages, sidebar, chat view, summary view, evaluation view.
- `config`: environment variables, settings defaults, and runtime config.
- `storage`: path handling, SQLite connection/migrations, repository methods, local secret file access.
- `ingestion`: PDF parsing, Markdown parsing, chunking, indexing orchestration.
- `retrieval`: embeddings, FAISS store, BM25 builder, hybrid rank fusion, citation formatting.
- `llm`: OpenAI client, prompts, answer generation, query rewrite, summarization.
- `evaluation`: eval cases, retrieval metrics, eval runner, MLflow logger.
- `utils`: hashing, text normalization, token counting helpers.

Decision: keep the core RAG pipeline self-built for MVP.

Reason: this project is for portfolio and learning; self-built components expose ingestion, chunking, retrieval, citation, and evaluation mechanics more clearly than a high-level framework.

Status: provisional, requires user approval.

## 9. Storage Design

Planned local storage layout:

```text
storage/
  app.db
  workspaces/
    <workspace_id>/
      documents/
        <document_id>__<original_filename>
      indexes/
        faiss.index
        faiss_meta.json
      summaries/
        <summary_cache_key>.json
      eval/
        exports/
      logs/
.local/
  secrets.local.json
```

Planned SQLite tables:

- `workspaces`
- `documents`
- `chunks`
- `chat_sessions`
- `chat_messages`
- `retrieval_events`
- `summary_cache`
- `eval_cases`
- `eval_runs`
- `settings`

Decision: store API keys outside SQLite in `.local/secrets.local.json`.

Reason: the requirement explicitly forbids storing API keys in SQLite. Keeping secrets in a separate Git-ignored file also makes backup/export behavior easier to reason about.

Status: final requirement from user, implementation details require Phase 00 approval.

## 10. Security Rules

Required rules:

- API keys may come from `.env`, temporary UI input, or a local saved secret file.
- Saved local API keys must include an owner name.
- UI displays only owner names, never raw or masked keys.
- Saved keys are plain text in a Git-ignored local secrets file for MVP.
- Never log or print API keys.
- Never store API keys in SQLite.
- Never commit `.env`, `.env.local`, `.local/`, or runtime storage data.
- `.env.example` must contain variable names only, not real secrets.

Decision: `.local/secrets.local.json` is the proposed MVP saved secret file.

Reason: it is explicit, easy to Git-ignore, easy to inspect locally, and avoids mixing secrets with application metadata.

Status: provisional location, security behavior is a final requirement.

## 11. Configuration Draft

Planned `.env.example` variables:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-nano
OPENAI_QUERY_REWRITE_MODEL=gpt-4.1-nano

EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=32

APP_STORAGE_DIR=storage
SQLITE_DB_PATH=storage/app.db
LOCAL_SECRETS_PATH=.local/secrets.local.json

AUTO_SUMMARY=false
ENABLE_QUERY_REWRITE=true
ALLOW_OUTSIDE_KNOWLEDGE=false

CHUNK_SIZE_TOKENS=700
CHUNK_OVERLAP_TOKENS=120
RETRIEVAL_TOP_K=6
DENSE_WEIGHT=0.65
SPARSE_WEIGHT=0.35

MLFLOW_TRACKING_URI=
```

Decision: default `OPENAI_MODEL` is `gpt-4.1-nano`, but model selection remains configurable.

Reason: the requested default is cost-conscious, while configurability avoids coupling the app to one model.

Status: provisional default, requires user approval.

Decision: default `AUTO_SUMMARY=false` in `.env.example`.

Reason: summary generation uses the OpenAI API and should be opt-in by default for cost safety during early development.

Status: final user decision.

Decision: use `pydantic-settings` from Phase 00 for typed settings and `.env` loading.

Reason: typed configuration reduces drift as the project grows and avoids replacing a temporary config approach later.

Status: final user decision.

## 12. Dependency Strategy

Dependency choices should be made phase by phase and documented before use.

Python version constraint:

```text
>=3.11,<3.13
```

Phase 00 dependencies:

- `streamlit`: minimal UI shell.
- `pydantic-settings`: typed settings and `.env` loading.
- `pytest`: scaffold tests.
- `ruff`: lint/format.

Deferred dependencies:

- Phase 01: `pymupdf` for PDF extraction and `markdown-it-py` for Markdown parsing.
- Phase 02: `sentence-transformers`, `faiss-cpu`, and `rank-bm25` for embeddings and retrieval.
- Phase 03: `openai` for generation, query rewrite, and summaries. It should not be included earlier unless a strong implementation reason is documented.
- Phase 05: `mlflow` for optional eval batch logging.
- Token-aware chunking phase: `tiktoken` or another tokenizer if needed.

Decision: defer `openai` until Phase 03.

Reason: Phase 00 is scaffold-only and does not call OpenAI; Phase 01 and Phase 02 can be implemented and tested without generation calls.

Status: final user decision unless a later phase documents a strong reason to change it.

Decision: keep `MLFLOW_TRACKING_URI=` in `.env.example`, but do not include `mlflow` as a Phase 00 dependency.

Reason: the app can reserve the configuration surface without installing MLflow before evaluation logging exists.

Status: final user decision.

Decision: do not introduce LangChain or LlamaIndex in MVP.

Reason: the project should demonstrate RAG internals directly; framework comparison can be a later optional phase.

Status: provisional, requires user approval.

## 13. Retrieval and QA Behavior

Retrieval:

- Dense retrieval uses FAISS.
- Sparse retrieval uses BM25.
- Hybrid fusion starts with configurable weighted score fusion.
- A future phase may compare reciprocal rank fusion or reranking.
- Retrieval is restricted to the selected documents for the current chat session.

QA:

- Default is grounded-only.
- If answer is not found and outside knowledge is disabled, the answer must say:

  ```text
  I could not find this information in the selected documents.
  ```

- If outside knowledge is enabled, output must separate:

  ```text
  From your documents:
  ...

  Outside the selected documents:
  ...
  ```

- Inline citations use `[S1]`, `[S2]`, etc.
- Source list appears below the answer.
- Source chunks are expandable.
- Dev panel preserves detailed source chunks and retrieval traces.

## 14. Chunking Plan

Default chunking target:

- 500-800 tokens per chunk.
- 100-150 token overlap.
- Proposed default: 700 tokens with 120 token overlap.

Chunking should prioritize retrieval quality. PDF chunks may cross page boundaries, but chunk metadata must preserve page start and page end for citation formatting.

Planned chunk metadata:

- `workspace_id`
- `document_id`
- `chunk_id`
- `source_type`
- `filename`
- `text`
- `page_start`
- `page_end`
- `heading_path`
- `token_count`
- `content_hash`

## 15. Evaluation Plan

Evaluation is workspace-specific.

Each eval case can include:

- question
- selected documents
- expected filename
- expected page or page range
- optional expected answer
- notes

MVP metrics:

- hit@k by filename
- hit@k by page
- hit@k by page range

Answer evaluation can be added later.

## 16. MLflow Plan

MLflow is optional and scoped to eval batch logging.

If `MLFLOW_TRACKING_URI` is configured, log:

- retrieval metrics
- retrieval config
- embedding model
- OpenAI model
- selected documents
- eval cases artifact
- eval results artifact

If MLflow is not configured, the app must still work normally.

## 17. Testing and Validation Strategy

Testing should be introduced as implementation begins:

- Unit tests for PDF parsing, Markdown parsing, chunking, citations, hybrid fusion, and deletion cleanup.
- Small synthetic fixtures to avoid large binary test files.
- Integration tests for workspace lifecycle and indexing workflow.
- Optional manual Streamlit smoke test after UI phases.

Expected commands by phase:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
```

## 18. Planning Assumptions

- The project is local-first and single-user for MVP.
- English-first document handling is acceptable for initial retrieval quality.
- Plain-text local secret storage is acceptable only for MVP and only if Git-ignored and clearly warned in UI.
- `faiss-cpu` is acceptable for MVP packaging, even when embeddings may run on CUDA through PyTorch.
- `BAAI/bge-base-en-v1.5` is a reasonable default embedding model candidate, subject to user approval.
- The OpenAI default model may be `gpt-4.1-nano`, subject to availability and user approval.
- The minimal Phase 00 Streamlit shell is allowed because it is scaffold-only and implements no RAG behavior.

## 19. Open Questions

Blocking before implementation:

- None for planning. Phase 00 still requires explicit user approval before implementation.

Non-blocking:

- Should the default embedding model be `BAAI/bge-base-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`, or another local model?
- Should `tiktoken` be used for token-aware chunking when chunking is implemented, or should the project start with a lighter tokenizer strategy?

## 20. Next Step

Review and approve or revise `docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md`.

Waiting for user approval before implementation.
