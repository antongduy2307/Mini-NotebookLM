# Phase 00 Repository Scaffold Plan

Status: planning draft for user and reviewer approval.

This phase plan defines the first implementation phase. It should not be executed until the user explicitly approves it.

## Phase Objective

Create a clean, minimal Python project scaffold for `mini-notebooklm-rag` using `uv`, a `src/` package layout, a scaffold-only Streamlit UI that starts with `uv run app`, typed configuration through `pydantic-settings`, explicit configuration examples, ignored local runtime storage, and baseline validation commands.

Phase 00 is not intended to implement RAG behavior. It creates the project foundation so later phases can add ingestion, retrieval, QA, summary, and evaluation features safely.

The minimal Streamlit shell is required in Phase 00, but it must remain scaffold-only. It must not parse documents, create a SQLite schema, call OpenAI, build indexes, retrieve chunks, summarize documents, or run evaluations.

## Scope

In scope for Phase 00 after approval:

- Packaging metadata.
- Dependency declaration.
- Minimal importable package.
- Minimal scaffold-only Streamlit startup shell.
- Console launcher for `uv run app`.
- Typed settings with `.env` loading through `pydantic-settings`.
- README setup instructions.
- `.gitignore`.
- `.env.example`.
- Empty test structure.
- Local storage placeholder.
- Baseline validation commands.

Out of scope for Phase 00:

- PDF parsing.
- Markdown parsing.
- SQLite schema and migrations.
- FAISS indexing.
- BM25 retrieval.
- OpenAI calls.
- API key save/load implementation.
- Chat sessions.
- Summaries.
- Evaluation UI.
- MLflow integration.

## Exact Files to Create in Phase 00

These files are proposed for Phase 00 implementation after approval:

```text
pyproject.toml
README.md
.gitignore
.env.example
docs/PROJECT_PLAN.md
docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md
src/mini_notebooklm_rag/__init__.py
src/mini_notebooklm_rag/app.py
src/mini_notebooklm_rag/streamlit_app.py
src/mini_notebooklm_rag/config.py
src/mini_notebooklm_rag/logging_config.py
tests/__init__.py
tests/test_scaffold.py
storage/.gitkeep
```

Notes:

- The two docs files already exist or are created by this planning run.
- `src/mini_notebooklm_rag/app.py` should be the console entrypoint for `uv run app`.
- `src/mini_notebooklm_rag/streamlit_app.py` should contain the minimal Streamlit UI shell.
- `tests/test_scaffold.py` should verify that the package imports and configuration defaults are available. It should not test unimplemented RAG behavior.
- `storage/.gitkeep` is only a placeholder; runtime files under `storage/` should be ignored.

## Proposed Minimal Dependencies

Phase 00 should keep dependencies small while still proving the project can start.

Runtime dependencies proposed for Phase 00:

- `streamlit`
- `pydantic-settings`

Development dependencies proposed for Phase 00:

- `pytest`
- `ruff`

Deferred dependencies:

- `pymupdf`: defer to Phase 01 when PDF ingestion is implemented.
- `markdown-it-py`: defer to Phase 01 when Markdown parsing is implemented.
- `sentence-transformers`: defer to Phase 02 when embeddings are implemented.
- `faiss-cpu`: defer to Phase 02 when dense indexing is implemented.
- `rank-bm25`: defer to Phase 02 when sparse retrieval is implemented.
- `openai`: defer to Phase 03 unless a strong reason is documented earlier.
- `mlflow`: defer to Phase 05.
- `tiktoken`: defer to the phase where token-aware chunking is implemented.

Design decision: defer heavy ingestion/retrieval dependencies until the phase that uses them.

Reason: this keeps Phase 00 fast to install and avoids dependency failures before the code needs those packages.

Status: provisional; requires user approval.

Design decision: use `pydantic-settings` in Phase 00 instead of `python-dotenv` as the main config solution.

Reason: typed settings are useful immediately for scaffold validation and reduce later config migration work.

Status: final user decision.

Design decision: defer `openai` to Phase 03.

Reason: Phase 00 does not call OpenAI, and earlier phases can establish local ingestion and retrieval without API dependency or key handling complexity.

Status: final user decision unless a later phase documents a strong reason to change it.

Alternative: include all known MVP dependencies in Phase 00.

Reason rejected: faster later implementation, but higher setup risk and less reviewable dependency surface.

## Proposed pyproject.toml Shape

Proposed project metadata:

```toml
[project]
name = "mini-notebooklm-rag"
version = "0.1.0"
description = "Local-first NotebookLM-like RAG application for learning and portfolio use."
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
  "streamlit",
  "pydantic-settings",
]

[project.scripts]
app = "mini_notebooklm_rag.app:main"

[dependency-groups]
dev = [
  "pytest",
  "ruff",
]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

The console script `app` should call `src/mini_notebooklm_rag/app.py`, which launches Streamlit for `src/mini_notebooklm_rag/streamlit_app.py`.

Design decision: use `uv run app` as the user-facing command.

Reason: this matches the project requirement and gives a stable command for README, tests, and reviewer verification.

Status: final requirement from user.

Open implementation detail: Streamlit normally runs via `streamlit run path/to/app.py`; the console script wrapper must be tested on Windows to ensure `uv run app` launches Streamlit correctly.

Design decision: use Python `>=3.11,<3.13`.

Reason: Python 3.11 gives broad package compatibility, while `<3.13` avoids early incompatibilities in ML and native packages.

Status: final user decision.

## Proposed Phase 00 Module Responsibilities

`src/mini_notebooklm_rag/__init__.py`

- Expose package version only.
- No side effects.

`src/mini_notebooklm_rag/app.py`

- Define `main()`.
- Locate `streamlit_app.py`.
- Launch Streamlit as the console entrypoint for `uv run app`.
- Avoid RAG logic.

`src/mini_notebooklm_rag/streamlit_app.py`

- Render the minimal scaffold-only Streamlit shell.
- Show project title and planning status.
- Avoid API calls and RAG feature implementation.

`src/mini_notebooklm_rag/config.py`

- Load `.env` using `pydantic-settings`.
- Define a typed settings object for scaffold-visible defaults.
- Include only settings needed for scaffold smoke tests.
- Do not read or print API keys in tests or startup logs.

`src/mini_notebooklm_rag/logging_config.py`

- Provide a minimal logging setup.
- Ensure future logs can avoid secrets.

Design decision: create module boundaries early but keep implementations minimal.

Reason: later phases can add behavior without reshuffling imports, while Phase 00 remains clearly non-functional for RAG.

Status: provisional; requires user approval.

## Storage Layout

Planned local runtime layout:

```text
storage/
  .gitkeep
  app.db
  workspaces/
    <workspace_id>/
      documents/
      indexes/
      summaries/
      eval/
      logs/
.local/
  secrets.local.json
```

Phase 00 should create only:

```text
storage/.gitkeep
```

Phase 00 should not create:

- `storage/app.db`
- workspace directories
- FAISS index files
- summary cache files
- `.local/secrets.local.json`

`.gitignore` should ignore runtime files:

```gitignore
.env
.env.local
.local/
storage/*
!storage/.gitkeep
```

`.gitignore` must not ignore:

```gitignore
docs/
docs/**
```

The current repository has an untracked `.gitignore` that ignores `docs/`. Phase 00 must correct this so planning documents are trackable by Git.

Design decision: use `storage/` for runtime data and `.local/` for local secrets.

Reason: runtime artifacts and secrets have different safety rules. Separating them makes Git ignore rules and future cleanup behavior more explicit.

Status: provisional; requires user approval.

## Security Rules

Phase 00 must establish these security rules in `.gitignore`, `.env.example`, and README:

- Do not commit real API keys.
- Do not commit `.env` or `.env.local`.
- Do not commit `.local/`.
- Do not commit runtime files under `storage/`.
- Do not store API keys in SQLite.
- Do not log or print API keys.
- Future saved API keys must include an owner name.
- Future UI must display only owner names, not raw or masked keys.
- Saved local API keys are plain text for MVP and must live only in a Git-ignored local secrets file.

Phase 00 should not implement secret persistence. It should only document and reserve the expected path:

```text
.local/secrets.local.json
```

Design decision: document secret handling before implementing it.

Reason: the project has explicit security constraints; documenting them in scaffold files prevents later accidental architecture drift.

Status: final behavior from user, path remains provisional.

## Proposed .env.example Defaults

Phase 00 should create `.env.example` with configuration names but no real secrets:

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

Design decision: `AUTO_SUMMARY=false` by default.

Reason: automatic summaries use paid API calls and should be opt-in for cost safety.

Status: final user decision.

Design decision: keep `MLFLOW_TRACKING_URI=` in `.env.example` without adding `mlflow` as a Phase 00 dependency.

Reason: the configuration key documents future support while keeping Phase 00 lightweight.

Status: final user decision.

## Acceptance Criteria

Phase 00 is accepted when all of the following are true:

- `pyproject.toml` exists and defines project metadata for `mini-notebooklm-rag`.
- Python package directory exists at `src/mini_notebooklm_rag/`.
- `src/mini_notebooklm_rag/app.py` is the console launcher.
- `src/mini_notebooklm_rag/streamlit_app.py` contains the minimal Streamlit UI shell.
- `uv run app` starts a minimal Streamlit app shell.
- `README.md` documents setup, run commands, and planning status.
- `.env.example` documents expected environment variables without real secrets.
- `.env.example` sets `AUTO_SUMMARY=false`.
- `.env.example` includes `MLFLOW_TRACKING_URI=`.
- `.gitignore` excludes `.env`, `.env.local`, `.local/`, and runtime storage files.
- `.gitignore` does not ignore `docs/` or the planning documents.
- `storage/.gitkeep` is tracked, but runtime storage contents are ignored.
- `tests/test_scaffold.py` exists and verifies the scaffold imports.
- `uv sync` succeeds.
- `uv run pytest` succeeds.
- `uv run ruff check .` succeeds.
- `uv run ruff format --check .` succeeds.
- `uv run app` is manually verified on Windows.
- No RAG application features are implemented in Phase 00.

## Test and Check Commands

Commands to run after Phase 00 implementation:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
git status --short
git check-ignore -v docs/PROJECT_PLAN.md docs/phases/PHASE_00_REPO_SCAFFOLD_PLAN.md
```

Expected results:

- `uv sync`: installs declared dependencies.
- `uv run pytest`: passes scaffold tests.
- `uv run ruff check .`: passes lint checks.
- `uv run ruff format --check .`: confirms formatting.
- `uv run app`: starts the Streamlit shell.
- `git status --short`: shows only intended scaffold files as changed/untracked.
- `git check-ignore -v ...`: returns no ignore rule for the planning docs.

Manual verification for `uv run app`:

- App starts locally on Windows from PowerShell.
- Page title renders.
- No API key is requested, printed, or logged.
- No SQLite database or FAISS index is created by just starting the scaffold app.

## Risks and Known Limitations

- Streamlit console script wrapper requires Windows-specific testing because `streamlit run` is normally a CLI command.
- Deferring heavy dependencies means later phases must update `pyproject.toml`; this is intentional but requires careful dependency review each phase.
- `pydantic-settings` is slightly heavier than plain environment reads, but it gives typed config from the start.
- Plain-text local secrets are acceptable only for MVP and require clear README/UI warnings before secret persistence is implemented.
- `gpt-4.1-nano` is a suggested default, but model availability and account access must be verified by the user before relying on it.
- The existing untracked `.gitignore` ignores `docs/`; Phase 00 must remove that ignore rule before Git tracking can verify the planning docs.
- No RAG functionality will exist after Phase 00; reviewers should judge it as scaffold only.

## Questions for User Review

Blocking before Phase 00 implementation:

None. The previously blocking Phase 00 questions have been resolved by user decision:

- Include a scaffold-only Streamlit shell.
- Use `pydantic-settings`.
- Defer heavy dependencies to later phases.
- Use Python `>=3.11,<3.13`.
- Set `AUTO_SUMMARY=false`.
- Keep `MLFLOW_TRACKING_URI=` without installing MLflow.
- Ensure `docs/` is not ignored.

Non-blocking before Phase 00 implementation:

1. Confirm whether `BAAI/bge-base-en-v1.5` should remain the planned default embedding model for Phase 02.
2. Confirm whether token-aware chunking should use `tiktoken` or another tokenizer when implemented.

## Approval Gate

Do not implement Phase 00 until the user explicitly approves this plan or provides requested revisions.

Waiting for user approval before implementation.
