# Phase 01 Ingestion and Workspace Foundation Plan

Status: planning draft for user and reviewer approval.

This phase plan defines the next implementation phase after the completed Phase 00 scaffold. Do not implement this phase until the user explicitly approves this plan.

## 1. Phase Objective

Implement the persistent local foundation for workspaces and document ingestion:

- Workspace create, select, list, and delete.
- SQLite initialization and repository layer.
- Local workspace/document storage layout.
- PDF text extraction with page metadata.
- Markdown parsing with heading metadata.
- File hashing and duplicate upload detection.
- Configurable approximate chunking.
- Document deletion cleanup.
- Streamlit UI updates for workspace and document management.
- Tests for storage, parsing, chunking, duplicate detection, and cleanup.

Phase 01 must preserve Phase 00 scaffold behavior and must not implement retrieval, generation, chat, summaries, evaluation, or API key persistence.

## 2. Scope and Non-Scope

### In Scope

- Add Phase 01 ingestion/storage dependencies only.
- Initialize `storage/app.db` when the app needs persistence.
- Create workspace directories under `storage/workspaces/<workspace_id>/`.
- Reserve future artifact directories under each workspace:
  - `documents/`
  - `indexes/`
  - `summaries/`
  - `eval/`
  - `logs/`
- Copy original uploaded PDF/Markdown files into the selected workspace.
- Persist workspace, document, and chunk metadata in SQLite.
- Delete workspace metadata and workspace directory.
- Delete document metadata, chunks, original file, and any future related artifact directories/files if present.
- Show workspace and document operations in Streamlit.
- Show ingestion status and chunk counts.
- Show a clear message that retrieval/chat are deferred to later phases.

### Non-Scope

Phase 01 must not implement:

- FAISS index files or FAISS indexing.
- BM25 retrieval.
- Embeddings.
- OpenAI calls.
- Query answering.
- Query rewriting.
- Chat sessions beyond passive placeholders in UI text.
- Summaries or summary cache files.
- Evaluation UI or eval artifacts.
- MLflow.
- API key persistence.
- OCR or scanned PDF support.
- LangChain or LlamaIndex.

## 3. Dependencies to Add and Why

Proposed runtime dependencies for Phase 01:

- `pymupdf`: required now for normal text PDF extraction with page metadata. It also supports generating small PDF fixtures in tests.
- `markdown-it-py`: required now for Markdown token parsing and heading-aware block extraction.

No additional database dependency is proposed. Use standard library `sqlite3`.

No tokenizer dependency is proposed for Phase 01. Chunking should use approximate token counting from standard library text processing.

Explicitly not added in Phase 01:

- `openai`
- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`
- `mlflow`
- `tiktoken`
- LangChain
- LlamaIndex

Decision: use `sqlite3` directly.

Reason: the schema is small, the app is local-first, and direct SQL keeps the MVP transparent.

Status: provisional, requires user approval.

Decision: avoid `tiktoken` in Phase 01 and use approximate token counts.

Reason: token-aware chunking can be refined later; Phase 01 only needs metadata-rich chunks for future retrieval.

Status: provisional, requires user approval.

## 4. Proposed File and Module Changes

Only after approval, Phase 01 should add or update these files:

```text
pyproject.toml
.env.example
README.md
src/mini_notebooklm_rag/config.py
src/mini_notebooklm_rag/streamlit_app.py
src/mini_notebooklm_rag/storage/__init__.py
src/mini_notebooklm_rag/storage/paths.py
src/mini_notebooklm_rag/storage/sqlite.py
src/mini_notebooklm_rag/storage/repositories.py
src/mini_notebooklm_rag/ingestion/__init__.py
src/mini_notebooklm_rag/ingestion/models.py
src/mini_notebooklm_rag/ingestion/pdf_parser.py
src/mini_notebooklm_rag/ingestion/markdown_parser.py
src/mini_notebooklm_rag/ingestion/chunker.py
src/mini_notebooklm_rag/ingestion/service.py
src/mini_notebooklm_rag/utils/__init__.py
src/mini_notebooklm_rag/utils/hashing.py
src/mini_notebooklm_rag/utils/filenames.py
tests/test_storage_sqlite.py
tests/test_workspace_repository.py
tests/test_document_repository.py
tests/test_pdf_parser.py
tests/test_markdown_parser.py
tests/test_chunker.py
tests/test_ingestion_service.py
tests/test_storage_paths.py
```

Notes:

- Existing Phase 00 tests should remain and continue to pass.
- `streamlit_app.py` can be updated directly for Phase 01, or split into small UI helper modules if it becomes hard to read. If split, keep the first split limited to workspace/document panels only.
- Do not create `retrieval/`, `llm/`, or `evaluation/` implementation modules in Phase 01.

## 5. Proposed SQLite Schema

Phase 01 should use simple idempotent initialization with `CREATE TABLE IF NOT EXISTS`. Migration files are not needed yet.

SQLite connection requirements:

- Use `sqlite3`.
- Enable foreign keys on every connection with `PRAGMA foreign_keys = ON`.
- Store timestamps as UTC ISO-8601 strings.
- Use explicit transactions for create/delete workflows.

### `workspaces`

```sql
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_workspaces_created_at
ON workspaces(created_at);
```

Behavior:

- `id` is a generated UUID hex string and is safe for filesystem paths.
- `name` is the display name.
- `name_normalized` prevents confusing duplicate names in the selector.
- Workspace delete removes documents and chunks through cascades.

Decision: make workspace names unique case-insensitively through `name_normalized`.

Reason: duplicate display names make Streamlit selection ambiguous.

Status: provisional, requires user approval.

### `documents`

```sql
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'markdown')),
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    UNIQUE (workspace_id, content_hash),
    UNIQUE (workspace_id, stored_filename)
);
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_documents_workspace_id
ON documents(workspace_id);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_hash
ON documents(workspace_id, content_hash);
```

Behavior:

- `display_name` is the original filename shown in UI.
- `stored_filename` should be `<document_id>__<sanitized_original_name>`.
- `relative_path` is relative to the configured storage root, not an absolute path.
- Duplicate detection is per workspace using `(workspace_id, content_hash)`.

### `chunks`

```sql
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'markdown')),
    filename TEXT NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    heading_path TEXT,
    approximate_token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id
ON chunks(workspace_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_workspace_document
ON chunks(workspace_id, document_id);
```

Behavior:

- `heading_path` is stored as JSON text for Markdown chunks and `NULL` for PDF chunks.
- `page_start` and `page_end` are user-facing 1-indexed page numbers for PDF chunks and `NULL` for Markdown chunks.
- `content_hash` hashes normalized chunk text plus relevant source metadata.

## 6. Proposed Storage Layout

Approved storage direction:

```text
storage/
  app.db
  workspaces/
    <workspace_id>/
      documents/
      indexes/
      summaries/
      eval/
      logs/
```

Phase 01 may create:

- `storage/app.db`
- `storage/workspaces/<workspace_id>/`
- `storage/workspaces/<workspace_id>/documents/`
- reserved empty directories:
  - `indexes/`
  - `summaries/`
  - `eval/`
  - `logs/`
- stored original uploaded files under `documents/`

Phase 01 must not create:

- FAISS index files.
- summary cache files.
- eval artifacts.
- `.local/secrets.local.json`.

Stored document path:

```text
storage/workspaces/<workspace_id>/documents/<document_id>__<sanitized_original_filename>
```

Decision: create reserved future directories when a workspace is created, but no future artifact files.

Reason: it keeps the storage layout stable for later phases without pretending FAISS/summaries/eval exist.

Status: provisional, requires user approval.

## 7. Workspace Lifecycle Design

Create workspace:

1. Validate name is non-empty after trimming.
2. Generate UUID hex workspace ID.
3. Normalize name for uniqueness.
4. Insert into SQLite.
5. Create workspace directory and reserved subdirectories.
6. If directory creation fails, rollback DB transaction or delete partial metadata.

Select workspace:

- UI loads workspace list from SQLite ordered by name or creation time.
- Selected workspace ID is stored in Streamlit session state.

Delete workspace:

1. Resolve workspace directory under the configured storage root.
2. Start SQLite transaction.
3. Delete workspace row; cascades delete documents and chunks.
4. Delete workspace directory only after safety checks prove it is inside storage root.
5. If filesystem delete fails after DB delete, surface a recovery warning with path and error.

No rename in MVP.

## 8. Document Lifecycle Design

Upload document:

1. Require selected workspace.
2. Accept only `.pdf`, `.md`, and `.markdown`.
3. Read uploaded bytes once.
4. Compute SHA-256 content hash.
5. Check `(workspace_id, content_hash)` duplicate.
6. If duplicate exists, skip copy/parsing/chunking and show a clear message.
7. Generate document ID.
8. Sanitize filename and copy original bytes to workspace `documents/`.
9. Parse document into structured blocks.
10. Chunk parsed blocks.
11. Insert document metadata and chunks in one SQLite transaction.
12. Update `chunk_count`.
13. On failure, remove copied file and rollback metadata.

List documents:

- Query documents by selected workspace.
- Show display name, source type, size, chunk count, created time, and duplicate status messages if applicable.

Delete document:

1. Load document metadata.
2. Start SQLite transaction and delete document row; chunks cascade.
3. Delete stored original file after path safety checks.
4. Delete any known future-related artifact files/directories for the document if present.
5. Do not touch FAISS files in Phase 01. Leave a documented TODO for Phase 02 to rebuild FAISS after document deletion.

Future-related artifact cleanup placeholder:

- Plan a helper that can remove document-scoped artifacts under `summaries/`, `eval/`, or future index metadata if they exist.
- In Phase 01, this helper should not create those artifacts.

## 9. PDF Parser Design

Dependency: `pymupdf`.

Parser input:

- Stored PDF path.
- Original filename for metadata.

Parser output:

```text
ParsedDocument(
  source_type="pdf",
  filename=<original filename>,
  blocks=[
    SourceBlock(text=<page text>, page_start=1, page_end=1, heading_path=None),
    ...
  ],
  page_count=<page count>
)
```

Behavior:

- Open PDF with PyMuPDF.
- Extract text page by page using normal text extraction.
- Preserve user-facing 1-indexed page numbers.
- Skip empty pages but preserve total page count.
- Treat scanned/no-text pages as empty and show a warning.
- No OCR.
- No table/formula/image-special handling.
- Target normal text PDFs around 100 pages.

Errors:

- Unsupported encrypted or unreadable PDFs should fail gracefully with a UI message.
- Parser errors should not leave metadata or copied files behind.

## 10. Markdown Parser Design

Dependency: `markdown-it-py`.

Parser input:

- Stored Markdown path.
- Original filename for metadata.

Parser output:

```text
ParsedDocument(
  source_type="markdown",
  filename=<original filename>,
  blocks=[
    SourceBlock(text=<block text>, page_start=None, page_end=None, heading_path=["Heading"]),
    ...
  ],
  page_count=None
)
```

Behavior:

- Parse Markdown tokens.
- Maintain a heading stack for `h1` through `h6`.
- Assign each text block the nearest heading path.
- If text appears before the first heading, use `["document start"]`.
- Include paragraph, list item, block quote, and code block text when available.
- Preserve enough block boundaries for heading-aware chunking.

Storage:

- Store `heading_path` as JSON text in SQLite.

## 11. Chunking Design

Phase 01 should implement configurable approximate chunking using existing settings:

- `CHUNK_SIZE_TOKENS=700`
- `CHUNK_OVERLAP_TOKENS=120`

Decision: use dependency-free approximate token counting.

Reason: current preference is to avoid `tiktoken` in Phase 01. Approximate chunking is sufficient for ingestion tests and can be upgraded later without changing stored source metadata shape.

Status: provisional, requires user approval.

Approximation approach:

- Split text into word-like units with a standard library regex.
- Estimate each unit as `max(1, ceil(len(unit) / 4))` approximate tokens.
- Build chunks until the approximate target is reached.
- Carry an approximate overlap by walking backward through prior units until the overlap target is reached.

PDF metadata:

- Chunks may cross page boundaries.
- `page_start` is the first contributing page.
- `page_end` is the last contributing page.

Markdown metadata:

- Prefer not to merge across top-level heading changes unless a single section exceeds the target.
- Preserve `heading_path` from the primary contributing block.
- Use `["document start"]` before the first heading.

Chunk row metadata:

- `workspace_id`
- `document_id`
- `chunk_id`
- `source_type`
- `filename`
- `text`
- `page_start`
- `page_end`
- `heading_path`
- `approximate_token_count`
- `content_hash`

## 12. Streamlit UI Changes

Phase 01 should update the scaffold UI to include:

- Workspace selector.
- Create workspace form.
- Delete selected workspace action with confirmation.
- PDF/Markdown upload control.
- Document list for selected workspace.
- Delete document action with confirmation.
- Basic ingestion status messages:
  - uploaded and indexed into chunks
  - duplicate skipped
  - unsupported type
  - parser failure
- Basic chunk count display per document.
- Clear disabled/placeholder area stating retrieval and chat arrive in later phases.

UI constraints:

- Do not request API keys.
- Do not show API key fields.
- Do not call OpenAI.
- Do not show Q&A input except disabled explanatory placeholder if helpful.

## 13. Error Handling

Expected user-facing errors:

- No workspace selected.
- Workspace name is empty or duplicate.
- Unsupported file extension.
- Duplicate file hash in selected workspace.
- PDF has no extractable text.
- PDF cannot be opened.
- Markdown cannot be decoded as UTF-8.
- File storage write failed.
- SQLite transaction failed.
- Delete blocked by path safety validation.

Implementation guidance:

- Use typed result objects for ingestion success/duplicate/failure states.
- Keep low-level exception details in logs, not UI, unless the message is safe and useful.
- Do not include document text in logs by default.
- Use transactions so DB metadata and filesystem changes do not drift silently.

## 14. Security and Path Safety Considerations

Rules:

- Do not touch API key persistence.
- Do not create `.local/secrets.local.json`.
- Do not log secrets.
- Store uploaded files only under the configured storage root.
- Use `Path(uploaded_filename).name` to remove any client-supplied path.
- Sanitize stored filenames to a conservative ASCII-safe subset.
- Prefix stored filenames with document ID to avoid collisions.
- Generate workspace IDs and document IDs with UUID hex strings.
- Before delete operations, resolve paths and verify they are inside the configured storage root.
- Never delete a computed path if the safety check fails.
- Store relative paths in SQLite.

Windows considerations:

- Avoid reserved Windows filename characters in sanitized names.
- Avoid trailing spaces/dots in stored filenames.
- Keep paths reasonably short.
- Use `pathlib.Path` for path composition.

## 15. Test Plan

Tests should not require OpenAI, FAISS, embeddings, MLflow, or network.

Planned tests:

- SQLite initialization:
  - creates `workspaces`, `documents`, and `chunks`
  - enables foreign keys
  - is idempotent
- Workspace repository:
  - create/list/get/delete workspace
  - duplicate normalized name rejected
  - delete cascades documents/chunks
- Document repository:
  - insert/list/delete document metadata
  - duplicate `(workspace_id, content_hash)` rejected or reported
  - chunk insert/list/delete through cascade
- Storage paths:
  - creates workspace directories
  - stores relative paths
  - rejects path traversal
  - refuses deletes outside storage root
- Hashing:
  - SHA-256 is stable for identical bytes
- PDF parser:
  - uses a small generated PyMuPDF PDF fixture
  - extracts page text with 1-indexed page metadata
  - handles empty page warning/no-text behavior
- Markdown parser:
  - heading path assignment for nested headings
  - content before first heading maps to `document start`
  - lists/code/paragraph text is preserved enough for chunking
- Chunker:
  - respects approximate target and overlap
  - preserves PDF page ranges
  - preserves Markdown heading paths
  - computes content hashes
- Ingestion service:
  - successful PDF/Markdown ingest copies original file, inserts metadata, inserts chunks
  - duplicate upload skips parsing/chunking
  - parser failure rolls back DB and removes copied file
  - document delete removes file and metadata
- UI smoke tests:
  - keep limited unless Streamlit testing utilities are already available
  - no browser automation required in Phase 01 unless the implementation meaningfully changes UI flow

## 16. Validation Commands

Run after Phase 01 implementation:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
git status --short
git check-ignore -v storage/app.db storage/workspaces/example/documents/example.pdf .local/secrets.local.json
```

Expected:

- `uv sync` installs only approved Phase 01 dependencies.
- `pytest` passes without network or external services.
- `ruff` passes.
- `uv run app` starts and permits workspace/document UI smoke testing.
- `storage/app.db` and workspace runtime files are ignored.
- `.local/secrets.local.json` is ignored and not created by Phase 01.

Manual app smoke:

- Create workspace.
- Upload small Markdown file.
- Upload small PDF file.
- Try duplicate upload and confirm it is skipped.
- Delete document.
- Delete workspace.
- Confirm retrieval/chat remain unavailable.

## 17. Acceptance Criteria

Phase 01 is accepted when:

- Only approved Phase 01 dependencies are added.
- SQLite initializes `workspaces`, `documents`, and `chunks`.
- Workspace create/select/delete works and persists.
- Workspace directories are created/deleted safely.
- PDF upload stores the original file, extracts page text, creates chunks, and stores metadata.
- Markdown upload stores the original file, extracts heading metadata, creates chunks, and stores metadata.
- Duplicate upload by file hash in the same workspace is skipped with a clear message.
- Document deletion removes original file, metadata, and chunks.
- No FAISS, BM25, embedding, OpenAI, summary, eval, MLflow, or API key persistence code exists.
- Tests cover storage, repositories, parsers, chunking, duplicate detection, and deletion cleanup.
- `uv run pytest`, `uv run ruff check .`, and `uv run ruff format --check .` pass.
- UI clearly states retrieval/chat are deferred.

## 18. Risks and Known Limitations

- SQLite schema risk: early schema choices may need migration later; use simple idempotent initialization now and add migrations only when schema churn becomes real.
- Parser limitation: PyMuPDF extracts normal text PDFs only; scanned PDFs will produce empty/no useful text.
- Markdown limitation: Markdown token-to-text reconstruction can miss custom extensions; MVP should support common headings, paragraphs, lists, quotes, and code blocks.
- Chunking limitation: approximate token counts will not match model tokenizer counts; Phase 02 or Phase 03 can revisit tokenizer choice.
- Windows path risk: uploaded names may contain reserved characters; strict sanitization and path containment checks are required.
- Delete safety risk: workspace/document deletion touches the filesystem; implementation must verify resolved paths stay under the storage root before deleting.
- Transaction drift risk: filesystem and SQLite operations are not one atomic transaction; implementation should order operations and cleanup failures carefully.
- Large PDF risk: 100-page PDFs are target size, not a guaranteed hard limit unless the user approves enforcing one.

## 19. Questions for User Review

Blocking before implementation:

- None. The requested Phase 01 scope and dependency boundaries are clear.

Non-blocking:

1. Should workspace display names be unique case-insensitively as proposed, or should duplicate names be allowed?
2. Should the 100-page PDF target be a warning only, or a hard upload limit in Phase 01?
3. Should Markdown code blocks be chunked as normal searchable text, or skipped by default?
4. Should Phase 01 create reserved empty directories (`indexes/`, `summaries/`, `eval/`, `logs/`) at workspace creation, or create only `documents/` and defer the rest?

## 20. Implementation Sequence for the Next Run

Recommended implementation order after user approval:

1. Update dependencies in `pyproject.toml` with `pymupdf` and `markdown-it-py`; run `uv sync`.
2. Extend `Settings` only if needed for Phase 01 storage/chunk config already present.
3. Add storage path safety helpers.
4. Add SQLite initialization with `CREATE TABLE IF NOT EXISTS` and foreign keys.
5. Add repository functions for workspaces, documents, and chunks.
6. Add hashing and filename sanitization utilities.
7. Add parser data models.
8. Implement PDF parser.
9. Implement Markdown parser.
10. Implement approximate chunker.
11. Implement ingestion service orchestration and rollback cleanup.
12. Update Streamlit UI for workspace/document management.
13. Add focused tests for each storage/parser/chunking/ingestion slice.
14. Run validation commands and manual app smoke.
15. Report results and stop before Phase 02.

Waiting for user approval before implementation.
