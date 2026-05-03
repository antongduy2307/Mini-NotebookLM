# Phase 04 Summary Plan Review

Status: planning draft for user and external reviewer approval.

This document plans Phase 04 only. It must not be treated as implementation approval. Phase 04
should preserve Phase 00 scaffold behavior, Phase 01 ingestion/storage behavior, Phase 02 retrieval
behavior, Phase 03 chat/QA behavior, and the CUDA/PyTorch environment configuration.

## 1. Phase Objective

Implement per-document summary generation and summary caching for already-ingested documents.

Phase 04 should add:

- Per-document summaries generated manually from the UI or automatically after upload when
  `AUTO_SUMMARY=true`.
- Cost-safe default behavior with `AUTO_SUMMARY=false`.
- Summary generation over existing SQLite chunks, not original full documents.
- Map-reduce summarization for long documents.
- Heading-aware Markdown grouping when chunk metadata includes headings.
- PDF grouping based on chunk/page metadata.
- Summary cache keyed by document hash, summary mode, model, prompt version, and summary config.
- Streamlit summary UI.
- Mocked tests for prompts, cache, service behavior, grouping, and no-key/no-cache paths.

Phase 04 should make the app feel more like NotebookLM's document overview behavior while staying
strictly per-document in the MVP.

## 2. Scope and Non-Scope

### In Scope

- Add a `summary/` package.
- Add SQLite summary cache table(s).
- Add summary repository/cache access.
- Add summary prompt builders.
- Add summary grouping and map-reduce orchestration.
- Reuse the existing Phase 03 OpenAI client wrapper and LLM response models.
- Use `.env` or temporary Streamlit UI API key only.
- Add a Streamlit Summary panel integrated with existing workspace/document UI.
- Add optional UI-triggered auto-summary behavior after a successful upload when
  `AUTO_SUMMARY=true`.
- Add tests with fake/mocked LLM clients only.

### Non-Scope

Phase 04 must not implement:

- evaluation UI
- MLflow
- Docker/deployment
- LangChain
- LlamaIndex
- answer-quality judging
- RAG evaluation metrics
- cross-document synthesis as default MVP behavior
- saved local API key manager
- keyring/secret manager
- OCR/scanned PDF support
- retrieval architecture changes
- chat architecture redesign
- reranking or RRF
- tokenizer-dependent chunking changes

## 3. Dependencies and Why No New Dependency Is Preferred

No new runtime dependency is proposed.

Already available and sufficient:

- `openai`: existing Phase 03 SDK wrapper.
- SQLite via standard library.
- Existing `chunks` table and document metadata.
- Existing `TokenUsage` and `LLMResponse` models.
- Existing `pydantic-settings` config values.
- Existing Streamlit UI.

Dependencies explicitly not added:

- `tiktoken`
- `mlflow`
- LangChain
- LlamaIndex
- eval/judge frameworks
- secret/keyring packages
- document parsing packages beyond the existing Phase 01 parsers

Decision: use approximate character/chunk budgeting instead of `tiktoken`.

Reason: Phase 04 summaries can be cost-controlled with existing chunk boundaries, character caps,
and max group counts. Adding tokenizer-specific behavior would broaden scope and make model/provider
coupling stronger.

Status: proposed, requires user approval.

## 4. Proposed File and Module Changes

Planned files to add:

```text
src/mini_notebooklm_rag/summary/__init__.py
src/mini_notebooklm_rag/summary/models.py
src/mini_notebooklm_rag/summary/prompts.py
src/mini_notebooklm_rag/summary/grouping.py
src/mini_notebooklm_rag/summary/repositories.py
src/mini_notebooklm_rag/summary/service.py
tests/test_summary_cache.py
tests/test_summary_grouping.py
tests/test_summary_prompts.py
tests/test_summary_service.py
```

Planned files to modify:

```text
README.md
src/mini_notebooklm_rag/storage/sqlite.py
src/mini_notebooklm_rag/storage/paths.py
src/mini_notebooklm_rag/streamlit_app.py
tests/test_storage_sqlite.py
tests/test_ingestion_service.py
```

Optional files if Streamlit grows too large:

```text
src/mini_notebooklm_rag/ui/__init__.py
src/mini_notebooklm_rag/ui/summary_panel.py
```

Expected responsibilities:

- `summary/models.py`: dataclasses for summary mode, config, cache key, cached summary, generation
  result, and source grouping.
- `summary/prompts.py`: map, reduce, and direct overview prompt builders.
- `summary/grouping.py`: chunk grouping for direct and map-reduce summaries, including
  Markdown-heading-aware grouping.
- `summary/repositories.py`: SQLite summary cache CRUD.
- `summary/service.py`: orchestration for cache lookup, generation, regeneration, and skipped/error
  behavior.
- `streamlit_app.py`: render Summary panel and invoke `SummaryService`; preserve existing ingestion,
  retrieval, and chat panels.

No `evaluation/`, `mlflow/`, Docker, LangChain, or LlamaIndex modules should be added.

## 5. Summary Modes and MVP Behavior

Recommended MVP mode:

```text
overview
```

The `overview` mode should produce:

- a concise document overview
- key points
- important caveats or limitations found in the document
- for PDFs, page-range hints when useful
- for Markdown, section/heading hints when useful

Reason to start with one mode:

- Keeps prompts, cache keys, UI, and tests smaller.
- Avoids prematurely designing multiple summary templates.
- Matches the portfolio/demo objective without expanding into academic review or synthesis tooling.

Deferred modes:

- `detailed`
- `key_points_only`
- `academic_paper`
- `method_contribution_gaps`

Plan for extensibility:

- Store `summary_mode` in the cache table even if only `overview` is implemented.
- UI may show a mode selector with only `overview`, or omit the selector until more modes exist.

Status: proposed, requires user approval.

## 6. Map-Reduce Summary Design

Summary generation should choose one of two paths:

1. Direct summary for small documents.
2. Map-reduce summary for long documents.

Inputs:

- `DocumentRecord`
- ordered `ChunkRecord` rows for the document
- summary mode
- model name
- summary config

Suggested config defaults:

```python
SUMMARY_PROMPT_VERSION = "summary-v1"
SUMMARY_MODE = "overview"
SUMMARY_DIRECT_MAX_CHARS = 12000
SUMMARY_MAP_GROUP_MAX_CHARS = 8000
SUMMARY_REDUCE_MAX_PARTIAL_CHARS = 12000
SUMMARY_MAX_GROUPS = 8
SUMMARY_MAX_CHUNKS = 80
```

These should live in code as Phase 04 constants unless the user wants `.env` settings. Keeping them
as constants avoids adding more config surface in the MVP.

Direct path:

1. Read document chunks ordered by `chunk_index`.
2. Concatenate chunk text with source hints until `SUMMARY_DIRECT_MAX_CHARS`.
3. If no truncation is needed, call the direct overview prompt once.
4. Store the final summary and metadata.

Map path:

1. Read document chunks ordered by `chunk_index`.
2. Group chunks by heading/page boundaries when possible while respecting group character budget.
3. Cap group count at `SUMMARY_MAX_GROUPS`.
4. For each group, call a map summary prompt.
5. Combine partial summaries in a reduce prompt.
6. Store the final summary and metadata.

Truncation behavior:

- If the document has more than `SUMMARY_MAX_CHUNKS` or more groups than `SUMMARY_MAX_GROUPS`, mark
  the result as partial.
- Store/display warnings such as:
  `Summary used the first 80 chunks only; the document was truncated for cost control.`
- Prompts must instruct the model not to claim coverage of omitted content.

No tokenizer dependency should be added. Use approximate character counts and existing chunk
boundaries.

## 7. Markdown Heading-Aware Summary Design

Markdown chunks already carry `heading_path`.

Grouping strategy:

1. Prefer grouping contiguous chunks by the same top-level heading path.
2. Preserve nested heading hints in group labels.
3. If a chunk has no heading path, label it as `document start`.
4. Split heading groups further when they exceed `SUMMARY_MAP_GROUP_MAX_CHARS`.

Prompt source block shape:

```text
Section: Parent > Child
Chunks: 4-7
Content:
...
```

Fallback:

- If all heading metadata is missing, use simple chunk-order grouping.

Markdown summary should mention major sections when useful but should not fabricate a structure that
is not present in the source.

## 8. PDF Summary Design

PDF chunks already carry `page_start` and `page_end`.

Grouping strategy:

1. Keep chunks in document order.
2. Prefer grouping contiguous chunks while preserving page ranges.
3. Include page hints in prompt blocks:

```text
Pages: 5-7
Chunks: 10-13
Content:
...
```

Summary behavior:

- The summary does not need QA-style inline `[S#]` citations.
- It should include page hints when they are useful and supported by chunk metadata.
- It should not cite pages not included in the summarized content.

PDF limitation:

- Normal text PDFs only, inherited from Phase 01.
- No OCR or scanned PDF support in Phase 04.

## 9. Cache and Storage Design

Recommended MVP cache storage: SQLite-only.

Reason:

- Summary text is small enough for SQLite in the MVP.
- SQLite cascade behavior makes document/workspace deletion cleanup straightforward.
- It avoids coordinating JSON artifact writes with DB writes.
- It keeps Phase 04 easier to test with `tmp_path`.

Reserved filesystem summary directories should remain available for later phases:

```text
storage/workspaces/<workspace_id>/summaries/
```

Phase 04 should not need JSON summary artifacts unless the user explicitly prefers filesystem
inspectability over cascade simplicity.

Cache key should include:

- `document_id`
- `document_content_hash`
- `summary_mode`
- `model_name`
- `prompt_version`
- `config_hash`

Config hash should be computed from stable JSON including:

- direct max characters
- map group max characters
- reduce max partial characters
- max groups
- max chunks
- grouping strategy version

Cache hit behavior:

- Default generation checks cache first.
- If a matching cache row exists, return it without calling OpenAI.
- Regenerate bypasses cache and updates/replaces the cache row.

## 10. SQLite Schema Changes

Add one table through idempotent `CREATE TABLE IF NOT EXISTS`.

```sql
CREATE TABLE IF NOT EXISTS document_summaries (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_content_hash TEXT NOT NULL,
    summary_mode TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    source_chunk_count INTEGER NOT NULL,
    source_character_count INTEGER NOT NULL,
    is_partial INTEGER NOT NULL DEFAULT 0,
    warnings TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE (
        document_id,
        document_content_hash,
        summary_mode,
        model_name,
        prompt_version,
        config_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_document_summaries_workspace
ON document_summaries(workspace_id);

CREATE INDEX IF NOT EXISTS idx_document_summaries_document
ON document_summaries(document_id);

CREATE INDEX IF NOT EXISTS idx_document_summaries_cache_key
ON document_summaries(
    document_id,
    document_content_hash,
    summary_mode,
    model_name,
    prompt_version,
    config_hash
);
```

Column notes:

- `summary_text` is stored in SQLite because it is a user-visible artifact.
- `warnings` stores JSON text.
- `config_json` stores the exact summary config used.
- `is_partial` is `0`/`1` for SQLite portability.
- Token usage is stored when the LLM response provides it.

Deletion behavior:

- Document deletion cascades summary rows.
- Workspace deletion cascades summary rows through workspace/document cascade.

No migration framework should be added in Phase 04.

## 11. OpenAI and LLM Integration

Reuse the Phase 03 wrapper:

- `mini_notebooklm_rag.llm.openai_client.OpenAIClient`
- `mini_notebooklm_rag.llm.models.LLMResponse`
- `mini_notebooklm_rag.llm.models.TokenUsage`

Summary service should accept an injected LLM client for tests:

```python
class SummaryService:
    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClientProtocol | None = None,
    ): ...
```

Generation should use:

- `.env` `OPENAI_API_KEY`, or
- temporary Streamlit UI key passed into the service call.

Rules:

- Do not duplicate OpenAI SDK calls.
- Do not persist API keys.
- Do not log API keys.
- Do not call OpenAI in tests.
- Do not call OpenAI if the document has no chunks.
- Do not call OpenAI if a valid cache hit exists and `regenerate=False`.

## 12. Prompt Design

Prompt builders should be explicit and unit-tested.

### Direct Overview Prompt

Purpose: summarize a small/medium document in one call.

Required behavior:

- Summarize only the provided source text.
- Mention that the summary is partial if source content is marked partial.
- Do not use outside knowledge.
- Do not invent claims, sections, or page references.
- Produce concise output suitable for a document overview panel.

Suggested output structure:

```text
Overview:

Key points:
- ...

Useful details:
- ...

Limitations or caveats in the document:
- ...
```

### Map Summary Prompt

Purpose: summarize one group of chunks.

Required behavior:

- Summarize only the group content.
- Preserve important heading/page hints.
- Keep output short enough for reduce step.
- State if the group content appears incomplete/truncated.

### Reduce Summary Prompt

Purpose: combine partial summaries into one final document summary.

Required behavior:

- Combine only the partial summaries.
- Avoid adding outside knowledge.
- Avoid claiming full document coverage if warnings say content was truncated.
- Preserve the most important page/heading hints.

### Optional Key Points Prompt

Recommendation: defer as a separate mode. The MVP `overview` output can include key points without
adding another cache mode.

### Optional Academic Template

Recommendation: defer. A later phase can add an `academic_paper` mode if the user wants method,
contribution, limitations, and gaps.

Prompt metadata should include:

- prompt type
- prompt version
- source chunk count
- source character count
- group count
- summary mode
- is partial

Prompt metadata must not include API keys.

## 13. Streamlit UI Changes

Add a Summary panel below document management and before or near Chat.

Controls:

- Document selector for the selected workspace.
- Summary mode selector if implemented; MVP can show only `overview`.
- Generate summary button.
- Regenerate summary button.
- Optional checkbox/notice for large documents before generation.

Outputs:

- Summary status:
  - `no summary`
  - `cached`
  - `generated`
  - `skipped due missing API key`
  - `failed`
- Cached summary text if available.
- Model name.
- Summary mode.
- Prompt version.
- Partial/truncation warnings.
- Token usage if available.
- Last generated/updated timestamp.

API key handling:

- Reuse the existing temporary API key input pattern where practical.
- If the existing chat panel owns the temporary key widget, extract a helper so the same
  `st.session_state` key can be used consistently.
- Do not persist the temporary key.

UI constraints:

- Keep retrieval debug and chat UI working.
- Do not add evaluation UI.
- Do not add MLflow UI.
- Do not implement cross-document synthesis.

## 14. Auto-Summary Behavior After Upload/Indexing

Default:

```env
AUTO_SUMMARY=false
```

Recommended implementation:

- Keep `IngestionService` focused on ingestion and do not make it import/call OpenAI.
- In Streamlit, after a successful upload/indexing result:
  - if `settings.auto_summary` is false, do nothing.
  - if true and no API key is available, show:
    `Auto-summary skipped because no OpenAI API key is configured.`
  - if true and API key is available, call `SummaryService.generate_for_document(...)`.
  - if summary generation fails, show a warning but do not fail ingestion.

Reason:

- This preserves Phase 01 ingestion purity.
- It avoids OpenAI side effects in lower-level storage/ingestion tests.
- It keeps auto-summary a UI orchestration feature for the MVP.

Manual summary should remain available regardless of `AUTO_SUMMARY`.

## 15. Document and Workspace Deletion Cleanup Behavior

SQLite cache cleanup:

- `document_summaries.document_id REFERENCES documents(id) ON DELETE CASCADE`
- `document_summaries.workspace_id REFERENCES workspaces(id) ON DELETE CASCADE`

Document deletion:

- Existing `IngestionService.delete_document(...)` deletes document metadata and original file.
- SQLite cascade should delete summary rows automatically.
- If future filesystem summary artifacts exist, deletion should remove them explicitly.
- Phase 04 SQLite-only cache means no extra file deletion is needed.

Workspace deletion:

- Existing workspace deletion removes workspace metadata and directory.
- SQLite cascade removes summary rows.
- Workspace directory removal also removes reserved `summaries/` directory.

Tests should verify:

- document delete removes cached summary rows
- workspace delete removes cached summary rows

## 16. Token and Cost Handling

Cost-control rules:

- `AUTO_SUMMARY=false` remains default.
- Manual generation is the main MVP path.
- Cache hit avoids repeated OpenAI calls.
- Regenerate is explicit.
- Long documents use map-reduce with group and chunk caps.
- Very long documents produce partial warnings.
- No OpenAI call if API key is missing.
- No OpenAI calls in tests.

Token usage:

- Store input/output/total tokens when `LLMResponse.token_usage` includes them.
- Show token usage in Streamlit summary metadata.
- Do not estimate dollar cost in Phase 04.

Approximate budgeting:

- Use character counts and existing chunks.
- Use existing `approximate_token_count` only as metadata, not as a hard tokenizer guarantee.
- Do not add `tiktoken`.

## 17. Security Rules

Required:

- Never log API keys.
- Never print API keys.
- Never store API keys in SQLite.
- Never persist temporary UI API key.
- Do not create `.local/secrets.local.json`.
- Do not add keyring/secret manager dependency.
- Do not log full document text by default.
- Summary text may be stored because it is a user-visible artifact; document this in README.
- Prompt metadata and summary metadata must not include API keys.
- Runtime summary filesystem artifacts are not planned in Phase 04; if later added, they must stay
  under Git-ignored `storage/`.

Privacy note:

- Summary text can contain document-derived information. Users should treat `storage/app.db` as
  local private runtime data.

## 18. Test Plan

Tests must use mocked/fake LLM clients only.

Planned tests:

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_summary_cache.py` | Cache key and repository behavior | insert/get, cache miss, unique key, regenerate update, document/workspace cascade | migration framework |
| `tests/test_summary_grouping.py` | Grouping and map-reduce planning | direct vs map path, max chunks/groups, partial warnings, Markdown heading grouping, PDF page grouping | real token counting |
| `tests/test_summary_prompts.py` | Prompt construction | no outside-knowledge instruction, partial warning, headings/pages included, metadata excludes keys | real model quality |
| `tests/test_summary_service.py` | Summary orchestration | no key skip, cache hit avoids LLM, regenerate bypasses cache, direct generation, map-reduce calls, token usage stored | real OpenAI calls |
| `tests/test_ingestion_service.py` | deletion regression | document deletion removes summary rows through cascade | filesystem summary artifact cleanup |
| `tests/test_storage_sqlite.py` | schema initialization | `document_summaries` table/indexes exist, foreign keys remain enabled | migrations |
| existing QA/chat/retrieval tests | regression | Phase 02/03 behavior still passes | browser automation |

Mocking strategy:

- Fake LLM client returns deterministic `LLMResponse`.
- Tests assert call counts for cache hit/regenerate.
- Tests use `tmp_path` storage only.
- No test should require OpenAI API key, network, FAISS rebuild, or real embedding model download.

Manual smoke after implementation:

1. Start `uv run app`.
2. Create/select workspace.
3. Upload small Markdown/PDF.
4. Generate summary manually.
5. Confirm summary appears as generated.
6. Reload/select document and confirm cached summary appears without another call.
7. Regenerate and confirm updated timestamp/token metadata changes.
8. Set/remove API key and confirm missing-key skip behavior.
9. Delete document and confirm summary disappears.
10. Confirm chat and retrieval still work.

## 19. Validation Commands

Run after Phase 04 implementation:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
git status --short
git check-ignore -v storage/app.db storage/workspaces/example/summaries/example.json
```

Expected:

- `uv sync`: no new runtime dependency unless explicitly approved.
- `uv run pytest`: all tests pass with mocked LLM clients.
- `uv run ruff check .`: passes.
- `uv run ruff format --check .`: passes.
- `uv run app`: Streamlit starts for manual summary smoke.
- `git check-ignore`: runtime DB and potential summary artifact paths are ignored.

## 20. Acceptance Criteria

Phase 04 is accepted when:

- No new runtime dependency is added unless explicitly approved.
- No evaluation UI, MLflow, Docker, LangChain, LlamaIndex, or retrieval architecture changes are
  added.
- Summary generation is per-document only.
- `AUTO_SUMMARY=false` remains default.
- Manual summary generation works from Streamlit.
- Auto-summary, if enabled, runs only after successful UI upload and does not fail ingestion if
  skipped/failed.
- Missing API key produces a friendly skipped/actionable message and no OpenAI call.
- Summary cache prevents repeated OpenAI calls for unchanged document/config.
- Regenerate bypasses cache.
- Long documents use map-reduce with chunk/group caps and partial warnings.
- Markdown summaries use heading metadata when available.
- PDF summaries use page metadata when available.
- Summary cache rows are deleted on document/workspace deletion.
- Token usage is stored/displayed when available.
- Tests use mocked LLM clients and no real OpenAI calls.
- Existing ingestion, retrieval, and chat behavior remains intact.

## 21. Risks and Known Limitations

- Summary quality depends on prompt compliance; no judge/eval framework is added.
- Approximate character budgeting can under/over-shoot real model token limits.
- Long documents may be partial due cost caps.
- SQLite stores summary text; this is simple but means `storage/app.db` contains document-derived
  content.
- Auto-summary can add unexpected API cost if enabled; default remains false.
- Streamlit file may grow large; a small `ui/summary_panel.py` helper may be justified if the panel
  becomes hard to review.
- Cache invalidation depends on stable config hashing and document content hash.
- No cross-document synthesis means workspace-level overview remains out of scope.
- No OCR means scanned PDFs still cannot be summarized unless text was extracted in Phase 01.

## 22. Questions for User Review

Blocking before implementation:

- None. The Phase 04 scope is clear enough to implement after approval.

Non-blocking review decisions:

1. Approve SQLite-only summary cache for Phase 04, with no JSON summary artifacts?
2. Approve `overview` as the only MVP summary mode while storing `summary_mode` for future modes?
3. Approve keeping summary budget constants in code instead of adding more `.env` settings now?
4. Approve UI-orchestrated auto-summary after upload, keeping `IngestionService` free of OpenAI
   calls?
5. Approve partial/truncated summaries for very long documents instead of failing generation?

## 23. Implementation Sequence for the Next Run

Recommended order after user approval:

1. Add summary dataclasses and constants in `summary/models.py`.
2. Add `document_summaries` schema to `storage/sqlite.py` and update schema tests.
3. Add summary repository/cache CRUD and cascade tests.
4. Add summary grouping helpers and tests for Markdown/PDF/direct/map paths.
5. Add prompt builders and prompt tests.
6. Add `SummaryService` with fake LLM tests for cache hit, regenerate, missing key, direct summary,
   map-reduce, token usage, and partial warnings.
7. Update Streamlit with a Summary panel and shared temporary API-key behavior.
8. Add optional UI auto-summary orchestration after successful upload when `AUTO_SUMMARY=true`.
9. Update README with Phase 04 summary behavior and privacy/cost notes.
10. Run validation commands and a manual app smoke.
11. Create a Phase 04 implementation review digest.
12. Stop before Phase 05.

## 24. Reviewer Checklist

- [ ] Plan is per-document summary only.
- [ ] Plan does not add eval UI, MLflow, Docker, LangChain, LlamaIndex, reranking, or retrieval
      architecture changes.
- [ ] No new runtime dependency is proposed.
- [ ] `AUTO_SUMMARY=false` remains default.
- [ ] Auto-summary does not make ingestion fail when skipped/failed.
- [ ] OpenAI integration reuses Phase 03 wrapper.
- [ ] API keys are not persisted.
- [ ] Summary cache key includes document hash, mode, model, prompt version, and config hash.
- [ ] SQLite schema has cascade cleanup for document/workspace deletion.
- [ ] Long documents use map-reduce with cost caps and warnings.
- [ ] Markdown heading metadata and PDF page metadata are used where available.
- [ ] Tests use mocked LLM clients only.
- [ ] Summary text storage privacy is documented.
- [ ] Implementation remains gated on explicit user approval.
