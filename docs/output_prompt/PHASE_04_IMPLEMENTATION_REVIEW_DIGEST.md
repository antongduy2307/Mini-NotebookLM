# Phase 04 Implementation Review Digest

## 1. Executive Summary

Phase 04 implemented per-document overview summaries with a SQLite-only cache. The app now supports manual summary generation from Streamlit and optional `AUTO_SUMMARY=true` orchestration after successful UI uploads. Summary generation reuses the Phase 03 OpenAI client wrapper through the existing non-streaming LLM protocol and does not add new runtime dependencies.

This phase did not implement cross-document synthesis, evaluation UI, MLflow, deployment, saved API keys, keyring support, LangChain, LlamaIndex, OCR, reranking, RRF, or retrieval/chat architecture redesign.

## 2. Files Changed

- `README.md`: updated current status, Phase 04 scope, run behavior, summary/API-key notes, and summary limitations.
- `src/mini_notebooklm_rag/storage/sqlite.py`: added the idempotent `document_summaries` schema and indexes.
- `src/mini_notebooklm_rag/streamlit_app.py`: added shared temporary API-key input, Summary panel, and optional auto-summary after UI uploads.
- `src/mini_notebooklm_rag/summary/__init__.py`: exports summary package constants, models, and service.
- `src/mini_notebooklm_rag/summary/models.py`: defines summary constants, cache key/config models, grouping models, persisted row models, and service result models.
- `src/mini_notebooklm_rag/summary/prompts.py`: builds direct, map, and reduce overview-summary prompts.
- `src/mini_notebooklm_rag/summary/grouping.py`: groups document chunks for direct or map-reduce summary generation.
- `src/mini_notebooklm_rag/summary/repositories.py`: provides SQLite cache read/upsert/delete helpers.
- `src/mini_notebooklm_rag/summary/service.py`: orchestrates cache lookup, skipped states, direct generation, map-reduce generation, and cache writes.
- `tests/test_storage_sqlite.py`: updated schema initialization assertion for `document_summaries`.
- `tests/test_summary_service.py`: added Phase 04 summary repository, grouping, prompt, service, cache, token, skip, and cascade tests.
- `docs/output_prompt/PHASE_04_IMPLEMENTATION_REVIEW_DIGEST.md`: this reviewer digest.

No dependency files were changed.

## 3. Module-by-Module Implementation Summary

### `summary/models.py`

Main items:

- `SUMMARY_MODE_OVERVIEW`
- `SUMMARY_PROMPT_VERSION`
- `SUMMARY_DIRECT_MAX_CHARS`
- `SUMMARY_MAP_GROUP_MAX_CHARS`
- `SUMMARY_REDUCE_MAX_PARTIAL_CHARS`
- `SUMMARY_MAX_GROUPS`
- `SUMMARY_MAX_CHUNKS`
- `SummaryConfig`
- `SummaryChunkGroup`
- `SummaryPlan`
- `SummaryCacheKey`
- `CachedSummary`
- `NewCachedSummary`
- `SummaryResult`

Responsibility: define typed summary configuration, cache identity, grouping plan data, persisted cache row data, and service result data.

Does not: call OpenAI, read files, write SQLite, or implement multiple summary modes.

### `summary/grouping.py`

Main functions:

- `build_summary_plan(document, chunks, config=None)`

Responsibility: choose direct vs map-reduce, cap chunks/groups for cost control, mark partial summaries, group Markdown chunks by contiguous heading path, and group PDFs by ordered page-aware chunk windows.

Does not: use tokenizers, call the LLM, persist summaries, or fabricate document structure.

### `summary/prompts.py`

Main functions/classes:

- `SummaryPrompt`
- `build_direct_overview_prompt(document, plan)`
- `build_map_summary_prompt(document, group, group_number, group_count, is_partial)`
- `build_reduce_summary_prompt(document, partial_summaries, plan, partials_truncated)`

Responsibility: construct prompts for overview summaries with explicit rules to summarize only provided source text, avoid outside knowledge, avoid invented claims/page references, and use the required section structure.

Does not: persist prompt text, include API keys, or implement chat/QA prompts.

### `summary/repositories.py`

Main functions/classes:

- `SummaryRepository.get_by_cache_key(key)`
- `SummaryRepository.latest_for_document(document_id, summary_mode)`
- `SummaryRepository.upsert(summary)`
- `SummaryRepository.delete_for_document(document_id)`
- `new_summary_id()`

Responsibility: store and retrieve summary cache rows from SQLite.

Does not: read source chunks, call OpenAI, or store full prompts/source chunk text.

### `summary/service.py`

Main functions/classes:

- `SummaryService`
- `SummaryService.get_cached_summary(...)`
- `SummaryService.latest_summary(...)`
- `SummaryService.generate_for_document(...)`
- `SummaryServiceError`
- `LLMClientProtocol`

Responsibility: coordinate document/chunk reads, cache keys, cache hits, missing-key/no-chunk skipped states, direct or map-reduce LLM calls, token usage aggregation, and cache writes.

Does not: persist API keys, run auto-summary inside ingestion, perform cross-document synthesis, or change retrieval/chat behavior.

### `streamlit_app.py`

Main additions:

- Shared temporary OpenAI API-key input near the top of the page.
- Summary panel with document selector, mode selector limited to `overview`, model input, cached summary display, Generate button, and Regenerate button.
- Auto-summary orchestration after successful UI upload when `AUTO_SUMMARY=true`.

Responsibility: expose manual and optional auto summary behavior while preserving ingestion, retrieval debug, and chat UI.

Does not: store temporary API keys, create summary JSON artifacts, call summary generation from `IngestionService`, or add evaluation UI.

## 4. Public/Internal API Summary

### `SummaryService(settings, llm_client=None)`

Inputs: typed app settings and optional fake/mock LLM client.

Output: service instance with initialized SQLite access.

Failure behavior: SQLite initialization or path creation errors surface normally.

Touches: filesystem only to ensure storage root exists; SQLite for schema initialization.

### `SummaryService.get_cached_summary(document_id, summary_mode="overview", model_name=None, config=None)`

Inputs: document ID, summary mode, model, optional summary config.

Output: `CachedSummary | None`.

Failure behavior: returns `None` if the document or cache row is missing.

Touches: SQLite only.

### `SummaryService.latest_summary(document_id, summary_mode="overview")`

Inputs: document ID and mode.

Output: latest `CachedSummary | None`.

Failure behavior: returns `None` if absent.

Touches: SQLite only.

### `SummaryService.generate_for_document(document_id, api_key, model_name=None, summary_mode="overview", regenerate=False, config=None)`

Inputs: document ID, API key string, model name, mode, regenerate flag, optional config.

Output: `SummaryResult` with status `cached`, `generated`, `skipped`, or `failed`.

Failure behavior:

- Unsupported mode returns `failed`.
- Missing document returns `failed`.
- No chunks returns `skipped`.
- Missing API key returns `skipped` when no fake LLM is injected.
- Existing cache returns `cached` without LLM call when `regenerate=False`.
- OpenAI wrapper errors return `failed`.

Touches: SQLite and OpenAI only when a real client is needed.

### `SummaryRepository.get_by_cache_key(key)`

Inputs: `SummaryCacheKey`.

Output: `CachedSummary | None`.

Failure behavior: SQLite errors surface normally.

Touches: SQLite only.

### `SummaryRepository.upsert(summary)`

Inputs: `NewCachedSummary`.

Output: persisted `CachedSummary`.

Failure behavior: SQLite integrity errors surface normally.

Touches: SQLite only.

### `build_summary_plan(document, chunks, config=None)`

Inputs: `DocumentRecord`, ordered chunks, optional config.

Output: `SummaryPlan`.

Failure behavior: does not raise for empty chunks; returns a plan with an empty group if called directly with empty input, though the service skips before calling it.

Touches: memory only.

## 5. Dependency Changes

No runtime or development dependencies were added in Phase 04.

Reused dependencies:

- `openai` via the existing Phase 03 OpenAI client wrapper.
- SQLite through the standard library.
- Existing ingestion chunks and metadata.

Deferred dependencies remain deferred:

- `tiktoken`
- `mlflow`
- LangChain
- LlamaIndex
- eval/judge frameworks
- secret manager/keyring dependencies

## 6. Final Summary Architecture

The summary flow is:

1. UI or service caller selects one document.
2. `SummaryService` loads document metadata and chunks from SQLite.
3. Service builds a cache key from document hash, summary mode, model, prompt version, and config hash.
4. If a valid cache row exists and `regenerate=False`, it returns the row without LLM calls.
5. If no API key exists, it returns a skipped result.
6. If chunks fit the direct budget, one prompt is sent.
7. If chunks exceed the direct budget, map prompts summarize bounded groups and a reduce prompt combines partial summaries.
8. The final summary and compact metadata are stored in SQLite.

`IngestionService` remains free of OpenAI calls. Auto-summary is only a Streamlit UI orchestration step after successful upload.

## 7. SQLite Summary Cache Schema

Final table:

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
    warnings TEXT NOT NULL,
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
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_document_summaries_workspace_document
ON document_summaries(workspace_id, document_id);

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

Cascade behavior:

- Workspace delete cascades summary rows through `workspace_id`.
- Document delete cascades summary rows through `document_id`.

## 8. Cache Key and Invalidation Behavior

Cache identity includes:

- `document_id`
- `document_content_hash`
- `summary_mode`
- `model_name`
- `prompt_version`
- `config_hash`

Invalidation behavior:

- Changed document content hash creates a cache miss.
- Changed model creates a cache miss.
- Changed prompt version creates a cache miss.
- Changed summary budget/config creates a cache miss.
- Regenerate bypasses cache and updates the matching row for the same key.
- Document/workspace deletion cascades cache rows.

## 9. Direct vs Map-Reduce Behavior

Direct path:

- Used when included chunk text fits `SUMMARY_DIRECT_MAX_CHARS`.
- Sends one prompt containing grouped source excerpts.

Map-reduce path:

- Used when source text exceeds the direct budget.
- Map step summarizes each bounded group.
- Reduce step combines partial summaries.
- Partial summaries are capped by `SUMMARY_REDUCE_MAX_PARTIAL_CHARS`.

Partial/truncated behavior:

- `SUMMARY_MAX_CHUNKS` caps included chunks.
- `SUMMARY_MAX_GROUPS` caps map groups.
- Truncated content sets `is_partial=True`.
- Warnings are persisted in JSON and shown in UI.

## 10. Markdown Heading-Aware Behavior

Markdown grouping uses existing chunk `heading_path` metadata:

- Contiguous chunks with the same heading path are grouped together when possible.
- Content before the first heading uses existing `document start` metadata.
- Heading hints are included in prompts.
- If heading metadata is missing, grouping falls back to `document start`/chunk order.

Code blocks remain included because Phase 01 stored them as searchable chunk text.

## 11. PDF Page-Aware Behavior

PDF grouping uses existing chunk page metadata:

- Groups preserve min/max `page_start`/`page_end`.
- Prompt excerpts include page hints such as `page 1` or `pages 1-2`.
- No OCR or scanned-PDF behavior was added.

## 12. Prompt Construction Behavior

Prompt builders exist for:

- Direct overview summary.
- Map-step partial summary.
- Reduce-step final summary.

All summary prompts instruct the model to:

- Summarize only provided source text.
- Avoid outside knowledge.
- Avoid invented claims, sections, structure, or page references.
- State when source content is insufficient.
- Use the exact sections:
  - `Overview`
  - `Key points`
  - `Useful details`
  - `Limitations or caveats in the document`

Prompt metadata excludes API keys and full source text. Full prompt/source text is not stored in SQLite.

## 13. Streamlit Summary UI Behavior

The app now has a Document summary panel with:

- Document selector scoped to selected workspace.
- Summary mode selector with only `overview`.
- Summary model input defaulting to `OPENAI_MODEL`.
- Cached summary display.
- Generate summary button.
- Regenerate summary button.
- Status messages for no summary, cached, generated, skipped, and failed states.
- Model, prompt version, source counts, partial flag, warnings, token usage, and updated timestamp display.

The top-level temporary OpenAI API-key input is shared by chat and summaries. It remains only in `st.session_state`.

## 14. Auto-Summary Behavior

Auto-summary is controlled by existing `AUTO_SUMMARY`.

Behavior after successful Streamlit upload:

- If `AUTO_SUMMARY=false`, no summary action is taken.
- If `AUTO_SUMMARY=true` and no API key exists, a skipped message is shown after rerun.
- If `AUTO_SUMMARY=true` and an API key exists, `SummaryService.generate_for_document(...)` is called.
- Summary failure is shown as a warning/error and does not fail ingestion.

`IngestionService` was not changed to call OpenAI.

## 15. Token/Cost Handling

Cost controls:

- Default `AUTO_SUMMARY=false`.
- Cache hit avoids LLM calls.
- Regenerate is explicit.
- Direct and map-reduce budgets are approximate character/chunk caps.
- Very long documents are summarized partially with warnings.
- Tests use fake LLM clients only.

Token handling:

- Input/output/total tokens are stored when the LLM response includes them.
- Map-reduce token usage is aggregated across map and reduce calls.

No cost estimation was added.

## 16. Security and Privacy Handling

Implemented rules:

- No API keys are persisted in SQLite or files.
- Temporary UI keys stay in Streamlit session state.
- Summary cache stores only user-visible summary text and compact metadata.
- Full prompt text and full source chunk text are not persisted in summary rows.
- Summary service does not log document text or secrets.
- Runtime summary JSON artifacts are not created.
- Runtime storage remains Git-ignored.

## 17. Deletion Cleanup Behavior

Document deletion:

- Existing `DocumentRepository.delete(document_id)` deletes the document row.
- SQLite cascades delete related chunks and summary rows.
- `IngestionService.delete_document(...)` continues to remove the stored original file.

Workspace deletion:

- Existing workspace delete removes workspace row.
- SQLite cascades documents, chunks, chat rows, and summary rows.
- Existing filesystem cleanup removes the workspace directory.

Known limitation: SQLite and filesystem deletion are still not one atomic transaction.

## 18. Test Coverage Matrix

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_summary_service.py` | Summary repository upsert/cache key | Cache conflict updates row and token usage | Concurrent summary writes |
| `tests/test_summary_service.py` | Cache hit behavior | Second request returns cached result and avoids LLM call | Real OpenAI response shape beyond wrapper tests |
| `tests/test_summary_service.py` | Regenerate behavior | Regenerate calls LLM again and updates timestamp/text | UI click automation |
| `tests/test_summary_service.py` | Skipped states | No chunks and missing API key return skipped results | Streamlit visual status rendering |
| `tests/test_summary_service.py` | Grouping | Direct vs map-reduce, partial warnings, Markdown headings, PDF page hints | Very large real PDFs |
| `tests/test_summary_service.py` | Prompt construction | Required no-outside-knowledge rule and overview sections | Prompt quality judged by humans |
| `tests/test_summary_service.py` | Cascade cleanup | Document/workspace delete removes summary rows | Filesystem failure during delete |
| `tests/test_storage_sqlite.py` | Schema init | `document_summaries` exists and init is idempotent | Migration from older on-disk DBs with partial schema drift |
| Existing full suite | Regression coverage | Ingestion, retrieval, chat, OpenAI wrapper, citation, and config tests still pass | Browser-driven manual workflows |

## 19. Validation Results

Commands run:

```bash
uv sync
```

Status: pass. Resolved 110 packages and checked 91 packages.

```bash
uv run pytest
```

Status: pass. 81 tests passed, 3 warnings from PyMuPDF/SWIG import deprecations.

Note: the first sandboxed targeted pytest run failed because pytest could not access the default Windows temp/cache directory. The command was rerun with approved escalation and passed. Final full-suite pytest also passed with approved escalation.

```bash
uv run ruff check .
```

Status: pass. All checks passed.

```bash
uv run ruff format --check .
```

Status: pass. 69 files already formatted.

```bash
uv run app
```

Status: pass for startup smoke. Streamlit started and reported:

- Local URL: `http://localhost:8501`

The process was stopped after the smoke test. Full interactive manual summary generation was not completed because this validation run did not use a real OpenAI API key or browser walkthrough.

```bash
git status --short
```

Status: pass. Shows intended modified/untracked files for Phase 04 plus the untracked Phase 04 plan file.

```bash
git check-ignore -v storage/app.db storage/workspaces/example/summaries/example.json
```

Status: pass. Both paths are ignored by `.gitignore:7:storage/*`.

## 20. Known Risks and Limitations

- Summary quality depends on the selected OpenAI model and prompt behavior.
- Character-based budgets are approximate and may not align with actual model context limits.
- Very long documents may produce partial summaries; warnings are shown and persisted.
- Map-reduce can still be costly for large documents, though caps reduce risk.
- No browser automation was used for the full manual Streamlit summary workflow.
- No real OpenAI call was made during tests.
- SQLite/filesystem cleanup remains non-atomic.
- Existing SQLite databases created before Phase 04 need app startup or service initialization to run idempotent schema creation before summary use.

## 21. Reviewer Checklist

- [ ] Confirm no new runtime dependencies were added.
- [ ] Confirm `IngestionService` does not call OpenAI.
- [ ] Confirm summaries are per-document only.
- [ ] Confirm summary mode is limited to `overview`.
- [ ] Confirm summary cache is SQLite-only and no summary JSON artifacts are created.
- [ ] Confirm cache key includes document hash, mode, model, prompt version, and config hash.
- [ ] Confirm API keys are not stored in SQLite or files.
- [ ] Confirm full prompt/source chunk text is not persisted in summary cache rows.
- [ ] Confirm document/workspace deletes cascade summary rows.
- [ ] Confirm ingestion, retrieval debug, and chat behavior remain present.
- [ ] Confirm tests pass with mocked/fake LLM clients only.

## 22. Next Recommended Step

Have the user/reviewer validate Phase 04 in the Streamlit UI with a real temporary or `.env` OpenAI API key, then commit Phase 04 if acceptable.

Do not start Phase 05 until explicitly approved.
