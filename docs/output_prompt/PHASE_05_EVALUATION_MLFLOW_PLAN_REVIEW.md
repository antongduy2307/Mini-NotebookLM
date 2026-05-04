# Phase 05 Evaluation and MLflow Plan Review

Status: planning draft for user and external reviewer approval.

This document plans Phase 05 only. It must not be treated as implementation approval. Phase 05 should preserve Phase 00 scaffold behavior, Phase 01 ingestion/storage behavior, Phase 02 retrieval behavior, Phase 03 chat/QA behavior, Phase 04 summary behavior, and the accepted app launcher fix.

## 1. Phase Objective

Implement workspace-specific retrieval evaluation and optional MLflow observability.

Phase 05 should add:

- Workspace-specific evaluation cases.
- Create, edit, delete, import, and export evaluation cases in Streamlit.
- Retrieval-only batch evaluation using the existing Phase 02 `RetrievalService`.
- Metrics:
  - hit@k by expected filename
  - hit@k by expected page
  - hit@k by expected page range
- Per-case retrieval result details and compact retrieval traces.
- Optional MLflow logging only when configured and available.
- Tests that use fake retrieval and mocked MLflow only.

Phase 05 should make retrieval quality inspectable without introducing answer judging, model-based eval, or retrieval architecture changes.

## 2. Scope and Non-Scope

### In Scope

- Add an `evaluation/` package.
- Add idempotent SQLite tables for eval cases, eval runs, and eval run items.
- Add repository/service APIs for eval case CRUD and run persistence.
- Add JSON import/export helpers for workspace eval cases.
- Add retrieval metric helpers for filename/page/page-range hits.
- Add an eval runner that uses `RetrievalService.retrieve(...)`.
- Add compact per-case retrieval trace/result serialization.
- Add optional MLflow logging behind a lazy import and config check.
- Add a Streamlit Evaluation panel.
- Add tests without OpenAI calls, network, real embedding downloads, or a real MLflow server.

### Non-Scope

Phase 05 must not implement:

- answer-quality LLM-as-judge
- model-based evaluation
- OpenAI evaluation calls
- MLflow server orchestration
- Docker/deployment
- LangChain
- LlamaIndex
- reranking
- RRF
- retrieval architecture redesign
- summary evaluation
- chat quality scoring
- saved API-key manager
- OCR/scanned PDF support

## 3. Dependency Strategy and MLflow Optionality

Recommendation: make MLflow optional, not a normal runtime dependency.

Proposed `pyproject.toml` approach:

```toml
[project.optional-dependencies]
observability = [
  "mlflow",
]
```

Usage:

```bash
uv sync --extra observability
```

Reason:

- Evaluation CRUD, JSON import/export, metric calculation, and local eval runs do not need MLflow.
- `mlflow` is a large dependency and can complicate local installs.
- The app requirement says it must still work when MLflow is not configured.
- Lazy optional import keeps the default app usable without installing MLflow.

Runtime behavior:

- If `MLFLOW_TRACKING_URI` is empty: show `MLflow logging disabled` and run local evaluation normally.
- If `MLFLOW_TRACKING_URI` is set but `mlflow` is not importable: run local evaluation normally and show an actionable warning: `MLflow is configured but the optional mlflow package is not installed. Install with uv sync --extra observability.`
- If `MLFLOW_TRACKING_URI` is set and `mlflow` imports successfully: log the eval batch as one MLflow run.

Tests should mock the MLflow module/logger boundary and should not require the optional dependency.

Status: proposed; requires user approval because it changes dependency packaging shape.

## 4. Proposed File and Module Changes

Planned files to add:

```text
src/mini_notebooklm_rag/evaluation/__init__.py
src/mini_notebooklm_rag/evaluation/models.py
src/mini_notebooklm_rag/evaluation/repositories.py
src/mini_notebooklm_rag/evaluation/import_export.py
src/mini_notebooklm_rag/evaluation/metrics.py
src/mini_notebooklm_rag/evaluation/runner.py
src/mini_notebooklm_rag/evaluation/mlflow_logger.py
tests/test_evaluation_repository.py
tests/test_evaluation_import_export.py
tests/test_evaluation_metrics.py
tests/test_evaluation_runner.py
tests/test_mlflow_logger.py
```

Planned files to modify:

```text
pyproject.toml
README.md
src/mini_notebooklm_rag/storage/sqlite.py
src/mini_notebooklm_rag/streamlit_app.py
tests/test_storage_sqlite.py
tests/test_scaffold.py
```

Optional files if the Streamlit page becomes too large:

```text
src/mini_notebooklm_rag/ui/__init__.py
src/mini_notebooklm_rag/ui/evaluation_panel.py
```

Expected responsibilities:

- `evaluation/models.py`: dataclasses for eval cases, import/export payloads, run configs, per-case results, aggregate metrics, and run records.
- `evaluation/repositories.py`: SQLite CRUD for `eval_cases`, `eval_runs`, and `eval_run_items`.
- `evaluation/import_export.py`: validate, parse, and serialize workspace eval JSON.
- `evaluation/metrics.py`: filename/page/page-range hit logic and optional reciprocal-rank helpers.
- `evaluation/runner.py`: run retrieval for each case, compute metrics, persist run results.
- `evaluation/mlflow_logger.py`: optional lazy MLflow logging boundary.
- `streamlit_app.py`: render Evaluation panel while preserving existing ingestion, summary, retrieval debug, and chat panels.

No OpenAI, summary-eval, chat-eval, LangChain, LlamaIndex, Docker, or retrieval architecture modules should be added.

## 5. Evaluation Case Model

Each eval case belongs to one workspace.

Suggested dataclasses:

```python
@dataclass(frozen=True)
class EvalCase:
    id: str
    workspace_id: str
    question: str
    selected_document_ids: list[str]
    expected_filename: str
    expected_page: int | None
    expected_page_start: int | None
    expected_page_end: int | None
    expected_answer: str | None
    notes: str
    created_at: str
    updated_at: str
```

Validation rules:

- `question` is required and non-empty.
- `selected_document_ids` may contain 1 to 3 document IDs for retrieval evaluation.
- `expected_filename` is required for filename-hit metrics.
- `expected_page` is optional.
- `expected_page_start` and `expected_page_end` are optional but must be supplied together for range metrics.
- If `expected_page` is set, it should be positive.
- If page range is set, both values should be positive and `expected_page_start <= expected_page_end`.
- `expected_answer` is stored for future answer evaluation but not used in Phase 05 metrics.
- Markdown cases can use filename-only expectations if page is unknown.

Expected page interpretation:

- Filename-only case: `expected_filename` set, no expected page/range.
- Exact-page case: `expected_page` set.
- Range case: `expected_page_start` and `expected_page_end` set.
- If both `expected_page` and page range are set, exact page is evaluated separately and page-range metric uses the range.

## 6. SQLite Schema Changes

Use idempotent `CREATE TABLE IF NOT EXISTS` in `storage/sqlite.py`. No migration framework should be added in Phase 05.

### `eval_cases`

```sql
CREATE TABLE IF NOT EXISTS eval_cases (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    question TEXT NOT NULL,
    selected_document_ids TEXT NOT NULL,
    expected_filename TEXT NOT NULL,
    expected_page INTEGER,
    expected_page_start INTEGER,
    expected_page_end INTEGER,
    expected_answer TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_cases_workspace_updated
ON eval_cases(workspace_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_eval_cases_workspace_filename
ON eval_cases(workspace_id, expected_filename);
```

Notes:

- `selected_document_ids` stores JSON text.
- Document deletion should not cascade eval cases, because historical expected cases may remain useful. The UI should warn when selected document IDs no longer exist.
- Workspace deletion cascades eval cases.

### `eval_runs`

```sql
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    top_k INTEGER NOT NULL,
    dense_weight REAL NOT NULL,
    sparse_weight REAL NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_device TEXT NOT NULL,
    eval_case_count INTEGER NOT NULL,
    filename_hit_count INTEGER NOT NULL,
    filename_hit_rate REAL NOT NULL,
    page_hit_count INTEGER NOT NULL,
    page_hit_rate REAL,
    page_range_hit_count INTEGER NOT NULL,
    page_range_hit_rate REAL,
    mean_reciprocal_rank REAL,
    mlflow_run_id TEXT,
    warnings TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_workspace_created
ON eval_runs(workspace_id, created_at);
```

Notes:

- `page_hit_rate` can be `NULL` when no evaluated case has `expected_page`.
- `page_range_hit_rate` can be `NULL` when no evaluated case has a page range.
- `mean_reciprocal_rank` is optional; set `NULL` if not implemented.
- `warnings` stores JSON text.

### `eval_run_items`

```sql
CREATE TABLE IF NOT EXISTS eval_run_items (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    question TEXT NOT NULL,
    selected_document_ids TEXT NOT NULL,
    expected_filename TEXT NOT NULL,
    expected_page INTEGER,
    expected_page_start INTEGER,
    expected_page_end INTEGER,
    filename_hit INTEGER NOT NULL,
    page_hit INTEGER,
    page_range_hit INTEGER,
    filename_hit_rank INTEGER,
    page_hit_rank INTEGER,
    page_range_hit_rank INTEGER,
    reciprocal_rank REAL,
    retrieved_results TEXT NOT NULL,
    retrieval_trace TEXT NOT NULL,
    warnings TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES eval_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (case_id) REFERENCES eval_cases(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_run_items_run_id
ON eval_run_items(run_id);

CREATE INDEX IF NOT EXISTS idx_eval_run_items_case_id
ON eval_run_items(case_id);
```

Notes:

- `retrieved_results` stores compact JSON: rank, chunk ID, document ID, filename, citation, page range, heading path, scores. It should not store full chunk text by default.
- `retrieval_trace` stores compact trace metadata, not full chunk text.
- Deleting an eval case cascades associated run items; aggregate run rows remain but may have missing items if cases are deleted later. This is acceptable for MVP, but the UI should show run-item count from stored items.

## 7. JSON Import/Export Design

JSON format version:

```json
{
  "format_version": 1,
  "workspace_id": "optional workspace id",
  "exported_at": "UTC ISO timestamp",
  "cases": [
    {
      "question": "What is the main finding?",
      "selected_document_ids": ["doc-id"],
      "expected_filename": "paper.pdf",
      "expected_page": 5,
      "expected_page_start": null,
      "expected_page_end": null,
      "expected_answer": null,
      "notes": "Optional reviewer note"
    }
  ]
}
```

Export behavior:

- Export cases for the selected workspace only.
- Include `format_version`.
- Include case IDs only if useful for round-trip review; recommendation: include `id` but treat it as advisory on import.
- Do not include API keys or secrets.
- Do not include full retrieved chunk text or eval run results in case export.

Import behavior:

- Validate JSON shape and `format_version`.
- Validate each case according to eval case rules.
- Bind imported cases to the currently selected workspace, not blindly to the JSON workspace ID.
- Avoid overwriting existing cases by default.
- If an imported case has an existing ID, generate a new ID unless the user explicitly chooses an overwrite/replace option.
- Report per-case import errors rather than failing the whole import where practical.

Initial MVP recommendation:

- Support append-only import by default.
- Defer overwrite/merge by ID unless the user explicitly asks in implementation approval.

## 8. Retrieval Evaluation Runner Design

The eval runner should use Phase 02 `RetrievalService` and should not reimplement retrieval.

Near-signature:

```python
class EvaluationRunner:
    def run_batch(
        self,
        workspace_id: str,
        case_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> EvalRunResult: ...
```

Per-case flow:

1. Load eval case from SQLite.
2. Validate selected document IDs count is at most 3.
3. Call `RetrievalService.retrieve(...)` with the case question and selected documents.
4. Compare retrieved results with expected filename/page/page range.
5. Store compact per-case result in `eval_run_items`.
6. Include warnings from retrieval and eval validation.

Batch flow:

1. Validate workspace and cases.
2. Capture retrieval config:
   - `top_k`
   - `dense_weight`
   - `sparse_weight`
   - embedding model
   - embedding device
3. Run cases sequentially for MVP.
4. Aggregate metrics.
5. Persist `eval_runs` and `eval_run_items`.
6. Optionally log to MLflow if configured/available.
7. Return run object, item results, metrics, warnings, and MLflow status.

Missing/stale index behavior:

- Do not silently rebuild indexes.
- If `RetrievalService.retrieve(...)` returns warnings for missing/stale/empty index and no results, store the case as miss with warnings.
- UI should show actionable message to rebuild the index from the retrieval debug panel.

Failure behavior:

- One case failure should be captured as a per-case warning if possible, not crash the whole batch.
- Fatal DB/service errors can mark the run `failed`.

## 9. Metric Definitions

All metrics are retrieval-only. No LLM answer quality is evaluated in Phase 05.

### Filename Hit@K

Definition:

```text
filename_hit = any(result.filename == expected_filename for result in top_k results)
```

Recommended comparison:

- Exact string match after trimming.
- Optional case-insensitive comparison can be proposed if filenames vary by OS; recommendation: case-insensitive for display filenames while preserving original values in stored results.

Aggregate:

```text
filename_hit_rate@k = filename_hit_count / eval_case_count
```

### Page Hit@K

Only evaluated when `expected_page` is set.

Definition:

```text
page_hit = any(
    result.filename matches expected_filename
    and result.page_start is not null
    and result.page_end is not null
    and result.page_start <= expected_page <= result.page_end
)
```

Aggregate:

```text
page_hit_rate@k = page_hit_count / page_evaluable_case_count
```

If no cases have `expected_page`, store/display `N/A`.

### Page Range Hit@K

Only evaluated when `expected_page_start` and `expected_page_end` are set.

Definition:

```text
page_range_hit = any(
    result.filename matches expected_filename
    and result.page_start is not null
    and result.page_end is not null
    and ranges_overlap(
        result.page_start,
        result.page_end,
        expected_page_start,
        expected_page_end,
    )
)
```

Overlap logic:

```text
max(result_start, expected_start) <= min(result_end, expected_end)
```

Aggregate:

```text
page_range_hit_rate@k = page_range_hit_count / page_range_evaluable_case_count
```

If no cases have page ranges, store/display `N/A`.

### Optional Mean Reciprocal Rank

Recommendation: implement reciprocal rank only if it remains simple.

Definition:

- Find the first rank that satisfies filename match.
- `reciprocal_rank = 1 / rank`
- `mean_reciprocal_rank = average reciprocal_rank across cases`

Status: optional, not required for acceptance.

### Markdown Cases

For Markdown documents:

- Filename hit applies.
- Page/page-range metrics are `N/A` unless page fields are intentionally absent.
- Heading-based expected fields are deferred unless explicitly approved later.

## 10. Streamlit Evaluation UI Design

Add an Evaluation panel after retrieval/chat/summary, or under an `st.expander("Evaluation")` if the page is too long.

Controls:

- Selected workspace comes from existing workspace selector.
- Eval case list for workspace.
- Create eval case form:
  - question
  - selected documents multiselect, max 3
  - expected filename
  - expected page
  - expected page start/end
  - optional expected answer
  - notes
- Edit eval case form for selected case.
- Delete eval case button with confirmation.
- Import JSON uploader.
- Export JSON download button.
- Eval run controls:
  - select cases to run
  - `top_k`
  - `dense_weight`
  - `sparse_weight`
  - Run eval batch button

Outputs:

- Aggregate metrics:
  - filename hit rate@k
  - page hit rate@k or `N/A`
  - page range hit rate@k or `N/A`
  - optional MRR if implemented
- Per-case result table:
  - question
  - expected filename/page/range
  - filename hit
  - page hit
  - page-range hit
  - first hit rank(s)
  - warnings
- Expandable per-case details:
  - compact retrieved results
  - citations
  - dense/sparse/fused scores
  - compact retrieval trace
- Current retrieval config and embedding model/device.
- MLflow status:
  - disabled
  - configured but package missing
  - logged with run ID
  - logging failed but local eval completed

UI constraints:

- No OpenAI key is required.
- Do not show summary evaluation or chat-quality controls.
- Do not log full chunk text to MLflow by default.
- UI must use evaluation service/repository functions, not direct SQL.

## 11. MLflow Logging Design

MLflow should be a thin optional boundary in `evaluation/mlflow_logger.py`.

Near-signature:

```python
class MLflowEvalLogger:
    def __init__(self, tracking_uri: str): ...
    def is_enabled(self) -> bool: ...
    def log_eval_run(self, run_result: EvalRunResult, artifact_dir: Path) -> MLflowLogResult: ...
```

Lazy import:

```python
try:
    import mlflow
except ImportError:
    return disabled_or_missing_package_result
```

If `MLFLOW_TRACKING_URI` is configured and MLflow is importable:

Log one eval batch as one MLflow run.

Params:

- `embedding_model`
- `embedding_device`
- `top_k`
- `dense_weight`
- `sparse_weight`
- `workspace_id`
- `eval_case_count`

Metrics:

- `filename_hit_rate_at_k`
- `page_hit_rate_at_k` when applicable
- `page_range_hit_rate_at_k` when applicable
- `mean_reciprocal_rank` if implemented

Artifacts:

- `eval_cases.json`
- `eval_results.json`
- `retrieval_config.json`

Artifact privacy:

- Use compact retrieved results.
- Do not include full chunk text by default.
- Include questions, expected filenames/pages, citations, scores, and warnings.

If MLflow logging fails:

- Local eval run remains stored.
- UI shows warning with sanitized error.
- App does not crash.

No MLflow server orchestration:

- Do not start `mlflow ui`.
- Do not manage servers, ports, or background processes.

## 12. Storage and Artifact Design

Preferred storage:

- SQLite for eval cases, eval run metadata, and per-case run results.
- JSON import/export for portability.
- Temporary/runtime JSON files only when needed for MLflow artifacts.

Workspace eval directory:

```text
storage/workspaces/<workspace_id>/eval/
```

Recommended use:

- It may hold transient MLflow artifact payloads under a run-specific directory:

```text
storage/workspaces/<workspace_id>/eval/runs/<eval_run_id>/
  eval_cases.json
  eval_results.json
  retrieval_config.json
```

Decision point:

- Option A: create temporary artifact files with Python `tempfile` and remove them after MLflow logging.
- Option B: write durable local artifacts under the workspace `eval/` directory.

Recommendation: use temporary files for MLflow logging in Phase 05 and rely on SQLite for durable local results. This keeps runtime storage smaller and avoids adding a second persistent eval result source.

Git behavior:

- All runtime eval artifacts are under `storage/` and remain ignored by Git.

## 13. Security and Privacy Rules

Rules:

- Do not log API keys.
- Do not require OpenAI API keys for evaluation.
- Do not create or modify saved API-key manager behavior.
- Eval cases and results may contain user questions and document-derived expectations; treat them as local private runtime data.
- `storage/app.db` may contain eval cases and compact retrieval result metadata.
- MLflow artifacts may include questions, expected filenames/pages, citations, and scores.
- Do not log full chunk text to MLflow by default.
- If the user configures a remote `MLFLOW_TRACKING_URI`, warn that eval artifacts may leave the local machine.
- Do not include `.env`, `.env.local`, `.local/`, API keys, or raw secrets in export artifacts.
- Sanitize MLflow error messages before showing them in UI.

Privacy recommendation for UI:

- Show a notice near MLflow controls:
  `MLflow artifacts can include eval questions and retrieval metadata. Use a local tracking URI unless you are comfortable sending this data to the configured server.`

## 14. Test Plan

Tests must not require:

- OpenAI
- real embedding model download
- real MLflow server
- network
- browser automation

Planned tests:

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_evaluation_repository.py` | SQLite eval case/run CRUD | create/list/update/delete cases, persist run/items, workspace cascade | migration framework |
| `tests/test_evaluation_import_export.py` | JSON import/export | format version, append-only import, validation errors, no secrets | overwrite/merge import |
| `tests/test_evaluation_metrics.py` | hit logic | filename match, page containment, page-range overlap, Markdown filename-only cases | heading metrics |
| `tests/test_evaluation_runner.py` | runner orchestration | fake retrieval calls, metrics aggregation, per-case warnings, compact results without text | real FAISS/model behavior |
| `tests/test_mlflow_logger.py` | MLflow optional behavior | disabled when URI empty, missing package warning, mocked logging calls/params/metrics/artifacts | real MLflow server |
| `tests/test_storage_sqlite.py` | schema initialization | eval tables and indexes exist, foreign keys remain enabled | migrations |
| Existing tests | regression | ingestion/retrieval/chat/summary remain passing | browser-level UI |

Fake retrieval strategy:

- Provide a fake `RetrievalService` with deterministic `retrieve(...)` responses.
- Include PDF results with page ranges and Markdown results without pages.
- Assert no OpenAI calls are involved.

MLflow test strategy:

- Inject or monkeypatch a fake MLflow module/client boundary.
- Do not import real `mlflow` in required tests.
- Assert params, metrics, and artifact file names are passed to the logger.

Streamlit test strategy:

- Keep service and helper logic unit-tested.
- Manual UI smoke after implementation should cover create/import/export/run behavior.

## 15. Validation Commands

Run after Phase 05 implementation:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
git status --short
git check-ignore -v storage/app.db storage/workspaces/example/eval/results.json
```

If MLflow is optional, also run one optional dependency check:

```bash
uv sync --extra observability
uv run python -c "import mlflow; print('mlflow import ok')"
```

Manual app smoke:

1. Start `uv run app`.
2. Create/select workspace with indexed documents.
3. Add eval cases for one PDF filename/page and one Markdown filename.
4. Export cases as JSON.
5. Import the exported JSON into the same workspace and confirm append behavior.
6. Run eval batch.
7. Confirm aggregate metrics render.
8. Expand per-case retrieval traces.
9. Confirm no OpenAI key is required.
10. Confirm MLflow disabled message when `MLFLOW_TRACKING_URI=` is empty.
11. If local MLflow tracking URI is configured and optional dependency installed, confirm run logging.

## 16. Acceptance Criteria

Phase 05 is accepted when:

- Evaluation cases are persisted per workspace.
- Users can create, edit, delete, import, and export eval cases in UI.
- JSON import/export uses a documented format version.
- Evaluation runs use existing `RetrievalService`.
- Evaluation enforces the selected-document max of 3.
- Filename hit@k, page hit@k, and page range hit@k are computed and displayed.
- Per-case compact retrieval results and traces are displayed.
- Eval runs and run items are stored locally.
- The app works when `MLFLOW_TRACKING_URI` is empty.
- The app works when `MLFLOW_TRACKING_URI` is set but MLflow is not installed, with an actionable warning.
- MLflow logging works when tracking URI is configured and optional dependency is installed.
- MLflow artifacts avoid full chunk text by default.
- Tests pass without OpenAI, network, real MLflow server, or real embedding downloads.
- Existing ingestion, retrieval, chat, summary, and launcher behavior remain intact.

## 17. Risks and Known Limitations

- Retrieval metrics only measure whether expected sources are retrieved, not whether answers are good.
- Filename matching can be brittle if documents are renamed in future phases; MVP has no document rename.
- Page metrics are only meaningful for PDF chunks with page metadata.
- Markdown has filename-only metrics in Phase 05 unless heading expectations are approved later.
- Query-local score normalization means scores are not comparable across runs except through rank/hit metrics.
- Stale or missing FAISS indexes can make eval runs miss all cases; UI must prompt rebuild rather than silently rebuilding.
- MLflow is optional; reviewers must test both disabled and configured paths if approving observability.
- Remote MLflow servers may receive eval questions and compact retrieval metadata.
- SQLite schema continues to use idempotent table creation; future migration tooling may be needed.
- Streamlit page is growing; a small evaluation UI helper module may be justified if implementation would otherwise make `streamlit_app.py` hard to review.

## 18. Questions for User Review

Blocking before implementation:

- None. The Phase 05 scope is clear enough to implement after approval.

Non-blocking review decisions:

1. Approve MLflow as an optional dependency under an `observability` extra rather than a normal dependency?
2. Approve append-only JSON import by default, with overwrite/merge deferred?
3. Approve storing eval runs and run items in SQLite while using JSON only for import/export and optional MLflow artifacts?
4. Approve compact MLflow artifacts that exclude full chunk text by default?
5. Approve optional mean reciprocal rank if it stays small, while keeping filename/page/page-range hit rates as required metrics?
6. Approve leaving Markdown heading-based expected metrics for a later phase?

## 19. Implementation Sequence for the Next Run

Recommended order after user approval:

1. Add optional MLflow dependency metadata under `observability`; do not make MLflow a required runtime dependency.
2. Add evaluation dataclasses in `evaluation/models.py`.
3. Add `eval_cases`, `eval_runs`, and `eval_run_items` schema to `storage/sqlite.py`; update schema tests.
4. Add evaluation repository CRUD and tests.
5. Add metric helpers and tests.
6. Add JSON import/export helpers and tests.
7. Add evaluation runner with fake-retrieval tests.
8. Add optional MLflow logger boundary with mocked tests for disabled, missing package, and successful logging states.
9. Update Streamlit with Evaluation panel using service/repository APIs.
10. Update README with Phase 05 usage, optional MLflow setup, and privacy notes.
11. Run validation commands.
12. Run manual UI smoke for local eval with MLflow disabled.
13. Optionally run MLflow import/logging smoke if the optional extra is installed.
14. Create a Phase 05 implementation review digest.
15. Stop before Docker/deployment or any later phase.

## 20. Reviewer Checklist

- [ ] Plan is retrieval-evaluation only, not answer-quality evaluation.
- [ ] Plan does not add OpenAI eval calls, LLM judging, Docker, LangChain, LlamaIndex, reranking, RRF, OCR, or saved-key behavior.
- [ ] MLflow is optional and lazy-imported.
- [ ] Eval UI works without MLflow installed or configured.
- [ ] SQLite schema defines eval cases, runs, run items, keys, indexes, cascades, and JSON fields.
- [ ] JSON import/export has format version and validation rules.
- [ ] Metrics define exact filename, page, and page-range hit logic.
- [ ] Runner uses existing `RetrievalService`.
- [ ] Missing/stale index behavior is actionable and does not silently rebuild.
- [ ] MLflow params, metrics, and artifacts are explicitly scoped.
- [ ] MLflow artifacts exclude full chunk text by default.
- [ ] Tests avoid OpenAI, network, real MLflow server, and real embedding downloads.
- [ ] Security/privacy notes cover eval data and remote MLflow risk.
- [ ] Implementation remains gated on explicit user approval.
