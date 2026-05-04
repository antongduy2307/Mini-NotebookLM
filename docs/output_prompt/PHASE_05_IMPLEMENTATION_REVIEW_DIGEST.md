# Phase 05 Implementation Review Digest

## 1. Executive Summary

Phase 05 implements workspace-specific retrieval evaluation and optional MLflow observability.

Implemented:
- Evaluation case CRUD in SQLite.
- Evaluation run and per-case result persistence in SQLite.
- JSON import/export for eval cases.
- Retrieval-only metrics: filename hit@k, page hit@k, page range hit@k, and mean reciprocal rank.
- Eval runner over the existing Phase 02 `RetrievalService`.
- Optional MLflow logging behind the `observability` extra.
- Streamlit Evaluation panel.
- Tests for repositories, import/export, metrics, runner behavior, optional MLflow logging, and schema creation.

Not implemented:
- Answer-quality judging.
- OpenAI evaluation calls.
- Model-based eval.
- Summary/chat quality scoring.
- MLflow server orchestration.
- Docker/deployment.
- Retrieval architecture changes.

## 2. Files Changed

Created:
- `src/mini_notebooklm_rag/evaluation/__init__.py`: public exports for evaluation helpers.
- `src/mini_notebooklm_rag/evaluation/models.py`: dataclasses for eval cases, runs, metrics, compact retrieved results, import errors, and MLflow status.
- `src/mini_notebooklm_rag/evaluation/repositories.py`: SQLite CRUD for eval cases, runs, and run items.
- `src/mini_notebooklm_rag/evaluation/import_export.py`: JSON import/export format version 1.
- `src/mini_notebooklm_rag/evaluation/metrics.py`: retrieval-only metric functions.
- `src/mini_notebooklm_rag/evaluation/runner.py`: batch runner using existing retrieval service.
- `src/mini_notebooklm_rag/evaluation/mlflow_logger.py`: lazy optional MLflow logging boundary.
- `tests/test_evaluation_import_export.py`: JSON import/export coverage.
- `tests/test_evaluation_metrics.py`: metric calculation coverage.
- `tests/test_evaluation_repository.py`: eval repository and cascade behavior coverage.
- `tests/test_evaluation_runner.py`: runner coverage with fake retrieval.
- `tests/test_mlflow_logger.py`: disabled, missing, and mocked MLflow logging coverage.
- `docs/output_prompt/PHASE_05_IMPLEMENTATION_REVIEW_DIGEST.md`: this review digest.

Modified:
- `pyproject.toml`: added optional `observability = ["mlflow"]` extra only.
- `uv.lock`: updated lock metadata for optional MLflow dependency graph.
- `src/mini_notebooklm_rag/storage/sqlite.py`: added eval tables and additive column guards for early local eval DBs.
- `src/mini_notebooklm_rag/streamlit_app.py`: added Evaluation UI panel.
- `tests/test_storage_sqlite.py`: updated expected tables.
- `README.md`: documented Phase 05 retrieval evaluation and optional MLflow setup.

No retrieval, ingestion, chat, summary, embedding, FAISS, or OpenAI evaluation architecture was redesigned.

## 3. Module-by-Module Implementation Summary

### `evaluation/models.py`

Main classes:
- `EvalCase`
- `NewEvalCase`
- `EvalRunConfig`
- `CompactRetrievedResult`
- `EvalCaseMetrics`
- `EvalRunItemResult`
- `EvalAggregateMetrics`
- `EvalRunRecord`
- `EvalRunResult`
- `MLflowLogResult`

Responsibility:
- Defines typed data contracts for retrieval eval.

Does not:
- Touch SQLite, filesystem, retrieval, OpenAI, or MLflow directly.

### `evaluation/repositories.py`

Main functions/classes:
- `EvaluationRepository`
- `validate_eval_case`
- `new_eval_id`

Responsibility:
- Persists eval cases, runs, and run items in SQLite.
- Validates required question, 1 to 3 selected documents, expected filename, and page/range consistency.
- Preserves imported IDs when available; generates a new ID if an imported ID already exists.

Does not:
- Run retrieval.
- Calculate metrics.
- Read/write JSON files.
- Delete eval cases when a document is deleted.

### `evaluation/import_export.py`

Main functions:
- `export_cases_payload`
- `export_cases_json`
- `parse_import_payload`

Responsibility:
- Serializes workspace eval cases as JSON format version 1.
- Parses imported JSON append-only and binds imported cases to the current workspace.
- Reports per-case validation errors.

Does not:
- Persist imported cases directly.
- Include secrets, retrieved chunk text, or eval run results.

### `evaluation/metrics.py`

Main functions:
- `filenames_match`
- `ranges_overlap`
- `evaluate_case`
- `aggregate_metrics`

Responsibility:
- Computes retrieval-only hit metrics.
- Uses case-insensitive filename matching.
- Treats page and page-range metrics as not applicable when expectations are absent.

Does not:
- Judge answer quality.
- Use LLMs.
- Evaluate Markdown heading expectations.

### `evaluation/runner.py`

Main classes/functions:
- `RetrievalServiceProtocol`
- `EvaluationRunner`

Responsibility:
- Loads selected eval cases.
- Runs existing `RetrievalService.retrieve(...)` per case.
- Converts retrieval results/traces to compact forms without full chunk text.
- Computes metrics, persists run/items, and invokes optional MLflow logging.

Does not:
- Rebuild indexes silently.
- Reimplement retrieval.
- Call OpenAI.
- Persist full chunk text.

### `evaluation/mlflow_logger.py`

Main class:
- `MLflowEvalLogger`

Responsibility:
- Lazy-imports MLflow only when `MLFLOW_TRACKING_URI` is configured.
- Logs one eval batch as one MLflow run when optional MLflow is installed.
- Logs params, metrics, and compact JSON artifacts.
- Returns disabled/missing/failed/logged status instead of crashing the app.

Does not:
- Add MLflow as a required dependency.
- Start or manage an MLflow server.
- Log full chunk text.

### `storage/sqlite.py`

Phase 05 additions:
- `eval_cases`
- `eval_runs`
- `eval_run_items`
- Indexes for workspace/run/case lookup.
- `_ensure_column(...)` additive compatibility guard for eval run metric count columns.

Responsibility:
- Idempotent schema initialization.

Does not:
- Add a migration framework.

### `streamlit_app.py`

Phase 05 additions:
- Evaluation panel with create/edit/delete eval case controls.
- JSON import/export.
- Eval batch runner controls.
- Aggregate and per-case results display.
- MLflow status display.

Does not:
- Require an OpenAI key.
- Add answer-quality controls.
- Trigger automatic index rebuilds.

## 4. Public/Internal API Summary

### `EvaluationRepository.create_case(case: NewEvalCase) -> EvalCase`

Inputs:
- New eval case data.

Outputs:
- Persisted `EvalCase`.

Failure behavior:
- Raises `EvalValidationError` for invalid case data.
- Raises `EvalRepositoryError` if insert cannot be read back.

Touches:
- SQLite only.

### `EvaluationRepository.update_case(case_id: str, case: NewEvalCase) -> EvalCase`

Inputs:
- Case ID and replacement case fields.

Outputs:
- Updated `EvalCase`.

Failure behavior:
- Raises validation error for invalid data.
- Raises repository error if target case is missing.

Touches:
- SQLite only.

### `EvaluationRepository.list_cases(workspace_id: str) -> list[EvalCase]`

Inputs:
- Workspace ID.

Outputs:
- Workspace eval cases ordered by updated time.

Failure behavior:
- Empty list when no cases exist.

Touches:
- SQLite only.

### `EvaluationRepository.create_run(...) -> EvalRunRecord`

Inputs:
- Workspace ID, run status, retrieval config, aggregate metrics, run items, warnings, optional MLflow run ID.

Outputs:
- Persisted `EvalRunRecord`.

Failure behavior:
- Raises repository error if run cannot be read back.

Touches:
- SQLite only.

### `parse_import_payload(raw_json: str, workspace_id: str) -> tuple[list[NewEvalCase], list[ImportValidationError]]`

Inputs:
- Uploaded JSON text and target workspace ID.

Outputs:
- Valid cases bound to target workspace plus per-case errors.

Failure behavior:
- Invalid JSON/version returns errors instead of raising.

Touches:
- Neither SQLite nor filesystem.

### `evaluate_case(eval_case: EvalCase, results: Sequence[CompactRetrievedResult]) -> EvalCaseMetrics`

Inputs:
- One eval case and compact retrieval results.

Outputs:
- Filename/page/page-range hit flags and hit ranks.

Failure behavior:
- No exceptions for empty results; returns miss metrics.

Touches:
- Neither SQLite nor filesystem.

### `EvaluationRunner.run_batch(...) -> EvalRunResult`

Inputs:
- Workspace ID, case IDs, top_k, dense_weight, sparse_weight.

Outputs:
- Persisted run, run items, and MLflow status.

Failure behavior:
- Missing selected cases produce run warning.
- Per-case retrieval exceptions become miss + warning; recoverable failures do not crash the batch.

Touches:
- SQLite and existing retrieval index/files through `RetrievalService`.

### `MLflowEvalLogger.log_eval_run(run_result, cases) -> MLflowLogResult`

Inputs:
- Eval run result and cases.

Outputs:
- Status: disabled, missing, logged, or failed.

Failure behavior:
- Missing optional package and MLflow exceptions return actionable statuses.

Touches:
- Temporary local files for artifacts during MLflow logging only.
- MLflow tracking backend only when configured.

## 5. Dependency and Optional Extra Changes

Required runtime dependencies:
- Unchanged.

Optional dependency:

```toml
[project.optional-dependencies]
observability = [
    "mlflow",
]
```

Default `uv sync` remains lightweight and does not keep MLflow installed. Optional validation installed the extra, verified import, then default `uv sync` removed the optional packages again.

## 6. Final Evaluation Architecture

Evaluation is layered as:

1. Streamlit Evaluation panel gathers case definitions and run parameters.
2. `EvaluationRepository` stores cases and prior run records in SQLite.
3. `EvaluationRunner` loads cases and calls existing `RetrievalService`.
4. Retrieved chunks are compacted to metadata only.
5. `metrics.py` computes hit metrics.
6. Run and item results are persisted in SQLite.
7. `MLflowEvalLogger` optionally logs compact params, metrics, and artifacts.

The retrieval implementation remains the Phase 02 hybrid FAISS + BM25 service.

## 7. SQLite Eval Schema

### `eval_cases`

Columns:
- `id TEXT PRIMARY KEY`
- `workspace_id TEXT NOT NULL`
- `question TEXT NOT NULL`
- `selected_document_ids TEXT NOT NULL`
- `expected_filename TEXT NOT NULL`
- `expected_page INTEGER`
- `expected_page_start INTEGER`
- `expected_page_end INTEGER`
- `expected_answer TEXT`
- `notes TEXT NOT NULL DEFAULT ''`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

Foreign keys:
- `workspace_id -> workspaces(id) ON DELETE CASCADE`

Indexes:
- `idx_eval_cases_workspace_updated`
- `idx_eval_cases_workspace_filename`

Document deletes do not cascade eval cases by design.

### `eval_runs`

Columns:
- `id TEXT PRIMARY KEY`
- `workspace_id TEXT NOT NULL`
- `status TEXT NOT NULL CHECK (status IN ('completed', 'failed'))`
- `top_k INTEGER NOT NULL`
- `dense_weight REAL NOT NULL`
- `sparse_weight REAL NOT NULL`
- `embedding_model TEXT NOT NULL`
- `embedding_device TEXT NOT NULL`
- `eval_case_count INTEGER NOT NULL`
- `filename_hit_count INTEGER NOT NULL`
- `filename_hit_rate REAL NOT NULL`
- `page_evaluable_count INTEGER NOT NULL`
- `page_hit_count INTEGER NOT NULL`
- `page_hit_rate REAL`
- `page_range_evaluable_count INTEGER NOT NULL`
- `page_range_hit_count INTEGER NOT NULL`
- `page_range_hit_rate REAL`
- `mean_reciprocal_rank REAL`
- `mlflow_run_id TEXT`
- `warnings TEXT NOT NULL`
- `created_at TEXT NOT NULL`
- `completed_at TEXT NOT NULL`

Foreign keys:
- `workspace_id -> workspaces(id) ON DELETE CASCADE`

Indexes:
- `idx_eval_runs_workspace_created`

### `eval_run_items`

Columns:
- `id TEXT PRIMARY KEY`
- `run_id TEXT NOT NULL`
- `workspace_id TEXT NOT NULL`
- `case_id TEXT NOT NULL`
- `question TEXT NOT NULL`
- `selected_document_ids TEXT NOT NULL`
- `expected_filename TEXT NOT NULL`
- `expected_page INTEGER`
- `expected_page_start INTEGER`
- `expected_page_end INTEGER`
- `filename_hit INTEGER NOT NULL`
- `page_hit INTEGER`
- `page_range_hit INTEGER`
- `filename_hit_rank INTEGER`
- `page_hit_rank INTEGER`
- `page_range_hit_rank INTEGER`
- `reciprocal_rank REAL`
- `retrieved_results TEXT NOT NULL`
- `retrieval_trace TEXT NOT NULL`
- `warnings TEXT NOT NULL`
- `created_at TEXT NOT NULL`

Foreign keys:
- `run_id -> eval_runs(id) ON DELETE CASCADE`
- `workspace_id -> workspaces(id) ON DELETE CASCADE`
- `case_id -> eval_cases(id) ON DELETE CASCADE`

Indexes:
- `idx_eval_run_items_run_id`
- `idx_eval_run_items_case_id`

## 8. Eval Case Behavior

Required:
- `question`
- `selected_document_ids`, 1 to 3 IDs
- `expected_filename`

Optional:
- `expected_page`
- `expected_page_start` and `expected_page_end` as a pair
- `expected_answer`, stored for future phases but unused in metrics
- `notes`

Validation:
- Page numbers must be positive.
- Page ranges must include both start and end.
- Page start cannot exceed page end.
- Filename matching during evaluation is case-insensitive.

## 9. JSON Import/Export Behavior

Export:
- Uses `format_version = 1`.
- Includes eval case definitions only.
- Excludes secrets, retrieval results, run results, and chunk text.

Import:
- Binds cases to the current workspace.
- Is append-only.
- Preserves imported ID when possible.
- Generates a new ID when the imported ID already exists.
- Reports per-case validation errors without failing all imports.

## 10. Metric Definitions and Implementation

Filename hit@k:
- True if any retrieved result filename matches `expected_filename` case-insensitively.

Page hit@k:
- Applicable only when `expected_page` is set.
- True if filename matches and `expected_page` lies within the retrieved chunk page range.

Page range hit@k:
- Applicable only when expected start/end are set.
- True if filename matches and retrieved chunk page range overlaps expected range.

Mean reciprocal rank:
- Implemented over filename hit rank.
- Returns `None` only when no reciprocal ranks exist.

Markdown heading-based metrics:
- Deferred.

## 11. Eval Runner Behavior

Runner behavior:
- Uses `RetrievalService.retrieve(...)`.
- Does not rebuild missing or stale indexes.
- Treats retrieval exceptions as per-case misses with warnings.
- Allows one case failure without failing the full batch.
- Stores compact retrieved results and compact retrieval trace only.

Compact result fields:
- rank
- chunk_id
- document_id
- filename
- citation
- source_type
- page_start/page_end
- heading_path
- dense_score
- sparse_score
- fused_score

Excluded:
- full chunk text
- full source document text

## 12. MLflow Optional Logging Behavior

Disabled:
- Empty `MLFLOW_TRACKING_URI` returns `MLflow logging disabled.`
- Local eval still runs.

Missing package:
- Configured tracking URI with missing optional package returns:
  `MLflow is configured but the optional mlflow package is not installed. Install with uv sync --extra observability.`
- Local eval still runs.

Logged:
- One eval batch maps to one MLflow run.
- Params logged:
  - embedding_model
  - embedding_device
  - top_k
  - dense_weight
  - sparse_weight
  - workspace_id
  - eval_case_count
- Metrics logged:
  - filename_hit_rate_at_k
  - page_hit_rate_at_k when applicable
  - page_range_hit_rate_at_k when applicable
  - mean_reciprocal_rank when applicable
- Artifacts logged:
  - `eval_cases.json`
  - `eval_results.json`
  - `retrieval_config.json`

Artifacts are generated in a temporary directory and exclude full chunk text by default.

## 13. Streamlit Evaluation UI Behavior

The Evaluation panel supports:
- Creating eval cases.
- Editing eval cases.
- Deleting eval cases.
- Exporting workspace eval cases as JSON.
- Importing eval cases JSON append-only.
- Selecting cases to run.
- Configuring top_k, dense_weight, and sparse_weight.
- Running retrieval eval batches.
- Displaying aggregate metrics.
- Displaying per-case compact retrieval details.
- Displaying MLflow status.

The UI requires no OpenAI key and exposes no answer-quality controls.

## 14. Storage/Artifact Behavior

Durable local storage:
- Eval cases, runs, and run items are stored in SQLite.

Portable storage:
- JSON is used for import/export only.

MLflow artifacts:
- Generated temporarily for logging.
- Compact.
- No full chunk text.

Git ignore:
- `storage/app.db` is ignored.
- `storage/workspaces/example/eval/results.json` is ignored.

## 15. Security and Privacy Handling

No API keys:
- Evaluation does not call OpenAI.
- Evaluation UI does not require an OpenAI key.
- No API keys are stored or logged.

Local privacy:
- Eval cases and results are local SQLite data and may contain user-authored questions or expected answers.
- Compact retrieval metadata may include filenames, citations, scores, and chunk IDs.

MLflow privacy:
- Remote MLflow tracking may receive eval questions and compact retrieval metadata.
- Full chunk text is excluded by default.
- Users should use local tracking unless comfortable sending this metadata to the configured server.

## 16. Test Coverage Matrix

| Test file | Behavior covered | Important assertions | Not covered |
| --- | --- | --- | --- |
| `tests/test_storage_sqlite.py` | Schema initialization | Eval tables exist; FK enabled | Migration framework |
| `tests/test_evaluation_metrics.py` | Hit metrics | Case-insensitive filename, page containment, page range overlap, N/A handling, aggregate rates | Markdown heading metrics |
| `tests/test_evaluation_import_export.py` | JSON import/export | Format version, workspace rebinding, per-case errors | UI upload interaction |
| `tests/test_evaluation_repository.py` | SQLite CRUD | Append import ID conflict, update/delete, document delete does not delete cases, workspace delete cascades, run item persistence | Concurrent writes |
| `tests/test_evaluation_runner.py` | Batch runner | Existing retrieval service is called, compact results exclude chunk text, recoverable retrieval failure becomes miss | Real FAISS index |
| `tests/test_mlflow_logger.py` | Optional MLflow boundary | Disabled state, missing package message, mocked successful logging and compact artifacts | Real MLflow server |
| Existing regression tests | Ingestion, retrieval, chat, summary, app scaffold | 96 total tests pass | Browser automation |

## 17. Validation Results

Commands run:

```bash
uv sync
```

Result:
- Initial attempt failed because prior local `app.exe`/Python processes locked `.venv\Scripts\app.exe`.
- Stopped project-local app/python processes with elevated permission.
- Re-run passed.

```bash
uv run pytest
```

Result:
- Passed.
- `96 passed, 3 warnings`.

```bash
uv run ruff check .
```

Result:
- Passed after formatting/import fixes.

```bash
uv run ruff format --check .
```

Result:
- Passed.
- `81 files already formatted`.

```bash
uv run app
```

Result:
- Bounded startup smoke passed.
- App started on `http://localhost:8877`.
- Project app processes were stopped afterward.

```bash
git status --short
```

Result:
- Shows Phase 05 modified/created files.
- Also shows pre-existing untracked `docs/output_prompt/PHASE_05_EVALUATION_MLFLOW_PLAN_REVIEW.md`.

```bash
git check-ignore -v storage/app.db storage/workspaces/example/eval/results.json
```

Result:
- Both ignored by `.gitignore:7:storage/*`.

```bash
uv sync --extra observability
```

Result:
- First attempt failed under restricted network sandbox while downloading `flask-cors`.
- Re-run with approved escalation passed and installed optional MLflow dependencies.

```bash
uv run python -c "import mlflow; print('mlflow import ok')"
```

Result:
- Passed.
- Printed `mlflow import ok`.

```bash
uv sync
```

Result:
- Passed.
- Restored default lightweight environment and uninstalled optional MLflow packages.

Manual UI smoke:
- Full browser click-through was not performed.
- Bounded Streamlit startup smoke was performed and passed.
- Service-level tests cover eval case CRUD, import/export, eval runner, metrics, and MLflow optional states without network or OpenAI.

## 18. Known Risks and Limitations

- Full interactive browser smoke was not completed in this run.
- Evaluation depends on existing retrieval indexes; missing/stale indexes are reported as misses/warnings and are not rebuilt silently.
- Eval run storage is compact and intentionally omits full chunk text, so forensic review of exact source text requires rerunning retrieval or using the retrieval debug UI.
- JSON import is append-only; overwrite/merge is deferred.
- Markdown heading-based expected metrics are deferred.
- MLflow artifacts may still include eval questions, filenames, citations, and compact retrieval metadata.
- Remote MLflow tracking can leak that metadata to the configured server.
- No MLflow server orchestration is provided.
- Per-case retrieval exceptions are recoverable, which is useful for batch robustness but can reduce metrics if a configuration problem affects many cases.

## 19. Reviewer Checklist

- [ ] MLflow is optional under `[project.optional-dependencies].observability`.
- [ ] `mlflow` is not a required runtime dependency.
- [ ] Evaluation uses existing `RetrievalService` and does not reimplement retrieval.
- [ ] Evaluation does not call OpenAI.
- [ ] Eval cases, runs, and run items are persisted in SQLite.
- [ ] JSON import/export excludes secrets and chunk text.
- [ ] Import is append-only and handles imported ID conflicts.
- [ ] Metrics match approved filename/page/page-range definitions.
- [ ] Runtime storage and eval artifacts remain Git-ignored.
- [ ] Tests avoid real OpenAI, real embedding downloads, real MLflow server, and network.
- [ ] Phase 05 does not implement Docker/deployment or later phases.

## 20. Next Recommended Step

Have the user/reviewer run an interactive UI smoke against real indexed documents:

1. Create/select a workspace with indexed documents.
2. Add one PDF eval case with filename/page expectation.
3. Add one Markdown eval case with filename-only expectation.
4. Export cases.
5. Import the export and confirm append behavior.
6. Run an eval batch.
7. Inspect aggregate metrics and compact per-case retrieval details.
8. Optionally set a local `MLFLOW_TRACKING_URI` and run with `uv sync --extra observability`.

Docker/deployment has not started.
