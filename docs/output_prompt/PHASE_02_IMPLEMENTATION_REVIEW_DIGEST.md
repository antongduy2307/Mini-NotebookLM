# Phase 02 Implementation Review Digest

## 1. Executive Summary

Phase 02 implemented local retrieval over Phase 01 SQLite chunks. The app now supports:

- lazy local embeddings through `sentence-transformers`
- `auto|cuda|cpu` embedding device selection
- normalized vectors with FAISS `IndexFlatIP`
- one FAISS index and metadata file per workspace
- in-memory BM25 rebuilt from SQLite chunks
- weighted dense/sparse hybrid retrieval
- selected-document filtering with a 3-document cap
- citation formatting for PDF and Markdown chunks
- structured retrieval traces
- a Streamlit retrieval debug panel

Phase 02 did not implement OpenAI calls, answer generation, query rewriting, chat sessions, summaries, evaluation UI, MLflow, API key persistence, LangChain, LlamaIndex, reranking, RRF, OCR, or tokenizer-dependent chunking changes.

## 2. Files Changed

Created:

- `src/mini_notebooklm_rag/retrieval/__init__.py`: retrieval package marker.
- `src/mini_notebooklm_rag/retrieval/models.py`: retrieval dataclasses.
- `src/mini_notebooklm_rag/retrieval/embeddings.py`: embedding wrapper and device resolver.
- `src/mini_notebooklm_rag/retrieval/faiss_store.py`: FAISS index build/load/search/status behavior.
- `src/mini_notebooklm_rag/retrieval/bm25_store.py`: in-memory BM25 sparse retrieval.
- `src/mini_notebooklm_rag/retrieval/hybrid.py`: dense/sparse normalization and fusion.
- `src/mini_notebooklm_rag/retrieval/citations.py`: citation formatter.
- `src/mini_notebooklm_rag/retrieval/service.py`: retrieval orchestration service.
- `tests/test_embedding_device.py`: device resolver tests.
- `tests/test_embedding_wrapper.py`: fake-model embedding wrapper tests.
- `tests/test_faiss_store.py`: FAISS build/search/save/load/stale tests.
- `tests/test_bm25_store.py`: BM25 tokenization/search/filter tests.
- `tests/test_hybrid_retrieval.py`: score normalization/fusion tests.
- `tests/test_citations.py`: citation formatter tests.
- `tests/test_retrieval_service.py`: retrieval service integration tests with fake embeddings.
- `docs/output_prompt/PHASE_02_IMPLEMENTATION_REVIEW_DIGEST.md`: this review digest.

Modified:

- `pyproject.toml`: added `sentence-transformers`, `faiss-cpu`, and `rank-bm25`.
- `uv.lock`: updated by `uv sync`.
- `README.md`: updated current status, Phase 02 scope, retrieval runtime notes, and planning doc links.
- `src/mini_notebooklm_rag/storage/paths.py`: added FAISS index path helpers.
- `src/mini_notebooklm_rag/storage/repositories.py`: added chunk read/count helper methods.
- `src/mini_notebooklm_rag/streamlit_app.py`: added retrieval debug UI while preserving ingestion UI.
- `tests/test_document_repository.py`: added coverage for new repository chunk helpers.

No source files for OpenAI, chat, summaries, evaluation, MLflow, LangChain, or LlamaIndex were created.

## 3. Module-by-Module Implementation Summary

`retrieval/models.py`

- Main classes: `EmbeddingInfo`, `DenseCandidate`, `SparseCandidate`, `RetrievedChunk`, `RetrievalTrace`, `FaissPosition`, `FaissMetadata`, `IndexStatus`, `RetrievalResponse`.
- Responsibility: provide typed internal API records for retrieval services and tests.
- Does not perform retrieval, I/O, logging, or model loading.

`retrieval/embeddings.py`

- Main functions/classes: `resolve_embedding_device`, `EmbeddingModel`, `EmbeddingDeviceError`, `EmbeddingModelError`.
- Responsibility: choose embedding device, lazy-load `SentenceTransformer`, encode text into normalized `float32` arrays, support fake model injection for tests.
- Does not persist embeddings, download models during tests, or log query/document text.

`retrieval/faiss_store.py`

- Main functions/classes: `compute_chunk_fingerprint`, `FaissStore`, `FaissIndexError`.
- Responsibility: build workspace FAISS indexes from chunks, save/load `faiss.index`, save/load `faiss_meta.json`, detect missing/current/stale/empty state, search dense candidates with selected-document filtering.
- Does not delete individual vectors, persist embedding caches, or retrieve from BM25.

`retrieval/bm25_store.py`

- Main functions/classes: `tokenize`, `BM25Store`.
- Responsibility: rebuild sparse retrieval state in memory from chunk records and return sparse candidates.
- Does not persist BM25 artifacts or implement language-specific stemming.

`retrieval/hybrid.py`

- Main functions/classes: `normalize_scores`, `fuse_results`, `HybridRetrievalError`.
- Responsibility: normalize dense/sparse scores to `[0, 1]`, normalize weights, reject invalid weights, produce ranked `RetrievedChunk` records with citations.
- Does not use RRF or reranking.

`retrieval/citations.py`

- Main function: `format_citation`.
- Responsibility: format PDF page/range citations and Markdown heading citations.
- Does not depend on SQLite, FAISS, BM25, or Streamlit.

`retrieval/service.py`

- Main classes/functions: `RetrievalService`, `RetrievalError`, `MAX_SELECTED_DOCUMENTS`.
- Responsibility: coordinate repositories, embedding model, FAISS, BM25, hybrid fusion, selected-document validation, index status, rebuild, and retrieval responses.
- Does not call OpenAI, create chat sessions, or log traces automatically.

`storage/paths.py`

- Added methods: `indexes_dir`, `faiss_index_path`, `faiss_metadata_path`.
- Responsibility: resolve runtime index paths under the existing storage root safety model.
- Does not create FAISS files by itself.

`storage/repositories.py`

- Added methods: `list_chunks_for_workspace`, `list_chunks_for_documents`, `get_chunks_by_ids`, `count_chunks_for_workspace`, `count_chunks_for_documents`.
- Responsibility: read chunk metadata for retrieval without schema changes.
- Does not perform filesystem operations or index building.

`streamlit_app.py`

- Added retrieval debug panel.
- Responsibility: expose index status, rebuild button, document selection, query input, retrieval controls, scores, citations, and expandable chunks.
- Does not talk to SQL directly, request API keys, call OpenAI, or implement chat.

## 4. Public/Internal API Summary

`resolve_embedding_device(requested_device, cuda_available=None) -> str`

- Inputs: requested device string and optional test hook.
- Outputs: `cpu` or `cuda`.
- Failure: raises `EmbeddingDeviceError` for invalid device or unavailable explicit CUDA.
- Touches: neither filesystem nor SQLite.

`EmbeddingModel.encode(texts: list[str]) -> np.ndarray`

- Inputs: text list.
- Outputs: normalized `float32` matrix.
- Failure: raises `EmbeddingModelError` for load/encode/shape issues.
- Touches: model loading only; no SQLite/filesystem writes.

`FaissStore.build(workspace_id, chunks) -> IndexStatus`

- Inputs: workspace ID and ordered chunk records.
- Outputs: index status and metadata.
- Failure: raises `FaissIndexError` for invalid embeddings.
- Touches: filesystem only, writing `faiss.index` and `faiss_meta.json`.

`FaissStore.status(workspace_id, chunks) -> IndexStatus`

- Inputs: workspace ID and current chunks.
- Outputs: `empty|missing|current|stale` status.
- Failure: returns stale status for unreadable metadata.
- Touches: filesystem reads.

`FaissStore.search(workspace_id, query, top_k, selected_document_ids=None) -> list[DenseCandidate]`

- Inputs: workspace ID, query, candidate count, optional document filter.
- Outputs: dense candidates.
- Failure: raises `FaissIndexError` if index missing.
- Touches: filesystem reads and embedding model.

`BM25Store.from_chunks(chunks) -> BM25Store`

- Inputs: chunk records.
- Outputs: in-memory BM25 store.
- Failure: none expected for valid chunks.
- Touches: neither filesystem nor SQLite.

`BM25Store.search(query, top_k, selected_document_ids=None) -> list[SparseCandidate]`

- Inputs: query, result count, optional document filter.
- Outputs: sparse candidates.
- Failure: empty query or no matches returns `[]`.
- Touches: neither filesystem nor SQLite.

`fuse_results(chunks, dense_candidates, sparse_candidates, top_k, dense_weight, sparse_weight) -> list[RetrievedChunk]`

- Inputs: chunk records, dense/sparse candidates, top-k, weights.
- Outputs: final ranked retrieval chunks.
- Failure: raises `HybridRetrievalError` for invalid top-k or weights.
- Touches: neither filesystem nor SQLite.

`RetrievalService.index_status(workspace_id) -> IndexStatus`

- Inputs: workspace ID.
- Outputs: workspace index status.
- Failure: returns status object for missing/stale states.
- Touches: SQLite reads and filesystem reads.

`RetrievalService.rebuild_index(workspace_id) -> IndexStatus`

- Inputs: workspace ID.
- Outputs: build status.
- Failure: propagates embedding/FAISS errors.
- Touches: SQLite reads and filesystem writes.

`RetrievalService.retrieve(workspace_id, query, selected_document_ids, top_k, dense_weight, sparse_weight) -> RetrievalResponse`

- Inputs: workspace ID, query, selected docs, top-k, weights.
- Outputs: final results, trace, warnings.
- Failure: raises `RetrievalError` for invalid inputs; missing/stale index returns empty response with warnings.
- Touches: SQLite reads, filesystem reads, embedding model.

## 5. Dependency Changes

Added direct runtime dependencies:

- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`

No direct `torch` dependency was added. `torch` was installed transitively by `sentence-transformers` during `uv sync`.

No `openai`, `mlflow`, `tiktoken`, LangChain, LlamaIndex, cross-encoder reranker, or vector DB server dependency was added.

## 6. Final Retrieval Architecture

Retrieval flow:

1. Phase 01 stores chunks in SQLite.
2. `RetrievalService.rebuild_index()` reads workspace chunks.
3. `EmbeddingModel` encodes and normalizes chunk text.
4. `FaissStore` writes a workspace `IndexFlatIP` index plus metadata.
5. At query time, `FaissStore` returns dense candidates.
6. `BM25Store` rebuilds from SQLite chunks in memory and returns sparse candidates.
7. `fuse_results()` normalizes and fuses candidate scores.
8. `RetrievalResponse` returns ranked chunks and a trace.
9. Streamlit renders citations, scores, and expandable source chunks.

No schema change was made in Phase 02.

## 7. Embedding Device and Normalization Behavior

Device behavior:

- `EMBEDDING_DEVICE=auto`: chooses CUDA if PyTorch reports CUDA available, otherwise CPU.
- `EMBEDDING_DEVICE=cpu`: forces CPU.
- `EMBEDDING_DEVICE=cuda`: raises an actionable error if CUDA is unavailable.

Normalization behavior:

- Embedding output is converted to `np.ndarray`.
- Vectors are coerced to `float32`.
- 1D output is reshaped to one row.
- Each row is L2-normalized.
- Zero vectors remain zero-safe by treating norm `0` as `1`.

Tests use fake model injection and do not download a real model.

## 8. FAISS Index Metadata Format and Stale Detection Behavior

FAISS runtime paths:

```text
storage/workspaces/<workspace_id>/indexes/faiss.index
storage/workspaces/<workspace_id>/indexes/faiss_meta.json
```

Metadata fields:

- `workspace_id`
- `embedding_model`
- `embedding_dimension`
- `normalized`
- `built_at`
- `chunk_count`
- `chunk_fingerprint`
- `positions`

Each `positions` item includes:

- `vector_index`
- `chunk_id`
- `document_id`

Stale detection compares:

- workspace ID
- embedding model
- normalization flag
- chunk fingerprint from ordered `(chunk_id, content_hash)` pairs

Status values:

- `empty`: workspace has no chunks.
- `missing`: chunks exist but index or metadata file is missing.
- `current`: metadata matches SQLite chunks and embedding settings.
- `stale`: metadata cannot be read or no longer matches chunks/settings.

FAISS vector deletion is not implemented. Document changes require rebuild.

## 9. BM25 Behavior

BM25 behavior:

- Rebuilt in memory from SQLite chunks.
- Not persisted to disk.
- Tokenization is deterministic and English-first: lowercase regex tokens from `[A-Za-z0-9_]+`.
- Selected-document filtering is applied before candidate ranking.
- Queries with no matching corpus tokens return no sparse candidates.

Known limitation: no stemming, stopword handling, multilingual segmentation, or persisted sparse cache.

## 10. Hybrid Fusion and Score Normalization Behavior

Fusion behavior:

- Dense scores and sparse scores are normalized independently to `[0, 1]`.
- If all scores in a non-empty set are equal, each receives `1.0`.
- Missing dense/sparse scores contribute `0.0`.
- Weights are normalized internally to sum to `1.0`.
- Both weights zero is rejected.

Sort order:

1. fused score descending
2. dense score descending
3. sparse score descending
4. document ID
5. chunk index

RRF and reranking are not implemented.

## 11. Selected-Document Filtering Behavior

Rules:

- At most 3 documents may be selected.
- Zero selected documents returns an empty response with a warning.
- Selected documents with no chunks return an empty response with a warning.
- FAISS over-retrieves with `max(top_k * 5, top_k + 20)` capped by index size.
- If filtered FAISS results are too few and the index is larger than the over-retrieval count, FAISS searches all vectors and filters again.
- BM25 filters selected document IDs before returning sparse candidates.

## 12. Citation Behavior

PDF citations:

- Same page: `filename, p. 5`
- Page range: `filename, pp. 5-6`
- Missing page metadata: `filename`

Markdown citations:

- Heading: `filename > Heading`
- Nested heading: `filename > Parent > Child`
- Missing heading: `filename > document start`

The formatter is independent and unit-tested.

## 13. Retrieval Trace Behavior

`RetrievalTrace` includes:

- original query
- selected document IDs
- embedding model
- embedding device
- top-k
- dense weight
- sparse weight
- dense candidate list
- sparse candidate list
- fused final list
- warnings

Traces are returned to callers but not automatically logged because final results can contain document text.

## 14. Streamlit UI Behavior

The app now shows:

- workspace and document ingestion UI from Phase 01
- retrieval debug panel for selected workspace
- embedding model/device info
- index status and chunk counts
- build/rebuild workspace index button
- document multiselect with a 3-document cap
- retrieval query input
- top-k, dense weight, sparse weight controls
- retrieval results with citations and scores
- expandable chunk text
- Phase 03 notice for answer generation and chat

The UI does not request API keys, call OpenAI, save secrets, implement chat, or implement summaries/evaluation.

## 15. Test Coverage Matrix

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_embedding_device.py` | Device resolver | `auto` CUDA/CPU behavior, explicit CUDA error, invalid device rejection | Real GPU runtime |
| `tests/test_embedding_wrapper.py` | Embedding wrapper | lazy fake model load, float32 output, L2 normalization, dimension exposure | Real model download |
| `tests/test_faiss_store.py` | FAISS store | index files written, metadata positions, search filtering, stale detection, empty cleanup | Large index performance |
| `tests/test_bm25_store.py` | BM25 store | tokenization, selected-doc filtering, unmatched query | stemming/multilingual behavior |
| `tests/test_hybrid_retrieval.py` | Hybrid fusion | equal-score normalization, missing score handling, weight normalization, zero-weight rejection | RRF/reranking |
| `tests/test_citations.py` | Citation formatter | PDF page/range and Markdown heading/document-start citations | citation grouping UI |
| `tests/test_retrieval_service.py` | Service orchestration | missing index warning, rebuild/retrieve, selected-doc cap, stale index, zero-weight rejection | real embedding model |
| `tests/test_document_repository.py` | Repository helpers | workspace chunk list, selected-doc chunk list, get-by-IDs order, chunk counts | migration behavior |
| Existing Phase 01 tests | Regression coverage | ingestion, parsers, chunking, paths, SQLite, workspace/document lifecycle | browser-level UI testing |

## 16. Validation Results

Commands run:

- `uv sync`: pass. Installed approved retrieval dependencies. `torch` appeared only as a transitive dependency of `sentence-transformers`.
- `uv run pytest`: initially failed due a service indentation error and two test issues; fixed. Final result: pass, 48 tests passed with 3 FAISS/PyMuPDF-related deprecation warnings.
- `uv run ruff check .`: initially failed on formatting/import ordering; fixed with `uv run ruff format .` and `uv run ruff check . --fix`. Final result: pass.
- `uv run ruff format --check .`: pass, 43 files already formatted.
- `uv run app`: bounded startup run printed Streamlit local URL `http://localhost:8501`; command timed out because Streamlit is long-running. Follow-up process check showed no `streamlit`, `uv`, or `python` process left running.
- `git check-ignore -v storage/workspaces/example/indexes/faiss.index storage/workspaces/example/indexes/faiss_meta.json`: pass. Both files are ignored by `.gitignore:7:storage/*`.
- `git status --short`: showed only expected Phase 02 source/test/doc/dependency changes before this digest was added.

Manual smoke:

- Browser-level manual UI smoke was not completed in this run because the shell-based process management attempts hit Windows `Start-Process`/`Start-Job` issues and real index building through the UI would download the default embedding model.
- Service-level smoke was completed with fake embeddings and ignored repo-local runtime storage:
  - created workspace
  - uploaded Markdown
  - uploaded PDF
  - duplicate upload skipped
  - built FAISS index
  - retrieved one selected-document result
  - deleted a document
  - confirmed index became stale
  - rebuilt index
  - deleted workspace
- Temporary smoke runtime directory `storage/phase02_smoke` was removed after validation.

## 17. Security, Path, and Cost Checks

Security:

- No API key persistence was added.
- No OpenAI calls were added.
- No secrets were added to `.env.example`, source, tests, or docs.
- Retrieval traces are not automatically logged.

Path safety:

- FAISS paths are resolved through `StoragePaths`.
- Runtime index files live under `storage/workspaces/<workspace_id>/indexes/`.
- Git ignore verification confirms FAISS runtime files are ignored.

Cost:

- Retrieval is local and has no API cost.
- First real embedding use may download `BAAI/bge-base-en-v1.5`.
- Tests avoid real model downloads through fake embedding injection.

## 18. Known Risks and Limitations

- Real `BAAI/bge-base-en-v1.5` loading was not smoke-tested to avoid model download during automated validation.
- CUDA behavior is unit-tested with mocked availability, not on real GPU hardware.
- `faiss-cpu` installed successfully here, but native packages can remain platform-sensitive.
- No embedding vector cache exists, so rebuilding FAISS re-embeds all workspace chunks.
- FAISS and SQLite writes are not one atomic transaction.
- BM25 tokenization is simple and English-first.
- Score normalization is per query and not comparable across queries.
- Browser-level Streamlit workflow was not fully smoke-tested in this run.
- Streamlit debug UI displays source chunk text by design; traces should still not be automatically logged.

## 19. Reviewer Checklist

- [ ] Only approved Phase 02 dependencies were added.
- [ ] No direct `torch` dependency was added.
- [ ] No OpenAI, chat, summary, eval, MLflow, API key, LangChain, or LlamaIndex implementation was added.
- [ ] Embedding device behavior follows the approved `auto|cpu|cuda` rules.
- [ ] Embeddings are normalized before FAISS indexing/search.
- [ ] FAISS uses one `IndexFlatIP` index per workspace.
- [ ] FAISS metadata maps vector positions to chunk/document IDs.
- [ ] Stale index detection compares chunk fingerprint and embedding settings.
- [ ] BM25 is rebuilt in memory and not persisted.
- [ ] Hybrid fusion uses weighted normalized scores, not RRF.
- [ ] Retrieval enforces at most 3 selected documents.
- [ ] Citation formatting matches PDF and Markdown requirements.
- [ ] Streamlit UI remains a retrieval debug panel, not chat.
- [ ] Tests avoid real embedding downloads.
- [ ] Runtime FAISS files are Git-ignored.

## 20. Next Recommended Step

User/reviewer should validate Phase 02, especially the Streamlit retrieval panel with a real local embedding model if model download is acceptable.

Do not start Phase 03 until Phase 02 is reviewed and approved.
