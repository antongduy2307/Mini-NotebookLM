# Phase 02 Retrieval Plan Review

Status: planning draft for user and external reviewer approval.

This document plans Phase 02 only. It does not approve implementation by itself. Phase 02 must preserve the completed Phase 00 app shell and Phase 01 workspace/document ingestion behavior.

## 1. Phase Objective

Implement the local retrieval foundation over chunks already persisted by Phase 01:

- Load a local open-source embedding model.
- Select CUDA when available for `EMBEDDING_DEVICE=auto`, otherwise fall back to CPU.
- Build and load one FAISS dense vector index per workspace.
- Rebuild BM25 sparse retrieval state from SQLite chunks when needed.
- Run hybrid dense + sparse retrieval over up to 3 selected documents.
- Return citation-ready retrieval results and structured trace metadata.
- Add a minimal Streamlit retrieval debug panel, not a chat interface.
- Add focused tests for retrieval components without OpenAI, network calls, or real model downloads.

Phase 02 should make retrieval inspectable and testable. It should not generate answers.

## 2. Scope and Non-Scope

### In Scope

- Add retrieval-only dependencies.
- Add a `retrieval/` package with embedding, FAISS, BM25, hybrid fusion, citation, trace, and service modules.
- Add repository read helpers for chunks already stored by Phase 01.
- Build/rebuild workspace FAISS indexes from SQLite chunks.
- Persist FAISS index files and metadata under each workspace `indexes/` directory.
- Rebuild BM25 in memory from SQLite chunks.
- Filter retrieval by selected document IDs, with a maximum of 3 selected documents per retrieval request.
- Expose configurable `top_k`, `dense_weight`, and `sparse_weight`.
- Format PDF and Markdown citations from chunk metadata.
- Show retrieval debug results in Streamlit.
- Show selected embedding device in UI/dev info.
- Add tests using fake embeddings and small synthetic chunks.

### Non-Scope

Phase 02 must not implement:

- OpenAI calls.
- Answer generation.
- Query rewriting.
- Conversational chat sessions.
- Conversation history.
- Summaries.
- Evaluation UI.
- MLflow.
- API key persistence.
- LangChain.
- LlamaIndex.
- Cross-encoder reranking.
- Reciprocal rank fusion unless approved in a later phase.
- Tokenizer-dependent chunking changes.
- OCR or scanned PDF behavior changes.

## 3. Dependencies to Add and Why

Proposed runtime dependencies:

- `sentence-transformers`: loads local embedding models and encodes chunk/query text.
- `faiss-cpu`: stores and searches dense vectors locally in a per-workspace FAISS index.
- `rank-bm25`: provides BM25 sparse retrieval over persisted chunk text.

No direct `torch` dependency is proposed unless `sentence-transformers` resolution or device detection fails without it. `sentence-transformers` normally brings PyTorch transitively. CUDA-enabled PyTorch installation is environment-specific, so Phase 02 should not encode CUDA wheel assumptions into `pyproject.toml`.

Dependencies explicitly not added in Phase 02:

- `openai`
- `mlflow`
- `tiktoken`
- LangChain
- LlamaIndex
- cross-encoder reranker packages
- vector database server packages

Decision: use `faiss-cpu` even when embeddings may run on CUDA.

Reason: the MVP stores indexes locally and does not require GPU FAISS. CUDA support in FAISS packaging is platform-specific and would make setup less reliable.

Status: proposed, requires user approval.

Decision: do not add a direct `torch` dependency in Phase 02 unless implementation proves it is necessary.

Reason: CUDA PyTorch installation varies by machine; `sentence-transformers` can supply the normal dependency path, and device detection can be wrapped defensively.

Status: proposed, requires user approval.

## 4. Proposed File and Module Changes

Planned files to add:

```text
src/mini_notebooklm_rag/retrieval/__init__.py
src/mini_notebooklm_rag/retrieval/models.py
src/mini_notebooklm_rag/retrieval/embeddings.py
src/mini_notebooklm_rag/retrieval/faiss_store.py
src/mini_notebooklm_rag/retrieval/bm25_store.py
src/mini_notebooklm_rag/retrieval/hybrid.py
src/mini_notebooklm_rag/retrieval/citations.py
src/mini_notebooklm_rag/retrieval/service.py
tests/test_embedding_device.py
tests/test_embedding_wrapper.py
tests/test_faiss_store.py
tests/test_bm25_store.py
tests/test_hybrid_retrieval.py
tests/test_citations.py
tests/test_retrieval_service.py
```

Planned files to modify:

```text
pyproject.toml
README.md
src/mini_notebooklm_rag/streamlit_app.py
src/mini_notebooklm_rag/storage/paths.py
src/mini_notebooklm_rag/storage/repositories.py
tests/test_document_repository.py
tests/test_scaffold.py
```

Expected responsibilities:

- `retrieval/models.py`: dataclasses for embedding info, retrieval candidates, final results, and traces.
- `retrieval/embeddings.py`: lazy sentence-transformers wrapper and device resolution.
- `retrieval/faiss_store.py`: build, save, load, stale-check, and search FAISS workspace indexes.
- `retrieval/bm25_store.py`: deterministic tokenization and in-memory BM25 retrieval.
- `retrieval/hybrid.py`: score normalization and weighted fusion.
- `retrieval/citations.py`: independent citation formatter.
- `retrieval/service.py`: orchestration layer used by Streamlit; no direct SQL in UI.
- `storage/paths.py`: add index path helpers for `faiss.index` and `faiss_meta.json`.
- `storage/repositories.py`: add chunk read helpers for workspace/document filtering.

No `llm/`, `evaluation/`, or API key modules should be added in Phase 02.

## 5. Embedding Model and Device Design

Default model:

```text
BAAI/bge-base-en-v1.5
```

Reason: it is already the planned default in `.env.example`, is a strong general English embedding model, and fits the project goal of local open-source retrieval.

Configuration already exists:

```env
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=32
```

Device resolver behavior:

- `auto`: use CUDA if importable PyTorch reports CUDA available; otherwise use CPU.
- `cpu`: force CPU.
- `cuda`: require CUDA and raise an actionable error if unavailable.

The stricter behavior for explicit `cuda` prevents silently running slower than requested. The `auto` path remains safe and portable.

Embedding wrapper near-signature:

```python
class EmbeddingModel:
    def __init__(self, model_name: str, requested_device: str, batch_size: int): ...
    @property
    def info(self) -> EmbeddingInfo: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
```

Behavior:

- Load the `SentenceTransformer` model lazily on first encode.
- Return a 2D `np.ndarray` of `float32`.
- Normalize embeddings before returning them.
- Expose model name, requested device, selected device, embedding dimension, and normalization flag.
- Never log full document text or query text by default.

Normalization decision:

Use normalized embeddings and FAISS inner product search.

Reason: normalized vectors with inner product produce cosine-like similarity, which is appropriate for sentence-transformer embeddings and gives dense scores with a predictable range for hybrid fusion.

Status: proposed, requires user approval.

## 6. FAISS Index Design

One FAISS index is stored per workspace:

```text
storage/workspaces/<workspace_id>/indexes/faiss.index
storage/workspaces/<workspace_id>/indexes/faiss_meta.json
```

Index type:

```text
faiss.IndexFlatIP
```

Reason: normalized embeddings plus inner product give cosine-like similarity, require no training step, and keep MVP behavior transparent.

Metadata file shape:

```json
{
  "workspace_id": "workspace id",
  "embedding_model": "BAAI/bge-base-en-v1.5",
  "embedding_dimension": 768,
  "normalized": true,
  "built_at": "UTC ISO timestamp",
  "chunk_count": 123,
  "chunk_fingerprint": "sha256 of ordered chunk ids and hashes",
  "positions": [
    {
      "vector_index": 0,
      "chunk_id": "chunk id",
      "document_id": "document id"
    }
  ]
}
```

Build flow:

1. Read all chunks for the workspace from SQLite.
2. If no chunks exist, do not create FAISS files and return a no-index status.
3. Encode chunk text in batches.
4. Normalize vectors through the embedding wrapper.
5. Build `IndexFlatIP`.
6. Save `faiss.index`.
7. Save `faiss_meta.json` with vector-position mapping and chunk fingerprint.

Search flow:

1. Load FAISS index and metadata.
2. Check metadata compatibility with current workspace, model, dimension, and chunk fingerprint.
3. Encode query.
4. Search dense candidates.
5. Map FAISS positions back to chunk IDs.
6. Fetch chunk records from SQLite.
7. Filter selected document IDs in Python.

Selected-document filtering:

- FAISS index is workspace-wide.
- Phase 02 should not build per-document FAISS indexes.
- Search should over-retrieve with `max(top_k * 5, top_k + 20)`.
- If filtering returns too few results and the workspace index is small enough, search all vectors and filter again.

Rationale: FAISS `IndexFlatIP` has no native metadata filter. Python filtering is acceptable for this local MVP and avoids additional index complexity.

## 7. BM25 Design

Use `rank-bm25` with an in-memory corpus rebuilt from SQLite chunks.

BM25 does not need to be persisted in Phase 02.

Reason: BM25 state is lightweight compared with embeddings, and rebuilding from SQLite avoids cache invalidation complexity.

Tokenization:

```text
lowercase text -> regex word tokens -> discard empty tokens
```

Suggested regex:

```python
r"[A-Za-z0-9_]+"
```

This keeps tokenization deterministic, dependency-light, and English-first.

BM25 store near-signature:

```python
class BM25Store:
    @classmethod
    def from_chunks(cls, chunks: list[ChunkRecord]) -> BM25Store: ...
    def search(
        self,
        query: str,
        top_k: int,
        selected_document_ids: set[str] | None = None,
    ) -> list[SparseCandidate]: ...
```

Selected-document filtering:

- The store may score all workspace chunks and filter selected document IDs before ranking.
- For small local MVP corpora this is acceptable.
- If performance becomes a problem, a later phase can build a temporary selected-document BM25 corpus per query.

## 8. Hybrid Retrieval and Score Normalization Design

Hybrid retrieval should start with weighted score fusion, not RRF.

Config:

```env
RETRIEVAL_TOP_K=6
DENSE_WEIGHT=0.65
SPARSE_WEIGHT=0.35
```

Validation:

- `top_k` must be positive.
- `dense_weight` and `sparse_weight` must be non-negative.
- If both weights are zero, reject the request with an actionable error.
- Normalize weights internally to sum to 1.0 before fusion.

Score normalization:

- Normalize dense candidate scores to `[0, 1]` within the query candidate set.
- Normalize sparse BM25 scores to `[0, 1]` within the query candidate set.
- If a non-empty candidate set has identical scores, assign `1.0` to each candidate in that set.
- Missing dense or sparse score for a chunk contributes `0.0`.

Fusion formula:

```text
fused_score = dense_weight_normalized * dense_score_normalized
            + sparse_weight_normalized * sparse_score_normalized
```

Final result fields:

- `chunk_id`
- `document_id`
- `filename`
- `text`
- `source_type`
- `page_start`
- `page_end`
- `heading_path`
- `dense_score`
- `sparse_score`
- `fused_score`
- `rank`
- `citation`

Tie-breakers:

1. Higher fused score.
2. Higher dense normalized score.
3. Higher sparse normalized score.
4. Lower original chunk index if available.

## 9. Citation Formatting Design

Add an independent formatter in `retrieval/citations.py`.

Near-signature:

```python
def format_citation(
    filename: str,
    source_type: str,
    page_start: int | None = None,
    page_end: int | None = None,
    heading_path: list[str] | None = None,
) -> str: ...
```

PDF behavior:

- Same page: `filename, p. 5`
- Page range: `filename, pp. 5-6`
- Missing page metadata: `filename`

Markdown behavior:

- Single heading: `filename > Heading`
- Nested headings: `filename > Parent > Child`
- Missing or empty heading path: `filename > document start`

This formatter should have no FAISS, BM25, Streamlit, or SQLite dependency.

## 10. Retrieval Trace Design

Add a serializable trace object for future dev panel and evaluation.

Suggested fields:

```python
@dataclass(frozen=True)
class RetrievalTrace:
    original_query: str
    selected_document_ids: list[str]
    embedding_model: str
    embedding_device: str
    top_k: int
    dense_weight: float
    sparse_weight: float
    dense_candidates: list[DenseCandidate]
    sparse_candidates: list[SparseCandidate]
    fused_results: list[RetrievedChunk]
    warnings: list[str]
```

Trace rules:

- Include chunk IDs, document IDs, scores, and ranks.
- Do not include API keys.
- Avoid logging trace objects automatically because they contain document text in final results.
- UI may render traces explicitly inside expanders.

## 11. SQLite and Repository Additions

No schema change is proposed for Phase 02.

Reason: Phase 01 already stores all metadata needed for retrieval: chunk text, source type, filename, page range, heading path, document ID, workspace ID, and content hash.

Proposed repository methods:

```python
class DocumentRepository:
    def list_chunks_for_workspace(self, workspace_id: str) -> list[ChunkRecord]: ...
    def list_chunks_for_documents(
        self,
        workspace_id: str,
        document_ids: list[str],
    ) -> list[ChunkRecord]: ...
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkRecord]: ...
    def count_chunks_for_workspace(self, workspace_id: str) -> int: ...
    def count_chunks_for_documents(
        self,
        workspace_id: str,
        document_ids: list[str],
    ) -> dict[str, int]: ...
```

Ordering requirements:

- Workspace chunk lists should order by `document_id`, then `chunk_index` for deterministic index builds.
- `get_chunks_by_ids` should return records in requested chunk ID order when practical.

Failure behavior:

- Unknown workspace/document IDs return empty lists where appropriate.
- Repository methods should not create files or mutate data.

## 12. Streamlit UI Changes

Add a minimal retrieval debug panel below the existing Phase 01 document management UI.

UI controls:

- Workspace selector remains the Phase 01 selector.
- Document multiselect for the selected workspace.
- Enforce at most 3 selected documents.
- Query text input.
- `top_k` numeric control.
- `dense_weight` and `sparse_weight` controls.
- Build/rebuild workspace index button.
- Run retrieval button.

UI outputs:

- Current embedding model.
- Selected embedding device.
- Index status:
  - missing
  - current
  - stale
  - empty workspace
- Retrieved chunks with:
  - rank
  - citation
  - dense score
  - sparse score
  - fused score
  - expandable chunk text
- Warning that answer generation and conversational chat are Phase 03.

UI constraints:

- Streamlit UI must use retrieval services, not direct SQL.
- UI must not call OpenAI.
- UI must not request or persist API keys.
- UI must not implement chat history or summaries.

## 13. Index Lifecycle and Stale-Index Handling

Initial build:

- User clicks "Build/rebuild workspace index" in the retrieval debug panel.
- Service reads all workspace chunks, builds embeddings, writes FAISS files, and writes metadata.

After new document ingestion:

- The new chunks make the stored chunk fingerprint differ from SQLite.
- UI should show index as stale and prompt rebuild.
- Optional implementation detail: after a successful upload in the same UI session, show a rebuild reminder rather than automatically embedding immediately.

After document deletion:

- Do not delete individual FAISS vectors.
- The old index becomes stale because metadata references deleted chunks.
- UI should show stale status and require rebuild.

Missing index:

- Retrieval should not crash.
- UI should show "Build the workspace index before dense retrieval."
- BM25-only retrieval may be possible, but the recommended Phase 02 default is to require index build for full hybrid retrieval and show an actionable message.

Workspace has no chunks:

- Do not create FAISS files.
- Return an empty-index status.
- UI should ask the user to upload documents first.

Selected documents have no chunks:

- Return no results with a clear warning.

Stale detection:

- Compute a chunk fingerprint from ordered `(chunk_id, content_hash)` pairs for the workspace.
- Compare it with `faiss_meta.json`.
- Also compare embedding model name and normalized flag.

## 14. Test Plan

Tests must not require OpenAI, network access, or downloading a real embedding model.

Planned tests:

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_embedding_device.py` | Device resolver | `auto` chooses mocked CUDA when available, falls back to CPU, explicit `cuda` errors when unavailable | Real GPU runtime |
| `tests/test_embedding_wrapper.py` | Embedding wrapper with fake model | lazy load, batch encode call, float32 output, normalized vectors | Real sentence-transformers download |
| `tests/test_faiss_store.py` | Build/search/save/load FAISS index | metadata position mapping, index reload, selected-doc over-retrieval behavior | Large indexes |
| `tests/test_bm25_store.py` | Tokenization and sparse search | deterministic tokens, selected-document filtering, empty query handling | Language-specific stemming |
| `tests/test_hybrid_retrieval.py` | Score normalization and fusion | dense/sparse normalization, missing score handling, weight normalization, tie-breakers | RRF or reranking |
| `tests/test_citations.py` | Citation formatting | PDF page/range and Markdown heading/document start formats | UI grouping of citations |
| `tests/test_retrieval_service.py` | End-to-end retrieval orchestration with fake embeddings | index status, rebuild, hybrid retrieval, selected-doc limit, stale warnings | Real model performance |
| `tests/test_document_repository.py` | New chunk read helpers | deterministic chunk ordering, selected-document chunk lists, counts | Schema migration |
| `tests/test_scaffold.py` | Settings defaults remain stable | retrieval config defaults still match `.env.example` | Streamlit runtime |

Manual smoke test after implementation:

- Start `uv run app`.
- Create/select a workspace with already-ingested Markdown/PDF documents.
- Build workspace index.
- Select up to 3 documents.
- Run a retrieval query.
- Confirm citations and scores render.
- Delete a document and confirm the index becomes stale.
- Rebuild and confirm retrieval still works.
- Confirm there is no answer-generation/chat behavior.

## 15. Validation Commands

Run after Phase 02 implementation:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
git status --short
git check-ignore -v storage/workspaces/example/indexes/faiss.index storage/workspaces/example/indexes/faiss_meta.json
```

Expected results:

- `uv sync`: installs only approved retrieval dependencies in addition to existing Phase 00/01 dependencies.
- `uv run pytest`: passes without network or OpenAI.
- `uv run ruff check .`: passes lint checks.
- `uv run ruff format --check .`: confirms formatting.
- `uv run app`: starts the app and allows manual retrieval debug smoke testing.
- `git status --short`: shows only intended source/test/doc/dependency changes.
- `git check-ignore -v ...`: confirms runtime index files are ignored through `storage/*`.

Optional manual dependency smoke:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers import ok')"
uv run python -c "import faiss; print('faiss import ok')"
```

Do not run tests that download a real model by default.

## 16. Acceptance Criteria

Phase 02 is accepted when:

- Only approved retrieval dependencies are added.
- No OpenAI, MLflow, LangChain, LlamaIndex, chat, summary, or eval implementation is added.
- Embedding wrapper loads lazily and supports `auto`, `cuda`, and `cpu` device configuration.
- `auto` uses CUDA when available and CPU otherwise.
- Workspace FAISS index builds from SQLite chunks and persists under the workspace `indexes/` directory.
- FAISS metadata maps vector positions to chunk/document IDs.
- Stale index detection works after chunk changes.
- BM25 rebuilds from SQLite chunks and supports selected-document filtering.
- Hybrid retrieval returns ranked results with dense, sparse, fused scores, and citations.
- Retrieval enforces a maximum of 3 selected documents.
- Streamlit exposes a retrieval debug panel only, with clear Phase 03 chat/generation notice.
- Tests cover device selection, fake embeddings, FAISS, BM25, fusion, citations, selected-doc filtering, and rebuild/stale behavior.
- Validation commands pass.

## 17. Risks and Known Limitations

- `sentence-transformers` may download the embedding model on first real use; automated tests should avoid this.
- CUDA availability depends on the user's PyTorch install, drivers, and hardware. Phase 02 should not promise GPU execution beyond detecting and using it when available.
- `faiss-cpu` can have platform-specific install issues on some Python/Windows combinations.
- Rebuilding FAISS after document changes re-embeds all workspace chunks in Phase 02; this is simpler but slower for large workspaces.
- No separate embedding vector cache is proposed in Phase 02, so rebuilds are less efficient.
- BM25 tokenization is English-first and does not stem or handle multilingual text well.
- Score normalization is query-local and not comparable across queries.
- Python-side filtering after FAISS over-retrieval can return fewer than `top_k` results if selected documents are sparse.
- Runtime index files are not transactional with SQLite changes; stale detection mitigates but does not eliminate DB/filesystem drift.
- Retrieval traces may contain document text and should not be logged automatically.

## 18. Questions for User Review

Blocking before implementation:

- None. The requested Phase 02 boundaries are clear.

Non-blocking review decisions:

1. Approve `IndexFlatIP` with normalized embeddings for cosine-like dense retrieval?
2. Approve no separate embedding vector cache in Phase 02, accepting full re-embedding on rebuild?
3. Approve explicit `EMBEDDING_DEVICE=cuda` raising an actionable error when CUDA is unavailable, while `auto` falls back to CPU?
4. Approve manual/stale-prompted index rebuild rather than automatic rebuild immediately after every upload/delete?
5. Approve BM25 in-memory rebuild only, with no persisted BM25 artifact in Phase 02?

## 19. Implementation Sequence for the Next Run

Recommended implementation order after user approval:

1. Add `sentence-transformers`, `faiss-cpu`, and `rank-bm25` to `pyproject.toml`; run `uv sync`.
2. Add retrieval dataclasses in `retrieval/models.py`.
3. Add path helpers for workspace FAISS index and metadata paths.
4. Add repository chunk read/count helpers and tests.
5. Implement citation formatter and tests.
6. Implement device resolver and fake-model-friendly embedding wrapper tests.
7. Implement FAISS store build/save/load/search and metadata/stale checks.
8. Implement BM25 store and tokenizer.
9. Implement hybrid score normalization and fusion.
10. Implement retrieval service orchestration with selected-document validation.
11. Update Streamlit with a minimal retrieval debug panel.
12. Update README with Phase 02 usage and non-scope.
13. Run unit tests, lint, format check, app smoke, and Git ignore verification.
14. Stop before Phase 03.

## 20. Reviewer Checklist

- [ ] The plan implements local retrieval only over Phase 01 chunks.
- [ ] The plan does not introduce OpenAI calls, answer generation, chat sessions, summaries, eval UI, MLflow, API key persistence, LangChain, or LlamaIndex.
- [ ] Proposed dependencies are limited to `sentence-transformers`, `faiss-cpu`, and `rank-bm25`, with direct `torch` deferred unless proven necessary.
- [ ] Embedding device behavior is explicit for `auto`, `cpu`, and `cuda`.
- [ ] FAISS index location and metadata mapping are explicit.
- [ ] BM25 rebuild behavior is explicit and does not require persistence.
- [ ] Hybrid score normalization and weighted fusion are specified.
- [ ] Citation formatting for PDF and Markdown is independently testable.
- [ ] Retrieval trace metadata is specified without automatic logging of document text.
- [ ] SQLite schema remains unchanged unless implementation uncovers a concrete need.
- [ ] Streamlit changes are limited to a retrieval debug panel, not chat or QA.
- [ ] Tests avoid real embedding downloads and external services.
- [ ] Runtime FAISS files remain Git-ignored under `storage/`.
- [ ] Remaining review decisions are listed in "Questions for User Review."
- [ ] Implementation is gated on explicit user approval.
