# Phase 03 Implementation Review Digest

## 1. Executive Summary

Phase 03 implements grounded QA and workspace chat over the existing Phase 02 retrieval
system. The implementation adds an OpenAI Responses API wrapper, prompt builders,
source mapping, chat session/message persistence, QA orchestration, Streamlit chat UI,
and mocked tests.

This phase does not implement summaries, evaluation UI, MLflow, Docker, LangChain,
LlamaIndex, reranking, saved API keys, keyring integration, streaming responses, OCR,
or retrieval/chunking redesign.

## 2. Files Changed

- `README.md`: Updated project status, Phase 03 scope, run notes, and API-key safety notes.
- `pyproject.toml`: Added `openai` as the only new direct runtime dependency.
- `uv.lock`: Updated lockfile for `openai` and its transitive dependencies.
- `src/mini_notebooklm_rag/storage/sqlite.py`: Added idempotent chat tables and indexes.
- `src/mini_notebooklm_rag/streamlit_app.py`: Added Phase 03 chat UI while preserving
  ingestion and retrieval debug panels.
- `src/mini_notebooklm_rag/llm/__init__.py`: LLM package marker.
- `src/mini_notebooklm_rag/llm/models.py`: App-level LLM response, token usage, and error models.
- `src/mini_notebooklm_rag/llm/openai_client.py`: Non-streaming OpenAI Responses API wrapper.
- `src/mini_notebooklm_rag/qa/__init__.py`: QA package marker.
- `src/mini_notebooklm_rag/qa/prompts.py`: Grounded QA, outside-knowledge QA, and rewrite prompt builders.
- `src/mini_notebooklm_rag/qa/source_mapping.py`: Retrieval-result to `[S#]` source mapping helpers.
- `src/mini_notebooklm_rag/qa/service.py`: QA orchestration across chat, retrieval, rewrite, and LLM generation.
- `src/mini_notebooklm_rag/chat/__init__.py`: Chat package marker.
- `src/mini_notebooklm_rag/chat/models.py`: Chat persistence data models.
- `src/mini_notebooklm_rag/chat/repositories.py`: SQLite repository for chat sessions and messages.
- `src/mini_notebooklm_rag/chat/service.py`: Service wrapper for chat lifecycle behavior.
- `tests/test_storage_sqlite.py`: Verifies chat tables are included in database initialization.
- `tests/test_chat_repository.py`: Chat persistence lifecycle and cascade behavior tests.
- `tests/test_chat_service.py`: Chat title derivation behavior test.
- `tests/test_openai_client.py`: Mocked OpenAI wrapper tests.
- `tests/test_prompt_construction.py`: Prompt template behavior tests.
- `tests/test_qa_service.py`: Grounded QA, outside knowledge, shortcut, and warning behavior tests.
- `tests/test_query_rewrite.py`: Query rewrite, clarify, no-history, disabled, and fallback tests.
- `tests/test_source_mapping.py`: Source ID mapping and unknown marker tests.

## 3. Module-by-Module Implementation Summary

### `src/mini_notebooklm_rag/llm/models.py`

Main classes:

- `OpenAIClientError`
- `TokenUsage`
- `LLMResponse`

Responsibility: Defines normalized LLM-layer return and error types.

Does not: Call OpenAI, persist responses, or know about retrieval/chat.

### `src/mini_notebooklm_rag/llm/openai_client.py`

Main class/functions:

- `OpenAIClient.generate(...)`
- `_extract_usage(...)`
- `_extract_output_text(...)`

Responsibility: Wraps `client.responses.create(...)` from the official OpenAI SDK,
normalizes text and token usage, and converts SDK exceptions into app-level errors.

Does not: Stream responses, log prompts/API keys, persist API keys, or implement QA logic.

### `src/mini_notebooklm_rag/qa/prompts.py`

Main classes/functions:

- `PromptBundle`
- `build_grounded_qa_prompt(...)`
- `build_outside_knowledge_prompt(...)`
- `build_query_rewrite_prompt(...)`
- `NOT_FOUND_MESSAGE`

Responsibility: Builds explicit prompts and compact prompt metadata.

Does not: Call OpenAI, retrieve chunks, parse model answers, or persist chat.

### `src/mini_notebooklm_rag/qa/source_mapping.py`

Main classes/functions:

- `SourceReference`
- `build_source_references(...)`
- `compact_source_map(...)`
- `find_unknown_source_markers(...)`
- `has_source_marker(...)`

Responsibility: Assigns `[S1]`, `[S2]`, etc. from retrieval result order and creates
compact source metadata for persistence.

Does not: Invent citations from model text, repair citation mistakes, or persist full chunk text.

### `src/mini_notebooklm_rag/qa/service.py`

Main classes/functions:

- `QAService.answer_question(...)`
- `RewriteResult`
- `QAResult`
- `LLMClientProtocol`
- `QAServiceError`

Responsibility: Coordinates user-message persistence, preflight checks, query rewrite,
retrieval, prompt construction, OpenAI generation, source warning checks, and assistant-message
persistence.

Does not: Rebuild indexes automatically, call OpenAI when selected documents/index state is
invalid, persist full chunk text, or implement summaries/evaluation.

### `src/mini_notebooklm_rag/chat/models.py`

Main classes:

- `ChatSession`
- `ChatMessage`
- `NewChatMessage`

Responsibility: Typed records for chat sessions and persisted messages.

Does not: Touch SQLite directly or perform UI logic.

### `src/mini_notebooklm_rag/chat/repositories.py`

Main class/functions:

- `ChatRepository.create_session(...)`
- `ChatRepository.list_sessions(...)`
- `ChatRepository.get_session(...)`
- `ChatRepository.update_session_documents(...)`
- `ChatRepository.update_session_title(...)`
- `ChatRepository.delete_session(...)`
- `ChatRepository.add_message(...)`
- `ChatRepository.list_messages(...)`

Responsibility: Performs SQLite CRUD for chat sessions/messages and JSON serialization for
selected documents and compact metadata.

Does not: Delete historical chat on document deletion, store API keys, or store full retrieval traces.

### `src/mini_notebooklm_rag/chat/service.py`

Main class/functions:

- `ChatService`
- `ChatService.maybe_title_from_question(...)`

Responsibility: Initializes storage/database and provides a small service boundary over
`ChatRepository`, including safe first-question title generation.

Does not: Call OpenAI, retrieve chunks, or render UI.

### `src/mini_notebooklm_rag/storage/sqlite.py`

Main function:

- `initialize_database(...)`

Responsibility: Adds idempotent `chat_sessions` and `chat_messages` tables alongside existing
workspace/document/chunk tables.

Does not: Add a migration framework or a `retrieval_events` table.

### `src/mini_notebooklm_rag/streamlit_app.py`

Main additions:

- `_render_chat_panel(...)`
- `_render_qa_debug(...)`

Responsibility: Renders the chat panel with temporary API-key input, model fields, selected
documents, chat history, outside-knowledge toggle, rewrite toggle, answer display, source list,
and current-response debug/source chunks.

Does not: Persist temporary API keys, request saved keys, call OpenAI directly, or implement
summary/eval controls.

## 4. Public/Internal API Summary

### OpenAI Client

`OpenAIClient(api_key: str, default_model: str, client: Any | None = None)`

- Inputs: API key, default model, optional fake SDK client.
- Output: client wrapper instance.
- Failure: raises `OpenAIClientError` if key is empty or SDK import fails.
- Touches: OpenAI SDK only when no injected client is supplied.

`OpenAIClient.generate(instructions: str, input_text: str, model: str | None = None, max_output_tokens: int | None = None) -> LLMResponse`

- Inputs: prompt instructions, prompt input, optional model, optional max output tokens.
- Output: `LLMResponse`.
- Failure: wraps SDK exceptions as `OpenAIClientError("OpenAI request failed.")`.
- Touches: OpenAI API through injected or real SDK client. Does not touch filesystem/SQLite.

### Chat Repository/Service

`ChatService.create_session(workspace_id: str, selected_document_ids: list[str], title: str = "New chat") -> ChatSession`

- Inputs: workspace ID, selected document IDs, title.
- Output: `ChatSession`.
- Failure: SQLite integrity errors propagate.
- Touches: SQLite.

`ChatService.list_sessions(workspace_id: str) -> list[ChatSession]`

- Inputs: workspace ID.
- Output: sessions sorted by update time descending.
- Failure: SQLite errors propagate.
- Touches: SQLite.

`ChatService.add_message(message: NewChatMessage) -> ChatMessage`

- Inputs: typed message with compact metadata.
- Output: persisted `ChatMessage`.
- Failure: SQLite integrity errors propagate.
- Touches: SQLite.

`ChatService.list_messages(session_id: str, limit: int | None = None) -> list[ChatMessage]`

- Inputs: session ID and optional limit.
- Output: messages sorted ascending by creation time.
- Failure: SQLite errors propagate.
- Touches: SQLite.

### QA Service

`QAService.answer_question(workspace_id: str, session_id: str, question: str, selected_document_ids: list[str], api_key: str, model: str | None = None, rewrite_model: str | None = None, allow_outside_knowledge: bool | None = None, enable_query_rewrite: bool | None = None, top_k: int | None = None, dense_weight: float | None = None, sparse_weight: float | None = None) -> QAResult`

- Inputs: workspace/session IDs, user question, selected document IDs, API key, model/config overrides.
- Output: `QAResult` containing persisted user/assistant messages, answer, current full sources,
  compact metadata, token usage, and warnings.
- Failure: raises `QAServiceError` for empty question or invalid session. OpenAI errors propagate as
  app-level LLM errors. Retrieval errors propagate from Phase 02 service.
- Touches: SQLite through `ChatService`, retrieval/index files through Phase 02 `RetrievalService`,
  OpenAI API only after preflight passes and an API key exists.

## 5. Dependency Changes

Direct runtime dependency added:

- `openai`

No other direct runtime dependencies were added. No `mlflow`, LangChain, LlamaIndex, keyring,
secret manager, streaming framework, tokenizer, reranker, or evaluation dependency was added.

`uv.lock` includes transitive dependencies required by `openai`.

## 6. Final QA/Chat Architecture

The Phase 03 flow is:

1. Streamlit collects workspace, chat session, selected document IDs, model settings, toggles,
   user question, and temporary API key.
2. `QAService` persists the user message.
3. `QAService` checks selected documents, chunk availability, and FAISS index status.
4. Query rewrite runs only when enabled, history exists, the query appears follow-up-like,
   preflight passes, and an API key exists.
5. Phase 02 `RetrievalService` retrieves chunks from the selected documents.
6. Source IDs are assigned from retrieval order.
7. A grounded or outside-knowledge prompt is built.
8. `OpenAIClient` calls the non-streaming Responses API.
9. The assistant message is persisted with compact metadata and token usage.
10. Full source chunks remain only in the current `QAResult` in Streamlit session state.

## 7. OpenAI Client Behavior

- Uses `from openai import OpenAI` lazily inside `_build_client`.
- Calls `client.responses.create(...)`.
- Uses non-streaming responses only.
- Extracts `response.output_text` when available.
- Falls back to walking `response.output[].content[].text`.
- Extracts `input_tokens`, `output_tokens`, and `total_tokens` when usage metadata exists.
- Computes total tokens if input and output tokens are present but total is missing.
- Normalizes SDK failures to `OpenAIClientError("OpenAI request failed.")`.
- Never logs, prints, returns, or persists the API key.
- Supports mocked SDK/client injection for tests.

## 8. Prompt Construction Behavior

Grounded prompt:

- Tells the model to answer only from provided sources.
- Requires factual claims to cite source IDs like `[S1]`.
- Requires the exact not-found response when unsupported.
- Forbids outside knowledge.

Outside-knowledge prompt:

- Requires exactly separated sections:
  - `From your documents:`
  - `Outside the selected documents:`
- Requires document-grounded claims to cite `[S#]`.
- Forbids source IDs on outside-knowledge-only content.

Query rewrite prompt:

- Uses only current session history.
- Requests JSON only.
- Supports `{"action":"rewrite","query":"..."}`.
- Supports `{"action":"clarify","question":"..."}`.

Source text is capped per source block with a simple character budget. Prompt metadata includes
prompt type, source count, source character count, and source IDs, but no API key.

## 9. Grounded QA Behavior

- Grounded-only is the default through `ALLOW_OUTSIDE_KNOWLEDGE=false`.
- If no documents are selected, the assistant returns an actionable message and OpenAI is not called.
- If more than 3 documents are selected, the assistant returns an actionable message and OpenAI is not called.
- If selected documents are missing or have no chunks, OpenAI is not called.
- If the index is missing, stale, or empty, OpenAI is not called.
- If retrieval returns no chunks and outside knowledge is disabled, the exact response is persisted:
  `I could not find this information in the selected documents.`
- Unknown `[S#]` markers produce a dev warning.
- A grounded answer without any source marker produces a dev warning unless it is the exact not-found answer.
- The service does not silently invent or repair citations.

## 10. Outside-Knowledge Behavior

- Controlled by Streamlit toggle/config.
- Still requires valid selected documents and current retrieval index before calling OpenAI.
- Prompt requires document and outside-document sections.
- Outside-document content must be labeled and must not use source IDs unless directly supported by retrieved sources.
- If selected documents or index state are invalid, the service returns an actionable message instead of making an outside-knowledge call.

## 11. Query Rewrite Behavior

- Uses only messages from the current chat session.
- Skips when disabled.
- Skips when there is no prior current-session history.
- Skips when the query appears standalone by a simple heuristic.
- Does not run when preflight fails or no API key is available.
- JSON parse failure falls back to the original query and records a warning.
- `clarify` action persists an assistant clarification message without retrieval or answer generation.
- The dev/debug state exposes original query, rewritten query, skipped reason, and clarification.

Known MVP limitation: standalone/follow-up detection is heuristic and intentionally conservative.

## 12. Chat SQLite Schema and Persistence Behavior

### `chat_sessions`

Columns:

- `id TEXT PRIMARY KEY`
- `workspace_id TEXT NOT NULL`
- `title TEXT NOT NULL`
- `selected_document_ids TEXT NOT NULL`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

Foreign keys:

- `workspace_id REFERENCES workspaces(id) ON DELETE CASCADE`

Indexes:

- `idx_chat_sessions_workspace_updated ON chat_sessions(workspace_id, updated_at)`

Behavior:

- Workspace deletion cascades to sessions.
- Session selected document IDs are JSON text.
- First user question can replace the default title.

### `chat_messages`

Columns:

- `id TEXT PRIMARY KEY`
- `workspace_id TEXT NOT NULL`
- `session_id TEXT NOT NULL`
- `role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system'))`
- `content TEXT NOT NULL`
- `selected_document_ids TEXT`
- `original_query TEXT`
- `rewritten_query TEXT`
- `answer_mode TEXT`
- `source_map TEXT`
- `retrieval_metadata TEXT`
- `prompt_metadata TEXT`
- `model_name TEXT`
- `input_tokens INTEGER`
- `output_tokens INTEGER`
- `total_tokens INTEGER`
- `created_at TEXT NOT NULL`

Foreign keys:

- `workspace_id REFERENCES workspaces(id) ON DELETE CASCADE`
- `session_id REFERENCES chat_sessions(id) ON DELETE CASCADE`

Indexes:

- `idx_chat_messages_session_created ON chat_messages(session_id, created_at)`
- `idx_chat_messages_workspace_created ON chat_messages(workspace_id, created_at)`

Behavior:

- Workspace deletion cascades to messages.
- Session deletion cascades to messages.
- Document deletion does not delete historical messages.
- Source map, retrieval metadata, and prompt metadata are stored as compact JSON.
- Full chunk text is not persisted by default.

## 13. Source/Citation Mapping Behavior

- Retrieval results are mapped in order: first result is `S1`, second is `S2`, etc.
- Source list is generated from the source map, not inferred from LLM text.
- Compact persisted source map includes source ID, chunk ID, document ID, filename, citation,
  source type, page range/heading, and scores.
- Full chunk text remains in `SourceReference` for the current UI response only.
- Unknown answer markers such as `[S99]` are detected and surfaced as warnings.

## 14. Retrieval Integration Behavior

- Uses existing Phase 02 `RetrievalService`.
- Does not reimplement dense, sparse, hybrid, citation, or FAISS logic.
- Does not silently rebuild indexes during chat.
- Preflight calls the existing index status path before any OpenAI call.
- Retrieval debug rebuild button remains available in Streamlit.
- Missing/stale/empty index state prevents unnecessary OpenAI calls.

## 15. Streamlit UI Behavior

Phase 03 Streamlit UI now includes:

- Existing workspace selector/create/delete.
- Existing PDF/Markdown ingestion and document delete.
- Existing retrieval debug panel and rebuild button.
- Chat panel with:
  - temporary OpenAI API key password input
  - generation model input
  - query rewrite model input
  - outside-knowledge toggle
  - query rewrite toggle
  - up-to-3 document multiselect
  - new chat button
  - chat history selector
  - delete selected chat button
  - chat input
  - answer display through Streamlit chat messages
  - persisted compact source list under assistant messages
  - latest-response debug expander with full source chunks and metadata

The UI does not show summary/eval controls and does not persist temporary API keys.

## 16. Token Usage Handling

- `OpenAIClient` extracts usage metadata when returned by the SDK.
- Assistant messages store `input_tokens`, `output_tokens`, and `total_tokens` when available.
- The UI dev/debug expander displays token usage for the latest response.
- The implementation does not estimate dollar cost.

## 17. Security and Secret Handling

- No real secrets were added.
- API keys are accepted only from `.env` settings or temporary Streamlit session input.
- Temporary UI keys use `st.text_input(..., type="password")` and `st.session_state`.
- API keys are not stored in SQLite, `.local/`, `storage/`, logs, or docs.
- `.env`, `.env.local`, `.local/`, and `storage/app.db` are Git-ignored.
- Prompt/retrieval metadata excludes API keys.
- Full document text is not automatically logged or persisted in chat metadata.
- Full source chunks are visible only in the current UI/dev response state.

## 18. Test Coverage Matrix

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_openai_client.py` | OpenAI wrapper with fake client | Responses API args, output extraction, usage extraction, sanitized errors | Real OpenAI API |
| `tests/test_prompt_construction.py` | Prompt builders | Grounded not-found instruction, source IDs, outside-knowledge sections, rewrite JSON contract | Prompt quality with real model |
| `tests/test_source_mapping.py` | Source reference mapping | Stable `[S#]`, compact map excludes text, unknown marker detection | Large source-list behavior |
| `tests/test_chat_repository.py` | Chat SQLite persistence | Session/message CRUD, cascade workspace delete, document delete preserves chat | Concurrent writes |
| `tests/test_chat_service.py` | Chat service helpers | First-question title truncation | Complex title generation |
| `tests/test_qa_service.py` | QA orchestration | No OpenAI on stale index/empty retrieval, grounded answer persistence, outside prompt, warnings | Real retrieval plus real model |
| `tests/test_query_rewrite.py` | Rewrite flow | Disabled/no-history skips, current-session-only history, clarify action, bad JSON fallback | Real model rewrite quality |
| `tests/test_storage_sqlite.py` | Database initialization | Chat tables are created idempotently | Schema migration from older deployed DB |

## 19. Validation Results

Commands run:

- `uv sync`: PASS. Installed/locked `openai==2.33.0` and required transitive packages.
- `uv run pytest`: PASS. `71 passed, 3 warnings in 2.85s`.
- `uv run ruff check .`: PASS. `All checks passed!`
- `uv run ruff format --check .`: PASS. `61 files already formatted`.
- `uv run app`: STARTUP PASS with bounded timeout. Streamlit printed local URL
  `http://localhost:8501`; command was intentionally stopped by the 20-second tool timeout.
- `git status --short`: PASS. Shows expected Phase 03 modified/untracked files only.
- `git check-ignore -v .env .env.local .local/secrets.local.json storage/app.db`: PASS.
  All paths are ignored by `.gitignore`.

Manual app smoke:

- Bounded app startup was performed successfully on Windows.
- Full browser walkthrough with a real OpenAI API key was not performed in this run. The OpenAI
  and QA flows are covered by mocked automated tests, and the final real-key UI path still needs
  reviewer/user smoke validation in a local browser.

Notes:

- PowerShell emitted the existing local profile warning:
  `profile.ps1 cannot be loaded because running scripts is disabled`.
- Pytest emitted existing SWIG/PyMuPDF deprecation warnings.

## 20. Known Risks and Limitations

- Real OpenAI API behavior was mocked in tests; a reviewer should run one real-key smoke test.
- Query rewrite standalone/follow-up detection is a simple heuristic and may skip useful rewrites.
- Model citation compliance is enforced through prompts plus warnings, not through automatic answer repair.
- OpenAI failures are intentionally sanitized, which protects secrets but may require dev inspection for details.
- Chat message and retrieval metadata persistence is compact by design; full historical chunk text is not stored.
- SQLite schema uses idempotent table creation, not migration files; future schema evolution will need care.
- Streamlit chat UI is functional MVP UI, not browser-automation tested.
- Index state must be rebuilt manually through the retrieval debug panel after ingestion/deletion.

## 21. Reviewer Checklist

- Confirm `openai` is the only new direct runtime dependency.
- Confirm no API key persistence was added to SQLite, `.local/`, or files.
- Confirm chat tables are idempotent and cascade only workspace/session deletion.
- Confirm document deletion does not delete historical chat messages.
- Confirm full chunk text is not persisted in chat metadata by default.
- Confirm query rewrite uses only current-session history.
- Confirm missing/stale indexes prevent OpenAI calls.
- Confirm grounded not-found shortcut avoids unnecessary OpenAI calls.
- Confirm outside-knowledge prompt requires separated sections.
- Confirm Streamlit still preserves Phase 01 ingestion and Phase 02 retrieval debug behavior.
- Run a real-key local browser smoke test before committing or advancing to Phase 04.

## 22. Next Recommended Step

Phase 03 appears ready for user/reviewer validation. Do not start Phase 04 until the reviewer
has checked the digest, run a real-key UI smoke test if desired, and explicitly approves the next
phase.
