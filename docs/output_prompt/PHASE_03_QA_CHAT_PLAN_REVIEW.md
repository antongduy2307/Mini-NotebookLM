# Phase 03 QA and Chat Plan Review

Status: planning draft for user and external reviewer approval.

This document plans Phase 03 only. It must not be treated as implementation approval. Phase 03 should preserve the completed Phase 00 app shell, Phase 01 ingestion/storage foundation, and Phase 02 local retrieval behavior.

## 1. Phase Objective

Implement grounded question answering and workspace chat over the Phase 02 retrieval system.

Phase 03 should add:

- OpenAI API client integration.
- Configurable generation and query rewrite models.
- Grounded-only answer generation by default.
- Inline source markers like `[S1]`, `[S2]`.
- Source list below each answer.
- Optional outside-knowledge mode with explicit answer separation.
- Workspace chat sessions and message history.
- Query rewriting using only the current chat session.
- Ambiguity handling that asks a clarifying question instead of hallucinating.
- Streamlit chat UI.
- Dev/debug panel showing original query, rewritten query, selected documents, retrieval trace, prompt metadata, source mapping, and token usage when available.
- Tests using mocked OpenAI clients only.

Phase 03 should make the app usable as a local-first NotebookLM-like grounded QA demo. It should not add summaries, evaluation, MLflow, Docker, deployment, LangChain, or LlamaIndex.

## 2. Scope and Non-Scope

### In Scope

- Add the `openai` dependency.
- Add an internal OpenAI client wrapper.
- Add prompt construction modules for grounded QA, outside-knowledge QA, query rewrite, clarification, and not-found refusal.
- Add source mapping from retrieved chunks to `[S1]`, `[S2]`, etc.
- Add SQLite persistence for chat sessions and chat messages.
- Optionally add retrieval event persistence if it is lightweight and useful for chat history/dev review.
- Add chat service orchestration around Phase 02 `RetrievalService`.
- Add Streamlit chat UI integrated with existing workspace/document/retrieval UI.
- Add temporary UI API key support for the current Streamlit session.
- Add tests with mocked OpenAI responses.

### Non-Scope

Phase 03 must not implement:

- document summaries
- summary cache
- evaluation UI
- MLflow
- Docker/deployment
- LangChain
- LlamaIndex
- reranking
- RRF
- OCR
- embedding redesign
- chunking redesign
- tokenizer-dependent chunking changes
- saved local API key manager
- API key persistence beyond `.env` and temporary UI input
- secret/keyring libraries
- web frameworks beyond the current Streamlit app

## 3. Dependencies to Add and Why

Add one runtime dependency:

- `openai`: official OpenAI Python SDK for Responses API calls.

No other dependency is proposed.

Dependencies explicitly not added:

- `mlflow`
- LangChain
- LlamaIndex
- eval or judge frameworks
- secret/keyring libraries
- additional web frameworks
- tokenizer packages

Decision: use the OpenAI Responses API through the official `openai` SDK.

Reason: OpenAI documentation describes Responses as the recommended primary API for new model integrations, and the official Python SDK exposes a convenient `client.responses.create(...)` interface. The wrapper should hide SDK details from the rest of the app.

Status: proposed for Phase 03, requires user approval.

References checked:

- OpenAI Responses API reference: <https://platform.openai.com/docs/api-reference/responses/create?api-mode=responses>
- OpenAI Python SDK repository: <https://github.com/openai/openai-python>
- OpenAI Responses migration guide: <https://platform.openai.com/docs/guides/responses-vs-chat-completions>

## 4. Proposed File and Module Changes

Planned files to add:

```text
src/mini_notebooklm_rag/llm/__init__.py
src/mini_notebooklm_rag/llm/openai_client.py
src/mini_notebooklm_rag/llm/models.py
src/mini_notebooklm_rag/qa/__init__.py
src/mini_notebooklm_rag/qa/prompts.py
src/mini_notebooklm_rag/qa/source_mapping.py
src/mini_notebooklm_rag/qa/service.py
src/mini_notebooklm_rag/chat/__init__.py
src/mini_notebooklm_rag/chat/models.py
src/mini_notebooklm_rag/chat/repositories.py
src/mini_notebooklm_rag/chat/service.py
tests/test_openai_client.py
tests/test_prompt_construction.py
tests/test_source_mapping.py
tests/test_qa_service.py
tests/test_query_rewrite.py
tests/test_chat_repository.py
tests/test_chat_service.py
```

Planned files to modify:

```text
pyproject.toml
uv.lock
README.md
src/mini_notebooklm_rag/config.py
src/mini_notebooklm_rag/storage/sqlite.py
src/mini_notebooklm_rag/streamlit_app.py
tests/test_scaffold.py
tests/test_storage_sqlite.py
```

Expected responsibilities:

- `llm/openai_client.py`: small wrapper around the OpenAI SDK; accepts API key from settings or temporary UI input; returns normalized response objects.
- `llm/models.py`: typed request/response records, token usage records, and LLM error types.
- `qa/prompts.py`: prompt templates and prompt assembly.
- `qa/source_mapping.py`: assign `[S1]` source IDs to retrieval results and build source list records.
- `qa/service.py`: orchestration for query rewrite, retrieval, answer generation, not-found/refusal, outside-knowledge mode, and dev metadata.
- `chat/models.py`: chat dataclasses.
- `chat/repositories.py`: SQLite persistence for chat sessions/messages.
- `chat/service.py`: workspace chat session lifecycle and message persistence.
- `streamlit_app.py`: add chat UI while preserving ingestion and retrieval debug behavior.

No `summary/`, `evaluation/`, `mlflow/`, Docker, LangChain, or LlamaIndex modules should be added.

## 5. OpenAI Client Design

Configuration inputs already exist:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-nano
OPENAI_QUERY_REWRITE_MODEL=gpt-4.1-nano
ALLOW_OUTSIDE_KNOWLEDGE=false
ENABLE_QUERY_REWRITE=true
```

API key sources for Phase 03:

1. `.env` via `OPENAI_API_KEY`
2. temporary Streamlit UI input stored only in `st.session_state`

No Phase 03 key persistence:

- Do not write API keys to SQLite.
- Do not write API keys to `.local/`.
- Do not implement owner-name saved key manager yet.
- Do not print/log raw or masked API keys.

Client wrapper near-signatures:

```python
@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw_finish_reason: str | None


class OpenAIClient:
    def __init__(self, api_key: str, default_model: str): ...
    def generate(
        self,
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...
```

Implementation approach:

- Use `from openai import OpenAI`.
- Construct `OpenAI(api_key=api_key)` only inside the wrapper.
- Call `client.responses.create(...)`.
- Use `response.output_text` when available.
- Extract token usage if present.
- Normalize SDK exceptions into app-level `OpenAIClientError`.
- Keep streaming out of Phase 03 unless explicitly approved.

Reason to defer streaming:

- Non-streaming responses are simpler to test, easier to map to persistence, and easier to validate for citation/source rules.

## 6. Prompt Design

Prompt construction should be explicit and unit-tested.

### Source Insertion

Retrieved chunks should be converted into source blocks before prompting:

```text
[S1]
Citation: paper.pdf, p. 5
Document ID: <document_id>
Chunk ID: <chunk_id>
Content:
<chunk text>

[S2]
Citation: notes.md > Heading
Document ID: <document_id>
Chunk ID: <chunk_id>
Content:
<chunk text>
```

Only the selected Phase 02 retrieval results should be inserted. The prompt should not include entire documents or all workspace history.

### Source ID Assignment

- Assign source IDs in final retrieval rank order.
- Use stable labels `[S1]`, `[S2]`, etc. within one answer.
- Each source ID maps to one retrieved chunk and its citation metadata.
- The source list shown below the answer should be generated from this map, not inferred from model text.

### Grounded QA Instructions

Core instructions:

```text
You answer questions using only the provided sources.
Every factual claim based on the sources must cite one or more source IDs like [S1].
Do not cite sources that do not support the claim.
If the answer is not supported by the provided sources, respond:
I could not find this information in the selected documents.
Do not use outside knowledge.
```

### Outside-Knowledge Instructions

Core instructions:

```text
Use the provided sources first.
Separate the answer into exactly these sections:
From your documents:
Outside the selected documents:
Document-grounded claims must cite source IDs like [S1].
Outside-knowledge content must be clearly labeled and must not use source IDs.
If the documents do not support part of the answer, say so before adding outside knowledge.
```

### Query Rewrite Instructions

The rewrite prompt should receive only:

- current chat session recent messages
- latest user question
- selected document display names/IDs for context

It should output one of:

```json
{"action":"rewrite","query":"standalone retrieval query"}
{"action":"clarify","question":"short clarification question"}
```

The JSON shape can be requested in prompt text without adding a structured-output dependency. If the OpenAI SDK supports JSON schema cleanly with the chosen model, implementation may use it, but this is optional and should be tested with mocks.

### Refusal/Not-Found Response

Application-side direct response when grounded-only retrieval has no useful context:

```text
I could not find this information in the selected documents.
```

Reason: avoid an unnecessary OpenAI call when there is no retrieved context and outside knowledge is disabled.

### Context Length Control

Phase 03 should avoid adding tokenizer dependencies. Use approximate context budgeting:

- start with top retrieval results from Phase 02
- cap source count, for example `top_k` from retrieval defaults or UI
- cap chunk text characters per source for prompt insertion if needed
- cap recent chat history for rewrite, for example last 6 messages
- expose prompt metadata in dev panel:
  - source count
  - approximate source characters
  - history message count
  - model name
  - query rewrite enabled/disabled

Do not expose API keys in prompt metadata.

## 7. Grounded QA Behavior

Default behavior:

- `ALLOW_OUTSIDE_KNOWLEDGE=false`
- Use retrieved chunks only.
- Every document-grounded claim should cite source IDs.
- If context does not support an answer, respond with the required not-found message.
- Do not invent citations.
- Do not answer from general model knowledge.

Pragmatic MVP flow:

1. Validate selected workspace, chat session, and selected documents.
2. Check selected document count is between 1 and 3.
3. Check workspace index status through `RetrievalService.index_status`.
4. If index is missing/stale/empty, show actionable message and do not call OpenAI.
5. Rewrite query only if enabled and useful.
6. Run Phase 02 retrieval.
7. If retrieval results are empty and outside knowledge is disabled, return not-found directly.
8. Build source map.
9. Build grounded QA prompt.
10. Call OpenAI.
11. Persist user message and assistant answer.
12. Render answer, sources, and dev metadata.

Potential answer validation:

- Phase 03 should not implement a second LLM judge.
- It can perform light deterministic checks:
  - If answer contains source markers not in source map, flag in dev warnings.
  - If answer has no `[S#]` markers and is not the not-found response, flag in dev warnings.
  - Do not silently fabricate or repair unsupported citations.

## 8. Outside-Knowledge Behavior

Outside knowledge can be enabled by:

- `ALLOW_OUTSIDE_KNOWLEDGE=true` in config, or
- a Streamlit toggle for the current chat request

Rules:

- The prompt must require two sections:
  - `From your documents:`
  - `Outside the selected documents:`
- Document-grounded content must cite sources.
- Outside-knowledge content must not cite document sources unless directly supported.
- If the selected documents contain no relevant context, the first section should say the information was not found in the selected documents.
- The UI must clearly show outside knowledge mode is enabled.

Avoid unnecessary calls:

- If outside knowledge is disabled and retrieval context is empty, return not-found directly.
- If outside knowledge is enabled, OpenAI may be called even with empty retrieval context, but the prompt must state that no relevant document context was found.

## 9. Query Rewrite Design

Configuration:

- `ENABLE_QUERY_REWRITE=true/false`
- `OPENAI_QUERY_REWRITE_MODEL`

Inputs:

- only the current chat session history
- latest user question
- selected document IDs/display names

Do not use:

- all workspace history
- messages from other chat sessions
- document text beyond selected document names in the rewrite prompt

When to skip rewrite:

- rewrite disabled
- no prior messages in current session
- user question is already standalone enough by a simple heuristic
- no API key available
- selected documents missing or invalid
- index missing/stale, because retrieval will not run

Pragmatic standalone heuristic:

- If no previous user/assistant messages exist, skip rewrite.
- If the query has no pronouns or follow-up markers such as `it`, `they`, `that`, `this`, `those`, `he`, `she`, `above`, `earlier`, `same`, `compare`, and is longer than a short threshold, skip rewrite.

Ambiguity handling:

- Query rewrite can return `{"action":"clarify","question":"..."}`.
- The app should persist/display the clarifying assistant question without running retrieval or answer generation.
- MVP ambiguity detection is prompt-driven plus simple heuristics:
  - very short query like `What about it?`
  - unresolved pronouns with no useful session history
  - selected documents empty or too broad for a referential question

Dev panel must show:

- original query
- rewritten query, if any
- rewrite model
- rewrite skipped reason or clarification question

## 10. Chat Session Persistence and SQLite Schema Changes

Phase 03 should extend the idempotent SQLite initialization used in `storage/sqlite.py`. No migration framework is required yet, but schema additions must be documented and tested.

### `chat_sessions`

```sql
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title TEXT NOT NULL,
    selected_document_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_workspace_updated
ON chat_sessions(workspace_id, updated_at);
```

Notes:

- `selected_document_ids` stores JSON text.
- Store IDs used for the session so history is understandable even if UI selection changes later.
- Title can be generated from the first user question, truncated to a safe length.

### `chat_messages`

```sql
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    selected_document_ids TEXT,
    original_query TEXT,
    rewritten_query TEXT,
    answer_mode TEXT,
    source_map TEXT,
    retrieval_trace TEXT,
    model_name TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
ON chat_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace_created
ON chat_messages(workspace_id, created_at);
```

Notes:

- `source_map` stores JSON text for `[S#]` mappings.
- `retrieval_trace` stores JSON text only if approved in implementation. Because traces can include document text, a safer alternative is storing retrieval metadata without full chunk text. Recommendation: store compact trace metadata in SQLite and render full trace from current retrieval response only in UI.
- `selected_document_ids` is duplicated on messages for auditability.

### Optional `retrieval_events`

Recommended for Phase 03 only if implementation remains small:

```sql
CREATE TABLE IF NOT EXISTS retrieval_events (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    session_id TEXT,
    message_id TEXT,
    original_query TEXT NOT NULL,
    rewritten_query TEXT,
    selected_document_ids TEXT NOT NULL,
    retrieval_trace TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE SET NULL,
    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_events_workspace_created
ON retrieval_events(workspace_id, created_at);
```

Recommendation: keep `retrieval_events` optional and defer if it complicates Phase 03. Chat messages can already store enough metadata for MVP.

Repository/service near-signatures:

```python
class ChatRepository:
    def create_session(workspace_id: str, selected_document_ids: list[str], title: str) -> ChatSession: ...
    def list_sessions(workspace_id: str) -> list[ChatSession]: ...
    def get_session(session_id: str) -> ChatSession | None: ...
    def delete_session(session_id: str) -> None: ...
    def add_message(message: NewChatMessage) -> ChatMessage: ...
    def list_messages(session_id: str, limit: int | None = None) -> list[ChatMessage]: ...
```

Cascade behavior:

- Deleting workspace deletes chat sessions and messages.
- Deleting chat session deletes messages.
- Deleting documents should not delete chat messages; existing messages remain historical, but future retrieval should not select missing documents.

## 11. Retrieval Integration

Phase 03 should use `RetrievalService`; it should not reimplement retrieval.

Pre-call guards:

- selected workspace exists
- selected documents count is 1 to 3
- selected documents belong to workspace
- index status is `current`
- API key exists before any OpenAI call

Index behavior:

- Missing/stale/empty index should show an actionable UI message.
- Do not silently rebuild during chat.
- Keep Phase 02 debug rebuild button available.

Retrieval response handling:

- If retrieval returns no results:
  - grounded-only mode: return not-found without OpenAI call
  - outside-knowledge mode: call OpenAI with empty/explicit no-context source block
- If retrieval returns warnings, show them in dev panel.

## 12. Streamlit UI Changes

Phase 03 should preserve the existing workspace/document/retrieval debug UI and add a chat area.

UI sections:

- Workspace selector.
- Document management panel from Phase 01.
- Retrieval debug panel from Phase 02, possibly collapsed under an expander.
- Chat panel.

Chat panel controls:

- temporary API key input with `type="password"`
- model display/input defaulting to `OPENAI_MODEL`
- document multiselect, max 3
- New Chat button
- chat session selector/history
- Delete Chat button
- outside-knowledge toggle
- query rewrite toggle
- chat input

Chat outputs:

- assistant answer with inline `[S#]` markers
- source list below answer
- expandable source chunks
- dev/debug expander:
  - original query
  - rewritten query
  - rewrite skipped reason or clarification
  - selected documents
  - retrieval trace
  - source mapping
  - generation model
  - query rewrite model
  - token usage if available
  - prompt metadata, but not raw API key

UI behavior:

- If API key is missing, show a clear request for `.env` or temporary session input.
- Do not persist temporary API key.
- Do not call OpenAI when selected documents are missing or index is stale/missing.
- Do not show summary or eval controls in Phase 03.

## 13. Source and Citation Mapping

Source mapping should be deterministic and independent of model output.

Suggested model:

```python
@dataclass(frozen=True)
class SourceReference:
    source_id: str  # "S1"
    chunk_id: str
    document_id: str
    filename: str
    citation: str
    text: str
    source_type: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str] | None
    dense_score: float
    sparse_score: float
    fused_score: float
```

Rules:

- Assign source IDs from final retrieval result order.
- Prompt uses bracketed IDs `[S1]`.
- UI source list displays `[S1] citation`.
- Expanders show chunk text and scores.
- If model returns a citation marker not in source map, flag a warning.
- If model omits citations in grounded answer, flag a warning.
- Do not invent or remap citations after generation.

## 14. Token and Cost Handling

Phase 03 should log token usage metadata when OpenAI returns it:

- input tokens
- output tokens
- total tokens

Where to store:

- `chat_messages.input_tokens`
- `chat_messages.output_tokens`
- `chat_messages.total_tokens`

Cost:

- Do not estimate cost unless implementation can do so simply and reliably.
- Display token usage, not dollar cost, in Phase 03.

Avoid unnecessary calls:

- no OpenAI call if selected documents are missing
- no OpenAI call if index missing/stale
- no answer generation if retrieval context is empty and outside knowledge is disabled
- skip query rewrite when no current-session history exists
- skip query rewrite if rewrite disabled
- skip query rewrite if no API key is available

Context budgeting:

- Use existing retrieval `top_k`.
- Limit chat history for rewriting to recent current-session messages.
- Limit prompt source text by source count and optional character cap.
- Do not add `tiktoken` in Phase 03.

## 15. Security Rules

Required rules:

- Never log API keys.
- Never print API keys.
- Never store API keys in SQLite.
- Never persist temporary UI API key.
- Do not create `.local/secrets.local.json`.
- Do not add keyring or secret manager dependency.
- Do not include full document text in logs by default.
- Retrieval traces may contain document text; render them only in UI/dev panel, not automatic logs.
- Prompt metadata shown in UI must not include API keys.
- Exceptions shown in UI should be sanitized and should not include request headers or secret-bearing config.

Testing expectations:

- Add tests that `Settings` repr excludes API key behavior still holds.
- Add tests for client construction and mocked calls without exposing API key in exceptions or logs.

## 16. Test Plan

Tests must not call the real OpenAI API.

Planned tests:

| Test file | Behavior covered | Important assertions | Not covered yet |
| --- | --- | --- | --- |
| `tests/test_openai_client.py` | OpenAI wrapper with fake SDK/client | model passed through, response text extracted, token usage extracted, no real network | real OpenAI account behavior |
| `tests/test_prompt_construction.py` | prompt templates | grounded prompt includes only sources, outside prompt separates sections, no API key in metadata | real model compliance |
| `tests/test_source_mapping.py` | `[S#]` assignment | deterministic source IDs, citation mapping, unknown marker warning | UI grouping |
| `tests/test_qa_service.py` | QA orchestration | not-found shortcut, no OpenAI call on missing/stale index, source mapping, token usage persistence | real model quality |
| `tests/test_query_rewrite.py` | rewrite behavior | disabled skip, no-history skip, mocked rewrite, mocked clarify action, current-session-only history | advanced ambiguity detection |
| `tests/test_chat_repository.py` | SQLite chat persistence | session create/list/delete, message insert/list, cascade delete, selected doc JSON | migrations |
| `tests/test_chat_service.py` | chat lifecycle | new chat, title generation, selected document metadata, history isolation by session | browser UI |
| `tests/test_storage_sqlite.py` | schema additions | new tables/indexes exist, foreign keys enabled | migration rollback |
| `tests/test_scaffold.py` | settings stability | OpenAI config defaults remain `.env.example` aligned | real secrets |

Mocking strategy:

- Fake OpenAI client object returns deterministic `LLMResponse`.
- Fake retrieval service returns deterministic `RetrievalResponse`.
- Tests assert no OpenAI calls are made for missing/stale index and grounded empty retrieval.
- Tests use `tmp_path` for SQLite and never write real project `storage/`.

Streamlit testing:

- Keep browser automation out unless it becomes necessary.
- Unit-test UI helper/service logic where possible.
- Manual smoke test should cover the integrated UI after implementation.

## 17. Validation Commands

Run after Phase 03 implementation:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run app
git status --short
git check-ignore -v .env .env.local .local/secrets.local.json storage/app.db
```

Manual app smoke:

1. Start `uv run app`.
2. Create/select workspace.
3. Upload small Markdown/PDF if needed.
4. Build/rebuild retrieval index if missing.
5. Select up to 3 documents.
6. Enter temporary API key or use `.env`.
7. Start a new chat.
8. Ask a document-grounded question.
9. Confirm answer includes `[S#]`.
10. Confirm source list and expandable chunks render.
11. Ask a follow-up question.
12. Confirm dev panel shows original and rewritten query.
13. Ask a question not found in selected documents with outside knowledge disabled.
14. Confirm not-found response.
15. Enable outside knowledge and confirm separated sections.
16. Confirm API key is not persisted in SQLite or files.

Optional manual checks:

- Delete a chat session and confirm messages disappear.
- Delete a workspace and confirm sessions/messages cascade.
- Confirm Phase 02 retrieval debug panel still works.

## 18. Acceptance Criteria

Phase 03 is accepted when:

- Only `openai` is added as a new runtime dependency.
- No summaries, eval UI, MLflow, Docker, LangChain, or LlamaIndex code is added.
- OpenAI client wrapper works with configurable models.
- API key can come from `.env` or temporary UI input.
- API keys are not logged, printed, persisted, or stored in SQLite.
- Chat sessions persist per workspace.
- Chat messages persist under the current chat session.
- Query rewrite uses only current session history.
- Rewrite can be disabled.
- Ambiguous follow-ups can trigger a clarifying question instead of retrieval/generation.
- Retrieval uses Phase 02 `RetrievalService`.
- Missing/stale index prevents unnecessary OpenAI answer calls.
- Grounded-only mode is default.
- Grounded no-context behavior returns the required not-found message.
- Answers include `[S#]` citations for grounded claims.
- Source list maps `[S#]` to citation strings and chunk metadata.
- Outside-knowledge mode separates document-grounded content from model knowledge.
- Dev panel shows original query, rewritten query, selected docs, retrieval trace, prompt metadata, model, token usage when available, and sources.
- Tests pass without real OpenAI API calls.

## 19. Risks and Known Limitations

- Model compliance risk: prompts can require citations, but an LLM may still omit or misuse markers. Phase 03 should flag deterministic citation warnings but not add a judge.
- Context budget risk: without tokenizer dependency, source/history limits are approximate.
- SQLite schema evolution risk: idempotent `CREATE TABLE IF NOT EXISTS` is acceptable for MVP but may need migrations later.
- Secret handling risk: temporary UI API key must stay in session state only; tests should cover no SQLite/file persistence.
- Cost risk: query rewrite can double API calls for follow-up turns; skip rewrite when unnecessary.
- Quality risk: ambiguity detection is heuristic and prompt-driven.
- Trace privacy risk: retrieval traces can include document text; they should not be automatically logged.
- UX risk: combining ingestion, retrieval debug, and chat in one Streamlit file may become large. If implementation grows too large, split minimal UI helper modules for chat panels only.

## 20. Questions for User Review

Blocking before implementation:

- None. The Phase 03 scope and dependency boundary are clear.

Non-blocking review decisions:

1. Approve using the OpenAI Responses API rather than Chat Completions for new Phase 03 calls?
2. Approve storing compact retrieval/prompt metadata on chat messages, while avoiding full trace persistence by default?
3. Approve non-streaming responses for Phase 03 MVP?
4. Approve simple idempotent SQLite table additions instead of migration files for Phase 03?
5. Approve temporary UI API key only in `st.session_state`, with no saved local key manager in Phase 03?

## 21. Implementation Sequence for the Next Run

Recommended order after user approval:

1. Add `openai` to `pyproject.toml`; run `uv sync`.
2. Add LLM dataclasses and OpenAI client wrapper with mocked tests.
3. Extend SQLite schema with `chat_sessions` and `chat_messages`; add repository tests.
4. Add chat models, repository, and service.
5. Add source mapping utilities and tests.
6. Add prompt templates and prompt-construction tests.
7. Add QA service orchestration with fake retrieval and fake LLM tests.
8. Add query rewrite handling and tests.
9. Update Streamlit UI with chat session controls and chat panel.
10. Add dev/debug panel rendering.
11. Update README with Phase 03 usage and security notes.
12. Run validation commands.
13. Run manual app smoke with a real OpenAI key only if the user has one configured or enters a temporary key.
14. Stop before Phase 04.

## 22. Reviewer Checklist

- [ ] Plan adds only `openai`.
- [ ] Plan does not include summaries, eval UI, MLflow, Docker, LangChain, or LlamaIndex.
- [ ] API key sources are `.env` and temporary UI input only.
- [ ] API keys are never persisted in SQLite or local files.
- [ ] OpenAI wrapper is isolated behind internal client/service APIs.
- [ ] Grounded-only behavior is the default.
- [ ] Not-found response is application-side when grounded context is empty.
- [ ] Outside-knowledge mode requires explicit separated sections.
- [ ] Query rewrite uses only current chat session history.
- [ ] Ambiguity handling can ask a clarifying question.
- [ ] Chat SQLite tables have columns, keys, indexes, and cascades defined.
- [ ] Retrieval integration uses Phase 02 `RetrievalService`.
- [ ] Missing/stale indexes prevent unnecessary OpenAI calls.
- [ ] Citation/source mapping is deterministic and unit-testable.
- [ ] Token usage is captured if available, without cost estimation.
- [ ] Tests use mocked OpenAI clients only.
- [ ] Implementation remains approval-gated.
