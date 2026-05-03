# Embedding Device UI Option Review

## Executive Summary

This change adds a Streamlit embedding device selector to the retrieval/chat UI without changing
retrieval architecture or adding dependencies. The selector offers `auto`, `cuda`, and `cpu`,
defaults from `settings.embedding_device`, and passes the selected value into the existing
`RetrievalService` initialization path for the current UI session.

FAISS remains CPU-only through `faiss-cpu`. The selector only controls the local embedding model
device resolution.

## Files Changed

- `.gitignore`
  - Added `.pytest-tmp/` and `.tmp/` for local test temp directories created during validation.
- `src/mini_notebooklm_rag/streamlit_app.py`
  - Added `EMBEDDING_DEVICE_OPTIONS = ["auto", "cuda", "cpu"]`.
  - Added a top-level Streamlit `selectbox` for embedding device selection.
  - Added `settings_for_embedding_device(...)` to create a runtime settings copy for the selected
    UI device.
  - Added `_embedding_device_index(...)` so the selector default follows current settings and
    falls back to `auto` for invalid config text.
  - Passes the runtime settings copy into `WorkspaceService`, `IngestionService`, `ChatService`,
    `RetrievalService`, and `QAService`.
  - Preserves resolved device display through the existing retrieval debug `embedding_info` block.
  - Guards `render()` with `if __name__ == "__main__"` so tests can import helper functions without
    rendering Streamlit.
- `src/mini_notebooklm_rag/retrieval/embeddings.py`
  - Changed explicit CUDA-unavailable error text to:
    `CUDA was requested but is not available in the current PyTorch environment.`
  - Kept existing `auto` behavior: CUDA when available, CPU fallback otherwise.
- `tests/test_embedding_device.py`
  - Updated explicit CUDA-unavailable assertion to require the user-facing actionable message.
- `tests/test_embedding_device_ui.py`
  - Added coverage for settings-copy behavior, selector default indexing, and `RetrievalService`
    receiving the UI-selected embedding device.

## Architecture Impact

No retrieval architecture changed.

The existing flow remains:

```text
Streamlit UI -> Settings -> RetrievalService -> EmbeddingModel -> device resolver
```

The only difference is that Streamlit now creates a runtime settings copy with the selected
`embedding_device` before constructing `RetrievalService`. No FAISS GPU path, vector store change,
embedding cache change, or retrieval algorithm change was introduced.

## UI Behavior

The UI now shows an `Embedding device` selectbox near the top of the app with these options:

- `auto`
- `cuda`
- `cpu`

Default selection:

- Uses `settings.embedding_device`.
- Falls back to `auto` if the configured value is not one of the supported UI options.

Resolved device display:

- The retrieval debug panel continues to display:
  - `embedding_model`
  - `requested_device`
  - `selected_device`
  - `dimension`
  - `normalized`

Error behavior:

- If the user selects `cuda` and CUDA is unavailable, `RetrievalService` initialization raises
  `EmbeddingDeviceError`.
- Streamlit catches that error and displays:
  `CUDA was requested but is not available in the current PyTorch environment.`

## Device Resolution Behavior

- `auto`: uses CUDA when `torch.cuda.is_available()` is true, otherwise CPU.
- `cuda`: requires CUDA; raises the actionable error above if unavailable.
- `cpu`: always uses CPU.

This continues to rely on the existing `resolve_embedding_device(...)` helper and the PyTorch
availability already pulled transitively through `sentence-transformers`.

## Dependency Check

No dependencies were added or removed.

FAISS remains `faiss-cpu`; this change does not introduce GPU FAISS or any CUDA-specific package.

## Test Coverage

| Test | Coverage |
| --- | --- |
| `tests/test_embedding_device.py::test_auto_falls_back_to_cpu_when_cuda_unavailable` | Confirms `auto` still falls back to CPU when CUDA is unavailable. |
| `tests/test_embedding_device.py::test_explicit_cuda_requires_cuda` | Confirms explicit CUDA request raises the exact actionable error. |
| `tests/test_embedding_device_ui.py::test_settings_for_embedding_device_returns_updated_copy` | Confirms UI helper updates only the runtime settings copy. |
| `tests/test_embedding_device_ui.py::test_embedding_device_index_defaults_to_settings_value` | Confirms selector default index follows settings and invalid config falls back to `auto`. |
| `tests/test_embedding_device_ui.py::test_retrieval_service_receives_ui_selected_embedding_device` | Confirms `RetrievalService` receives the UI-selected requested device. |

## Validation Results

Commands run:

- `uv run pytest tests/test_embedding_device.py tests/test_embedding_device_ui.py`
  - PASS: `7 passed`.
- `uv run pytest`
  - PASS outside sandbox: `74 passed, 3 warnings`.
  - Inside the sandbox, full-suite setup hit Windows temp-directory `Access denied` errors before
    many existing `tmp_path` tests could start. The standard command was rerun outside the sandbox
    and passed.
- `uv run ruff check .`
  - PASS: `All checks passed!`
- `uv run ruff format --check .`
  - PASS: `62 files already formatted`.
- `uv run app`
  - STARTUP PASS with bounded timeout. Streamlit printed `Local URL: http://localhost:8501`.

Known validation noise:

- PowerShell still reports the local profile execution-policy warning.
- Pytest still reports existing SWIG/PyMuPDF deprecation warnings.

## Risks and Limitations

- The UI selector changes device only for the current Streamlit session/runtime settings copy; it
  does not write `.env` or persist a preference.
- Selecting `cuda` requires the current PyTorch environment to have CUDA support. CPU-only PyTorch
  installations will show the actionable error.
- Switching embedding device does not automatically rebuild existing FAISS indexes. Existing Phase 02
  manual rebuild behavior remains unchanged.
- No browser automation was run; startup and unit tests validate the wiring.

## Reviewer Checklist

- Confirm no new dependencies were added.
- Confirm `faiss-cpu` remains unchanged.
- Confirm the selector options are exactly `auto`, `cuda`, and `cpu`.
- Confirm default selection comes from `settings.embedding_device`.
- Confirm `RetrievalService` receives the selected device through runtime settings.
- Confirm resolved requested/selected device and embedding model still display in retrieval debug UI.
- Confirm explicit CUDA-unavailable error text is actionable and matches requirements.
- Confirm tests cover selected-device wiring, explicit CUDA error, and auto CPU fallback.
