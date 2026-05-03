# App Launcher Fix Review

## 1. Root Cause Hypothesis

`uv run app` used Streamlit's internal in-process CLI API:

```python
from streamlit.web import cli as streamlit_cli
streamlit_cli.main()
```

That path mutates `sys.argv` and runs Streamlit inside the console entrypoint process. The reported behavior, where direct `uv run streamlit run src/mini_notebooklm_rag/streamlit_app.py --logger.level=debug` remains stable while `uv run app` can crash after UI interactions, points to launcher/process lifecycle instability rather than Streamlit UI, SQLite, retrieval, chat, ingestion, or summary logic.

The safer MVP launcher is to delegate to Streamlit as a child process using the current Python executable. This matches the stable direct command more closely and avoids importing/calling Streamlit's internal CLI API from application code.

## 2. Files Changed

- `src/mini_notebooklm_rag/app.py`: replaced in-process Streamlit CLI invocation with a subprocess launcher using `[sys.executable, "-m", "streamlit", "run", str(streamlit_app)]`; added simple passthrough args.
- `tests/test_scaffold.py`: added a unit test that verifies the launcher command uses Python module execution and preserves passthrough args.
- `README.md`: documented the direct Streamlit fallback command.
- `docs/output_prompt/APP_LAUNCHER_FIX_REVIEW.md`: this review digest.

No retrieval, chat, summary, ingestion, storage, dependency, or schema behavior was changed by this launcher fix.

## 3. Exact Launcher Behavior Before and After

Before:

```python
from streamlit.web import cli as streamlit_cli

sys.argv = [
    "streamlit",
    "run",
    str(streamlit_app),
    "--server.headless=true",
]
streamlit_cli.main()
```

Behavior:

- Imported Streamlit's internal CLI module.
- Mutated `sys.argv`.
- Ran Streamlit in the same Python process as the console launcher.
- Hard-coded `--server.headless=true`.

After:

```python
[
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(streamlit_app),
    *extra_args,
]
```

Behavior:

- Uses the current environment's Python executable.
- Starts Streamlit as a child process through `subprocess.run(...)`.
- Preserves Streamlit's exit code with `SystemExit(completed.returncode)`.
- Supports simple passthrough args, for example:

```bash
uv run app -- --logger.level=debug
```

No default Streamlit flags are forced. For headless smoke tests, pass:

```bash
uv run app -- --server.headless=true
```

## 4. Commands Run and Results

```bash
uv run pytest
```

Status: pass. 82 tests passed, 3 warnings from PyMuPDF/SWIG import deprecations.

```bash
uv run ruff check .
```

Status: pass. All checks passed.

```bash
uv run ruff format --check .
```

Status: pass. 69 files already formatted.

```bash
uv run app -- --server.headless=true --server.port=8766
```

Status: pass for bounded startup smoke. Streamlit started with no immediate launcher error and reported:

```text
Local URL: http://localhost:8766
```

The process was stopped after the smoke test.

## 5. Whether `uv run app` Now Behaves Like Direct `streamlit run`

Yes, for launcher mechanics. `uv run app` now delegates to:

```bash
python -m streamlit run src/mini_notebooklm_rag/streamlit_app.py
```

inside the active `uv` environment, which is equivalent in shape to the stable direct command:

```bash
uv run streamlit run src/mini_notebooklm_rag/streamlit_app.py
```

The remaining difference is that `uv run app` goes through the project console script first, then spawns Streamlit as a child process with the same Python executable.

## 6. Remaining Risks

- The smoke test verifies startup only; it does not prove long multi-click browser stability.
- Streamlit itself still owns server lifecycle, browser state, and rerun behavior.
- If a user passes unsupported Streamlit flags through `uv run app -- ...`, Streamlit will reject them.
- The subprocess launcher blocks until Streamlit exits, which is expected for a CLI app command.
- This change removes the prior default `--server.headless=true`; use passthrough args for headless runs.

## 7. Next Recommended Step

Run the app interactively with:

```bash
uv run app
```

Then repeat the clicks that previously crashed the UI. If instability remains, compare logs from:

```bash
uv run app -- --logger.level=debug
```

and:

```bash
uv run streamlit run src/mini_notebooklm_rag/streamlit_app.py --logger.level=debug
```

Phase 05 should not start until the launcher behavior is accepted.
