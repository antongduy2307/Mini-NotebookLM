# Streamlit Watcher Config Fix

## Summary

Added a project-level Streamlit config to disable Streamlit file watching:

```toml
[server]
fileWatcherType = "none"
```

This targets file watcher tracebacks/noise caused by `transformers` optional vision module imports. No `torchvision` dependency was added.

## Files Changed

- `.streamlit/config.toml`
  - New Streamlit configuration file.
  - Disables Streamlit file watching with `fileWatcherType = "none"`.

- `README.md`
  - Updated run notes with:
    - default `uv run app`
    - debug fallback `uv run app -- --logger.level=debug`
    - note that Streamlit watcher is disabled by config if dependency watcher tracebacks appear

## Explicit Non-Changes

- No retrieval logic changed.
- No chat logic changed.
- No summary logic changed.
- No evaluation logic changed.
- No dependencies added.
- `torchvision` was not added.

## Validation Results

```bash
uv run pytest
```

Result: passed.

```text
96 passed, 3 warnings
```

```bash
uv run ruff check .
```

Result: passed.

```text
All checks passed!
```

```bash
uv run ruff format --check .
```

Result: passed.

```text
81 files already formatted
```

```bash
uv run app
```

Result: bounded startup smoke passed.

```text
uv run app bounded smoke state: Running
Local URL: http://localhost:8877
```

Project-local app/python processes left by the smoke test were stopped afterward.

## Remaining Risks

- Disabling file watching means Streamlit may not auto-rerun on source edits during development. Restarting the app remains the reliable development workflow.
- This does not fix missing optional `transformers` vision dependencies; it avoids Streamlit watcher traversal/import noise around them.

## Next Recommended Step

Run the app normally with:

```bash
uv run app
```

Use debug logging when needed:

```bash
uv run app -- --logger.level=debug
```
