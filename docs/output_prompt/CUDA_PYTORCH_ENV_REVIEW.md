# CUDA PyTorch Environment Review

## 1. Executive Summary

This was an environment/configuration task before Phase 04, not a feature phase. The project was
configured so `uv sync` installs a CUDA-enabled PyTorch wheel for `sentence-transformers`
embeddings when the local machine has a compatible NVIDIA driver.

Retrieval architecture was not changed. FAISS remains `faiss-cpu`. No NVIDIA system drivers were
installed. No full system CUDA Toolkit was installed. No OpenAI, MLflow, LangChain, LlamaIndex, or
unrelated dependencies were added.

Final result on this machine:

- `torch`: `2.11.0+cu126`
- `torch.cuda.is_available()`: `True`
- `torch.version.cuda`: `12.6`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- `sentence-transformers` import: OK

## 2. Environment Diagnostics

Shell Python:

- `python --version`: `Python 3.13.12`

Project Python through uv:

- `uv run python -c "import sys; print(sys.version); print(sys.executable)"`
- Version: `3.12.12`
- Executable: `E:\allPythonProject\mini_notebooklm_rag\.venv\Scripts\python.exe`

uv:

- `uv --version`: `uv 0.11.6 (65950801c 2026-04-09 x86_64-pc-windows-msvc)`

NVIDIA driver/GPU:

- `nvidia-smi`: available
- Driver: `595.97`
- Driver-reported CUDA version: `13.2`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- VRAM: `8188 MiB`

uv PyTorch backend support:

- `uv pip install --help` exposes `--torch-backend`.
- Supported values include `auto`, `cpu`, and CUDA backends including `cu126`.
- `uv sync --help` does not expose `--torch-backend`, so project configuration was needed for
  normal `uv sync` reproducibility.

## 3. Current Torch/Sentence-Transformers Status

Before configuration:

```text
torch 2.11.0+cpu
cuda available False
torch cuda None
device CPU only
```

After configuration and `uv sync`:

```text
torch: 2.11.0+cu126
cuda available: True
torch cuda: 12.6
cuda device count: 1
device: NVIDIA GeForce RTX 4060 Laptop GPU
```

Sentence Transformers:

```text
sentence-transformers import ok
```

## 4. Whether CUDA Is Available

CUDA is available to PyTorch in the project virtual environment after this task.

`EMBEDDING_DEVICE=auto` will now resolve to `cuda` on this machine. `EMBEDDING_DEVICE=cuda` should
also work on this machine. CPU fallback remains available with `EMBEDDING_DEVICE=cpu`.

## 5. Changes Made

### `pyproject.toml`

Added `torch` as an explicit project dependency so uv can apply a package source override to it.
`torch` was already present transitively through `sentence-transformers`; making it explicit is what
makes CUDA wheel selection reproducible under `uv sync`.

Added a dedicated explicit PyTorch CUDA 12.6 wheel index:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```

### `uv.lock`

Updated the lockfile so `torch` resolves from the PyTorch CUDA 12.6 wheel index:

- `torch` changed from `2.11.0` to `2.11.0+cu126`.
- The selected torch source is `https://download.pytorch.org/whl/cu126`.

The lockfile includes PyTorch wheel runtime package metadata. This is not a system driver install
and does not replace or install NVIDIA system CUDA Toolkit components. On Windows, the installed
torch wheel provides the runtime pieces needed by PyTorch itself.

### `scripts/check_cuda.py`

Added a small diagnostic script that prints:

- Python version
- Python executable
- torch version
- CUDA availability
- torch CUDA runtime version
- CUDA device count
- GPU name or `CPU only`

### `.env.example`

Added comments explaining:

- `auto` prefers CUDA only when CUDA-enabled PyTorch and a compatible driver are available.
- `cuda` requires CUDA-enabled PyTorch.
- `cpu` forces CPU embeddings.

### `README.md`

Added an embedding-device setup section with:

- `.env` values for `auto`, `cuda`, and `cpu`
- `uv run python scripts/check_cuda.py`
- note that FAISS remains CPU-only
- note that full CUDA Toolkit is not required by this project

## 6. Exact Commands Run

Diagnostics:

```bash
python --version
uv --version
uv run python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('torch cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
uv run python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers import ok')"
uv run python -c "from mini_notebooklm_rag.config import get_settings; from mini_notebooklm_rag.retrieval.embeddings import resolve_embedding_device; settings=get_settings(); print('config embedding device', settings.embedding_device); print('resolved device', resolve_embedding_device(settings.embedding_device))"
nvidia-smi
uv pip install --help
uv sync --help
uv add --help
uv lock --help
uv tree
```

PyTorch backend checks:

```bash
uv pip install torch --torch-backend auto --dry-run
uv pip install torch --torch-backend cu126 --reinstall-package torch --dry-run
uv pip install torch --torch-backend cu126 --reinstall-package torch
uv sync --dry-run
uv lock --upgrade-package torch
uv sync
```

Validation:

```bash
uv run python scripts/check_cuda.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Notes:

- The first sandboxed PyTorch CUDA dry-run could not reach `https://download.pytorch.org` because
  network access was restricted. It was rerun with approved escalation and resolved cleanly.
- Installing `torch==2.11.0+cu126` downloaded a large PyTorch wheel, about `2.4 GiB`.

## 7. Validation Results

`uv sync`

- PASS.
- Installed `torch==2.11.0+cu126`.

`uv run python scripts/check_cuda.py`

- PASS.

```text
python: 3.12.12
executable: E:\allPythonProject\mini_notebooklm_rag\.venv\Scripts\python.exe
torch: 2.11.0+cu126
cuda available: True
torch cuda: 12.6
cuda device count: 1
device: NVIDIA GeForce RTX 4060 Laptop GPU
```

`uv run pytest`

- PASS.
- `74 passed, 3 warnings in 5.35s`.
- Warnings are existing SWIG/PyMuPDF deprecation warnings.

`uv run ruff check .`

- PASS.
- `All checks passed!`

`uv run ruff format --check .`

- PASS.
- `63 files already formatted`.

## 8. Remaining User Actions If Driver Changes Are Needed

No NVIDIA driver change is needed on this machine. `nvidia-smi` reports driver `595.97`, CUDA
`13.2`, and PyTorch CUDA is available.

On another machine, if `scripts/check_cuda.py` reports `cuda available: False` after `uv sync`:

1. Run `nvidia-smi`.
2. If `nvidia-smi` is missing or reports a driver error, install or repair the NVIDIA driver outside
   this project.
3. Do not install the full CUDA Toolkit for this project unless a separate non-project workflow
   requires it.
4. Rerun `uv sync`.
5. Rerun `uv run python scripts/check_cuda.py`.

## 9. How to Set `.env`

Recommended default:

```env
EMBEDDING_DEVICE=auto
```

Behavior:

- Uses CUDA when CUDA-enabled PyTorch and a compatible NVIDIA driver are available.
- Falls back to CPU otherwise.

Require CUDA and fail clearly if unavailable:

```env
EMBEDDING_DEVICE=cuda
```

Behavior:

- Uses CUDA only.
- If CUDA is unavailable, the app shows:
  `CUDA was requested but is not available in the current PyTorch environment.`

Force CPU:

```env
EMBEDDING_DEVICE=cpu
```

Behavior:

- Uses CPU embeddings even when CUDA is available.

## 10. Risks and Rollback Instructions

Risks:

- The CUDA PyTorch wheel is large and increases environment sync/download size.
- CUDA availability depends on a compatible NVIDIA driver on each machine.
- The project now has `torch` as an explicit dependency to make uv source selection reproducible.
- The lockfile includes CUDA runtime package metadata for PyTorch wheels. This is Python package
  metadata, not a system driver install.
- CPU-only machines can still run the app with `EMBEDDING_DEVICE=auto` or `EMBEDDING_DEVICE=cpu`,
  but they may download a larger CUDA wheel unless the project is rolled back or a CPU-specific
  install path is used.

Rollback:

1. Remove `"torch"` from `[project.dependencies]` in `pyproject.toml`.
2. Remove the `[[tool.uv.index]]` entry named `pytorch-cu126`.
3. Remove `[tool.uv.sources] torch = { index = "pytorch-cu126" }`.
4. Run:

   ```bash
   uv lock --upgrade-package torch
   uv sync
   uv run python scripts/check_cuda.py
   ```

5. Set `.env` to:

   ```env
   EMBEDDING_DEVICE=cpu
   ```

## 11. Next Recommended Step

Phase 04 is still waiting. Before Phase 04, the user/reviewer should confirm that the larger CUDA
PyTorch wheel is acceptable for this portfolio project and that CPU-only reviewers understand the
rollback or CPU-device path.
