"""Print local PyTorch/CUDA and sentence-transformers diagnostics."""

from __future__ import annotations

import importlib.util
import sys


def main() -> None:
    """Print dependency availability without changing app behavior."""
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        print("torch: not installed")
        print("cuda available: false")
    else:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"torch: {torch.__version__}")
        print(f"cuda available: {cuda_available}")
        print(f"torch cuda: {torch.version.cuda}")
        print(f"cuda device count: {torch.cuda.device_count() if cuda_available else 0}")
        print(f"device: {torch.cuda.get_device_name(0) if cuda_available else 'CPU only'}")

    sentence_transformers_spec = importlib.util.find_spec("sentence_transformers")
    print(
        "sentence-transformers: "
        f"{'installed' if sentence_transformers_spec is not None else 'not installed'}"
    )


if __name__ == "__main__":
    main()
