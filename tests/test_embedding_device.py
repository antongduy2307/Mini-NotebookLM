from __future__ import annotations

import pytest

from mini_notebooklm_rag.retrieval.embeddings import (
    EmbeddingDeviceError,
    resolve_embedding_device,
)


def test_auto_prefers_cuda_when_available() -> None:
    assert resolve_embedding_device("auto", cuda_available=lambda: True) == "cuda"


def test_auto_falls_back_to_cpu_when_cuda_unavailable() -> None:
    assert resolve_embedding_device("auto", cuda_available=lambda: False) == "cpu"


def test_explicit_cuda_requires_cuda() -> None:
    with pytest.raises(EmbeddingDeviceError, match="CUDA is not available"):
        resolve_embedding_device("cuda", cuda_available=lambda: False)


def test_invalid_embedding_device_is_rejected() -> None:
    with pytest.raises(EmbeddingDeviceError, match="auto, cpu, cuda"):
        resolve_embedding_device("metal", cuda_available=lambda: False)
