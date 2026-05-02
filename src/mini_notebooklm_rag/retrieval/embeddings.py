"""Local embedding model wrapper with device selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mini_notebooklm_rag.retrieval.models import EmbeddingInfo


class EmbeddingDeviceError(RuntimeError):
    """Raised when the requested embedding device cannot be used."""


class EmbeddingModelError(RuntimeError):
    """Raised when embedding model loading or encoding fails."""


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def resolve_embedding_device(
    requested_device: str,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    """Resolve `auto`, `cpu`, or `cuda` into the device used by embeddings."""
    requested = requested_device.strip().lower()
    has_cuda = (cuda_available or _torch_cuda_available)()

    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if has_cuda:
            return "cuda"
        raise EmbeddingDeviceError(
            "EMBEDDING_DEVICE=cuda was requested, but CUDA is not available. "
            "Use EMBEDDING_DEVICE=auto or install a CUDA-enabled PyTorch environment."
        )
    raise EmbeddingDeviceError(
        f"EMBEDDING_DEVICE must be one of: auto, cpu, cuda. Received: {requested_device!r}."
    )


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32", copy=False)


class EmbeddingModel:
    """Lazy local embedding wrapper.

    `model_factory` exists for tests so they do not download a real model.
    """

    def __init__(
        self,
        model_name: str,
        requested_device: str = "auto",
        batch_size: int = 32,
        model_factory: Callable[[str, str], Any] | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ):
        self.model_name = model_name
        self.requested_device = requested_device
        self.selected_device = resolve_embedding_device(requested_device, cuda_available)
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model: Any | None = None
        self._dimension: int | None = None
        self.normalized = True

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            model_name=self.model_name,
            requested_device=self.requested_device,
            selected_device=self.selected_device,
            dimension=self._dimension,
            normalized=self.normalized,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors."""
        if not texts:
            return np.empty((0, self._dimension or 0), dtype="float32")

        model = self._load_model()
        try:
            raw_vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except TypeError:
            raw_vectors = model.encode(texts)
        except Exception as exc:
            raise EmbeddingModelError("Embedding encode failed.") from exc

        vectors = np.asarray(raw_vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise EmbeddingModelError("Embedding model returned vectors with invalid shape.")

        self._dimension = int(vectors.shape[1])
        return _normalize_rows(vectors)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if self._model_factory is not None:
            self._model = self._model_factory(self.model_name, self.selected_device)
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingModelError("sentence-transformers is not installed.") from exc

        try:
            self._model = SentenceTransformer(self.model_name, device=self.selected_device)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model {self.model_name!r} "
                f"on device {self.selected_device!r}."
            ) from exc
        return self._model
