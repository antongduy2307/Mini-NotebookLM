from __future__ import annotations

import numpy as np

from mini_notebooklm_rag.retrieval.embeddings import EmbeddingModel


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.array([[3.0, 4.0] if "a" in text else [0.0, 2.0] for text in texts])


def test_embedding_model_loads_lazily_and_normalizes_vectors() -> None:
    created: list[FakeModel] = []

    def factory(model_name: str, device: str) -> FakeModel:
        assert model_name == "fake-model"
        assert device == "cpu"
        model = FakeModel()
        created.append(model)
        return model

    embeddings = EmbeddingModel(
        model_name="fake-model",
        requested_device="cpu",
        batch_size=2,
        model_factory=factory,
    )

    assert embeddings.info.dimension is None
    assert created == []

    vectors = embeddings.encode(["alpha", "zzz"])

    assert len(created) == 1
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert embeddings.info.dimension == 2
    assert embeddings.info.normalized is True
