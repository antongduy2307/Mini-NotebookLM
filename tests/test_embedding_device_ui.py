from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.retrieval import service as retrieval_service
from mini_notebooklm_rag.retrieval.models import EmbeddingInfo
from mini_notebooklm_rag.streamlit_app import _embedding_device_index, settings_for_embedding_device


def test_settings_for_embedding_device_returns_updated_copy() -> None:
    settings = Settings(
        _env_file=None,
        app_storage_dir="storage-test",
        embedding_device="auto",
    )

    runtime_settings = settings_for_embedding_device(settings, "cpu")

    assert settings.embedding_device == "auto"
    assert runtime_settings.embedding_device == "cpu"


def test_embedding_device_index_defaults_to_settings_value() -> None:
    assert _embedding_device_index("cuda") == 1
    assert _embedding_device_index("cpu") == 2
    assert _embedding_device_index("unsupported") == 0


def test_retrieval_service_receives_ui_selected_embedding_device(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeEmbeddingModel:
        def __init__(self, model_name: str, requested_device: str, batch_size: int):
            captured["model_name"] = model_name
            captured["requested_device"] = requested_device
            captured["batch_size"] = str(batch_size)
            self.model_name = model_name
            self.requested_device = requested_device
            self.selected_device = "cpu"
            self.info = EmbeddingInfo(
                model_name=model_name,
                requested_device=requested_device,
                selected_device="cpu",
                dimension=None,
                normalized=True,
            )

    monkeypatch.setattr(retrieval_service, "EmbeddingModel", FakeEmbeddingModel)
    storage_dir = Path(".tmp") / f"embedding-device-ui-{uuid4().hex}" / "storage"
    settings = Settings(
        _env_file=None,
        app_storage_dir=str(storage_dir),
        embedding_model_name="fake-model",
        embedding_device="auto",
        embedding_batch_size=7,
    )

    runtime_settings = settings_for_embedding_device(settings, "cuda")
    service = retrieval_service.RetrievalService(runtime_settings)

    assert captured == {
        "model_name": "fake-model",
        "requested_device": "cuda",
        "batch_size": "7",
    }
    assert service.embedding_info.requested_device == "cuda"
    assert service.embedding_info.selected_device == "cpu"
