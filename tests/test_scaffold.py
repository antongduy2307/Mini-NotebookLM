from __future__ import annotations

from mini_notebooklm_rag import __version__
from mini_notebooklm_rag.app import build_streamlit_command, main
from mini_notebooklm_rag.config import Settings

ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_QUERY_REWRITE_MODEL",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "APP_STORAGE_DIR",
    "SQLITE_DB_PATH",
    "LOCAL_SECRETS_PATH",
    "AUTO_SUMMARY",
    "ENABLE_QUERY_REWRITE",
    "ALLOW_OUTSIDE_KNOWLEDGE",
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "RETRIEVAL_TOP_K",
    "DENSE_WEIGHT",
    "SPARSE_WEIGHT",
    "MLFLOW_TRACKING_URI",
)


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults_match_phase_00_plan(monkeypatch) -> None:
    for env_name in ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == ""
    assert settings.openai_model == "gpt-4.1-nano"
    assert settings.openai_query_rewrite_model == "gpt-4.1-nano"
    assert settings.embedding_model_name == "BAAI/bge-base-en-v1.5"
    assert settings.embedding_device == "auto"
    assert settings.embedding_batch_size == 32
    assert settings.app_storage_dir == "storage"
    assert settings.sqlite_db_path == "storage/app.db"
    assert settings.local_secrets_path == ".local/secrets.local.json"
    assert settings.auto_summary is False
    assert settings.enable_query_rewrite is True
    assert settings.allow_outside_knowledge is False
    assert settings.chunk_size_tokens == 700
    assert settings.chunk_overlap_tokens == 120
    assert settings.retrieval_top_k == 6
    assert settings.dense_weight == 0.65
    assert settings.sparse_weight == 0.35
    assert settings.mlflow_tracking_uri == ""


def test_auto_summary_default_is_false() -> None:
    assert Settings(_env_file=None).auto_summary is False


def test_app_launcher_exposes_main_without_starting_streamlit() -> None:
    assert callable(main)


def test_app_launcher_uses_python_module_subprocess() -> None:
    command = build_streamlit_command(["--logger.level=debug"])

    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[-1] == "--logger.level=debug"
    assert command[4].endswith("streamlit_app.py")
