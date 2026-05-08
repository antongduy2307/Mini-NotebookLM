"""Typed application settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration defaults matching `.env.example`.

    Secret-bearing values are accepted for future phases but are excluded from
    model representations so accidental logs do not reveal them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-4.1-nano"
    openai_query_rewrite_model: str = "gpt-4.1-nano"

    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "auto"
    embedding_batch_size: int = 32

    app_storage_dir: str = "storage"
    sqlite_db_path: str = "storage/app.db"
    local_secrets_path: str = ".local/secrets.local.json"

    auto_summary: bool = False
    enable_query_rewrite: bool = True
    allow_outside_knowledge: bool = False

    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 120
    retrieval_top_k: int = 6
    dense_weight: float = 0.65
    sparse_weight: float = 0.35

    mlflow_tracking_uri: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return cached settings without logging or printing secret values."""
    return Settings()
