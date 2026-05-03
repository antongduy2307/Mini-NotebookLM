"""Data models and constants for document summaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from mini_notebooklm_rag.llm.models import TokenUsage
from mini_notebooklm_rag.storage.repositories import ChunkRecord

SUMMARY_MODE_OVERVIEW = "overview"
SUMMARY_PROMPT_VERSION = "summary-overview-v1"
SUMMARY_DIRECT_MAX_CHARS = 12_000
SUMMARY_MAP_GROUP_MAX_CHARS = 8_000
SUMMARY_REDUCE_MAX_PARTIAL_CHARS = 12_000
SUMMARY_MAX_GROUPS = 8
SUMMARY_MAX_CHUNKS = 80

SummaryStatus = Literal["cached", "generated", "skipped", "failed"]


@dataclass(frozen=True)
class SummaryConfig:
    """Budget and behavior knobs for overview summaries."""

    direct_max_chars: int = SUMMARY_DIRECT_MAX_CHARS
    map_group_max_chars: int = SUMMARY_MAP_GROUP_MAX_CHARS
    reduce_max_partial_chars: int = SUMMARY_REDUCE_MAX_PARTIAL_CHARS
    max_groups: int = SUMMARY_MAX_GROUPS
    max_chunks: int = SUMMARY_MAX_CHUNKS

    def to_dict(self) -> dict[str, int]:
        return {
            "direct_max_chars": self.direct_max_chars,
            "map_group_max_chars": self.map_group_max_chars,
            "reduce_max_partial_chars": self.reduce_max_partial_chars,
            "max_groups": self.max_groups,
            "max_chunks": self.max_chunks,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SummaryChunkGroup:
    """A bounded group of chunks sent to one summary prompt."""

    label: str
    chunks: tuple[ChunkRecord, ...]
    text: str
    source_character_count: int
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] | None = None


@dataclass(frozen=True)
class SummaryPlan:
    """Prompt grouping plan for one document."""

    groups: tuple[SummaryChunkGroup, ...]
    use_map_reduce: bool
    source_chunk_count: int
    source_character_count: int
    is_partial: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryCacheKey:
    """Stable cache key for a document summary."""

    document_id: str
    document_content_hash: str
    summary_mode: str
    model_name: str
    prompt_version: str
    config_hash: str


@dataclass(frozen=True)
class CachedSummary:
    """Persisted summary row."""

    id: str
    workspace_id: str
    document_id: str
    document_content_hash: str
    summary_mode: str
    model_name: str
    prompt_version: str
    config_hash: str
    config_json: str
    summary_text: str
    source_chunk_count: int
    source_character_count: int
    is_partial: bool
    warnings: list[str]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class NewCachedSummary:
    """New or replacement summary row."""

    id: str
    workspace_id: str
    document_id: str
    document_content_hash: str
    summary_mode: str
    model_name: str
    prompt_version: str
    config_hash: str
    config_json: str
    summary_text: str
    source_chunk_count: int
    source_character_count: int
    is_partial: bool
    warnings: list[str]
    token_usage: TokenUsage


@dataclass(frozen=True)
class SummaryResult:
    """Service result for cache, generation, and skip outcomes."""

    status: SummaryStatus
    message: str
    summary: CachedSummary | None = None
    generated: bool = False
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)
    model_name: str | None = None
    prompt_version: str = SUMMARY_PROMPT_VERSION
    token_usage: TokenUsage = field(default_factory=TokenUsage)
