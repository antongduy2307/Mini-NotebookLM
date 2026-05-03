"""Summary service orchestration with SQLite caching."""

from __future__ import annotations

from typing import Protocol

from mini_notebooklm_rag.config import Settings
from mini_notebooklm_rag.llm.models import LLMResponse, TokenUsage
from mini_notebooklm_rag.llm.openai_client import OpenAIClient, OpenAIClientError
from mini_notebooklm_rag.storage.paths import StoragePaths
from mini_notebooklm_rag.storage.repositories import DocumentRepository
from mini_notebooklm_rag.storage.sqlite import initialize_database
from mini_notebooklm_rag.summary.grouping import build_summary_plan
from mini_notebooklm_rag.summary.models import (
    SUMMARY_MODE_OVERVIEW,
    SUMMARY_PROMPT_VERSION,
    NewCachedSummary,
    SummaryCacheKey,
    SummaryConfig,
    SummaryResult,
)
from mini_notebooklm_rag.summary.prompts import (
    build_direct_overview_prompt,
    build_map_summary_prompt,
    build_reduce_summary_prompt,
)
from mini_notebooklm_rag.summary.repositories import SummaryRepository, new_summary_id


class SummaryServiceError(RuntimeError):
    """Raised for summary service failures."""


class LLMClientProtocol(Protocol):
    """Minimal LLM client shape used by summary generation."""

    def generate(
        self,
        instructions: str,
        input_text: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text from a prompt."""


class SummaryService:
    """Generate and cache per-document summaries."""

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClientProtocol | None = None,
    ):
        self.settings = settings
        storage_paths = StoragePaths(root=settings.app_storage_dir)
        storage_paths.ensure_root()
        initialize_database(storage_paths.db_path)
        self.documents = DocumentRepository(storage_paths.db_path)
        self.summaries = SummaryRepository(storage_paths.db_path)
        self._llm_client = llm_client

    def get_cached_summary(
        self,
        document_id: str,
        summary_mode: str = SUMMARY_MODE_OVERVIEW,
        model_name: str | None = None,
        config: SummaryConfig | None = None,
    ):
        """Return the current cache hit for this document/config, if any."""
        document = self.documents.get(document_id)
        if document is None:
            return None
        summary_config = config or SummaryConfig()
        cache_key = self._cache_key(
            document_id,
            document.content_hash,
            summary_mode,
            model_name,
            summary_config,
        )
        return self.summaries.get_by_cache_key(cache_key)

    def latest_summary(
        self,
        document_id: str,
        summary_mode: str = SUMMARY_MODE_OVERVIEW,
    ):
        """Return the most recently updated summary for a document."""
        return self.summaries.latest_for_document(document_id, summary_mode)

    def generate_for_document(
        self,
        document_id: str,
        api_key: str,
        model_name: str | None = None,
        summary_mode: str = SUMMARY_MODE_OVERVIEW,
        regenerate: bool = False,
        config: SummaryConfig | None = None,
    ) -> SummaryResult:
        """Generate or fetch a cached per-document summary."""
        if summary_mode != SUMMARY_MODE_OVERVIEW:
            return SummaryResult(
                status="failed",
                message=f"Unsupported summary mode: {summary_mode}",
                model_name=model_name or self.settings.openai_model,
            )

        document = self.documents.get(document_id)
        if document is None:
            return SummaryResult(
                status="failed",
                message="Document was not found.",
                model_name=model_name or self.settings.openai_model,
            )

        chunks = self.documents.list_chunks(document_id)
        if not chunks:
            return SummaryResult(
                status="skipped",
                message="Summary skipped because the document has no chunks.",
                model_name=model_name or self.settings.openai_model,
            )

        summary_config = config or SummaryConfig()
        effective_model = model_name or self.settings.openai_model
        cache_key = self._cache_key(
            document.id,
            document.content_hash,
            summary_mode,
            effective_model,
            summary_config,
        )
        if not regenerate:
            cached = self.summaries.get_by_cache_key(cache_key)
            if cached is not None:
                return SummaryResult(
                    status="cached",
                    message="Loaded cached summary.",
                    summary=cached,
                    from_cache=True,
                    warnings=cached.warnings,
                    model_name=cached.model_name,
                    token_usage=TokenUsage(
                        input_tokens=cached.input_tokens or 0,
                        output_tokens=cached.output_tokens or 0,
                        total_tokens=cached.total_tokens or 0,
                    ),
                )

        if not api_key and self._llm_client is None:
            return SummaryResult(
                status="skipped",
                message="Summary skipped because no OpenAI API key is configured.",
                model_name=effective_model,
            )

        plan = build_summary_plan(document, chunks, summary_config)
        try:
            if plan.use_map_reduce:
                summary_text, token_usage, warnings = self._generate_map_reduce(
                    document,
                    plan,
                    effective_model,
                    summary_config,
                    api_key,
                )
            else:
                summary_text, token_usage, warnings = self._generate_direct(
                    document,
                    plan,
                    effective_model,
                    api_key,
                )
        except OpenAIClientError as exc:
            return SummaryResult(
                status="failed",
                message=f"Summary generation failed: {exc}",
                warnings=plan.warnings,
                model_name=effective_model,
            )

        if not summary_text.strip():
            return SummaryResult(
                status="failed",
                message="Summary generation returned no text.",
                warnings=warnings,
                model_name=effective_model,
                token_usage=token_usage,
            )

        cached = self.summaries.upsert(
            NewCachedSummary(
                id=new_summary_id(),
                workspace_id=document.workspace_id,
                document_id=document.id,
                document_content_hash=document.content_hash,
                summary_mode=summary_mode,
                model_name=effective_model,
                prompt_version=SUMMARY_PROMPT_VERSION,
                config_hash=summary_config.hash(),
                config_json=summary_config.to_json(),
                summary_text=summary_text.strip(),
                source_chunk_count=plan.source_chunk_count,
                source_character_count=plan.source_character_count,
                is_partial=plan.is_partial,
                warnings=warnings,
                token_usage=token_usage,
            )
        )
        return SummaryResult(
            status="generated",
            message="Generated summary.",
            summary=cached,
            generated=True,
            warnings=warnings,
            model_name=effective_model,
            token_usage=token_usage,
        )

    def _generate_direct(
        self,
        document,
        plan,
        model_name: str,
        api_key: str,
    ) -> tuple[str, TokenUsage, list[str]]:
        prompt = build_direct_overview_prompt(document, plan)
        response = self._client(api_key, model_name).generate(
            prompt.instructions,
            prompt.input_text,
            model=model_name,
        )
        return response.text, response.token_usage, list(plan.warnings)

    def _generate_map_reduce(
        self,
        document,
        plan,
        model_name: str,
        config: SummaryConfig,
        api_key: str,
    ) -> tuple[str, TokenUsage, list[str]]:
        client = self._client(api_key, model_name)
        partials: list[str] = []
        token_usage = TokenUsage()
        warnings = list(plan.warnings)
        for index, group in enumerate(plan.groups, start=1):
            prompt = build_map_summary_prompt(
                document,
                group,
                index,
                len(plan.groups),
                plan.is_partial,
            )
            response = client.generate(prompt.instructions, prompt.input_text, model=model_name)
            partials.append(response.text.strip())
            token_usage = _add_usage(token_usage, response.token_usage)

        partials_text = "\n\n".join(partials)
        partials_truncated = False
        if len(partials_text) > config.reduce_max_partial_chars:
            partials_text = partials_text[: config.reduce_max_partial_chars]
            partials = [partials_text]
            partials_truncated = True
            warnings.append(
                "Partial summaries were truncated before the reduce step for cost control."
            )

        reduce_prompt = build_reduce_summary_prompt(document, partials, plan, partials_truncated)
        response = client.generate(
            reduce_prompt.instructions,
            reduce_prompt.input_text,
            model=model_name,
        )
        token_usage = _add_usage(token_usage, response.token_usage)
        return response.text, token_usage, warnings

    def _client(self, api_key: str, model_name: str) -> LLMClientProtocol:
        if self._llm_client is not None:
            return self._llm_client
        return OpenAIClient(api_key=api_key, default_model=model_name)

    def _cache_key(
        self,
        document_id: str,
        document_content_hash: str,
        summary_mode: str,
        model_name: str | None,
        config: SummaryConfig,
    ) -> SummaryCacheKey:
        return SummaryCacheKey(
            document_id=document_id,
            document_content_hash=document_content_hash,
            summary_mode=summary_mode,
            model_name=model_name or self.settings.openai_model,
            prompt_version=SUMMARY_PROMPT_VERSION,
            config_hash=config.hash(),
        )


def _add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=(left.input_tokens or 0) + (right.input_tokens or 0),
        output_tokens=(left.output_tokens or 0) + (right.output_tokens or 0),
        total_tokens=(left.total_tokens or 0) + (right.total_tokens or 0),
    )
