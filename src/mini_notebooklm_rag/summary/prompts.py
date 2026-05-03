"""Prompt builders for per-document summaries."""

from __future__ import annotations

from dataclasses import dataclass

from mini_notebooklm_rag.storage.repositories import DocumentRecord
from mini_notebooklm_rag.summary.models import (
    SUMMARY_MODE_OVERVIEW,
    SUMMARY_PROMPT_VERSION,
    SummaryChunkGroup,
    SummaryPlan,
)


@dataclass(frozen=True)
class SummaryPrompt:
    """Prompt payload plus non-secret metadata."""

    instructions: str
    input_text: str
    metadata: dict[str, object]


def build_direct_overview_prompt(document: DocumentRecord, plan: SummaryPlan) -> SummaryPrompt:
    """Build a single-call overview summary prompt."""
    group = plan.groups[0]
    return SummaryPrompt(
        instructions=_base_instructions()
        + "\nSummarize the provided document excerpts directly into the required sections.",
        input_text=_input_header(document, plan)
        + _partial_notice(plan)
        + "\n\nSource excerpts:\n"
        + group.text,
        metadata=_metadata("direct", document, plan),
    )


def build_map_summary_prompt(
    document: DocumentRecord,
    group: SummaryChunkGroup,
    group_number: int,
    group_count: int,
    is_partial: bool,
) -> SummaryPrompt:
    """Build the map-step prompt for one chunk group."""
    partial_notice = (
        "\nThis document summary is partial; this is one included source group."
        if is_partial
        else ""
    )
    return SummaryPrompt(
        instructions=_base_instructions()
        + "\nCreate a concise partial summary of only this source group.",
        input_text=(
            f"Document: {document.display_name}\n"
            f"Source type: {document.source_type}\n"
            f"Group: {group_number} of {group_count}\n"
            f"Group label: {group.label}"
            f"{partial_notice}\n\n"
            f"Source excerpts:\n{group.text}"
        ),
        metadata={
            "prompt_type": "map",
            "prompt_version": SUMMARY_PROMPT_VERSION,
            "summary_mode": SUMMARY_MODE_OVERVIEW,
            "document_id": document.id,
            "group_number": group_number,
            "group_count": group_count,
            "source_character_count": group.source_character_count,
        },
    )


def build_reduce_summary_prompt(
    document: DocumentRecord,
    partial_summaries: list[str],
    plan: SummaryPlan,
    partials_truncated: bool,
) -> SummaryPrompt:
    """Build the reduce-step prompt from partial summaries."""
    warning_text = ""
    if plan.is_partial or partials_truncated:
        warning_text = (
            "\nCoverage warning: some source content or partial summaries were omitted for "
            "cost control. The final summary must explicitly mention this limitation."
        )
    input_text = (
        f"Document: {document.display_name}\n"
        f"Source type: {document.source_type}\n"
        f"Partial summaries: {len(partial_summaries)}\n"
        f"{warning_text}\n\n"
        "Partial summaries:\n"
        + "\n\n".join(
            f"Partial {index + 1}:\n{summary}" for index, summary in enumerate(partial_summaries)
        )
    )
    return SummaryPrompt(
        instructions=_base_instructions()
        + "\nCombine the partial summaries into one coherent document overview.",
        input_text=input_text,
        metadata=_metadata("reduce", document, plan)
        | {"partial_summary_count": len(partial_summaries)},
    )


def _base_instructions() -> str:
    return (
        "You summarize only the provided source text. Do not use outside knowledge. "
        "Do not invent claims, document structure, page references, methods, or caveats. "
        "If the source is insufficient for a section, say so briefly. "
        "Use this exact section structure:\n"
        "Overview\n"
        "Key points\n"
        "Useful details\n"
        "Limitations or caveats in the document"
    )


def _input_header(document: DocumentRecord, plan: SummaryPlan) -> str:
    return (
        f"Document: {document.display_name}\n"
        f"Source type: {document.source_type}\n"
        f"Included chunks: {plan.source_chunk_count}\n"
        f"Included source characters: {plan.source_character_count}\n"
    )


def _partial_notice(plan: SummaryPlan) -> str:
    if not plan.is_partial:
        return ""
    warnings = "\n".join(f"- {warning}" for warning in plan.warnings)
    return f"\nCoverage is partial because:\n{warnings}\nMention this limitation in the summary."


def _metadata(prompt_type: str, document: DocumentRecord, plan: SummaryPlan) -> dict[str, object]:
    return {
        "prompt_type": prompt_type,
        "prompt_version": SUMMARY_PROMPT_VERSION,
        "summary_mode": SUMMARY_MODE_OVERVIEW,
        "document_id": document.id,
        "document_content_hash": document.content_hash,
        "source_chunk_count": plan.source_chunk_count,
        "source_character_count": plan.source_character_count,
        "is_partial": plan.is_partial,
        "group_count": len(plan.groups),
    }
