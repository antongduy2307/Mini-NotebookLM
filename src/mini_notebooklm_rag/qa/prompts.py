"""Prompt builders for grounded QA and query rewriting."""

from __future__ import annotations

from dataclasses import dataclass

from mini_notebooklm_rag.chat.models import ChatMessage
from mini_notebooklm_rag.qa.source_mapping import SourceReference

NOT_FOUND_MESSAGE = "I could not find this information in the selected documents."
DEFAULT_SOURCE_TEXT_CHAR_LIMIT = 2500


@dataclass(frozen=True)
class PromptBundle:
    instructions: str
    input_text: str
    metadata: dict


def build_grounded_qa_prompt(
    question: str,
    sources: list[SourceReference],
    max_source_chars: int = DEFAULT_SOURCE_TEXT_CHAR_LIMIT,
) -> PromptBundle:
    source_text = _format_sources(sources, max_source_chars)
    instructions = "\n".join(
        [
            "You answer questions using only the provided sources.",
            "Every factual claim based on the sources must cite one or more source IDs like [S1].",
            "Do not cite sources that do not support the claim.",
            "If the answer is not supported by the provided sources, respond exactly: "
            f"{NOT_FOUND_MESSAGE}",
            "Do not use outside knowledge.",
        ]
    )
    input_text = f"Question:\n{question}\n\nSources:\n{source_text}"
    return PromptBundle(
        instructions=instructions,
        input_text=input_text,
        metadata=_prompt_metadata("grounded", sources, source_text),
    )


def build_outside_knowledge_prompt(
    question: str,
    sources: list[SourceReference],
    max_source_chars: int = DEFAULT_SOURCE_TEXT_CHAR_LIMIT,
) -> PromptBundle:
    source_text = _format_sources(sources, max_source_chars) if sources else "No relevant sources."
    instructions = "\n".join(
        [
            "Use the provided sources first.",
            "Separate the answer into exactly these sections:",
            "From your documents:",
            "Outside the selected documents:",
            "Document-grounded claims must cite source IDs like [S1].",
            "Outside-knowledge content must be clearly labeled and must not use source IDs.",
            "If the documents do not support part of the answer, say so before adding "
            "outside knowledge.",
        ]
    )
    input_text = f"Question:\n{question}\n\nSources:\n{source_text}"
    return PromptBundle(
        instructions=instructions,
        input_text=input_text,
        metadata=_prompt_metadata("outside_knowledge", sources, source_text),
    )


def build_query_rewrite_prompt(
    question: str,
    history: list[ChatMessage],
    selected_document_ids: list[str],
    max_history_messages: int = 6,
) -> PromptBundle:
    recent_history = history[-max_history_messages:]
    history_text = (
        "\n".join(f"{message.role}: {message.content}" for message in recent_history)
        or "No prior messages."
    )
    instructions = "\n".join(
        [
            "Rewrite follow-up questions into standalone retrieval queries.",
            "Use only the current chat history and selected document IDs.",
            "If the question is too ambiguous, ask a short clarifying question.",
            'Return only JSON: {"action":"rewrite","query":"..."} or '
            '{"action":"clarify","question":"..."}',
        ]
    )
    input_text = "\n".join(
        [
            f"Selected document IDs: {', '.join(selected_document_ids)}",
            "Current chat history:",
            history_text,
            "Latest user question:",
            question,
        ]
    )
    return PromptBundle(
        instructions=instructions,
        input_text=input_text,
        metadata={
            "prompt_type": "query_rewrite",
            "history_message_count": len(recent_history),
            "selected_document_count": len(selected_document_ids),
        },
    )


def _format_sources(sources: list[SourceReference], max_source_chars: int) -> str:
    blocks = [source.to_prompt_block(max_source_chars) for source in sources]
    return "\n\n".join(blocks) if blocks else "No relevant sources."


def _prompt_metadata(
    prompt_type: str,
    sources: list[SourceReference],
    source_text: str,
) -> dict:
    return {
        "prompt_type": prompt_type,
        "source_count": len(sources),
        "source_characters": len(source_text),
        "source_ids": [source.source_id for source in sources],
    }
