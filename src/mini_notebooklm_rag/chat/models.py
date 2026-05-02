"""Typed chat persistence models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatSession:
    id: str
    workspace_id: str
    title: str
    selected_document_ids: list[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChatMessage:
    id: str
    workspace_id: str
    session_id: str
    role: str
    content: str
    selected_document_ids: list[str] | None
    original_query: str | None
    rewritten_query: str | None
    answer_mode: str | None
    source_map: list[dict] | None
    retrieval_metadata: dict | None
    prompt_metadata: dict | None
    model_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    created_at: str


@dataclass(frozen=True)
class NewChatMessage:
    workspace_id: str
    session_id: str
    role: str
    content: str
    selected_document_ids: list[str] | None = None
    original_query: str | None = None
    rewritten_query: str | None = None
    answer_mode: str | None = None
    source_map: list[dict] | None = None
    retrieval_metadata: dict | None = None
    prompt_metadata: dict | None = None
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
