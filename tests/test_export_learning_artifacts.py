from __future__ import annotations

from mini_notebooklm_rag.export import (
    artifact_export_filename,
    flashcard_set_to_json_dict,
    flashcard_set_to_markdown,
    quiz_set_to_json_dict,
    quiz_set_to_markdown,
)
from mini_notebooklm_rag.learning.models import (
    LEARNING_MODE_QUERY,
    LEARNING_PROMPT_VERSION,
    Flashcard,
    FlashcardSet,
    QuizItem,
    QuizSet,
)
from mini_notebooklm_rag.llm.models import TokenUsage
from mini_notebooklm_rag.qa.source_mapping import SourceReference


def test_quiz_markdown_and_json_exports_are_compact() -> None:
    quiz_set = _quiz_set()

    markdown = quiz_set_to_markdown(quiz_set)
    payload = quiz_set_to_json_dict(quiz_set)

    assert "# Quiz" in markdown
    assert "What does hybrid retrieval combine?" in markdown
    assert "**Answer:** A" in markdown
    assert "[S1] sample.md > Retrieval" in markdown
    assert payload["format_version"] == 1
    assert payload["artifact_type"] == "quiz"
    assert "text" not in payload["source_map"][0]
    assert payload["items"][0]["correct_index"] == 0


def test_flashcard_markdown_and_json_exports_are_compact() -> None:
    flashcard_set = _flashcard_set()

    markdown = flashcard_set_to_markdown(flashcard_set)
    payload = flashcard_set_to_json_dict(flashcard_set)

    assert "# Flashcards" in markdown
    assert "What is hybrid retrieval?" in markdown
    assert "Two retrieval styles" in markdown
    assert payload["format_version"] == 1
    assert payload["artifact_type"] == "flashcards"
    assert "text" not in payload["source_map"][0]
    assert payload["cards"][0]["front"] == "What is hybrid retrieval?"


def test_artifact_export_filename_is_safe() -> None:
    filename = artifact_export_filename("Portfolio Demo/Bad", "Quiz Set", "json")

    assert filename.startswith("quiz_set_portfolio_demo_bad_")
    assert filename.endswith(".json")
    assert "/" not in filename


def _quiz_set() -> QuizSet:
    return QuizSet(
        id="quiz-1",
        workspace_id="workspace-1",
        selected_document_ids=["doc-1"],
        mode=LEARNING_MODE_QUERY,
        topic_or_query="retrieval",
        model_name="fake-model",
        prompt_version=LEARNING_PROMPT_VERSION,
        items=[
            QuizItem(
                question="What does hybrid retrieval combine?",
                options=["Dense and sparse", "PDFs", "Summaries", "Evals"],
                correct_index=0,
                explanation="It combines dense and sparse retrieval. [S1]",
                source_markers=["[S1]"],
            )
        ],
        source_map=_sources(),
        warnings=[],
        token_usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        created_at="2026-05-10T00:00:00+00:00",
    )


def _flashcard_set() -> FlashcardSet:
    return FlashcardSet(
        id="cards-1",
        workspace_id="workspace-1",
        selected_document_ids=["doc-1"],
        mode=LEARNING_MODE_QUERY,
        topic_or_query="retrieval",
        model_name="fake-model",
        prompt_version=LEARNING_PROMPT_VERSION,
        cards=[
            Flashcard(
                front="What is hybrid retrieval?",
                back="Combining dense and sparse retrieval. [S1]",
                hint="Two retrieval styles",
                source_markers=["[S1]"],
            )
        ],
        source_map=_sources(),
        warnings=[],
        token_usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        created_at="2026-05-10T00:00:00+00:00",
    )


def _sources() -> list[SourceReference]:
    return [
        SourceReference(
            source_id="S1",
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="sample.md",
            citation="sample.md > Retrieval",
            text="Hybrid retrieval combines dense and sparse retrieval.",
            source_type="markdown",
            page_start=None,
            page_end=None,
            heading_path=["Retrieval"],
            dense_score=0.9,
            sparse_score=0.8,
            fused_score=0.85,
        )
    ]
