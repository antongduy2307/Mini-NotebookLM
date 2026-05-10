from __future__ import annotations

import pytest

from mini_notebooklm_rag.learning.parsing import LearningParseError, parse_learning_json


def test_parse_valid_quiz_json() -> None:
    parsed = parse_learning_json(
        '{"items":[{"question":"Q","options":["A","B","C","D"],'
        '"correct_index":0,"explanation":"E [S1]","source_markers":["[S1]"]}],'
        '"warnings":[]}'
    )

    assert parsed.data["items"][0]["question"] == "Q"
    assert parsed.warnings == []


def test_parse_valid_flashcard_json() -> None:
    parsed = parse_learning_json(
        '{"cards":[{"front":"F","back":"B [S1]","source_markers":["[S1]"]}],"warnings":[]}'
    )

    assert parsed.data["cards"][0]["front"] == "F"


def test_parse_fenced_json_with_warning() -> None:
    parsed = parse_learning_json('```json\n{"items":[],"warnings":[]}\n```')

    assert parsed.data["items"] == []
    assert parsed.warnings == ["LLM returned fenced JSON; extracted JSON content defensively."]


def test_parse_non_json_fails_actionably() -> None:
    with pytest.raises(LearningParseError, match="not valid JSON"):
        parse_learning_json("not json")
