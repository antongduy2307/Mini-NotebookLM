"""Defensive JSON parsing for learning artifact LLM output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class LearningParseError(ValueError):
    """Raised when learning artifact JSON cannot be parsed."""


@dataclass(frozen=True)
class ParsedLearningPayload:
    """Parsed JSON payload plus non-fatal warnings."""

    data: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


_FENCED_JSON_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.I)


def parse_learning_json(text: str) -> ParsedLearningPayload:
    """Parse strict JSON, with a warning when fenced JSON is defensively extracted."""
    raw = text.strip()
    warnings: list[str] = []
    fenced = _FENCED_JSON_RE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
        warnings.append("LLM returned fenced JSON; extracted JSON content defensively.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LearningParseError(f"Learning artifact response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise LearningParseError("Learning artifact response must be a JSON object.")
    return ParsedLearningPayload(data=data, warnings=warnings)
