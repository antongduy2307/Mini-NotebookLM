"""Filename normalization helpers."""

from __future__ import annotations

import re
from pathlib import Path

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str, fallback: str = "uploaded_file") -> str:
    """Return a conservative filesystem-safe filename."""
    name = Path(filename).name.strip().strip(". ")
    if not name:
        name = fallback

    sanitized = _SAFE_CHARS.sub("_", name).strip("._ ")
    if not sanitized:
        sanitized = fallback

    stem = Path(sanitized).stem
    suffix = Path(sanitized).suffix
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        sanitized = f"{stem}_{suffix}" if suffix else f"{stem}_"

    return sanitized[:180]


def normalize_workspace_name(name: str) -> str:
    """Normalize workspace names for case-insensitive uniqueness."""
    return " ".join(name.strip().casefold().split())
