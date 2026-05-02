"""Hashing utilities."""

from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest for UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))
