"""Citation formatting for retrieval results."""

from __future__ import annotations


def format_citation(
    filename: str,
    source_type: str,
    page_start: int | None = None,
    page_end: int | None = None,
    heading_path: list[str] | None = None,
) -> str:
    """Format a citation string from chunk metadata."""
    if source_type == "pdf":
        if page_start is None:
            return filename
        if page_end is None or page_end == page_start:
            return f"{filename}, p. {page_start}"
        return f"{filename}, pp. {page_start}-{page_end}"

    if source_type == "markdown":
        path = heading_path or ["document start"]
        clean_path = [part for part in path if part]
        return f"{filename} > {' > '.join(clean_path or ['document start'])}"

    return filename
