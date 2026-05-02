from __future__ import annotations

from mini_notebooklm_rag.ingestion.markdown_parser import parse_markdown


def test_markdown_parser_assigns_heading_paths(tmp_path) -> None:
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "Preamble\n\n# Chapter\n\nIntro text\n\n## Detail\n\n- item\n\n```python\nprint('x')\n```",
        encoding="utf-8",
    )

    parsed = parse_markdown(markdown_path, "notes.md")

    assert parsed.source_type == "markdown"
    assert parsed.blocks[0].heading_path == ["document start"]
    assert any(block.heading_path == ["Chapter"] for block in parsed.blocks)
    assert any(block.heading_path == ["Chapter", "Detail"] for block in parsed.blocks)
    assert any("print" in block.text for block in parsed.blocks)
