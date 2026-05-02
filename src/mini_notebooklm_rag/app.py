"""Console entrypoint for the scaffold-only Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    """Launch the Streamlit UI shell for `uv run app`."""
    streamlit_app = Path(__file__).with_name("streamlit_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.headless=true",
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
