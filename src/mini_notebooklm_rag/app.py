"""Console entrypoint for the Streamlit app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_streamlit_command(extra_args: list[str] | None = None) -> list[str]:
    """Build the subprocess command used by `uv run app`."""
    streamlit_app = Path(__file__).with_name("streamlit_app.py")
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        *(extra_args or []),
    ]


def main() -> None:
    """Launch the Streamlit UI in a separate process for `uv run app`."""
    extra_args = sys.argv[1:]
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    completed = subprocess.run(build_streamlit_command(extra_args), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
