"""CLI launcher for the Streamlit dashboard.

Entry point: ``agent-memory-dashboard``
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Dashboard requires streamlit.  "
            "Install with:\n\n"
            "    pip install agent-memory-sdk[dashboard]\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        # Older streamlit versions
        from streamlit import cli as stcli  # type: ignore[no-redef]

    app_path = str(Path(__file__).parent / "app.py")
    sys.argv = ["streamlit", "run", app_path, "--"] + sys.argv[1:]
    sys.exit(stcli.main())
