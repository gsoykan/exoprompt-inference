"""
Debug launcher for Streamlit app.

Run this file directly in PyCharm debugger instead of using streamlit CLI.
This allows PyCharm to properly attach the debugger.

Usage:
    Right-click this file → Debug 'debug_app'
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent / "app.py"),
        "--server.port=8501",
        "--server.headless=true",  # Don't auto-open browser
    ]

    sys.exit(stcli.main())
