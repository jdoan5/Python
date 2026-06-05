"""Wrap the Streamlit app in a native macOS window via PyWebView.

Run with:
    pip install -e ".[desktop]"
    python desktop.py

This launches Streamlit headlessly on a free local port, then opens a
PyWebView window pointing at it. Closing the window kills Streamlit.
"""

from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
from contextlib import closing

import webview

WINDOW_TITLE = "AI Agents Portfolio"
WIDTH = 1400
HEIGHT = 900


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (OSError, socket.timeout):
                time.sleep(0.3)
    return False


def main() -> None:
    port = find_free_port()

    streamlit_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.headless=true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats=false",
        ]
    )

    atexit.register(lambda: streamlit_proc.terminate())

    if not wait_for_server(port):
        print(f"Streamlit failed to start on port {port}", file=sys.stderr)
        streamlit_proc.terminate()
        sys.exit(1)

    webview.create_window(
        WINDOW_TITLE,
        f"http://127.0.0.1:{port}",
        width=WIDTH,
        height=HEIGHT,
    )
    webview.start()


if __name__ == "__main__":
    main()
