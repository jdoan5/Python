from __future__ import annotations

import pytest

from job_agent.fetcher import FetchError, fetch_posting, html_to_text


def test_html_to_text_strips_tags_and_scripts() -> None:
    raw = "<html><script>x()</script><h1>Engineer</h1><p>Build &amp; ship</p></html>"
    text = html_to_text(raw)
    assert "Engineer" in text
    assert "Build & ship" in text
    assert "x()" not in text


def test_html_to_text_preserves_line_structure() -> None:
    raw = "<li>Python</li><li>SQL</li>"
    text = html_to_text(raw)
    assert "Python" in text and "SQL" in text
    assert "\n" in text


def test_fetch_rejects_non_http() -> None:
    with pytest.raises(FetchError, match="Refusing"):
        fetch_posting("file:///etc/passwd")
    with pytest.raises(FetchError, match="Refusing"):
        fetch_posting("ftp://example.com/x")


def test_fetch_rejects_private_hosts() -> None:
    # SSRF guard: loopback and private ranges must be refused before any request.
    for url in (
        "http://127.0.0.1/admin",
        "http://localhost:8080/",
        "http://192.168.1.1/router",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
    ):
        with pytest.raises(FetchError, match="non-public|resolve"):
            fetch_posting(url)


def test_html_to_text_strips_unterminated_script() -> None:
    raw = "<p>Real job text</p><script>var secret = 'leaks'"
    text = html_to_text(raw)
    assert "Real job text" in text
    assert "secret" not in text
