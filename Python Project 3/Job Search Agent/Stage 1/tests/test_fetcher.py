from __future__ import annotations

import socket

import httpx
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


def test_fetch_malformed_url_raises_fetch_error() -> None:
    # Bad ports / invalid hosts raise outside httpx.HTTPError — they must
    # still surface as FetchError, never a raw ValueError/InvalidURL.
    with pytest.raises(FetchError):
        fetch_posting("http://example.com:99999/")  # port out of range
    with pytest.raises(FetchError):
        fetch_posting("http://exa mple.com/job")    # space in host


def test_fetch_connects_to_the_address_it_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    # DNS rebinding: a hostile resolver answers the SSRF check with a public
    # address and the following lookup with the cloud metadata endpoint. The
    # fetch must use the answer it actually vetted, and must still present the
    # hostname in Host/SNI so TLS verification of real postings keeps working.
    answers = [
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))],
    ]

    def rebinding_getaddrinfo(host, *args, **kwargs):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    seen = {}

    def handle_request(self, request):
        seen["connect_host"] = request.url.host
        seen["host_header"] = request.headers.get("host")
        seen["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, html="<p>" + "Senior Python Engineer. " * 40 + "</p>",
                              request=request)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)

    fetch_posting("https://jobs.example.com/posting/1")

    assert seen["connect_host"] == "93.184.216.34"
    assert seen["host_header"] == "jobs.example.com"
    assert seen["sni_hostname"] == "jobs.example.com"


def test_html_to_text_drops_script_and_style_bodies() -> None:
    raw = (
        "<p>Real job text</p>"
        "<SCRIPT TYPE='text/javascript'>alsoHidden()</SCRIPT>"
        "<style>body{color:red}</style>"
    )
    text = html_to_text(raw)
    assert "Real job text" in text
    for leaked in ("alsoHidden()", "color:red"):
        assert leaked not in text


def test_html_to_text_handles_angle_bracket_inside_attribute() -> None:
    # The old `<[^>]+>` strip ended the tag at the '>' inside the attribute
    # value, spilling `b">` into the prompt. A parser reads the quoting.
    raw = '<div title="a > b">Salary &gt; 100k</div>'
    text = html_to_text(raw)
    assert text == "Salary > 100k"


def test_html_to_text_ignores_comments_and_selfclosing_breaks() -> None:
    raw = "<p>Alpha</p><!-- hidden note --><br/><p>Beta</p>"
    text = html_to_text(raw)
    assert "Alpha" in text and "Beta" in text
    assert "hidden note" not in text
