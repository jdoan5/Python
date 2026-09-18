"""Fetch a job posting URL and reduce it to plain text.

Security posture: this tool fetches URLs the *user* pastes, but the URL may
point anywhere (shortlinks, redirects), so we still refuse private/internal
destinations (SSRF guard) and validate every redirect hop — not just the
first URL.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

MAX_FETCH_BYTES = 800_000
MAX_TEXT_CHARS = 40_000
MAX_REDIRECTS = 5
TIMEOUT_SECONDS = 15.0

TRUNCATION_MARKER = "\n\n[... posting truncated for length ...]"


class FetchError(Exception):
    """Raised when a posting cannot be fetched or yields no usable text."""


def _assert_public_host(url: str) -> str:
    """Reject URLs whose host resolves to a private/loopback/link-local address.

    Returns the vetted address. The caller must connect to *that*, not to the
    name again: httpx resolves independently, so a rebinding DNS server can
    answer this check with a public address and the connection with an internal
    one (169.254.169.254, 127.0.0.1) in the window between the two lookups.
    """
    host = urlparse(url).hostname
    if not host:
        raise FetchError(f"URL has no host: {url}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FetchError(f"Cannot resolve host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise FetchError(
                f"Refusing to fetch {host!r}: resolves to a non-public address ({ip})."
            )
    return infos[0][4][0]


class _TextExtractor(HTMLParser):
    """Collect visible text, dropping <script>/<style> contents entirely.

    A real parser rather than regexes: regex tag-stripping is defeated by
    nested, malformed or truncated markup, and a fetched page is attacker-
    influenced input whose leftovers would land in an LLM prompt. The parser
    also handles an unterminated <script> for free — it stays in CDATA mode to
    end of input, so nothing after it is ever emitted as text.
    """

    _SKIP = {"script", "style"}
    _BREAK = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # entities decoded for us
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        if tag in self._BREAK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BREAK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html_str: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_str)
    parser.close()
    text = parser.text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _read_capped(response: httpx.Response) -> str:
    """Read at most MAX_FETCH_BYTES from the wire, then decode."""
    chunks = []
    total = 0
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_FETCH_BYTES:
            break
    encoding = response.charset_encoding or "utf-8"
    return b"".join(chunks)[:MAX_FETCH_BYTES].decode(encoding, errors="ignore")


def fetch_posting(url: str) -> str:
    """Fetch a job posting and return cleaned text. Raises FetchError."""
    if not url.startswith(("http://", "https://")):
        raise FetchError(f"Refusing non-http URL: {url}")

    headers = {"User-Agent": "Mozilla/5.0 (JobSearchAgent/0.1; personal use)"}
    current = url
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False,
                          headers=headers) as client:
            for _ in range(MAX_REDIRECTS + 1):
                pinned_ip = _assert_public_host(current)
                target = httpx.URL(current)
                # Dial the vetted address itself, but keep the real hostname in
                # Host and SNI so certificate verification is unaffected.
                with client.stream(
                    "GET",
                    target.copy_with(host=pinned_ip),
                    headers={"Host": target.netloc.decode("ascii")},
                    extensions={"sni_hostname": target.raw_host.decode("ascii")},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError(f"Redirect with no Location header from {current}")
                        current = str(httpx.URL(current).join(location))
                        continue
                    response.raise_for_status()
                    raw = _read_capped(response)
                    break
            else:
                raise FetchError(f"Too many redirects (> {MAX_REDIRECTS}) fetching {url}")
    except httpx.HTTPStatusError as e:
        raise FetchError(
            f"HTTP {e.response.status_code} fetching {current}. "
            "Some job boards block bots — try saving the posting text to a file "
            "and passing the file path instead."
        ) from e
    except httpx.HTTPError as e:
        raise FetchError(f"Network error fetching {current}: {e}") from e
    except (httpx.InvalidURL, UnicodeError, ValueError) as e:
        # Malformed URLs (bad ports, invalid IDNA hosts) raise outside the
        # HTTPError hierarchy — normalize them into the FetchError contract.
        raise FetchError(f"Invalid URL {current!r}: {e}") from e

    text = html_to_text(raw)
    if len(text) < 200:
        raise FetchError(
            "Fetched page contained almost no text (likely a JavaScript-rendered "
            "posting). Save the posting text to a file and pass the file path instead."
        )
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + TRUNCATION_MARKER
    return text
