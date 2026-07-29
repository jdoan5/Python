"""Fetch a job posting URL and reduce it to plain text.

Security posture: this tool fetches URLs the *user* pastes, but the URL may
point anywhere (shortlinks, redirects), so we still refuse private/internal
destinations (SSRF guard) and validate every redirect hop — not just the
first URL.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx

MAX_FETCH_BYTES = 800_000
MAX_TEXT_CHARS = 40_000
MAX_REDIRECTS = 5
TIMEOUT_SECONDS = 15.0

TRUNCATION_MARKER = "\n\n[... posting truncated for length ...]"


class FetchError(Exception):
    """Raised when a posting cannot be fetched or yields no usable text."""


def _assert_public_host(url: str) -> None:
    """Reject URLs whose host resolves to a private/loopback/link-local address."""
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


def html_to_text(html_str: str) -> str:
    # Paired script/style blocks first, then any unterminated tail (a truncated
    # or malformed page would otherwise leak raw JS/CSS into the model prompt).
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html_str, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(?:script|style)\b[^>]*>.*$", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<(br|p|div|li|tr|h[1-6])\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
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
                _assert_public_host(current)
                with client.stream("GET", current) as response:
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
