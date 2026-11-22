"""Turn FastAPI's mkdocs-flavoured Markdown into clean text plus a heading map.

The cleaned text is what every downstream offset refers to: chunk spans now,
and from Stage 2 the gold-answer sections. So cleaning is deterministic,
line-based and fence-aware, and it records each heading's offset *as it emits
the line* rather than re-deriving positions afterwards.

Fence awareness is the part that matters most: inside a ```python block,
`# create the app` is a comment, not a heading, and a naive splitter would cut
a section in half there. Where Markdown's own rules decide a question (what
opens a fence, what counts as a code span) this follows CommonMark.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

IncludeResolver = Callable[[str], Optional[str]]

_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([^\s}]+)[^}]*\})?\s*$")
_INCLUDE_STAR_RE = re.compile(r"^\s*\{\*\s*(\S+)((?:\s+[a-z]+\[[^\]]*\])*)\s*\*\}\s*$")
_INCLUDE_BANG_RE = re.compile(r"^\s*\{!>?\s*(\S+?)\s*!\}\s*$")
_LINE_RANGE_RE = re.compile(r"\bln\[([^\]]*)\]")
_BLOCK_OPEN_RE = re.compile(r"^/{3,}\s*([A-Za-z][\w-]*)\s*(?:\|\s*(.*?))?\s*$")
_BLOCK_CLOSE_RE = re.compile(r"^/{3,}\s*$")
_MKDOCSTRINGS_RE = re.compile(r"^:::\s+\S")
_JINJA_STMT_LINE_RE = re.compile(r"^\s*\{%.*%\}\s*$")
_JINJA_EXPR_RE = re.compile(r"\{\{.*?\}\}")
_FRONT_MATTER_KEY_RE = re.compile(r"^[A-Za-z_][\w-]*\s*:")
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+")
_ABBR_RE = re.compile(r'<abbr\s+title="([^"]*)"\s*>(.*?)</abbr>', re.IGNORECASE)
_AUTOLINK_RE = re.compile(r"<((?:https?|ftp|mailto):[^\s<>]*)>", re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
# Tags that separate content (table cells, list items, line breaks) become a
# space; anything else (<strong>, <font>, <span>) is removed in place.
_SEPARATING_TAG_RE = re.compile(
    r"</?(?:address|article|aside|blockquote|br|dd|details|div|dl|dt|figcaption|figure|footer"
    r"|h[1-6]|header|hr|li|ol|p|pre|section|summary|table|tbody|td|tfoot|th|thead|tr|ul)\b[^>]*>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_RAW_TEXT_OPEN_RE = re.compile(r"<(style|script)\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_INNER_SPACES_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")

_LANG_BY_SUFFIX = {
    ".py": "python", ".html": "html", ".js": "javascript", ".css": "css",
    ".json": "json", ".toml": "toml", ".yml": "yaml", ".yaml": "yaml",
    ".sh": "bash", ".txt": "text", ".md": "markdown", ".jinja": "jinja",
}


@dataclass(frozen=True)
class Heading:
    level: int
    title: str   # display title, inline markup removed
    anchor: str  # the author's { #anchor } when present, else an mkdocs-style slug
    offset: int  # where the heading line starts in the cleaned text


@dataclass(frozen=True)
class CleanResult:
    text: str
    headings: tuple[Heading, ...]
    unresolved_includes: tuple[str, ...]


# ---------------------------------------------------------------- fences (shared with chunking)

def fence_open(line: str) -> tuple[str, str, int, str] | None:
    """(indent, char, length, info) if `line` opens a code fence, else None.

    Per CommonMark, a backtick fence's info string may not contain a backtick,
    so "```js``` is inline code" is prose, not a fence opening.
    """
    m = _FENCE_RE.match(line)
    if not m:
        return None
    marker, info = m.group(2), m.group(3)
    if marker[0] == "`" and "`" in info:
        return None
    return m.group(1), marker[0], len(marker), info


def fence_closes(line: str, char: str, length: int) -> bool:
    """A bare run of the same fence char, at least as long, closes the fence."""
    m = _FENCE_RE.match(line)
    return bool(m) and m.group(2)[0] == char and len(m.group(2)) >= length and not m.group(3).strip()


# ---------------------------------------------------------------- small helpers

def slugify(title: str) -> str:
    """Python-Markdown's default toc slug, so fallback anchors match the site."""
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def display_title(raw_title: str) -> str:
    return re.sub(r"\s+", " ", raw_title.replace("`", "").replace("**", "")).strip()


def _code_spans(line: str) -> list[tuple[int, int]]:
    return [m.span() for m in _INLINE_CODE_RE.finditer(line)]


def _in_spans(i: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= i < b for a, b in spans)


def _find_outside_code(line: str, token: str) -> int:
    spans = _code_spans(line)
    pos = line.find(token)
    while pos != -1 and _in_spans(pos, spans):
        pos = line.find(token, pos + 1)
    return pos


def _unescape(text: str) -> str:
    """Decode only properly terminated entities.

    `html.unescape` also decodes legacy forms without a semicolon, which would
    turn a URL's `&copy=2` into `©=2`.
    """
    return _ENTITY_RE.sub(lambda m: html.unescape(m.group(0)), text)


def _reduce_links(line: str) -> str:
    """`[text](url)` and `![alt](src)` -> their text, unless inside a code span.

    Runs on the whole line before prose and code are separated, because link
    text can itself contain code — [`Depends`](...) — which would otherwise
    split the link across segments and leave its URL in the text.
    """
    spans = _code_spans(line)
    out, last = [], 0
    for m in _MD_LINK_RE.finditer(line):
        if _in_spans(m.start(), spans):
            continue
        out.append(line[last:m.start()])
        out.append(m.group(1))
        last = m.end()
    out.append(line[last:])
    return "".join(out)


def _clean_prose_segment(text: str) -> str:
    text = _ABBR_RE.sub(lambda m: f"{m.group(2)} ({m.group(1)})", text)
    text = _AUTOLINK_RE.sub(r"\1", text)
    text = _IMG_TAG_RE.sub("", text)
    text = _SEPARATING_TAG_RE.sub(" ", text)
    text = _TAG_RE.sub("", text)
    text = _JINJA_EXPR_RE.sub("", text)
    return _INNER_SPACES_RE.sub(" ", _unescape(text))


def _clean_prose(line: str) -> str:
    """Clean a prose line, leaving inline `code` spans untouched.

    Prose such as "an HTML `<form>`" must keep its `<form>`: the tag stripping
    meant for decorative HTML would otherwise eat it.
    """
    indent = line[: len(line) - len(line.lstrip())]
    line = _reduce_links(line)
    out, last = [], 0
    for a, b in _code_spans(line):
        out.append(_clean_prose_segment(line[last:a]))
        out.append(line[a:b])
        last = b
    out.append(_clean_prose_segment(line[last:]))
    # Keep the author's indentation (it is structure), but not whitespace that a
    # removed leading tag left behind.
    return indent + "".join(out).lstrip()


def _apply_line_ranges(content: str, spec: str) -> str:
    """mkdocs `ln[a:b,c:d]`: 1-based inclusive ranges, joined by '...'."""
    lines = content.splitlines()
    picked: list[str] = []
    try:
        for part in (p.strip() for p in spec.split(",")):
            if not part:
                continue
            if ":" in part:
                a, b = part.split(":", 1)
                start = int(a) if a.strip() else 1
                end = int(b) if b.strip() else len(lines)
            else:
                start = end = int(part)
            if picked:
                picked.append("...")
            picked.extend(lines[max(start, 1) - 1 : end])
    except ValueError:
        return content  # malformed spec: showing the whole file beats dropping it
    return "\n".join(picked)


def _fence_lang(info: str) -> str:
    token = info.strip().split(None, 1)[0] if info.strip() else ""
    if not token or "=" in token or token.startswith("{"):
        return ""
    return token.lower()


def _strip_front_matter(raw: str) -> str:
    """Drop YAML front matter — but only real front matter.

    A document may also open with `---` as a horizontal rule; it is front
    matter only if the next line is a YAML `key:`.
    """
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        first_line = raw[4:].split("\n", 1)[0]
        if end != -1 and _FRONT_MATTER_KEY_RE.match(first_line):
            return raw[end + len("\n---\n"):]
    return raw


def _unique(base: str, used: set[str]) -> str:
    candidate, n = base, 0
    while candidate in used:
        n += 1
        candidate = f"{base}_{n}"
    used.add(candidate)
    return candidate


class _Emitter:
    """Accumulates output lines while tracking each line's start offset."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.length = 0  # chars emitted so far, counting the joining newlines

    def emit(self, line: str) -> int:
        line = line.rstrip()
        if not line and (not self.lines or not self.lines[-1]):
            return self.length  # no leading blanks, no runs of blank lines
        offset = self.length + (1 if self.lines else 0)
        self.length = offset + len(line)
        self.lines.append(line)
        return offset

    def text(self) -> str:
        return "\n".join(self.lines).rstrip()


# ---------------------------------------------------------------- the cleaner

def clean_markdown(raw: str, resolve_include: IncludeResolver | None = None) -> CleanResult:
    out = _Emitter()
    pending: list[tuple[int, str, str | None, int]] = []  # level, title, authored anchor, offset
    unresolved: list[str] = []

    fence: tuple[str, int] | None = None  # (char, length) while inside a code fence
    fence_lang = ""
    in_comment = False
    raw_text_tag: str | None = None  # inside <style>/<script>: content is never prose
    skipping_directive = False

    def emit_include(ref: str, wrap: bool, options: str = "") -> None:
        content = resolve_include(ref) if resolve_include else None
        if content is None:
            unresolved.append(ref)
            return
        ranges = _LINE_RANGE_RE.search(options)
        if ranges:
            content = _apply_line_ranges(content, ranges.group(1))
        if wrap:
            name = ref.rsplit("/", 1)[-1]
            suffix = name[name.rfind("."):].lower() if "." in name else ""
            out.emit("```" + _LANG_BY_SUFFIX.get(suffix, ""))
        for code_line in content.rstrip("\n").split("\n"):
            out.emit(code_line)
        if wrap:
            out.emit("```")

    lines = _strip_front_matter(raw.replace("\r\n", "\n")).split("\n")
    for line in lines:
        # ---------------- inside a code fence: emit verbatim ----------------
        if fence is not None:
            if fence_closes(line, *fence):
                indent = _FENCE_RE.match(line).group(1)
                out.emit(indent + fence[0] * fence[1])
                fence = None
                continue
            inc = _INCLUDE_BANG_RE.match(line)
            if inc:
                emit_include(inc.group(1), wrap=False)
                continue
            if fence_lang == "console":
                # Terminal transcripts are styled with <font>/<span>; keep the text.
                line = _unescape(_TAG_RE.sub("", line))
            out.emit(line)
            continue

        # ---------------- raw-text blocks and HTML comments ----------------
        removed = False
        if raw_text_tag is not None:
            close = line.lower().find(f"</{raw_text_tag}>")
            if close == -1:
                continue
            line, raw_text_tag, removed = line[close + len(raw_text_tag) + 3:], None, True
        if in_comment:
            close = line.find("-->")
            if close == -1:
                continue
            line, in_comment, removed = line[close + 3:], False, True
        while (start := _find_outside_code(line, "<!--")) != -1:
            removed = True
            close = line.find("-->", start + 4)
            if close == -1:
                line, in_comment = line[:start], True
                break
            line = line[:start] + line[close + 3:]
        m = _RAW_TEXT_OPEN_RE.search(line)
        if m and not _in_spans(m.start(), _code_spans(line)):
            removed = True
            tag = m.group(1).lower()
            close = line.lower().find(f"</{tag}>", m.end())
            if close == -1:
                line, raw_text_tag = line[:m.start()], tag
            else:
                line = line[:m.start()] + line[close + len(tag) + 3:]
        if removed and not line.strip():
            # A line that held only a comment or a <style> must not become a blank
            # line: that would invent a paragraph break, and paragraphs are chunk boundaries.
            continue

        # ---------------- mkdocstrings `::: module.Name` + its options ----------------
        if skipping_directive:
            if not line.strip() or line[:1] in (" ", "\t"):
                continue
            skipping_directive = False
        if _MKDOCSTRINGS_RE.match(line):
            skipping_directive = True
            continue

        # ---------------- opening code fence ----------------
        opened = fence_open(line)
        if opened:
            indent, char, length, info = opened
            fence, fence_lang = (char, length), _fence_lang(info)
            out.emit(indent + char * length + fence_lang)
            continue

        # ---------------- include directives ----------------
        inc = _INCLUDE_STAR_RE.match(line)
        if inc:
            emit_include(inc.group(1), wrap=True, options=inc.group(2))
            continue
        inc = _INCLUDE_BANG_RE.match(line)
        if inc:
            emit_include(inc.group(1), wrap=False)
            continue

        # ---------------- headings ----------------
        h = _HEADING_RE.match(line)
        if h:
            raw_title = _clean_prose(h.group(2)).strip()
            out.emit("")  # headings always start a new block
            offset = out.emit(f"{h.group(1)} {raw_title}")
            pending.append((len(h.group(1)), display_title(raw_title), h.group(3), offset))
            continue

        # ---------------- admonitions / tabs: `/// tip | Title` ... `///` ----------------
        b = _BLOCK_OPEN_RE.match(line)
        if b:
            kind = b.group(1).capitalize()
            label = display_title(_clean_prose(b.group(2) or ""))
            out.emit("")
            out.emit(f"**{kind}: {label}**" if label else f"**{kind}**")
            continue
        if _BLOCK_CLOSE_RE.match(line):
            out.emit("")
            continue

        # ---------------- templating and link definitions ----------------
        if _JINJA_STMT_LINE_RE.match(line) or _REF_DEF_RE.match(line):
            continue

        out.emit(_clean_prose(line))

    # Anchors are resolved after the pass. An author's { #anchor } is kept
    # verbatim — Stage 2's gold labels reference these — so it is reserved
    # first; generated slugs then take a numbered suffix around anything taken.
    used: set[str] = set()
    anchors: list[str] = [""] * len(pending)
    for i, (_, _, authored, _) in enumerate(pending):
        if authored:
            anchors[i] = _unique(authored, used)
    for i, (_, title, authored, _) in enumerate(pending):
        if not authored:
            anchors[i] = _unique(slugify(title) or "section", used)
    headings = tuple(
        Heading(level=level, title=title, anchor=anchors[i], offset=offset)
        for i, (level, title, _, offset) in enumerate(pending)
    )
    return CleanResult(text=out.text(), headings=headings, unresolved_includes=tuple(unresolved))


def extract_title(result: CleanResult, fallback: str) -> str:
    for h in result.headings:
        if h.level == 1:
            return h.title
    return result.headings[0].title if result.headings else fallback
