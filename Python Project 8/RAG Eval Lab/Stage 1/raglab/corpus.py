"""Corpus: fetch a pinned copy of the FastAPI docs, verify it, load it.

Every number a later stage reports (recall@k, faithfulness, cost) is only
meaningful against a known corpus, so the corpus is pinned like a dependency.
`corpus.lock.json` records the upstream tag, the commit it resolved to, and a
SHA-256 per file. Fetching refuses to continue if the tag has been moved to a
different commit, and every load re-verifies the on-disk copy against the
lock, so a corpus that has drifted can never be used silently.

The docs are third-party content (FastAPI, MIT licence). They are fetched into
a git-ignored cache rather than committed, only ever parsed as text, and
treated as untrusted: nothing in the upstream tree — a symlink, a `..` in an
include path, an odd ref — can make the fetch read or write outside the corpus.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath

from raglab.markdown import CleanResult, Heading, clean_markdown, extract_title

FASTAPI_REPO = "https://github.com/fastapi/fastapi.git"
DOCS_SUBDIR = "docs/en/docs"
SRC_SUBDIR = "docs_src"
GIT_TIMEOUT_SECONDS = 300
MAX_FILE_BYTES = 5_000_000  # the largest real page is ~0.75 MB; anything bigger is not documentation
# Byte-identical checkouts on every OS: Git for Windows defaults to autocrlf=true,
# which would rewrite line endings and change every file's hash.
_GIT_CONFIG = ("-c", "core.autocrlf=false", "-c", "core.eol=lf")
# Files an OS drops into folders it displays; never part of the corpus.
_JUNK_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
# Release tags only. Excludes refs/pull/... (anyone's PR) and anything starting
# with "-", which git would parse as an option rather than a ref.
_TAG_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,99}$")

# Pinned but deliberately not indexed. Each is upstream content that is not
# documentation a user would ask questions of, and the first would distort
# every term statistic BM25 and TF-IDF compute.
EXCLUDED_DOCS: dict[str, str] = {
    "release-notes.md": "changelog of PR titles — 48% of the documentation by size; it would dominate term statistics",
    "_llm-test.md": "fixture for FastAPI's translation LLM, not documentation",
    "agent-banner.md": "banner snippet included into other pages",
    "translation-banner.md": "banner snippet included into other pages",
    "fastapi-people.md": "rendered from YAML at build time; the Markdown holds only template code",
    "external-links.md": "rendered from YAML at build time; the Markdown holds only template code",
}

_INCLUDE_REF_RE = re.compile(r"\.\./\.\./docs_src/([^\s*!]+)")
_LOCK_REQUIRED = ("repo", "ref", "commit", "fingerprint", "files")
_LOCK_OPTIONAL = ("missing_includes",)


class CorpusError(Exception):
    """The corpus is missing, unpinned, untrusted, or does not match its lock."""


def validate_ref(ref: object) -> str:
    if not isinstance(ref, str) or not _TAG_RE.match(ref):
        raise CorpusError(
            f"{ref!r} is not a release tag name (letters, digits, '.', '_', '+', '-'; "
            "starting with a letter or digit), e.g. 0.165.0"
        )
    return ref


def project_root() -> Path:
    """The `RAG Eval Lab` folder every stage shares (lock + data cache)."""
    override = os.environ.get("RAGLAB_HOME")
    if override:
        return Path(override).expanduser().resolve()
    # raglab/ -> Stage N/ -> RAG Eval Lab/
    return Path(__file__).resolve().parents[2]


def default_lock_path() -> Path:
    return project_root() / "corpus.lock.json"


def default_corpus_dir(ref: str) -> Path:
    return project_root() / "data" / "corpus" / f"fastapi-{validate_ref(ref)}"


def fingerprint_of(files: dict[str, str]) -> str:
    """One hash for the whole corpus: order-independent, content-sensitive."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode("utf-8"))
    return h.hexdigest()


@dataclass
class CorpusLock:
    repo: str
    ref: str
    commit: str
    fingerprint: str
    files: dict[str, str] = field(default_factory=dict)  # relative path -> sha256
    missing_includes: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def save(self, path: Path) -> None:
        data = asdict(self)
        data["file_count"] = self.file_count
        # newline="\n": the lock is committed and must be byte-identical on every OS.
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> "CorpusLock":
        if not path.is_file():
            raise CorpusError(
                f"No corpus lock at {path}. Run `python -m raglab corpus fetch --ref <tag>` first."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise CorpusError(f"{path.name} is unreadable or not valid JSON ({e}). Restore it from git.") from e
        if not isinstance(data, dict):
            raise CorpusError(f"{path.name} must hold a JSON object. Restore it from git.")
        declared_count = data.pop("file_count", None)
        missing = [k for k in _LOCK_REQUIRED if k not in data]
        unknown = sorted(set(data) - set(_LOCK_REQUIRED) - set(_LOCK_OPTIONAL))
        if missing or unknown:
            raise CorpusError(f"{path.name} is malformed (missing {missing}, unexpected {unknown}). Restore it from git.")
        files = data["files"]
        if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
            raise CorpusError(f"{path.name}: 'files' must map relative paths to SHA-256 strings.")
        validate_ref(data["ref"])
        if not isinstance(data["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", data["commit"]):
            raise CorpusError(f"{path.name}: 'commit' must be a full git commit SHA.")
        lock = cls(**data)
        if lock.fingerprint != fingerprint_of(lock.files) or declared_count not in (None, lock.file_count):
            raise CorpusError(
                f"{path.name} is internally inconsistent — its fingerprint or file count does not "
                "match its own file list, so it was edited by hand or corrupted. Restore it from git."
            )
        return lock


def hash_tree(root: Path) -> dict[str, str]:
    files = {}
    for p in sorted(root.rglob("*")):
        if p.name in _JUNK_FILES or not p.is_file():
            continue
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError as e:
            raise CorpusError(f"Cannot read {p}: {e}") from e
        # as_posix(): the lock must read the same on macOS, Linux and Windows.
        files[p.relative_to(root).as_posix()] = digest
    return files


def _git(args: list[str], cwd: Path) -> str:
    if shutil.which("git") is None:
        raise CorpusError("`git` is not installed or not on PATH — it is needed to fetch the corpus.")
    try:
        result = subprocess.run(
            ["git", *_GIT_CONFIG, *args], cwd=cwd, check=True, capture_output=True, text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as e:
        raise CorpusError(f"git {' '.join(args)} failed: {e.stderr.strip() or e}") from e
    except subprocess.TimeoutExpired as e:
        raise CorpusError(f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS}s") from e
    return result.stdout.strip()


def _checked_dir(work: Path, subdir: str) -> Path | None:
    """`work/subdir` if it is a real directory reached without any symlink."""
    path = work / subdir
    if not path.exists() and not path.is_symlink():
        return None
    if path.resolve() != work.resolve() / subdir or not path.is_dir():
        raise CorpusError(f"Upstream {subdir}/ is (or sits under) a symlink; refusing to build a corpus from it.")
    return path


def _reject_symlinks(root: Path) -> None:
    # os.walk(followlinks=False) lists symlinked directories without entering them.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            if (Path(dirpath) / name).is_symlink():
                rel = (Path(dirpath) / name).relative_to(root.parent)
                raise CorpusError(f"Upstream contains a symlink ({rel}); refusing to build a corpus from it.")


def _copy_regular(source: Path, target: Path, root: Path) -> None:
    # Last line of defence: whatever the path checks upstream of here missed,
    # a copy can only ever land inside the staging tree.
    if not target.resolve().is_relative_to(root.resolve()):
        raise CorpusError(f"Refusing to write outside the corpus: {target}")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise CorpusError(f"{source.name} is over {MAX_FILE_BYTES:,} bytes — not documentation; refusing.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target, follow_symlinks=False)


def _previous(dest: Path) -> Path:
    return dest.with_name(dest.name + ".previous")


def _recover_interrupted_swap(dest: Path) -> None:
    """If a swap died between its two renames, put the old corpus back."""
    if _previous(dest).is_dir() and not dest.exists():
        _previous(dest).rename(dest)


def _swap_into_place(staging: Path, dest: Path) -> None:
    """Replace `dest` with `staging` so that every failure leaves a whole corpus.

    The old corpus is renamed aside (atomic on one filesystem), the new one is
    renamed in, and only then is the old one deleted. If the second rename
    fails, the first is undone.
    """
    previous = _previous(dest)
    if previous.exists():
        shutil.rmtree(previous)
    had_old = dest.exists()
    if had_old:
        dest.rename(previous)
    try:
        staging.rename(dest)
    except BaseException:
        if had_old and not dest.exists():
            previous.rename(dest)
        raise
    if had_old:
        shutil.rmtree(previous, ignore_errors=True)


def fetch_corpus(ref: str, dest: Path, expected: CorpusLock | None = None) -> CorpusLock:
    """Download the English docs at release tag `ref`, plus the examples they include.

    A shallow, blob-filtered, sparse fetch transfers only what is needed — not
    the history, the library source, or the translated doc trees. Of
    `docs_src/`, only files a page actually includes are kept. When `expected`
    is given (re-fetching from a lock), the resolved commit and every file hash
    must match it. A failure at any step leaves any existing corpus intact.
    """
    validate_ref(ref)
    work: Path | None = None
    staging = dest.with_name(dest.name + ".staging")
    try:
        _recover_interrupted_swap(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".fetch-", dir=dest.parent))
        _git(["init", "-q"], work)
        _git(["remote", "add", "origin", FASTAPI_REPO], work)
        _git(["sparse-checkout", "set", "--no-cone", f"/{DOCS_SUBDIR}/", f"/{SRC_SUBDIR}/"], work)
        _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", f"refs/tags/{ref}"], work)
        _git(["checkout", "-q", "FETCH_HEAD"], work)
        commit = _git(["rev-parse", "HEAD"], work)
        if expected is not None and commit != expected.commit:
            raise CorpusError(
                f"Tag {ref!r} now resolves to {commit[:12]}, but the lock pins "
                f"{expected.commit[:12]}. Upstream moved the tag; refusing to mix corpora."
            )

        docs_src = _checked_dir(work, DOCS_SUBDIR)
        if docs_src is None:
            raise CorpusError(f"{DOCS_SUBDIR} not found at {ref} — wrong tag?")
        code_src = _checked_dir(work, SRC_SUBDIR)
        _reject_symlinks(docs_src)
        if code_src is not None:
            _reject_symlinks(code_src)
        if staging.exists():
            shutil.rmtree(staging)

        referenced: set[str] = set()
        for md in sorted(docs_src.rglob("*.md")):
            _copy_regular(md, staging / "docs" / md.relative_to(docs_src), staging)
            referenced.update(_INCLUDE_REF_RE.findall(md.read_text(encoding="utf-8", errors="replace")))

        missing = []
        for rel in sorted(referenced):
            pure = PurePosixPath(rel)
            # With symlinks already refused, a relative POSIX path without ".." cannot
            # leave docs_src/. "\" and ":" are refused too: on Windows they would act
            # as a separator and a drive prefix.
            if any(c in rel for c in "\0\\:") or pure.is_absolute() or ".." in pure.parts:
                missing.append(rel)
                continue
            source = code_src / rel if code_src is not None else None
            if source is None or not source.is_file():
                missing.append(rel)
                continue
            _copy_regular(source, staging / SRC_SUBDIR / rel, staging)

        files = hash_tree(staging)
        lock = CorpusLock(
            repo=FASTAPI_REPO, ref=ref, commit=commit,
            fingerprint=fingerprint_of(files), files=files, missing_includes=missing,
        )
        if expected is not None and lock.fingerprint != expected.fingerprint:
            raise CorpusError("Fetched files do not match the lock's hashes.")
        _swap_into_place(staging, dest)
        return lock
    except OSError as e:
        raise CorpusError(f"Filesystem error while fetching the corpus: {e}") from e
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)


def verify_corpus(corpus_dir: Path, lock: CorpusLock) -> list[str]:
    """Human-readable problems; an empty list means byte-identical to the lock."""
    if not corpus_dir.is_dir():
        return [f"corpus directory missing: {corpus_dir}"]
    actual = hash_tree(corpus_dir)
    problems = [f"missing: {rel}" for rel in sorted(set(lock.files) - set(actual))]
    problems += [f"unexpected: {rel}" for rel in sorted(set(actual) - set(lock.files))]
    problems += [
        f"modified: {rel}"
        for rel in sorted(set(actual) & set(lock.files))
        if actual[rel] != lock.files[rel]
    ]
    return problems


@dataclass(frozen=True)
class Section:
    """A heading's own content: from its heading line to the next heading of any level.

    Sections partition a document, which is what lets Stage 2 label gold
    answers as (doc_id, anchor) and stay valid across every chunking strategy.
    """
    anchor: str                    # "" for text before the first heading
    heading_path: tuple[str, ...]  # e.g. ("First Steps", "Interactive API docs")
    level: int
    start: int
    end: int                       # exclusive


@dataclass(frozen=True)
class Document:
    doc_id: str   # path under docs/ without ".md", e.g. "tutorial/first-steps"
    title: str
    text: str     # cleaned text — every chunk's offsets refer to this
    headings: tuple[Heading, ...]

    @cached_property
    def sections(self) -> tuple[Section, ...]:
        sections: list[Section] = []
        first = self.headings[0].offset if self.headings else len(self.text)
        if first > 0:
            sections.append(Section("", (self.title,), 0, 0, first))
        stack: list[Heading] = []
        for i, h in enumerate(self.headings):
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
            end = self.headings[i + 1].offset if i + 1 < len(self.headings) else len(self.text)
            sections.append(Section(h.anchor, tuple(x.title for x in stack), h.level, h.offset, end))
        return tuple(sections)

    @cached_property
    def _section_starts(self) -> list[int]:
        return [s.start for s in self.sections]

    def section_at(self, offset: int) -> Section:
        i = bisect.bisect_right(self._section_starts, offset) - 1
        return self.sections[max(i, 0)]


def make_include_resolver(corpus_dir: Path):
    """Map `../../docs_src/x.py` to the fetched copy, refusing to escape it.

    The docs are third-party input, so an include path is untrusted: a
    `../../docs_src/../../../etc/passwd` must not read outside the corpus, and
    a malformed path (a NUL byte, say) is unresolved rather than a crash.
    """
    src_root = (corpus_dir / SRC_SUBDIR).resolve()

    def resolve(ref: str) -> str | None:
        marker = "docs_src/"
        i = ref.find(marker)
        if i < 0 or "\0" in ref:
            return None
        try:
            target = (src_root / ref[i + len(marker):]).resolve()
            if not target.is_relative_to(src_root) or not target.is_file():
                return None
            return target.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None

    return resolve


@dataclass(frozen=True)
class LoadReport:
    documents: tuple[Document, ...]
    excluded: tuple[str, ...]
    unresolved_includes: tuple[str, ...]


def load_corpus(
    corpus_dir: Path, excluded: dict[str, str] | None = None, lock: CorpusLock | None = None
) -> LoadReport:
    """Load and clean every doc page, in a stable (sorted) order.

    Given a lock, the corpus is verified first: a stray file (an iCloud
    "index 2.md" conflict copy, say) or an edited page is an error, not a
    silent change to every downstream number.
    """
    if lock is not None:
        problems = verify_corpus(corpus_dir, lock)
        if problems:
            raise CorpusError(
                f"The corpus at {corpus_dir} does not match corpus.lock.json "
                f"({len(problems)} difference(s), e.g. {problems[0]}). "
                "Run `python -m raglab corpus fetch` to restore it."
            )
    excluded = EXCLUDED_DOCS if excluded is None else excluded
    docs_root = corpus_dir / "docs"
    if not docs_root.is_dir():
        raise CorpusError(
            f"Corpus not found at {corpus_dir}. Run `python -m raglab corpus fetch` first."
        )
    resolve = make_include_resolver(corpus_dir)
    documents, skipped, unresolved = [], [], []
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(docs_root).as_posix()
        if rel in excluded:
            skipped.append(rel)
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise CorpusError(f"Cannot read docs/{rel}: {e}") from e
        result: CleanResult = clean_markdown(raw, resolve)
        unresolved.extend(f"{rel}: {r}" for r in result.unresolved_includes)
        if not result.text.strip():
            continue
        doc_id = rel[: -len(".md")]
        documents.append(
            Document(doc_id=doc_id, title=extract_title(result, doc_id), text=result.text, headings=result.headings)
        )
    if not documents:
        raise CorpusError(f"No usable documents under {docs_root}.")
    return LoadReport(tuple(documents), tuple(skipped), tuple(unresolved))


def load_documents(corpus_dir: Path, lock: CorpusLock | None = None) -> list[Document]:
    return list(load_corpus(corpus_dir, lock=lock).documents)
