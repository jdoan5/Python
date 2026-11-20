from __future__ import annotations

import os
from pathlib import Path

import pytest

from raglab.corpus import CorpusLock, Document, default_corpus_dir, default_lock_path, load_corpus
from raglab.markdown import clean_markdown, extract_title

FIRST_STEPS = """\
# First Steps { #first-steps }

The simplest FastAPI file could look like this:

{* ../../docs_src/first/app.py *}

/// tip | Editor support

Use the official extension.

///

## Interactive API docs { #interactive-api-docs }

Now go to the docs page to see Swagger UI.

## Raise errors { #raise-errors }

Use `HTTPException` to return a 404 status code when an item is not found.
"""


def make_doc(markdown: str, doc_id: str = "page", resolve=None) -> Document:
    result = clean_markdown(markdown, resolve)
    return Document(doc_id=doc_id, title=extract_title(result, doc_id), text=result.text, headings=result.headings)


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    """A three-page corpus in the fetched layout: docs/ plus the docs_src/ they include."""
    root = tmp_path / "corpus"
    (root / "docs" / "tutorial").mkdir(parents=True)
    (root / "docs_src" / "first").mkdir(parents=True)
    (root / "docs_src" / "first" / "app.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/')\ndef root():\n    return {'hello': 'world'}\n",
        encoding="utf-8",
    )
    (root / "docs" / "index.md").write_text("# Home { #home }\n\nWelcome to the FastAPI docs.\n", encoding="utf-8")
    (root / "docs" / "tutorial" / "first-steps.md").write_text(FIRST_STEPS, encoding="utf-8")
    (root / "docs" / "release-notes.md").write_text("# Release Notes\n\n## 0.1.0\n\n* Fix things.\n", encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def real_docs() -> list[Document]:
    """The fetched FastAPI corpus, verified against its lock.

    Skips only when the corpus genuinely isn't there (a fresh checkout, since
    data/ is git-ignored). A corrupt lock, a drifted corpus or a loader crash
    is a failure, never a skip — otherwise a CI gate could pass vacuously.
    Set RAGLAB_REQUIRE_CORPUS=1 (CI does, after fetching) to make absence fail too.
    """
    lock_path = default_lock_path()
    lock = CorpusLock.load(lock_path) if lock_path.is_file() else None
    corpus_dir = default_corpus_dir(lock.ref) if lock else None
    if lock is None or not corpus_dir.is_dir():
        reason = "real corpus not fetched — run `python -m raglab corpus fetch`"
        if os.environ.get("RAGLAB_REQUIRE_CORPUS") == "1":
            pytest.fail(f"{reason} (RAGLAB_REQUIRE_CORPUS=1)")
        pytest.skip(reason)
    return list(load_corpus(corpus_dir, lock=lock).documents)
