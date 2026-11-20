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


# --- Stage 3: stand-ins for the neural models ----------------------------------------------------

REAL_MODELS_LOCK = Path(__file__).resolve().parents[2] / "models.lock.json"


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(w) > 2]


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts that share words point the same way.

    Records every (instruction, text) it embeds, so tests can check what was asked and how often.
    """

    dim = 64

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or {"model": "fake", "dim": self.dim}
        self.calls: list[tuple[str, str]] = []

    def embed(self, texts, instruction):
        import hashlib

        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            self.calls.append((instruction, text))
            for w in _words(text):
                out[i, int(hashlib.sha256(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            out[i] = out[i] / norm if norm else out[i]
            if not norm:
                out[i, 0] = 1.0
        return out


class FakeScorer:
    """Scores a pair by the words it shares; records every pair it scores."""

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or {"model": "fake-reranker"}
        self.calls: list[tuple[str, str]] = []

    def score(self, query, texts):
        import numpy as np

        q = set(_words(query))
        self.calls.extend((query, t) for t in texts)
        return np.array([float(len(q & set(_words(t)))) for t in texts], dtype=np.float32)


@pytest.fixture(autouse=True)
def _fake_models_unless_marked(request, monkeypatch):
    """Every test runs on FakeEmbedder/FakeScorer unless it is marked `models`.

    So the unit tests never load PyTorch weights, never depend on a download,
    and stay fast. Tests marked `models` use the pinned models themselves.
    """
    if request.node.get_closest_marker("models"):
        return
    from raglab import neural
    from raglab.models import ModelLock

    monkeypatch.setattr(neural, "model_lock", lambda: ModelLock.load(REAL_MODELS_LOCK))
    monkeypatch.setattr(neural, "default_embedder", lambda: neural.CachedEmbedder(FakeEmbedder(), None))
    monkeypatch.setattr(neural, "default_scorer", lambda: neural.CachedScorer(FakeScorer(), None))


@pytest.fixture(scope="session")
def real_models():
    """The pinned models, verified against models.lock.json.

    Skips only when they genuinely aren't downloaded. A lock that doesn't
    match is a failure. Set RAGLAB_REQUIRE_MODELS=1 to make absence fail too.
    """
    from raglab.models import ModelLock, local_dir, require

    lock = ModelLock.load(REAL_MODELS_LOCK)
    if any(local_dir(lock[role]) is None for role in ("embedder", "reranker")):
        reason = "pinned models not downloaded — run `python -m raglab models fetch`"
        if os.environ.get("RAGLAB_REQUIRE_MODELS") == "1":
            pytest.fail(f"{reason} (RAGLAB_REQUIRE_MODELS=1)")
        pytest.skip(reason)
    return {role: require(lock[role]) for role in ("embedder", "reranker")}
