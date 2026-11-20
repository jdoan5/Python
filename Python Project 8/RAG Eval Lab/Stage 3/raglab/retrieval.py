"""The lexical retrievers, and the registry of every retriever Stage 3 measures.

Stage 3 adds three neural retrievers (dense, hybrid, hybrid+rerank) in
neural.py. They are imported only when one is built, so the lexical path never
loads PyTorch. The two lexical ones live here:

- tfidf: RAG Mini Chat's retriever, reproduced exactly — scikit-learn's
  TfidfVectorizer(ngram_range=(1, 2), stop_words="english") scored by cosine.
- bm25:  the Okapi BM25 scorer from the Job Search Agent (same k1, b and IDF
  floor), rewritten over an inverted index so it scales to thousands of chunks.

Both use scikit-learn's analyzer (token pattern, lowercasing, English stop
words), so the comparison isolates the scoring function — the one remaining
difference being the baseline's bigrams, kept for fidelity to RAG Mini Chat.
Ties break by corpus order, so every ranking is reproducible run to run.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import ClassVar, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from raglab.chunking import Chunk

# Okapi BM25 constants, unchanged from the Job Search Agent.
K1 = 1.5
B = 0.75
IDF_FLOOR = 0.01  # BM25+-style: keeps IDF positive for terms in most chunks


class RetrievalError(Exception):
    """The retriever cannot be built (e.g. no chunks, or nothing indexable)."""


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk: Chunk


class Retriever(Protocol):
    name: ClassVar[str]

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


def _rank(scores: np.ndarray, chunks: Sequence[Chunk], k: int) -> list[Hit]:
    """Top-k positive scores; ties go to the earlier chunk so rankings are stable."""
    candidates = np.flatnonzero(scores > 0)
    if candidates.size == 0 or k <= 0:
        return []
    order = candidates[np.lexsort((candidates, -scores[candidates]))][:k]
    return [Hit(rank=i + 1, score=float(scores[j]), chunk=chunks[j]) for i, j in enumerate(order)]


class TfidfRetriever:
    name: ClassVar[str] = "tfidf"

    def __init__(self, chunks: Sequence[Chunk], with_context: bool = True) -> None:
        if not chunks:
            raise RetrievalError("no chunks to index")
        self.chunks = list(chunks)
        self.with_context = with_context
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            self.matrix = self.vectorizer.fit_transform([c.index_text(with_context) for c in self.chunks])
        except ValueError as e:  # scikit-learn's "empty vocabulary"
            raise RetrievalError(f"nothing indexable: {e}") from e

    def search(self, query: str, k: int = 10) -> list[Hit]:
        query_vec = self.vectorizer.transform([query])
        # Rows and query are L2-normalised, so the dot product is the cosine.
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        return _rank(scores, self.chunks, k)


class Bm25Retriever:
    name: ClassVar[str] = "bm25"

    def __init__(
        self, chunks: Sequence[Chunk], with_context: bool = True, k1: float = K1, b: float = B
    ) -> None:
        if not chunks:
            raise RetrievalError("no chunks to index")
        self.chunks = list(chunks)
        self.with_context = with_context
        self.k1, self.b = k1, b
        self.analyzer = CountVectorizer(stop_words="english").build_analyzer()

        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        lengths = np.zeros(len(self.chunks))
        for i, chunk in enumerate(self.chunks):
            terms = self.analyzer(chunk.index_text(with_context))
            lengths[i] = len(terms)
            for term, tf in Counter(terms).items():
                postings[term][0].append(i)
                postings[term][1].append(tf)
        if not postings:
            raise RetrievalError("nothing indexable: every chunk is stop words or punctuation")

        n = len(self.chunks)
        self._postings = {
            t: (np.asarray(idx), np.asarray(tf, dtype=float)) for t, (idx, tf) in postings.items()
        }
        self._idf = {
            t: max(IDF_FLOOR, math.log((n - len(idx) + 0.5) / (len(idx) + 0.5) + 1.0))
            for t, (idx, _) in self._postings.items()
        }
        avg_len = lengths.mean() or 1.0
        self._length_norm = k1 * (1 - b + b * lengths / avg_len)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = np.zeros(len(self.chunks))
        # A repeated query term counts each time, as in the Job Search Agent.
        for term in self.analyzer(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            idx, tf = posting
            scores[idx] += self._idf[term] * tf * (self.k1 + 1) / (tf + self._length_norm[idx])
        return _rank(scores, self.chunks, k)


RETRIEVERS: dict[str, type] = {r.name: r for r in (Bm25Retriever, TfidfRetriever)}
NEURAL = ("dense", "hybrid", "hybrid+rerank")  # built by raglab.neural
# Every retriever, in report order: Stage 2's two (as Stage 2 ordered them), then Stage 3's three.
RETRIEVER_NAMES: tuple[str, ...] = (*sorted(RETRIEVERS), *NEURAL)


def _unknown(name: str) -> ValueError:
    return ValueError(f"unknown retriever {name!r}; choose from {', '.join(RETRIEVER_NAMES)}")


def retriever_config(name: str) -> dict:
    """Parameters for run manifests: everything that decides a retriever's scores."""
    if name == "bm25":
        return {"name": name, "k1": K1, "b": B, "idf_floor": IDF_FLOOR, "analyzer": "scikit-learn word, english stop words"}
    if name == "tfidf":
        return {"name": name, "ngram_range": [1, 2], "stop_words": "english", "similarity": "cosine"}
    if name in NEURAL:
        from raglab.neural import neural_config

        return neural_config(name)
    raise _unknown(name)


def build_retriever(name: str, chunks: Sequence[Chunk], with_context: bool = True) -> Retriever:
    if name in RETRIEVERS:
        return RETRIEVERS[name](chunks, with_context=with_context)
    if name in NEURAL:
        from raglab.neural import NEURAL_RETRIEVERS

        return NEURAL_RETRIEVERS[name](chunks, with_context=with_context)
    raise _unknown(name)
