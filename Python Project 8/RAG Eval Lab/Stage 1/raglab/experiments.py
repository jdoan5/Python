"""The retrieval configurations Stage 2 will score against the golden set.

A configuration is three independent switches — where chunks are cut, how
they are scored, and whether each chunk carries its heading breadcrumb — so
an effect can be attributed to one switch at a time. `BASELINE` is RAG Mini
Chat exactly as it shipped: paragraph chunks, TF-IDF, no breadcrumb.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglab.chunking import CHUNKERS, Chunk, chunk_corpus, get_chunker
from raglab.corpus import Document
from raglab.retrieval import RETRIEVERS, Retriever, build_retriever


@dataclass(frozen=True)
class RetrievalConfig:
    chunker: str
    retriever: str
    with_context: bool = True
    label: str = ""

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"{self.chunker}+{self.retriever}" + ("" if self.with_context else " (no breadcrumb)")


BASELINE = RetrievalConfig("paragraph", "tfidf", with_context=False, label="baseline (RAG Mini Chat)")

GRID: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever) for chunker in CHUNKERS for retriever in sorted(RETRIEVERS)
)


def build_retrievers(
    docs: list[Document], configs: tuple[RetrievalConfig, ...] = GRID
) -> dict[str, Retriever]:
    """One retriever per config; each chunking strategy is computed only once."""
    chunk_cache: dict[str, list[Chunk]] = {}
    retrievers: dict[str, Retriever] = {}
    for config in configs:
        if config.chunker not in chunk_cache:
            chunk_cache[config.chunker] = chunk_corpus(docs, get_chunker(config.chunker))
        retrievers[config.name] = build_retriever(
            config.retriever, chunk_cache[config.chunker], with_context=config.with_context
        )
    return retrievers
