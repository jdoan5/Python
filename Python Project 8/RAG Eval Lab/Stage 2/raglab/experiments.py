"""The retrieval configurations scored against the golden set.

A configuration is three independent switches: where chunks are cut, how
they are scored, and whether each chunk carries its heading breadcrumb.
Stage 2 evaluates the full 3 x 2 x 2 factorial, so each switch's effect can be
read off by comparing configurations that differ in that switch alone.
`BASELINE` is RAG Mini Chat exactly as it shipped: paragraph chunks, TF-IDF,
no breadcrumb. It is also one cell of the factorial.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglab.chunking import CHUNKERS, Chunk, chunk_corpus, chunker_config, get_chunker
from raglab.corpus import Document
from raglab.retrieval import RETRIEVERS, Retriever, build_retriever, retriever_config


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

    @property
    def cell(self) -> tuple[str, str, bool]:
        """Its position in the factorial, ignoring the label."""
        return (self.chunker, self.retriever, self.with_context)

    def manifest(self) -> dict:
        """Every parameter that decides this configuration's rankings."""
        return {
            "name": self.name,
            "chunker": chunker_config(get_chunker(self.chunker)),
            "retriever": retriever_config(self.retriever),
            "breadcrumb": self.with_context,
        }


BASELINE = RetrievalConfig("paragraph", "tfidf", with_context=False, label="baseline (RAG Mini Chat)")

# Stage 1's side-by-side view: the baseline plus every strategy x scorer, breadcrumb on.
GRID: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever) for chunker in CHUNKERS for retriever in sorted(RETRIEVERS)
)

# Stage 2's evaluation: all 12 cells. (paragraph, tfidf, no breadcrumb) *is* the baseline.
FACTORIAL: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever, with_context)
    for chunker in CHUNKERS
    for retriever in sorted(RETRIEVERS)
    for with_context in (True, False)
    if (chunker, retriever, with_context) != BASELINE.cell
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
