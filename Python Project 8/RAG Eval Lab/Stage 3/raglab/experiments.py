"""The retrieval configurations scored against the golden set.

A configuration is three independent switches: where chunks are cut, how
they are scored, and whether each chunk carries its heading breadcrumb.
Stage 3 evaluates the full 3 x 5 x 2 factorial (Stage 2's 3 x 2 x 2 plus the
dense, hybrid and hybrid+rerank retrievers), so each switch's effect can be
read off by comparing configurations that differ in that switch alone.
`BASELINE` is RAG Mini Chat exactly as it shipped: paragraph chunks, TF-IDF,
no breadcrumb. It is also one cell of the factorial.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglab.chunking import CHUNKERS, Chunk, chunk_corpus, chunker_config, get_chunker
from raglab.corpus import Document
from raglab.retrieval import RETRIEVER_NAMES, Retriever, build_retriever, retriever_config


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

# The side-by-side view (`raglab compare`): the baseline plus every strategy x retriever, breadcrumb on.
GRID: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever) for chunker in CHUNKERS for retriever in RETRIEVER_NAMES
)

# Stage 3's evaluation: all 30 cells. (paragraph, tfidf, no breadcrumb) *is* the baseline.
FACTORIAL: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever, with_context)
    for chunker in CHUNKERS
    for retriever in RETRIEVER_NAMES
    for with_context in (True, False)
    if (chunker, retriever, with_context) != BASELINE.cell
)


def build_retrievers(
    docs: list[Document], configs: tuple[RetrievalConfig, ...] = GRID
) -> dict[str, Retriever]:
    """One retriever per config, sharing work: each chunking is computed once, and a
    hybrid reuses the bm25 and dense retrievers over the same chunks (which a reranker
    in turn reuses), so no index or embedding is built twice."""
    chunk_cache: dict[str, list[Chunk]] = {}
    built: dict[tuple[str, str, bool], Retriever] = {}

    def get(chunker: str, retriever: str, with_context: bool) -> Retriever:
        key = (chunker, retriever, with_context)
        if key not in built:
            if chunker not in chunk_cache:
                chunk_cache[chunker] = chunk_corpus(docs, get_chunker(chunker))
            chunks = chunk_cache[chunker]
            if retriever == "hybrid":
                from raglab.neural import HybridRetriever

                built[key] = HybridRetriever(chunks, with_context, lexical=get(chunker, "bm25", with_context),
                                             dense=get(chunker, "dense", with_context))
            elif retriever == "hybrid+rerank":
                from raglab.neural import RerankRetriever

                built[key] = RerankRetriever(chunks, with_context, first_stage=get(chunker, "hybrid", with_context))
            else:
                built[key] = build_retriever(retriever, chunks, with_context=with_context)
        return built[key]

    return {config.name: get(*config.cell) for config in configs}
