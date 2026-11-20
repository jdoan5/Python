"""The retrieval configurations scored against the golden set.

A configuration is three independent switches: where chunks are cut, how
they are scored, and whether each chunk carries its heading breadcrumb.
Stage 3 evaluates the full 3 x 5 x 2 factorial (Stage 2's 3 x 2 x 2 plus the
dense, hybrid and hybrid+rerank retrievers), so each switch's effect can be
read off by comparing configurations that differ in that switch alone, plus
five extras: a bm25+rerank control and four variants (EXTRAS).
`BASELINE` is RAG Mini Chat exactly as it shipped: paragraph chunks, TF-IDF,
no breadcrumb. It is also one cell of the factorial.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglab.chunking import CHUNKERS, Chunk, chunk_corpus, chunker_config, get_chunker
from raglab.corpus import Document
from raglab.retrieval import Retriever, build_retriever, retriever_config

# The retrievers crossed with every chunker and breadcrumb setting, in report order.
FACTORIAL_RETRIEVERS = ("bm25", "tfidf", "dense", "hybrid", "hybrid+rerank")


# Named departures from a configuration's defaults, evaluated outside the factorial.
VARIANTS: dict[str, dict] = {
    "no instruction": {"instruction": ""},  # dense without the model card's query instruction
    "depth 10": {"depth": 10},              # the rerank-depth curve
    "depth 20": {"depth": 20},
    "depth 50": {"depth": 50},
}


@dataclass(frozen=True)
class RetrievalConfig:
    chunker: str
    retriever: str
    with_context: bool = True
    label: str = ""
    variant: str = ""

    def __post_init__(self) -> None:
        if self.variant and self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; choose from {', '.join(VARIANTS)}")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return (f"{self.chunker}+{self.retriever}" + ("" if self.with_context else " (no breadcrumb)")
                + (f" [{self.variant}]" if self.variant else ""))

    @property
    def cell(self) -> tuple[str, str, bool, str]:
        """Its position: chunker, retriever, breadcrumb, then the variant (empty in the factorial)."""
        return (self.chunker, self.retriever, self.with_context, self.variant)

    def manifest(self) -> dict:
        """Every parameter that decides this configuration's rankings."""
        return {
            "name": self.name,
            "chunker": chunker_config(get_chunker(self.chunker)),
            "retriever": retriever_config(self.retriever, VARIANTS.get(self.variant)),
            "breadcrumb": self.with_context,
            "variant": self.variant or None,
        }


BASELINE = RetrievalConfig("paragraph", "tfidf", with_context=False, label="baseline (RAG Mini Chat)")

# Stage 3's evaluation: all 30 cells. (paragraph, tfidf, no breadcrumb) *is* the baseline.
FACTORIAL: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever, with_context)
    for chunker in CHUNKERS
    for retriever in FACTORIAL_RETRIEVERS
    for with_context in (True, False)
    if (chunker, retriever, with_context, "") != BASELINE.cell
)

# The control: the same cross-encoder over BM25's top 100. Was the vector index needed at all?
CONTROL = RetrievalConfig("heading", "bm25+rerank")

# The side-by-side view (`raglab compare`): the baseline, every chunker x retriever with the breadcrumb, the control.
GRID: tuple[RetrievalConfig, ...] = (BASELINE,) + tuple(
    RetrievalConfig(chunker, retriever) for chunker in CHUNKERS for retriever in FACTORIAL_RETRIEVERS
) + (CONTROL,)

# The headline comparison, named before any Stage 3 result: the full pipeline against Stage 2's best.
CHALLENGER = RetrievalConfig("heading", "hybrid+rerank")
INCUMBENT = RetrievalConfig("heading", "bm25")

# The confirmatory chain, on heading chunks with the breadcrumb, in the order it is tested (PLAN.md):
# (a, b, predicted) asks whether b beats a; predicted is +1 (b better) or 0 (no prediction).
# The first link is the headline. Each later link counts only if every earlier one was significant.
CHAIN: tuple[tuple[RetrievalConfig, RetrievalConfig, int], ...] = tuple(
    (RetrievalConfig("heading", a), RetrievalConfig("heading", b), predicted) for a, b, predicted in (
        ("bm25", "hybrid+rerank", 1),          # the headline: the full pipeline against Stage 2's best
        ("bm25", "hybrid", 1),                 # fusion alone, without the reranker
        ("bm25", "dense", 0),                  # embeddings alone: no prediction overall
        ("hybrid", "hybrid+rerank", 1),        # what the reranker adds to the hybrid
        ("bm25+rerank", "hybrid+rerank", 1),   # what dense candidates add once a reranker is used
    )
)
# Component comparisons reported beside the chain, without claims.
EXPLORATORY_PAIRS: tuple[tuple[RetrievalConfig, RetrievalConfig], ...] = (
    (RetrievalConfig("heading", "dense"), RetrievalConfig("heading", "hybrid")),
    (RetrievalConfig("heading", "bm25"), CONTROL),
)

# Pooled and scored with the factorial, but outside it: the control, the instruction's effect, and the
# rerank-depth curve (depth 100 is the factorial's own hybrid+rerank).
EXTRAS: tuple[RetrievalConfig, ...] = (
    CONTROL,
    RetrievalConfig("heading", "dense", variant="no instruction"),
    *(RetrievalConfig("heading", "hybrid+rerank", variant=f"depth {d}") for d in (10, 20, 50)),
)

EVALUATED: tuple[RetrievalConfig, ...] = FACTORIAL + EXTRAS


def build_retrievers(
    docs: list[Document], configs: tuple[RetrievalConfig, ...] = GRID
) -> dict[str, Retriever]:
    """One retriever per config, sharing work: each chunking is computed once, a hybrid
    reuses the bm25 and dense retrievers over the same chunks, and a reranker reuses its
    first stage, so no index or embedding is built twice."""
    chunk_cache: dict[str, list[Chunk]] = {}
    built: dict[tuple[str, str, bool, str], Retriever] = {}

    def get(chunker: str, retriever: str, with_context: bool, variant: str = "") -> Retriever:
        key = (chunker, retriever, with_context, variant)
        if key not in built:
            if chunker not in chunk_cache:
                chunk_cache[chunker] = chunk_corpus(docs, get_chunker(chunker))
            chunks = chunk_cache[chunker]
            options = VARIANTS.get(variant, {})
            if retriever == "dense" and variant:
                from raglab.neural import DenseRetriever

                built[key] = DenseRetriever(chunks, with_context, instruction=options["instruction"])
            elif retriever == "hybrid":
                from raglab.neural import HybridRetriever

                built[key] = HybridRetriever(chunks, with_context, lexical=get(chunker, "bm25", with_context),
                                             dense=get(chunker, "dense", with_context))
            elif retriever.endswith("+rerank"):
                from raglab.neural import RerankRetriever

                first = retriever.removesuffix("+rerank")
                built[key] = RerankRetriever(chunks, with_context, first_stage=get(chunker, first, with_context),
                                             **({"depth": options["depth"]} if "depth" in options else {}))
            else:
                built[key] = build_retriever(retriever, chunks, with_context=with_context)
        return built[key]

    return {config.name: get(*config.cell) for config in configs}
