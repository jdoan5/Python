"""The dense, hybrid and reranking retrievers, and the caches under them, on stand-in models."""

from __future__ import annotations

import numpy as np
import pytest

from raglab import neural
from raglab.chunking import Chunk
from raglab.neural import (
    QUERY_INSTRUCTION, RERANK_DEPTH, RRF_K, CachedEmbedder, CachedScorer, DenseRetriever, HybridRetriever,
    RerankRetriever, _Store, neural_config,
)
from raglab.retrieval import Bm25Retriever, Hit, RetrievalError
from conftest import FakeEmbedder, FakeScorer


def chunk(text: str, n: int, heading: tuple[str, ...] = ("Page",)) -> Chunk:
    return Chunk(f"doc{n}::heading-{n:03d}", f"doc{n}", "heading", 0, len(text), text, heading, f"a{n}")


DOCS = [
    chunk("Raise an HTTPException to return a 404 error when an item is missing.", 0, ("Tutorial", "Errors")),
    chunk("Declare the request body with a Pydantic model and FastAPI validates it.", 1, ("Tutorial", "Body")),
    chunk("Mount StaticFiles to serve images and CSS from a directory.", 2, ("Tutorial", "Static Files")),
    chunk("Use Depends to share a database session between path operations.", 3, ("Tutorial", "Dependencies")),
]


def cached(cache_dir=None, settings=None):
    fake = FakeEmbedder(settings)
    return fake, CachedEmbedder(fake, cache_dir)


# ---------------------------------------------------------------- the on-disk store


def test_a_store_round_trips_exactly_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "cache" / "e.npz"
    store = _Store(path)
    rows = {f"{i:064x}": np.random.default_rng(i).standard_normal(8).astype(np.float32) for i in range(5)}
    for k, v in rows.items():
        store.put(k, v)
    store.save()
    again = _Store(path)
    assert len(again) == 5 and all(np.array_equal(again.get(k), v) for k, v in rows.items())
    assert sorted(p.name for p in path.parent.iterdir()) == ["e.npz"]


def test_an_unreadable_store_is_moved_aside_never_trusted(tmp_path):
    path = tmp_path / "e.npz"
    path.write_bytes(b"not a zip file")
    with pytest.warns(RuntimeWarning, match="unreadable"):
        store = _Store(path)
    assert len(store) == 0 and (tmp_path / "e.npz.corrupt").exists() and not path.exists()


def test_a_store_with_misaligned_arrays_is_moved_aside(tmp_path):
    path = tmp_path / "e.npz"
    np.savez(path, keys=np.array(["a" * 64, "b" * 64], dtype="U64"), rows=np.zeros((3, 4), dtype=np.float32))
    with pytest.warns(RuntimeWarning):
        assert len(_Store(path)) == 0


# ---------------------------------------------------------------- the caching wrappers


def test_each_text_is_embedded_once_then_read_from_the_cache(tmp_path):
    fake, embedder = cached(tmp_path)
    first = embedder.embed(["alpha beta", "gamma delta", "alpha beta"], "")
    assert len(fake.calls) == 2  # the repeated text is embedded once
    second = embedder.embed(["gamma delta", "alpha beta"], "")
    assert len(fake.calls) == 2 and np.array_equal(second, first[[1, 0]])
    fresh_fake, fresh = cached(tmp_path)  # a new process: the file on disk answers
    assert np.array_equal(fresh.embed(["alpha beta"], ""), first[:1]) and fresh_fake.calls == []


def test_the_instruction_is_part_of_the_key(tmp_path):
    fake, embedder = cached(tmp_path)
    embedder.embed(["how do I return a 404"], "")
    embedder.embed(["how do I return a 404"], QUERY_INSTRUCTION)
    assert fake.calls == [("", "how do I return a 404"), (QUERY_INSTRUCTION, "how do I return a 404")]


def test_changing_any_setting_starts_a_separate_cache(tmp_path):
    _, a = cached(tmp_path, {"model": "m", "revision": "1", "threads": 4})
    a.embed(["alpha"], "")
    fake_b, b = cached(tmp_path, {"model": "m", "revision": "2", "threads": 4})
    b.embed(["alpha"], "")
    fake_c, c = cached(tmp_path, {"model": "m", "revision": "1", "threads": 8})
    c.embed(["alpha"], "")
    assert fake_b.calls == [("", "alpha")] and fake_c.calls == [("", "alpha")]
    assert len(list(tmp_path.glob("embeddings-*.npz"))) == 3


def test_reranker_scores_are_cached_per_question_and_text(tmp_path):
    fake = FakeScorer()
    scorer = CachedScorer(fake, tmp_path)
    s1 = scorer.score("return a 404 error", ["a 404 error", "static files"])
    s2 = scorer.score("return a 404 error", ["static files", "a 404 error"])
    assert list(s2) == list(s1[::-1]) and len(fake.calls) == 2
    scorer.score("serve static files", ["static files"])
    assert len(fake.calls) == 3  # a new question is a new pair
    again = FakeScorer()
    assert list(CachedScorer(again, tmp_path).score("return a 404 error", ["a 404 error"])) == [s1[0]]
    assert again.calls == []


# ---------------------------------------------------------------- dense


def test_dense_retrieval_ranks_by_cosine_and_uses_the_query_instruction():
    fake, embedder = cached()
    retriever = DenseRetriever(DOCS, with_context=True, embedder=embedder)
    hits = retriever.search("return a 404 error for a missing item", k=2)
    assert hits[0].chunk is DOCS[0] and [h.rank for h in hits] == [1, 2]
    assert hits[0].score == pytest.approx(float(retriever.scores("return a 404 error for a missing item")[0]))
    instructions = {(i, t) for i, t in fake.calls}
    assert ("", DOCS[0].index_text(True)) in instructions          # chunks: no instruction, breadcrumb on
    assert (QUERY_INSTRUCTION, "return a 404 error for a missing item") in instructions


def test_dense_indexes_the_breadcrumb_only_when_asked():
    fake, embedder = cached()
    DenseRetriever(DOCS, with_context=False, embedder=embedder)
    assert sorted(t for _, t in fake.calls) == sorted(c.text for c in DOCS)


def test_dense_needs_chunks():
    with pytest.raises(RetrievalError):
        DenseRetriever([], embedder=cached()[1])


# ---------------------------------------------------------------- hybrid (reciprocal rank fusion)


class Fixed:
    """A retriever that returns a fixed ranking of chunk positions."""

    def __init__(self, chunks, order, with_context=True):
        self.chunks, self.order, self.with_context = list(chunks), order, with_context

    def search(self, query, k=10):
        return [Hit(rank=r + 1, score=1.0 / (r + 1), chunk=self.chunks[i]) for r, i in enumerate(self.order[:k])]


def test_fusion_adds_one_over_sixty_plus_rank_from_each_ranking():
    hybrid = HybridRetriever(DOCS, lexical=Fixed(DOCS, [2, 0]), dense=Fixed(DOCS, [0, 1, 2, 3]))
    scores = hybrid.scores("q")
    assert RRF_K == 60
    assert scores[0] == pytest.approx(1 / 62 + 1 / 61)      # 2nd for BM25, 1st for dense
    assert scores[2] == pytest.approx(1 / 61 + 1 / 63)      # 1st for BM25, 3rd for dense
    assert scores[1] == pytest.approx(1 / 62)               # BM25 never scored it: nothing from BM25
    assert [h.chunk for h in hybrid.search("q", 4)] == [DOCS[0], DOCS[2], DOCS[1], DOCS[3]]


def test_fusion_ties_go_to_corpus_order():
    hybrid = HybridRetriever(DOCS, lexical=Fixed(DOCS, [3, 1]), dense=Fixed(DOCS, [1, 3]))
    assert [h.chunk for h in hybrid.search("q", 2)] == [DOCS[1], DOCS[3]]  # 1/61 + 1/62 each


def test_a_hybrid_refuses_parts_built_over_other_chunks_or_settings():
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(DOCS, lexical=Fixed(DOCS[:3], [0]), dense=Fixed(DOCS, [0]))
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(DOCS, with_context=True, lexical=Fixed(DOCS, [0], with_context=False), dense=Fixed(DOCS, [0]))


def test_the_default_hybrid_fuses_real_bm25_with_dense():
    hybrid = HybridRetriever(DOCS)
    assert isinstance(hybrid.lexical, Bm25Retriever) and isinstance(hybrid.dense, DenseRetriever)
    assert hybrid.search("serve static files", 1)[0].chunk is DOCS[2]


# ---------------------------------------------------------------- hybrid+rerank


def rerank(order, depth=RERANK_DEPTH, with_context=True):
    fake = FakeScorer()
    first = HybridRetriever(DOCS, with_context=with_context,
                            lexical=Fixed(DOCS, order, with_context), dense=Fixed(DOCS, order, with_context))
    return fake, RerankRetriever(DOCS, with_context=with_context, first_stage=first,
                                 scorer=CachedScorer(fake, None), depth=depth)


def test_reranking_reorders_the_head_and_keeps_the_tail_in_first_stage_order():
    fake, rr = rerank([3, 2, 1, 0], depth=3)
    hits = rr.search("Depends shares database session", k=4)
    # head = DOCS 3, 2, 1, re-ordered by shared words with the question; DOCS[0] stays 4th (below the depth)
    assert hits[0].chunk is DOCS[3] and hits[-1].chunk is DOCS[0]
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert len(fake.calls) == 3  # only the top `depth` are scored


def test_reranker_ties_keep_the_first_stage_order():
    _, rr = rerank([2, 1, 3, 0], depth=4)
    hits = rr.search("zzz", k=4)  # every pair shares nothing: all tie
    assert [h.chunk for h in hits] == [DOCS[2], DOCS[1], DOCS[3], DOCS[0]]


def test_the_reranker_sees_the_text_as_indexed():
    fake, rr = rerank([0, 1, 2, 3], with_context=True)
    rr.search("404", k=1)
    assert ("404", DOCS[0].index_text(True)) in fake.calls
    fake, rr = rerank([0, 1, 2, 3], with_context=False)
    rr.search("404", k=1)
    assert ("404", DOCS[0].text) in fake.calls


def test_a_reranker_needs_a_positive_depth_and_matching_chunks():
    with pytest.raises(ValueError, match="depth"):
        RerankRetriever(DOCS, first_stage=HybridRetriever(DOCS), scorer=CachedScorer(FakeScorer(), None), depth=0)
    with pytest.raises(ValueError, match="same chunks"):
        RerankRetriever(DOCS[:2], first_stage=HybridRetriever(DOCS), scorer=CachedScorer(FakeScorer(), None))


def test_search_respects_k():
    _, rr = rerank([0, 1, 2, 3])
    assert len(rr.search("q", 2)) == 2 and rr.search("q", 0) == []


# ---------------------------------------------------------------- manifests


def test_neural_configs_name_every_parameter_that_decides_a_ranking():
    dense, hybrid, rr = neural_config("dense"), neural_config("hybrid"), neural_config("hybrid+rerank")
    assert dense["model"]["repo"] == "BAAI/bge-small-en-v1.5" and dense["query_instruction"] == QUERY_INSTRUCTION
    assert len(dense["model"]["revision"]) == 40 and dense["batch_size"] == 1
    assert hybrid == {"name": "hybrid", "fusion": "reciprocal rank", "k": 60, "inputs": ["bm25", "dense"],
                      "ties": "corpus order"}
    assert rr["model"]["repo"] == "cross-encoder/ms-marco-MiniLM-L6-v2" and rr["depth"] == 100
    with pytest.raises(ValueError, match="unknown neural retriever"):
        neural_config("colbert")


def test_the_unit_tests_run_on_stand_in_models():
    assert isinstance(neural.default_embedder().embedder, FakeEmbedder)
    assert isinstance(neural.default_scorer().scorer, FakeScorer)


# ---------------------------------------------------------------- the real models, on questions outside the golden set


@pytest.mark.models
@pytest.mark.corpus
def test_the_real_neural_retrievers_find_the_cors_page(real_docs, real_models):
    from raglab.chunking import HeadingChunker, chunk_corpus

    chunks = chunk_corpus(real_docs, HeadingChunker())
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(chunks, dense=dense)
    rr = RerankRetriever(chunks, first_stage=hybrid)
    for retriever in (dense, hybrid, rr):
        hits = retriever.search("allow my React frontend on another origin to call the API", k=5)
        assert any(h.chunk.doc_id == "tutorial/cors" for h in hits), retriever.name
