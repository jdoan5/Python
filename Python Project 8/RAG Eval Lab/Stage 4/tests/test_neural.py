"""The dense, hybrid and reranking retrievers, and the caches under them, on stand-in models."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeEmbedder, FakeScorer
from raglab import neural
from raglab.chunking import Chunk
from raglab.neural import (
    QUERY_INSTRUCTION, RERANK_DEPTH, RRF_K, TIE_SEED, CachedEmbedder, CachedScorer, DenseRetriever, HybridRetriever,
    RerankRetriever, SentenceTransformerEmbedder, _Store, competition_ranks, neural_config, tie_order,
)
from raglab.retrieval import Bm25Retriever, RetrievalError


def chunk(text: str, n: int, heading: tuple[str, ...] = ("Page",), length: int | None = None) -> Chunk:
    return Chunk(f"doc{n}::heading-{n:03d}", f"doc{n}", "heading", 0, length if length is not None else len(text),
                 text, heading, f"a{n}")


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
    store = _Store(path, "settings-A")
    rows = {f"{i:064x}": np.random.default_rng(i).standard_normal(8).astype(np.float32) for i in range(5)}
    for k, v in rows.items():
        store.put(k, v)
    store.save()
    again = _Store(path, "settings-A")
    assert len(again) == 5 and all(np.array_equal(again.get(k), v) for k, v in rows.items())
    assert sorted(p.name for p in path.parent.iterdir()) == ["e.npz"]


@pytest.mark.parametrize("damage", ["not a zip", "other settings", "misaligned", "not unit length"])
def test_a_store_that_fails_any_check_is_moved_aside_never_trusted(tmp_path, damage):
    path = tmp_path / "e.npz"
    keys = np.array(["a" * 64, "b" * 64], dtype="U64")
    unit = np.eye(2, 4, dtype=np.float32)
    if damage == "not a zip":
        path.write_bytes(b"not a zip file")
    elif damage == "other settings":
        np.savez(path, settings=np.array("settings-B"), keys=keys, rows=unit)
    elif damage == "misaligned":
        np.savez(path, settings=np.array("settings-A"), keys=keys, rows=np.zeros((3, 4), dtype=np.float32))
    else:
        np.savez(path, settings=np.array("settings-A"), keys=keys, rows=2 * unit)
    with pytest.warns(RuntimeWarning, match="unusable"):
        store = _Store(path, "settings-A", unit_rows=True)
    assert len(store) == 0 and (tmp_path / "e.npz.corrupt").exists() and not path.exists()


def test_a_store_refuses_to_cache_a_non_finite_value(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        _Store(tmp_path / "e.npz", "s").put("a" * 64, np.array([np.nan], dtype=np.float32))


def test_caches_refuse_icloud_synced_folders(tmp_path):
    synced = tmp_path / "Mobile Documents" / "proj" / "data" / "cache"
    with pytest.raises(neural.CacheError, match="iCloud"):
        _Store(synced / "e.npz", "s")


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


def test_a_fully_cached_run_never_loads_a_model(tmp_path):
    spec = neural.model_lock()["embedder"]

    def must_not_load(_spec):
        raise AssertionError("the model was loaded although every vector was cached")

    lazy = SentenceTransformerEmbedder(spec, folder=must_not_load)
    store = CachedEmbedder(lazy, tmp_path)
    fake = FakeEmbedder()
    for text, vector in zip(["alpha", "beta"], fake.embed(["alpha", "beta"], "")):
        store._store.put(neural._key("", text), vector)
    assert store.embed(["beta", "alpha"], "").shape == (2, FakeEmbedder.dim)
    with pytest.raises(AssertionError, match="was loaded"):
        store.embed(["gamma"], "")


def test_model_settings_name_everything_that_decides_a_number():
    settings = neural.model_settings(neural.model_lock()["embedder"], "embedder")
    for key in ("fingerprint", "revision", "device", "dtype", "threads", "batch_size", "max_tokens", "platform",
                "torch", "transformers", "sentence-transformers", "tokenizers", "safetensors"):
        assert key in settings
    assert settings["threads"] == 4 and settings["batch_size"] == 1 and settings["device"] == "cpu"


# ---------------------------------------------------------------- dense


def test_dense_ranks_every_chunk_by_cosine_and_uses_the_query_instruction():
    fake, embedder = cached()
    retriever = DenseRetriever(DOCS, with_context=True, embedder=embedder)
    hits = retriever.search("return a 404 error for a missing item", k=len(DOCS))
    assert [h.rank for h in hits] == [1, 2, 3, 4] and hits[0].chunk is DOCS[0]
    assert len(retriever.search("zzz qqq", k=100)) == len(DOCS)  # even when nothing matches at all
    assert ("", DOCS[0].index_text(True)) in fake.calls
    assert (QUERY_INSTRUCTION, "return a 404 error for a missing item") in fake.calls


def test_dense_without_the_instruction_embeds_the_bare_question():
    fake, embedder = cached()
    DenseRetriever(DOCS, embedder=embedder, instruction="").search("return a 404", 1)
    assert ("", "return a 404") in fake.calls and all(i == "" for i, _ in fake.calls)


def hash_order(query, chunks):
    import hashlib

    return sorted(chunks, key=lambda c: hashlib.sha256(f"{TIE_SEED}:{query}:{c.chunk_id}".encode()).hexdigest())


def test_identical_texts_get_identical_scores_and_ties_go_to_the_seeded_hash():
    twins = [chunk("serve static files", 0), chunk("other words here", 1), chunk("serve static files", 2)]
    fake, embedder = cached()
    retriever = DenseRetriever(twins, with_context=False, embedder=embedder)
    assert sorted(t for _, t in fake.calls) == ["other words here", "serve static files"]  # each text once
    scores = retriever.scores("serve static files")
    assert scores[0] == scores[2]
    assert [h.chunk for h in retriever.search("serve static files", 2)] == hash_order("serve static files",
                                                                                      [twins[0], twins[2]])


def test_the_tie_order_is_by_score_then_a_hash_that_ignores_corpus_position():
    many = [chunk(f"text {i}", i) for i in range(12)]
    scores = np.array([0.5] * 6 + [0.9] + [0.5] * 5)
    order = tie_order(scores, "q", many)
    assert order[0] == 6  # the best score first, whatever its position
    assert [many[i] for i in order[1:]] == hash_order("q", [c for i, c in enumerate(many) if i != 6])
    assert [many[i] for i in tie_order(scores, "another question", many)[1:]] != [many[i] for i in order[1:]]


def test_dense_indexes_the_breadcrumb_only_when_asked():
    fake, embedder = cached()
    DenseRetriever(DOCS, with_context=False, embedder=embedder)
    assert sorted(t for _, t in fake.calls) == sorted(c.text for c in DOCS)


def test_dense_needs_chunks():
    with pytest.raises(RetrievalError):
        DenseRetriever([], embedder=cached()[1])


# ---------------------------------------------------------------- hybrid (reciprocal rank fusion)


class Fixed:
    """A first stage with fixed scores, one per chunk."""

    name = "fixed"

    def __init__(self, chunks, scores, with_context=True):
        self.chunks, self._scores, self.with_context = list(chunks), np.asarray(scores, dtype=float), with_context

    def scores(self, query):
        return self._scores

    def search(self, query, k=10):
        from raglab.retrieval import _rank

        return _rank(self._scores, self.chunks, k)


def test_competition_ranks_share_the_best_rank_on_ties():
    scores = np.array([5.0, 3.0, 3.0, 1.0, 0.0])
    assert list(competition_ranks(scores, np.arange(5))) == [1, 2, 2, 4, 5]
    assert list(competition_ranks(scores, np.array([1, 3]))) == [1, 2]


def test_fusion_adds_one_over_sixty_plus_rank_from_each_ranking():
    # BM25 scores chunks 2 and 0 (1st and 2nd) and nothing else; dense ranks 0, 1, 2, 3.
    hybrid = HybridRetriever(DOCS, lexical=Fixed(DOCS, [2.0, 0.0, 3.0, 0.0]), dense=Fixed(DOCS, [0.9, 0.8, 0.7, 0.6]))
    score, dense_rank, lexical_rank = hybrid.fused("q")
    assert RRF_K == 60
    assert score[0] == pytest.approx(1 / 61 + 1 / 62) and score[2] == pytest.approx(1 / 63 + 1 / 61)
    assert score[1] == pytest.approx(1 / 62) and score[3] == pytest.approx(1 / 64)  # BM25 scored them 0: no BM25 term
    assert list(lexical_rank) == [2, np.inf, 1, np.inf]
    assert [h.chunk for h in hybrid.search("q", 4)] == [DOCS[0], DOCS[2], DOCS[1], DOCS[3]]


def test_a_swapped_pair_ties_exactly_and_neither_input_is_favoured():
    # chunk 1: BM25 1st, dense 2nd; chunk 3: BM25 2nd, dense 1st. Their fused scores are bit-identical.
    hybrid = HybridRetriever(DOCS, lexical=Fixed(DOCS, [0, 2.0, 0, 1.0]), dense=Fixed(DOCS, [0.1, 0.8, 0.2, 0.9]))
    score, _, _ = hybrid.fused("q")
    assert score[1] == score[3]
    assert [h.chunk for h in hybrid.search("q", 2)] == hash_order("q", [DOCS[1], DOCS[3]])


def test_tied_input_scores_share_a_rank_before_fusion():
    hybrid = HybridRetriever(DOCS, lexical=Fixed(DOCS, [1.0, 1.0, 0, 0]), dense=Fixed(DOCS, [0.5, 0.5, 0.4, 0.3]))
    score, dense_rank, lexical_rank = hybrid.fused("q")
    assert list(lexical_rank[:2]) == [1, 1] and list(dense_rank[:2]) == [1, 1] and score[0] == score[1]
    assert [h.chunk for h in hybrid.search("q", 2)] == hash_order("q", [DOCS[0], DOCS[1]])


def test_the_hybrid_ranks_every_chunk():
    hybrid = HybridRetriever(DOCS, lexical=Fixed(DOCS, [0, 0, 0, 0]), dense=Fixed(DOCS, [-0.1, -0.2, -0.3, -0.4]))
    assert len(hybrid.search("q", 10)) == len(DOCS)


def test_a_hybrid_refuses_parts_built_over_other_chunks_or_settings():
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(DOCS, lexical=Fixed(DOCS[:3], [1, 1, 1]), dense=Fixed(DOCS, [1, 1, 1, 1]))
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(DOCS, with_context=True, lexical=Fixed(DOCS, [1] * 4, with_context=False),
                        dense=Fixed(DOCS, [1] * 4))


def test_the_default_hybrid_fuses_real_bm25_with_dense():
    hybrid = HybridRetriever(DOCS)
    assert isinstance(hybrid.lexical, Bm25Retriever) and isinstance(hybrid.dense, DenseRetriever)
    assert hybrid.search("serve static files", 1)[0].chunk is DOCS[2]


# ---------------------------------------------------------------- reranking


def rerank(first_scores, depth=RERANK_DEPTH, with_context=True, chunks=DOCS):
    fake = FakeScorer()
    first = Fixed(chunks, first_scores, with_context)
    return fake, RerankRetriever(chunks, with_context=with_context, first_stage=first,
                                 scorer=CachedScorer(fake, None), depth=depth)


def test_reranking_reorders_the_head_and_keeps_the_tail_in_first_stage_order():
    fake, rr = rerank([1.0, 2.0, 3.0, 4.0], depth=3)  # first stage: 3, 2, 1, 0
    hits = rr.search("Depends shares database session", k=4)
    assert hits[0].chunk is DOCS[3] and hits[-1].chunk is DOCS[0]  # DOCS[0] is below the depth: stays 4th
    assert [h.rank for h in hits] == [1, 2, 3, 4] and len(fake.calls) == 3


def test_negative_logits_are_ranked_not_dropped():
    class Negative(FakeScorer):
        def score(self, query, texts):
            return -10.0 + super().score(query, texts)

    first = Fixed(DOCS, [1.0, 2.0, 3.0, 4.0])
    rr = RerankRetriever(DOCS, first_stage=first, scorer=CachedScorer(Negative(), None), depth=4)
    hits = rr.search("Depends database session", k=4)
    assert len(hits) == 4 and all(h.score < 0 for h in hits) and hits[0].chunk is DOCS[3]


def test_reranker_ties_keep_the_first_stage_order():
    _, rr = rerank([1.0, 3.0, 4.0, 2.0], depth=4)  # first stage: 2, 1, 3, 0
    assert [h.chunk for h in rr.search("zzz", k=4)] == [DOCS[2], DOCS[1], DOCS[3], DOCS[0]]


def test_the_top_k_is_always_a_prefix_of_the_full_ranking():
    _, rr = rerank([1.0, 2.0, 3.0, 4.0], depth=3)
    full = [h.chunk for h in rr.search("database session 404 static", k=len(DOCS))]
    for k in range(len(DOCS) + 1):
        assert [h.chunk for h in rr.search("database session 404 static", k=k)] == full[:k]


def test_the_default_depth_is_the_first_stage_s_top_100():
    assert RERANK_DEPTH == 100
    many = [chunk(f"text {i}", i) for i in range(150)]
    fake, rr = rerank(list(range(150, 0, -1)), chunks=many)
    assert rr.depth_for(rr.first_stage.search("q", 150)) == 100
    hits = rr.search("q", 150)
    assert len(fake.calls) == 100 and [h.chunk for h in hits[100:]] == many[100:]  # ranks 101+ keep their order
    few = [chunk(f"text {i}", i) for i in range(30)]
    _, rr = rerank(list(range(30, 0, -1)), chunks=few)
    assert rr.depth_for(rr.first_stage.search("q", 30)) == 30  # the whole first stage when it holds fewer


def test_the_reranker_sees_the_text_as_indexed():
    fake, rr = rerank([4.0, 3.0, 2.0, 1.0], with_context=True)
    rr.search("404", k=1)
    assert ("404", DOCS[0].index_text(True)) in fake.calls
    fake, rr = rerank([4.0, 3.0, 2.0, 1.0], with_context=False)
    rr.search("404", k=1)
    assert ("404", DOCS[0].text) in fake.calls


def test_a_reranker_is_named_for_its_first_stage_and_checks_its_inputs():
    assert RerankRetriever(DOCS, first_stage=Bm25Retriever(DOCS), scorer=CachedScorer(FakeScorer(), None)).name \
        == "bm25+rerank"
    assert RerankRetriever(DOCS, scorer=CachedScorer(FakeScorer(), None)).name == "hybrid+rerank"
    with pytest.raises(ValueError, match="depth"):
        RerankRetriever(DOCS, scorer=CachedScorer(FakeScorer(), None), depth=0)
    with pytest.raises(ValueError, match="same chunks"):
        RerankRetriever(DOCS[:2], first_stage=HybridRetriever(DOCS), scorer=CachedScorer(FakeScorer(), None))


# ---------------------------------------------------------------- manifests


def test_neural_configs_name_every_parameter_that_decides_a_ranking():
    dense, hybrid = neural_config("dense"), neural_config("hybrid")
    bm25_rr, hybrid_rr = neural_config("bm25+rerank"), neural_config("hybrid+rerank")
    assert dense["model"]["repo"] == "BAAI/bge-small-en-v1.5" and dense["query_instruction"] == QUERY_INSTRUCTION
    assert dense["ranks"] == "every chunk" and dense["model"]["batch_size"] == 1
    assert hybrid["k"] == 60 and "competition ranking" in hybrid["input_ranks"]
    assert hybrid["ties"] == dense["ties"] and TIE_SEED in hybrid["ties"]
    assert bm25_rr["first_stage"] == "bm25" and hybrid_rr["first_stage"] == "hybrid"
    assert hybrid_rr["depth"] == 100 and hybrid_rr["model"]["repo"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert neural_config("hybrid+rerank", {"depth": 20})["depth"] == 20
    assert neural_config("dense", {"instruction": ""})["query_instruction"] == ""
    with pytest.raises(ValueError, match="unknown neural retriever"):
        neural_config("colbert")


def test_the_unit_tests_run_on_stand_in_models():
    assert isinstance(neural.default_embedder().embedder, FakeEmbedder)
    assert isinstance(neural.default_scorer().scorer, FakeScorer)


# ---------------------------------------------------------------- the real models, on a question outside the golden set


@pytest.mark.models
@pytest.mark.corpus
def test_the_real_neural_retrievers_rank_every_chunk_and_find_the_cors_page(real_docs, real_models):
    from raglab.chunking import HeadingChunker, chunk_corpus

    chunks = chunk_corpus(real_docs, HeadingChunker())
    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever(chunks, lexical=bm25, dense=dense)
    question = "allow my React frontend on another origin to call the API"
    for retriever in (dense, hybrid, RerankRetriever(chunks, first_stage=hybrid),
                      RerankRetriever(chunks, first_stage=bm25)):
        full = retriever.search(question, len(chunks))
        assert len(full) == (len(chunks) if retriever.name != "bm25+rerank" else len(bm25.search(question, len(chunks))))
        assert any(h.chunk.doc_id == "tutorial/cors" for h in full[:5]), retriever.name
        assert [h.chunk.chunk_id for h in retriever.search(question, 7)] == [h.chunk.chunk_id for h in full[:7]]
