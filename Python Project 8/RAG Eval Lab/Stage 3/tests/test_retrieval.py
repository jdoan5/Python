from __future__ import annotations

import math

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from raglab import experiments
from raglab.chunking import Chunk
from raglab.experiments import BASELINE, GRID, build_retrievers
from raglab.retrieval import Bm25Retriever, RetrievalError, TfidfRetriever, build_retriever


def chunk(text: str, n: int = 0, path: tuple[str, ...] = ()) -> Chunk:
    return Chunk(f"doc::t-{n:03d}", "doc", "t", 0, len(text), text, path, "")


FRUIT = [chunk("apples bananas", 0), chunk("apples cherries cherries", 1), chunk("dates", 2)]


# ---------------------------------------------------------------- BM25

def test_bm25_score_matches_the_okapi_formula_computed_by_hand():
    hits = Bm25Retriever(FRUIT, with_context=False).search("cherries")
    # Independent reference: n=3 chunks, lengths 2/3/1 (avg 2), "cherries" in 1 chunk with tf=2.
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / 2)
    expected = idf * 2 * (1.5 + 1) / (2 + norm)
    assert [h.chunk.chunk_id for h in hits] == ["doc::t-001"]
    assert hits[0].score == pytest.approx(expected)


def test_bm25_ranks_the_rarer_term_higher_and_drops_non_matches():
    hits = Bm25Retriever(FRUIT, with_context=False).search("dates apples")
    assert hits[0].chunk.text == "dates"
    assert {h.chunk.text for h in hits} == {"dates", "apples bananas", "apples cherries cherries"}
    assert Bm25Retriever(FRUIT, with_context=False).search("kiwi") == []


def test_a_repeated_query_term_counts_each_time():
    r = Bm25Retriever(FRUIT, with_context=False)
    once, twice = r.search("cherries")[0].score, r.search("cherries cherries")[0].score
    assert twice == pytest.approx(2 * once)


def test_k_limits_results_and_ranks_are_one_based():
    hits = Bm25Retriever(FRUIT, with_context=False).search("apples", k=1)
    assert [h.rank for h in hits] == [1] and len(hits) == 1


@pytest.mark.parametrize("cls", [Bm25Retriever, TfidfRetriever])
def test_ties_break_by_corpus_order_so_rankings_are_reproducible(cls):
    # 40 tied chunks: an unstable sort (e.g. quicksort) scrambles this; two would not show it.
    many = [chunk("identical text here", i) for i in range(40)]
    hits = cls(many, with_context=False).search("identical", k=40)
    assert [h.chunk.chunk_id for h in hits] == [f"doc::t-{i:03d}" for i in range(40)]


@pytest.mark.parametrize("cls", [Bm25Retriever, TfidfRetriever])
def test_a_query_of_only_stop_words_returns_nothing(cls):
    assert cls(FRUIT, with_context=False).search("the and of which") == []


# ---------------------------------------------------------------- TF-IDF (the baseline)

def test_tfidf_is_configured_exactly_like_rag_mini_chat():
    r = TfidfRetriever(FRUIT)
    assert r.vectorizer.ngram_range == (1, 2)
    assert r.vectorizer.stop_words == "english"


def test_tfidf_ranking_equals_scikit_learn_cosine_similarity():
    texts = ["path parameters in the url", "query parameters with defaults", "request body models"]
    chunks = [chunk(t, i) for i, t in enumerate(texts)]
    query = "query parameters"
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(texts)
    sims = cosine_similarity(vec.transform([query]), vec.transform(texts))[0]
    expected = sorted(((i, s) for i, s in enumerate(sims) if s > 0), key=lambda p: (-p[1], p[0]))
    hits = TfidfRetriever(chunks, with_context=False).search(query, k=3)
    # Which chunk got each score — not just the multiset of scores.
    assert [h.chunk.chunk_id for h in hits] == [f"doc::t-{i:03d}" for i, _ in expected]
    assert [h.score for h in hits] == pytest.approx([s for _, s in expected])


# ---------------------------------------------------------------- shared behaviour

@pytest.mark.parametrize("cls", [Bm25Retriever, TfidfRetriever])
def test_the_breadcrumb_is_searchable_only_with_context(cls):
    chunks = [chunk("set the scopes here", 0, ("Security", "OAuth2")), chunk("unrelated words", 1)]
    assert cls(chunks, with_context=True).search("oauth2")
    assert cls(chunks, with_context=False).search("oauth2") == []


@pytest.mark.parametrize("cls", [Bm25Retriever, TfidfRetriever])
def test_nothing_to_index_is_a_clear_error(cls):
    with pytest.raises(RetrievalError):
        cls([])
    with pytest.raises(RetrievalError):
        cls([chunk("the and of", 0)], with_context=False)


def test_build_retriever_names_the_choices_on_a_typo():
    with pytest.raises(ValueError, match="bm25, tfidf, dense, hybrid, bm25\\+rerank, hybrid\\+rerank"):
        build_retriever("colbert", FRUIT)


# ---------------------------------------------------------------- the experiment grid

def test_the_grid_is_the_baseline_plus_every_chunker_retriever_pair():
    names = [c.name for c in GRID]
    assert GRID[0] == BASELINE
    assert (BASELINE.chunker, BASELINE.retriever, BASELINE.with_context) == ("paragraph", "tfidf", False)
    # the baseline, then 3 chunkers x 5 retrievers with the breadcrumb on, then the bm25+rerank control
    assert len(GRID) == 17 and len(set(names)) == 17
    assert {c.retriever for c in GRID[1:]} == {"bm25", "tfidf", "dense", "hybrid", "bm25+rerank", "hybrid+rerank"}
    assert GRID[-1].name == "heading+bm25+rerank"


def test_build_retrievers_chunks_each_strategy_only_once(monkeypatch, mini_corpus):
    from raglab.corpus import load_corpus

    calls = []
    real = experiments.chunk_corpus
    monkeypatch.setattr(experiments, "chunk_corpus", lambda docs, c: calls.append(c.name) or real(docs, c))
    retrievers = build_retrievers(list(load_corpus(mini_corpus).documents))
    assert len(retrievers) == 17
    assert sorted(calls) == ["fixed", "heading", "paragraph"]


def test_build_retrievers_builds_each_index_once_and_shares_it(mini_corpus):
    from raglab.corpus import load_corpus

    retrievers = build_retrievers(list(load_corpus(mini_corpus).documents))
    for chunker in ("paragraph", "fixed", "heading"):
        hybrid, rerank = retrievers[f"{chunker}+hybrid"], retrievers[f"{chunker}+hybrid+rerank"]
        assert hybrid.lexical is retrievers[f"{chunker}+bm25"]
        assert hybrid.dense is retrievers[f"{chunker}+dense"]
        assert rerank.first_stage is hybrid
    assert retrievers["heading+bm25+rerank"].first_stage is retrievers["heading+bm25"]


def test_variants_share_the_main_configuration_s_components(mini_corpus):
    from raglab.corpus import load_corpus
    from raglab.experiments import EXTRAS

    retrievers = build_retrievers(list(load_corpus(mini_corpus).documents), GRID + EXTRAS)
    assert retrievers["heading+dense [no instruction]"].instruction == ""
    for depth in (10, 20, 50):
        variant = retrievers[f"heading+hybrid+rerank [depth {depth}]"]
        assert variant.depth == depth and variant.first_stage is retrievers["heading+hybrid"]
    assert retrievers["heading+hybrid+rerank"].depth == 100


# ---------------------------------------------------------------- the real corpus

@pytest.mark.corpus
def test_bm25_finds_the_error_handling_page_for_a_404_question(real_docs):
    from raglab.chunking import HeadingChunker, chunk_corpus

    hits = Bm25Retriever(chunk_corpus(real_docs, HeadingChunker())).search(
        "raise an HTTPException to return a 404 error", k=5
    )
    assert any(h.chunk.doc_id == "tutorial/handling-errors" for h in hits)
