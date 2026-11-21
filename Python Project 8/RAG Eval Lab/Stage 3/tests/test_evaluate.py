from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import MappingProxyType
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from raglab import evaluate
from raglab.corpus import CorpusLock, default_lock_path, fingerprint_of, hash_tree, load_corpus
from raglab.evaluate import (
    QuestionResult, build_manifest, code_version, load_run, run_configs, save_run, summarize, unjudged,
    unjudged_in_windows,
)
from raglab.experiments import BASELINE, EVALUATED, FACTORIAL, RetrievalConfig
from raglab.golden import Question, check_against_corpus, default_golden_path, dump_golden, load_golden, questions_fingerprint
from raglab.metrics import BINARY_METRICS, METRICS
from raglab.sections import SectionIndex

RAISE = "tutorial/first-steps#raise-errors"
DOCS = "tutorial/first-steps#interactive-api-docs"
HOME = "index#home"
FROZEN_QUESTIONS = "c4673dddd9bdce5b73ae05f94672ad9e603c591d80b9469133ea442b821fcb4a"
FIXED_TIME = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
STAGE = Path(__file__).resolve().parents[1]
V1 = default_golden_path(1)
# Stage 2's twelve configurations: the lexical cells of Stage 3's factorial.
LEXICAL = tuple(c for c in FACTORIAL if c.retriever in ("bm25", "tfidf"))
needs_v2 = pytest.mark.skipif(not default_golden_path(2).is_file(), reason="golden/v2.yaml is not built yet")
# Tests that score Stage 3 configurations against v2 wait until the labels are final (PLAN.md, freeze point 2).
needs_final_v2 = pytest.mark.skipif(not (default_golden_path(2).parent / "v2-blind.json").is_file(),
                                    reason="v2's labels are not final yet (no golden/v2-blind.json)")
needs_results = pytest.mark.skipif(not (STAGE / "results" / "manifest.json").is_file(),
                                   reason="Stage 3's results/ is not generated yet")


@pytest.fixture
def setup(tmp_path, mini_corpus):
    files = hash_tree(mini_corpus)
    lock = CorpusLock("repo", "0.1.0", "c" * 40, fingerprint_of(files), files)
    docs = list(load_corpus(mini_corpus).documents)
    raw = {
        "version": 1,
        "corpus": {"ref": lock.ref, "fingerprint": lock.fingerprint},
        "questions": [
            {"id": "q001", "question": "How do I return a 404 status code when an item is not found?",
             "style": "paraphrase", "target": RAISE, "judgments": {RAISE: 2}},
            {"id": "q002", "question": "Where do I see the Swagger UI docs page?",
             "style": "keyword", "target": DOCS, "judgments": {DOCS: 2, HOME: 1}},
            {"id": "q003", "question": "Welcome page of the docs", "style": "scenario",
             "target": HOME, "judgments": {HOME: 2}},
        ],
    }
    path = tmp_path / "golden.yaml"
    dump_golden(raw, path)
    golden = load_golden(path)
    index = SectionIndex(docs)
    assert check_against_corpus(golden, lock, index) == []
    return lock, docs, index, golden


def test_every_configuration_scores_every_question(setup):
    _, docs, index, golden = setup
    results = run_configs(docs, golden.questions, FACTORIAL, index)
    assert list(results) == [c.name for c in FACTORIAL] and len(results) == 30
    for per_question in results.values():
        assert len(per_question) == 3
        for r in per_question:
            assert set(r.values) == set(METRICS)
            assert all(0.0 <= v <= 1.0 for v in r.values.values())
            assert len(r.ranking) <= 10


def test_the_factorial_is_complete_and_contains_the_baseline_once():
    cells = [c.cell for c in FACTORIAL]
    assert len(set(cells)) == 30 and BASELINE.cell in cells
    assert sum(c.name == BASELINE.name for c in FACTORIAL) == 1
    assert len(LEXICAL) == 12 and len(EVALUATED) == 35 and len({c.name for c in EVALUATED}) == 35


def test_summary_has_intervals_comparisons_and_switch_effects(setup):
    _, docs, index, golden = setup
    results = run_configs(docs, golden.questions, FACTORIAL, index)
    summary = summarize(golden.questions, FACTORIAL, results, resamples=1000)
    for name in summary["configs"]:
        for m in METRICS:
            cell = summary["summary"][name][m]
            assert cell["lo"] <= cell["mean"] <= cell["hi"]
    assert BASELINE.name not in summary["vs_baseline"] and len(summary["vs_baseline"]) == 29
    for per_metric in summary["vs_baseline"].values():
        for row in per_metric.values():
            assert row["p"] <= row["p_holm"] <= 1.0
    for e in summary["effects"]:
        for pair in e["pairs"].values():  # each matched pair carries its own paired test
            assert pair["lo"] <= pair["delta"] <= pair["hi"] and pair["p"] <= pair["p_holm"] <= 1.0
    for style in summary["by_style"].values():
        for per_config in style["configs"].values():
            cell = per_config["recall@5"]
            assert cell["lo"] <= cell["mean"] <= cell["hi"] and "success@5" in per_config
    pairs = {(e["factor"], e["within"], str(e["from"]), str(e["to"])): len(e["pairs"]) for e in summary["effects"]}
    for a, b in (("tfidf", "bm25"), ("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid"),
                 ("hybrid", "hybrid+rerank")):
        assert pairs[("retriever", None, a, b)] == 6                   # 3 chunkers x 2 breadcrumb settings
    assert pairs[("breadcrumb", "lexical", "False", "True")] == 6      # 3 chunkers x 2 lexical retrievers
    assert pairs[("chunker", "lexical", "paragraph", "heading")] == 4  # 2 lexical retrievers x 2 breadcrumbs
    for family in ("dense", "hybrid", "hybrid+rerank"):                # never pooled across families
        assert pairs[("breadcrumb", family, "False", "True")] == 3
        assert pairs[("chunker", family, "paragraph", "heading")] == 2
    assert summary["confirmatory"] is None  # the chain needs the control, which the factorial alone lacks
    assert set(summary["by_style_paired"]) == {"keyword", "paraphrase", "scenario"}
    assert set(summary["by_style"]) == {"keyword", "paraphrase", "scenario"}
    for d in summary["diagnostics"].values():
        assert 0.0 <= d["judged@10"] <= 1.0 and 0.0 <= d["empty_slots@10"] <= 1.0


def test_the_baseline_must_be_among_the_configurations(setup):
    _, docs, index, golden = setup
    others = tuple(c for c in FACTORIAL if c != BASELINE)
    with pytest.raises(ValueError, match="baseline"):
        summarize(golden.questions, others, run_configs(docs, golden.questions, others, index), resamples=1000)


def test_configuration_names_must_be_unique(setup):
    _, docs, index, golden = setup
    twins = (BASELINE, RetrievalConfig("fixed", "bm25", label=BASELINE.name))
    with pytest.raises(ValueError, match="unique"):
        run_configs(docs, golden.questions, twins, index)


def test_pooling_lists_only_sections_nobody_judged(setup):
    _, _, _, golden = setup
    results = {
        "a": [QuestionResult({}, (RAISE, None, "x#1"), 0), QuestionResult({}, (DOCS,), 0), QuestionResult({}, (), 0)],
        "b": [QuestionResult({}, ("x#1", "x#2"), 0), QuestionResult({}, (HOME, "y#1"), 0), QuestionResult({}, (), 0)],
    }
    assert unjudged(golden.questions, results, depth=2) == {"q001": {"x#1": ["b"], "x#2": ["b"]}, "q002": {"y#1": ["b"]}}
    assert unjudged(golden.questions, results, depth=3)["q001"] == {"x#1": ["a", "b"], "x#2": ["b"]}


def test_window_pooling_lists_unjudged_sections_inside_a_budget(setup):
    _, _, _, golden = setup

    def r(window):
        return QuestionResult({}, (), 0, frozenset(window), {4800: frozenset(window), 9600: frozenset(window) | {"z#9"}})

    results = {"a": [r({RAISE, "w#1"}), r({DOCS}), r(set())], "b": [r({"w#1", "w#2"}), r({HOME}), r({"w#3"})]}
    assert unjudged_in_windows(golden.questions, results) == {
        "q001": {"w#1": ["a", "b"], "w#2": ["b"]}, "q003": {"w#3": ["b"]}}
    assert unjudged_in_windows(golden.questions, results, configs=["a"]) == {"q001": {"w#1": ["a"]}}
    assert "z#9" in unjudged_in_windows(golden.questions, results, budget=9600)["q001"]


def _run(setup, out: Path) -> dict:
    lock, docs, index, golden = setup
    manifest = build_manifest(golden, lock, FACTORIAL, resamples=1000, now=FIXED_TIME)
    results = run_configs(docs, golden.questions, FACTORIAL, index)
    summary = summarize(golden.questions, FACTORIAL, results, resamples=1000)
    save_run(out, manifest, summary, golden.questions, results)
    return manifest


def test_runs_are_reproducible_byte_for_byte(setup, tmp_path):
    _run(setup, tmp_path / "one")
    _run(setup, tmp_path / "two")
    for name in ("summary.json", "questions.jsonl", "manifest.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name


def test_the_manifest_pins_everything_and_carries_no_personal_paths(setup, tmp_path):
    lock, _, _, golden = setup
    manifest = _run(setup, tmp_path / "run")
    assert manifest["corpus"]["fingerprint"] == lock.fingerprint
    assert manifest["golden"] == {"path": "golden.yaml", "version": 1, "sha256": golden.sha256, "questions": 3}
    assert len(manifest["code"]["package_sha256"]) == 64
    assert {c["name"] for c in manifest["configs"]} == {c.name for c in FACTORIAL}
    bm25 = next(c for c in manifest["configs"] if c["retriever"]["name"] == "bm25")
    assert bm25["retriever"]["k1"] == 1.5 and bm25["chunker"]["name"] in ("paragraph", "fixed", "heading")
    assert manifest["schema"] == 4
    assert manifest["models"]["embedder"]["repo"] == "BAAI/bge-small-en-v1.5"
    assert manifest["models"]["reranker"]["threads"] == manifest["numerics"]["threads"] == 4
    assert manifest["confirmatory_plan"]["primary_metric"] == "mrr@10"
    assert manifest["confirmatory_plan"]["chain"][0] == ["heading+bm25", "heading+hybrid+rerank", 1]
    assert manifest["plan"]["path"].endswith("PLAN.md")
    text = (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    for personal in (str(Path.home()), str(tmp_path), "/Users/", "/home/"):
        assert personal not in text


def test_code_version_works_without_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(evaluate.subprocess, "run", no_git)
    version = code_version()
    assert version["git_commit"] is None and version["git_dirty"] is None
    assert len(version["package_sha256"]) == 64


def test_code_version_reports_git_when_it_answers(monkeypatch):
    def fake_git(args, **kwargs):
        out = "a" * 40 + "\n" if "rev-parse" in args else " M raglab/cli.py\n"
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(evaluate.subprocess, "run", fake_git)
    assert code_version()["git_commit"] == "a" * 40 and code_version()["git_dirty"] is True


def test_a_saved_run_is_never_silently_replaced(setup, tmp_path):
    _run(setup, tmp_path / "run")
    with pytest.raises(FileExistsError):
        _run(setup, tmp_path / "run")
    manifest, summary = load_run(tmp_path / "run")
    assert manifest["run_id"].startswith("20261120-120000-") and summary["questions"] == 3


def test_loading_something_that_is_not_a_run_is_a_clean_error(tmp_path):
    with pytest.raises(ValueError, match="not a readable run"):
        load_run(tmp_path)
    (tmp_path / "manifest.json").write_text('{"schema": 99}', encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different run format"):
        load_run(tmp_path)


# --- Stage 2's numbers, recomputed by Stage 3's code ------------------------------------------


@pytest.mark.corpus
def test_stage_3_code_reproduces_stage_2_s_summary_on_its_twelve_configurations(real_docs):
    # Everything Stage 3 changed in the shared machinery (ranking helpers, effects within families,
    # new summary blocks) must leave Stage 2's committed numbers exactly as they were.
    from raglab.evaluate import _round_tree

    golden = load_golden(V1)
    stage2 = json.loads((STAGE.parent / "Stage 2" / "results" / "summary.json").read_text(encoding="utf-8"))
    run = run_configs(real_docs, golden.questions, LEXICAL, SectionIndex(real_docs))
    summary = _round_tree(json.loads(json.dumps(summarize(golden.questions, LEXICAL, run))))
    new = ("ndcg_answers@10",)  # Stage 3 adds metrics; everything Stage 2 reported must be unchanged
    assert {n: {m: v for m, v in per.items() if m not in new} for n, per in summary["summary"].items()} == stage2["summary"]
    assert {n: {m: v for m, v in per.items() if m not in new}
            for n, per in summary["vs_baseline"].items()} == stage2["vs_baseline"]
    for style, block in stage2["by_style"].items():
        assert {n: {m: v for m, v in per.items() if m not in new}
                for n, per in summary["by_style"][style]["configs"].items()} == block["configs"]
    assert summary["confirmatory"] is None and summary["interaction"] is None
    mine = [{k: v for k, v in e.items() if k != "within"} for e in summary["effects"]
            if e["within"] in (None, "lexical") and e["metric"] not in new]
    assert mine == stage2["effects"]
    for name, diagnostics in stage2["diagnostics"].items():
        assert {k: summary["diagnostics"][name][k] for k in diagnostics} == diagnostics


# --- the real golden set -------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run(real_docs):
    golden = load_golden(default_golden_path())
    index = SectionIndex(real_docs)
    return golden, index, run_configs(real_docs, golden.questions, EVALUATED, index)


@pytest.mark.corpus
@pytest.mark.parametrize("version", [1, pytest.param(2, marks=needs_v2)])
def test_the_golden_set_matches_the_pinned_corpus(real_docs, version):
    golden = load_golden(default_golden_path(version))
    assert check_against_corpus(golden, CorpusLock.load(default_lock_path()), SectionIndex(real_docs)) == []


@pytest.mark.corpus
@pytest.mark.parametrize("version", [1, pytest.param(2, marks=needs_v2)])
def test_no_question_was_edited_after_it_was_frozen(version):
    # Judging adds labels; it must never change what was asked.
    assert questions_fingerprint(load_golden(default_golden_path(version)).questions) == FROZEN_QUESTIONS


@needs_final_v2
@pytest.mark.corpus
@pytest.mark.models
def test_every_section_any_configuration_ranks_in_its_top_ten_has_been_judged(real_run, real_models):
    # MRR@10, nDCG@10 and Recall@10 read ten ranks deep, so the pool must reach that deep too.
    golden, _, results = real_run
    missing = unjudged(golden.questions, results, depth=10)
    assert not missing, f"{sum(len(v) for v in missing.values())} unjudged; run `raglab golden pool`"


@needs_final_v2
@pytest.mark.corpus
@pytest.mark.models
def test_real_scores_are_bounded_and_rankings_are_real_sections(real_run, real_models):
    golden, index, results = real_run
    for per_question in results.values():
        for r in per_question:
            assert all(0.0 <= v <= 1.0 for v in r.values.values())
            assert all(loc is None or loc in index for loc in r.ranking)


@pytest.mark.corpus
def test_the_reliability_studies_are_the_documented_draws_from_this_golden_set(real_docs):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    from samples import agreement_draw, unjudged_draw, unjudged_population

    golden = load_golden(V1)
    index = SectionIndex(real_docs)
    folder = V1.parent
    agreement = json.loads((folder / "v1-agreement.json").read_text(encoding="utf-8"))["items"]
    assert [(i["question"], i["location"]) for i in agreement] == [(q, loc) for q, loc, _, _ in agreement_draw(golden)]
    for item in agreement:  # the first-pass grade quoted is the grade in v1.yaml
        q = golden.by_id(item["question"])
        assert q.judgments[item["location"]] == item["first"], item
        assert item["is_target"] == (item["location"] == q.target)
    sample = json.loads((folder / "v1-unjudged-sample.json").read_text(encoding="utf-8"))
    population = unjudged_population(real_docs, index, golden)
    assert sample["population"] == {s: sum(v == s for v in population.values()) for s in ("focused", "scattered")}
    assert sorted((i["question"], i["location"]) for i in sample["items"]) == \
        sorted((q, loc) for q, loc, _ in unjudged_draw(population))
    for item in sample["items"]:  # ...and the unjudged sample really is unjudged there
        assert item["location"] in index and item["location"] not in golden.by_id(item["question"]).judgments


@needs_final_v2
@pytest.mark.corpus
@pytest.mark.models
def test_the_real_evaluation_reproduces_byte_for_byte(real_docs, real_models, tmp_path):
    # The claim the README makes about results/: same inputs, same bytes.
    golden = load_golden(default_golden_path())
    index = SectionIndex(real_docs)
    lock = CorpusLock.load(default_lock_path())
    for name in ("one", "two"):
        results = run_configs(real_docs, golden.questions, EVALUATED, index)
        summary = summarize(golden.questions, EVALUATED, results, resamples=2000)
        save_run(tmp_path / name, build_manifest(golden, lock, EVALUATED, resamples=2000, now=FIXED_TIME),
                 summary, golden.questions, results)
    for file in ("summary.json", "questions.jsonl", "manifest.json"):
        assert (tmp_path / "one" / file).read_bytes() == (tmp_path / "two" / file).read_bytes(), file



def test_summary_signs_and_holm_adjustment_on_synthetic_results():
    # The baseline never answers; "heading+bm25" always does; the rest are coin flips.
    rng = np.random.default_rng(4)
    questions = [Question(f"q{i:03d}", f"question {i}", "keyword", "t#a", MappingProxyType({"t#a": 2})) for i in range(40)]

    def result(value: float) -> QuestionResult:
        return QuestionResult({m: value for m in METRICS}, (), 0)

    results = {}
    for c in LEXICAL:
        if c == BASELINE:
            results[c.name] = [result(0.0) for _ in questions]
        elif c.name == "heading+bm25":
            results[c.name] = [result(1.0) for _ in questions]
        else:
            results[c.name] = [result(float(rng.random() < 0.3)) for _ in questions]
    summary = summarize(questions, LEXICAL, results, resamples=2000)
    best = summary["vs_baseline"]["heading+bm25"]["recall@5"]
    assert best["delta"] == pytest.approx(1.0) and best["lo"] == pytest.approx(1.0)  # direction: b - baseline
    rows = [row for per in summary["vs_baseline"].values() for row in per.values()]
    assert any(row["p_holm"] > row["p"] for row in rows)  # the correction really is applied
    effect = next(e for e in summary["effects"] if e["factor"] == "retriever" and e["metric"] == "recall@5")
    assert all(set(p) == {"delta", "lo", "hi", "p", "p_holm"} for p in effect["pairs"].values())
    # Direction of a switch: heading+bm25 always answers and heading+tfidf only sometimes, so
    # the heading tfidf -> bm25 pair must come out positive, with its interval above zero.
    pair = effect["pairs"]["heading+tfidf -> heading+bm25"]
    assert pair["delta"] > 0 and pair["lo"] > 0
    pairs = [p for e in summary["effects"] for p in e["pairs"].values()]
    assert any(p["p_holm"] > p["p"] for p in pairs)  # Holm is applied within each switch too


def test_the_default_golden_set_is_recorded_by_a_project_relative_path(tmp_path, monkeypatch, mini_corpus):
    files = hash_tree(mini_corpus)
    lock = CorpusLock("repo", "0.1.0", "c" * 40, fingerprint_of(files), files)
    monkeypatch.setenv("RAGLAB_HOME", str(tmp_path))
    (tmp_path / "golden").mkdir()
    dump_golden({"version": 1, "corpus": {"ref": lock.ref, "fingerprint": lock.fingerprint},
                 "questions": [{"id": "q001", "question": "Where?", "style": "keyword", "target": RAISE,
                                "judgments": {RAISE: 2}}]}, tmp_path / "golden" / "v2.yaml")
    manifest = build_manifest(load_golden(default_golden_path()), lock, FACTORIAL, resamples=1000, now=FIXED_TIME)
    assert manifest["golden"]["path"] == "golden/v2.yaml"
    assert str(tmp_path) not in json.dumps(manifest)


def test_the_code_hash_changes_when_the_code_does(tmp_path, monkeypatch):
    package = Path(evaluate.__file__).resolve().parent
    copy_dir = tmp_path / "raglab"
    shutil.copytree(package, copy_dir, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(evaluate, "__file__", str(copy_dir / "evaluate.py"))
    monkeypatch.setattr(evaluate, "_git_state", lambda _: {"git_commit": None, "git_dirty": None})
    before = code_version()["package_sha256"]
    (copy_dir / "metrics.py").write_text((copy_dir / "metrics.py").read_text() + "\n# edit\n")
    assert code_version()["package_sha256"] != before


@pytest.mark.corpus
def test_the_largest_budget_is_not_cut_short_by_retrieval_depth(real_docs):
    # q030 under the baseline: its top 200 chunks hold under 3,700 characters.
    from raglab.evaluate import ranked_spans
    from raglab.experiments import build_retrievers
    from raglab.metrics import BUDGETS, score

    golden = load_golden(V1)
    q = golden.by_id("q030")
    retriever = build_retrievers(real_docs, (BASELINE,))[BASELINE.name]
    spans = ranked_spans(retriever, q.text)
    assert sum(e - s for _, s, e in spans) >= max(BUDGETS)
    values, _ = score(spans, q.primary, q.judgments, SectionIndex(real_docs))
    assert values[f"budget_recall@{max(BUDGETS)}"] == 1.0



@pytest.mark.corpus
def test_the_regrade_log_matches_the_golden_set_and_the_readme():
    log = json.loads((V1.parent / "v1-regrade.json").read_text(encoding="utf-8"))["items"]
    golden = load_golden(V1)
    for item in log:
        assert golden.by_id(item["question"]).judgments[item["location"]] == item["final"], item
        assert (item["tiebreak"] is None) == (item["first"] == item["full_text"])
    changed = [i for i in log if i["final"] != i["first"]]
    assert (len(log), len(changed)) == (186, 28) and all(i["final"] > i["first"] for i in changed)
    assert sum(i["final"] == 2 and i["first"] != 2 for i in log) == 18


@needs_final_v2
@needs_results
@pytest.mark.corpus
@pytest.mark.models
def test_a_fresh_run_reproduces_the_committed_results(real_docs, real_models, tmp_path):
    # The README's claim about results/, checked against the committed files themselves.
    results = Path(__file__).resolve().parents[1] / "results"
    golden = load_golden(default_golden_path())
    lock = CorpusLock.load(default_lock_path())
    index = SectionIndex(real_docs)
    run = run_configs(real_docs, golden.questions, EVALUATED, index)
    save_run(tmp_path, build_manifest(golden, lock, EVALUATED), summarize(golden.questions, EVALUATED, run),
             golden.questions, run)
    for name in ("summary.json", "questions.jsonl"):
        assert (tmp_path / name).read_bytes() == (results / name).read_bytes(), name
    fresh, committed = (json.loads((d / "manifest.json").read_text()) for d in (tmp_path, results))
    for volatile in ("run_id", "created_utc"):
        fresh.pop(volatile), committed.pop(volatile)
    for manifest in (fresh, committed):  # git state and the machine depend on the checkout, not on the run
        manifest["code"].pop("git_commit"), manifest["code"].pop("git_dirty")
        manifest.pop("environment")
    assert fresh == committed


@needs_results
def test_every_table_in_the_readme_is_the_generated_one():
    from raglab.report import markdown_report

    stage = Path(__file__).resolve().parents[1]
    manifest, summary = load_run(stage / "results")
    readme = (stage / "README.md").read_text(encoding="utf-8")
    report = markdown_report(summary, manifest)
    tables = [block for block in report.split("\n\n") if block.startswith("| ")]
    assert len(tables) >= 8
    for table in tables:
        assert table in readme, table.splitlines()[0]


# --- Stage 3: the confirmatory chain, on synthetic results with known answers --------------------


def _synthetic(per_config, styles=("keyword", "paraphrase", "scenario"), n=60):
    """Questions on n/2 pages (two per page); per_config maps a name to a function (question index) -> value."""
    from raglab.experiments import EVALUATED

    questions = [Question(f"q{i:03d}", f"question {i}", styles[i % len(styles)], f"page{i // 2}#t",
                          MappingProxyType({f"page{i // 2}#t": 2})) for i in range(n)]
    results = {}
    for c in EVALUATED:
        value = per_config.get(c.name, lambda i: 0.3)
        results[c.name] = [QuestionResult({m: float(round(value(i)) if m in BINARY_METRICS else value(i))
                                           for m in METRICS}, (), 0) for i in range(n)]
    return questions, results


def test_the_chain_stops_at_the_first_link_that_is_not_significant():
    better, same = (lambda i: 0.9), (lambda i: 0.3)
    questions, results = _synthetic({
        "heading+bm25": same, "heading+hybrid+rerank": better,   # 1: clearly better -> significant
        "heading+hybrid": same,                                  # 2: no difference -> chain stops here
        "heading+dense": better,                                 # 3: would be significant, but is never reached
    })
    c = summarize(questions, EVALUATED, results, resamples=2000)["confirmatory"]
    first, second, third = c["chain"][:3]
    assert first["significant"] and first["verdict"] == "supported" and first["tested"]
    assert second["tested"] and not second["significant"] and second["verdict"] == "inconclusive"
    assert not third["tested"] and third["significant"] is None and third["p"] < 0.05
    assert third["verdict"].startswith("not reached")
    assert all(not t["tested"] for t in c["chain"][2:])


def test_supporting_metrics_count_only_when_the_headline_holds():
    questions, results = _synthetic({"heading+bm25": lambda i: 0.3, "heading+hybrid+rerank": lambda i: 0.3})
    c = summarize(questions, EVALUATED, results, resamples=2000)["confirmatory"]
    assert not c["chain"][0]["significant"]
    assert all(t["verdict"].startswith("exploratory") for t in c["supporting"])
    questions, results = _synthetic({"heading+bm25": lambda i: 0.3, "heading+hybrid+rerank": lambda i: 0.9})
    c = summarize(questions, EVALUATED, results, resamples=2000)["confirmatory"]
    assert [t["metric"] for t in c["supporting"]][:2] == ["success@5", "ndcg_answers@10"]
    assert all(t["p"] <= t["p_holm"] <= 1 and t["verdict"] == "supported" for t in c["supporting"])
    assert c["ndcg_gain_robust"] is True


def test_success_changes_count_fixed_and_broken_questions():
    questions, results = _synthetic({"heading+bm25": lambda i: float(i % 3 == 0),
                                     "heading+hybrid+rerank": lambda i: float(i % 2 == 0)})
    changes = summarize(questions, EVALUATED, results, resamples=1000)["confirmatory"]["success@5_changes"]
    # i % 6: 0 both, 1 neither, 2 fixed, 3 broken, 4 fixed, 5 neither -> per 60 questions: 10, 20, 20, 10
    assert changes == {"fixed": 20, "broken": 10, "both_found": 10, "both_missed": 20}


def test_a_gain_carried_by_whole_pages_is_marked_fragile_when_pages_disagree():
    # Within each page the two questions agree, so the per-question test overcounts the evidence.
    rng = np.random.default_rng(11)
    page_effect = rng.normal(0.08, 0.35, 30)
    questions, results = _synthetic({"heading+bm25": lambda i: 0.4,
                                     "heading+hybrid+rerank": lambda i: 0.4 + page_effect[i // 2]})
    headline = summarize(questions, EVALUATED, results, resamples=4000)["confirmatory"]["chain"][0]
    assert headline["page_level"]["p"] >= headline["p"]
    assert headline["fragile"] == (headline["p"] < 0.05 <= headline["page_level"]["p"])
    assert headline["resolvable"] > 0


def test_the_style_interaction_compares_paraphrase_with_keyword_questions():
    styles = ("keyword", "paraphrase")
    questions, results = _synthetic({"heading+bm25": lambda i: 0.5,
                                     "heading+dense": lambda i: 0.5 + (0.4 if i % 2 else 0.0)}, styles=styles)
    inter = summarize(questions, EVALUATED, results, resamples=2000)["interaction"]
    m = inter["metrics"]["ndcg_answers@10"]
    assert inter["pre_specified_metric"] == "ndcg_answers@10"
    assert m["paraphrase_mean"] == pytest.approx(0.4) and m["keyword_mean"] == pytest.approx(0.0)
    # every paraphrase gains exactly 0.4 and no keyword question gains, so the interval collapses onto 0.4
    assert m["difference"] == pytest.approx(0.4) and m["lo"] == pytest.approx(0.4) == m["hi"] and m["p"] < 0.05


def test_by_style_paired_differences_cover_the_headline_and_every_chain_link():
    questions, results = _synthetic({"heading+hybrid+rerank": lambda i: 0.8})
    block = summarize(questions, EVALUATED, results, resamples=1000)["by_style_paired"]
    pairs = block["paraphrase"]["pairs"]
    assert "heading+bm25 -> heading+hybrid+rerank" in pairs and "heading+bm25+rerank -> heading+hybrid+rerank" in pairs
    assert pairs["heading+bm25 -> heading+hybrid+rerank"]["mrr@10"]["delta"] == pytest.approx(0.5)
    assert block["paraphrase"]["n"] == 20


def test_new_diagnostics_report_related_slots_and_unreranked_windows(setup):
    _, docs, index, golden = setup
    results = run_configs(docs, golden.questions, EVALUATED, index)
    diagnostics = summarize(golden.questions, EVALUATED, results, resamples=1000)["diagnostics"]
    assert diagnostics["heading+bm25"]["unreranked_share_of_window"] is None
    assert set(diagnostics["heading+hybrid+rerank"]["unreranked_share_of_window"]) == {"1200", "2400", "4800", "9600"}
    assert diagnostics["heading+hybrid+rerank [depth 10]"]["rerank_depth"]["max"] <= 10
    for d in diagnostics.values():
        assert 0.0 <= d["related_share@10"] <= 1.0 and set(d["judged_in_budget"]) == {"1200", "2400", "4800", "9600"}


# --- Stage 3: drawing labels never scores; the label check; provenance hashes ---------------------


def test_ranking_only_runs_never_read_a_grade(setup, monkeypatch):
    _, docs, index, golden = setup

    def no_scoring(*args, **kwargs):
        raise AssertionError("a grade was read")

    monkeypatch.setattr(evaluate, "score", no_scoring)
    results = run_configs(docs, golden.questions, EVALUATED, index, scored=False)
    assert all(r.values == {} and r.ranking is not None for per in results.values() for r in per)
    assert unjudged(golden.questions, results, depth=10) is not None


def test_the_baseline_is_compared_with_the_29_factorial_cells_only(setup):
    _, docs, index, golden = setup
    summary = summarize(golden.questions, EVALUATED, run_configs(docs, golden.questions, EVALUATED, index), resamples=1000)
    assert len(summary["vs_baseline"]) == 29 and "heading+bm25+rerank" not in summary["vs_baseline"]
    assert "heading+bm25+rerank" in summary["summary"]  # scored and reported, just outside that Holm family


def _blind_rows(questions, results, grade_of):
    rows = []
    for i, q in enumerate(questions):
        for name in ("heading+bm25", "heading+hybrid+rerank"):
            for loc in results[name][i].ranking:
                if loc and not any(r["question"] == q.id and r["location"] == loc for r in rows):
                    rows.append({"question": q.id, "location": loc, "groups": ["union"], "new_in_v2": True,
                                 "first": q.judgments.get(loc, 0), "blind": grade_of(q, loc)})
    return rows


def _ranked_synthetic(winner_first: bool, n: int = 40):
    """Rankings where heading+hybrid+rerank puts the answer first and heading+bm25 puts it third (or the reverse)."""
    questions = [Question(f"q{i:03d}", f"question {i}", ("keyword", "paraphrase")[i % 2], f"page{i // 2}#t",
                          MappingProxyType({f"page{i // 2}#t": 2, f"x{i}#a": 0, f"x{i}#b": 0}))
                 for i in range(n)]
    results = {}
    for c in EVALUATED:
        rows = []
        for q in questions:
            good = (q.target, f"{q.id}x#a", f"{q.id}x#b")
            bad = (f"{q.id}x#a", f"{q.id}x#b", q.target)
            better = c.name == "heading+hybrid+rerank"
            ranking = good if better == winner_first else bad
            mrr = 1.0 if ranking[0] == q.target else 1 / 3
            rows.append(QuestionResult({m: (1.0 if mrr == 1.0 else 0.0) if m in BINARY_METRICS else mrr
                                        for m in METRICS}, ranking, 0))
        results[c.name] = rows
    return questions, results


def test_the_label_check_confirms_a_headline_only_when_the_blind_grades_agree():
    from raglab.evaluate import label_check, confirmatory, _matrix
    from raglab.stats import bootstrap_indices

    questions, results = _ranked_synthetic(winner_first=True)
    idx = bootstrap_indices(len(questions), 2000)
    confirmed = confirmatory(questions, _matrix(results), idx, 2000)
    assert confirmed["chain"][0]["verdict"] == "supported"
    agree = _blind_rows(questions, results, lambda q, loc: 2 if loc == q.target else 0)
    check = label_check(questions, results, confirmed, agree, idx, 2000)
    assert check["verdict"] == "supported" and check["stage4_inherits"] == "heading+hybrid+rerank"
    assert check["kappa_answer_vs_not"] == pytest.approx(1.0) and not check["lead_with_both_label_sets"]
    # The blind judge calls every target merely related: no answers on the blind labels, so no difference.
    doubt = _blind_rows(questions, results, lambda q, loc: 1)
    check = label_check(questions, results, confirmed, doubt, idx, 2000)
    assert check["on_blind_labels"]["delta"] == 0 and check["stage4_inherits"] == "heading+bm25"
    assert check["verdict"] == "supported on v2's labels, not confirmed by the blind round"
    assert check["lead_with_both_label_sets"]  # agreement collapses, so the README shows both label sets
    with pytest.raises(ValueError, match="no blind grade"):
        label_check(questions, results, confirmed, agree[1:], idx, 2000)


def test_an_unsupported_headline_stays_unsupported_whatever_the_blind_grades_say():
    from raglab.evaluate import label_check, confirmatory, _matrix
    from raglab.stats import bootstrap_indices

    questions, results = _ranked_synthetic(winner_first=False)
    idx = bootstrap_indices(len(questions), 2000)
    confirmed = confirmatory(questions, _matrix(results), idx, 2000)
    check = label_check(questions, results, confirmed,
                        _blind_rows(questions, results, lambda q, loc: 2 if loc == q.target else 0), idx, 2000)
    assert check["verdict"] == confirmed["chain"][0]["verdict"] == "contradicted"
    assert check["stage4_inherits"] == "heading+bm25"


def test_the_manifest_records_plain_file_hashes_of_the_plan_and_the_locks(setup, tmp_path):
    import hashlib

    manifest = _run(setup, tmp_path / "run")
    assert manifest["plan"]["sha256"] == hashlib.sha256((STAGE / "PLAN.md").read_bytes()).hexdigest()
    assert manifest["numerics"]["requirements_lock_sha256"] == \
        hashlib.sha256((STAGE / "requirements-lock.txt").read_bytes()).hexdigest()


@pytest.mark.models
@pytest.mark.corpus
def test_a_seeded_sample_of_the_cache_is_exactly_what_the_models_compute(real_docs, real_models):
    # The byte-for-byte tests read the cache; this one checks the cache against the models themselves.
    import random

    from raglab import neural
    from raglab.chunking import chunk_corpus, get_chunker
    from raglab.models import ModelLock

    embedder, scorer = neural.default_embedder(), neural.default_scorer()
    lock = ModelLock.load(STAGE.parent / "models.lock.json")
    fresh_embedder = neural.SentenceTransformerEmbedder(lock["embedder"])
    fresh_scorer = neural.CrossEncoderScorer(lock["reranker"])
    texts = sorted({c.index_text(True) for c in chunk_corpus(real_docs, get_chunker("heading"))})
    sample = random.Random(20261120).sample(texts, 40)
    assert np.array_equal(embedder.embed(sample, ""), fresh_embedder.embed(sample, ""))
    question = "How do I add a background task that runs after the response is sent?"
    assert np.array_equal(scorer.score(question, sample[:20]), fresh_scorer.score(question, sample[:20]))


@needs_final_v2
@pytest.mark.corpus
@pytest.mark.models
@pytest.mark.skipif(os.environ.get("RAGLAB_COLD") != "1", reason="about 15 minutes; set RAGLAB_COLD=1 to run it")
def test_the_evaluation_reproduces_from_an_empty_cache(real_docs, real_models, tmp_path, monkeypatch):
    # Run one computes every vector and score from the models; run two reads the filled cache. Same bytes.
    from raglab import neural

    golden = load_golden(default_golden_path())
    index = SectionIndex(real_docs)
    lock = CorpusLock.load(default_lock_path())
    real_cache = neural.default_cache_dir()
    for name, cache in (("cold", tmp_path / "empty-cache"), ("warm", real_cache)):
        monkeypatch.setattr(neural, "default_cache_dir", lambda cache=cache: cache)
        neural._DEFAULTS.clear()
        results = run_configs(real_docs, golden.questions, EVALUATED, index)
        save_run(tmp_path / name, build_manifest(golden, lock, EVALUATED, resamples=2000, now=FIXED_TIME),
                 summarize(golden.questions, EVALUATED, results, resamples=2000), golden.questions, results)
    neural._DEFAULTS.clear()
    for file in ("summary.json", "questions.jsonl"):
        assert (tmp_path / "cold" / file).read_bytes() == (tmp_path / "warm" / file).read_bytes(), file
