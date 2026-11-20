from __future__ import annotations

import json
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
)
from raglab.experiments import BASELINE, FACTORIAL, RetrievalConfig
from raglab.golden import Question, check_against_corpus, default_golden_path, dump_golden, load_golden, questions_fingerprint
from raglab.metrics import METRICS
from raglab.sections import SectionIndex

RAISE = "tutorial/first-steps#raise-errors"
DOCS = "tutorial/first-steps#interactive-api-docs"
HOME = "index#home"
FROZEN_QUESTIONS = "c4673dddd9bdce5b73ae05f94672ad9e603c591d80b9469133ea442b821fcb4a"
FIXED_TIME = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


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
    assert list(results) == [c.name for c in FACTORIAL] and len(results) == 12
    for per_question in results.values():
        assert len(per_question) == 3
        for r in per_question:
            assert set(r.values) == set(METRICS)
            assert all(0.0 <= v <= 1.0 for v in r.values.values())
            assert len(r.ranking) <= 10


def test_the_factorial_is_complete_and_contains_the_baseline_once():
    cells = [c.cell for c in FACTORIAL]
    assert len(set(cells)) == 12 and BASELINE.cell in cells
    assert sum(c.name == BASELINE.name for c in FACTORIAL) == 1


def test_summary_has_intervals_comparisons_and_switch_effects(setup):
    _, docs, index, golden = setup
    results = run_configs(docs, golden.questions, FACTORIAL, index)
    summary = summarize(golden.questions, FACTORIAL, results, resamples=1000)
    for name in summary["configs"]:
        for m in METRICS:
            cell = summary["summary"][name][m]
            assert cell["lo"] <= cell["mean"] <= cell["hi"]
    assert BASELINE.name not in summary["vs_baseline"] and len(summary["vs_baseline"]) == 11
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
    pairs = {(e["factor"], str(e["from"]), str(e["to"])): len(e["pairs"]) for e in summary["effects"]}
    assert pairs[("retriever", "tfidf", "bm25")] == 6        # 3 chunkers x 2 breadcrumb settings
    assert pairs[("breadcrumb", "False", "True")] == 6       # 3 chunkers x 2 retrievers
    assert pairs[("chunker", "paragraph", "heading")] == 4   # 2 retrievers x 2 breadcrumb settings
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


# --- the real golden set -------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run(real_docs):
    golden = load_golden(default_golden_path())
    index = SectionIndex(real_docs)
    return golden, index, run_configs(real_docs, golden.questions, FACTORIAL, index)


@pytest.mark.corpus
def test_the_golden_set_matches_the_pinned_corpus(real_docs):
    golden = load_golden(default_golden_path())
    assert check_against_corpus(golden, CorpusLock.load(default_lock_path()), SectionIndex(real_docs)) == []


@pytest.mark.corpus
def test_no_question_was_edited_after_it_was_frozen():
    # Judging adds labels; it must never change what was asked.
    assert questions_fingerprint(load_golden(default_golden_path()).questions) == FROZEN_QUESTIONS


@pytest.mark.corpus
def test_every_section_any_configuration_ranks_in_its_top_ten_has_been_judged(real_run):
    # MRR@10, nDCG@10 and Recall@10 read ten ranks deep, so the pool must reach that deep too.
    golden, _, results = real_run
    missing = unjudged(golden.questions, results, depth=10)
    assert not missing, f"{sum(len(v) for v in missing.values())} unjudged; run `raglab golden pool`"


@pytest.mark.corpus
def test_real_scores_are_bounded_and_rankings_are_real_sections(real_run):
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

    golden = load_golden(default_golden_path())
    index = SectionIndex(real_docs)
    folder = default_golden_path().parent
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


@pytest.mark.corpus
def test_the_real_evaluation_reproduces_byte_for_byte(real_docs, tmp_path):
    # The claim the README makes about results/: same inputs, same bytes.
    golden = load_golden(default_golden_path())
    index = SectionIndex(real_docs)
    lock = CorpusLock.load(default_lock_path())
    for name in ("one", "two"):
        results = run_configs(real_docs, golden.questions, FACTORIAL, index)
        summary = summarize(golden.questions, FACTORIAL, results, resamples=2000)
        save_run(tmp_path / name, build_manifest(golden, lock, FACTORIAL, resamples=2000, now=FIXED_TIME),
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
    for c in FACTORIAL:
        if c == BASELINE:
            results[c.name] = [result(0.0) for _ in questions]
        elif c.name == "heading+bm25":
            results[c.name] = [result(1.0) for _ in questions]
        else:
            results[c.name] = [result(float(rng.random() < 0.3)) for _ in questions]
    summary = summarize(questions, FACTORIAL, results, resamples=2000)
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
                                "judgments": {RAISE: 2}}]}, tmp_path / "golden" / "v1.yaml")
    manifest = build_manifest(load_golden(default_golden_path()), lock, FACTORIAL, resamples=1000, now=FIXED_TIME)
    assert manifest["golden"]["path"] == "golden/v1.yaml"
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

    golden = load_golden(default_golden_path())
    q = golden.by_id("q030")
    retriever = build_retrievers(real_docs, (BASELINE,))[BASELINE.name]
    spans = ranked_spans(retriever, q.text)
    assert sum(e - s for _, s, e in spans) >= max(BUDGETS)
    values, _ = score(spans, q.primary, q.judgments, SectionIndex(real_docs))
    assert values[f"budget_recall@{max(BUDGETS)}"] == 1.0



@pytest.mark.corpus
def test_the_regrade_log_matches_the_golden_set_and_the_readme():
    log = json.loads((default_golden_path().parent / "v1-regrade.json").read_text(encoding="utf-8"))["items"]
    golden = load_golden(default_golden_path())
    for item in log:
        assert golden.by_id(item["question"]).judgments[item["location"]] == item["final"], item
        assert (item["tiebreak"] is None) == (item["first"] == item["full_text"])
    changed = [i for i in log if i["final"] != i["first"]]
    assert (len(log), len(changed)) == (186, 28) and all(i["final"] > i["first"] for i in changed)
    assert sum(i["final"] == 2 and i["first"] != 2 for i in log) == 18


@pytest.mark.corpus
def test_a_fresh_run_reproduces_the_committed_results(real_docs, tmp_path):
    # The README's claim about results/, checked against the committed files themselves.
    results = Path(__file__).resolve().parents[1] / "results"
    golden = load_golden(default_golden_path())
    lock = CorpusLock.load(default_lock_path())
    index = SectionIndex(real_docs)
    run = run_configs(real_docs, golden.questions, FACTORIAL, index)
    save_run(tmp_path, build_manifest(golden, lock, FACTORIAL), summarize(golden.questions, FACTORIAL, run),
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
