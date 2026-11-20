from __future__ import annotations

import copy

import pytest
import yaml

from raglab.corpus import CorpusLock, fingerprint_of, hash_tree, load_corpus
from raglab.golden import (
    SAMPLE_EXCLUDED, GoldenError, area_of, check_against_corpus, dump_golden, load_golden, merge_judgments,
    questions_fingerprint, sample_targets,
)
from raglab.sections import SectionIndex

RAISE = "tutorial/first-steps#raise-errors"
DOCS = "tutorial/first-steps#interactive-api-docs"


@pytest.fixture
def lock(mini_corpus):
    files = hash_tree(mini_corpus)
    return CorpusLock("repo", "0.1.0", "c" * 40, fingerprint_of(files), files)


@pytest.fixture
def index(mini_corpus):
    return SectionIndex(load_corpus(mini_corpus).documents)


@pytest.fixture
def raw(lock):
    return {
        "version": 1,
        "corpus": {"ref": lock.ref, "fingerprint": lock.fingerprint},
        "questions": [
            {"id": "q001", "question": "How do I return a 404?", "style": "paraphrase",
             "target": RAISE, "judgments": {RAISE: 2, DOCS: 0}},
            {"id": "q002", "question": "Where is Swagger UI?", "style": "keyword",
             "target": DOCS, "judgments": {DOCS: 2}},
        ],
    }


def write(tmp_path, raw):
    path = tmp_path / "golden.yaml"
    dump_golden(raw, path)
    return path


def test_a_valid_file_loads_with_its_answers(tmp_path, raw):
    gs = load_golden(write(tmp_path, raw))
    assert [q.id for q in gs.questions] == ["q001", "q002"]
    q = gs.by_id("q001")
    assert q.primary == frozenset({RAISE}) and q.relevant == {RAISE: 2}
    assert len(gs.sha256) == 64


def test_written_files_have_a_header_and_unix_line_endings(tmp_path, raw):
    data = write(tmp_path, raw).read_bytes()
    assert data.startswith(b"# RAG Eval Lab") and b"\r\n" not in data


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("corpus"), "missing 'corpus'"),
        (lambda r: r.update(extra=1), "unknown key 'extra'"),
        (lambda r: r.update(version=True), "version"),
        (lambda r: r["corpus"].update(fingerprint="abc"), "corpus.fingerprint"),
        (lambda r: r["questions"][0].update(id="Q1"), "id must look like q001"),
        (lambda r: r["questions"][1].update(id="q001"), "duplicate id"),
        (lambda r: r["questions"][1].update(question="how do I  return a 404?"), "same question as q001"),
        (lambda r: r["questions"][0].update(style="clever"), "style must be one of"),
        (lambda r: r["questions"][0]["judgments"].update({DOCS: 3}), "grades are 0, 1 or 2"),
        (lambda r: r["questions"][0]["judgments"].update({DOCS: True}), "grades are 0, 1 or 2"),
        (lambda r: r["questions"][0]["judgments"].update({RAISE: 1}), "no section judged 2"),
        (lambda r: r["questions"][1].update(target=RAISE), "must be judged 2"),
        # The target judged 1 while another section answers: the target rule itself must fire.
        (lambda r: r["questions"][0]["judgments"].update({RAISE: 1, DOCS: 2}), "must be judged 2"),
        (lambda r: r["questions"][0].update(id="q001\n"), "id must look like q001"),
        (lambda r: r["corpus"].update(fingerprint=r["corpus"]["fingerprint"] + "\n"), "corpus.fingerprint"),
        (lambda r: r["questions"][0].update(judgments=[]), "judgments must be a non-empty mapping"),
        (lambda r: r["questions"][0].update(question="x" * 301), "longer than 300"),
        (lambda r: r.update(skipped=[{"location": RAISE}]), "skipped"),
    ],
)
def test_malformed_files_are_rejected_with_a_reason(tmp_path, raw, mutate, message):
    bad = copy.deepcopy(raw)
    mutate(bad)
    with pytest.raises(GoldenError, match=message):
        load_golden(write(tmp_path, bad))


def test_every_problem_is_reported_at_once(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["questions"][0]["style"] = "clever"
    bad["questions"][1]["id"] = "nope"
    with pytest.raises(GoldenError) as err:
        load_golden(write(tmp_path, bad))
    assert "2 problem(s)" in str(err.value)


def test_not_yaml_is_a_clean_error(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(GoldenError, match="not valid UTF-8 YAML"):
        load_golden(path)


def test_labels_must_point_at_sections_with_text(tmp_path, raw, lock, index):
    gs = load_golden(write(tmp_path, raw))
    assert check_against_corpus(gs, lock, index) == []
    bad = copy.deepcopy(raw)
    bad["questions"][0]["judgments"]["tutorial/first-steps#gone"] = 0
    problems = check_against_corpus(load_golden(write(tmp_path, bad)), lock, index)
    assert problems == ["q001: tutorial/first-steps#gone is not a section in the corpus"]


def test_labels_made_against_another_corpus_are_flagged(tmp_path, raw, lock, index):
    bad = copy.deepcopy(raw)
    bad["corpus"]["fingerprint"] = "0" * 64
    problems = check_against_corpus(load_golden(write(tmp_path, bad)), lock, index)
    assert any("re-judge" in p for p in problems)


def test_merge_adds_judgments_but_never_changes_one(raw):
    merged, added, changed = merge_judgments(raw, {"q002": {RAISE: 1}})
    assert (added, changed) == (1, 0) and merged["questions"][1]["judgments"] == {DOCS: 2, RAISE: 1}
    assert raw["questions"][1]["judgments"] == {DOCS: 2}  # the input is untouched
    _, added, _ = merge_judgments(raw, {"q001": {RAISE: 2}})  # same grade again: a no-op
    assert added == 0
    with pytest.raises(GoldenError, match="already judged 0, not 1"):
        merge_judgments(raw, {"q001": {DOCS: 1}})
    with pytest.raises(GoldenError, match="no such question"):
        merge_judgments(raw, {"q999": {DOCS: 1}})
    with pytest.raises(GoldenError, match="grades are 0, 1 or 2"):
        merge_judgments(raw, {"q002": {RAISE: 5}})


def test_regrade_changes_grades_on_request_but_never_a_target(raw):
    merged, added, changed = merge_judgments(raw, {"q001": {DOCS: 1}}, regrade=True)
    assert (added, changed) == (0, 1) and merged["questions"][0]["judgments"][DOCS] == 1
    with pytest.raises(GoldenError, match="stays judged 2"):
        merge_judgments(raw, {"q001": {RAISE: 1}}, regrade=True)


def test_merged_judgments_list_the_target_first_then_by_grade(raw):
    merged, *_ = merge_judgments(raw, {"q001": {"index#home": 1, "aaa#first": 0}})
    assert list(merged["questions"][0]["judgments"]) == [RAISE, "index#home", "aaa#first", DOCS]


def test_questions_fingerprint_ignores_judgments_but_not_questions(tmp_path, raw):
    before = questions_fingerprint(load_golden(write(tmp_path, raw)).questions)
    judged, *_ = merge_judgments(raw, {"q002": {RAISE: 1}})
    assert questions_fingerprint(load_golden(write(tmp_path, judged)).questions) == before
    edited = copy.deepcopy(raw)
    edited["questions"][0]["question"] = "How do I return a 404 error?"
    assert questions_fingerprint(load_golden(write(tmp_path, edited)).questions) != before


# --- sampling --------------------------------------------------------------------------------


@pytest.mark.corpus
def test_sampling_is_reproducible_stratified_and_skips_excluded_pages(real_docs):
    index = SectionIndex(real_docs)
    first = sample_targets(index, 100, seed=20261120)
    assert [s.ref.location for s in first] == [s.ref.location for s in sample_targets(index, 100, seed=20261120)]
    assert [s.ref.location for s in first] != [s.ref.location for s in sample_targets(index, 100, seed=1)]
    assert len({s.ref.location for s in first}) == 100
    for s in first:
        assert s.ref.length >= 150
        assert not any(s.ref.doc_id.startswith(p) if p.endswith("/") else s.ref.doc_id == p for p in SAMPLE_EXCLUDED)
    per_doc: dict[str, int] = {}
    for s in first:
        per_doc[s.ref.doc_id] = per_doc.get(s.ref.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2
    areas = {a: sum(s.area == a for s in first) for a in {s.area for s in first}}
    assert areas["tutorial"] > areas["advanced"] > areas["how-to"]  # in proportion to the docs
    assert {s.style for s in first} == {"keyword", "paraphrase", "scenario"}


def test_area_of_groups_top_level_pages():
    assert area_of("tutorial/body") == "tutorial" and area_of("async") == "(top)"


def test_agreement_report_counts_matches_and_targets():
    from raglab.golden import agreement_report

    items = [
        {"question": "q1", "location": "a", "first": 2, "second": 2, "is_target": True},
        {"question": "q1", "location": "b", "first": 2, "second": 1, "is_target": False},
        {"question": "q2", "location": "c", "first": 0, "second": 0, "is_target": False},
        {"question": "q2", "location": "d", "first": 1, "second": 1, "is_target": False},
    ]
    r = agreement_report(items)
    assert r["exact"] == 0.75
    assert r["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert r["agreement_by_first_grade"] == {0: 1.0, 1: 1.0, 2: 0.5}
    assert (r["targets"], r["targets_confirmed"]) == (1, 1)
    assert 0 < r["kappa"] < 1 and r["kappa_linear"] > r["kappa"]
    with pytest.raises(ValueError):
        agreement_report([{"question": "q", "location": "x", "first": 3, "second": 0}])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("version: 1\nversion: 2\n", "duplicate key 'version'"),
        ("questions:\n- judgments:\n    a#x: 2\n    a#x: 0\n", "duplicate key 'a#x'"),
        ("base: &b {x: 1}\nother: *b\n", "anchors and aliases are not allowed"),
        ("? [1, 2]\n: 3\n", "a key must be text or a number"),
    ],
)
def test_yaml_that_could_lose_or_multiply_labels_is_refused(tmp_path, text, message):
    path = tmp_path / "golden.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(GoldenError, match=message):
        load_golden(path)



@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("questions: " + "[" * 3000 + "]" * 3000 + "\n", "not a golden set"),
        ("version: " + "9" * 5001 + "\n", "not a golden set"),
    ],
)
def test_hostile_yaml_is_a_clean_error_not_a_traceback(tmp_path, text, message):
    path = tmp_path / "golden.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(GoldenError, match=message):
        load_golden(path)


def test_labels_for_another_release_are_flagged_even_with_a_matching_fingerprint(tmp_path, raw, lock, index):
    other = copy.deepcopy(raw)
    other["corpus"]["ref"] = "0.999.0"
    problems = check_against_corpus(load_golden(write(tmp_path, other)), lock, index)
    assert any("labelled against FastAPI 0.999.0" in p for p in problems)


def test_allocation_breaks_exact_ties_by_area_name():
    from raglab.golden import _allocate

    # Remainders for a0 and a3 tie exactly (4/7); the name decides, not float noise.
    alloc = _allocate(984, {"a0": 346, "a1": 168, "a2": 120, "a3": 192, "a4": 322})
    assert sum(alloc.values()) == 984 and (alloc["a0"], alloc["a3"]) == (297, 164)


def test_agreement_report_rejects_booleans_and_says_n_a_for_empty_grades():
    from raglab.golden import agreement_report

    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        agreement_report([{"question": "q", "location": "x", "first": True, "second": 1}])
    r = agreement_report([{"question": "q", "location": "x", "first": 2, "second": 2},
                          {"question": "q", "location": "y", "first": 0, "second": 0}])
    assert r["agreement_by_first_grade"][1] is None and r["exact"] == 1.0


def test_strict_json_refuses_repeated_keys():
    from raglab.golden import strict_json

    with pytest.raises(ValueError, match="duplicate key 'q001'"):
        strict_json('{"q001": {"a": 2}, "q001": {"b": 0}}')
    with pytest.raises(ValueError, match="duplicate key 'a'"):
        strict_json('{"q001": {"a": 2, "a": 0}}')
    assert strict_json('{"q001": {"a": 2}}') == {"q001": {"a": 2}}


def test_the_file_reads_in_a_human_order(tmp_path, raw):
    # Even when the data arrives alphabetised (as a YAML round trip leaves it), each
    # question is written before its labels and the rubric before the questions.
    shuffled = yaml.safe_load(yaml.safe_dump(raw, sort_keys=True))
    shuffled["rubric"] = {0: "no", 1: "related", 2: "answers"}
    text = write(tmp_path, shuffled).read_text(encoding="utf-8")
    assert text.index("rubric:") < text.index("questions:")
    assert "- id: q001\n  question: How do I return a 404?\n  style: paraphrase\n  target:" in text
    assert text.index("2: answers") < text.index("0: 'no'")
