"""The rules that build golden/v2.yaml and its studies (analysis/v2_labels.py), on small made-up inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import v2_labels  # noqa: E402
from raglab.golden import GoldenSet, Question  # noqa: E402


def question(qid: str, target: str, judgments: dict, style: str = "keyword") -> Question:
    return Question(qid, f"text of {qid}", style, target, MappingProxyType(judgments))


def golden(questions: list[Question]) -> GoldenSet:
    return GoldenSet(1, "0.1.0", "f" * 64, tuple(questions), "0" * 64, Path("v1.yaml"),
                     {"version": 1, "questions": [{"id": q.id, "judgments": dict(q.judgments)} for q in questions]})


def test_calibration_strata_are_by_grade_split_by_heading_bm25_s_top_10_and_prefer_off_page_sections(monkeypatch):
    monkeypatch.setattr(v2_labels, "CALIBRATION", {"related": 2, "rejected": 2, "answers": 2})
    q1 = question("q001", "a#t", {"a#t": 2, "a#x": 1, "b#y": 0, "c#z": 1, "d#w": 2, "a#v": 2, "h#0": 0})
    q2 = question("q002", "e#t", {"e#t": 2, "e#u": 0, "f#u": 1, "g#u": 2})
    drawn = v2_labels.calibration_draw(golden([q1, q2]), {"q001": {"a#x", "d#w", "b#y"}, "q002": {"f#u"}})
    assert all(d["location"] not in ("a#t", "e#t") for d in drawn)  # targets are never calibration items
    by = {s: [d for d in drawn if d["stratum"] == s] for s in ("related", "rejected", "answers")}
    assert [len(v) for v in by.values()] == [2, 2, 2]
    assert {d["v1_grade"] for d in by["related"]} == {1} and {d["v1_grade"] for d in by["rejected"]} == {0}
    assert {d["v1_grade"] for d in by["answers"]} == {2}
    assert all(sum(d["in_heading_bm25_top10"] for d in v) == 1 for v in by.values())  # half and half
    # answers: one from inside heading+bm25's top 10 (d#w), one from outside, off the target page first (g#u)
    assert {(d["question"], d["location"]) for d in by["answers"]} == {("q001", "d#w"), ("q002", "g#u")}


def test_a_short_half_is_filled_from_the_other(monkeypatch):
    monkeypatch.setattr(v2_labels, "CALIBRATION", {"related": 0, "rejected": 4, "answers": 0})
    q1 = question("q001", "a#t", {"a#t": 2, "b#1": 0, "b#2": 0, "b#3": 0, "b#4": 0})
    drawn = v2_labels.calibration_draw(golden([q1]), {"q001": {"b#1"}})
    assert len(drawn) == 4 and sum(d["in_heading_bm25_top10"] for d in drawn) == 1


def test_batches_hold_whole_questions_in_order_and_close_at_the_size():
    assert v2_labels.batches(["q1", "q2", "q3", "q4", "q5"], {"q1": 30, "q2": 31, "q3": 5, "q5": 70}, size=60) == \
        [["q1", "q2"], ["q3", "q5"]]  # q4 has nothing to judge


def test_a_batch_must_return_every_item_exactly_once():
    expected = {("q1", "a#1"), ("q1", "a#2")}
    ok = [{"question": "q1", "location": "a#1", "grade": 2}, {"question": "q1", "location": "a#2", "grade": 0}]
    assert v2_labels.check_batch(expected, ok) == []
    assert v2_labels.check_batch(expected, None) == ["no grades returned"]
    assert "missing ('q1', 'a#2')" in v2_labels.check_batch(expected, ok[:1])
    assert any("twice" in p for p in v2_labels.check_batch(expected, ok + ok[:1]))
    assert any("unexpected" in p for p in v2_labels.check_batch(expected, ok + [{"question": "q1", "location": "z#9", "grade": 1}]))
    assert any("bad grade" in p for p in v2_labels.check_batch(expected, [ok[0], {**ok[1], "grade": True}]))


class FakeIndex:
    class Ref:
        def __init__(self, loc):
            self.doc_id, self.start, self.end = loc.split("#")[0], 0, 5
            self.heading_path = ("Page", loc)

    def get(self, loc):
        return self.Ref(loc)


class FakeDoc:
    def __init__(self, doc_id):
        self.doc_id, self.text = doc_id, f"text-{doc_id}-and-more"


def test_worksheets_carry_full_text_name_no_configuration_and_withhold_answer_controls(tmp_path):
    q1 = question("q001", "a#t", {"a#t": 2, "a#v": 2, "b#y": 0})
    pool_items = [{"question": "q001", "location": "c#new", "top10": ["heading+dense"], "window": []}]
    calibration = [{"question": "q001", "location": "a#v", "v1_grade": 2, "stratum": "answers"},
                   {"question": "q001", "location": "b#y", "v1_grade": 0, "stratum": "promotable"}]
    docs = [FakeDoc(d) for d in ("a", "b", "c")]
    manifest = v2_labels.worksheets(golden([q1]), docs, FakeIndex(), pool_items, calibration, tmp_path)
    assert len(manifest) == 1 and manifest[0]["items"] == 3
    folder = Path(manifest[0]["path"]).parent
    assert [p.name for p in folder.iterdir()] == ["worksheet.json"]  # alone in its folder
    text = Path(manifest[0]["path"]).read_text()
    entry = json.loads(text)[0]
    assert set(entry["answers"]) == {"a#t"}  # a#v is a hidden calibration item: not shown as an answer
    assert [c["location"] for c in entry["candidates"]] == ["a#v", "b#y", "c#new"]  # sorted, nothing marks the controls
    assert "heading+dense" not in text and "top10" not in text and "calibration" not in text
    assert entry["candidates"][2]["text"] == "text-"  # the section's text, as the index cuts it


def test_the_agreement_draw_keeps_its_paraphrase_share_and_skips_the_union(monkeypatch):
    monkeypatch.setattr(v2_labels, "AGREEMENT", {2: 5, 1: 0, 0: 0})
    qs = [question(f"q{i:03d}", f"t{i}#t", {f"t{i}#t": 2, **{f"n{i}#{j}": 2 for j in range(3)}},
                   style="paraphrase" if i < 2 else "keyword") for i in range(6)]
    g = golden(qs)
    pool_items = [{"question": q.id, "location": loc} for q in qs for loc in q.judgments if loc != q.target]
    union = {("q000", "n0#0")}
    drawn = v2_labels.agreement_draw(g, pool_items, union)
    assert len(drawn) == 5 and ("q000", "n0#0", 2) not in drawn
    assert sum(qid in ("q000", "q001") for qid, _, _ in drawn) >= 2  # ceil(0.4 x 5) paraphrase items


def test_blind_worksheets_hold_no_grades_answers_or_groups(tmp_path):
    items = [{"item": f"i{k}", "question_id": "q001", "groups": ["union"], "new_in_v2": True, "quote": False,
              "question": "Q?", "location": f"a#{k}", "heading_path": "Page", "text": "body"} for k in range(3)]
    manifest = v2_labels.blind_worksheets(items, tmp_path, size=2)
    assert [len(m["items"]) for m in manifest] == [2, 1]
    entry = json.loads(Path(manifest[0]["path"]).read_text())[0]
    assert set(entry) == {"item", "question", "location", "heading_path", "text"}


@pytest.mark.parametrize("quote, holds", [
    ("Use HTTPException", True), ("use   HTTPException\n to return", True), ("", False),
    ("HTTPException with code 418", False),
])
def test_a_quote_counts_only_if_it_is_in_the_section(quote, holds):
    text = "To fail a request, use HTTPException\nto return an error."
    assert v2_labels.quote_holds(quote.replace("Use", "use") if quote.startswith("Use") else quote, text) is holds


def test_the_bridge_draw_is_a_fixed_40_from_v1_s_study_outside_the_union():
    items = [{"question": f"q{i:03d}", "location": f"x#{i}", "first": 1, "second": 1} for i in range(100)]
    a, b = v2_labels.bridge_draw(items, set()), v2_labels.bridge_draw(list(reversed(items)), set())
    assert len(a) == 40 and a == b
    union = {(i["question"], i["location"]) for i in a[:10]}
    c = v2_labels.bridge_draw(items, union)
    assert len(c) == 40 and not union & {(i["question"], i["location"]) for i in c}


def test_the_quote_judge_sees_everything_outside_the_union_and_the_union_s_new_answers():
    q = question("q001", "a#t", {"a#t": 2, "b#new2": 2, "c#new1": 1, "d#old2": 2, "e#agree": 0})
    union = [("q001", "a#t"), ("q001", "b#new2"), ("q001", "c#new1"), ("q001", "d#old2")]
    items = v2_labels.blind_items(golden([q]), [FakeDoc(d) for d in "abcde"], FakeIndex(), union,
                                  [("q001", "e#agree", 0)], [], new_keys={("q001", "b#new2"), ("q001", "c#new1"),
                                                                           ("q001", "e#agree")})
    quoted = {i["location"] for i in items if i["quote"]}
    assert quoted == {"b#new2", "e#agree"}  # a new answer in the union, and the agreement item
    assert {i["location"]: i["groups"] for i in items}["e#agree"] == ["agreement"]


def test_a_blind_batch_must_answer_every_item_once_with_a_valid_answer():
    ok = [{"item": "i1", "grade": 2}, {"item": "i2", "grade": 0}]
    assert v2_labels.check_blind_batch(["i1", "i2"], ok, quote=False) == []
    assert v2_labels.check_blind_batch(["i1", "i2"], ok[:1], quote=False) == ["missing i2"]
    assert any("twice" in p for p in v2_labels.check_blind_batch(["i1", "i2"], ok + ok[:1], quote=False))
    assert any("bad grade" in p for p in v2_labels.check_blind_batch(["i1"], [{"item": "i1", "grade": "2"}], quote=False))
    good_quote = [{"item": "i1", "answers": True, "quote": "x"}]
    assert v2_labels.check_blind_batch(["i1"], good_quote, quote=True) == []
    assert v2_labels.check_blind_batch(["i1"], [{"item": "i1", "answers": "yes", "quote": "x"}], quote=True)


def test_a_batch_that_passed_can_never_be_judged_again(tmp_path):
    manifest = [{"label": "blind-01", "items": ["i1"]}]
    first = tmp_path / "first.json"
    first.write_text(json.dumps([{"label": "blind-01", "grades": [{"item": "i1", "grade": 1}]}]))
    again = tmp_path / "again.json"
    again.write_text(json.dumps([{"label": "blind-01", "grades": [{"item": "i1", "grade": 2}]}]))
    with pytest.raises(SystemExit, match="passed validation"):
        v2_labels._round_answers(first, [again], manifest, "grades", quote=False)
    first.write_text(json.dumps([{"label": "blind-01", "grades": []}]))  # failed: nothing returned for i1
    answers, log = v2_labels._round_answers(first, [again], manifest, "grades", quote=False)
    assert answers["i1"]["grade"] == 2 and log[0]["batch"] == "blind-01" and log[0]["first_run"] == []
