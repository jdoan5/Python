"""Build golden/v2.yaml, and the studies that check its new labels (PLAN.md, "Labels").

Every set below is a rule applied to committed inputs, so `verify` can show
that each file in golden/ is exactly what the rule produces.

  init         write golden/v2.yaml: v1's 98 questions and every v1 judgment, as version 2
  worksheets   the judging round. The pool is every unjudged section in some
               configuration's top 10, plus every unjudged section inside the
               4.8k window of a heading or fixed configuration. 200 hidden
               calibration items (v1 labels, graded again) are mixed in. One
               worksheet per batch goes into a folder of its own, outside the
               project. Writes v2-provenance.json.
  merge        check every batch's grades, merge the pool's into v2.yaml, and write
               v2-judgments.json (every grade as returned) and v2-calibration.json
  blind        the blind round: every section in either headline configuration's
               top 10, an agreement sample of v2's other new labels, and 40 bridge
               items from v1's agreement study; question and section only
  merge-blind  write v2-blind.json and v2-quote.json from the blind and quote judges' answers
  verify       check that the committed files are exactly these draws

    cd "Stage 3" && ../.venv/bin/python analysis/v2_labels.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.evaluate import code_version, run_configs, unjudged, unjudged_in_windows  # noqa: E402
from raglab.experiments import CHALLENGER, EVALUATED, INCUMBENT  # noqa: E402
from raglab.golden import GoldenSet, default_golden_path, dump_golden, load_golden, merge_judgments  # noqa: E402
from raglab.metrics import HEADLINE_BUDGET  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402

SEED = 20261120
WINDOW_CHUNKERS = ("heading", "fixed")   # their 4.8k windows are judged in full; paragraph windows aren't
BATCH_ITEMS = 60                         # a batch closes once it holds at least this many items
CALIBRATION = {"promotable": 100, "answers": 100}  # v1 grade 0/1, and non-target v1 grade 2
AGREEMENT = {2: 100, 1: 60, 0: 60}       # the agreement sample of v2's other new labels, by first grade
PARAPHRASE_SHARE = 0.4                   # at least this share of each agreement stratum is paraphrase questions
BRIDGE = 40                              # items re-drawn from v1's agreement study
JUDGE_MODEL = "claude-opus-5-5"          # the model v1's judges ran as, and v2's
GOLDEN = default_golden_path(1).parent
PROMPTS = GOLDEN / "prompts"


def rank_key(kind: str, *parts: str) -> str:
    return hashlib.sha256(":".join((str(SEED), kind, *parts)).encode("utf-8")).hexdigest()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context():
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    return docs, SectionIndex(docs)


def body(docs_by_id, index, location: str) -> str:
    ref = index.get(location)
    return docs_by_id[ref.doc_id].text[ref.start:ref.end]


def page_of(location: str) -> str:
    return location.split("#", 1)[0]


# --- init ---------------------------------------------------------------------------------------


def init_v2() -> Path:
    v1, v2 = default_golden_path(1), default_golden_path(2)
    if v2.exists():
        raise SystemExit(f"{v2.name} already exists; it is never rebuilt over its labels")
    raw = json.loads(json.dumps(load_golden(v1).raw))
    raw["version"] = 2
    dump_golden(raw, v2)
    return v2


# --- the judging round --------------------------------------------------------------------------


def pool(golden: GoldenSet, results) -> list[dict]:
    """Every section v2 must judge, with which configurations surfaced it and how."""
    top10 = unjudged(golden.questions, results, depth=10)
    windows = unjudged_in_windows(golden.questions, results, HEADLINE_BUDGET,
                                  [c.name for c in EVALUATED if c.chunker in WINDOW_CHUNKERS])
    items = []
    for q in golden.questions:
        for loc in sorted(set(top10.get(q.id, {})) | set(windows.get(q.id, {}))):
            items.append({"question": q.id, "location": loc,
                          "top10": sorted(top10.get(q.id, {}).get(loc, [])),
                          "window": sorted(windows.get(q.id, {}).get(loc, []))})
    return items


def calibration_draw(v1: GoldenSet, incumbent_top10: dict[str, set[str]]) -> list[dict]:
    """200 v1 labels to grade again, hidden among the new candidates.

    Two strata: v1 grades 0 or 1 (a lenient round could promote them) and
    non-target v1 answers (a strict round could demote them). Each stratum is
    half sections in heading+bm25's top 10, the side of the headline whose
    labels all come from v1, and half not. Within each half, off-target-page
    sections come first, because that is where v2's new labels fall, then
    sha256 order. A half that runs short is filled from the other half.
    """
    drawn = []
    for stratum, quota in CALIBRATION.items():
        halves = {True: [], False: []}
        for q in v1.questions:
            for loc, grade in q.judgments.items():
                if loc == q.target or (grade == 2) != (stratum == "answers"):
                    continue
                halves[loc in incumbent_top10.get(q.id, set())].append(
                    (page_of(loc) == page_of(q.target), rank_key("calibration", q.id, loc), q.id, loc, grade))
        for half in halves.values():
            half.sort()
        take = {True: halves[True][: quota // 2], False: halves[False][: quota // 2]}
        for side in (True, False):
            short = quota // 2 - len(take[side])
            if short > 0:
                take[not side] = halves[not side][: quota // 2 + short]
        for side in (True, False):
            for on_page, _, qid, loc, grade in take[side]:
                drawn.append({"question": qid, "location": loc, "v1_grade": grade, "stratum": stratum,
                              "in_heading_bm25_top10": side, "on_target_page": on_page})
    return sorted(drawn, key=lambda d: (d["question"], d["location"]))


def batches(question_ids: list[str], items_per_question: dict[str, int], size: int = BATCH_ITEMS) -> list[list[str]]:
    """Whole questions in id order; a batch closes once it holds at least `size` items."""
    out, current, count = [], [], 0
    for qid in question_ids:
        if not items_per_question.get(qid):
            continue
        current.append(qid)
        count += items_per_question[qid]
        if count >= size:
            out.append(current)
            current, count = [], 0
    if current:
        out.append(current)
    return out


def worksheets(v1: GoldenSet, docs, index, pool_items: list[dict], calibration: list[dict], out: Path) -> list[dict]:
    """One worksheet per batch, each alone in its own folder. Returns the batch manifest.

    A worksheet carries what a v1 worksheet carried and nothing more: the
    question, its v1 answers in full (only to show what it asks), and each
    candidate's location, heading path and full text, sorted by location.
    A calibration item that v1 graded 2 is left out of its own question's
    answers, so the judge can't see that it is one.
    """
    by_id = {d.doc_id: d for d in docs}
    candidates: dict[str, list[str]] = {}
    for item in pool_items + calibration:
        candidates.setdefault(item["question"], []).append(item["location"])
    withheld = {(c["question"], c["location"]) for c in calibration if c["v1_grade"] == 2}
    order = [q.id for q in v1.questions]
    manifest = []
    for k, qids in enumerate(batches(order, {q: len(v) for q, v in candidates.items()}), 1):
        entries = []
        for qid in qids:
            q = v1.by_id(qid)
            entries.append({
                "id": qid,
                "question": q.text,
                "answers": {loc: body(by_id, index, loc) for loc in sorted(q.primary) if (qid, loc) not in withheld},
                "candidates": [{"location": loc, "heading_path": " > ".join(index.get(loc).heading_path),
                                "text": body(by_id, index, loc)} for loc in sorted(candidates[qid])],
            })
        folder = out / f"batch-{k:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "worksheet.json"
        path.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        manifest.append({"label": f"batch-{k:02d}", "path": str(path), "questions": qids,
                         "items": sum(len(e["candidates"]) for e in entries), "sha256": sha256_of(path)})
    return manifest


def check_batch(expected: set[tuple[str, str]], graded: list[dict] | None) -> list[str]:
    """Problems with one batch's grades: every expected item exactly once, with a valid grade."""
    if graded is None:
        return ["no grades returned"]
    seen: dict[tuple[str, str], int] = {}
    problems = []
    for g in graded:
        key = (g.get("question"), g.get("location"))
        if key not in expected:
            problems.append(f"unexpected item {key}")
        elif key in seen:
            problems.append(f"graded twice: {key}")
        elif g.get("grade") not in (0, 1, 2) or isinstance(g.get("grade"), bool):
            problems.append(f"bad grade for {key}: {g.get('grade')!r}")
        seen[key] = g.get("grade")
    problems += [f"missing {key}" for key in sorted(expected - set(seen))]
    return problems


# --- the blind round --------------------------------------------------------------------------


def headline_union(golden: GoldenSet, results) -> list[tuple[str, str]]:
    """Every section either headline configuration ranks in its top 10: every label the primary metric reads."""
    out = []
    for i, q in enumerate(golden.questions):
        locs = {loc for name in (INCUMBENT.name, CHALLENGER.name) for loc in results[name][i].ranking if loc}
        out += [(q.id, loc) for loc in sorted(locs)]
    return out


def agreement_draw(v2: GoldenSet, pool_items: list[dict], union: set[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """v2's other new labels, stratified by grade; at least 40% of each stratum from paraphrase questions."""
    styles = {q.id: q.style for q in v2.questions}
    drawn = []
    for grade, quota in AGREEMENT.items():
        stratum = sorted(
            (rank_key("agreement", it["question"], it["location"]), it["question"], it["location"])
            for it in pool_items
            if (it["question"], it["location"]) not in union and v2.by_id(it["question"]).judgments[it["location"]] == grade
        )
        paraphrase = [s for s in stratum if styles[s[1]] == "paraphrase"]
        first = paraphrase[: math.ceil(PARAPHRASE_SHARE * quota)]
        rest = [s for s in stratum if s not in first][: quota - len(first)]
        drawn += [(qid, loc, grade) for _, qid, loc in sorted(first + rest)]
    return sorted(drawn)


def bridge_draw(v1_agreement: list[dict]) -> list[dict]:
    """40 items from v1's agreement study, re-judged to show whether the blind judge itself has moved."""
    ranked = sorted(v1_agreement, key=lambda i: rank_key("bridge", i["question"], i["location"]))
    return sorted(ranked[:BRIDGE], key=lambda i: (i["question"], i["location"]))


def blind_items(v2: GoldenSet, docs, index, union, agreement, bridge) -> list[dict]:
    by_id = {d.doc_id: d for d in docs}
    items = {}
    for qid, loc in union:
        items[(qid, loc)] = "union"
    for qid, loc, _ in agreement:
        items.setdefault((qid, loc), "agreement")
    for b in bridge:
        items.setdefault((b["question"], b["location"]), "bridge")
    out = []
    for (qid, loc), group in items.items():
        out.append({"item": rank_key("item", qid, loc)[:12], "question_id": qid, "group": group,
                    "question": v2.by_id(qid).text, "location": loc,
                    "heading_path": " > ".join(index.get(loc).heading_path), "text": body(by_id, index, loc)})
    return sorted(out, key=lambda i: rank_key("order", i["item"]))


def blind_worksheets(items: list[dict], out: Path, size: int = BATCH_ITEMS) -> list[dict]:
    """Batches of items in sha256 order; a worksheet holds only item id, question, location, heading and text."""
    manifest = []
    for k in range(0, len(items), size):
        chunk = [{key: i[key] for key in ("item", "question", "location", "heading_path", "text")}
                 for i in items[k:k + size]]
        folder = out / f"blind-{k // size + 1:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "worksheet.json"
        path.write_text(json.dumps(chunk, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        manifest.append({"label": folder.name, "path": str(path), "items": [i["item"] for i in chunk],
                         "sha256": sha256_of(path)})
    return manifest


def quote_holds(quote: str, text: str) -> bool:
    """A quote counts only if it is in the section, ignoring differences in whitespace."""
    squash = lambda s: " ".join(s.split())  # noqa: E731
    return bool(quote.strip()) and squash(quote) in squash(text)


# --- commands ---------------------------------------------------------------------------------


def _rankings(golden: GoldenSet):
    docs, index = context()
    return docs, index, run_configs(docs, golden.questions, EVALUATED, index)


def cmd_worksheets(out: Path) -> None:
    v1, v2 = load_golden(default_golden_path(1)), load_golden(default_golden_path(2))
    docs, index, results = _rankings(v2)
    items = pool(v2, results)
    incumbent = {q.id: {loc for loc in results[INCUMBENT.name][i].ranking if loc} for i, q in enumerate(v1.questions)}
    calibration = calibration_draw(v1, incumbent)
    manifest = worksheets(v1, docs, index, items, calibration, out)
    provenance = {
        "pool": {"configurations": [c.name for c in EVALUATED], "depth": 10,
                 "windows": {"budget": HEADLINE_BUDGET, "chunkers": list(WINDOW_CHUNKERS)},
                 "code_sha256": code_version()["package_sha256"], "items": items},
        "calibration": {"rule": calibration_draw.__doc__.strip(), "items": calibration},
        "judging": {
            "prompt": "golden/prompts/v2-pool-judge.md", "prompt_sha256": sha256_of(PROMPTS / "v2-pool-judge.md"),
            "prompt_differs_from_v1": "one added sentence: 'Open no file other than the input file.'",
            "model": JUDGE_MODEL, "batch_rule": f"whole questions in id order; a batch closes at {BATCH_ITEMS}+ items",
            "references": "the question's v1 answers in full; a calibration item graded 2 in v1 is withheld "
                          "from its own question's answers",
            "isolation": "each worksheet alone in its own folder outside the project; judges never saw which "
                         "configuration found a section, how many did, or any ranking, result or plan",
            "batches": [{k: v for k, v in b.items() if k != "path"} for b in manifest],
        },
    }
    (GOLDEN / "v2-provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    (out / "plan.json").write_text(json.dumps([{"label": b["label"], "path": b["path"]} for b in manifest]))
    print(f"{len(items)} pool items + {len(calibration)} calibration items in {len(manifest)} batches -> {out}")


def cmd_merge(graded_path: Path, rejudged_paths: list[Path]) -> None:
    provenance = json.loads((GOLDEN / "v2-provenance.json").read_text())
    pool_keys = {(i["question"], i["location"]) for i in provenance["pool"]["items"]}
    calibration = {(c["question"], c["location"]): c for c in provenance["calibration"]["items"]}
    expected = {b["label"]: {(qid, loc) for qid, loc in pool_keys | set(calibration) if qid in b["questions"]}
                for b in provenance["judging"]["batches"]}
    by_batch = {r["label"]: r for r in json.loads(graded_path.read_text())}
    rejudged = []
    for path in rejudged_paths:  # whole batches judged again after failing validation; every one is logged
        for r in json.loads(path.read_text()):
            first = by_batch.get(r["label"], {}).get("graded")
            rejudged.append({"batch": r["label"], "first_run_problems": check_batch(expected[r["label"]], first)[:5]})
            by_batch[r["label"]] = r
    v2 = load_golden(default_golden_path(2))
    additions, records, failed = {}, [], []
    for b in provenance["judging"]["batches"]:
        graded = (by_batch.get(b["label"]) or {}).get("graded")
        problems = check_batch(expected[b["label"]], graded)
        if problems:
            failed.append((b["label"], problems[:3]))
            continue
        for g in graded:
            key = (g["question"], g["location"])
            records.append({"question": key[0], "location": key[1], "grade": g["grade"], "batch": b["label"]})
            if key in pool_keys:
                additions.setdefault(key[0], {})[key[1]] = g["grade"]
            else:
                calibration[key] = {**calibration[key], "v2_round_grade": g["grade"]}
    if failed:
        raise SystemExit("re-judge these whole batches before merging: " + json.dumps(failed))
    data, added, _ = merge_judgments(v2.raw, additions)
    dump_golden(data, default_golden_path(2))
    provenance["judging"]["rejudged_batches"] = rejudged
    (GOLDEN / "v2-provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    (GOLDEN / "v2-judgments.json").write_text(json.dumps(sorted(records, key=lambda r: (r["question"], r["location"])),
                                                         indent=1) + "\n", encoding="utf-8")
    (GOLDEN / "v2-calibration.json").write_text(json.dumps(
        {"rule": provenance["calibration"]["rule"], "items": sorted(calibration.values(),
                                                                    key=lambda c: (c["question"], c["location"]))},
        indent=1) + "\n", encoding="utf-8")
    print(f"merged {added} new judgments into v2.yaml; {len(calibration)} calibration grades kept apart")


def cmd_blind(out: Path) -> None:
    v2 = load_golden(default_golden_path(2))
    docs, index, results = _rankings(v2)
    provenance = json.loads((GOLDEN / "v2-provenance.json").read_text())
    union = headline_union(v2, results)
    agreement = agreement_draw(v2, provenance["pool"]["items"], set(union))
    bridge = bridge_draw(json.loads((GOLDEN / "v1-agreement.json").read_text())["items"])
    items = blind_items(v2, docs, index, union, agreement, bridge)
    blind = blind_worksheets(items, out / "blind")
    quote = blind_worksheets([i for i in items if i["group"] != "union"], out / "quote")
    (out / "blind-items.json").write_text(json.dumps(
        [{k: i[k] for k in ("item", "question_id", "group", "location")} for i in items], indent=1))
    (out / "plan.json").write_text(json.dumps({"blind": [{"label": b["label"], "path": b["path"]} for b in blind],
                                               "quote": [{"label": b["label"], "path": b["path"]} for b in quote]}))
    provenance["blind"] = {
        "prompt": "golden/prompts/v2-blind-judge.md", "prompt_sha256": sha256_of(PROMPTS / "v2-blind-judge.md"),
        "quote_prompt": "golden/prompts/v2-quote-judge.md", "quote_prompt_sha256": sha256_of(PROMPTS / "v2-quote-judge.md"),
        "model": JUDGE_MODEL, "union": len(union), "agreement": len(agreement), "bridge": len(bridge),
        "batches": [{k: v for k, v in b.items() if k != "path"} for b in blind],
        "quote_batches": [{k: v for k, v in b.items() if k != "path"} for b in quote],
    }
    (GOLDEN / "v2-provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    print(f"blind: {len(union)} union + {len(agreement)} agreement + {len(bridge)} bridge = {len(items)} items "
          f"in {len(blind)} batches; quote judge: {len(items) - len(union)} items")


def cmd_merge_blind(blind_path: Path, quote_path: Path, items_path: Path) -> None:
    v2 = load_golden(default_golden_path(2))
    docs, index = context()
    by_id = {d.doc_id: d for d in docs}
    meta = {i["item"]: i for i in json.loads(items_path.read_text())}
    v1_agreement = {(i["question"], i["location"]): i for i in json.loads((GOLDEN / "v1-agreement.json").read_text())["items"]}
    blind = {g["item"]: g["grade"] for r in json.loads(blind_path.read_text()) for g in (r.get("grades") or [])}
    quote = {g["item"]: g for r in json.loads(quote_path.read_text()) for g in (r.get("judgments") or [])}
    missing = sorted(set(meta) - set(blind)) + sorted(i for i in meta if meta[i]["group"] != "union" and i not in quote)
    if missing:
        raise SystemExit(f"{len(missing)} items have no grade; re-judge their batches: {missing[:5]}")
    rows, quotes = [], []
    for item, m in sorted(meta.items(), key=lambda kv: (kv[1]["question_id"], kv[1]["location"])):
        key = (m["question_id"], m["location"])
        first = v1_agreement[key]["second"] if m["group"] == "bridge" else v2.by_id(key[0]).judgments.get(key[1])
        rows.append({"question": key[0], "location": key[1], "group": m["group"], "first": first, "blind": blind[item]})
        if item in quote:
            q = quote[item]
            holds = bool(q.get("answers")) and quote_holds(q.get("quote", ""), body(by_id, index, key[1]))
            quotes.append({"question": key[0], "location": key[1], "group": m["group"], "first": first,
                           "answers": bool(q.get("answers")), "quote": q.get("quote", ""), "quote_holds": holds})
    (GOLDEN / "v2-blind.json").write_text(json.dumps({"items": rows}, indent=1) + "\n", encoding="utf-8")
    (GOLDEN / "v2-quote.json").write_text(json.dumps({"items": quotes}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} blind grades and {len(quotes)} quote judgments written")


def cmd_verify() -> None:
    v1, v2 = load_golden(default_golden_path(1)), load_golden(default_golden_path(2))
    provenance = json.loads((GOLDEN / "v2-provenance.json").read_text())
    docs, index, results = _rankings(v1)
    incumbent = {q.id: {loc for loc in results[INCUMBENT.name][i].ranking if loc} for i, q in enumerate(v1.questions)}
    problems = []
    if [{k: c[k] for k in ("question", "location", "v1_grade", "stratum")} for c in calibration_draw(v1, incumbent)] != \
            [{k: c[k] for k in ("question", "location", "v1_grade", "stratum")} for c in provenance["calibration"]["items"]]:
        problems.append("calibration items are not the documented draw")
    judged = json.loads((GOLDEN / "v2-judgments.json").read_text())
    pool_keys = {(i["question"], i["location"]) for i in provenance["pool"]["items"]}
    rebuilt = {}
    for r in judged:
        if (r["question"], r["location"]) in pool_keys:
            rebuilt.setdefault(r["question"], {})[r["location"]] = r["grade"]
    data, _, _ = merge_judgments(v1.raw, rebuilt)
    if [q["judgments"] for q in data["questions"]] != [dict(q.judgments) for q in v2.questions]:
        problems.append("v2.yaml is not v1.yaml plus the pool grades in v2-judgments.json")
    if pool_keys != {(r["question"], r["location"]) for r in judged} - {(c["question"], c["location"])
                                                                       for c in provenance["calibration"]["items"]}:
        problems.append("v2-judgments.json doesn't hold exactly the pool and calibration items")
    if problems:
        raise SystemExit("NOT OK: " + "; ".join(problems))
    print("OK: v2.yaml is v1.yaml plus the round's pool grades, and the calibration items are the documented draw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    w = sub.add_parser("worksheets")
    w.add_argument("out", type=Path, help="an empty folder outside the project")
    m = sub.add_parser("merge")
    m.add_argument("graded", type=Path, help="the judging workflow's output: [{label, graded: [...]}]")
    m.add_argument("--rejudged", type=Path, nargs="*", default=[],
                   help="outputs of whole batches judged again after failing validation (logged in provenance)")
    b = sub.add_parser("blind")
    b.add_argument("out", type=Path)
    mb = sub.add_parser("merge-blind")
    mb.add_argument("blind", type=Path)
    mb.add_argument("quote", type=Path)
    mb.add_argument("items", type=Path)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "init":
        print(f"wrote {init_v2()}")
    elif args.command == "worksheets":
        if args.out.resolve().is_relative_to(GOLDEN.parent.resolve()):
            raise SystemExit("write worksheets outside the project, so judges can't reach anything else in it")
        cmd_worksheets(args.out)
    elif args.command == "merge":
        cmd_merge(args.graded, args.rejudged)
    elif args.command == "blind":
        cmd_blind(args.out)
    elif args.command == "merge-blind":
        cmd_merge_blind(args.blind, args.quote, args.items)
    else:
        cmd_verify()


if __name__ == "__main__":
    main()
