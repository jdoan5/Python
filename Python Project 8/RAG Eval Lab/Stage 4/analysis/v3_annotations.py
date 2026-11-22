"""Stage 4's annotations: section labels, the sufficiency audit (S) and the answer keys (PLAN.md, "Annotations").

They are made the way Stage 3 made v2's labels (analysis/v2_labels.py): every
worksheet sits alone in its own folder outside the project, is judged by
claude-opus-5-5 in Claude Code at no API cost, and only a whole batch that
fails validation is judged again. This file covers the pilot, frozen at freeze
point 1:

  pilot-worksheets   the pilot's pool (every section on each target page and in
                     each pilot window) and its windows for S, one worksheet per batch
  pilot-merge        check every batch, then write pilot/labels.json and pilot/sufficiency.json
  pilot-keys         the key worksheets: each question's reference, built by the v3 rule
  pilot-merge-keys   check every key, then write pilot/keys.json (keys and reference lists)
  pilot-swaps        after the pilot's answers: a blind worksheet confirming each near-miss swap candidate
  pilot-merge-swaps  check it, then write pilot/checks.json (which candidates are confirmed)

    cd "Stage 4" && ../.venv/bin/python analysis/v3_annotations.py pilot-worksheets --out <folder outside the project>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab import params  # noqa: E402
from raglab.blocks import window_documents  # noqa: E402
from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.judge import quote_holds  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402
from raglab.windows import CONDITIONS, Windows  # noqa: E402

STAGE = Path(__file__).resolve().parents[1]
PILOT = STAGE / "pilot"
PROMPTS = STAGE.parent / "golden" / "prompts"
JUDGE_MODEL = "claude-opus-5-5"
POOL_BATCH_ITEMS = 60       # a pool batch closes once it holds at least this many items (v2's rule)
WINDOW_BATCH = 8            # windows per S worksheet: a baseline window runs to about 8,500 characters
PILOT_WINDOWS = ("hhr-4.8k-med", "hhr-9.6k-med", "base-4.8k-med")


class AnnotationError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(kind: str, *parts: str) -> str:
    return hashlib.sha256(":".join((kind,) + parts).encode("utf-8")).hexdigest()


def context():
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    return lock, docs, {d.doc_id: d for d in docs}, SectionIndex(docs)


def section_text(by_id, index, location: str) -> str:
    ref = index.get(location)
    if ref is None:
        raise AnnotationError(f"no section {location}")
    return by_id[ref.doc_id].text[ref.start:ref.end]


def outside_project(out: Path) -> Path:
    out = out.resolve()
    project = STAGE.parent.resolve()
    if out == project or project in out.parents:
        raise AnnotationError(f"worksheets go outside the project, not in {out}")
    return out


def load_pilot() -> dict:
    data = yaml.safe_load((PILOT / "questions.yaml").read_text(encoding="utf-8"))
    qs = data["questions"]
    if len(qs) != params.PILOT["questions"] or len({q["id"] for q in qs}) != len(qs):
        raise AnnotationError(f"the pilot needs {params.PILOT['questions']} distinct questions")
    return data


def pilot_windows(qs, docs, index) -> dict[tuple[str, str], list]:
    """Every pilot window that S judges and that the pool covers, keyed by (question id, condition)."""
    w = Windows(docs, index)
    return {(q["id"], c): w.window(CONDITIONS[c], q["question"]) for q in qs for c in PILOT_WINDOWS}


def refuse_if_frozen(phase: str, *rels: str) -> None:
    """A merge never rewrites a file once the phase it decides has been authorized: from then on the file is
    frozen (PLAN.md, "Freeze points": freeze point 1 is the commit the pilot's first authorization is given at).
    Before that, a pushed file is a draft and may be merged again. Authorization records are append-only, so
    this never reopens."""
    from raglab import cli4
    from raglab.authorize import _load

    given = [r for r in _load(cli4.api_dir() / "authorization.json") if r["phase"] == phase]
    if given:
        raise AnnotationError(f"{', '.join(rels)} is frozen: phase {phase!r} was authorized at commit "
                              f"{given[0]['commit'][:8]}. A change now is a deviation, logged in the README first")


def write_worksheet(folder: Path, entries: list) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "worksheet.json"
    path.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"label": folder.name, "path": str(path), "sha256": sha256_of(path), "entries": len(entries)}


def cmd_pilot_worksheets(out: Path) -> None:
    out = outside_project(out)
    data = load_pilot()
    qs = data["questions"]
    _, docs, by_id, index = context()
    windows = pilot_windows(qs, docs, index)
    pool, manifest = {}, {"pool": [], "sufficiency": []}
    for q in qs:
        cands = set()
        if q.get("target"):
            page = q["target"].split("#", 1)[0]
            cands |= {r.location for r in index if r.doc_id == page}
        for c in PILOT_WINDOWS:
            cands |= {ref.location for span in windows[(q["id"], c)] for ref, _ in index.retrieved_by(*span)}
        cands.discard(q.get("target"))
        pool[q["id"]] = sorted(cands)
    # Pool batches: whole questions in id order, closing at POOL_BATCH_ITEMS or more items.
    batch, size, k = [], 0, 0
    for q in sorted(qs, key=lambda q: q["id"]):
        answers = {q["target"]: section_text(by_id, index, q["target"])} if q.get("target") else {}
        batch.append({"id": q["id"], "question": q["question"], "answers": answers,
                      "candidates": [{"location": loc, "heading_path": " > ".join(index.get(loc).heading_path),
                                      "text": section_text(by_id, index, loc)} for loc in pool[q["id"]]]})
        size += len(pool[q["id"]])
        if size >= POOL_BATCH_ITEMS:
            k += 1
            manifest["pool"].append(write_worksheet(out / f"pool-{k:02d}", batch) | {"questions": [e["id"] for e in batch]})
            batch, size = [], 0
    if batch:
        k += 1
        manifest["pool"].append(write_worksheet(out / f"pool-{k:02d}", batch) | {"questions": [e["id"] for e in batch]})
    # S batches: windows from every condition mixed, in sha256 order.
    items = []
    for (qid, cond), spans in windows.items():
        q = next(x for x in qs if x["id"] == qid)
        wdocs = window_documents(spans, by_id, index)
        items.append({"id": f"{qid}|{rank('s', qid, cond)[:8]}", "question": q["question"],
                      "documents": [{"title": d.title, "text": "\n\n".join(b.text for b in d.blocks)} for d in wdocs],
                      "_window": cond, "_blocks": evidence_blocks(wdocs)})
    items.sort(key=lambda e: rank("s-order", e["id"]))
    key_of = {e["id"]: e.pop("_window") for e in items}
    blocks_of = {e["id"]: e.pop("_blocks") for e in items}  # kept beside the worksheets, never shown to the judge
    for k, start in enumerate(range(0, len(items), WINDOW_BATCH), 1):
        chunk = items[start:start + WINDOW_BATCH]
        manifest["sufficiency"].append(write_worksheet(out / f"sufficiency-{k:02d}", chunk)
                                       | {"items": [e["id"] for e in chunk]})
    manifest["window_of"] = key_of  # the id -> condition map stays with the worksheets, never shown to a judge
    manifest["blocks_of"] = blocks_of  # what an S quote may come from: one block, never a heading line
    manifest["pool_items"] = {qid: locs for qid, locs in pool.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    n_pool = sum(len(v) for v in pool.values())
    print(f"{len(manifest['pool'])} pool worksheets ({n_pool} sections) and {len(manifest['sufficiency'])} S "
          f"worksheets ({len(items)} windows) in {out}")


def check_pool_batch(expected: set[tuple[str, str]], graded: list[dict]) -> list[str]:
    seen, problems = set(), []
    for g in graded:
        key = (g.get("question"), g.get("location"))
        if key not in expected:
            problems.append(f"unexpected item {key}")
        elif key in seen:
            problems.append(f"graded twice: {key}")
        elif g.get("grade") not in (0, 1, 2) or isinstance(g.get("grade"), bool):
            problems.append(f"bad grade for {key}: {g.get('grade')!r}")
        seen.add(key)
    return problems + [f"missing {key}" for key in sorted(expected - seen)]


def evidence_blocks(wdocs) -> list[str]:
    """The texts an S quote may come from (PLAN.md, "S"): each block of each document on its own, leaving out
    heading lines, so neither a title nor a heading can count as evidence."""
    return [b.text for d in wdocs for b in d.blocks if b.section is not None]


def check_s_batch(expected: list[str], answers: list[dict], blocks_of: dict[str, list[str]]) -> list[str]:
    seen, problems = set(), []
    for a in answers:
        i = a.get("id")
        if i not in expected:
            problems.append(f"unexpected item {i}")
        elif i in seen:
            problems.append(f"judged twice: {i}")
        elif a.get("answers") not in ("yes", "partly", "no"):
            problems.append(f"bad verdict for {i}: {a.get('answers')!r}")
        elif a["answers"] != "no" and not any(quote_holds(a.get("quote", ""), b) for b in blocks_of[i]):
            problems.append(f"{i}: the quote isn't inside one block of the window (heading lines don't count)")
        seen.add(i)
    return problems + [f"missing {i}" for i in expected if i not in seen]


def cmd_pilot_merge(manifest_path: Path, pool_path: Path, s_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_grades = json.loads(pool_path.read_text(encoding="utf-8"))   # {batch label: [grades]}
    s_grades = json.loads(s_path.read_text(encoding="utf-8"))         # {batch label: [verdicts]}
    problems = []
    labels: dict[str, dict[str, int]] = {}
    for b in manifest["pool"]:
        if sha256_of(Path(b["path"])) != b["sha256"]:
            problems.append(f"{b['label']}: the worksheet changed after it was written")
        expected = {(qid, loc) for qid in b["questions"] for loc in manifest["pool_items"][qid]}
        got = pool_grades.get(b["label"])
        bad = ["no grades returned"] if got is None else check_pool_batch(expected, got)
        problems += [f"{b['label']}: {p}" for p in bad]
        for g in got or ():
            labels.setdefault(g["question"], {})[g["location"]] = g["grade"]
    sufficiency = {}
    for b in manifest["sufficiency"]:
        if sha256_of(Path(b["path"])) != b["sha256"]:
            problems.append(f"{b['label']}: the worksheet changed after it was written")
        got = s_grades.get(b["label"])
        bad = ["no verdicts returned"] if got is None else check_s_batch(b["items"], got, manifest["blocks_of"])
        problems += [f"{b['label']}: {p}" for p in bad]
        for a in got or ():
            qid = a["id"].split("|", 1)[0]
            sufficiency.setdefault(qid, {})[manifest["window_of"][a["id"]]] = {"answers": a["answers"],
                                                                                "quote": a.get("quote", "")}
    if problems:
        raise AnnotationError("not merged; judge these batches again:\n  " + "\n  ".join(problems))
    refuse_if_frozen("pilot", "pilot/labels.json", "pilot/sufficiency.json")
    data = load_pilot()
    for q in data["questions"]:
        if q.get("target"):
            labels.setdefault(q["id"], {})[q["target"]] = 2  # the section each question was written from
    provenance = {"judge_model": JUDGE_MODEL, "prompts": {n: sha256_of(PROMPTS / n) for n in
                                                          ("v2-pool-judge.md", "v3-sufficiency.md")},
                  "worksheets": {b["label"]: b["sha256"] for b in manifest["pool"] + manifest["sufficiency"]},
                  "plan_sha256": sha256_of(STAGE / "PLAN.md")}
    (PILOT / "labels.json").write_text(json.dumps({"provenance": provenance, "labels": labels}, indent=1,
                                                  sort_keys=True) + "\n", encoding="utf-8")
    (PILOT / "sufficiency.json").write_text(json.dumps({"provenance": provenance, "windows": sufficiency}, indent=1,
                                                       sort_keys=True) + "\n", encoding="utf-8")
    print(f"pilot labels for {len(labels)} questions and S for {sum(len(v) for v in sufficiency.values())} windows")


# ---------------------------------------------------------------------------------------------------- keys


def delivered_shares(spans_by_window, answering, index) -> dict[str, tuple[int, float]]:
    """For each answering section: how many of the given windows deliver it, and the largest share of its
    text any of them holds (Stage 3's coverage_fractions)."""
    from raglab.metrics import coverage_fractions

    out: dict[str, tuple[int, float]] = {}
    for loc in answering:
        ref = index.get(loc)
        shares = [coverage_fractions(spans, [ref], index)[0] for spans in spans_by_window]
        out[loc] = (sum(1 for x in shares if x > 0), max(shares, default=0.0))
    return out


def reference_for(labels: dict[str, int], target: str | None, delivered: dict[str, tuple[int, float]], by_id,
                  index, trim: int = 0) -> tuple[list[dict], list[dict]]:
    """PLAN.md's rule. Which sections are kept, and the order they are shown in, are separate:
    - kept, in priority order while they fit under the cap: the target always; then the sections the chain's
      windows deliver, by the largest share of the section any window holds (so a section touched only by a
      3% fragment counts last), then by how many windows deliver it; then the rest, in document order;
    - shown in document order, so a section's position says nothing about whether it is the target.
    `trim` is the key rule's fallback: when no fact holds in every section, the last `trim` kept sections in
    priority order are dropped (never the target), and the trim is logged.
    Returns the kept sections and the sections left out: by the cap (delivered ones, with their shares) or by
    the trim."""
    answering = [loc for loc, g in labels.items() if g == 2]

    def doc_order(loc: str):
        ref = index.get(loc)
        return (ref.doc_id, ref.start)

    def priority(loc: str):
        count, share = delivered.get(loc, (0, 0.0))
        return (loc != target, -share, -count, doc_order(loc))

    def whole(loc: str) -> dict:
        ref = index.get(loc)
        doc = by_id[ref.doc_id]
        section = next(s for s in doc.sections if s.start <= ref.start < s.end)
        return {"location": loc, "heading": " > ".join(ref.heading_path),
                "text": doc.text[section.start:ref.end].strip()}  # the heading line and the body, whole

    kept, removed, used = [], [], 0
    for loc in sorted(answering, key=priority):
        sec = whole(loc)
        if loc != target and used + len(sec["text"]) > params.REFERENCE_CAP_CHARS:
            if delivered.get(loc, (0, 0.0))[0]:
                removed.append({"location": loc, "windows": delivered[loc][0], "largest_share": delivered[loc][1]})
            continue
        used += len(sec["text"])
        kept.append(sec)
    if trim:
        droppable = [sec for sec in kept if sec["location"] != target]
        if trim > len(droppable):
            raise AnnotationError(f"can't trim {trim} sections: only {len(droppable)} besides the target")
        for sec in droppable[len(droppable) - trim:]:
            kept.remove(sec)
            removed.append({"location": sec["location"], "trimmed": True})
    kept.sort(key=lambda sec: doc_order(sec["location"]))
    return kept, removed


def check_key(entry: dict, reference: list[dict]) -> list[str]:
    """The v3 key rules, applied by code (PLAN.md, "Keys and references")."""
    problems = []
    nuggets = entry.get("nuggets") or []
    k = params.KEYS
    if not k["min_nuggets"] <= len(nuggets) <= k["max_nuggets"]:
        problems.append(f"{len(nuggets)} nuggets, not {k['min_nuggets']}-{k['max_nuggets']}")
    essential = [n for n in nuggets if n.get("essential")]
    if not k["min_essential"] <= len(essential) <= k["max_essential"]:
        problems.append(f"{len(essential)} essential nuggets, not {k['min_essential']}-{k['max_essential']}")
    sections = {f"R{i}": s["text"] for i, s in enumerate(reference, start=1)}
    for n, nugget in enumerate(nuggets, start=1):
        quoted = set()
        for q in nugget.get("quotes") or ():
            sec = q.get("section")
            if sec not in sections:
                problems.append(f"nugget {n}: quote names unknown section {sec!r}")
            elif not quote_holds(q.get("quote", ""), sections[sec]):
                problems.append(f"nugget {n}: a quote isn't in {sec}")
            else:
                quoted.add(sec)
        if not quoted:
            problems.append(f"nugget {n}: no quote that holds")
        if nugget.get("essential") and quoted != set(sections):
            problems.append(f"essential nugget {n} isn't quoted from every section: missing "
                            + ", ".join(sorted(set(sections) - quoted)))
    return problems


def cmd_pilot_keys(out: Path, trims: dict[str, int] | None = None) -> None:
    trims = trims or {}
    out = outside_project(out)
    data = load_pilot()
    labels = json.loads((PILOT / "labels.json").read_text(encoding="utf-8"))["labels"]
    _, docs, by_id, index = context()
    windows = pilot_windows(data["questions"], docs, index)
    entries, refs = [], {}
    removed_all = {}
    for q in data["questions"]:
        grades = labels.get(q["id"], {})
        answering = [loc for loc, g in grades.items() if g == 2]
        delivered = delivered_shares([windows[(q["id"], c)] for c in ("hhr-4.8k-med", "base-4.8k-med")], answering,
                                     index)
        ref, removed = reference_for(grades, q.get("target"), delivered, by_id, index, trims.get(q["id"], 0))
        refs[q["id"]], removed_all[q["id"]] = ref, removed
        if ref:  # a question no section answers has no reference and no key
            entries.append({"id": q["id"], "question": q["question"],
                            "reference": [{"id": f"R{i}", "heading": s["heading"], "text": s["text"]}
                                          for i, s in enumerate(ref, start=1)]})
    info = write_worksheet(out / "keys-01", entries)
    (out / "keys-manifest.json").write_text(json.dumps({"keys": info, "references": refs, "removed_by_cap": removed_all,
                                                        "trims": trims}, indent=1) + "\n", encoding="utf-8")
    print(f"a key worksheet for {len(entries)} questions in {out / 'keys-01'}")


def cmd_pilot_merge_keys(manifest_path: Path, keys_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = json.loads(keys_path.read_text(encoding="utf-8"))  # a list of entries
    refs = manifest["references"]
    problems, out = [], {}
    by_id = {e.get("id"): e for e in keys}
    expected = [qid for qid, ref in refs.items() if ref]
    for qid in expected:
        if qid not in by_id:
            problems.append(f"{qid}: no key")
            continue
        bad = check_key(by_id[qid], refs[qid])
        problems += [f"{qid}: {p}" for p in bad]
        out[qid] = {"nuggets": [{"id": f"N{i}", "text": n["text"], "essential": bool(n["essential"]),
                                 "alternatives": list(n.get("alternatives") or []), "quotes": n.get("quotes") or []}
                                for i, n in enumerate(by_id[qid]["nuggets"], start=1)],
                    "reference": [s["location"] for s in refs[qid]], "trimmed": manifest.get("trims", {}).get(qid, 0)}
    problems += [f"unexpected key {qid}" for qid in by_id if qid not in expected]
    if problems:
        raise AnnotationError("not merged; write these keys again:\n  " + "\n  ".join(problems))
    refuse_if_frozen("pilot", "pilot/keys.json")
    provenance = {"judge_model": JUDGE_MODEL, "prompt": sha256_of(PROMPTS / "v3-keys.md"),
                  "worksheet": manifest["keys"]["sha256"], "plan_sha256": sha256_of(STAGE / "PLAN.md"),
                  "trims": manifest.get("trims", {})}
    (PILOT / "keys.json").write_text(json.dumps({"provenance": provenance, "keys": out}, indent=1, sort_keys=True)
                                     + "\n", encoding="utf-8")
    print(f"keys for {len(out)} pilot questions")


# ---------------------------------------------------------------------------------------------------- swaps


def swap_entries(candidates) -> tuple[list[dict], dict[str, dict]]:
    """Blind worksheet entries for every candidate that needs confirming, and the id map kept beside them.

    An entry shows a claim and a passage, nothing else: not the kind, the arm or
    what answer is expected. Ids are opaque and entries are in sha256 order.
    """
    entries, key_of = [], {}
    for c in candidates:
        if not c.needs_confirmation:
            continue
        eid = "c" + rank("swap-check", c.answered, c.name)[:10]
        entries.append({"id": eid, "claim": c.claim, "passage": "\n\n".join(c.texts),
                        "other_passages": list(c.context)})
        key_of[eid] = {"candidate": f"{c.answered}:{c.name}", "kind": c.kind}
    entries.sort(key=lambda e: rank("swap-order", e["id"]))
    return entries, key_of


def check_swaps(entries: list[dict], answers: list[dict]) -> list[str]:
    """Every entry answered once, "yes" with a quote that holds in its passage, or "no" with none."""
    problems = []
    passage = {e["id"]: e["passage"] for e in entries}
    got = [a.get("id") for a in answers]
    if sorted(got) != sorted(passage) or len(set(got)) != len(got):
        problems.append(f"ids don't match: missing {sorted(set(passage) - set(got))}, "
                        f"unexpected {sorted(set(got) - set(passage))}")
        return problems
    for a in answers:
        if a.get("others") not in ("yes", "no"):
            problems.append(f"{a['id']}: others must be yes or no")
        if a.get("states") not in ("yes", "no"):
            problems.append(f"{a['id']}: states must be yes or no")
        elif a["states"] == "yes" and not quote_holds(a.get("quote", ""), passage[a["id"]]):
            problems.append(f"{a['id']}: a yes whose quote isn't in the passage")
        elif a["states"] == "no" and (a.get("quote") or "").strip():
            problems.append(f"{a['id']}: a no with a quote")
    return problems


def cmd_pilot_swaps(out: Path) -> None:
    from raglab import cli4_judge

    out = outside_project(out)
    candidates = cli4_judge._candidates(cli4_judge.pilot_world())
    entries, key_of = swap_entries(candidates)
    info = write_worksheet(out / "swaps-01", entries)
    (out / "swaps-manifest.json").write_text(json.dumps({"swaps": info, "key_of": key_of}, indent=1) + "\n",
                                             encoding="utf-8")
    print(f"a swap-check worksheet of {len(entries)} entries in {out / 'swaps-01'}")


def cmd_pilot_merge_swaps(manifest_path: Path, answers_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = json.loads(Path(manifest["swaps"]["path"]).read_text(encoding="utf-8"))
    if sha256_of(Path(manifest["swaps"]["path"])) != manifest["swaps"]["sha256"]:
        raise AnnotationError("the worksheet changed after it was written")
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    problems = check_swaps(entries, answers)
    if problems:
        raise AnnotationError("not merged; judge this worksheet again:\n  " + "\n  ".join(problems))
    refuse_if_frozen("pilot-judging", "pilot/checks.json")
    import hashlib

    passage = {e["id"]: e["passage"] for e in entries}
    confirmed, cross_unit, passage_sha = {}, {}, {}
    for a in answers:
        meta = manifest["key_of"][a["id"]]
        passage_sha[meta["candidate"]] = hashlib.sha256(passage[a["id"]].encode("utf-8")).hexdigest()
        # A swap is confirmed when its block does NOT state the claim; a cross-section plant when it does.
        confirmed[meta["candidate"]] = (a["states"] == "no") if meta["kind"].startswith("swap") else (a["states"] == "yes")
        if meta["kind"].startswith("swap"):
            cross_unit[meta["candidate"]] = a["others"] == "yes"  # another unit's excerpts state the claim
    provenance = {"judge_model": JUDGE_MODEL, "prompt": sha256_of(PROMPTS / "v3-swap-check.md"),
                  "worksheet": manifest["swaps"]["sha256"], "plan_sha256": sha256_of(STAGE / "PLAN.md")}
    (PILOT / "checks.json").write_text(json.dumps({"provenance": provenance, "confirmed": confirmed,
                                                   "cross_unit": cross_unit, "passage_sha256": passage_sha},
                                                  indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{sum(confirmed.values())} of {len(confirmed)} candidates confirmed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("pilot-worksheets")
    w.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("pilot-merge")
    m.add_argument("--manifest", type=Path, required=True)
    m.add_argument("--pool", type=Path, required=True)
    m.add_argument("--sufficiency", type=Path, required=True)
    k = sub.add_parser("pilot-keys")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--trim", action="append", default=[], metavar="QID=N",
                   help="the key rule's fallback: drop the last N kept sections of QID's reference (logged)")
    mk = sub.add_parser("pilot-merge-keys")
    mk.add_argument("--manifest", type=Path, required=True)
    mk.add_argument("--keys", type=Path, required=True)
    sw = sub.add_parser("pilot-swaps")
    sw.add_argument("--out", type=Path, required=True)
    ms = sub.add_parser("pilot-merge-swaps")
    ms.add_argument("--manifest", type=Path, required=True)
    ms.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "pilot-worksheets":
            cmd_pilot_worksheets(args.out)
        elif args.command == "pilot-merge":
            cmd_pilot_merge(args.manifest, args.pool, args.sufficiency)
        elif args.command == "pilot-keys":
            cmd_pilot_keys(args.out, {t.split("=")[0]: int(t.split("=")[1]) for t in args.trim})
        elif args.command == "pilot-merge-keys":
            cmd_pilot_merge_keys(args.manifest, args.keys)
        elif args.command == "pilot-swaps":
            cmd_pilot_swaps(args.out)
        else:
            cmd_pilot_merge_swaps(args.manifest, args.answers)
    except AnnotationError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
