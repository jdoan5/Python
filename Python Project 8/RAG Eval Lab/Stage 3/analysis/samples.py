"""The two judged samples behind golden/'s reliability files, drawn by exact rules.

agreement   A stratified sample of v1's labels for a blind second judge: up to
            80 sections graded 2, 60 graded 1 and 60 graded 0 (targets
            excluded), plus 40 question targets. Within a stratum, items are
            taken in order of
            sha256("20261120:agreement:{question}:{location}"). The judge sees
            them in order of sha256("20261120:order:{question}:{location}"),
            with only the question and the section's full text.

unjudged    The (question, section) pairs some configuration puts in its
            4,800-character context that v1 leaves unjudged, in two strata.
            "focused": pairs inside a heading or fixed configuration's context;
            these are all judged. "scattered": pairs that appear only in
            paragraph-family contexts; 200 are judged, taken in order of
            sha256("20261120:unjudged:{question}:{location}").

`python analysis/samples.py agreement|unjudged --out DIR` writes worksheets
for judging; `python analysis/samples.py verify` checks that the committed
files in golden/ are exactly these draws from the current v1.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.experiments import FACTORIAL, build_retrievers  # noqa: E402
from raglab.golden import default_golden_path, load_golden  # noqa: E402
from raglab.metrics import HEADLINE_BUDGET, pack  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402

SEED = 20261120
AGREEMENT_QUOTA = {2: 80, 1: 60, 0: 60}
AGREEMENT_TARGETS = 40
SCATTERED_SAMPLE = 200
PARAGRAPH_FAMILY = ("paragraph", "baseline")


def rank(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, (SEED, *parts))).encode("utf-8")).hexdigest()


def load():
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    return docs, SectionIndex(docs), load_golden(default_golden_path())


def agreement_draw(golden) -> list[tuple[str, str, int, bool]]:
    """(question, location, grade, is_target), in presentation order."""
    strata: dict[object, list[tuple[str, str, int, bool]]] = {0: [], 1: [], 2: [], "target": []}
    for q in golden.questions:
        for loc, grade in q.judgments.items():
            item = (q.id, loc, grade, loc == q.target)
            strata["target" if loc == q.target else grade].append(item)
    chosen = []
    for key, quota in (*AGREEMENT_QUOTA.items(), ("target", AGREEMENT_TARGETS)):
        chosen += sorted(strata[key], key=lambda t: rank("agreement", t[0], t[1]))[:quota]
    return sorted(chosen, key=lambda t: rank("order", t[0], t[1]))


def unjudged_population(docs, index, golden) -> dict[tuple[str, str], str]:
    """Every unjudged pair inside some configuration's headline-budget context -> its stratum."""
    retrievers = build_retrievers(docs, FACTORIAL)
    configs_by_pair: dict[tuple[str, str], set[str]] = {}
    for config in FACTORIAL:
        for q in golden.questions:
            hits = retrievers[config.name].search(q.text, len(retrievers[config.name].chunks))
            spans = [(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits]
            for doc_id, start, end in pack(spans, HEADLINE_BUDGET):
                for ref, _ in index.retrieved_by(doc_id, start, end):
                    if ref.location not in q.judgments:
                        configs_by_pair.setdefault((q.id, ref.location), set()).add(config.name)
    return {
        pair: "scattered" if all(name.startswith(PARAGRAPH_FAMILY) for name in names) else "focused"
        for pair, names in configs_by_pair.items()
    }


def unjudged_draw(population: dict[tuple[str, str], str]) -> list[tuple[str, str, str]]:
    focused = sorted(p for p, s in population.items() if s == "focused")
    scattered = sorted((p for p, s in population.items() if s == "scattered"), key=lambda p: rank("unjudged", *p))
    return [(q, loc, "focused") for q, loc in focused] + [(q, loc, "scattered") for q, loc in scattered[:SCATTERED_SAMPLE]]


def worksheet(index, docs, golden, pairs, with_answers: bool) -> list[dict]:
    text = {d.doc_id: d.text for d in docs}

    def body(loc: str) -> str:
        ref = index.get(loc)
        return text[ref.doc_id][ref.start:ref.end]

    by_q: dict[str, dict] = {}
    for qid, loc in pairs:
        q = golden.by_id(qid)
        entry = by_q.setdefault(qid, {"id": qid, "question": q.text, "candidates": []})
        if with_answers:
            entry["answers"] = {a: body(a) for a in sorted(q.primary)}
        entry["candidates"].append({"location": loc, "heading_path": " > ".join(index.get(loc).heading_path), "text": body(loc)})
    return list(by_q.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("agreement", "unjudged", "verify"))
    parser.add_argument("--out", type=Path, help="folder for worksheets (agreement, unjudged)")
    args = parser.parse_args()
    docs, index, golden = load()
    folder = default_golden_path().parent

    if args.what == "agreement":
        draw = agreement_draw(golden)
        text = {d.doc_id: d.text for d in docs}
        items, key = [], {}
        for n, (qid, loc, grade, is_target) in enumerate(draw, start=1):
            ref = index.get(loc)
            items.append({"item": f"a{n:03d}", "question": golden.by_id(qid).text, "location": loc,
                          "heading_path": " > ".join(ref.heading_path), "text": text[ref.doc_id][ref.start:ref.end]})
            key[f"a{n:03d}"] = {"question": qid, "location": loc, "first": grade, "is_target": is_target}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "agreement_items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        (args.out / "agreement_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        print(f"{len(items)} agreement items -> {args.out}")
    elif args.what == "unjudged":
        population = unjudged_population(docs, index, golden)
        draw = unjudged_draw(population)
        args.out.mkdir(parents=True, exist_ok=True)
        for stratum in ("focused", "scattered"):
            pairs = [(q, loc) for q, loc, s in draw if s == stratum]
            (args.out / f"unjudged_{stratum}.json").write_text(
                json.dumps(worksheet(index, docs, golden, pairs, with_answers=True), indent=1, ensure_ascii=False),
                encoding="utf-8")
        sizes = {s: sum(1 for v in population.values() if v == s) for s in ("focused", "scattered")}
        (args.out / "unjudged_population.json").write_text(json.dumps(sizes), encoding="utf-8")
        print(f"population {sizes}; drawn {len(draw)} -> {args.out}")
    else:
        agreement = json.loads((folder / "v1-agreement.json").read_text(encoding="utf-8"))["items"]
        expected = [(q, loc) for q, loc, _, _ in agreement_draw(golden)]
        assert [(i["question"], i["location"]) for i in agreement] == expected, "agreement sample is not the documented draw"
        sample = json.loads((folder / "v1-unjudged-sample.json").read_text(encoding="utf-8"))
        population = unjudged_population(docs, index, golden)
        assert sample["population"] == {s: sum(1 for v in population.values() if v == s) for s in ("focused", "scattered")}
        drawn = [(q, loc) for q, loc, _ in unjudged_draw(population)]
        assert sorted((i["question"], i["location"]) for i in sample["items"]) == sorted(drawn), "unjudged sample is not the documented draw"
        print(f"OK: both committed samples are exactly the documented draws from {golden.path.name}")


if __name__ == "__main__":
    main()
