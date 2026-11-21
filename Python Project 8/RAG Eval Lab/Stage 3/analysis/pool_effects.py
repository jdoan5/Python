"""The pool analyses and bias checks (PLAN.md, "Exploratory"). Exploratory: no claims.

Run only after freeze point 2: these score Stage 3's configurations.

- stale pool: the new configurations scored on v1 (unjudged counts as not
  relevant) against v2, with each one's unjudged share of its v1 top 10;
- Stage 2 on the wider pool: Stage 2's 12 configurations on v1 against v2, by
  style. Success@5 and MRR@10 can't change, by construction; Recall, nDCG and
  coverage can, as answer sets grow;
- leaving one family out: drop the new v2 judgments that only one new retriever
  family surfaced (per v2-provenance.json), re-score, and compare the 35
  configurations' MRR@10 orderings (Kendall's tau);
- answers Stage 2 couldn't reach: of the 60 v1 answers in no lexical top 10,
  how many each new configuration ranks in its top 5 and top 10; likewise the
  questions with no answer in any lexical top 10;
- target-page recall: Recall@10 over target-page answers other than the target;
- source left out: on questions with an answer besides the target, remove the
  target from the answers and every ranking (sections below move up) and test
  link 1's MRR@10 difference.

    cd "Stage 3" && ../.venv/bin/python analysis/pool_effects.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.evaluate import paired_test, run_configs  # noqa: E402
from raglab.experiments import CHALLENGER, EVALUATED, INCUMBENT, build_retrievers  # noqa: E402
from raglab.golden import Question, default_golden_path, load_golden  # noqa: E402
from raglab.metrics import RANK_DEPTH, reciprocal_rank, section_ranking  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402
from raglab.stats import bootstrap_indices  # noqa: E402

LEXICAL = ("bm25", "tfidf")
NEW_FAMILIES = ("dense", "hybrid", "hybrid+rerank", "bm25+rerank")
COMPARED = ("mrr@10", "success@5", "recall@5", "ndcg@10", "ndcg_answers@10", "coverage@4800")


def family_of(config_name: str) -> str:
    """A configuration's retriever family: 'heading+hybrid+rerank [depth 10]' -> 'hybrid+rerank'."""
    config = next(c for c in EVALUATED if c.name == config_name)
    return config.retriever


def single_family_judgments(provenance: dict) -> dict[str, set[tuple[str, str]]]:
    """For each new family, the new judgments that only that family's top 10s or windows surfaced."""
    out: dict[str, set[tuple[str, str]]] = {f: set() for f in NEW_FAMILIES}
    for item in provenance["pool"]["items"]:
        families = {family_of(n) for n in item["top10"] + item["window"]}
        if len(families) == 1 and (family := next(iter(families))) in out:
            out[family].add((item["question"], item["location"]))
    return out


def without(questions, dropped: set[tuple[str, str]]) -> list[Question]:
    return [Question(q.id, q.text, q.style, q.target,
                     MappingProxyType({loc: g for loc, g in q.judgments.items() if (q.id, loc) not in dropped}), q.notes)
            for q in questions]


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Kendall's tau-a between two orderings of the same items."""
    n, concordant, discordant = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            concordant += s > 0
            discordant += s < 0
    return float((concordant - discordant) / (n * (n - 1) / 2))


def means(results, metric: str) -> dict[str, float]:
    return {name: float(np.mean([r.values[metric] for r in per])) for name, per in results.items()}


def unreached(v1, results) -> tuple[set[tuple[str, str]], set[str]]:
    """v1 answers in no lexical configuration's top 10, and the questions with none in any lexical top 10."""
    lexical = [n for n in results if family_of(n) in LEXICAL]
    answers, questions = set(), set()
    for i, q in enumerate(v1.questions):
        reached = {loc for n in lexical for loc in results[n][i].ranking if loc}
        missed = {loc for loc in q.primary if loc not in reached}
        answers |= {(q.id, loc) for loc in missed}
        if not q.primary & reached:
            questions.add(q.id)
    return answers, questions


def reach(v1, results, answers, questions) -> dict:
    out = {}
    for name in results:
        if family_of(name) in LEXICAL:
            continue
        ranking = {q.id: results[name][i].ranking for i, q in enumerate(v1.questions)}
        out[name] = {
            "answers_in_top5": sum(loc in ranking[qid][:5] for qid, loc in answers),
            "answers_in_top10": sum(loc in ranking[qid][:10] for qid, loc in answers),
            "questions_answered_in_top5": sum(bool(v1.by_id(qid).primary & set(ranking[qid][:5])) for qid in questions),
            "questions_answered_in_top10": sum(bool(v1.by_id(qid).primary & set(ranking[qid][:10])) for qid in questions),
        }
    return {"of_answers": len(answers), "of_questions": len(questions), "configurations": out}


def target_page_recall(questions, results) -> dict:
    """Recall@10 over each question's target-page answers other than the target: judged in full, whatever ranked."""
    page = lambda loc: loc.split("#", 1)[0]  # noqa: E731
    sets = {q.id: {loc for loc in q.primary if loc != q.target and page(loc) == page(q.target)} for q in questions}
    asked = [i for i, q in enumerate(questions) if sets[q.id]]
    out = {}
    for name, per in results.items():
        out[name] = float(np.mean([len(sets[questions[i].id] & set(per[i].ranking[:10])) / len(sets[questions[i].id])
                                   for i in asked]))
    return {"questions": len(asked), "answers": sum(len(sets[questions[i].id]) for i in asked), "recall@10": out}


def source_left_out_mrr(question: Question, ranking: list[str | None]) -> float | None:
    """MRR@10 with the target removed from the answers and the ranking (sections below it move up)."""
    answers = question.primary - {question.target}
    if not answers:
        return None
    shifted = [loc for loc in ranking if loc != question.target][:RANK_DEPTH]
    return reciprocal_rank(shifted, frozenset(answers))


def main() -> None:
    golden = default_golden_path(2).parent
    if not (golden / "v2-blind.json").is_file():
        raise SystemExit("v2's labels aren't final (no v2-blind.json); these analyses run after freeze point 2")
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    index = SectionIndex(docs)
    v1, v2 = load_golden(default_golden_path(1)), load_golden(default_golden_path(2))
    provenance = json.loads((golden / "v2-provenance.json").read_text())
    on_v1, on_v2 = run_configs(docs, v1.questions, EVALUATED, index), run_configs(docs, v2.questions, EVALUATED, index)
    out: dict = {}
    out["stale_pool"] = {
        name: {"unjudged_share_of_v1_top10": float(np.mean([
            sum(loc is not None and loc not in q.judgments for loc in on_v1[name][i].ranking) / RANK_DEPTH
            for i, q in enumerate(v1.questions)])),
            **{m: {"v1": means(on_v1, m)[name], "v2": means(on_v2, m)[name]} for m in COMPARED}}
        for name in on_v2 if family_of(name) not in LEXICAL
    }
    out["stage2_on_the_wider_pool"] = {
        name: {m: {"v1": means(on_v1, m)[name], "v2": means(on_v2, m)[name]} for m in COMPARED}
        for name in on_v2 if family_of(name) in LEXICAL
    }
    base = means(on_v2, "mrr@10")
    names = sorted(base)
    out["leave_one_family_out"] = {}
    for family, dropped in single_family_judgments(provenance).items():
        rescored = means(run_configs(docs, without(v2.questions, dropped), EVALUATED, index), "mrr@10")
        out["leave_one_family_out"][family] = {
            "judgments_dropped": len(dropped),
            "mrr@10_change": {n: rescored[n] - base[n] for n in names},
            "kendall_tau": kendall_tau([base[n] for n in names], [rescored[n] for n in names]),
        }
    answers, questions = unreached(v1, on_v1)
    out["answers_stage2_could_not_reach"] = reach(v1, on_v2, answers, questions)
    out["target_page_recall"] = target_page_recall(v2.questions, on_v2)
    retrievers = build_retrievers(docs, (INCUMBENT, CHALLENGER))
    deep = {name: [section_ranking([(h.chunk.doc_id, h.chunk.start, h.chunk.end)
                                    for h in retrievers[name].search(q.text, len(retrievers[name].chunks))],
                                   index, depth=RANK_DEPTH + 1) for q in v2.questions]
            for name in (INCUMBENT.name, CHALLENGER.name)}
    kept = [i for i, q in enumerate(v2.questions) if q.primary - {q.target}]
    a = np.array([source_left_out_mrr(v2.questions[i], deep[INCUMBENT.name][i]) for i in kept])
    b = np.array([source_left_out_mrr(v2.questions[i], deep[CHALLENGER.name][i]) for i in kept])
    out["source_left_out"] = {"questions": len(kept), "link1_mrr@10": paired_test(
        [v2.questions[i] for i in kept], a, b, 1, bootstrap_indices(len(kept)))}
    path = Path(__file__).resolve().parents[1] / "results" / "pool_effects.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
