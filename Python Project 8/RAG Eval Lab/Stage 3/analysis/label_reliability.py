"""How far v2's new labels can be trusted (PLAN.md, "Labels"). Reads grades only; computes no metric.

- Drift between the rounds, from the 200 hidden calibration items: promotions
  (v1 grade 0 or 1, graded 2 this round) and demotions (v1 answers graded 0 or
  1), each with a Wilson interval, overall and for each half; demotions also
  for the off-target-page answers alone.
- The blind round. The label check's figure: Cohen's kappa on answer versus
  not, unweighted, between the blind and v2 grades over the union's new
  labels. The agreement sample, also reweighted to the grade counts of all
  new labels as v1's study was. The bridge: the blind judge now against v1's
  blind judge on the same 40 items.
- The quote-checking judge: for each first grade, the share of items it calls
  answers with a quotation that is really in the section.

    cd "Stage 3" && ../.venv/bin/python analysis/label_reliability.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from raglab.golden import default_golden_path  # noqa: E402
from raglab.stats import cohen_kappa, wilson_ci  # noqa: E402

GOLDEN = default_golden_path(2).parent


def rate(hits: list[bool]) -> dict:
    if not hits:
        return {"n": 0}
    mean, lo, hi = wilson_ci([float(h) for h in hits])
    return {"n": len(hits), "count": int(sum(hits)), "rate": mean, "lo": lo, "hi": hi}


def agreement(first: list[int], second: list[int], weights: list[float] | None = None) -> dict:
    if len(first) < 2:
        return {"n": len(first)}
    w = np.ones(len(first)) if weights is None else np.asarray(weights, dtype=float)
    a, b = np.asarray(first), np.asarray(second)
    return {
        "n": len(first),
        "exact": float(np.sum(w * (a == b)) / np.sum(w)),
        "kappa": cohen_kappa(first, second, sample_weight=None if weights is None else list(w)),
        "answer_kappa": cohen_kappa([int(x == 2) for x in first], [int(x == 2) for x in second], categories=(0, 1),
                                    sample_weight=None if weights is None else list(w)),
        "promoted_to_answer": int(np.sum((a != 2) & (b == 2))),
        "demoted_from_answer": int(np.sum((a == 2) & (b != 2))),
    }


def drift(calibration: list[dict]) -> dict:
    out = {}
    for label, keep in (("all", lambda c: True), ("in heading+bm25's top 10", lambda c: c["in_heading_bm25_top10"]),
                        ("not in it", lambda c: not c["in_heading_bm25_top10"])):
        items = [c for c in calibration if keep(c)]
        out[label] = {
            "promotions": rate([c["v2_round_grade"] == 2 for c in items if c["v1_grade"] in (0, 1)]),
            "promotions_from_1": rate([c["v2_round_grade"] == 2 for c in items if c["v1_grade"] == 1]),
            "demotions": rate([c["v2_round_grade"] != 2 for c in items if c["v1_grade"] == 2]),
        }
    off_page = [c for c in calibration if c["v1_grade"] == 2 and not c["on_target_page"]]
    out["demotions of off-target-page answers"] = rate([c["v2_round_grade"] != 2 for c in off_page])
    out["test_retest"] = agreement([c["v1_grade"] for c in calibration], [c["v2_round_grade"] for c in calibration])
    return out


def blind_report(rows: list[dict], new_grade_counts: Counter) -> dict:
    union_new = [r for r in rows if "union" in r["groups"] and r["new_in_v2"]]
    union_old = [r for r in rows if "union" in r["groups"] and not r["new_in_v2"]]
    sample = [r for r in rows if "agreement" in r["groups"]]
    bridge = [r for r in rows if "bridge" in r["groups"]]
    in_sample = Counter(r["first"] for r in sample)
    weights = [new_grade_counts[r["first"]] / in_sample[r["first"]] for r in sample]
    label_check = (cohen_kappa([int(r["first"] == 2) for r in union_new], [int(r["blind"] == 2) for r in union_new],
                               categories=(0, 1)) if len(union_new) > 1 else None)
    return {
        "label_check_kappa_answer_vs_not": label_check,
        "union_new_labels": agreement([r["first"] for r in union_new], [r["blind"] for r in union_new]),
        "union_v1_labels": agreement([r["first"] for r in union_old], [r["blind"] for r in union_old]),
        "agreement_sample": agreement([r["first"] for r in sample], [r["blind"] for r in sample]),
        "agreement_sample_reweighted_to_all_new_labels": agreement([r["first"] for r in sample],
                                                                   [r["blind"] for r in sample], weights),
        "bridge_blind_now_vs_v1_blind": agreement([r["v1_blind"] for r in bridge], [r["blind"] for r in bridge]),
    }


def quote_report(items: list[dict]) -> dict:
    out = {}
    for grade in (2, 1, 0):
        these = [q for q in items if q["first"] == grade]
        out[f"first grade {grade}"] = {
            "answers_with_a_quote_that_holds": rate([q["answers"] and q["quote_holds"] for q in these]),
            "answers_whose_quote_fails": rate([q["answers"] and not q["quote_holds"] for q in these]),
        }
    return out


def main() -> None:
    calibration = json.loads((GOLDEN / "v2-calibration.json").read_text())["items"]
    rows = json.loads((GOLDEN / "v2-blind.json").read_text())["items"]
    quotes = json.loads((GOLDEN / "v2-quote.json").read_text())["items"]
    judged = json.loads((GOLDEN / "v2-judgments.json").read_text())
    provenance = json.loads((GOLDEN / "v2-provenance.json").read_text())
    pool_keys = {(i["question"], i["location"]) for i in provenance["pool"]["items"]}
    new_grade_counts = Counter(r["grade"] for r in judged if (r["question"], r["location"]) in pool_keys)
    result = {"new_labels": dict(sorted(new_grade_counts.items())), "drift": drift(calibration),
              "blind": blind_report(rows, new_grade_counts), "quote": quote_report(quotes)}
    path = Path(__file__).resolve().parents[1] / "results" / "label_reliability.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
