"""`pilot report`: the pilot's rules 1-7 applied to its stored answers and verdicts (PLAN.md, "The pilot").

It writes pilot/REPORT.json, which later code reads (golden max_tokens, the
answer prompt, whether low effort passed, the unit costs, the freeze point 1
commit), and pilot/REPORT.md, the same for a person. No rule here changes a
test, a metric or a decision rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from raglab.judge import outcome
from raglab.judging import needs_judging
from raglab.pilot import (
    flip_rate, integrity_stop, low_effort_rule, max_tokens_rule, prompt_fix_needed, uncited_claim_share, unit_costs,
)
from raglab.windows import CONDITIONS

SPLITTER_FIX_FIELDS = ("rule", "answers", "reason", "commit", "date")
ARMS = {"hhr": ("hhr-4.8k-med", 0), "baseline": ("base-4.8k-med", 0)}
M1_GRID = (0.35, 0.50, 0.65, 0.85)


def effort_key(condition: str) -> str:
    c = CONDITIONS[condition]
    return "off" if c.thinking == "disabled" else c.effort


def _answer(stored, condition: str, question: str, sample: int):
    for cid, s in stored.items():
        if s.line.condition == condition and s.line.question == question and s.line.sample == sample:
            return cid, s
    return None, None


def verdict_of(judgments: Mapping, judge: str, answered: str, item: str, sample: int = 0):
    j = judgments.get((judge, answered, item, None, sample))
    return j.verdict if j is not None and j.state == "verdict" else None


def sc_of(s, f, c, needs_c: bool) -> int | None:
    """SC for one answer: 0 for a code-flagged abstention at any effort, None if a needed judge is missing."""
    if not needs_judging(s.answer):
        return 0
    if f is None or (needs_c and c is None):
        return None
    return outcome(s.answer, f, c).sc


def rule3_inputs(stored, questions, graded, judgments, swap_results) -> tuple[dict, dict]:
    """Per arm: unit labels, nugget labels and SC at medium and at low, aligned answer by answer. SC is kept per
    question too ("sc_by_question"), so the two arms are paired on the same questions."""
    arms = {}
    for arm, (condition, sample) in ARMS.items():
        units_m, units_l, nug_m, nug_l, sc_m, sc_l = [], [], [], [], [], []
        by_q: dict[str, tuple] = {}
        for q in questions:
            cid, s = _answer(stored, condition, q.id, sample)
            if s is None:
                continue
            g = graded[q.id]
            fm, fl = (verdict_of(judgments, "F", cid, item) for item in ("pilot-medium", "pilot-low"))
            cm, cl = (verdict_of(judgments, "C", cid, item) for item in ("pilot-medium", "pilot-low"))
            if needs_judging(s.answer):
                for u in s.answer.units:
                    units_m.append(next((v.support for v in fm.units if v.id == u.id), None) if fm else None)
                    units_l.append(next((v.support for v in fl.units if v.id == u.id), None) if fl else None)
                for n in (g.key or ()):
                    nug_m.append(cm.conveyed.get(n.id) if cm else None)
                    nug_l.append(cl.conveyed.get(n.id) if cl else None)
            sc_m.append(sc_of(s, fm, cm, g.key is not None))
            sc_l.append(sc_of(s, fl, cl, g.key is not None))
            by_q[q.id] = (sc_m[-1], sc_l[-1])
        arms[arm] = {"units": (units_m, units_l), "nuggets": (nug_m, nug_l), "sc": (sc_m, sc_l), "sc_by_question": by_q}
    return arms, swap_results


def paired_difference(hhr: Mapping[str, int | None], baseline: Mapping[str, int | None]) -> dict:
    """The pilot's hhr-minus-baseline SC difference, paired question by question over those both arms have a
    verdict on."""
    pairs = [(hhr[q], baseline[q]) for q in sorted(set(hhr) & set(baseline))
             if hhr[q] is not None and baseline[q] is not None]
    return {"questions": len(pairs), "difference": (sum(a - b for a, b in pairs) / len(pairs)) if pairs else None}


def swap_results(gated, judgments, effort: str) -> dict:
    """Per arm and kind, (caught, judged) for the gated swaps at one effort. A swap is caught when F doesn't call
    the swapped unit supported with a holding quote; an unjudged swap counts as judged and not caught."""
    item = f"pilot-swaps-{effort}"
    arm_of = {cond: arm for arm, (cond, _) in ARMS.items()}
    out: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for c in gated:
        arm = arm_of[c.arm]
        j = judgments.get(("F", c.answered, item, c.name, 0))
        v = j.verdict if j is not None and j.state == "verdict" else None
        unit = next(u for u in v.units if u.id == c.unit) if v is not None else None
        caught = unit is not None and unit.claim and not unit.supported
        out[arm][c.kind][0] += int(caught)
        out[arm][c.kind][1] += 1
    return {arm: {k: tuple(v) for k, v in kinds.items()} for arm, kinds in out.items()}


def relabelled_swaps(gated, judgments, effort: str) -> int:
    """Swaps F waved through by calling the swapped unit "not a claim": never counted as caught, reported apart."""
    n = 0
    for c in gated:
        j = judgments.get(("F", c.answered, f"pilot-swaps-{effort}", c.name, 0))
        if j is not None and j.state == "verdict":
            n += int(not next(u for u in j.verdict.units if u.id == c.unit).claim)
    return n


def whole_window_sc(stored, questions, graded, judgments, windows, index) -> tuple[float | None, int]:
    """Rule 4's input: hhr run 1's SC at medium on questions whose 4.8k window holds a whole grade-2 section."""
    from raglab.metrics import coverage_fractions

    scores = []
    condition, sample = ARMS["hhr"]
    for q in questions:
        if not q.answering:
            continue
        refs = [index.get(loc) for loc in sorted(q.answering) if index.get(loc) is not None]
        spans = windows.window(CONDITIONS[condition], q.text, q.answering, q.target)
        if not refs or max(coverage_fractions(spans, refs, index)) < 0.999:
            continue
        cid, s = _answer(stored, condition, q.id, sample)
        if s is None:
            continue
        sc = sc_of(s, verdict_of(judgments, "F", cid, "pilot-medium"), verdict_of(judgments, "C", cid, "pilot-medium"),
                   graded[q.id].key is not None)
        if sc is not None:
            scores.append(sc)
    return (sum(scores) / len(scores) if scores else None), len(scores)


POWER_GRID = Path(__file__).resolve().parents[1] / "analysis" / "power-grid.json"


def power_range(m1: float, grid_path: Path = POWER_GRID) -> list[dict]:
    """Rule 4: the frozen grid's rows (analysis/power.py's own run, committed at freeze point 1) at the grid m1
    nearest the pilot's, over e in {0.03, 0.10}, gamma in {0.25, 0.50} and one or two runs. Read, never
    re-simulated, so the rows are exactly the ones the plan quotes."""
    grid = json.loads(grid_path.read_text(encoding="utf-8"))["link1"]
    rows = [{"m1": r["m1"], "e": r["e"], "gamma": r["gamma"], "runs": r["R"], "delta": r["delta"],
             "resolvable": r["resolvable"], "power": r["power"]}
            for r in grid if r["m1"] == m1 and r["e"] in (0.03, 0.10) and r["gamma"] in (0.25, 0.5)]
    if len(rows) != 8:
        raise ValueError(f"analysis/power-grid.json has {len(rows)} rows for m1 {m1}, not 8")
    return sorted(rows, key=lambda r: (r["e"], r["gamma"], r["runs"]))


def assemble(*, stored, questions, graded, judgments, gated, generation_ids, judge_kinds, settled_usage,
             smoke: Mapping, fp1_commit: str, answer_v2_exists: bool, windows, index, power=None,
             splitter_fix: Mapping | None = None) -> dict:
    """The report. `generation_ids` are the pilot's batch answers (not smoke or timing); `judge_kinds` maps each
    judge request's custom_id to its call kind ("F-medium", ...); `settled_usage` is the ledger's usage by custom_id."""
    gen = {cid: stored[cid] for cid in generation_ids if cid in stored}
    haiku = defaultdict(list)
    by_condition = defaultdict(list)
    for s in gen.values():
        haiku[effort_key(s.line.condition)].append(int((settled_usage.get(s.line.custom_id) or {})
                                                        .get("output_tokens") or 0))
        by_condition[s.line.condition].append(s.answer)
    judges = defaultdict(list)
    kinds = defaultdict(list)
    for cid, kind in judge_kinds.items():
        if cid in settled_usage:
            judges["support" if kind.startswith("F") else "correctness"].append(
                int(settled_usage[cid].get("output_tokens") or 0))
            kinds[kind].append(settled_usage[cid])
    rule1 = max_tokens_rule(dict(haiku), dict(judges))

    rule2 = prompt_fix_needed(dict(by_condition))
    units_by_arm = {}
    for arm, (condition, sample) in ARMS.items():
        pairs = []
        for q in questions:
            cid, s = _answer(stored, condition, q.id, sample)
            f = verdict_of(judgments, "F", cid, "pilot-medium") if s else None
            if f is not None:
                pairs += [(bool(u.citations), v.claim) for u, v in zip(s.answer.units, f.units)]
        units_by_arm[arm] = pairs
    rule2["uncited_claim_share"] = uncited_claim_share(units_by_arm)

    arms, swaps_low = rule3_inputs(stored, questions, graded, judgments, swap_results(gated, judgments, "low"))
    rule3 = low_effort_rule({a: {k: v for k, v in x.items() if k != "sc_by_question"} for a, x in arms.items()},
                            swaps_low)
    rule3["swaps_medium"] = swap_results(gated, judgments, "medium")
    rule3["swaps_relabelled"] = {e: relabelled_swaps(gated, judgments, e) for e in ("medium", "low")}
    rule3["paired_difference"] = {
        effort: paired_difference({q: v[i] for q, v in arms["hhr"]["sc_by_question"].items()},
                                  {q: v[i] for q, v in arms["baseline"]["sc_by_question"].items()})
        for i, effort in enumerate(("medium", "low"))}
    for a in arms.values():
        a.pop("sc_by_question")

    hhr_cond, _ = ARMS["hhr"]
    first, again, run2 = [], [], []
    for q in questions:
        cid, s = _answer(stored, hhr_cond, q.id, 0)
        cid2, s2 = _answer(stored, hhr_cond, q.id, 1)
        if s is None:
            continue
        needs_c = graded[q.id].key is not None
        a = sc_of(s, verdict_of(judgments, "F", cid, "pilot-medium"), verdict_of(judgments, "C", cid, "pilot-medium"),
                  needs_c)
        b = sc_of(s, verdict_of(judgments, "F", cid, "pilot-noise", 2), verdict_of(judgments, "C", cid, "pilot-noise", 2),
                  needs_c)
        c = sc_of(s2, verdict_of(judgments, "F", cid2, "pilot-medium"),
                  verdict_of(judgments, "C", cid2, "pilot-medium"), needs_c) if s2 else None
        if a is not None and b is not None:
            first.append(a)
            again.append(b)
        if a is not None and c is not None:
            run2.append((a, c))
    rejudge_flip = flip_rate(first, again)
    run_flip = flip_rate([x for x, _ in run2], [y for _, y in run2])
    sc_whole, n_whole = whole_window_sc(stored, questions, graded, judgments, windows, index)
    e = (rejudge_flip or 0.0) / 2
    m1 = None if sc_whole is None else min(max((sc_whole - e) / (1 - 2 * e), 0.0), 1.0)
    grid = None if m1 is None else min(M1_GRID, key=lambda g: abs(g - m1))
    rule4 = {"whole_window_sc": sc_whole, "whole_window_questions": n_whole, "rejudge_flip_rate": rejudge_flip,
             "e": e, "m1": m1, "m1_grid": grid, "hhr_run_to_run_flip_rate": run_flip,
             "power": (power or power_range)(grid) if grid is not None else []}

    rule5 = unit_costs(dict(kinds))
    # Every pilot answer with citations: the batches, the smoke test and the timing probe.
    rule6 = integrity_stop([s.answer for s in stored.values()])
    fix = rule2["needed"]
    if fix and not answer_v2_exists:
        prompt = None  # the developer writes prompts/answer-v2.md, logs the reason, and runs the report again
    else:
        prompt = "answer-v2.md" if fix else "answer-v1.md"
    return {
        "fp1_commit": fp1_commit,
        "answer_prompt": prompt,
        "max_tokens": rule1,
        "rule2_prompt_fix": rule2,
        "rule3_low_effort": rule3,
        "low_passed": bool(rule3["passes"]),
        "rule4_power": rule4,
        "rule5_unit_costs": rule5,
        "rule6_citation_failures": rule6,
        "rule7_splitter_fix": dict(splitter_fix) if splitter_fix else None,
        "smoke": dict(smoke),
        "stop": bool(rule6) or prompt is None,
    }


def render(report: Mapping) -> str:
    """REPORT.md: the same content, for a person."""
    r2, r3, r4 = report["rule2_prompt_fix"], report["rule3_low_effort"], report["rule4_power"]
    lines = ["# Stage 4 pilot report", "",
             f"Freeze point 1: commit `{report['fp1_commit']}`. Every rule below was fixed there (PLAN.md, "
             "\"The pilot\"); none changes a test, a metric or a decision rule.", ""]
    if report["stop"]:
        lines += ["**The stage is stopped:** " + ("citations failed their check (rule 6)" if report["rule6_citation_failures"]
                                                   else "rule 2 calls for a prompt fix: write prompts/answer-v2.md, "
                                                        "log why, and run the report again"), ""]
    mt = report["max_tokens"]
    lines += ["## Rule 1: max_tokens for golden requests", "",
              "| calls | max_tokens |", "|---|---:|"]
    lines += [f"| Haiku, {k} | {v:,} |" for k, v in sorted(mt["haiku"].items())]
    lines += [f"| {k} judge | {v:,} |" for k, v in sorted(mt["judge"].items())]
    lines += ["", "## Rule 2: the answer prompt", "",
              f"Prompt for golden generation: `{report['answer_prompt']}`.",
              f"Fix needed: {'yes' if r2['needed'] else 'no'}" + (f" ({'; '.join(r2['reasons'])})" if r2["reasons"] else "")
              + f". Format failures: {len(r2['format_failures'])}; truncated: {r2['truncated']}.",
              f"Cited share of 40+ character units: hhr {_pct(r2['prose_citation_share']['hhr'])}, baseline "
              f"{_pct(r2['prose_citation_share']['baseline'])}; code units {_pct(r2['code_citation_share'])}.",
              f"Uncited units F labels claims: hhr {_pct(r2['uncited_claim_share'].get('hhr'))}, baseline "
              f"{_pct(r2['uncited_claim_share'].get('baseline'))}.", "",
              "## Rule 3: low judge effort", "",
              f"Low effort {'passed' if report['low_passed'] else 'failed'}: {r3['swaps_caught']} of "
              f"{r3['swaps_judged']} swaps caught ({r3['swaps_relabelled']['low']} waved through as not a claim). "
              f"The paired hhr-minus-baseline SC difference is {_num(r3['paired_difference']['medium']['difference'])} "
              f"at medium and {_num(r3['paired_difference']['low']['difference'])} at low.", ""]
    for arm, a in r3["arms"].items():
        lines.append(f"- {arm}: units {_pct(a['units'])}, nuggets {_pct(a['nuggets'])}, SC agrees on "
                     f"{a['sc_agree']} of {a['sc_compared']}; swaps caught {a['caught']} of {a['judged']} "
                     f"({', '.join(f'{k} {c}/{j}' for k, (c, j) in a['swaps'].items())}).")
    lines += ["", "## Rule 4: what the design can resolve", "",
              f"Whole-window SC {_num(r4['whole_window_sc'])} on {r4['whole_window_questions']} questions; re-judge flip "
              f"rate {_num(r4['rejudge_flip_rate'])} (e = {_num(r4['e'])}); m1 = {_num(r4['m1'])}, nearest grid value "
              f"{r4['m1_grid']}; hhr run-to-run flip rate {_num(r4['hhr_run_to_run_flip_rate'])}.", "",
              "| e | gamma | runs | expected difference | resolvable | power |", "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {p['e']:.2f} | {p['gamma']:.2f} | {p['runs']} | {p['delta']:+.3f} | {p['resolvable']:.3f} | "
              f"{p['power']:.2f} |" for p in r4["power"]]
    lines += ["", "## Rule 5: unit costs (mean output tokens per call)", "", "| call | calls | mean output tokens |",
              "|---|---:|---:|"]
    lines += [f"| {k} | {v['calls']} | {v['mean_output_tokens']} |" for k, v in sorted(report["rule5_unit_costs"].items())]
    lines += ["", "## Rule 6: citation integrity", "",
              "Every citation mapped." if not report["rule6_citation_failures"] else
              "Failed citations (a mapping bug: the stage stops):\n" + "\n".join(f"- {f}" for f in
                                                                                report["rule6_citation_failures"]),
              "", "## Rule 7: the unit splitter", "", "No fix." if report["rule7_splitter_fix"] is None else
              _fix_text(report["rule7_splitter_fix"]), ""]
    return "\n".join(lines)


def _fix_text(fix: Mapping) -> str:
    return (f"Fixed once ({fix['date']}, commit `{fix['commit']}`): {fix['rule']}. Shown by: "
            f"{', '.join(fix['answers'])}. {fix['reason']}")


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _num(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def write(report: Mapping, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
                                        encoding="utf-8")
    (folder / "REPORT.md").write_text(render(report), encoding="utf-8")

