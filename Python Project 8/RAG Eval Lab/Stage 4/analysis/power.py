"""What link 1 can resolve, written down before any answer exists (PLAN.md, "What the design can resolve").

A simulation, not a measurement. Each question's chance of a supported-correct
answer (SC) depends on its window in Stage 3's committed results:
- no answering section in the 4.8k window: p0;
- an answering section: p1 * h + p0 * (1 - h), where h = best_coverage ** gamma
  (gamma 0 means a fragment is as good as the whole section).
p1 varies by question, drawn from Beta(m1 * c, (1 - m1) * c) with c = 1, the
same for both arms (some questions are hard for any window). The judge flips a
verdict with probability e, in either direction, in both arms alike. Each arm
gets R judged runs, averaged per question. The test is Stage 3's two-sided
sign-flip test at alpha 0.05. "Power" counts runs that are significant in the
predicted direction.

The last block adds what the simulation otherwise leaves out: a judge (or a
rule) that is harsher on one arm. With no true difference, a penalty d on the
baseline's SC makes a "significant" link 1 by itself.

Usage (from Stage 4/):  ../.venv/bin/python analysis/power.py [--sims 600]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
SEED = 20261121
Z80 = 1.959964 + 0.841621
BASELINE, BM25, HYBRID, HHR = ("baseline (RAG Mini Chat)", "heading+bm25", "heading+hybrid",
                               "heading+hybrid+rerank")


def load_windows():
    metrics = json.loads((ROOT / "Stage 3" / "results" / "summary.json").read_text(encoding="utf-8"))["metrics"]
    rows = [json.loads(line) for line in (ROOT / "Stage 3" / "results" / "questions.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    qs = yaml.safe_load((ROOT / "golden" / "v2.yaml").read_text(encoding="utf-8"))["questions"]
    ids = [q["id"] for q in qs]
    pages = np.unique([q["target"].split("#")[0] for q in qs], return_inverse=True)[1]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["config"]][r["question"]] = r

    def arm(config: str):
        def vec(m):
            return np.array([by[config][q]["values"][metrics.index(m)] for q in ids], dtype=float)
        return vec("budget_success@4800"), vec("best_coverage@4800")

    return ids, pages, {name: arm(name) for name in (BASELINE, BM25, HYBRID, HHR)}


class Sim:
    def __init__(self, n: int, n_pages: int, seed: int = SEED) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.signs = self.rng.choice((-1.0, 1.0), size=(2000, n))

    def pvalue(self, d: np.ndarray) -> float:
        obs = abs(d.mean())
        if obs == 0:
            return 1.0
        return (np.count_nonzero(np.abs(self.signs @ d) / self.n >= obs * (1 - 1e-12)) + 1) / (len(self.signs) + 1)

    def probs(self, G, C, p1, p0, gamma):
        h = np.where(G > 0, np.power(np.clip(C, 1e-9, 1), gamma), 0.0)
        return G * (p1 * h + (1 - h) * p0) + (1 - G) * p0

    def run(self, a, b, *, m1, e, gamma, R, p0=0.10, conc=1.0, penalty_a=0.0, sims=600) -> dict:
        (GA, CA), (GB, CB) = a, b
        hits, deltas, sds, flips = [], [], [], []
        for _ in range(sims):
            p1 = self.rng.beta(m1 * conc, (1 - m1) * conc, size=self.n)
            pa = np.clip(self.probs(GA, CA, p1, p0, gamma) - penalty_a, 0, 1)
            pb = self.probs(GB, CB, p1, p0, gamma)
            pa = pa * (1 - e) + (1 - pa) * e
            pb = pb * (1 - e) + (1 - pb) * e
            ra = self.rng.binomial(1, pa[:, None], size=(self.n, R))
            rb = self.rng.binomial(1, pb[:, None], size=(self.n, R))
            if R >= 2:
                flips.append(float(np.mean(rb[:, 0] != rb[:, 1])))  # b's run-to-run SC flip rate
            ya, yb = ra.mean(1), rb.mean(1)
            d = yb - ya
            deltas.append(d.mean())
            sds.append(d.std(ddof=1))
            hits.append(self.pvalue(d) < 0.05 and d.mean() > 0)
        sd = float(np.mean(sds))
        out = {"delta": float(np.mean(deltas)), "resolvable": Z80 * sd / np.sqrt(self.n), "power": float(np.mean(hits))}
        if flips:
            out["flip"] = float(np.mean(flips))
        return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=600)
    parser.add_argument("--json", type=Path, help="also write the grid as JSON")
    args = parser.parse_args(argv)
    ids, pages, arms = load_windows()
    sim = Sim(len(ids), int(pages.max()) + 1)
    out = {"seed": SEED, "sims": args.sims, "link1": [], "link2": [], "bias": [], "run_to_run": []}

    print("Link 1 (baseline -> heading+hybrid+rerank), 4.8k windows from Stage 3/results/questions.jsonl\n")
    print("| SC with a whole answer in context (m1) | judge error e | partial credit (gamma) | runs per arm | "
          "expected difference | resolvable | power |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for m1 in (0.35, 0.50, 0.65, 0.85):
        for e in (0.03, 0.10, 0.15):
            for gamma in (0.5, 0.25):
                for R in (1, 2):
                    r = sim.run(arms[BASELINE], arms[HHR], m1=m1, e=e, gamma=gamma, R=R, sims=args.sims)
                    out["link1"].append({"m1": m1, "e": e, "gamma": gamma, "R": R, **r})
                    print(f"| {m1:.2f} | {e:.2f} | {gamma:.2f} | {R} | {r['delta']:+.3f} | {r['resolvable']:.3f} "
                          f"| {r['power']:.2f} |")
    print("\nLink 2 (heading+bm25 -> heading+hybrid+rerank)\n")
    print("| m1 | e | gamma | runs | expected difference | resolvable | power |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for m1, e in ((0.65, 0.03), (0.85, 0.03), (0.65, 0.10)):
        for R in (1, 2):
            r = sim.run(arms[BM25], arms[HHR], m1=m1, e=e, gamma=0.5, R=R, sims=args.sims)
            out["link2"].append({"m1": m1, "e": e, "gamma": 0.5, "R": R, **r})
            print(f"| {m1:.2f} | {e:.2f} | 0.50 | {R} | {r['delta']:+.3f} | {r['resolvable']:.3f} | {r['power']:.2f} |")
    print("\nRun to run: how much a second run adds depends on how often one question's SC flips between runs.")
    print("Each question's chance is fixed across runs; a lower concentration c makes questions more all-or-nothing,")
    print("so runs agree more (m1 0.65, e 0.03, gamma 0.50). Compare the pilot's measured hhr flip rate.\n")
    print("| concentration c | runs | hhr run-to-run SC flip rate | resolvable | power |")
    print("|---:|---:|---:|---:|---:|")
    out["run_to_run"] = []
    for conc in (0.1, 1.0):
        for R in (1, 2):
            r = sim.run(arms[BASELINE], arms[HHR], m1=0.65, e=0.03, gamma=0.5, R=R, conc=conc, sims=args.sims)
            out["run_to_run"].append({"conc": conc, "R": R, **r})
            flip = f"{r['flip']:.2f}" if "flip" in r else "-"
            print(f"| {conc:.1f} | {R} | {flip} | {r['resolvable']:.3f} | {r['power']:.2f} |")
    print("\nNo true difference (gamma 0, m1 0.65, e 0.03, one run), but SC is lowered on the baseline by d\n")
    print("| penalty d on the baseline | significant for heading+hybrid+rerank, one run | two runs |")
    print("|---:|---:|---:|")
    for d in (0.0, 0.05, 0.10):
        r1 = sim.run(arms[BASELINE], arms[HHR], m1=0.65, e=0.03, gamma=0.0, R=1, penalty_a=d, sims=args.sims)
        r2 = sim.run(arms[BASELINE], arms[HHR], m1=0.65, e=0.03, gamma=0.0, R=2, penalty_a=d, sims=args.sims)
        out["bias"].append({"penalty": d, "one_run": r1["power"], "two_runs": r2["power"]})
        print(f"| {d:.2f} | {r1['power']:.2f} | {r2['power']:.2f} |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
