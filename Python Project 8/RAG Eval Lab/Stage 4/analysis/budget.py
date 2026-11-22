"""The Stage 4 budget, computed from token assumptions and pricing.json (PLAN.md, "Cost").

Every figure in the plan's budget tables comes from this script, so the
arithmetic can be re-run rather than trusted. The token counts are assumptions
until the pilot measures them; count_tokens replaces every input figure before
any golden request is reserved. Costs here are expected costs at list prices
(global inference); reservations, which bound the worst case, also assume the
1.1x US-only rate and are not what is shown.

The judged items are scheduled by the scope rule itself (raglab/scope.py), the
same function `judge plan` applies at freeze point 4: items in the developer's
priority order while the projected total (fixed costs, the reserve, and each
judged item's expected cost x 1.15) stays at or under the planning ceiling,
the reference check sized by what else is judged, and the effort rule choosing
medium or low.

Usage (from Stage 4/):  ../.venv/bin/python analysis/budget.py
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from raglab import params, scope  # noqa: E402
from raglab.ledger import Pricing  # noqa: E402
from raglab.params import CAP, PLANNING_CEILING as CEILING, PROJECTION, RESERVE  # noqa: E402

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
FACTOR = Decimal(str(PROJECTION["factor"]))
REKEYED = 7  # questions whose reference the check rebuilds: an assumption until the top-up is done

# (central, pessimistic) input tokens per Haiku request, by window
GEN_INPUT = {"hhr-4.8k": (2560, 2900), "base-4.8k": (4600, 5040), "base-4.8k-notitle": (3350, 3700),
             "bm25-4.8k": (2600, 2950), "hybrid-4.8k": (2580, 2920), "masked-4.8k": (2560, 2900),
             "hhr-2.4k": (1590, 1800), "hhr-9.6k": (4460, 5000), "oracle-4.8k": (1750, 2000), "closed": (400, 450),
             "huge-pilot-window": (40000, 46000)}
# (central, pessimistic) output tokens, thinking included
GEN_OUTPUT = {"off": (350, 700), "low": (600, 1200), "med": (1100, 2500), "high": (2200, 5000),
              "xhigh": (4000, 9000), "max": (7000, 16000)}
# Judge calls, ((input, output) central, (input, output) pessimistic). F's input carries the whole answer, every
# unit again and each unit's excerpts; C's the key, the reference (20,000-character cap) and the answer.
F_MED = ((2400, 1400), (2900, 2600))
C_MED = ((2800, 1050), (3300, 2250))
F_LOW = ((2400, 700), (2900, 1300))
C_LOW = ((2800, 500), (3300, 1100))

GENERATION = [
    ("hhr-4.8k", "med", 3 * 98), ("base-4.8k", "med", 2 * 98), ("bm25-4.8k", "med", 98),
    ("masked-4.8k", "med", 84), ("closed", "med", 98), ("base-4.8k-notitle", "med", 98),
    ("hhr-4.8k", "off", 98), ("hhr-4.8k", "low", 98), ("hhr-4.8k", "high", 98),
    ("hhr-4.8k", "xhigh", 98), ("hhr-4.8k", "max", 98), ("hhr-9.6k", "med", 98), ("hhr-2.4k", "med", 98),
    ("hybrid-4.8k", "med", 98), ("oracle-4.8k", "med", 98),
]
PILOT_GENERATION = [("hhr-4.8k", "med", 24), ("base-4.8k", "med", 12), ("base-4.8k-notitle", "med", 12),
                    ("closed", "med", 12), ("hhr-4.8k", "off", 12), ("hhr-4.8k", "low", 12),
                    ("hhr-4.8k", "high", 12), ("hhr-4.8k", "xhigh", 12), ("hhr-4.8k", "max", 12),
                    ("hhr-9.6k", "med", 12)]


def answers(rows) -> int:
    return sum(n for _, _, n in rows)


class Book:
    def __init__(self, pricing: Pricing, case: int, low: bool = False) -> None:
        self.p, self.case, self.low = pricing, case, low  # case 0 = central, 1 = pessimistic

    def call(self, model: str, tokens: tuple[int, int], batch: bool) -> Decimal:
        r = self.p.rates(model)
        inp, out = tokens
        return (inp * r.input + out * r.output) * (self.p.batch_multiplier if batch else 1)

    def gen(self, rows, batch=True) -> Decimal:
        return sum((n * self.call(HAIKU, (GEN_INPUT[w][self.case], GEN_OUTPUT[e][self.case]), batch)
                    for w, e, n in rows), Decimal(0))

    def f(self, n: int, low: bool | None = None, batch: bool = True) -> Decimal:
        return n * self.call(OPUS, (F_LOW if (self.low if low is None else low) else F_MED)[self.case], batch)

    def c(self, n: int, low: bool | None = None, batch: bool = True) -> Decimal:
        return n * self.call(OPUS, (C_LOW if (self.low if low is None else low) else C_MED)[self.case], batch)

    def fc(self, n: int, low: bool | None = None) -> Decimal:
        return self.f(n, low) + self.c(n, low)


def fixed_items(b: Book) -> list[tuple[str, Decimal]]:
    smoke = (b.gen([("hhr-4.8k", "med", 1), ("base-4.8k", "med", 1), ("closed", "med", 1),
                    ("huge-pilot-window", "med", 2)], batch=False)
             + b.f(2, low=False, batch=False) + b.c(1, low=False, batch=False))
    pilot_gen = b.gen(PILOT_GENERATION) + 48 * b.call(
        HAIKU, (GEN_INPUT["hhr-4.8k"][b.case], GEN_OUTPUT["med"][b.case]), False)
    # C is never run on the 2 questions no section answers (no key), so its counts are 10 per 12 answers.
    pilot_judging = (b.f(36, low=False) + b.c(30, low=False) + b.f(12, low=False) + b.c(10, low=False)
                     + b.f(24, low=True) + b.c(20, low=True) + b.f(12, low=False) + b.f(12, low=True)
                     + b.c(10, low=False))
    n_pilot, n_gen = answers(PILOT_GENERATION), answers(GENERATION)
    return [
        ("smoke test, standard API: 5 answers (2 of them ~300-document windows) under pilot; 2 support calls and "
         "1 correctness call first under pilot-judging", smoke),
        (f"pilot generation: {n_pilot} answers, plus 48 timing-probe calls", pilot_gen),
        ("pilot judging: F on 36 and C on 30 at medium, 12 and 10 re-judged, F on 24 and C on 20 at low (both "
         "arms), 12 swaps at both efforts, C on 10 closed-book", pilot_judging),
        (f"golden generation: {n_gen:,} answers, batch", b.gen(GENERATION)),
        ("timing run: 200 standard streaming calls after freeze point 4",
         200 * b.call(HAIKU, (GEN_INPUT["hhr-4.8k"][b.case], GEN_OUTPUT["med"][b.case]), False)),
        ("reserve: one re-run per failed judgment, truncation re-runs, overruns", RESERVE),
    ]


def _calls(b: Book, judge: str, n: int, effort: str) -> scope.Calls:
    table = {("F", "medium"): F_MED, ("C", "medium"): C_MED, ("F", "low"): F_LOW, ("C", "low"): C_LOW}
    return scope.Calls(judge, effort, n, n * table[(judge, effort)][b.case][0])


def _part(b: Book, f: int = 0, c: int = 0, requires: str | None = None) -> scope.Part:
    def calls(effort: str) -> list[scope.Calls]:
        return ([_calls(b, "F", f, effort)] if f else []) + ([_calls(b, "C", c, effort)] if c else [])

    return scope.Part({e: calls(e) for e in scope.EFFORTS}, requires)


LABELS = {
    "core": "core: link 1 run 1 in both arms (support + correctness), 20 near-miss swaps and 20 true-support plants",
    "link1-run2": "1. link 1 run 2 in both arms",
    "closed-book": "2. closed-book control (correctness)",
    "reference-check": f"3. reference check: C re-judges every judged link-1 and closed-book answer of ~{REKEYED} "
                       "re-keyed questions",
    "self-agreement": "4. self-agreement: the 24 hand-graded answers judged again",
    "link2": "5. link 2: heading+bm25",
    "masked": "6. masked windows: 40 that don't abstain",
}
EXPLORATORY = params.PRIORITY[6:]


def items(b: Book) -> list[scope.Item]:
    """The golden judge items as the scope rule sees them, the core first, then params.PRIORITY's order."""
    out = [scope.Item("core", [_part(b, f=196, c=196), _part(b, f=40)]),
           scope.Item("link1-run2", [_part(b, f=196, c=196)]),
           scope.Item("closed-book", [_part(b, c=98)]),
           scope.Item("reference-check", [_part(b, c=2 * REKEYED), _part(b, c=2 * REKEYED, requires="link1-run2"),
                                          _part(b, c=REKEYED, requires="closed-book")]),
           scope.Item("self-agreement", [_part(b, f=24, c=24)]),
           scope.Item("link2", [_part(b, f=98, c=98)]),
           scope.Item("masked", [_part(b, f=40, c=40)])]
    out += [scope.Item(name, [_part(b, f=98, c=98)]) for name in EXPLORATORY]
    assert [i.name for i in out[1:]] == list(params.PRIORITY), "the items must follow the plan's priority order"
    return out


def prices(b: Book) -> scope.Prices:
    r = b.p.rates(OPUS)
    table = {("F", "medium"): F_MED, ("C", "medium"): C_MED, ("F", "low"): F_LOW, ("C", "low"): C_LOW}
    return scope.Prices(r.input * b.p.batch_multiplier, r.output * b.p.batch_multiplier,
                        {k: Decimal(v[b.case][1]) for k, v in table.items()})


def label(name: str) -> str:
    return LABELS.get(name) or f"{8 + EXPLORATORY.index(name) - 1}. exploratory arm {name}"


def columns(b: Book, fixed: Decimal, effort: str) -> tuple[scope.Decision, dict[str, tuple[Decimal, str]]]:
    """One case's schedule at one effort, with each item's expected cost and the running projection."""
    its, pr = items(b), prices(b)
    d = scope.schedule(its, effort, fixed, pr)
    running, judged, col = fixed, set(), {}
    for it in its:
        cost = scope._item_cost(it, effort, judged, pr, its)
        if it.name in d.judged:
            running += cost * FACTOR
            judged.add(it.name)
            col[it.name] = (cost, money(running))
        else:
            col[it.name] = (cost, "not run")
    return d, col


def money(x: Decimal) -> str:
    return f"${x.quantize(Decimal('0.01'))}"


def main() -> int:
    pricing = Pricing.load(STAGE / "pricing.json")
    books = [Book(pricing, 0), Book(pricing, 1)]
    print("Unit costs (batch), per 1,000 calls: support (F) central "
          f"{money(books[0].f(1000))}, correctness (C) {money(books[0].c(1000))}; pessimistic "
          f"{money(books[1].f(1000))} and {money(books[1].c(1000))}\n")
    print("| item | central | pessimistic |\n|---|---:|---:|")
    for (name, a), (_, c) in zip(fixed_items(books[0]), fixed_items(books[1])):
        print(f"| {name} | {money(a)} | {money(c)} |")
    fixed = [sum(c for _, c in fixed_items(bk)) for bk in books]  # the reserve is one of them
    print(f"| **fixed items** | **{money(fixed[0])}** | **{money(fixed[1])}** |\n")
    for effort in scope.EFFORTS:
        (d0, c0), (d1, c1) = columns(books[0], fixed[0], effort), columns(books[1], fixed[1], effort)
        print(f"Golden judging at {effort} effort: judged in order while the projection (expected x {FACTOR}) "
              f"stays at or under {money(CEILING)}\n")
        print("| item | central | projected total | pessimistic | projected total |\n|---|---:|---:|---:|---:|")
        for it in items(books[0]):
            (a, ta), (c, tc) = c0[it.name], c1[it.name]
            print(f"| {label(it.name)} | {money(a)} | {ta} | {money(c)} | {tc} |")
        spend = [fx + sum(col[n][0] for n in d.judged) for fx, col, d in ((fixed[0], c0, d0), (fixed[1], c1, d1))]
        print(f"\nExpected spend at {effort}: {money(spend[0])} central, {money(spend[1])} pessimistic "
              f"(hard cap {money(CAP)}).\n")
    print("The scope rule's decision (raglab/scope.decide):\n")
    for low_passed in (True, False):
        for name, bk, fx in zip(("central", "pessimistic"), books, fixed):
            d = scope.decide(items(bk), fx, prices(bk), low_passed=low_passed)
            _, col = columns(bk, fx, d.effort)
            spend = fx + sum(col[n][0] for n in d.judged)
            print(f"- {name}, low {'passed' if low_passed else 'failed'} rule 3: {d.effort} effort; judged "
                  f"{', '.join(d.judged)}; projected {money(d.projected)}; expected spend {money(spend)} "
                  f"({'; '.join(d.log)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
