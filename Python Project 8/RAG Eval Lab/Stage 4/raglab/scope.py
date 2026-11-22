"""The scope rule: which golden judge items are judged, and at which effort (PLAN.md, "Scope rule").

`judge plan` applies it at freeze point 4, and again after the core wave from
the ledger's settled usage; analysis/budget.py applies the same function to the
plan's token assumptions, so the plan's tables are this rule's output.

**The projection.**
- projected total = committed spend (settled, plus open reservations at their
  bounds) + the timing run while it hasn't run + the $1.00 reserve + the sum,
  over the items judged, of 1.15 x each item's expected cost;
- an item's expected cost = the sum over its calls of (count_tokens input x
  the batch input rate + the pilot's mean output tokens for that call kind x
  the batch output rate), x 1.1 if US-only inference was accepted for the
  judging phase (a pilot acceptance never carries over);
- every comparison is with the $24.50 planning ceiling, the core's too.

**The order.** The core (link 1 run 1 in both arms, and the swaps and plants)
is always judged. Then each item in the developer's priority order is judged if
the projected total stays at or under the ceiling; one that doesn't fit is
skipped and later items are still considered. Run 2 is one item, so it is
judged only whole, in both arms. An item's parts can depend on another item:
the reference check re-judges run 2's answers only if run 2 is judged, and the
closed-book answers only if the closed-book control is.

**The effort.** Medium, unless low passed pilot rule 3 in both arms and one of
these holds; then both judges run at low for every golden judgment:
1. the core doesn't fit at medium but does at low;
2. run 2 fits at low but not at medium;
3. run 2 fits at neither, and the closed-book control fits at low but not at
   medium (the developer's choice, 2026-11-21).
If the core fits at neither effort, the stage stops before any golden judge
call, and the developer decides as a logged deviation.

**After the core wave** the rule runs again at the effort already chosen, on
the calls not yet sent, with unit costs from the ledger's settled usage. What
was sent is committed spend, so it is never counted twice. Items already
started stay; run 2 may be added only whole, its reference-check re-judges
inside its own ceiling check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from raglab import params

EFFORTS = ("medium", "low")
CORE = "core"
if params.PROJECTION["statistic"] != "mean":  # the unit costs below are means (pilot rule 5)
    raise ImportError("the scope rule projects with mean unit costs")


@dataclass(frozen=True)
class Calls:
    """Some judge calls of one kind ("F" or "C") at one effort: how many, and their counted input tokens."""

    judge: str
    effort: str
    n: int
    input_tokens: int


@dataclass(frozen=True)
class Part:
    """A piece of an item, judged only if `requires` (another item) is judged too, or always if it is None."""

    calls: Mapping[str, Sequence[Calls]]  # effort -> its calls
    requires: str | None = None


@dataclass(frozen=True)
class Item:
    name: str
    parts: Sequence[Part]


@dataclass(frozen=True)
class Prices:
    """Batch rates per token for the judge model, mean output tokens per call kind, and the US multiplier."""

    input_rate: Decimal
    output_rate: Decimal
    mean_output: Mapping[tuple[str, str], Decimal]  # (judge, effort) -> mean output tokens per call
    multiplier: Decimal = Decimal(1)

    def cost(self, calls: Sequence[Calls]) -> Decimal:
        return sum(((c.input_tokens * self.input_rate + c.n * self.mean_output[(c.judge, c.effort)] * self.output_rate)
                    * self.multiplier for c in calls), Decimal(0))


@dataclass
class Decision:
    effort: str | None
    judged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    projected: Decimal = Decimal(0)
    stop: bool = False
    log: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"effort": self.effort, "judged": self.judged, "skipped": self.skipped,
                "projected": f"{self.projected:.2f}", "stop": self.stop, "log": self.log}


def _item_cost(item: Item, effort: str, judged: set[str], prices: Prices, items: Sequence[Item]) -> Decimal:
    """An item's expected cost if it joins `judged`: its own parts whose requirement is met, plus the parts of
    items already judged that were waiting for it (run 2's reference-check re-judges, say)."""
    own = sum((prices.cost(p.calls[effort]) for p in item.parts if p.requires is None or p.requires in judged),
              Decimal(0))
    waiting = sum((prices.cost(p.calls[effort]) for other in items if other.name in judged
                   for p in other.parts if p.requires == item.name), Decimal(0))
    return own + waiting


def schedule(items: Sequence[Item], effort: str, base: Decimal, prices: Prices,
             started: Sequence[str] = ()) -> Decision:
    """Items in order (the core first) at one effort, each judged while the projection stays under the ceiling.

    `base` is committed spend + the timing run (if not yet run) + the reserve.
    At freeze point 4 nothing has started and `items` hold every call. For a
    re-projection, `items` hold only the calls not yet sent (what was sent is in
    committed spend, so in `base`) and `started` names the items under way: they
    are judged from the start and never dropped. A started item's unsent part
    whose requirement is already judged is charged at once; one waiting on an
    item not yet judged is charged when that item is admitted, inside its
    ceiling check.
    """
    factor = Decimal(str(params.PROJECTION["factor"]))
    ceiling = params.PLANNING_CEILING
    started = set(started)
    unknown = started - {i.name for i in items}
    if unknown:
        raise ValueError(f"started items that aren't in the plan: {sorted(unknown)}")
    d = Decision(effort, projected=base)
    judged: set[str] = set(started)
    for item in items:
        if item.name in started:
            pending = sum((prices.cost(p.calls[effort]) for p in item.parts if p.requires is None
                           or p.requires in judged), Decimal(0))
            d.projected += pending * factor
    for item in items:
        if item.name in started:
            d.judged.append(item.name)
            continue
        cost = _item_cost(item, effort, judged, prices, items) * factor
        if (item.name == CORE and not started) or d.projected + cost <= ceiling:
            d.projected += cost
            d.judged.append(item.name)
            judged.add(item.name)
        else:
            d.skipped.append(item.name)
    return d


def decide(items: Sequence[Item], base: Decimal, prices: Prices, low_passed: bool) -> Decision:
    """The scope decision at freeze point 4: the effort, then the items judged at it."""
    if not items or items[0].name != CORE:
        raise ValueError("the core comes first")
    ceiling = params.PLANNING_CEILING
    medium, low = schedule(items, "medium", base, prices), schedule(items, "low", base, prices)
    core_med = schedule(items[:1], "medium", base, prices).projected
    core_low = schedule(items[:1], "low", base, prices).projected
    if core_med > ceiling:
        if low_passed and core_low <= ceiling:
            low.log.append(f"the core projects {core_med:.2f} at medium, over {ceiling}; it fits at low")
            return low
        return Decision(None, projected=min(core_med, core_low), stop=True,
                        log=[f"the core projects {core_med:.2f} at medium and {core_low:.2f} at low, over {ceiling}: "
                             "stop before any golden judge call; the developer decides (a logged deviation)"])
    if low_passed and "link1-run2" in low.judged and "link1-run2" not in medium.judged:
        low.log.append("run 2 fits at low but not at medium")
        return low
    if (low_passed and "link1-run2" not in low.judged and "link1-run2" not in medium.judged
            and "closed-book" in low.judged and "closed-book" not in medium.judged):
        low.log.append("run 2 fits at neither effort; low buys the closed-book control")
        return low
    medium.log.append("medium effort")
    return medium


def reproject(remaining: Sequence[Item], effort: str, base: Decimal, prices: Prices,
              started: Sequence[str]) -> Decision:
    """After the core wave: the same order and ceiling at the chosen effort. `remaining` holds each item's calls not
    yet sent (`judge plan` builds it from the judge list and the ledger); `base` holds everything committed, so
    nothing already sent is counted twice. What has started is kept."""
    d = schedule(remaining, effort, base, prices, started=started)
    d.log.append(f"re-projected after the core wave at {effort}; started: {', '.join(started)}")
    return d


def base_total(committed: Decimal, timing_expected: Decimal | None) -> Decimal:
    """What the projection starts from: committed spend, the timing run while it hasn't run, and the reserve."""
    return committed + (timing_expected or Decimal(0)) + params.RESERVE
