"""The scope rule (raglab/scope.py) on the plan's own budget: it must reproduce the plan's branches."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

from raglab import params, scope

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "analysis"))

import budget  # noqa: E402
from raglab.ledger import Pricing  # noqa: E402


def case(n: int):
    book = budget.Book(Pricing.load(STAGE / "pricing.json"), n)
    fixed = sum(c for _, c in budget.fixed_items(book))
    return budget.items(book), fixed, budget.prices(book)


def test_central_costs_judge_run_2_the_closed_book_control_and_the_reference_check_at_medium() -> None:
    items, fixed, prices = case(0)
    d = scope.decide(items, fixed, prices, low_passed=True)
    assert d.effort == "medium" and d.judged == ["core", "link1-run2", "closed-book", "reference-check"]
    assert f"{d.projected:.2f}" == "23.96" and "self-agreement" in d.skipped


def test_pessimistic_costs_switch_to_low_when_that_buys_the_closed_book_control() -> None:
    items, fixed, prices = case(1)
    d = scope.decide(items, fixed, prices, low_passed=True)
    assert d.effort == "low" and d.judged == ["core", "closed-book", "reference-check", "self-agreement", "masked"]
    assert f"{d.projected:.2f}" == "22.13"
    stay = scope.decide(items, fixed, prices, low_passed=False)  # low failed rule 3: medium, run 1 only
    assert stay.effort == "medium" and stay.judged == ["core", "reference-check"] and f"{stay.projected:.2f}" == "23.63"


def test_the_reference_check_is_sized_by_what_else_is_judged() -> None:
    items, fixed, prices = case(1)
    ref = next(i for i in items if i.name == "reference-check")
    alone = scope._item_cost(ref, "medium", {"core"}, prices, items)
    full = scope._item_cost(ref, "medium", {"core", "link1-run2", "closed-book"}, prices, items)
    assert alone < full and full == pytest.approx(alone * Decimal(5) / Decimal(2), rel=Decimal("0.001"))
    # Run 2 added after the core wave brings its reference-check re-judges with it.
    run2 = next(i for i in items if i.name == "link1-run2")
    with_ref = scope._item_cost(run2, "medium", {"core", "reference-check"}, prices, items)
    assert with_ref > scope._item_cost(run2, "medium", {"core"}, prices, items)


def test_the_core_that_fits_at_neither_effort_stops_the_stage() -> None:
    items, _, prices = case(1)
    d = scope.decide(items, Decimal("20.00"), prices, low_passed=True)
    assert d.stop and d.effort is None and "developer decides" in d.log[0]
    low = scope.decide(items, Decimal("14.00"), prices, low_passed=True)
    assert low.effort == "low" and "core" in low.judged and "fits at low" in low.log[0]


def _without_sent(items, sent: dict[str, set[int]]):
    """What `judge plan` hands the re-projection: each item's parts minus those already sent."""
    return [scope.Item(i.name, [p for n, p in enumerate(i.parts) if n not in sent.get(i.name, set())]) for i in items]


def test_a_core_that_settles_at_its_expected_cost_keeps_run_2() -> None:
    items, fixed, prices = case(0)
    core = next(i for i in items if i.name == "core")
    core_cost = scope._item_cost(core, "medium", set(), prices, items)
    remaining = _without_sent(items, {"core": {0, 1}})
    d = scope.reproject(remaining, "medium", fixed + core_cost, prices, started=["core"])
    assert d.judged[:2] == ["core", "link1-run2"] and "closed-book" in d.judged and "reference-check" in d.judged
    # The core's 1.15 factor is released once its cost is known, so self-agreement now fits too.
    assert "self-agreement" in d.judged and f"{d.projected:.2f}" == "23.79"


def test_run_2_brings_its_reference_check_re_judges_inside_its_own_ceiling_check() -> None:
    items, fixed, prices = case(0)
    ref = next(i for i in items if i.name == "reference-check")
    run2 = next(i for i in items if i.name == "link1-run2")
    # The core, the closed-book control and the reference check's run-1 and closed-book re-judges were sent;
    # the reference check's run-2 part waits for run 2.
    remaining = _without_sent(items, {"core": {0, 1}, "closed-book": {0}, "reference-check": {0, 2}})
    started = ["core", "reference-check", "closed-book"]
    alone = scope._item_cost(run2, "medium", set(), prices, items)
    waiting = prices.cost(ref.parts[1].calls["medium"])
    factor = Decimal("1.15")
    base = scope.params.PLANNING_CEILING - (alone + waiting) * factor + Decimal("0.01")  # just too little room
    d = scope.reproject(remaining, "medium", base, prices, started=started)
    assert "link1-run2" in d.skipped  # without its re-judges it would have fit
    roomy = scope.reproject(remaining, "medium", base - Decimal("0.02"), prices, started=started)
    assert "link1-run2" in roomy.judged
    assert roomy.projected <= scope.params.PLANNING_CEILING  # never past the ceiling


def test_a_started_item_is_never_dropped_and_must_be_in_the_plan() -> None:
    items, fixed, prices = case(0)
    remaining = _without_sent(items, {"core": {0, 1}, "closed-book": {0}})
    d = scope.reproject(remaining, "medium", Decimal("24.40"), prices, started=["core", "closed-book"])
    assert "closed-book" in d.judged and "link1-run2" in d.skipped
    with pytest.raises(ValueError, match="aren't in the plan"):
        scope.reproject(remaining, "medium", fixed, prices, started=["core", "masked-everything"])


def test_the_projection_starts_from_committed_spend_the_timing_run_and_the_reserve() -> None:
    assert scope.base_total(Decimal("10.00"), Decimal("0.16")) == Decimal("11.16")
    assert scope.base_total(Decimal("10.00"), None) == Decimal("10.00") + params.RESERVE
    us = scope.Prices(Decimal("0.0000025"), Decimal("0.0000125"), {("F", "medium"): Decimal(1000)}, Decimal("1.1"))
    plain = scope.Prices(Decimal("0.0000025"), Decimal("0.0000125"), {("F", "medium"): Decimal(1000)})
    calls = [scope.Calls("F", "medium", 10, 24000)]
    assert us.cost(calls) == plain.cost(calls) * Decimal("1.1")
