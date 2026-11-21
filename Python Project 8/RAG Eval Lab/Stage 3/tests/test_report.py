from __future__ import annotations

from raglab.report import _delta, style_table


def test_comparisons_are_bold_only_when_significant_after_holm():
    row = {"delta": 0.1, "lo": 0.02, "hi": 0.18, "p": 0.01}
    assert not _delta({**row, "p_holm": 0.2}).startswith("**")   # raw p alone is not enough
    assert _delta({**row, "p_holm": 0.03}).startswith("**")
    assert _delta(row | {"p_holm": 0.03}) == "**+0.10 (+0.02 to +0.18)**"


def test_switch_effects_are_bold_only_when_the_interval_excludes_zero():
    assert _delta({"delta": 0.03, "lo": 0.001, "hi": 0.06}, None).startswith("**")
    assert not _delta({"delta": 0.03, "lo": -0.001, "hi": 0.06}, None).startswith("**")
    assert _delta({"delta": -0.05, "lo": -0.09, "hi": -0.01}, None).startswith("**")


def test_style_table_shows_intervals():
    cell = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    summary = {"by_style": {"keyword": {"n": 3, "configs": {"c": {"recall@5": cell}}}}}
    assert "0.50 (0.40–0.60)" in style_table(summary, ["c"])


def test_against_the_baseline_bolds_on_holm_at_the_call_site():
    from raglab.report import vs_baseline_table

    row = {"delta": 0.1, "lo": 0.01, "hi": 0.2, "p": 0.01, "p_holm": 0.08}  # significant raw, not after Holm
    summary = {"baseline": "b", "configs": ["b", "c"], "vs_baseline": {"c": {"recall@5": row}}}
    table = vs_baseline_table(summary, ("recall@5",))
    assert "| c | +0.10 (+0.01 to +0.20) |" in table


def test_tables_list_configurations_in_design_order_never_by_score():
    from raglab.report import results_table

    cell = lambda m: {"mean": m, "lo": m, "hi": m}  # noqa: E731
    summary = {"baseline": "a", "configs": ["a", "b", "c"],
               "summary": {"a": {"recall@5": cell(0.1)}, "b": {"recall@5": cell(0.9)}, "c": {"recall@5": cell(0.5)}}}
    rows = [line.split(" | ")[0] for line in results_table(summary, ("recall@5",)).splitlines()[2:]]
    assert rows == ["| **a**", "| b", "| c"]  # the order given, although b scores highest


def test_the_chart_plots_success_first_and_mean_coverage_second():
    from raglab.report import budget_chart_svg

    cell = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    metrics = [f"{m}@{b}" for m in ("budget_success", "coverage") for b in (1200, 2400)]
    summary = {"configs": ["x", "y"], "summary": {n: {m: dict(cell, mean=0.1 + 0.1 * i) for i, m in enumerate(metrics)}
                                                  for n in ("x", "y")}}
    svg = budget_chart_svg(summary, ["x"], (1200, 2400), "t", "s")
    assert svg.index("An answering section arrives (yes/no)") < svg.index("How much of the answering text arrives")
    assert "x 0.10/0.20" in svg  # the first panel's description quotes budget_success, not recall
