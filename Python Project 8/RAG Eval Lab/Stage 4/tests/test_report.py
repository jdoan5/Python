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


# --- Stage 3's cost chart and cost table (added after freeze point 2) --------------------------------


def _cost_inputs(names, metric="mrr@10"):
    cell = lambda m: {"mean": m, "lo": m - 0.05, "hi": m + 0.05}  # noqa: E731
    summary = {"summary": {n: {metric: cell(0.5 + 0.01 * i)} for i, n in enumerate(names)}}
    cost = {"configurations": {n: {"p50_ms": 0.05 * 10 ** i, "p95_ms": 0.1 * 10 ** i, "mean_ms": 0.05 * 10 ** i,
                                   "cpu_seconds_per_1000_questions": 0.05 * 10 ** i,
                                   "steps_mean_ms": {"reranking": 7.0 if "rerank" in n else 0.0}}
                               for i, n in enumerate(names)}}
    return summary, cost


def test_the_cost_chart_draws_every_pipeline_in_design_order_with_its_values():
    from raglab.report import COST_CHART_ROWS, quality_cost_svg

    names = [n for group in COST_CHART_ROWS for n, _, _ in group]
    summary, cost = _cost_inputs(list(reversed(names)))  # scores and timings given in the opposite order
    svg = quality_cost_svg(summary, cost, "a <title>", "s")
    labels = [label for group in COST_CHART_ROWS for _, label, _ in group]
    assert [svg.index(f">{label}</text>") for label in labels] == sorted(svg.index(f">{label}</text>") for label in labels)
    assert "a &lt;title&gt;" in svg and "<title>" not in svg.replace('<title id="t">', "")
    desc = svg[svg.index('<desc id="d">'):svg.index("</desc>")]
    assert all(label in desc for label in labels) and "95% interval" in desc
    assert svg.count("prefers-color-scheme: dark") == 1


def test_the_cost_chart_skips_pipelines_missing_from_either_input_and_refuses_an_empty_chart():
    import pytest

    from raglab.report import quality_cost_svg

    summary, cost = _cost_inputs(["heading+bm25", "heading+hybrid"])
    del cost["configurations"]["heading+hybrid"]
    svg = quality_cost_svg(summary, cost, "t", "s")
    assert ">BM25</text>" in svg and "hybrid (BM25 + dense)" not in svg
    with pytest.raises(ValueError):
        quality_cost_svg(summary, {"configurations": {}}, "t", "s")


def test_the_cost_table_lists_the_pipelines_in_design_order_and_marks_no_reranking():
    from raglab.report import cost_table

    summary, cost = _cost_inputs(["heading+hybrid+rerank", "heading+bm25"])
    rows = cost_table(cost).splitlines()[2:]
    assert [r.split(" | ")[0] for r in rows] == ["| heading+bm25", "| heading+hybrid+rerank"]
    assert rows[0].split(" | ")[3] == "–" and rows[1].split(" | ")[3] == "7.0"
    assert rows[1].split(" | ")[1] == "0.05"  # under 0.1 ms keeps two decimals


def test_the_label_check_verdict_says_fragile_when_its_blind_test_is():
    from raglab.report import confirmatory_table

    t = {"from": "a", "to": "b", "metric": "mrr@10", "delta": 0.09, "lo": 0.02, "hi": 0.16, "p": 0.01,
         "p_holm": 0.01, "page_level": {"p": 0.02, "p_holm": 0.02}, "resolvable": 0.1, "verdict": "supported",
         "fragile": False}
    summary = {"confirmatory": {"chain": [t], "supporting": [], "success@5_changes":
                                {"fixed": 1, "broken": 0, "both_found": 1, "both_missed": 1}},
               "label_check": {"metric": "mrr@10", "on_blind_labels": dict(t, p=0.04, fragile=True),
                               "kappa_answer_vs_not": 0.9, "kappa_items": 10, "verdict": "supported",
                               "stage4_inherits": "b"}}
    assert "Verdict: **supported**, its blind-round confirmation fragile per page." in confirmatory_table(summary)
    summary["label_check"]["on_blind_labels"]["fragile"] = False
    assert "Verdict: **supported**. Stage 4" in confirmatory_table(summary)
