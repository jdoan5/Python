"""README-ready tables, generated from a saved run so no number is ever hand-copied.

`python -m raglab eval report <run> --markdown` prints the README's tables:
the confirmatory tests, then the heading-chunk pipelines, the baseline and the
control. `--full` prints every table for every configuration (TABLES.md).
Configurations are always listed in design order, never by score, so no row
is singled out as the best. Each value is a mean with its 95% interval:
percentile bootstrap for scores, Wilson for yes/no rates. Bold in an
exploratory table describes (Holm-adjusted p below 0.05 against the baseline;
an interval that excludes zero for a switch); it is not a claim. Only the
confirmatory table's verdicts are claims.
"""

from __future__ import annotations

from typing import Mapping

from raglab.experiments import CHALLENGER, CONTROL, INCUMBENT
from raglab.metrics import HEADLINE_BUDGET, HEADLINE_METRICS, LABELS

ALPHA = 0.05
# The README's two table families: how well sections are ranked, and what a fixed-size prompt receives.
RANK_TABLE = ("success@5", "recall@5", "mrr@10", "ndcg@10", "ndcg_answers@10")
BUDGET_TABLE = tuple(f"{m}@{HEADLINE_BUDGET}" for m in ("budget_success", "budget_recall", "coverage", "best_coverage"))
# The README core: Stage 1's baseline, every heading-chunk pipeline with the breadcrumb, and the control.
README_CONFIGS = ("baseline (RAG Mini Chat)", "heading+tfidf", "heading+bm25", "heading+dense", "heading+hybrid",
                  "heading+hybrid+rerank", CONTROL.name)
SWITCH_LABELS = {
    ("retriever", "tfidf", "bm25"): "BM25 instead of TF-IDF",
    ("retriever", "bm25", "dense"): "dense instead of BM25",
    ("retriever", "bm25", "hybrid"): "hybrid instead of BM25",
    ("retriever", "dense", "hybrid"): "hybrid instead of dense",
    ("retriever", "hybrid", "hybrid+rerank"): "reranking the hybrid",
    ("retriever", "bm25", "bm25+rerank"): "reranking BM25",
    ("retriever", "bm25+rerank", "hybrid+rerank"): "hybrid instead of BM25, both reranked",
    ("breadcrumb", False, True): "breadcrumb on",
    ("chunker", "paragraph", "heading"): "heading chunks instead of paragraphs",
    ("chunker", "paragraph", "fixed"): "fixed windows instead of paragraphs",
    ("chunker", "fixed", "heading"): "heading chunks instead of fixed windows",
}


def _ci(cell: Mapping) -> str:
    return f"{cell['mean']:.2f} ({cell['lo']:.2f}–{cell['hi']:.2f})"


def _delta(row: Mapping, p_key: str | None = "p_holm") -> str:
    text = f"{row['delta']:+.2f} ({row['lo']:+.2f} to {row['hi']:+.2f})"
    if p_key is None:  # effects: judged by whether the interval excludes zero
        significant = row["lo"] > 0 or row["hi"] < 0
    else:
        significant = row[p_key] < ALPHA
    return f"**{text}**" if significant else text


def _header(first: str, metrics: tuple[str, ...]) -> list[str]:
    return [
        "| " + " | ".join([first, *(LABELS[m] for m in metrics)]) + " |",
        "|" + "---|" + "---:|" * len(metrics),
    ]


def _in_order(summary: Mapping, names) -> list[str]:
    return [n for n in summary["configs"] if names is None or n in names]


def results_table(summary: Mapping, metrics: tuple[str, ...] = HEADLINE_METRICS, configs=None) -> str:
    """The configurations in design order; the baseline in bold."""
    order = _in_order(summary, configs)
    lines = _header("configuration", metrics)
    for name in order:
        label = f"**{name}**" if name == summary["baseline"] else name
        lines.append("| " + " | ".join([label, *(_ci(summary["summary"][name][m]) for m in metrics)]) + " |")
    return "\n".join(lines)


def vs_baseline_table(summary: Mapping, metrics: tuple[str, ...] = HEADLINE_METRICS, configs=None) -> str:
    order = [n for n in _in_order(summary, configs) if n in summary["vs_baseline"]]
    lines = _header(f"vs {summary['baseline']}", metrics)
    for name in order:
        cells = [_delta(summary["vs_baseline"][name][m]) for m in metrics]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)


def effects_table(summary: Mapping, metrics: tuple[str, ...] = HEADLINE_METRICS) -> str:
    """Average effect of flipping one switch, over every setting of the other two."""
    rows: dict[tuple, dict] = {}
    for e in summary["effects"]:
        key = (e["factor"], e["from"], e["to"], e.get("within"))
        rows.setdefault(key, {"pairs": len(e["pairs"])})[e["metric"]] = e
    lines = _header("switch (pairs compared)", metrics)
    for (factor, a, b, within), by_metric in rows.items():
        name = SWITCH_LABELS.get((factor, a, b), f"{factor}: {a} -> {b}")
        label = f"{name}{f', {within} retrievers' if within else ''} ({by_metric['pairs']})"
        lines.append("| " + " | ".join([label, *(_delta(by_metric[m], None) for m in metrics)]) + " |")
    return "\n".join(lines)


def style_table(summary: Mapping, configs: list[str], metric: str = "recall@5") -> str:
    styles = list(summary["by_style"])
    lines = [
        "| " + " | ".join([f"{LABELS[metric]} by question style", *(f"{s} (n={summary['by_style'][s]['n']})" for s in styles)]) + " |",
        "|---|" + "---:|" * len(styles),
    ]
    for name in configs:
        cells = [_ci(summary["by_style"][s]["configs"][name][metric]) for s in styles]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)


def diagnostics_table(summary: Mapping, budget: int = HEADLINE_BUDGET, configs=None) -> str:
    lines = [
        f"| configuration | top-10 slots that are lone headings | chars in top 5 chunks | sections in {budget / 1000:g}k chars "
        f"| top-10 sections judged | sections in {budget / 1000:g}k chars judged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in _in_order(summary, configs):
        d = summary["diagnostics"][name]
        lines.append(
            f"| {name} | {d['empty_slots@10']:.0%} | {d['top5_chars']:,.0f} | {d[f'sections_in_budget@{budget}']:.1f} "
            f"| {d['judged@10']:.0%} | {d[f'judged_in_budget@{budget}']:.0%} |"
        )
    return "\n".join(lines)


def _test_row(label: str, t: Mapping, holm: bool = False) -> str:
    p = t["p_holm"] if holm else t["p"]
    page = t["page_level"]["p_holm"] if holm else t["page_level"]["p"]
    verdict = t["verdict"] + (" (fragile per page)" if t.get("fragile") else "")
    return (f"| {label} | {t['delta']:+.3f} ({t['lo']:+.3f} to {t['hi']:+.3f}) | {p:.4f} | {page:.4f} "
            f"| ±{t['resolvable']:.3f} | {verdict} |")


def confirmatory_table(summary: Mapping) -> str:
    """The chain, the supporting metrics and the label check, with their verdicts: the only claims."""
    c = summary["confirmatory"]
    if c is None:
        return "_The confirmatory tests need every chain configuration; this run lacks one._"
    head = ["| test | difference (95% CI) | p | p per page | resolvable | verdict |", "|---|---:|---:|---:|---:|---|"]
    lines = list(head)
    for i, t in enumerate(c["chain"], 1):
        lines.append(_test_row(f"link {i}: {t['to']} vs {t['from']}, {LABELS[t['metric']]}", t))
    lines += ["", "Supporting metrics, headline pair (Holm-adjusted p):", "", *head]
    for t in c["supporting"]:
        lines.append(_test_row(LABELS[t["metric"]], t, holm=True))
    changes = c["success@5_changes"]
    lines += ["", f"Success@5: {changes['fixed']} questions fixed, {changes['broken']} broken, "
                  f"{changes['both_found']} found by both, {changes['both_missed']} missed by both."]
    check = summary.get("label_check")
    if check:
        lines += ["", "Label check, link 1 on the blind round's grades:", "", *head,
                  _test_row(LABELS[check["metric"]], check["on_blind_labels"]), "",
                  f"Answer-versus-not agreement with the new labels: κ = {check['kappa_answer_vs_not']:.2f} "
                  f"over {check['kappa_items']} labels. Verdict: **{check['verdict']}**"
                  # PLAN.md: a fragile Supported is called fragile wherever it is stated.
                  f"{', its blind-round confirmation fragile per page' if check['on_blind_labels'].get('fragile') else ''}. "
                  f"Stage 4 inherits {check['stage4_inherits']}."]
    return "\n".join(lines)


def markdown_report(summary: Mapping, manifest: Mapping, full: bool = False) -> str:
    """The README's tables, or with `full` every table for every configuration."""
    code, corpus, golden = manifest["code"], manifest["corpus"], manifest["golden"]
    configs = None if full else [n for n in README_CONFIGS if n in summary["configs"]]
    styles = (list(summary["configs"]) if full else
              [n for n in (summary["baseline"], INCUMBENT.name, CHALLENGER.name) if n in summary["configs"]])
    parts = [
        f"Run `{manifest['run_id']}` · {summary['questions']} questions · golden v{golden['version']} "
        f"`{golden['sha256'][:12]}` · FastAPI {corpus['ref']} `{corpus['fingerprint'][:12]}` · "
        f"code `{code['package_sha256'][:12]}` · {manifest['statistics']['bootstrap_resamples']:,} resamples",
        "", "### The confirmatory tests", "", confirmatory_table(summary),
        "", "### Ranking quality", "", results_table(summary, RANK_TABLE, configs),
        "", "### What a 4,800-character prompt receives", "", results_table(summary, BUDGET_TABLE, configs),
        "", "### Against the baseline: ranking", "", vs_baseline_table(summary, RANK_TABLE, configs),
        "", "### Against the baseline: the prompt", "", vs_baseline_table(summary, BUDGET_TABLE, configs),
    ]
    if full:
        parts += ["", "### What each switch is worth: ranking", "", effects_table(summary, RANK_TABLE),
                  "", "### What each switch is worth: the prompt", "", effects_table(summary, BUDGET_TABLE)]
    parts += ["", "### By question style", "", style_table(summary, styles, "success@5"),
              "", "### Diagnostics", "", diagnostics_table(summary, configs=configs)]
    return "\n".join(parts) + "\n"


# --- the README chart: success and coverage as the context budget grows ---------------------------------

# Reference palette (validated: scripts/validate_palette.js, 3 slots, both modes).
_THEME = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "context": "#cfcdc6", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "context": "#4a4a47", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70"},
}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def budget_chart_svg(
    summary: Mapping,
    highlight: list[str],
    budgets: tuple[int, ...],
    title: str,
    subtitle: str,
    panels: tuple[tuple[str, str], ...] = (
        ("budget_success", "An answering section arrives (yes/no)"),
        ("coverage", "How much of the answering text arrives (mean)"),
    ),
) -> str:
    """Small multiples, one panel per metric, against context budget.

    Up to three configurations are highlighted (legend plus end values); every
    other configuration is drawn in gray as context, so the highlighted ones are
    read against the whole field rather than in isolation.
    """
    if not 1 <= len(highlight) <= 3:
        raise ValueError("highlight one to three configurations; the rest are drawn as context")
    panel_w, gap, left_pad = 330, 168, 64
    width = left_pad + len(panels) * panel_w + (len(panels) - 1) * gap + 96
    height = 450
    top, bottom = 150, 372

    def y(v: float) -> float:
        return bottom - v * (bottom - top)

    def series(name: str, metric: str) -> list[float]:
        return [summary["summary"][name][f"{metric}@{b}"]["mean"] for b in budgets]

    def css(mode: str) -> str:
        return "".join(f"--{k}:{v};" for k, v in _THEME[mode].items())

    desc = "; ".join(
        f"{label}: " + ", ".join(
            f"{n} " + "/".join(f"{v:.2f}" for v in series(n, metric)) for n in highlight
        ) + f" at {', '.join(f'{b:,}' for b in budgets)} characters"
        for metric, label in panels
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-labelledby="t d" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        f'<title id="t">{_esc(title)}</title>',
        f'<desc id="d">{_esc(desc)}</desc>',
        f"<style>svg{{{css('light')}}}@media (prefers-color-scheme: dark){{svg{{{css('dark')}}}}}</style>",
        f'<rect width="{width}" height="{height}" rx="8" fill="var(--surface)"/>',
        f'<text x="24" y="34" font-size="17" font-weight="600" fill="var(--text)">{_esc(title)}</text>',
        f'<text x="24" y="56" font-size="13" fill="var(--text2)">{_esc(subtitle)}</text>',
    ]
    lx = 24.0
    for i, name in enumerate(highlight, start=1):
        out.append(f'<line x1="{lx}" y1="84" x2="{lx + 18}" y2="84" stroke="var(--s{i})" stroke-width="2.5" stroke-linecap="round"/>')
        out.append(f'<text x="{lx + 24}" y="88" font-size="12.5" fill="var(--text2)">{_esc(name)}</text>')
        lx += 24 + 6.7 * len(name) + 20
    out.append(f'<line x1="{lx}" y1="84" x2="{lx + 18}" y2="84" stroke="var(--context)" stroke-width="2" stroke-linecap="round"/>')
    out.append(f'<text x="{lx + 24}" y="88" font-size="12.5" fill="var(--text2)">the other configurations</text>')

    for p, (metric, label) in enumerate(panels):
        x0 = left_pad + p * (panel_w + gap)
        x1 = x0 + panel_w
        xs = [x0 + i * panel_w / (len(budgets) - 1) for i in range(len(budgets))]
        out.append(f'<text x="{x0 - 40}" y="{top - 22}" font-size="13.5" font-weight="600" fill="var(--text)">{_esc(label)}</text>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            stroke = "var(--axis)" if tick == 0 else "var(--grid)"
            out.append(f'<line x1="{x0}" y1="{y(tick):.1f}" x2="{x1}" y2="{y(tick):.1f}" stroke="{stroke}" stroke-width="1"/>')
            out.append(f'<text x="{x0 - 10}" y="{y(tick) + 4:.1f}" font-size="12" text-anchor="end" fill="var(--muted)">{tick:g}</text>')
        for x, b in zip(xs, budgets):
            out.append(f'<text x="{x:.1f}" y="{bottom + 22}" font-size="12" text-anchor="middle" fill="var(--muted)">{b:,}</text>')
        out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{bottom + 44}" font-size="12" text-anchor="middle" '
                   f'fill="var(--muted)">context budget (characters)</text>')
        for name in summary["configs"]:
            if name not in highlight:
                pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, series(name, metric)))
                out.append(f'<polyline points="{pts}" fill="none" stroke="var(--context)" stroke-width="1.5" '
                           f'stroke-linejoin="round" stroke-linecap="round"/>')
        ends = []
        for i, name in enumerate(highlight, start=1):
            vals = series(name, metric)
            pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals))
            out.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{i})" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
            for x, v in zip(xs, vals):
                out.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="4" fill="var(--s{i})" stroke="var(--surface)" stroke-width="2"/>')
            ends.append([y(vals[-1]), vals[-1], i])
        # End values beside the last marker; nudged apart with a leader line when they'd collide.
        ends.sort()
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 16)
        for label_y, value, i in ends:
            line_y = y(value)
            if abs(label_y - line_y) > 1:
                out.append(f'<line x1="{x1 + 7}" y1="{line_y:.1f}" x2="{x1 + 15}" y2="{label_y - 4:.1f}" '
                           f'stroke="var(--muted)" stroke-width="1"/>')
            out.append(f'<line x1="{x1 + 18}" y1="{label_y - 4:.1f}" x2="{x1 + 28}" y2="{label_y - 4:.1f}" '
                       f'stroke="var(--s{i})" stroke-width="2.5" stroke-linecap="round"/>')
            out.append(f'<text x="{x1 + 33}" y="{label_y:.1f}" font-size="12.5" font-weight="600" fill="var(--text)">'
                       f'{value:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Stage 3's chart: ranking quality beside what a question costs (added after freeze point 2) -------

# The heading-chunk pipelines with the breadcrumb, grouped by family: (configuration, label, palette slot).
COST_CHART_ROWS = (
    (("heading+tfidf", "TF-IDF", 2), ("heading+bm25", "BM25", 2)),
    (("heading+dense", "dense", 1), ("heading+hybrid", "hybrid (BM25 + dense)", 1)),
    (("heading+hybrid+rerank [depth 10]", "hybrid, top 10 reranked", 3),
     ("heading+hybrid+rerank [depth 20]", "hybrid, top 20 reranked", 3),
     ("heading+hybrid+rerank [depth 50]", "hybrid, top 50 reranked", 3),
     ("heading+hybrid+rerank", "hybrid, top 100 reranked", 3),
     ("heading+bm25+rerank", "BM25, top 100 reranked", 3)),
)
_FAMILIES = ((2, "lexical"), (1, "embeddings"), (3, "reranked by the cross-encoder"))


def quality_cost_svg(summary: Mapping, cost: Mapping, title: str, subtitle: str, metric: str = "mrr@10") -> str:
    """Two panels sharing one row per pipeline: a metric's mean with its 95% interval, and the median time
    per question on a log scale. One axis per panel; rows in design order, grouped by family, never by score."""
    import math

    groups = [[r for r in g if r[0] in summary["summary"] and r[0] in cost["configurations"]] for g in COST_CHART_ROWS]
    groups = [g for g in groups if g]
    rows = [r for g in groups for r in g]
    if not rows:
        raise ValueError("none of the chart's pipelines is in both the run and the timings")
    cells = {n: summary["summary"][n][metric] for n, _, _ in rows}
    ms = {n: cost["configurations"][n]["p50_ms"] for n, _, _ in rows}
    row_h, gap, top = 28, 14, 150
    height = top + len(rows) * row_h + (len(groups) - 1) * gap + 64
    width = 760
    a0, a1 = 222, 452  # the metric panel
    b0, b1 = 512, 686  # the time panel
    v_lo = math.floor(min(c["lo"] for c in cells.values()) * 10) / 10
    v_hi = math.ceil(max(c["hi"] for c in cells.values()) * 10) / 10
    t_lo = 10 ** math.floor(math.log10(max(min(ms.values()), 1e-3)))
    t_hi = 10 ** math.ceil(math.log10(max(ms.values())))

    def xa(v: float) -> float:
        return a0 + (v - v_lo) / (v_hi - v_lo) * (a1 - a0)

    def xb(t: float) -> float:
        return b0 + (math.log10(t) - math.log10(t_lo)) / (math.log10(t_hi) - math.log10(t_lo)) * (b1 - b0)

    def css(mode: str) -> str:
        return "".join(f"--{k}:{v};" for k, v in _THEME[mode].items())

    def fmt_ms(t: float) -> str:
        return f"{t:,.0f} ms" if t >= 10 else f"{t:.1f} ms" if t >= 0.1 else f"{t:.2f} ms"

    desc = "; ".join(f"{label}: {LABELS[metric]} {cells[n]['mean']:.2f} (95% interval {cells[n]['lo']:.2f} to "
                     f"{cells[n]['hi']:.2f}), median {fmt_ms(ms[n])} per question" for n, label, _ in rows)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-labelledby="t d" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        f'<title id="t">{_esc(title)}</title>', f'<desc id="d">{_esc(desc)}</desc>',
        f"<style>svg{{{css('light')}}}@media (prefers-color-scheme: dark){{svg{{{css('dark')}}}}}</style>",
        f'<rect width="{width}" height="{height}" rx="8" fill="var(--surface)"/>',
        f'<text x="24" y="34" font-size="17" font-weight="600" fill="var(--text)">{_esc(title)}</text>',
        f'<text x="24" y="56" font-size="13" fill="var(--text2)">{_esc(subtitle)}</text>',
    ]
    lx = 24.0
    for slot, label in _FAMILIES:
        out.append(f'<circle cx="{lx + 5}" cy="84" r="5" fill="var(--s{slot})"/>')
        out.append(f'<text x="{lx + 15}" y="88" font-size="12.5" fill="var(--text2)">{_esc(label)}</text>')
        lx += 15 + 6.7 * len(label) + 22
    bottom = height - 64
    out.append(f'<text x="{a0}" y="{top - 22}" font-size="13.5" font-weight="600" fill="var(--text)">'
               f'{_esc(LABELS[metric])}, mean and 95% interval</text>')
    out.append(f'<text x="{b0}" y="{top - 22}" font-size="13.5" font-weight="600" fill="var(--text)">'
               'Median time per question</text>')
    steps = round((v_hi - v_lo) / 0.1)
    for i in range(steps + 1):
        v = v_lo + i * 0.1
        out.append(f'<line x1="{xa(v):.1f}" y1="{top - 8}" x2="{xa(v):.1f}" y2="{bottom}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{xa(v):.1f}" y="{bottom + 20}" font-size="12" text-anchor="middle" fill="var(--muted)">{v:.1f}</text>')
    t = t_lo
    while t <= t_hi * 1.0001:
        out.append(f'<line x1="{xb(t):.1f}" y1="{top - 8}" x2="{xb(t):.1f}" y2="{bottom}" stroke="var(--grid)" stroke-width="1"/>')
        label = f"{t:,.0f}" if t >= 1 else f"{t:g}"
        out.append(f'<text x="{xb(t):.1f}" y="{bottom + 20}" font-size="12" text-anchor="middle" fill="var(--muted)">{label}</text>')
        t *= 10
    out.append(f'<text x="{(b0 + b1) / 2:.1f}" y="{bottom + 40}" font-size="12" text-anchor="middle" fill="var(--muted)">'
               'milliseconds, log scale, caches off</text>')
    if "heading+bm25" in cells:  # the incumbent, as a reference for every row
        ref = xa(cells["heading+bm25"]["mean"])
        out.append(f'<line x1="{ref:.1f}" y1="{top - 8}" x2="{ref:.1f}" y2="{bottom}" stroke="var(--axis)" '
                   'stroke-width="1.5" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{ref:.1f}" y="{bottom + 40}" font-size="12" text-anchor="middle" fill="var(--muted)">'
                   f'BM25&#8217;s mean</text>')
    y = float(top)
    for g, group in enumerate(groups):
        if g:
            y += gap
        for n, label, slot in group:
            cy = y + row_h / 2
            c = cells[n]
            out.append(f'<text x="24" y="{cy + 4:.1f}" font-size="12.5" fill="var(--text)">{_esc(label)}</text>')
            out.append(f'<line x1="{xa(c["lo"]):.1f}" y1="{cy:.1f}" x2="{xa(c["hi"]):.1f}" y2="{cy:.1f}" '
                       f'stroke="var(--s{slot})" stroke-width="2" stroke-linecap="round"/>')
            out.append(f'<circle cx="{xa(c["mean"]):.1f}" cy="{cy:.1f}" r="5" fill="var(--s{slot})" '
                       'stroke="var(--surface)" stroke-width="2"/>')
            out.append(f'<text x="{xa(c["hi"]) + 8:.1f}" y="{cy + 4:.1f}" font-size="12" paint-order="stroke" stroke="var(--surface)" stroke-width="4" stroke-linejoin="round" fill="var(--text2)">'
                       f'{c["mean"]:.2f}</text>')
            out.append(f'<circle cx="{xb(ms[n]):.1f}" cy="{cy:.1f}" r="5" fill="var(--s{slot})" '
                       'stroke="var(--surface)" stroke-width="2"/>')
            out.append(f'<text x="{xb(ms[n]) + 9:.1f}" y="{cy + 4:.1f}" font-size="12" paint-order="stroke" stroke="var(--surface)" stroke-width="4" stroke-linejoin="round" fill="var(--text2)">'
                       f'{_esc(fmt_ms(ms[n]))}</text>')
            y += row_h
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _amount(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}" if x >= 0.1 else f"{x:.2f}"


def cost_table(cost: Mapping) -> str:
    """What each heading-chunk pipeline costs per question, from `raglab bench` (results/cost.json; added after
    freeze point 2). Measured on one machine, outside the reproducible results; nothing in the evaluation reads it."""
    lines = ["| configuration | median ms per question | 95th percentile ms | reranking, mean ms "
             "| CPU-seconds per 1,000 questions |", "|---|---:|---:|---:|---:|"]
    for group in COST_CHART_ROWS:
        for name, _, _ in group:
            c = cost["configurations"].get(name)
            if c is None:
                continue
            rerank = c["steps_mean_ms"]["reranking"]
            lines.append(f"| {name} | {_amount(c['p50_ms'])} | {_amount(c['p95_ms'])} "
                         f"| {_amount(rerank) if rerank else '–'} | {_amount(c['cpu_seconds_per_1000_questions'])} |")
    return "\n".join(lines)
