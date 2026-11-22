"""Stage 4's fixed settings, in one place. PLAN.md's machine-readable block must equal these (tests/test_plan.py)."""

from __future__ import annotations

from decimal import Decimal

GENERATOR = "claude-haiku-5-5"
JUDGE = "claude-opus-5-5"

CONDITIONS = ("hhr-4.8k-med", "base-4.8k-med", "bm25-4.8k-med", "closed-med", "masked-4.8k-med",
              "hhr-4.8k-off", "hhr-4.8k-low", "hhr-4.8k-high", "hhr-9.6k-med", "hhr-2.4k-med",
              "base-4.8k-med-notitle", "hybrid-4.8k-med", "oracle-4.8k-med", "hhr-4.8k-xhigh", "hhr-4.8k-max")
RUNS = {"hhr-4.8k-med": 3, "base-4.8k-med": 2}  # every other condition: one run

PRIMARY_METRIC = "supported_correct"
CHAIN = (("base-4.8k-med", "hhr-4.8k-med", 1), ("bm25-4.8k-med", "hhr-4.8k-med", 1))
SUPPORTING_METRICS = ("correct", "claim_support_rate", "penalized")

ABSTENTION_SENTENCE = "The provided documentation does not answer this question."
REFERENCE_CAP_CHARS = 20_000
KEYS = {"min_nuggets": 2, "max_nuggets": 5, "min_essential": 1, "max_essential": 3}
QUOTE_MAX_CHARS = 300
REASON_MAX_WORDS = 15
CODE_GATE_MIN_CHARS = 40
CITATION_FAILURE_STOP = 0.005

INSTRUMENT_CHECK = {"kappa": 0.70, "swaps": 20, "swaps_gate": 18, "swaps_gate_per_arm": 8, "plants": 20,
                    "plants_gate": 18, "plants_gate_per_kind": 8}
HAND_GRADED = {"questions": 12, "floor": 8}
PLANTED = {"swaps_per_arm": 10, "lexical_per_arm": 5, "cross_section_plants_per_arm": 5, "shape_plants": 10,
           "candidates_per_kind": 30}
MASKED_JUDGED = 40
LENIENCY_WILSON_LOWER = 0.05

PILOT = {"questions": 12, "swaps": 12, "swaps_per_arm": 6, "lexical_per_arm": 3, "swap_gate": 11,
         "swap_gate_per_arm": 5, "low_effort_agreement": 0.90, "low_effort_sc_agree": 11,
         "format_failures_allowed": 1, "code_citation_share": 0.90, "unit_citation_share": 0.90,
         "unit_citation_gap": 0.10}
MAX_TOKENS_PILOT = {"haiku_batch": 16_000, "haiku_xhigh_max_batch": 32_000, "haiku_standard": 8_000, "judge": 8_000}
MAX_TOKENS_BOUNDS = {"haiku": (4_000, 16_000), "haiku_xhigh_max": (4_000, 32_000), "judge": (3_000, 12_000),
                     "judge_truncation_retry": 16_000}

INPUT_MARGIN = Decimal("0.05")
INPUT_SLACK = 64
BATCH_MAX = {"claude-haiku-5-5": 100, "claude-opus-5-5": 50}
CAP = Decimal("25.00")
PLANNING_CEILING = Decimal("24.50")
RESERVE = Decimal("1.00")  # one re-run per failed judgment, truncation re-runs, overruns: kept in every projection
PHASE_CAPS = {"pilot": Decimal("6.50"), "generation": Decimal("3.00")}
PROJECTION = {"statistic": "mean", "factor": 1.15}
PRIORITY = ("link1-run2", "closed-book", "reference-check", "self-agreement", "link2", "masked",
            "base-4.8k-med-notitle", "hhr-4.8k-off", "hhr-4.8k-low", "hhr-4.8k-high", "hhr-9.6k-med", "hhr-2.4k-med",
            "hybrid-4.8k-med", "oracle-4.8k-med")


def as_plan_block() -> dict:
    """These settings as PLAN.md's JSON block spells them."""
    from raglab import metrics, stats
    from raglab.golden import default_golden_path, load_golden, questions_fingerprint
    from raglab.llm import PINNED_SDK
    from raglab.store import API_VERSION

    return {
        "questions_fingerprint": questions_fingerprint(load_golden(default_golden_path(2)).questions),
        "generator": GENERATOR, "judge": JUDGE, "api_version": API_VERSION, "sdk": PINNED_SDK,
        "headline_budget": metrics.HEADLINE_BUDGET,
        "conditions": list(CONDITIONS), "runs": dict(RUNS),
        "primary_metric": PRIMARY_METRIC, "chain": [list(c) for c in CHAIN],
        "supporting_metrics": list(SUPPORTING_METRICS),
        "alpha": 0.05, "resamples": stats.RESAMPLES, "seed": stats.SEED, "sign_flip_seed": stats.SEED + 1,
        "abstention_sentence": ABSTENTION_SENTENCE, "reference_cap_chars": REFERENCE_CAP_CHARS, "keys": dict(KEYS),
        "quote_max_chars": QUOTE_MAX_CHARS, "reason_max_words": REASON_MAX_WORDS,
        "code_gate_min_chars": CODE_GATE_MIN_CHARS, "citation_failure_stop": CITATION_FAILURE_STOP,
        "instrument_check": dict(INSTRUMENT_CHECK), "hand_graded": dict(HAND_GRADED), "planted": dict(PLANTED),
        "masked_judged": MASKED_JUDGED, "leniency_wilson_lower": LENIENCY_WILSON_LOWER, "pilot": dict(PILOT),
        "max_tokens_pilot": dict(MAX_TOKENS_PILOT),
        "max_tokens_bounds": {k: list(v) if isinstance(v, tuple) else v for k, v in MAX_TOKENS_BOUNDS.items()},
        "input_margin": float(INPUT_MARGIN), "input_slack": INPUT_SLACK, "batch_max": dict(BATCH_MAX),
        "cap": f"{CAP:.2f}", "planning_ceiling": f"{PLANNING_CEILING:.2f}", "reserve": f"{RESERVE:.2f}",
        "phase_caps": {**{k: f"{v:.2f}" for k, v in PHASE_CAPS.items()}, "judging": "remainder"},
        "projection": dict(PROJECTION),
        "priority": list(PRIORITY),
    }
