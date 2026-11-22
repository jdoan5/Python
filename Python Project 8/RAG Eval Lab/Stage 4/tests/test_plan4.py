"""Request lists: unique ids, counted tokens, bounds, and bodies checked against their plan."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from raglab.ledger import Pricing
from raglab.plan4 import as_request, count_all, line_for, read_plan, total_bound, write_plan

STAGE = Path(__file__).resolve().parents[1]
BODY = {"model": "claude-haiku-5-5", "max_tokens": 16000, "messages": [{"role": "user", "content": "q"}]}


def test_identical_bodies_in_two_conditions_get_two_lines_and_one_count(tmp_path: Path) -> None:
    pricing = Pricing.load(STAGE / "pricing.json")
    a, b = line_for("hhr-2.4k-med", "q022", 0, BODY), line_for("oracle-4.8k-med", "q022", 0, BODY)
    assert a.custom_id != b.custom_id and a.body_sha256 == b.body_sha256
    calls = []
    lines = count_all([a, b], {a.custom_id: BODY, b.custom_id: BODY}, lambda body: calls.append(1) or 1000, pricing)
    assert len(calls) == 1
    assert lines[0].input_tokens == 1114  # 1000 * 1.05 + 64
    # (1114 * 0.10 + 16000 * 0.50) / 1e6, halved for a batch
    assert Decimal(lines[0].bound) == (1114 * Decimal("0.10") + 16000 * Decimal("0.50")) / 10**6 / 2 * Decimal("1.1")
    assert total_bound(lines) == 2 * Decimal(lines[0].bound)
    write_plan(tmp_path / "requests.jsonl", lines)
    assert read_plan(tmp_path / "requests.jsonl") == lines
    with pytest.raises(ValueError, match="share a custom_id"):
        write_plan(tmp_path / "dupes.jsonl", [lines[0], lines[0]])


def test_a_body_that_changed_since_planning_is_refused() -> None:
    pricing = Pricing.load(STAGE / "pricing.json")
    planned = line_for("hhr-4.8k-med", "q001", 1, BODY)
    (counted_line,) = count_all([planned], {planned.custom_id: BODY}, lambda body: 10, pricing)
    req = as_request(counted_line, BODY)
    assert req.custom_id == planned.custom_id and req.sample == 1
    with pytest.raises(ValueError, match="differs from the planned one"):
        as_request(counted_line, {**BODY, "max_tokens": 1})
    with pytest.raises(ValueError, match="never counted"):
        as_request(planned, BODY)
