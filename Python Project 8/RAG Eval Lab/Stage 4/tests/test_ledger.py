"""The cost guard: reservations, caps, settlement, crashes, duplicates. No network, no key."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from raglab.ledger import (
    CapExceeded, Caps, Ledger, LedgerError, Planned, Pricing, fold, input_bound, money, read_events,
)

STAGE = Path(__file__).resolve().parents[1]
HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
AUTH = "a" * 64  # stands for the SHA-256 of an authorization record
US = Decimal("1.1")  # every bound assumes US-only inference


@pytest.fixture
def pricing() -> Pricing:
    return Pricing.load(STAGE / "pricing.json")


@pytest.fixture
def caps() -> Caps:
    return Caps(total=Decimal("1.00"), phases={"pilot": Decimal("0.60"), "main": Decimal("0.90")})


def planned(custom_id: str, model: str = OPUS, input_tokens: int = 1000, max_tokens: int = 1000,
            cache: bool = False) -> Planned:
    return Planned(custom_id, model, input_tokens, max_tokens, cache)


ONE = (1000 * Decimal(4) + 1000 * Decimal(20)) / 10**6 * US  # one standard Opus call at 1,000 in and 1,000 out


def test_the_price_table_matches_the_list_prices(pricing: Pricing) -> None:
    haiku, opus = pricing.rates(HAIKU), pricing.rates(OPUS)
    assert haiku.input * 1_000_000 == Decimal("0.10") and haiku.output * 1_000_000 == Decimal("0.50")
    assert opus.input * 1_000_000 == Decimal("4") and opus.output * 1_000_000 == Decimal("20")
    assert opus.cache_read * 1_000_000 == Decimal("0.20")  # 0.05x on Opus 5.5
    assert haiku.cache_read * 1_000_000 == Decimal("0.01")  # 0.1x elsewhere
    for r in (haiku, opus):
        assert r.cache_write_5m == r.input * Decimal("1.25") and r.cache_write_1h == r.input * 2
    assert pricing.batch_multiplier == Decimal("0.5") and pricing.us_inference_multiplier == US


def test_a_bound_is_every_input_token_at_the_dearest_rate_plus_max_tokens_of_output(pricing: Pricing) -> None:
    # 1,000 in at $4/M and 2,000 out at $20/M = $0.044, times 1.1 in case the workspace defaults to US-only
    assert pricing.bound(OPUS, 1000, 2000, False, "standard") == Decimal("0.044") * US
    assert pricing.bound(OPUS, 1000, 2000, True, "standard") == Decimal("0.045") * US  # a cache marker: 1.25x input
    assert pricing.bound(OPUS, 1000, 2000, False, "batch") == Decimal("0.022") * US


def test_a_prompt_on_haikus_second_rate_card_is_refused(pricing: Pricing) -> None:
    pricing.bound(HAIKU, 100_000, 10, False, "batch")
    with pytest.raises(LedgerError, match="second rate card"):
        pricing.bound(HAIKU, 100_001, 10, False, "batch")


def test_an_unknown_model_or_mode_is_refused(pricing: Pricing) -> None:
    with pytest.raises(LedgerError, match="no price"):
        pricing.bound("claude-sonnet-5-5", 10, 10, False, "batch")
    with pytest.raises(LedgerError, match="mode"):
        pricing.bound(OPUS, 10, 10, False, "priority")


def test_usage_is_priced_field_by_field(pricing: Pricing) -> None:
    usage = {"input_tokens": 100, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 2000,
             "output_tokens": 500, "inference_geo": "global", "service_tier": "batch",
             "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0}}
    cost, flags = pricing.usage_cost(OPUS, usage, "standard")
    assert cost == Decimal(100 * 4 + 1000 * 5 + 2000 * 0.2 + 500 * 20) / 1_000_000 and flags == []
    batch, _ = pricing.usage_cost(OPUS, usage, "batch")
    assert batch == cost / 2


def test_us_only_inference_costs_more_and_is_flagged(pricing: Pricing) -> None:
    base = {"input_tokens": 1000, "output_tokens": 1000}
    global_cost, _ = pricing.usage_cost(OPUS, base | {"inference_geo": "global"}, "batch")
    us_cost, flags = pricing.usage_cost(OPUS, base | {"inference_geo": "us"}, "batch")
    assert us_cost == global_cost * US and any("US-only" in f for f in flags)


def test_anything_that_changes_the_price_is_flagged(pricing: Pricing) -> None:
    for usage, word in (({"service_tier": "priority"}, "service_tier"), ({"speed": "fast"}, "speed"),
                        ({"iterations": [{}, {}]}, "iterations"), ({"web_search_requests": 3}, "unpriced"),
                        ({"cache_creation_input_tokens": 5, "cache_creation": {"ephemeral_1h_input_tokens": 5}},
                         "1-hour")):
        _, flags = pricing.usage_cost(HAIKU, {"input_tokens": 1, "output_tokens": 1} | usage, "batch")
        assert any(word in f for f in flags), (usage, flags)


def test_the_input_bound_adds_the_margin() -> None:
    assert input_bound(1000, Decimal("0.05"), 64) == 1114
    assert input_bound(0, Decimal("0.05"), 64) == 64
    with pytest.raises(LedgerError):
        input_bound(-1, Decimal("0.05"), 64)


def test_a_reservation_must_name_its_authorization(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        for missing in (None, ""):
            with pytest.raises(LedgerError, match="authorization"):
                ledger.reserve("pilot", "standard", [planned("x")], missing)
        ledger.reserve("pilot", "standard", [planned("x")], AUTH)
    assert read_events(tmp_path / "ledger.jsonl")[0][0]["authorization"] == AUTH


def test_reservations_are_refused_past_either_cap(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        n = int(caps.phases["pilot"] / ONE)
        for i in range(n):
            ledger.reserve("pilot", "standard", [planned(f"p{i}")], AUTH)
        with pytest.raises(CapExceeded, match="pilot"):
            ledger.reserve("pilot", "standard", [planned("one-too-many")], AUTH)
        # the pilot's cap is spent, but the stage cap still has room for another phase, up to its own limit
        m = int((caps.total - ledger.state.committed()) / ONE)
        for i in range(m):
            ledger.reserve("main", "standard", [planned(f"m{i}")], AUTH)
        with pytest.raises(CapExceeded):
            ledger.reserve("main", "standard", [planned("over-the-total")], AUTH)
        assert ledger.state.committed() <= caps.total


def test_a_batch_is_reserved_whole_or_not_at_all(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        per_item = ONE / 2
        fits = int(caps.phases["pilot"] / per_item)
        with pytest.raises(CapExceeded):
            ledger.reserve("pilot", "batch", [planned(f"b{i}") for i in range(fits + 1)], AUTH)
        assert ledger.state.committed() == 0 and not list(ledger.events())
        group = ledger.reserve("pilot", "batch", [planned(f"b{i}") for i in range(fits)], AUTH)
        assert ledger.state.committed("pilot") == fits * per_item and group == "pilot-00001"


def test_one_open_attempt_per_request_across_groups(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        first = ledger.reserve("pilot", "batch", [planned("x"), planned("y")], AUTH)
        with pytest.raises(LedgerError, match="already has an attempt"):
            ledger.reserve("pilot", "batch", [planned("y"), planned("z")], AUTH)
        ledger.mark_sent(first, batch_id="msgbatch_1")
        with pytest.raises(LedgerError, match="already has an attempt that is sent"):
            ledger.reserve("pilot", "standard", [planned("x")], AUTH)
        ledger.settle(first, "x", "succeeded", OPUS, {"input_tokens": 1, "output_tokens": 1})
        ledger.settle(first, "y", "expired", None, None)
        ledger.reserve("pilot", "standard", [planned("y")], AUTH)  # an expired item may be asked again


def test_an_answered_request_is_never_paid_for_twice(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    """Even if its response went missing from the store: the ledger alone refuses a second attempt."""
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        group = ledger.reserve("pilot", "standard", [planned("done")], AUTH)
        ledger.mark_sent(group)
        ledger.settle(group, "done", "succeeded", OPUS, {"input_tokens": 1, "output_tokens": 1})
        with pytest.raises(LedgerError, match="already answered"):
            ledger.reserve("pilot", "batch", [planned("other"), planned("done")], AUTH)
        assert "other" not in {c for (_, c) in ledger.state.items}  # refused whole: nothing was reserved
        failed = ledger.reserve("pilot", "standard", [planned("bad")], AUTH)
        ledger.mark_sent(failed)
        ledger.settle(failed, "bad", "errored", None, None)
        ledger.reserve("pilot", "standard", [planned("bad")], AUTH)  # an error was never answered: one resubmission


def test_an_unknown_attempt_blocks_until_an_explicit_resend_and_stays_spent(tmp_path: Path, pricing: Pricing,
                                                                           caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        group = ledger.reserve("pilot", "standard", [planned("lost")], AUTH)
        ledger.mark_sent(group)
        ledger.mark_unknown(group, "lost", "connection dropped after sending")
        assert ledger.state.committed() == ONE
        with pytest.raises(LedgerError, match="unknown"):
            ledger.reserve("pilot", "standard", [planned("lost")], AUTH)  # a second process can't resend it either
        ledger.resend(group, "lost", "the developer checked the Console: it never ran")
        again = ledger.reserve("pilot", "standard", [planned("lost")], AUTH)
        assert ledger.state.committed() == 2 * ONE  # the unknown attempt still counts: it may have been billed
        with pytest.raises(LedgerError, match="already cleared"):
            ledger.resend(group, "lost", "twice")
        assert again != group


def test_settling_replaces_the_bound_with_the_cost(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        group = ledger.reserve("pilot", "batch", [planned("a"), planned("b"), planned("c"), planned("d")], AUTH)
        ledger.mark_sent(group, batch_id="msgbatch_x")
        cost = ledger.settle(group, "a", "succeeded", OPUS, {"input_tokens": 1000, "output_tokens": 500})
        assert cost == Decimal("0.007")  # (1000*4 + 500*20)/1e6, halved; no US surcharge reported
        for cid, outcome in (("b", "errored"), ("c", "canceled"), ("d", "expired")):
            assert ledger.settle(group, cid, outcome, None, None) == 0
        assert ledger.state.committed() == Decimal("0.007") == ledger.state.settled("pilot")


def test_only_an_unsent_reservation_can_be_released(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        unsent = ledger.reserve("pilot", "batch", [planned("u")], AUTH)
        ledger.release(unsent, "batch create never reached the API (reconcile found no batch)")
        assert ledger.state.committed() == 0
        sent = ledger.reserve("pilot", "batch", [planned("s")], AUTH)
        ledger.mark_sent(sent, batch_id="msgbatch_y")
        with pytest.raises(LedgerError, match="was sent"):
            ledger.release(sent, "no")
        with pytest.raises(LedgerError, match="already sent"):
            ledger.mark_sent(sent, batch_id="msgbatch_z")


def test_a_request_must_be_sent_before_it_is_settled(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        group = ledger.reserve("pilot", "standard", [planned("x")], AUTH)
        with pytest.raises(LedgerError, match="only a sent request"):
            ledger.settle(group, "x", "succeeded", OPUS, {"input_tokens": 1, "output_tokens": 1})
        ledger.mark_sent(group, request_id="req_1")
        ledger.settle(group, "x", "succeeded", OPUS, {"input_tokens": 1, "output_tokens": 1})
        with pytest.raises(LedgerError, match="only a sent request"):
            ledger.settle(group, "x", "succeeded", OPUS, {"input_tokens": 1, "output_tokens": 1})


def test_a_cost_above_its_bound_or_from_another_model_is_flagged(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with Ledger(tmp_path / "ledger.jsonl", pricing, caps) as ledger:
        group = ledger.reserve("pilot", "standard", [planned("x", max_tokens=10, input_tokens=10)], AUTH)
        ledger.mark_sent(group)
        ledger.settle(group, "x", "succeeded", HAIKU, {"input_tokens": 500_000, "output_tokens": 10})
        flags = ledger.summary()["flags"]
        assert any("served by claude-haiku-5-5" in f for f in flags)
        assert any("exceeded its bound" in f for f in flags)


def test_the_ledger_survives_a_restart_and_a_torn_last_line(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    path = tmp_path / "ledger.jsonl"
    with Ledger(path, pricing, caps) as ledger:
        group = ledger.reserve("pilot", "batch", [planned("a"), planned("b")], AUTH)
        ledger.mark_sent(group, batch_id="msgbatch_x")
        ledger.settle(group, "a", "succeeded", OPUS, {"input_tokens": 1000, "output_tokens": 500})
    with path.open("a", encoding="utf-8") as f:  # a crash halfway through writing the next event
        f.write('{"event": "settle", "group": "pilot-00001", "custom_id": "b", "outco')
    events, torn = read_events(path)
    assert len(events) == 3 and torn
    with pytest.raises(LedgerError, match="torn"):  # never repaired silently
        Ledger(path, pricing, caps).__enter__()
    with Ledger(path, pricing, caps, repair=True) as ledger:
        assert ledger.repairs and "torn" in ledger.repairs[0]
        assert ledger.state.items[(group, "b")].status == "sent"  # still open, still counted at its bound
        assert ledger.state.committed() == Decimal("0.007") + ONE / 2
        ledger.settle(group, "b", "expired", None, None)
    assert path.read_bytes().endswith(b"\n")
    assert fold(read_events(path)[0]).committed() == Decimal("0.007")


def test_corruption_before_the_last_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"event": "reserve"\n{"event": "sent"}\n', encoding="utf-8")
    with pytest.raises(LedgerError, match="line 1"):
        read_events(path)


def test_one_process_at_a_time(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    path = tmp_path / "ledger.jsonl"
    with Ledger(path, pricing, caps):
        with pytest.raises(LedgerError, match="another process"):
            Ledger(path, pricing, caps).__enter__()
    with Ledger(path, pricing, caps):  # released on exit
        pass


def test_writing_needs_the_lock(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    with pytest.raises(LedgerError, match="open the ledger"):
        Ledger(tmp_path / "ledger.jsonl", pricing, caps).reserve("pilot", "standard", [planned("x")], AUTH)


def test_amounts_are_stored_as_exact_decimals(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    path = tmp_path / "ledger.jsonl"
    with Ledger(path, pricing, caps) as ledger:
        ledger.reserve("pilot", "batch", [planned("a", model=HAIKU, input_tokens=3, max_tokens=7)], AUTH)
    (event,) = read_events(path)[0]
    assert event["items"][0]["bound"] == "0.00000209"  # (3*0.10 + 7*0.50)/1e6 / 2 * 1.1, no float, no exponent
    assert money(Decimal("1E-7")) == "0.0000001"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["phase"] == "pilot"


def test_every_event_is_mirrored_and_a_lost_copy_is_never_silent(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    path, mirror = tmp_path / "data" / "ledger.jsonl", tmp_path / "elsewhere" / "ledger.jsonl"
    with Ledger(path, pricing, caps, mirror=mirror) as ledger:
        group = ledger.reserve("pilot", "standard", [planned("a")], AUTH)
        ledger.mark_sent(group)
    assert path.read_bytes() == mirror.read_bytes()
    path.unlink()  # data/ was lost: the spend must not reset to $0
    with pytest.raises(LedgerError, match="missing"):
        Ledger(path, pricing, caps, mirror=mirror).__enter__()
    with Ledger(path, pricing, caps, mirror=mirror, repair=True) as ledger:
        assert ledger.state.committed() == ONE
    assert path.read_bytes() == mirror.read_bytes()


def test_losing_both_copies_is_refused_once_responses_exist(tmp_path: Path, pricing: Pricing, caps: Caps) -> None:
    path, mirror = tmp_path / "ledger.jsonl", tmp_path / "mirror.jsonl"
    with pytest.raises(LedgerError, match="spend record is missing"):
        Ledger(path, pricing, caps, mirror=mirror, expect_history=True).__enter__()
    with Ledger(path, pricing, caps, mirror=mirror, expect_history=False):  # a fresh start is fine
        pass


def test_a_crash_between_the_two_writes_heals_and_anything_else_is_refused(tmp_path: Path, pricing: Pricing,
                                                                          caps: Caps) -> None:
    path, mirror = tmp_path / "ledger.jsonl", tmp_path / "mirror.jsonl"
    with Ledger(path, pricing, caps, mirror=mirror) as ledger:
        ledger.reserve("pilot", "standard", [planned("a")], AUTH)
        ledger.reserve("pilot", "standard", [planned("b")], AUTH)
    lines = mirror.read_bytes().splitlines(keepends=True)
    mirror.write_bytes(b"".join(lines[:1]))  # the mirror missed the last event
    with Ledger(path, pricing, caps, mirror=mirror) as ledger:
        assert ledger.repairs == [f"copied the last event to {mirror}"]
    assert path.read_bytes() == mirror.read_bytes()
    mirror.write_bytes(b"")  # two events behind: not a crash between writes
    with pytest.raises(LedgerError, match="disagree"):
        Ledger(path, pricing, caps, mirror=mirror).__enter__()
