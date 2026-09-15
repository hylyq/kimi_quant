"""Regression tests for RiskManager validation.

Covers the fixes:
  - opposite-direction entries require an explicit CLOSE (flip)
  - take profit is mandatory for LONG/SHORT (closes the R:R bypass)
  - existing checks: SL required / distance, R:R, margin, circuit breaker
"""

import pytest

from kimi_quant.llm import TradingSignal
from kimi_quant.risk import RiskManager


@pytest.fixture
def risk() -> RiskManager:
    return RiskManager()


def _sig(
    actions: list[str],
    confidence: float = 0.8,
    size: float = 0.01,
    entry: float = 100000.0,
    sl: float | None = 99000.0,
    tp: float | None = 102000.0,
    **kw,
) -> TradingSignal:
    return TradingSignal(
        actions=actions,
        confidence=confidence,
        reasoning="test",
        size=size,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        **kw,
    )


MID = 100000.0


# ─── Direction: flips must include CLOSE ──────────────────────────────────


def test_shorts_while_long_rejected_without_close(risk):
    check = risk.validate_sequence(
        _sig(["SHORT"]), current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert not check.passed
    assert "CLOSE" in check.reason


def test_longs_while_short_rejected_without_close(risk):
    check = risk.validate_sequence(
        _sig(["LONG"]), current_position_size=0.01,
        current_position_side="short", mid_price=MID,
    )
    assert not check.passed
    assert "CLOSE" in check.reason


def test_flip_with_explicit_close_passes(risk):
    check = risk.validate_sequence(
        _sig(["CLOSE", "SHORT"], sl=101000.0, tp=98000.0),
        current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert check.passed


def test_same_side_still_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"]), current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert not check.passed


def test_close_without_position_rejected(risk):
    check = risk.validate_sequence(
        _sig(["CLOSE"]), current_position_size=0.0,
        current_position_side="none", mid_price=MID,
    )
    assert not check.passed


# ─── Take profit mandatory (R:R bypass) ───────────────────────────────────


def test_long_without_tp_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], tp=None), mid_price=MID,
    )
    assert not check.passed
    assert "Take profit" in check.reason


def test_short_without_tp_rejected(risk):
    check = risk.validate_sequence(
        _sig(["SHORT"], tp=None), mid_price=MID,
    )
    assert not check.passed


def test_close_does_not_require_tp(risk):
    check = risk.validate_sequence(
        _sig(["CLOSE"], tp=None), current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert check.passed


# ─── Stop loss ────────────────────────────────────────────────────────────


def test_long_without_sl_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], sl=None), mid_price=MID,
    )
    assert not check.passed
    assert "Stop loss" in check.reason


def test_sl_too_tight_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], sl=99990.0),  # 0.01% < 0.5% minimum
        mid_price=MID,
    )
    assert not check.passed
    assert "too tight" in check.reason


# ─── R:R ratio ────────────────────────────────────────────────────────────


def test_rr_below_minimum_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], sl=99000.0, tp=100500.0),  # R:R = 0.5
        mid_price=MID,
    )
    assert not check.passed
    assert "1.5" in check.reason


def test_rr_above_minimum_passes(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], sl=99000.0, tp=103000.0),  # R:R = 3.0
        mid_price=MID,
    )
    assert check.passed


# ─── Confidence & circuit breaker ─────────────────────────────────────────


def test_low_confidence_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], confidence=0.5), mid_price=MID,
    )
    assert not check.passed
    assert "Confidence" in check.reason


def test_circuit_breaker_blocks_new_positions(risk):
    risk.cooldown_remaining = 3
    risk.consecutive_losses = 4
    check = risk.validate_sequence(_sig(["LONG"]), mid_price=MID)
    assert not check.passed
    assert "Circuit breaker" in check.reason


def test_circuit_breaker_allows_risk_reducing_actions(risk):
    risk.cooldown_remaining = 3
    risk.consecutive_losses = 4
    check = risk.validate_sequence(
        _sig(["CLOSE"]), current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert check.passed


def test_cooldown_ticks_down_and_resets(risk):
    risk.cooldown_remaining = 2
    risk.tick_cooldown()
    assert risk.cooldown_remaining == 1
    assert risk.is_blocked()
    risk.tick_cooldown()
    assert risk.cooldown_remaining == 0
    assert not risk.is_blocked()
    assert risk.consecutive_losses == 0


# ─── Daily drawdown (resets at UTC midnight) ──────────────────────────────


def test_daily_drawdown_blocks_new_positions(risk):
    risk.initial_balance = 1000.0
    risk.record_loss(-60.0)  # -6% of $1000 in one trade
    check = risk.validate_sequence(_sig(["LONG"]), mid_price=MID)
    assert not check.passed
    assert "daily drawdown" in check.reason.lower()


def test_daily_drawdown_resets_on_date_change(risk, monkeypatch):
    import kimi_quant.risk as risk_mod

    risk.initial_balance = 1000.0
    risk.record_loss(-60.0)
    assert risk._daily_pnl == pytest.approx(-60.0)

    # Simulate UTC midnight rollover: the counter must reset so the cap
    # applies to the NEW day only.
    day_after = risk_mod.datetime.now(risk_mod.timezone.utc).strftime("%Y-%m-%d")
    monkeypatch.setattr(
        risk_mod, "datetime",
        type("FrozenDT", (), {
            "now": staticmethod(lambda tz: type("DT", (), {
                "strftime": lambda self, fmt: "2999-01-01" if fmt == "%Y-%m-%d" else day_after,
            })()),
        }),
    )
    risk._roll_daily()
    assert risk._daily_pnl == 0.0
    check = risk.validate_sequence(_sig(["LONG"]), mid_price=MID)
    assert check.passed


def test_seed_daily_pnl_counts_only_today(risk, tmp_path):
    from kimi_quant.analytics import TradeLogger

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger = TradeLogger(log_path=str(tmp_path / "t.jsonl"))
    logger.open_trade(side="long", size=0.01, entry_price=100000.0)
    logger.close_trade(exit_price=99000.0, reason="stop_loss")  # -$10 today
    logger.open_trade(side="long", size=0.01, entry_price=100000.0)
    old = logger.close_trade(exit_price=101000.0, reason="take_profit")
    old.closed_at = "2000-01-01T00:00:00+00:00"  # pretend it closed long ago

    risk.seed_daily_pnl(logger.get_all_trades())
    assert risk._daily_pnl == pytest.approx(logger.get_all_trades()[0].net_pnl)
    assert risk._daily_date == today


# ─── Margin / risk budget ─────────────────────────────────────────────────


def test_margin_exceeded_rejected(risk):
    # size=0.01 (within max) at $100k = $1000 notional → margin $333 at 3x,
    # which exceeds 95% of a $300 balance.
    check = risk.validate_sequence(
        _sig(["LONG"], size=0.01),
        mid_price=MID,
        account_balance=300.0,
    )
    assert not check.passed
    assert "Margin" in check.reason


def test_risk_amount_over_2pct_rejected(risk):
    # risk = |100000 - 90000| * 0.01 = $100 = 10% of $1000 balance
    check = risk.validate_sequence(
        _sig(["LONG"], size=0.01, sl=90000.0, tp=110000.0),
        mid_price=MID,
        account_balance=1000.0,
    )
    assert not check.passed
    assert "2%" in check.reason


# ─── Multi-action sequence simulation ─────────────────────────────────────


def test_sequence_simulates_state_transitions(risk):
    """['CLOSE','SHORT']: CLOSE validated vs long, SHORT vs simulated flat."""
    check = risk.validate_sequence(
        _sig(["CLOSE", "SHORT"], sl=101000.0, tp=98000.0),
        current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert check.passed


def test_hold_always_passes(risk):
    check = risk.validate_sequence(_sig(["HOLD"], confidence=0.0))
    assert check.passed


# ─── SL/TP side validation (wrong-side orders would trigger instantly) ────


def test_wrong_side_sl_rejected_for_long(risk):
    """LONG with SL ABOVE entry would trigger the moment it is placed."""
    check = risk.validate_sequence(_sig(["LONG"], sl=100500.0), mid_price=MID)
    assert not check.passed
    assert "ABOVE" in check.reason


def test_wrong_side_sl_rejected_for_short(risk):
    check = risk.validate_sequence(_sig(["SHORT"], sl=99500.0, tp=98000.0),
                                   mid_price=MID)
    assert not check.passed
    assert "BELOW" in check.reason


def test_wrong_side_tp_rejected_for_long(risk):
    """LONG with TP below entry is a guaranteed-loss structure."""
    check = risk.validate_sequence(_sig(["LONG"], tp=99500.0), mid_price=MID)
    assert not check.passed
    assert "TP must be above entry" in check.reason


def test_wrong_side_tp_rejected_for_short(risk):
    check = risk.validate_sequence(
        _sig(["SHORT"], sl=101000.0, tp=100500.0), mid_price=MID,
    )
    assert not check.passed
    assert "TP must be below entry" in check.reason


# ─── MODIFY_SL side guard ─────────────────────────────────────────────────


def test_modify_sl_wrong_side_rejected(risk):
    sig = TradingSignal(action="MODIFY_SL", confidence=0.5, reasoning="r",
                        modify_sl_to=100500.0)
    check = risk.validate(sig, 0.01, "long", mid_price=MID)
    assert not check.passed
    assert "trigger instantly" in check.reason


def test_modify_sl_correct_side_passes(risk):
    sig = TradingSignal(action="MODIFY_SL", confidence=0.5, reasoning="r",
                        modify_sl_to=99500.0)
    assert risk.validate(sig, 0.01, "long", mid_price=MID).passed


# ─── CANCEL_STALE is risk-neutral ─────────────────────────────────────────


def test_cancel_stale_allowed_during_cooldown(risk):
    risk.cooldown_remaining = 3
    risk.consecutive_losses = 4
    sig = TradingSignal(action="CANCEL_STALE", confidence=0.1, reasoning="r")
    assert risk.validate(sig, 0.0, "none").passed


# ─── Close discipline (anti-churn) ────────────────────────────────────────

from datetime import datetime, timedelta, timezone

from kimi_quant.risk import PositionContext


def _ctx(minutes_ago: float = 0.0, entry: float = 100000.0, sl: float = 99000.0):
    """PositionContext for a LONG opened `minutes_ago`."""
    return PositionContext(
        side="long",
        entry_price=entry,
        sl_price=sl,
        entry_time=(
            datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        ).isoformat(),
    )


def test_early_close_blocked(risk):
    """CLOSE one cycle after entry with no price movement → rejected."""
    check = risk.validate_sequence(
        _sig(["CLOSE"], confidence=0.8),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100000.0,
        position_ctx=_ctx(minutes_ago=10),  # held 10min, price unchanged
    )
    assert not check.passed
    assert "close discipline" in check.reason.lower()


def test_close_allowed_after_min_hold(risk):
    check = risk.validate_sequence(
        _sig(["CLOSE"], confidence=0.8),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100000.0,
        position_ctx=_ctx(minutes_ago=45),  # > 30min
    )
    assert check.passed


def test_close_allowed_after_real_move(risk):
    # SL distance = 1000 (1%); price moved 600 = 60% of SL distance
    check = risk.validate_sequence(
        _sig(["CLOSE"], confidence=0.8),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100600.0,
        position_ctx=_ctx(minutes_ago=10),
    )
    assert check.passed


def test_close_allowed_with_override_confidence(risk):
    check = risk.validate_sequence(
        _sig(["CLOSE"], confidence=0.93),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100000.0,
        position_ctx=_ctx(minutes_ago=10),
    )
    assert check.passed


def test_close_fails_open_without_context(risk):
    """Recovered position (no entry_time) must never trap capital."""
    check = risk.validate_sequence(
        _sig(["CLOSE"], confidence=0.8),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100000.0,
        position_ctx=PositionContext(
            side="long", entry_price=100000.0, sl_price=99000.0, entry_time=""
        ),
    )
    assert check.passed


def test_flip_with_early_close_blocked(risk):
    """The CLOSE inside ["CLOSE", "SHORT"] is subject to the same gate."""
    check = risk.validate_sequence(
        _sig(["CLOSE", "SHORT"], confidence=0.8, sl=101000.0, tp=98500.0),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100000.0,
        position_ctx=_ctx(minutes_ago=10),
    )
    assert not check.passed
    assert "close discipline" in check.reason.lower()


def test_sl_tp_exits_unaffected(risk):
    """HOLD/MODIFY_SL while position is young are never blocked."""
    check = risk.validate_sequence(
        _sig(["MODIFY_SL"], modify_sl_to=99500.0),
        current_position_size=0.01,
        current_position_side="long",
        mid_price=100200.0,
        position_ctx=_ctx(minutes_ago=5),
    )
    assert check.passed


# ─── SL distance band (max) ───────────────────────────────────────────────


def test_sl_too_wide_rejected(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], sl=97000.0, tp=106000.0),  # SL 3% away, R:R 2:1
        mid_price=MID,
    )
    assert not check.passed
    assert "too wide" in check.reason


# ─── Expected value gate ──────────────────────────────────────────────────


def test_ev_gate_rejects_confidence_below_breakeven(risk):
    """R:R 1:1 → breakeven 50%; confidence 45% is negative-EV."""
    risk.min_confidence = 0.4
    risk.MIN_RR_RATIO = 1.0
    check = risk.validate_sequence(
        _sig(["LONG"], confidence=0.45, sl=99000.0, tp=101000.0),
        mid_price=MID,
    )
    assert not check.passed
    assert "expected value" in check.reason.lower()


def test_ev_gate_passes_when_confidence_exceeds_breakeven(risk):
    risk.min_confidence = 0.4
    risk.MIN_RR_RATIO = 1.0
    check = risk.validate_sequence(
        _sig(["LONG"], confidence=0.6, sl=99000.0, tp=101000.0),
        mid_price=MID,
    )
    assert check.passed


def test_tp_fee_floor_rejects_penny_targets(risk):
    """A TP closer than 3× round-trip fees can never pay for itself."""
    risk.MIN_RR_RATIO = 0.5
    risk.MIN_SL_DISTANCE = 0.001
    check = risk.validate_sequence(
        _sig(["LONG"], sl=99900.0, tp=100100.0),  # reward 0.1% < 0.27%
        mid_price=MID,
    )
    assert not check.passed
    assert "cannot pay" in check.reason


# ─── Trend alignment gate ─────────────────────────────────────────────────


def test_counter_trend_short_rejected(risk):
    check = risk.validate_sequence(
        _sig(["SHORT"], confidence=0.75, sl=101000.0, tp=98500.0),
        mid_price=MID,
        trend_1h="up", trend_4h="up",
    )
    assert not check.passed
    assert "Counter-trend" in check.reason


def test_counter_trend_short_allowed_with_high_confidence(risk):
    check = risk.validate_sequence(
        _sig(["SHORT"], confidence=0.85, sl=101000.0, tp=98500.0),
        mid_price=MID,
        trend_1h="up", trend_4h="up",
    )
    assert check.passed


def test_with_trend_entry_ignores_gate(risk):
    check = risk.validate_sequence(
        _sig(["LONG"], confidence=0.75),
        mid_price=MID,
        trend_1h="up", trend_4h="up",
    )
    assert check.passed


def test_sideways_trend_is_neutral(risk):
    check = risk.validate_sequence(
        _sig(["SHORT"], confidence=0.75, sl=101000.0, tp=98500.0),
        mid_price=MID,
        trend_1h="sideways", trend_4h="up",
    )
    assert check.passed


# ─── Size normalization ───────────────────────────────────────────────────


def test_clamp_size_caps_by_risk_budget(risk):
    # balance $100, leverage 3 (conftest): margin cap = 0.95*100*3/100000
    # = 0.00285; risk cap with 1% SL = 100*0.01/1000 = 0.001 → 0.001 wins
    sig = _sig(["LONG"], size=0.01, sl=99000.0)
    assert risk.clamp_size(sig, 100000.0, 100.0) == pytest.approx(0.001)


def test_clamp_size_respects_hard_position_cap(risk):
    sig = _sig(["LONG"], size=0.05, sl=99000.0)  # above max (0.01, conftest)
    assert risk.clamp_size(sig, 100000.0, None) == pytest.approx(0.01)


def test_clamp_size_defaults_when_size_missing(risk):
    sig = _sig(["LONG"], size=None, sl=99000.0)
    assert risk.clamp_size(sig, 100000.0, None) == pytest.approx(0.01)
