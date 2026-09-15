"""RiskManager validation tests — direction, SL/TP side, daily drawdown."""

import pytest

from kimi_quant.llm import TradingSignal
from kimi_quant.risk import RiskManager


def make_signal(**kw) -> TradingSignal:
    defaults = dict(
        action="LONG",
        confidence=0.8,
        reasoning="test",
        size=0.01,
        entry_price=100_000.0,
        stop_loss=99_000.0,
        take_profit=102_000.0,
    )
    defaults.update(kw)
    return TradingSignal(**defaults)


# ─── Direction ────────────────────────────────────────────────────────────


def test_same_direction_rejected():
    risk = RiskManager()
    assert not risk.validate(make_signal(action="LONG"), 0.01, "long").passed
    assert not risk.validate(make_signal(action="SHORT"), 0.01, "short").passed


def test_naked_opposite_entry_rejected():
    """Bare SHORT while long (no CLOSE) must be rejected — it corrupts tracker state."""
    risk = RiskManager()
    check = risk.validate(make_signal(action="SHORT"), 0.01, "long")
    assert not check.passed
    assert "CLOSE" in check.reason

    check = risk.validate(make_signal(action="LONG"), 0.01, "short")
    assert not check.passed
    assert "CLOSE" in check.reason


def test_flip_sequence_passes():
    """["CLOSE", "SHORT"] while long is the sanctioned flip path."""
    risk = RiskManager()
    signal = TradingSignal(
        actions=["CLOSE", "SHORT"],
        confidence=0.8,
        reasoning="flip",
        size=0.01,
        entry_price=100_000.0,
        stop_loss=101_000.0,  # valid for the new SHORT
        take_profit=98_000.0,
    )
    assert risk.validate_sequence(signal, 0.01, "long").passed


def test_flip_sequence_without_close_rejected():
    risk = RiskManager()
    signal = TradingSignal(
        actions=["SHORT"],
        confidence=0.8,
        reasoning="naked flip",
        size=0.01,
        entry_price=100_000.0,
        stop_loss=101_000.0,
        take_profit=98_000.0,
    )
    check = risk.validate_sequence(signal, 0.01, "long")
    assert not check.passed
    assert "CLOSE" in check.reason


def test_close_without_position_rejected():
    risk = RiskManager()
    assert not risk.validate(make_signal(action="CLOSE"), 0.0, "none").passed


# ─── Stop loss side + distance ────────────────────────────────────────────


def test_wrong_side_sl_rejected_for_long():
    """LONG with SL ABOVE entry would trigger instantly — must be caught."""
    risk = RiskManager()
    check = risk.validate(
        make_signal(stop_loss=100_500.0),  # above entry 100k
        0.0, "none", mid_price=100_000.0,
    )
    assert not check.passed
    assert "ABOVE" in check.reason


def test_wrong_side_sl_rejected_for_short():
    risk = RiskManager()
    check = risk.validate(
        make_signal(action="SHORT", stop_loss=99_500.0, take_profit=98_000.0),
        0.0, "none", mid_price=100_000.0,
    )
    assert not check.passed
    assert "BELOW" in check.reason


def test_sl_distance_too_tight_rejected():
    risk = RiskManager()
    check = risk.validate(
        make_signal(stop_loss=99_800.0),  # 0.2% < 0.5% min
        0.0, "none", mid_price=100_000.0,
    )
    assert not check.passed
    assert "too tight" in check.reason


def test_valid_long_passes_all_checks():
    risk = RiskManager()
    check = risk.validate(
        make_signal(),  # SL 1% below, TP 2% above → R:R 2:1
        0.0, "none", mid_price=100_000.0,
    )
    assert check.passed


def test_missing_sl_rejected_for_directional():
    risk = RiskManager()
    check = risk.validate(make_signal(stop_loss=None), 0.0, "none")
    assert not check.passed
    assert "required" in check.reason


# ─── Take profit side + R:R ───────────────────────────────────────────────


def test_wrong_side_tp_rejected_for_long():
    """LONG with TP below entry is a guaranteed-loss structure."""
    risk = RiskManager()
    check = risk.validate(
        make_signal(take_profit=99_500.0),
        0.0, "none", mid_price=100_000.0,
    )
    assert not check.passed
    assert "TP must be above entry" in check.reason


def test_tp_none_skips_rr_check():
    """TP is optional — R:R is not enforced when absent."""
    risk = RiskManager()
    check = risk.validate(make_signal(take_profit=None), 0.0, "none")
    assert check.passed


def test_rr_below_minimum_rejected():
    risk = RiskManager()
    check = risk.validate(
        make_signal(stop_loss=99_000.0, take_profit=100_500.0),  # R:R 0.5
        0.0, "none", mid_price=100_000.0,
    )
    assert not check.passed
    assert "Risk/reward" in check.reason


# ─── MODIFY_SL side validation ────────────────────────────────────────────


def test_modify_sl_wrong_side_rejected():
    risk = RiskManager()
    signal = TradingSignal(action="MODIFY_SL", confidence=0.5, reasoning="r",
                           modify_sl_to=100_500.0)
    check = risk.validate(signal, 0.01, "long", mid_price=100_000.0)
    assert not check.passed
    assert "trigger instantly" in check.reason


def test_modify_sl_correct_side_passes():
    risk = RiskManager()
    signal = TradingSignal(action="MODIFY_SL", confidence=0.5, reasoning="r",
                           modify_sl_to=99_500.0)
    assert risk.validate(signal, 0.01, "long", mid_price=100_000.0).passed


# ─── Confidence + circuit breaker ─────────────────────────────────────────


def test_low_confidence_rejected():
    risk = RiskManager()
    check = risk.validate(make_signal(confidence=0.5), 0.0, "none")
    assert not check.passed


def test_circuit_breaker_blocks_new_positions_only():
    risk = RiskManager()
    risk.cooldown_remaining = 3
    assert not risk.validate(make_signal(), 0.0, "none").passed
    # Risk-reducing actions stay available during cooldown
    for action in ("CLOSE", "MODIFY_SL", "MODIFY_TP", "CANCEL_STALE", "HOLD"):
        sig = TradingSignal(action=action, confidence=0.1, reasoning="r")
        assert risk.validate(sig, 0.01, "long").passed, action


# ─── Daily drawdown ───────────────────────────────────────────────────────


def test_daily_drawdown_blocks_new_positions():
    risk = RiskManager()
    risk.initial_balance = 1_000.0
    risk.total_realized_pnl = -60.0
    risk.day_start_realized_pnl = 0.0  # all losses today
    check = risk.validate(make_signal(), 0.0, "none")
    assert not check.passed
    assert "daily drawdown" in check.reason


def test_daily_baseline_limits_drawdown_to_today():
    """Yesterday's losses must not count toward today's drawdown."""
    risk = RiskManager()
    risk.initial_balance = 1_000.0
    risk.total_realized_pnl = -60.0
    # -50 realized yesterday, -10 today → daily = -10 → above -5% cap
    risk.seed_daily(daily_pnl_so_far=-10.0)
    assert risk.get_daily_pnl() == pytest.approx(-10.0)
    assert risk.validate(make_signal(), 0.0, "none").passed


def test_day_rollover_resets_baseline():
    risk = RiskManager()
    risk.total_realized_pnl = -100.0
    risk.day_start_realized_pnl = 0.0
    risk._current_day = "2000-01-01"  # force rollover on next tick
    risk.tick_cooldown()
    assert risk.get_daily_pnl() == pytest.approx(0.0)


def test_record_win_resets_consecutive_losses():
    risk = RiskManager()
    risk.record_loss(-5)
    risk.record_loss(-5)
    assert risk.consecutive_losses == 2
    risk.record_win(10)
    assert risk.consecutive_losses == 0
    assert risk.get_daily_pnl() == pytest.approx(0.0)  # -10 + 10
