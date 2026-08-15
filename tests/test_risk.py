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
        _sig(["CLOSE", "SHORT"]), current_position_size=0.01,
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
        _sig(["CLOSE", "SHORT"]), current_position_size=0.01,
        current_position_side="long", mid_price=MID,
    )
    assert check.passed


def test_hold_always_passes(risk):
    check = risk.validate_sequence(_sig(["HOLD"], confidence=0.0))
    assert check.passed
