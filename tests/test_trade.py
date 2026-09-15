"""Tests for TradeLogger P&L math and main.py feedback helpers."""

import pytest

from kimi_quant.analytics import TradeLogger
from kimi_quant.main import _build_last_cycle_feedback, _fail_safe_account_unavailable


@pytest.fixture
def logger(tmp_path) -> TradeLogger:
    return TradeLogger(log_path=str(tmp_path / "trades.jsonl"))


# ─── P&L math ─────────────────────────────────────────────────────────────


def test_long_pnl_and_net_after_fees(logger):
    logger.open_trade(side="long", size=0.01, entry_price=100000.0)
    trade = logger.close_trade(exit_price=101000.0, reason="take_profit")

    assert trade is not None
    assert trade.pnl == pytest.approx(10.0)          # 1000 * 0.01
    assert trade.pnl_pct == pytest.approx(1.0)
    assert trade.is_win
    # fees = (100000 + 101000) * 0.01 * 0.00035
    assert trade.fees_est == pytest.approx((100000 + 101000) * 0.01 * 0.00035)
    assert trade.net_pnl == pytest.approx(trade.pnl - trade.fees_est)


def test_short_pnl(logger):
    logger.open_trade(side="short", size=0.01, entry_price=100000.0)
    trade = logger.close_trade(exit_price=99000.0, reason="take_profit")
    assert trade is not None
    assert trade.pnl == pytest.approx(10.0)
    assert trade.is_win


def test_loss_trade(logger):
    logger.open_trade(side="long", size=0.01, entry_price=100000.0)
    trade = logger.close_trade(exit_price=99000.0, reason="stop_loss")
    assert trade is not None
    assert trade.pnl == pytest.approx(-10.0)
    assert not trade.is_win
    assert trade.close_reason == "stop_loss"


def test_stats_computation(logger):
    logger.open_trade(side="long", size=0.01, entry_price=100000.0)
    logger.close_trade(exit_price=101000.0, reason="take_profit")
    logger.open_trade(side="long", size=0.01, entry_price=100000.0)
    logger.close_trade(exit_price=99000.0, reason="stop_loss")

    stats = logger.get_stats()
    assert stats.total_trades == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.win_rate == pytest.approx(50.0)
    assert stats.net_pnl == pytest.approx(-stats.total_fees)


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    lg1 = TradeLogger(log_path=path)
    lg1.open_trade(side="long", size=0.01, entry_price=100000.0)
    lg1.close_trade(exit_price=101000.0, reason="take_profit")

    lg2 = TradeLogger(log_path=path)
    assert len(lg2.get_all_trades()) == 1
    t = lg2.get_all_trades()[0]
    assert t.pnl == pytest.approx(10.0)
    assert t.close_reason == "take_profit"


def test_recover_trade(logger):
    logger.recover_trade(side="long", size=0.01, entry_price=100000.0)
    assert logger.has_pending
    trade = logger.close_trade(exit_price=99000.0, reason="manual")
    assert trade is not None
    assert trade.close_reason == "manual"


# ─── main.py helpers ──────────────────────────────────────────────────────


def test_last_cycle_feedback_executed():
    fb = _build_last_cycle_feedback({
        "signal": "LONG", "confidence": 0.8,
        "reasoning": "trend up", "status": "executed",
    })
    assert "EXECUTED" in fb
    assert "LONG" in fb


def test_last_cycle_feedback_rejected():
    fb = _build_last_cycle_feedback({
        "signal": "SHORT", "confidence": 0.7,
        "reasoning": "x", "status": "rejected",
    })
    assert "REJECTED" in fb


def test_last_cycle_feedback_empty_when_no_prev():
    assert _build_last_cycle_feedback(None) == ""


def test_fail_safe_account_unavailable_skips():
    result = _fail_safe_account_unavailable()
    assert result["status"] == "skipped"
    assert "unavailable" in result["reason"].lower()
