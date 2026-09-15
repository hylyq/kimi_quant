"""TradeRecord / TradeLogger P&L tests."""

import pytest

from kimi_quant.analytics import TAKER_FEE_RATE, TradeLogger, TradeRecord


def test_long_pnl_and_fees():
    t = TradeRecord(opened_at="2026-01-01T00:00:00+00:00", side="long",
                    size=0.01, entry_price=100_000.0)
    t.close(101_000.0, "take_profit")
    assert t.pnl == pytest.approx(10.0)
    expected_fees = (100_000.0 * 0.01 + 101_000.0 * 0.01) * TAKER_FEE_RATE
    assert t.fees_est == pytest.approx(expected_fees)
    assert t.net_pnl == pytest.approx(10.0 - expected_fees)
    assert t.is_win


def test_short_pnl():
    t = TradeRecord(opened_at="2026-01-01T00:00:00+00:00", side="short",
                    size=0.01, entry_price=100_000.0)
    t.close(99_000.0, "take_profit")
    assert t.pnl == pytest.approx(10.0)


def test_is_win_is_net_of_fees():
    """A $1 gross gain on a $5 fee bill is a LOSS for breaker purposes."""
    t = TradeRecord(opened_at="2026-01-01T00:00:00+00:00", side="long",
                    size=1.0, entry_price=100_000.0)
    t.close(100_001.0, "signal")
    assert t.pnl == pytest.approx(1.0)
    assert t.fees_est > 5.0
    assert not t.is_win


def test_logger_roundtrip(tmp_path):
    log = tmp_path / "trades.jsonl"
    logger = TradeLogger(log_path=str(log))
    logger.open_trade("long", 0.01, 100_000.0, dry_run=True)
    assert logger.has_pending
    trade = logger.close_trade(101_000.0, "take_profit")
    assert trade is not None and trade.is_win
    assert not logger.has_pending
    assert log.exists()

    reloaded = TradeLogger(log_path=str(log))
    assert len(reloaded.get_all_trades()) == 1
    loaded = reloaded.get_all_trades()[0]
    assert loaded.entry_price == 100_000.0
    assert loaded.exit_price == 101_000.0


def test_cancel_pending(tmp_path):
    logger = TradeLogger(log_path=str(tmp_path / "t.jsonl"))
    logger.open_trade("long", 0.01, 100_000.0)
    logger.cancel_pending()
    assert not logger.has_pending
    assert logger.close_trade(101_000.0) is None  # nothing to close
