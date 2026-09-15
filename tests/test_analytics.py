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


# ─── Real fee rates (verified against live fills: 4.5bp taker / 1.5bp maker)


def test_taker_fee_constants():
    from kimi_quant.analytics import MAKER_FEE_RATE, ROUNDTRIP_TAKER_FEE
    assert TAKER_FEE_RATE == pytest.approx(0.00045)
    assert MAKER_FEE_RATE == pytest.approx(0.00015)
    assert ROUNDTRIP_TAKER_FEE == pytest.approx(0.0009)


def test_maker_entry_pays_maker_fee_only_on_entry():
    t = TradeRecord(opened_at="2026-01-01T00:00:00+00:00", side="long",
                    size=0.01, entry_price=100_000.0, entry_type="maker")
    t.close(101_000.0, "take_profit")
    expected = (100_000.0 * 0.01 * 0.00015
                + 101_000.0 * 0.01 * TAKER_FEE_RATE)
    assert t.fees_est == pytest.approx(expected)


# ─── Calibration fields survive persistence ───────────────────────────────


def test_confidence_fields_roundtrip(tmp_path):
    log = tmp_path / "t.jsonl"
    logger = TradeLogger(log_path=str(log))
    logger.open_trade("long", 0.01, 100_000.0, dry_run=False,
                      entry_confidence=0.82, planned_sl=99_000.0,
                      planned_tp=102_000.0, entry_type="maker")
    logger.close_trade(101_000.0, "take_profit")

    loaded = TradeLogger(log_path=str(log)).get_all_trades()[0]
    assert loaded.entry_confidence == pytest.approx(0.82)
    assert loaded.planned_sl == pytest.approx(99_000.0)
    assert loaded.planned_tp == pytest.approx(102_000.0)
    assert loaded.entry_type == "maker"


def test_old_records_load_without_new_fields(tmp_path):
    import json
    log = tmp_path / "legacy.jsonl"
    log.write_text(json.dumps({
        "opened_at": "2026-07-18T19:07:55+00:00", "side": "short",
        "size": 0.0005, "entry_price": 64000.0, "dry_run": False,
        "closed_at": "2026-07-18T20:08:35+00:00", "exit_price": 64135.0,
        "close_reason": "stop_loss", "pnl": -0.075, "pnl_pct": -0.23,
        "fees_est": 0.029, "net_pnl": -0.104, "is_win": False,
    }) + "\n")
    loaded = TradeLogger(log_path=str(log)).get_all_trades()[0]
    assert loaded.entry_confidence == 0.0
    assert loaded.entry_type == "market"


# ─── Calibration / exit attribution report ────────────────────────────────


def _seed_trades(logger: TradeLogger):
    """6 real trades: 3 at conf 0.75 (1 win), 3 at conf 0.85 (2 wins)."""
    rows = [
        (0.75, 100_000.0, 101_500.0, "take_profit"),   # win
        (0.75, 100_000.0, 99_400.0, "stop_loss"),      # loss
        (0.75, 100_000.0, 100_010.0, "signal"),        # fee-scratch loss
        (0.85, 100_000.0, 102_000.0, "take_profit"),   # win
        (0.85, 100_000.0, 101_800.0, "take_profit"),   # win
        (0.85, 100_000.0, 99_500.0, "stop_loss"),      # loss
    ]
    for conf, entry, exit_, reason in rows:
        logger.open_trade("long", 0.01, entry, dry_run=False,
                          entry_confidence=conf)
        logger.close_trade(exit_, reason)


def test_calibration_context_buckets_and_attribution(tmp_path):
    logger = TradeLogger(log_path=str(tmp_path / "t.jsonl"))
    _seed_trades(logger)
    ctx = logger.get_calibration_context()
    assert "0.75" in ctx and "0.85" in ctx
    assert "1/3 wins" in ctx          # 0.75 bucket
    assert "2/3 wins" in ctx          # 0.85 bucket
    assert "LLM CLOSE" in ctx and "SL hit" in ctx and "TP hit" in ctx


def test_calibration_context_needs_five_tagged_trades(tmp_path):
    logger = TradeLogger(log_path=str(tmp_path / "t.jsonl"))
    for i in range(4):  # only 4 confidence-tagged trades
        logger.open_trade("long", 0.01, 100_000.0, dry_run=False,
                          entry_confidence=0.8)
        logger.close_trade(101_000.0, "take_profit")
    assert logger.get_calibration_context() == ""


def test_untagged_trades_do_not_count(tmp_path):
    """Historical trades without entry_confidence never trigger the report."""
    logger = TradeLogger(log_path=str(tmp_path / "t.jsonl"))
    for i in range(10):
        logger.open_trade("long", 0.01, 100_000.0, dry_run=False)
        logger.close_trade(101_000.0, "take_profit")
    assert logger.get_calibration_context() == ""
