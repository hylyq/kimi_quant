"""TradeExecutor / PositionTracker tests (dry-run, no network)."""

import pytest

from kimi_quant.executor import (
    PositionTracker,
    TradeExecutor,
    _is_trigger_order,
    _round_to_tick,
)
from kimi_quant.llm import TradingSignal
from kimi_quant.monitor import EventType, OrderEvent


# ─── Pure helpers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("price,tick,expected", [
    (65_906.8, 0.5, 65_907.0),
    (65_906.4, 0.5, 65_906.5),
    (65_906.2, 0.5, 65_906.0),
    (100.04, 0.1, 100.0),
    (100.06, 0.1, 100.1),
    (99.999, 1.0, 100.0),
    (123.45, 0.0, 123.45),  # tick=0 → untouched
])
def test_round_to_tick(price, tick, expected):
    assert _round_to_tick(price, tick) == pytest.approx(expected)


def test_is_trigger_order():
    assert _is_trigger_order({"orderType": {"trigger": {"triggerPx": "64200"}}})
    assert _is_trigger_order({"orderType": "Trigger"})
    assert not _is_trigger_order({"orderType": {"limit": {}}})
    assert not _is_trigger_order({"orderType": "limit"})
    assert not _is_trigger_order({})


# ─── Tracker state machine ────────────────────────────────────────────────


def make_event(etype, oid, price=100.0, size=0.01) -> OrderEvent:
    return OrderEvent(
        event_type=etype, coin="BTC", side="sell", order_id=oid,
        filled_size=size, total_size=size, fill_price=price,
    )


def test_ws_entry_fill_promotes_resting():
    t = PositionTracker()
    t.update_from_open("long", 0.01, 100_000.0, {"entry": 11})
    assert t.state == "resting"
    t.apply_ws_event(make_event(EventType.ORDER_FILLED, 11, price=100_050.0))
    assert t.state == "active"
    assert t.entry_price == 100_050.0


def test_ws_sl_fill_clears_and_captures_price():
    t = PositionTracker()
    t.update_from_open("long", 0.01, 100_000.0,
                       {"entry": 11, "sl": 22, "tp": 33},
                       sl_price=99_000.0, tp_price=102_000.0)
    t.state = "active"

    t.apply_ws_event(make_event(EventType.ORDER_FILLED, 22, price=98_900.0))
    assert t.state == "none"
    reason, price = t.consume_ws_close()
    assert reason == "stop_loss"
    assert price == 98_900.0
    # Consumed once
    assert t.consume_ws_close() == (None, 0.0)


def test_ws_close_reason_survives_clear():
    """clear() must NOT wipe the WS close bridge — the main loop reads it after."""
    t = PositionTracker()
    t._last_ws_close_reason = "take_profit"
    t._last_ws_close_price = 101_000.0
    t.clear()
    assert t.consume_ws_close() == ("take_profit", 101_000.0)


def test_ws_cancel_clears_sl_oid_only():
    t = PositionTracker()
    t.update_from_open("long", 0.01, 100_000.0,
                       {"entry": 11, "sl": 22, "tp": 33},
                       sl_price=99_000.0, tp_price=102_000.0)
    t.state = "active"
    t.apply_ws_event(make_event(EventType.ORDER_CANCELLED, 22))
    assert t.sl_oid is None and t.sl_price == 0.0
    assert t.tp_oid == 33  # untouched
    assert t.state == "active"


def test_sync_active_corrects_drifted_side():
    t = PositionTracker()
    t.update_from_open("short", 0.01, 100_000.0, {})
    t.state = "active"
    # Chain now reports a long (external flip) — side must follow the chain.
    t.sync_active("long", 0.008, 101_000.0)
    assert t.side == "long"
    assert t.size == 0.008
    assert t.entry_price == 101_000.0


# ─── Executor (dry-run) ───────────────────────────────────────────────────


@pytest.fixture
def executor():
    return TradeExecutor()  # conftest forces config.dry_run=True


def test_verify_flags_missing_sl_without_oid(executor):
    """Live mode: sl_oid None + active position = unprotected, must be flagged."""
    executor.tracker.update_from_open("long", 0.01, 100_000.0, {})
    executor.tracker.state = "active"

    executor.dry_run = False  # simulate live for the check only
    status = executor.verify_tracked_orders([])
    assert status["sl_missing"] is True

    executor.dry_run = True  # dry-run trackers never carry oids — no spam
    status = executor.verify_tracked_orders([])
    assert status["sl_missing"] is False


def test_verify_flags_sl_oid_not_on_chain(executor):
    executor.tracker.update_from_open("long", 0.01, 100_000.0, {"sl": 22})
    executor.tracker.state = "active"
    executor.dry_run = False
    status = executor.verify_tracked_orders([{"coin": "BTC", "oid": 99}])
    assert status["sl_missing"] is True


def test_sync_with_chain_recovers_unknown_position(executor):
    executor.sync_with_chain("long", 0.02, 99_500.0)
    assert executor.tracker.has_position()
    assert executor.tracker.side == "long"
    assert executor.tracker.size == 0.02


def test_sync_with_chain_fixes_side_mismatch(executor):
    executor.tracker.update_from_open("short", 0.01, 100_000.0, {})
    executor.tracker.state = "active"
    executor.sync_with_chain("long", 0.01, 100_000.0)
    assert executor.tracker.side == "long"


def test_dispatch_cancel_stale_dry_run(executor):
    result = executor._dispatch("CANCEL_STALE", TradingSignal(
        action="CANCEL_STALE", confidence=0.5, reasoning="cleanup"))
    assert result["executed"] is True and result["dry_run"] is True


def test_modify_sl_dry_run_updates_tracker(executor):
    executor.tracker.update_from_open("long", 0.01, 100_000.0, {})
    executor.tracker.state = "active"
    result = executor._dispatch("MODIFY_SL", TradingSignal(
        action="MODIFY_SL", confidence=0.6, reasoning="trail",
        modify_sl_to=100_200.0))
    assert result["executed"] is True
    assert executor.tracker.sl_price == 100_200.0


def test_build_entry_orders_aborts_without_mid(executor):
    """all_mids unavailable → must raise, never place a blind Ioc order."""
    executor.dry_run = False  # _build_entry_orders is a live-only path
    executor.info = None     # force all_mids failure
    signal = TradingSignal(action="LONG", confidence=0.9, reasoning="r",
                           size=0.01, stop_loss=99_000.0, take_profit=102_000.0)
    with pytest.raises(ValueError, match="all_mids"):
        executor._build_entry_orders(signal, is_buy=True, size=0.01)


def test_execute_hold_not_a_failure(executor):
    result = executor.execute(TradingSignal(
        action="HOLD", confidence=0.4, reasoning="wait"))
    assert result["executed"] is False
    assert result["action"] == "HOLD"


# ─── Passive price clamping (maker entries) ───────────────────────────────


from kimi_quant.executor import _clamp_passive_price


def test_aggressive_buy_clamped_to_bid():
    # LLM wants to buy at the ask — clamp to the bid so it rests as maker
    assert _clamp_passive_price(100_050.0, 100_000.0, 100_050.0, True) == 100_000.0


def test_passive_buy_kept():
    assert _clamp_passive_price(99_800.0, 100_000.0, 100_050.0, True) == 99_800.0


def test_aggressive_sell_clamped_to_ask():
    assert _clamp_passive_price(99_900.0, 100_000.0, 100_050.0, False) == 100_050.0


def test_passive_sell_kept():
    assert _clamp_passive_price(100_200.0, 100_000.0, 100_050.0, False) == 100_200.0


# ─── ensure_protective_orders (dry-run no-op / no position) ──────────────


def test_ensure_protective_orders_noop_without_position():
    ex = TradeExecutor()  # dry-run (conftest)
    assert ex.ensure_protective_orders() == {"sl_placed": False, "tp_placed": False}


def test_ensure_protective_orders_noop_in_dry_run():
    ex = TradeExecutor()
    ex.tracker.update_from_open(
        "long", 0.01, 100_000.0, {},
        sl_price=99_000.0, tp_price=102_000.0,
    )
    ex.tracker.state = "active"
    # Dry-run: no orders exist to place — must not raise or "place" anything
    assert ex.ensure_protective_orders() == {"sl_placed": False, "tp_placed": False}


# ─── Maker fill flow: resting → filled → protection pending ──────────────


def test_maker_flow_resting_then_ws_fill_retains_planned_sl_tp():
    t = PositionTracker()
    # What _open_position_maker records when the GTC order is acked:
    t.update_from_open("long", 0.01, 99_950.0, {"entry": 42},
                       sl_price=99_000.0, tp_price=102_000.0)
    assert t.state == "resting"
    assert t.sl_oid is None and t.tp_oid is None  # deferred until fill
    # WS reports the fill:
    t.apply_ws_event(make_event(EventType.ORDER_FILLED, 42, price=99_950.0))
    assert t.state == "active"
    # Planned levels survive so ensure_protective_orders can use them:
    assert t.sl_price == pytest.approx(99_000.0)
    assert t.tp_price == pytest.approx(102_000.0)
    assert t.sl_oid is None  # still unplaced → ensure will handle it
