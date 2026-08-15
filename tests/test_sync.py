"""Regression tests for PositionTracker / sync_with_chain state machine.

Covers the critical fix: an UNKNOWN chain state (account API failure or
dry-run) must never be interpreted as "position gone", which previously
cleared the tracker and allowed a new entry to double the real position.
"""

import pytest

from kimi_quant.executor import PositionTracker, TradeExecutor


def _make_executor(dry_run: bool = True) -> TradeExecutor:
    """Build a TradeExecutor without network access (skips __init__)."""
    ex = TradeExecutor.__new__(TradeExecutor)
    ex.tracker = PositionTracker()
    ex.dry_run = dry_run
    ex.coin = "BTC"
    return ex


def _activate(ex: TradeExecutor, side: str = "long", size: float = 0.01,
              entry: float = 100000.0) -> None:
    ex.tracker.update_from_open(
        side, size, entry, {"entry": 101, "sl": 102, "tp": 103},
        sl_price=99000.0, tp_price=102000.0,
    )
    ex.tracker.state = "active"  # chain-confirmed


# ─── Case 1: resting → filled ─────────────────────────────────────────────


def test_resting_to_active_on_chain_position():
    ex = _make_executor()
    ex.tracker.update_from_open("long", 0.01, 100000.0, {"entry": 1})
    assert ex.tracker.has_resting_order()

    ex.sync_with_chain("long", 0.01, 100500.0, account_known=True)

    assert ex.tracker.has_position()
    assert ex.tracker.entry_price == 100500.0


# ─── Case 2: resting → still resting ──────────────────────────────────────


def test_resting_ticks_and_times_out():
    ex = _make_executor()
    ex.tracker.update_from_open("long", 0.01, 100000.0, {"entry": 1})

    ex.sync_with_chain("none", 0.0, 0.0, account_known=True)
    assert ex.tracker.resting_cycles == 1
    assert ex.tracker.has_resting_order()

    ex.tracker.resting_cycles = ex.tracker.max_resting_cycles  # timed out
    ex.sync_with_chain("none", 0.0, 0.0, account_known=True)
    assert not ex.tracker.has_resting_order()
    assert not ex.tracker.has_position()


# ─── Case 3: active → gone ────────────────────────────────────────────────


def test_active_cleared_when_chain_confirms_flat():
    ex = _make_executor()
    _activate(ex)
    ex.sync_with_chain("none", 0.0, 0.0, account_known=True)
    assert not ex.tracker.has_position()


def test_active_NOT_cleared_when_chain_unknown():
    """CRITICAL REGRESSION: account API failure (account_known=False) must
    NOT clear the tracker — otherwise the bot would think it is flat and
    could open a new position on top of the real one."""
    ex = _make_executor(dry_run=False)
    _activate(ex)

    ex.sync_with_chain("none", 0.0, 0.0, account_known=False)

    assert ex.tracker.has_position(), (
        "Tracker must survive an unknown chain state"
    )
    assert ex.tracker.side == "long"


def test_active_NOT_cleared_in_dry_run():
    """CRITICAL REGRESSION: dry-run has no chain — the tracker is the source
    of truth. Positions must persist across cycles until an explicit CLOSE
    (previously every dry-run position was auto-closed the next cycle)."""
    ex = _make_executor(dry_run=True)
    _activate(ex)

    ex.sync_with_chain("none", 0.0, 0.0, account_known=False)

    assert ex.tracker.has_position()
    assert ex.tracker.size == 0.01


# ─── Case 4: active → still there ─────────────────────────────────────────


def test_active_entry_and_size_refreshed_from_chain():
    ex = _make_executor()
    _activate(ex, entry=100000.0)
    ex.sync_with_chain("long", 0.012, 101000.0, account_known=True)
    assert ex.tracker.has_position()
    assert ex.tracker.entry_price == 101000.0
    assert ex.tracker.size == 0.012


# ─── Case 5: recovery (no tracker state, chain has position) ──────────────


def test_recovers_position_from_chain():
    ex = _make_executor()
    ex.sync_with_chain("short", 0.02, 99000.0, account_known=True)
    assert ex.tracker.has_position()
    assert ex.tracker.side == "short"
    assert ex.tracker.size == 0.02
    assert ex.tracker.entry_price == 99000.0


# ─── WS event bridge ──────────────────────────────────────────────────────


def test_ws_sl_fill_clears_tracker_and_preserves_reason():
    from kimi_quant.monitor import EventType, OrderEvent

    ex = _make_executor()
    _activate(ex)

    ev = OrderEvent(
        event_type=EventType.ORDER_FILLED,
        order_id=ex.tracker.sl_oid,
        fill_price=98900.0,
        filled_size=0.01,
        total_size=0.01,
    )
    changed = ex.tracker.apply_ws_event(ev)

    assert changed == "sl_filled oid=102"
    assert not ex.tracker.has_position()
    assert ex.tracker.consume_ws_close_reason() == "stop_loss"


def test_ws_entry_fill_promotes_resting_to_active():
    from kimi_quant.monitor import EventType, OrderEvent

    ex = _make_executor()
    ex.tracker.update_from_open("long", 0.01, 100000.0, {"entry": 7})

    ev = OrderEvent(
        event_type=EventType.ORDER_FILLED,
        order_id=7,
        fill_price=100100.0,
        filled_size=0.01,
        total_size=0.01,
    )
    assert ex.tracker.apply_ws_event(ev) == "entry_filled oid=7"
    assert ex.tracker.has_position()
    assert ex.tracker.entry_price == 100100.0
