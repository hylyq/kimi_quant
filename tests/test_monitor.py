"""Regression tests for WebSocket event parsing in OrderMonitor.

Covers the fixes:
  - batched orderUpdates / userFills return ALL events (none dropped)
  - partial-fill total_size is the ORIGINAL size, not the remaining size
"""

import pytest

from kimi_quant.monitor import EventType, OrderMonitor


@pytest.fixture
def monitor() -> OrderMonitor:
    return OrderMonitor(
        base_url="https://api.hyperliquid.xyz",
        address="0x0000000000000000000000000000000000000000",
    )


# ─── Batched orderUpdates ─────────────────────────────────────────────────


def test_batch_returns_all_filled_events(monitor):
    data = {"data": [
        {"order": {"oid": 111, "coin": "BTC", "side": "B", "sz": "0.01"},
         "status": "filled"},
        {"order": {"oid": 222, "coin": "BTC", "side": "A", "sz": "0.02"},
         "status": "filled"},
    ]}
    events = monitor._parse_order_update(data)
    assert len(events) == 2, "batched events must NOT be dropped"
    assert [e.order_id for e in events] == [111, 222]
    assert all(e.event_type == EventType.ORDER_FILLED for e in events)


def test_batch_skips_non_significant_keeps_significant(monitor):
    data = {"data": [
        {"order": {"oid": 1, "coin": "BTC", "side": "B", "sz": "0.01"},
         "status": "open"},  # not partial → not significant
        {"order": {"oid": 2, "coin": "BTC", "side": "A", "sz": "0.01"},
         "status": "canceled"},
    ]}
    events = monitor._parse_order_update(data)
    assert len(events) == 1
    assert events[0].order_id == 2
    assert events[0].event_type == EventType.ORDER_CANCELLED


def test_single_dict_input_works(monitor):
    data = {"order": {"oid": 5, "coin": "BTC", "side": "B", "sz": "0.01"},
            "status": "filled"}
    events = monitor._parse_order_update(data)
    assert len(events) == 1
    assert events[0].order_id == 5


def test_sentinel_returns_empty(monitor):
    assert monitor._parse_order_update({"_sentinel": True}) == []
    assert monitor._parse_fill_update({"_sentinel": True}) == []


# ─── Partial fills ────────────────────────────────────────────────────────


def test_partial_fill_total_is_original_size(monitor):
    """sz=0.007 remaining of origSz=0.01 → filled 0.003, TOTAL 0.01
    (previously total was wrongly set to the remaining 0.007)."""
    data = {"data": [
        {"order": {"oid": 333, "coin": "BTC", "side": "B",
                   "sz": "0.007", "origSz": "0.01"},
         "status": "open"},
    ]}
    events = monitor._parse_order_update(data)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == EventType.ORDER_PARTIAL
    assert ev.filled_size == pytest.approx(0.003)
    assert ev.total_size == pytest.approx(0.01)
    assert ev.remaining_size == pytest.approx(0.007)
    assert ev.fill_pct == pytest.approx(30.0)


def test_full_fill_total_is_order_size(monitor):
    data = {"data": [
        {"order": {"oid": 444, "coin": "BTC", "side": "A", "sz": "0.01"},
         "status": "filled"},
    ]}
    events = monitor._parse_order_update(data)
    ev = events[0]
    assert ev.filled_size == pytest.approx(0.01)
    assert ev.total_size == pytest.approx(0.01)
    assert ev.fill_pct == pytest.approx(100.0)


# ─── Batched userFills ────────────────────────────────────────────────────


def test_fill_batch_returns_all_fills(monitor):
    data = {"data": [
        {"oid": 101, "coin": "BTC", "side": "B", "px": "100000", "sz": "0.005"},
        {"oid": 102, "coin": "BTC", "side": "B", "px": "100010", "sz": "0.005"},
    ]}
    events = monitor._parse_fill_update(data)
    assert len(events) == 2
    assert [e.order_id for e in events] == [101, 102]
    assert events[0].fill_price == pytest.approx(100000.0)
    assert events[0].event_type == EventType.ORDER_FILLED
