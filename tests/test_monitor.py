"""OrderMonitor parsing tests — batched WS payloads must not lose events.

Covers the fixes:
  - batched orderUpdates / userFills return ALL events (none dropped)
  - partial-fill total_size is the ORIGINAL size, not the remaining size
  - fills are reported from userFills only (real fill price, no duplicates)
"""

import pytest

from kimi_quant.monitor import EventType, FlashReporter, OrderMonitor, OrderEvent


@pytest.fixture
def monitor() -> OrderMonitor:
    return OrderMonitor(base_url="https://example.invalid", address="0xabc")


# ─── Batched orderUpdates ─────────────────────────────────────────────────


def test_batched_updates_all_returned(monitor):
    """A batch with cancel + reject must yield BOTH events, not just the first."""
    payload = {"data": [
        {"order": {"oid": "1", "coin": "BTC", "side": "B", "sz": "0.01"},
         "status": "canceled"},
        {"order": {"oid": "2", "coin": "BTC", "side": "A", "sz": "0.02"},
         "status": "rejected"},
    ]}
    events = monitor._parse_order_update(payload)
    assert [e.order_id for e in events] == [1, 2]
    assert events[0].event_type == EventType.ORDER_CANCELLED
    assert events[1].event_type == EventType.ORDER_REJECTED


def test_tpsl_batch_keeps_both_transitions(monitor):
    """SL filled + TP auto-cancelled often arrive in ONE orderUpdates batch."""
    payload = {"data": [
        {"order": {"oid": "33", "coin": "BTC", "side": "A", "sz": "0.01"},
         "status": "filled"},  # reported via userFills — skipped here
        {"order": {"oid": "22", "coin": "BTC", "side": "A", "sz": "0.01"},
         "status": "canceled"},  # the orphaned TP — must not be lost
    ]}
    events = monitor._parse_order_update(payload)
    assert [e.order_id for e in events] == [22]
    assert events[0].event_type == EventType.ORDER_CANCELLED


def test_filled_status_skipped_in_order_updates(monitor):
    """Fills are reported by userFills (real price) — orderUpdates skips them."""
    payload = {"order": {"oid": "7", "coin": "BTC", "side": "B", "sz": "0.01"},
               "status": "filled"}
    assert monitor._parse_order_update(payload) == []


def test_plain_open_skipped(monitor):
    payload = {"order": {"oid": "5", "coin": "BTC", "side": "B",
                         "sz": "0.01", "origSz": "0.01"},
               "status": "open"}
    assert monitor._parse_order_update(payload) == []


def test_non_dict_and_sentinel(monitor):
    assert monitor._parse_order_update("not a dict") == []
    assert monitor._parse_order_update({"_sentinel": True}) == []
    assert monitor._parse_fill_update({"_sentinel": True}) == []


# ─── Partial fills ────────────────────────────────────────────────────────


def test_partial_fill_detected(monitor):
    payload = {"order": {"oid": "5", "coin": "BTC", "side": "B",
                         "sz": "0.004", "origSz": "0.01"},
               "status": "open"}
    events = monitor._parse_order_update(payload)
    assert len(events) == 1
    assert events[0].event_type == EventType.ORDER_PARTIAL
    assert events[0].filled_size == pytest.approx(0.006)


def test_partial_fill_total_is_original_size(monitor):
    """sz=0.007 remaining of origSz=0.01 → filled 0.003, TOTAL 0.01
    (previously total was wrongly set to the remaining 0.007)."""
    payload = {"data": [
        {"order": {"oid": "333", "coin": "BTC", "side": "B",
                   "sz": "0.007", "origSz": "0.01"},
         "status": "open"},
    ]}
    events = monitor._parse_order_update(payload)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == EventType.ORDER_PARTIAL
    assert ev.filled_size == pytest.approx(0.003)
    assert ev.total_size == pytest.approx(0.01)
    assert ev.remaining_size == pytest.approx(0.007)
    assert ev.fill_pct == pytest.approx(30.0)


# ─── userFills ────────────────────────────────────────────────────────────


def test_batched_fills_all_returned_with_real_price(monitor):
    payload = {"data": [
        {"oid": "101", "coin": "BTC", "side": "B", "px": "100_123.5", "sz": "0.005"},
        {"oid": "102", "coin": "BTC", "side": "A", "px": "100_200.0", "sz": "0.01"},
    ]}
    events = monitor._parse_fill_update(payload)
    assert [e.order_id for e in events] == [101, 102]
    assert events[0].fill_price == 100_123.5
    assert all(e.event_type == EventType.ORDER_FILLED for e in events)


def test_fill_missing_coin_skipped(monitor):
    payload = {"data": [{"oid": "1", "px": "1", "sz": "1"}]}
    assert monitor._parse_fill_update(payload) == []


# ─── FlashReporter fallback formatting ────────────────────────────────────


def test_fallback_cancelled_format():
    r = FlashReporter(event_queue=None, api_key=None)
    ev = OrderEvent(event_type=EventType.ORDER_CANCELLED, coin="BTC",
                    order_id=42)
    text = r._format_fallback(ev)
    assert "42" in text and "取消" in text
