"""OrderMonitor parsing tests — batched WS payloads must not lose events."""

from kimi_quant.monitor import EventType, FlashReporter, OrderMonitor


def make_monitor() -> OrderMonitor:
    return OrderMonitor(base_url="https://example.invalid", address="0xabc")


# ─── orderUpdates ─────────────────────────────────────────────────────────


def test_batched_updates_all_returned():
    """A batch with cancel + reject must yield BOTH events, not just the first."""
    m = make_monitor()
    payload = {"data": [
        {"order": {"oid": "1", "coin": "BTC", "side": "B", "sz": "0.01"},
         "status": "canceled"},
        {"order": {"oid": "2", "coin": "BTC", "side": "A", "sz": "0.02"},
         "status": "rejected"},
    ]}
    events = m._parse_order_update(payload)
    assert [e.order_id for e in events] == [1, 2]
    assert events[0].event_type == EventType.ORDER_CANCELLED
    assert events[1].event_type == EventType.ORDER_REJECTED


def test_tpsl_batch_keeps_both_transitions():
    """SL filled + TP auto-cancelled often arrive in ONE orderUpdates batch."""
    m = make_monitor()
    payload = {"data": [
        {"order": {"oid": "33", "coin": "BTC", "side": "A", "sz": "0.01"},
         "status": "filled"},  # reported via userFills — skipped here
        {"order": {"oid": "22", "coin": "BTC", "side": "A", "sz": "0.01"},
         "status": "canceled"},  # the orphaned TP — must not be lost
    ]}
    events = m._parse_order_update(payload)
    assert [e.order_id for e in events] == [22]
    assert events[0].event_type == EventType.ORDER_CANCELLED


def test_filled_status_skipped_in_order_updates():
    """Fills are reported by userFills (real price) — orderUpdates skips them."""
    m = make_monitor()
    payload = {"order": {"oid": "7", "coin": "BTC", "side": "B", "sz": "0.01"},
               "status": "filled"}
    assert m._parse_order_update(payload) == []


def test_partial_fill_detected():
    m = make_monitor()
    payload = {"order": {"oid": "5", "coin": "BTC", "side": "B",
                         "sz": "0.004", "origSz": "0.01"},
               "status": "open"}
    events = m._parse_order_update(payload)
    assert len(events) == 1
    assert events[0].event_type == EventType.ORDER_PARTIAL
    assert events[0].filled_size == 0.006


def test_plain_open_skipped():
    m = make_monitor()
    payload = {"order": {"oid": "5", "coin": "BTC", "side": "B",
                         "sz": "0.01", "origSz": "0.01"},
               "status": "open"}
    assert m._parse_order_update(payload) == []


def test_non_dict_and_sentinel():
    m = make_monitor()
    assert m._parse_order_update("not a dict") == []
    assert m._parse_order_update({"_sentinel": True}) == []


# ─── userFills ────────────────────────────────────────────────────────────


def test_batched_fills_all_returned_with_real_price():
    m = make_monitor()
    payload = {"data": [
        {"oid": "101", "coin": "BTC", "side": "B", "px": "100_123.5", "sz": "0.005"},
        {"oid": "102", "coin": "BTC", "side": "A", "px": "100_200.0", "sz": "0.01"},
    ]}
    events = m._parse_fill_update(payload)
    assert [e.order_id for e in events] == [101, 102]
    assert events[0].fill_price == 100_123.5
    assert all(e.event_type == EventType.ORDER_FILLED for e in events)


def test_fill_missing_coin_skipped():
    m = make_monitor()
    payload = {"data": [{"oid": "1", "px": "1", "sz": "1"}]}
    assert m._parse_fill_update(payload) == []


# ─── FlashReporter fallback formatting ────────────────────────────────────


def test_fallback_cancelled_format():
    r = FlashReporter(event_queue=None, api_key=None)
    from kimi_quant.monitor import OrderEvent
    ev = OrderEvent(event_type=EventType.ORDER_CANCELLED, coin="BTC",
                    order_id=42)
    text = r._format_fallback(ev)
    assert "42" in text and "取消" in text
