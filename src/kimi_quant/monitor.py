"""Real-time order monitoring via Hyperliquid WebSocket + Flash LLM reporting.

Architecture:
  - OrderMonitor: background thread with asyncio event loop, subscribes to
    Hyperliquid WebSocket for orderUpdates and userFills events.
  - FlashReporter: consumes events from a thread-safe queue, uses a cheap
    Flash-level LLM to generate natural-language notifications, sends them
    via the existing Notifier.

Only meaningful state transitions trigger a notification:
  - Order filled (entry / stop loss / take profit)
  - Order partially filled
  - Order cancelled or rejected
  - Liquidation detected (via user events)

Usage:
    from kimi_quant.monitor import OrderMonitor

    monitor = OrderMonitor(
        base_url="https://api.hyperliquid.xyz",
        address="0x...",
        tracker=executor.tracker,   # PositionTracker reference
    )
    monitor.start()
    ...
    monitor.stop()
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kimi_quant.config import config

logger = logging.getLogger(__name__)


# ─── Event Types ────────────────────────────────────────────────────────────


class EventType:
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL = "order_partial"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_CLOSED = "position_closed"  # SL/TP triggered
    LIQUIDATED = "liquidated"
    UNKNOWN = "unknown"


@dataclass
class OrderEvent:
    """Normalized order event from WebSocket feed."""

    event_type: str  # one of EventType values
    coin: str = ""
    side: str = ""  # "buy" | "sell"
    order_id: int | None = None
    order_type: str = ""  # "limit" | "market" | "stop_loss" | "take_profit" | ""
    filled_size: float = 0.0
    total_size: float = 0.0
    fill_price: float = 0.0
    remaining_size: float = 0.0
    status: str = ""  # "filled" | "partial_fill" | "cancelled" | "rejected"
    raw: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def fill_pct(self) -> float:
        if self.total_size > 0:
            return (self.filled_size / self.total_size) * 100
        return 0.0

    @property
    def is_significant(self) -> bool:
        """Only report events that matter."""
        return self.event_type != EventType.UNKNOWN


# ─── Order Monitor ──────────────────────────────────────────────────────────


class OrderMonitor:
    """WebSocket-based order tracker running in a background thread.

    Subscribes to orderUpdates and userFills from Hyperliquid.
    Pushes normalized OrderEvents to a thread-safe queue consumed by
    FlashReporter.
    """

    def __init__(
        self,
        base_url: str,
        address: str,
        tracker: Any = None,  # PositionTracker (avoids circular import)
    ):
        self.base_url = base_url
        self.address = address
        self.tracker = tracker
        self._queue: queue.Queue[OrderEvent] = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._error_count = 0
        self._last_heartbeat: float = 0.0

    @property
    def events(self) -> queue.Queue:
        """Queue of normalized OrderEvents for the reporter to consume."""
        return self._queue

    def start(self) -> None:
        """Launch the WebSocket monitor in a daemon background thread."""
        if self._started:
            logger.warning("OrderMonitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="order-monitor",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info("OrderMonitor started (address=%s)", self.address[:10] + "...")

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the monitor to stop and wait for clean shutdown."""
        if not self._started:
            return

        logger.info("Stopping OrderMonitor...")
        self._stop_event.set()
        # Push a sentinel so the event loop wakes up
        try:
            self._queue.put_nowait(
                OrderEvent(event_type=EventType.UNKNOWN, raw={"_sentinel": True})
            )
        except queue.Full:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("OrderMonitor thread did not stop within %.1fs", timeout)

        self._started = False
        logger.info("OrderMonitor stopped (errors=%d)", self._error_count)

    def is_running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    # ─── Internal: WebSocket Event Loop ──────────────────────────────────

    _RESTART_DELAY = 5.0   # seconds between restart attempts
    _MAX_CONSECUTIVE_CRASHES = 10  # give up after this many crashes

    def _run_loop(self) -> None:
        """Main event loop running in the background thread.

        Auto-restarts on crash up to _MAX_CONSECUTIVE_CRASHES times,
        with an exponential backoff capped at 60 seconds.
        """
        consecutive = 0
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._async_run())
                # Clean exit (stop requested) — don't restart
                break
            except Exception:
                consecutive += 1
                self._error_count += 1
                if consecutive >= self._MAX_CONSECUTIVE_CRASHES:
                    logger.error(
                        "OrderMonitor: %d consecutive crashes, giving up",
                        consecutive,
                    )
                    break
                delay = min(self._RESTART_DELAY * (2 ** (consecutive - 1)), 60.0)
                logger.error(
                    "OrderMonitor event loop crashed (crash %d/%d), "
                    "restarting in %.1fs",
                    consecutive, self._MAX_CONSECUTIVE_CRASHES, delay,
                    exc_info=True,
                )
                self._stop_event.wait(delay)

    async def _async_run(self) -> None:
        """Async setup: create Info, subscribe, keep running until stopped."""
        from hyperliquid.info import Info
        from hyperliquid.utils.types import (
            OrderUpdatesSubscription,
            UserFillsSubscription,
        )

        # Create a separate Info instance with WebSocket enabled.
        # The main executor's Info has skip_ws=True — we need our own.
        info = Info(base_url=self.base_url, skip_ws=False)

        # Subscribe to order updates (status changes: filled, cancelled, rejected)
        order_sub = OrderUpdatesSubscription(
            type="orderUpdates",
            user=self.address,
        )
        # Subscribe to user fills (detailed fill events with prices)
        fill_sub = UserFillsSubscription(
            type="userFills",
            user=self.address,
        )

        sub_id_orders: int | None = None
        sub_id_fills: int | None = None

        try:
            sub_id_orders = info.subscribe(order_sub, self._on_order_update)
            sub_id_fills = info.subscribe(fill_sub, self._on_fill_update)
            logger.info(
                "WebSocket subscribed: orderUpdates(#%s) + userFills(#%s)",
                sub_id_orders,
                sub_id_fills,
            )

            # Keep the event loop alive. The ws_manager runs its own
            # background thread for the actual WebSocket connection.
            # We just need to not exit until stop is requested.
            while not self._stop_event.is_set():
                self._last_heartbeat = time.monotonic()
                await asyncio.sleep(1.0)

        except Exception:
            logger.error("WebSocket subscription failed", exc_info=True)
            self._error_count += 1
        finally:
            # Clean unsubscribe
            try:
                if sub_id_orders is not None:
                    info.unsubscribe(order_sub, sub_id_orders)
                if sub_id_fills is not None:
                    info.unsubscribe(fill_sub, sub_id_fills)
            except Exception:
                pass
            try:
                info.disconnect_websocket()
            except Exception:
                pass

    # ─── WebSocket Callbacks ─────────────────────────────────────────────

    def _on_order_update(self, data: Any) -> None:
        """Callback for orderUpdates subscription."""
        try:
            event = self._parse_order_update(data)
            if event and event.is_significant:
                # Sync to PositionTracker first (so LLM sees latest state)
                self._sync_to_tracker(event)
                self._enqueue(event)
                logger.debug(
                    "Order update: type=%s oid=%s status=%s fill=%.4f/%.4f",
                    event.event_type,
                    event.order_id,
                    event.status,
                    event.filled_size,
                    event.total_size,
                )
        except Exception:
            logger.error("Failed to parse order update: %s", data, exc_info=True)

    def _on_fill_update(self, data: Any) -> None:
        """Callback for userFills subscription."""
        try:
            event = self._parse_fill_update(data)
            if event and event.is_significant:
                self._sync_to_tracker(event)
                self._enqueue(event)
                logger.debug(
                    "Fill update: oid=%s side=%s px=%.1f sz=%.4f",
                    event.order_id,
                    event.side,
                    event.fill_price,
                    event.filled_size,
                )
        except Exception:
            logger.error("Failed to parse fill update: %s", data, exc_info=True)

    def _sync_to_tracker(self, event: OrderEvent) -> None:
        """Apply the event to the PositionTracker (thread-safe)."""
        if self.tracker is None:
            return
        try:
            changed = self.tracker.apply_ws_event(event)
            if changed:
                logger.info("WS → tracker synced: %s", changed)
        except Exception:
            logger.error("Failed to sync event to tracker", exc_info=True)

    def _enqueue(self, event: OrderEvent) -> None:
        """Push event to queue, dropping oldest if full (non-blocking)."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                logger.warning("Event queue full — dropped event")

    # ─── Parsing ─────────────────────────────────────────────────────────

    def _parse_order_update(self, data: Any) -> OrderEvent | None:
        """Normalize a raw orderUpdate payload into an OrderEvent.

        Hyperliquid orderUpdates contain one or more order status entries.
        Each entry has: order -> {oid, coin, side, sz, limitPx, orderType},
        status (e.g. "filled", "open", "canceled", "rejected"), and
        statusTimestamp.
        """
        if not isinstance(data, dict):
            return None

        # Sentinel check
        if data.get("_sentinel"):
            return None

        # The WebSocket may batch multiple updates in a list under 'data'
        # or send single-order dicts directly.
        entries: list[dict] = []
        if "data" in data and isinstance(data["data"], list):
            entries = data["data"]
        else:
            entries = [data]

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            order = entry.get("order", {})
            status = entry.get("status", "").lower()
            coin = order.get("coin", "")
            oid = order.get("oid")
            sz = float(order.get("sz") or 0)
            side = "buy" if order.get("side", "") == "B" else "sell"

            # Determine event type from status
            event_type = EventType.UNKNOWN
            filled_sz = 0.0

            if status == "filled":
                event_type = EventType.ORDER_FILLED
                filled_sz = sz  # fully filled
            elif status == "open":
                # Check if partial fill
                orig_sz = float(order.get("origSz") or sz)
                if orig_sz > sz > 0:
                    event_type = EventType.ORDER_PARTIAL
                    filled_sz = orig_sz - sz
                else:
                    # Just an open order — not significant enough to report
                    continue
            elif status in ("canceled", "cancelled"):
                event_type = EventType.ORDER_CANCELLED
            elif status == "rejected":
                event_type = EventType.ORDER_REJECTED
            else:
                continue

            return OrderEvent(
                event_type=event_type,
                coin=coin,
                side=side,
                order_id=int(oid) if oid is not None else None,
                order_type=str(order.get("orderType", "")),
                filled_size=filled_sz,
                total_size=sz if filled_sz > 0 else sz,
                fill_price=float(order.get("limitPx") or 0),
                remaining_size=sz,
                status=status,
                raw=entry,
            )

        return None

    def _parse_fill_update(self, data: Any) -> OrderEvent | None:
        """Normalize a raw userFills payload into an OrderEvent.

        Hyperliquid userFills contain fill details: oid, coin, px, sz, side.
        These are more detailed than orderUpdates — they have the exact fill
        price and size for each individual fill.
        """
        if not isinstance(data, dict):
            return None

        if data.get("_sentinel"):
            return None

        entries: list[dict] = []
        if "data" in data and isinstance(data["data"], list):
            entries = data["data"]
        else:
            entries = [data]

        for entry in entries:
            if not isinstance(entry, dict) or "coin" not in entry:
                continue

            oid = entry.get("oid")
            px = float(entry.get("px") or 0)
            sz = float(entry.get("sz") or 0)
            side = "buy" if entry.get("side", "") == "B" else "sell"
            coin = entry.get("coin", "")

            return OrderEvent(
                event_type=EventType.ORDER_FILLED,
                coin=coin,
                side=side,
                order_id=int(oid) if oid is not None else None,
                order_type="",  # not available in fill data
                filled_size=sz,
                total_size=sz,
                fill_price=px,
                remaining_size=0.0,
                status="filled",
                raw=entry,
            )

        return None


# ─── Flash Reporter ─────────────────────────────────────────────────────────


# System prompt for the Flash reporter agent.
# Kept minimal — the model only needs to format, not analyze.
FLASH_SYSTEM_PROMPT = """\
You are a trading assistant that reports order status changes. \
Given an order event, produce a concise, one-line notification in Chinese.

Rules:
- Be specific: include side (多/空), size (BTC), price ($), and what happened.
- Use these emoji prefixes by event type:
  - order_filled → ✅ (entry filled) or 🎯 (take profit) or 🛑 (stop loss)
  - order_partial → ⏳
  - order_cancelled → ❌
  - order_rejected → 🚫
  - position_closed → 🏁
  - liquidated → 💀
- If you can infer SL/TP from context, mention it.
- Keep it under 120 characters — it's a push notification.
- Output ONLY the notification text, no markdown, no explanation.
"""


def _build_flash_prompt(event: OrderEvent) -> str:
    """Build a compact prompt for the Flash model from an OrderEvent."""
    pct = f" ({event.fill_pct:.0f}%)" if event.total_size > 0 else ""
    return (
        f"Event: {event.event_type}\n"
        f"Side: {event.side} | Coin: {event.coin}\n"
        f"Size: {event.filled_size:.4f}/{event.total_size:.4f} BTC{pct}\n"
        f"Price: ${event.fill_price:.1f}\n"
        f"Order ID: {event.order_id}\n"
        f"Status: {event.status}\n"
    )


class FlashReporter:
    """Consumes OrderEvents from the monitor queue and reports via Flash LLM.

    Runs in its own daemon thread. Falls back to plain-text formatting
    if the LLM is unavailable or errors out.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._queue = event_queue
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._llm = None  # lazy init in thread
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._error_count = 0
        self._report_count = 0

    @property
    def report_count(self) -> int:
        return self._report_count

    def start(self) -> None:
        """Start the reporter thread."""
        if self._started:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="flash-reporter",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info(
            "FlashReporter started (model=%s, llm=%s)",
            self._model,
            "enabled" if self._has_creds() else "fallback-only",
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the reporter to stop."""
        if not self._started:
            return

        self._stop_event.set()
        # Unblock the queue get
        try:
            self._queue.put_nowait(
                OrderEvent(event_type=EventType.UNKNOWN, raw={"_sentinel": True})
            )
        except queue.Full:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._started = False
        logger.info(
            "FlashReporter stopped (reports=%d, errors=%d)",
            self._report_count,
            self._error_count,
        )

    def _has_creds(self) -> bool:
        return bool(self._api_key)

    def _run_loop(self) -> None:
        """Main reporting loop."""
        # Lazy-init LLM (must be in thread context for some SDKs)
        if self._has_creds():
            try:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    model=self._model,
                    temperature=0.0,
                    max_tokens=128,
                )
                logger.info("FlashReporter LLM ready: %s", self._model)
            except Exception as e:
                logger.warning("Flash LLM init failed, using fallback: %s", e)
                self._llm = None

        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event.raw.get("_sentinel"):
                continue

            if not event.is_significant:
                continue

            try:
                text = self._generate_report(event)
                if text:
                    self._send(text)
                    self._report_count += 1
            except Exception:
                logger.error("Failed to process event: %s", event, exc_info=True)
                self._error_count += 1

    def _generate_report(self, event: OrderEvent) -> str:
        """Generate a notification text for the event.

        Tries Flash LLM first, falls back to deterministic formatting.
        """
        # Try LLM
        if self._llm is not None:
            try:
                prompt = _build_flash_prompt(event)
                response = self._llm.invoke(
                    [
                        ("system", FLASH_SYSTEM_PROMPT),
                        ("user", prompt),
                    ]
                )
                text = response.content.strip() if hasattr(response, "content") else str(response).strip()
                if text and len(text) <= 200:
                    return text
                # If LLM returned something too long, truncate
                if text:
                    return text[:197] + "..."
            except Exception as e:
                logger.warning("Flash LLM call failed, using fallback: %s", e)
                self._llm = None  # disable LLM after first failure to avoid spam

        # Fallback: deterministic formatting
        return self._format_fallback(event)

    @staticmethod
    def _format_fallback(event: OrderEvent) -> str:
        """Deterministic fallback formatting when Flash LLM is unavailable."""
        oid_str = f"#{event.order_id}" if event.order_id else ""
        side_label = "多" if event.side == "buy" else "空"

        if event.event_type == EventType.ORDER_FILLED:
            if event.fill_pct >= 99:
                return (
                    f"✅ 订单成交 {oid_str}\n"
                    f"{side_label} {event.filled_size:.4f} BTC @ ${event.fill_price:.1f}"
                )
            else:
                return (
                    f"⏳ 部分成交 {oid_str}\n"
                    f"{side_label} {event.filled_size:.4f}/{event.total_size:.4f} BTC @ ${event.fill_price:.1f}"
                )
        elif event.event_type == EventType.ORDER_PARTIAL:
            return (
                f"⏳ 部分成交 {oid_str}\n"
                f"{side_label} {event.filled_size:.4f}/{event.total_size:.4f} BTC "
                f"({event.fill_pct:.0f}%) @ ${event.fill_price:.1f}"
            )
        elif event.event_type == EventType.ORDER_CANCELLED:
            return f"❌ 订单已取消 {oid_str}"
        elif event.event_type == EventType.ORDER_REJECTED:
            return f"🚫 订单被拒 {oid_str}"
        elif event.event_type == EventType.POSITION_CLOSED:
            return (
                f"🏁 仓位已平 {oid_str}\n"
                f"{side_label} {event.filled_size:.4f} BTC @ ${event.fill_price:.1f}"
            )
        elif event.event_type == EventType.LIQUIDATED:
            return f"💀 仓位被清算 {oid_str}\n{event.filled_size:.4f} BTC @ ${event.fill_price:.1f}"
        else:
            return f"📢 {event.event_type} {oid_str}"

    def _send(self, text: str) -> None:
        """Send notification through the existing Notifier."""
        from kimi_quant.notify import notify

        try:
            notify.send(text, priority="high")
            logger.info("FlashReporter sent: %s", text[:80])
        except Exception:
            logger.error("Failed to send notification", exc_info=True)
