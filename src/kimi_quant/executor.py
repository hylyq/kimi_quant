"""Trade execution via Hyperliquid Exchange API.

Full life-cycle:
  - Startup recovery from chain state (positions + open orders)
  - Market open/close with proper execution price tracking
  - Limit orders with fill verification per cycle
  - Trigger orders (stop loss, take profit)
  - Bulk orders (atomic open + SL + TP)
  - Order modification with tracked oids
  - Schedule cancel (dead man's switch)
  - Isolated margin management
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

# TLS fingerprint patching for Hyperliquid SDK (shared with data.py).
# Must be imported BEFORE Exchange/Info construction since __init__ calls API.
from kimi_quant.tls import _cf_requests as _cf_requests_exec  # noqa: F401

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _round_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the nearest valid tick size.

    Hyperliquid requires all prices (limit_px, triggerPx) to be divisible
    by the asset's tick size. BTC typically uses 0.5.

    Two-step rounding prevents floating-point drift:
      1. Snap to nearest tick multiple (e.g. 65906.8 → 65907.0 for tick=0.5)
      2. Re-round to tick's decimal precision to prevent IEEE 754 drift
         (e.g. tick=0.1 → final round to 1 decimal; tick=0.01 → 2 decimals)
    """
    if tick_size <= 0:
        return price

    import math

    # Step 1: snap to nearest tick
    result = round(price / tick_size) * tick_size

    # Step 2: re-round to tick's native decimal precision
    # tick=0.5 → decimals=1, tick=0.1 → decimals=1, tick=0.01 → decimals=2
    decimals = max(0, math.ceil(-math.log10(tick_size)))
    return round(result, decimals)


def _is_trigger_order(order: dict) -> bool:
    """Check whether an open order is a trigger/TPSL order.

    Hyperliquid returns orderType as a dict like {"trigger": {...}} or
    {"limit": {...}} for trigger/TPSL orders. We check for the presence
    of a triggerPx sub-field as the most reliable signal.
    """
    ot = order.get("orderType", "")
    if isinstance(ot, dict):
        # e.g. {"trigger": {"triggerPx": "64200", "isMarket": true, "tpsl": "sl"}}
        return "trigger" in ot
    if isinstance(ot, str):
        return "trigger" in ot.lower()
    return False


# ─── Order State Tracking ────────────────────────────────────────────────


@dataclass
class PositionTracker:
    """Tracks the current position and order IDs across cycles.

    States:
      - "none":    No position, no pending orders
      - "resting": Limit order placed, not yet filled (entry_oid active, no chain pos)
      - "active":  Position confirmed on-chain (with SL/TP oids)
    """

    coin: str = ""
    side: str = "none"  # "long" | "short" | "none"
    size: float = 0.0
    entry_price: float = 0.0
    state: str = "none"  # "none" | "resting" | "active"

    # Order IDs and prices (prices stored so LLM can see current SL/TP levels)
    entry_oid: int | None = None
    sl_oid: int | None = None
    tp_oid: int | None = None
    sl_price: float = 0.0  # current stop loss trigger price
    tp_price: float = 0.0  # current take profit trigger price

    # ── Position Memory (for LLM Step 0 thesis validation) ────────────
    entry_time: str = ""           # ISO timestamp when position was opened
    entry_reason: str = ""         # LLM's reasoning at time of entry
    entry_confidence: float = 0.0  # confidence at time of entry
    peak_favorable: float = 0.0    # best uPNL (positive = in profit, tracking MAE)
    peak_adverse: float = 0.0      # worst uPNL (tracking MFE — how far against)

    # How many cycles has the limit order been resting?
    resting_cycles: int = 0
    max_resting_cycles: int = 3  # cancel limit order after this many unfilled cycles

    # Thread safety: the main loop and the WebSocket monitor both access
    # tracker state. All public mutations must hold this lock.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    # ─── Read-only helpers ───────────────────────────────────────────

    def has_position(self) -> bool:
        return self.state == "active"

    def has_resting_order(self) -> bool:
        return self.state == "resting"

    def should_cancel_resting(self) -> bool:
        return self.state == "resting" and self.resting_cycles >= self.max_resting_cycles

    # ─── Mutations (thread-safe via _lock) ───────────────────────────

    def clear(self) -> None:
        """Reset tracker after position is closed."""
        with self._lock:
            self.side = "none"
            self.size = 0.0
            self.entry_price = 0.0
            self.state = "none"
            self.entry_oid = None
            self.sl_oid = None
            self.tp_oid = None
            self.sl_price = 0.0
            self.tp_price = 0.0
            self.resting_cycles = 0
            # Reset position memory
            self.entry_time = ""
            self.entry_reason = ""
            self.entry_confidence = 0.0
            self.peak_favorable = 0.0
            self.peak_adverse = 0.0

    def update_from_open(
        self,
        side: str,
        size: float,
        entry_price: float | None,
        oids: dict[str, int | None],
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        entry_reason: str = "",
        entry_confidence: float = 0.0,
    ) -> None:
        """Record a new position/order intent with entry thesis for Step 0 validation."""
        from datetime import datetime, timezone

        with self._lock:
            self.coin = config.trading_pair
            self.side = side
            self.size = size
            self.entry_price = entry_price or 0.0
            self.entry_oid = oids.get("entry")
            self.sl_oid = oids.get("sl")
            self.tp_oid = oids.get("tp")
            self.sl_price = sl_price
            self.tp_price = tp_price
            # Start as "resting" — will be promoted to "active" when chain confirms
            self.state = "resting"
            self.resting_cycles = 0
            # Store entry thesis for future cycle Step 0 validation
            self.entry_time = datetime.now(timezone.utc).isoformat()
            self.entry_reason = entry_reason
            self.entry_confidence = entry_confidence
            self.peak_favorable = 0.0
            self.peak_adverse = 0.0

    def confirm_active(self, chain_entry_price: float, chain_size: float) -> None:
        """Chain confirms position exists — promote to active."""
        with self._lock:
            self.state = "active"
            self.resting_cycles = 0
            if chain_entry_price > 0:
                self.entry_price = chain_entry_price
            if chain_size > 0:
                self.size = chain_size

    def cancel_resting(self) -> None:
        """Limit order timed out — clear the tracker."""
        logger.warning(
            "Limit order #%d unfilled after %d cycles — cancelling intent",
            self.entry_oid, self.resting_cycles,
        )
        self.clear()

    def tick_resting(self) -> None:
        """Increment resting counter."""
        with self._lock:
            if self.state == "resting":
                self.resting_cycles += 1

    def update_pnl_extremes(self, unrealized_pnl: float) -> None:
        """Track best/worst uPNL seen during this position's life (MAE/MFE).

        Called each cycle from main loop so the LLM can see how far the
        position has gone in its favor (MFE) and against it (MAE).
        """
        with self._lock:
            if self.state != "active":
                return
            if unrealized_pnl > self.peak_favorable:
                self.peak_favorable = unrealized_pnl
            if unrealized_pnl < self.peak_adverse:
                self.peak_adverse = unrealized_pnl

    # ─── WebSocket Event Sync ─────────────────────────────────────────

    def apply_ws_event(self, event: Any) -> str:
        """Apply a WebSocket order/fill event to tracker state. Thread-safe.

        Called from the OrderMonitor's WebSocket callbacks (background thread).
        Matches the event's oid against tracked entry/sl/tp oids and updates
        state accordingly.

        Returns a short description of what changed (empty string = no change).
        """
        # Lazy import to avoid circular dependency at module level
        from kimi_quant.monitor import EventType

        with self._lock:
            oid = event.order_id
            etype = event.event_type

            if etype == EventType.ORDER_FILLED:
                # Entry order filled → promote to active
                if oid is not None and oid == self.entry_oid and self.state == "resting":
                    self.state = "active"
                    self.resting_cycles = 0
                    if event.fill_price > 0:
                        self.entry_price = event.fill_price
                    if event.filled_size > 0:
                        self.size = event.filled_size
                    logger.info(
                        "WS sync: entry #%d filled @ %.1f (state: resting→active)",
                        oid, event.fill_price,
                    )
                    return f"entry_filled oid={oid}"

                # SL filled → position closed
                if oid is not None and oid == self.sl_oid:
                    logger.info(
                        "WS sync: stop loss #%d filled @ %.1f — position closed",
                        oid, event.fill_price,
                    )
                    self.clear()
                    return f"sl_filled oid={oid}"

                # TP filled → position closed
                if oid is not None and oid == self.tp_oid:
                    logger.info(
                        "WS sync: take profit #%d filled @ %.1f — position closed",
                        oid, event.fill_price,
                    )
                    self.clear()
                    return f"tp_filled oid={oid}"

                # Unknown oid — log for visibility
                if oid is not None:
                    logger.debug(
                        "WS sync: untracked fill oid=%d (tracked: entry=%s, sl=%s, tp=%s)",
                        oid, self.entry_oid, self.sl_oid, self.tp_oid,
                    )

            elif etype == EventType.ORDER_PARTIAL:
                if oid is not None and oid == self.entry_oid and self.state == "resting":
                    logger.info(
                        "WS sync: entry #%d partial fill %.4f/%.4f",
                        oid, event.filled_size, event.total_size,
                    )
                    return f"entry_partial oid={oid}"

            elif etype == EventType.ORDER_CANCELLED:
                if oid is not None and oid == self.entry_oid:
                    logger.info("WS sync: entry #%d cancelled", oid)
                    self.clear()
                    return f"entry_cancelled oid={oid}"
                if oid is not None and oid == self.sl_oid:
                    logger.info("WS sync: SL #%d cancelled", oid)
                    self.sl_oid = None
                    self.sl_price = 0.0
                    return f"sl_cancelled oid={oid}"
                if oid is not None and oid == self.tp_oid:
                    logger.info("WS sync: TP #%d cancelled", oid)
                    self.tp_oid = None
                    self.tp_price = 0.0
                    return f"tp_cancelled oid={oid}"

            elif etype == EventType.ORDER_REJECTED:
                if oid is not None and oid == self.entry_oid:
                    logger.warning("WS sync: entry #%d rejected — clearing tracker", oid)
                    self.clear()
                    return f"entry_rejected oid={oid}"

            return ""  # no tracked state change

    # ─── Display ──────────────────────────────────────────────────────

    def to_summary(self) -> str:
        if self.state == "none":
            return "No position"
        state_label = "RESTING" if self.state == "resting" else "ACTIVE"
        sl_info = f"${self.sl_price:.0f}" if self.sl_price else "NONE"
        tp_info = f"${self.tp_price:.0f}" if self.tp_price else "NONE"
        return (
            f"[{state_label}] {self.side.upper()} {self.size:.4f} {self.coin} "
            f"@ ${self.entry_price:.1f} SL={sl_info} TP={tp_info} "
            f"(entry_oid={self.entry_oid}, sl_oid={self.sl_oid}, tp_oid={self.tp_oid})"
        )

    def to_orders_summary(self) -> str:
        """Generate a compact summary of open orders for LLM context.

        Returns empty string if no open orders (no position and no resting).
        """
        if self.state == "none":
            return ""

        parts = []
        if self.entry_oid is not None:
            order_type = "LIMIT" if self.state == "resting" else "filled (position active)"
            parts.append(f"Entry: oid={self.entry_oid} ({order_type})")

        if self.sl_price and self.sl_oid is not None:
            parts.append(f"Stop Loss: ${self.sl_price:.0f} (oid={self.sl_oid})")
        elif self.sl_oid is not None:
            parts.append(f"Stop Loss: oid={self.sl_oid} (price unknown)")
        else:
            parts.append("Stop Loss: NOT SET ⚠️")

        if self.tp_price and self.tp_oid is not None:
            parts.append(f"Take Profit: ${self.tp_price:.0f} (oid={self.tp_oid})")
        elif self.tp_oid is not None:
            parts.append(f"Take Profit: oid={self.tp_oid} (price unknown)")
        else:
            parts.append("Take Profit: NOT SET")

        return (
            f"Open orders for {self.side.upper()} {self.size:.4f} {self.coin}: "
            + ", ".join(parts)
        )

    def to_position_memory(self, unrealized_pnl: float = 0.0) -> str:
        """Build the Position Memory section for the LLM prompt.

        Gives the LLM full context for Step 0 thesis validation:
        why the position was opened, how long it's been held, and
        how far price has moved in favor/against.

        Returns empty string if no active position with memory.
        """
        if self.state != "active" or not self.entry_time or not self.entry_reason:
            return ""

        from datetime import datetime, timezone

        try:
            opened_at = datetime.fromisoformat(self.entry_time)
            elapsed = datetime.now(timezone.utc) - opened_at.replace(
                tzinfo=timezone.utc
            )
            if elapsed.total_seconds() < 60:
                duration = f"{elapsed.total_seconds():.0f}s"
            elif elapsed.total_seconds() < 3600:
                duration = f"{elapsed.total_seconds() / 60:.0f}min"
            else:
                hours = elapsed.total_seconds() / 3600
                duration = f"{hours:.1f}h"
        except (ValueError, TypeError):
            duration = "unknown"

        notional = self.entry_price * self.size if self.entry_price > 0 else 0
        u_pnl_pct = (
            (unrealized_pnl / notional * 100) if notional > 0 else 0.0
        )

        lines = [
            "# 📌 Position Memory",
            f"Holding: {self.side.upper()} {self.size:.4f} {self.coin} @ ${self.entry_price:.1f}",
            f"Opened: {duration} ago | Entry confidence: {self.entry_confidence:.2f}",
            f"Entry thesis: \"{self.entry_reason[:200]}\"",
            f"Current uPNL: ${unrealized_pnl:+.2f} ({u_pnl_pct:+.2f}%)",
        ]
        if self.peak_favorable > 0 or self.peak_adverse < 0:
            lines.append(
                f"Best: ${self.peak_favorable:+.2f} | Worst: ${self.peak_adverse:+.2f}"
            )
        lines.append(
            "\n⚠️  Step 0 — THESIS VALIDATION: "
            "Has the original entry thesis held? "
            "If the market has negated this thesis, CLOSE the position. "
            "If it's working, consider MODIFY_SL to lock in profit."
        )

        return "\n".join(lines)


def _extract_errors(result: Any, num_orders: int) -> list[str]:
    """Check a bulk_orders response for per-order errors.

    The top-level may say 'ok' while individual statuses contain errors.
    Returns list of error messages (empty if all orders succeeded).
    """
    errors: list[str] = []
    try:
        statuses = None
        if isinstance(result, dict):
            statuses = (
                result.get("response", {})
                .get("data", {})
                .get("statuses", [])
            )
        elif isinstance(result, list):
            statuses = result

        if statuses:
            for s in statuses:
                if isinstance(s, dict) and "error" in s:
                    errors.append(s["error"])
    except Exception:
        pass
    return errors


def _parse_oids_from_result(result: Any, num_orders: int) -> dict[str, int | None]:
    """Extract order IDs from a bulk_orders or order response."""
    oids: dict[str, int | None] = {"entry": None, "sl": None, "tp": None}
    keys = ["entry", "sl", "tp"]

    try:
        statuses = None
        if isinstance(result, dict):
            statuses = (
                result.get("response", {})
                .get("data", {})
                .get("statuses", [])
            )
        elif isinstance(result, list):
            statuses = result

        if statuses:
            for i in range(min(len(statuses), num_orders, len(keys))):
                s = statuses[i]
                if isinstance(s, dict):
                    resting = s.get("resting", {})
                    filled = s.get("filled", {})
                    oid = resting.get("oid") or filled.get("oid")
                    oids[keys[i]] = int(oid) if oid is not None else None
    except Exception as e:
        logger.warning("Failed to parse oids from result: %s", e)

    logger.info("Parsed oids: %s", oids)
    return oids


# ─── Trade Executor ──────────────────────────────────────────────────────


class TradeExecutor:
    """Executes trades with chain-state-aware position tracking.

    Startup recovery: queries on-chain positions and open orders to
    rebuild the tracker — survives crashes and restarts.
    """

    def __init__(self):
        self.dry_run = config.dry_run
        self.coin = config.trading_pair
        self.tracker = PositionTracker()

        if self.dry_run:
            self.exchange = None
            self.info = None
            self.address = "0x_dry_run"
            self._tick_size = 0.5  # BTC default for dry-run
            logger.info("TradeExecutor initialized in DRY RUN mode")
        else:
            if not config.hl_private_key:
                raise ValueError("Private key required for live trading")

            account: LocalAccount = Account.from_key(config.hl_private_key)
            base_url = (
                "https://api.hyperliquid-testnet.xyz"
                if config.hl_testnet
                else config.hl_base_url
            )
            self.exchange = Exchange(wallet=account, base_url=base_url)
            self.info = Info(base_url=base_url, skip_ws=True)
            self.address = account.address

            # Fetch tick size for price validation BEFORE any order is built
            self._tick_size = self._fetch_tick_size()

            logger.info(
                "TradeExecutor initialized (address=%s, testnet=%s, curl_cffi=%s, "
                "tick=%.1f)",
                self.address,
                config.hl_testnet,
                _cf_requests_exec is not None,
                self._tick_size,
            )

            # Recover state from chain
            self._recover_state()

    # ─── Startup Recovery ────────────────────────────────────────────────

    def _recover_state(self) -> None:
        """Rebuild tracker from on-chain position and open orders.

        If the bot restarted while holding a position, this ensures we
        know about it and can manage (modify SL, close, etc.).

        Position recovery and order recovery are independent: failure
        in open_orders does not prevent position recovery from user_state.
        """
        if self.dry_run:
            return

        from kimi_quant.data import retry_api_call

        # ── 1. Recover position from user_state ──────────────────────────
        try:
            user_state = retry_api_call(
                lambda: self.info.user_state(self.address),
                description="user_state (recovery)",
            )
            positions = user_state.get("assetPositions", [])
            for pos in positions:
                if pos["position"]["coin"] == self.coin:
                    p = pos["position"]
                    size = float(p.get("szi", 0))
                    if abs(size) > 0:
                        side = "long" if size > 0 else "short"
                        size = abs(size)
                        entry_px = float(p.get("entryPx", 0))
                        self.tracker.side = side
                        self.tracker.size = size
                        self.tracker.entry_price = entry_px
                        self.tracker.state = "active"
                        self.tracker.coin = self.coin
                        logger.info(
                            "Recovered position: %s %.4f @ $%.1f",
                            side.upper(), size, entry_px,
                        )
        except Exception as e:
            logger.warning(
                "Startup position recovery failed (non-fatal): %s", e
            )

        # ── 2. Recover SL/TP orders from open_orders ─────────────────────
        try:
            open_orders = retry_api_call(
                lambda: self.info.open_orders(self.address),
                description="open_orders (recovery)",
            )
            for o in open_orders:
                if o["coin"] == self.coin:
                    oid = int(o["oid"])
                    limit_px = float(o.get("limitPx", 0))

                    # Only trigger/TPSL orders are SL/TP candidates.
                    # Regular limit orders (entry, take-profit limit) should
                    # not be misclassified as stop-loss or take-profit.
                    if not _is_trigger_order(o):
                        if self.tracker.state != "active":
                            logger.info(
                                "Recovered resting order #%d (price=$%.1f) "
                                "— no position, may be a pending entry",
                                oid, limit_px,
                            )
                        continue

                    if self.tracker.state == "active":
                        # Infer SL vs TP from price relative to entry
                        if limit_px < self.tracker.entry_price:
                            if self.tracker.side == "long":
                                self.tracker.sl_oid = oid
                                self.tracker.sl_price = limit_px
                            else:
                                self.tracker.tp_oid = oid
                                self.tracker.tp_price = limit_px
                        else:
                            if self.tracker.side == "long":
                                self.tracker.tp_oid = oid
                                self.tracker.tp_price = limit_px
                            else:
                                self.tracker.sl_oid = oid
                                self.tracker.sl_price = limit_px
                        logger.info(
                            "Recovered SL/TP order #%d (price=$%.1f)", oid, limit_px
                        )
        except Exception as e:
            logger.warning(
                "Startup order recovery failed (non-fatal): %s", e
            )

        if self.tracker.has_position():
            logger.info("Startup recovery: %s", self.tracker.to_summary())
        else:
            logger.info("Startup recovery: no active position found")

    def _fetch_tick_size(self) -> float:
        """Fetch the price tick size for the trading pair from exchange metadata.

        Hyperliquid requires all order prices to be divisible by the
        asset's tick size (pxTick). For BTC this is typically 0.5.

        Falls back to 0.5 if the metadata query fails.
        """
        from kimi_quant.data import retry_api_call as _retry

        try:
            meta, _ = _retry(
                lambda: self.info.meta_and_asset_ctxs(),
                description="meta (tick_size)",
            )
            for asset in meta.get("universe", []):
                if asset.get("name") == self.coin:
                    tick = float(asset.get("pxTick", 0))
                    if tick > 0:
                        logger.info("Tick size for %s: %.1f", self.coin, tick)
                        return tick
        except Exception as e:
            logger.warning(
                "Failed to fetch tick size from exchange (%s) — "
                "falling back to 0.5 for %s",
                e, self.coin,
            )

        logger.info("Tick size for %s: 0.5 (default)", self.coin)
        return 0.5

    # ─── Main entry point ────────────────────────────────────────────────

    def execute(self, signal: TradingSignal) -> dict:
        """Execute a trading signal. Supports multi-action sequences.

        Actions are executed in order. Execution stops on the first
        non-HOLD failure to avoid partial state.
        """
        actions = signal.get_actions()
        results: list[dict] = []

        for i, action in enumerate(actions):
            is_first = (i == 0)

            # Pre-check: if we have a resting limit order that's unfilled,
            # don't place new entry orders — check chain state first.
            # Only applies to the first action (subsequent actions in a flip
            # should proceed since CLOSE already cleared the tracker).
            if is_first and action in ("LONG", "SHORT") and self.tracker.has_resting_order():
                logger.warning(
                    "Limit order #%d still resting — skip new entry",
                    self.tracker.entry_oid,
                )
                return {
                    "action": "/".join(actions),
                    "executed": False,
                    "reason": "Previous limit order still resting",
                    "position": self.tracker.to_summary(),
                }

            result = self._dispatch(action, signal)
            results.append(result)

            # Stop on first failure. HOLD is never "executed" but is not a
            # failure — it's an intentional no-op. Only real failures stop.
            if action != "HOLD" and not result.get("executed"):
                logger.warning(
                    "Action %d/%d (%s) failed — stopping sequence. %s",
                    i + 1, len(actions), action, result.get("reason", ""),
                )
                break

        return self._merge_results(actions, results, signal)

    def _dispatch(self, action: str, signal: TradingSignal) -> dict:
        """Execute a single action from a trading signal."""
        if action == "HOLD":
            logger.info("HOLD — no action taken")
            return {"action": "HOLD", "executed": False,
                    "reason": signal.reasoning}

        elif action == "CLOSE":
            return self._close_position()

        elif action == "MODIFY_SL":
            return self._handle_modify_sl(signal)

        elif action == "MODIFY_TP":
            return self._handle_modify_tp(signal)

        elif action in ("LONG", "SHORT"):
            return self._open_position(
                signal, is_buy=(action == "LONG")
            )

        else:
            return {"action": action, "executed": False,
                    "reason": f"Unknown action: {action}"}

    def _merge_results(
        self, actions: list[str], results: list[dict], signal: TradingSignal
    ) -> dict:
        """Merge individual action results into a single result dict."""
        executed = all(r.get("executed") for r in results)
        action_label = "/".join(actions)

        merged: dict = {
            "action": action_label,
            "executed": executed,
            "actions": actions,
            "results": results,
        }

        # Forward key fields from the last result (price/size info)
        if results:
            last = results[-1]
            for key in ("dry_run", "side", "size", "entry_price",
                        "stop_loss", "take_profit", "oids", "result",
                        "new_sl", "new_tp"):
                if key in last:
                    merged[key] = last[key]

        # If any action failed, include the first failure reason
        if not executed:
            for r in results:
                if not r.get("executed") and r.get("reason"):
                    merged["reason"] = r["reason"]
                    break

        merged["position"] = self.tracker.to_summary()
        return merged

    # ─── Chain State Sync (called from main loop each cycle) ─────────────

    def sync_with_chain(self, chain_side: str, chain_size: float,
                        chain_entry: float) -> None:
        """Sync tracker with on-chain state each cycle.

        Called by main.py's _validate_and_execute before risk checks.
        """
        has_chain_pos = chain_side != "none" and chain_size > 0
        tracker_has = self.tracker.has_position()
        tracker_resting = self.tracker.has_resting_order()

        # Case 1: Resting order → now filled (chain has position but tracker says resting)
        if tracker_resting and has_chain_pos:
            if abs(chain_size - self.tracker.size) > 1e-8:
                logger.info(
                    "Partial fill detected: requested=%.4f, filled=%.4f (%.1f%%)",
                    self.tracker.size, chain_size,
                    (chain_size / self.tracker.size * 100) if self.tracker.size > 0 else 0,
                )
            else:
                logger.info("Limit order filled! Confirming position on-chain.")
            self.tracker.confirm_active(chain_entry, chain_size)

        # Case 2: Resting order → still resting (chain has no position)
        elif tracker_resting and not has_chain_pos:
            self.tracker.tick_resting()
            if self.tracker.should_cancel_resting():
                logger.warning("Limit order timed out — cancelling.")
                if self.tracker.entry_oid and not self.dry_run:
                    try:
                        self.exchange.cancel(self.coin, self.tracker.entry_oid)
                    except Exception:
                        pass
                self.tracker.cancel_resting()
            else:
                logger.info(
                    "Limit order still resting (cycle %d/%d)",
                    self.tracker.resting_cycles,
                    self.tracker.max_resting_cycles,
                )

        # Case 3: Active position → now gone (SL/TP filled or manually closed)
        elif tracker_has and not has_chain_pos:
            logger.warning(
                "Position gone from chain — SL/TP likely filled. Clearing tracker."
            )
            # main.py records the trade via TradeLogger before clearing
            self.tracker.clear()

        # Case 4: Active position → still there (update entry/size from chain)
        elif tracker_has and has_chain_pos:
            if chain_entry > 0:
                self.tracker.entry_price = chain_entry
            if chain_size > 0:
                self.tracker.size = chain_size

        # Case 5: No tracker position but chain has one (recovery)
        elif not tracker_has and not tracker_resting and has_chain_pos:
            logger.warning("Chain has position but tracker doesn't — recovering.")
            self.tracker.side = chain_side
            self.tracker.size = chain_size
            self.tracker.entry_price = chain_entry
            self.tracker.state = "active"
            self.tracker.coin = self.coin

    def verify_tracked_orders(self, open_orders_raw: list[dict]) -> dict:
        """Cross-reference tracker SL/TP oids against chain open orders.

        Called each cycle when there's an active position. Returns a dict
        indicating which tracked orders are missing from the exchange.

        Args:
            open_orders_raw: List of open order dicts from the chain API.

        Returns:
            {"sl_missing": bool, "tp_missing": bool}
        """
        result = {"sl_missing": False, "tp_missing": False}

        if not self.tracker.has_position():
            return result

        # Build set of chain order IDs for this coin
        chain_oids: set[int] = set()
        for o in open_orders_raw:
            if o.get("coin") == self.coin:
                oid = o.get("oid")
                if oid is not None:
                    chain_oids.add(int(oid))

        # Check SL
        if self.tracker.sl_oid is not None and self.tracker.sl_oid not in chain_oids:
            logger.warning(
                "SL order #%d ($%.0f) NOT FOUND on chain! Position is unprotected.",
                self.tracker.sl_oid, self.tracker.sl_price,
            )
            result["sl_missing"] = True

        # Check TP
        if self.tracker.tp_oid is not None and self.tracker.tp_oid not in chain_oids:
            logger.warning(
                "TP order #%d ($%.0f) NOT FOUND on chain!",
                self.tracker.tp_oid, self.tracker.tp_price,
            )
            result["tp_missing"] = True

        if not result["sl_missing"] and not result["tp_missing"]:
            logger.debug(
                "SL/TP verification OK: SL=#%d ($%.0f), TP=#%d ($%.0f) both on chain",
                self.tracker.sl_oid, self.tracker.sl_price,
                self.tracker.tp_oid, self.tracker.tp_price,
            )

        return result

    # ─── Open Position ───────────────────────────────────────────────────

    def _open_position(self, signal: TradingSignal, is_buy: bool) -> dict:
        """Open a position with stop loss and take profit.

        After placing orders, marks tracker as "resting".
        The next cycle's sync_with_chain() will confirm or cancel.
        """
        size = signal.size or config.max_position_size
        side = "long" if is_buy else "short"
        side_label = "BUY" if is_buy else "SELL"

        if self.dry_run:
            logger.info(
                "DRY RUN: %s %.4f %s | SL=$%.1f TP=$%.1f",
                side_label, size, self.coin,
                signal.stop_loss or 0,
                signal.take_profit or 0,
            )
            self.tracker.update_from_open(
                side, size, signal.entry_price, {},
                sl_price=signal.stop_loss or 0.0,
                tp_price=signal.take_profit or 0.0,
                entry_reason=signal.reasoning or "",
                entry_confidence=signal.confidence,
            )
            self.tracker.state = "active"  # in dry-run, pretend filled instantly
            return {
                "action": signal.action,
                "executed": True,
                "dry_run": True,
                "side": side,
                "size": size,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "reasoning": signal.reasoning,
            }

        try:
            # Set leverage first
            self.exchange.update_leverage(config.max_leverage, self.coin)

            orders = self._build_entry_orders(signal, is_buy, size)
            num_orders = len(orders)

            if num_orders > 1:
                result = self.exchange.bulk_orders(
                    orders, grouping="normalTpsl"
                )
                logger.info(
                    "Bulk order placed (%d orders): open + SL + TP", num_orders
                )
            else:
                result = self.exchange.order(
                    name=orders[0]["coin"],
                    is_buy=orders[0]["is_buy"],
                    sz=orders[0]["sz"],
                    limit_px=orders[0]["limit_px"],
                    order_type=orders[0]["order_type"],
                )
                logger.info("Single order placed (no SL/TP)")

            # Check for errors in bulk order response
            errors = _extract_errors(result, num_orders)
            if errors:
                logger.error("Order rejected by exchange: %s", errors)
                return {
                    "action": signal.action,
                    "executed": False,
                    "error": f"Exchange rejected order: {errors}",
                    "reasoning": signal.reasoning,
                }

            oids = _parse_oids_from_result(result, num_orders)
            self.tracker.update_from_open(
                side, size, signal.entry_price, oids,
                sl_price=signal.stop_loss or 0.0,
                tp_price=signal.take_profit or 0.0,
                entry_reason=signal.reasoning or "",
                entry_confidence=signal.confidence,
            )

            # All entry orders are Ioc market orders — fill immediately
            logger.info(
                "Market (Ioc) order placed%s; will confirm fill next cycle",
                f" (ref=${signal.entry_price:.0f})" if signal.entry_price else "",
            )

            return {
                "action": signal.action,
                "executed": True,
                "dry_run": False,
                "side": side,
                "size": size,
                "oids": oids,
                "result": result,
                "reasoning": signal.reasoning,
            }

        except Exception as e:
            logger.error("Order execution failed: %s", e, exc_info=True)
            return {
                "action": signal.action,
                "executed": False,
                "error": str(e),
                "reasoning": signal.reasoning,
            }

    def _build_entry_orders(
        self, signal: TradingSignal, is_buy: bool, size: float
    ) -> list[dict]:
        """Build the order request list for atomic open + SL + TP.

        All entry orders use Ioc (market) with a wide limit price
        (mid_price ± 2%) to ensure execution while protecting against
        extreme slippage. The LLM's entry_price is ignored for order
        construction — it is used only for risk checks and trade logging.
        """
        orders: list[dict] = []

        # Order 0: Open position — always market order via Ioc
        try:
            mids = self.info.all_mids()
            mid = float(mids.get(self.coin, 0))
        except Exception:
            mid = 0
        buffer = mid * 0.02 if mid > 0 else 1000  # 2% buffer
        px = mid + buffer if is_buy else max(mid - buffer, 1)
        orders.append({
            "coin": self.coin,
            "is_buy": is_buy,
            "sz": size,
            "limit_px": _round_to_tick(px, self._tick_size),
            "order_type": {"limit": {"tif": "Ioc"}},
            "reduce_only": False,
        })

        # Order 1: Stop Loss (trigger, market execution)
        if signal.stop_loss:
            sl_px = _round_to_tick(signal.stop_loss, self._tick_size)
            orders.append({
                "coin": self.coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": sl_px,
                "order_type": {
                    "trigger": {
                        "triggerPx": sl_px,
                        "isMarket": True,
                        "tpsl": "sl",
                    }
                },
                "reduce_only": True,
            })

        # Order 2: Take Profit (trigger, market execution)
        if signal.take_profit:
            tp_px = _round_to_tick(signal.take_profit, self._tick_size)
            orders.append({
                "coin": self.coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": tp_px,
                "order_type": {
                    "trigger": {
                        "triggerPx": tp_px,
                        "isMarket": True,
                        "tpsl": "tp",
                    }
                },
                "reduce_only": True,
            })

        logger.info(
            "Built %d orders: MARKET(Ioc) @market %sSL %sTP",
            len(orders),
            f"${signal.stop_loss:.0f} " if signal.stop_loss else "NO ",
            f"${signal.take_profit:.0f}" if signal.take_profit else "NO",
        )
        return orders

    # ─── Close Position ──────────────────────────────────────────────────

    def _close_position(self) -> dict:
        """Close the current position using market_close."""
        if self.dry_run:
            logger.info("DRY RUN: Would close %s position", self.coin)
            self.tracker.clear()
            return {"action": "CLOSE", "executed": True, "dry_run": True}

        try:
            result = self.exchange.market_close(self.coin)
            logger.info("Position closed: %s", result)
            self.tracker.clear()
            return {
                "action": "CLOSE",
                "executed": True,
                "dry_run": False,
                "result": result,
            }
        except Exception as e:
            logger.error("Failed to close position: %s", e)
            return {"action": "CLOSE", "executed": False, "error": str(e)}

    # ─── Cancel Orders ──────────────────────────────────────────────────

    def cancel_order(self, oid: int) -> dict:
        """Cancel a single order by exchange-assigned order ID."""
        if self.dry_run:
            logger.info("DRY RUN: Cancel order #%d", oid)
            return {"action": "cancel", "executed": True, "dry_run": True}

        try:
            result = self.exchange.cancel(self.coin, oid)
            logger.info("Order #%d cancelled", oid)
            if oid == self.tracker.entry_oid:
                self.tracker.entry_oid = None
            elif oid == self.tracker.sl_oid:
                self.tracker.sl_oid = None
            elif oid == self.tracker.tp_oid:
                self.tracker.tp_oid = None
            return {"action": "cancel", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to cancel order #%d: %s", oid, e)
            return {"action": "cancel", "executed": False, "error": str(e)}

    def cancel_by_cloid(self, cloid: str) -> dict:
        """Cancel an order by client-assigned order ID."""
        if self.dry_run:
            logger.info("DRY RUN: Cancel order cloid=%s", cloid)
            return {"action": "cancel", "executed": True, "dry_run": True}

        try:
            result = self.exchange.cancel_by_cloid(self.coin, cloid)
            logger.info("Order cloid=%s cancelled", cloid)
            return {"action": "cancel", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to cancel cloid=%s: %s", cloid, e)
            return {"action": "cancel", "executed": False, "error": str(e)}

    def cancel_by_cloids(self, cloids: list[str]) -> dict:
        """Cancel multiple orders by their client-assigned IDs."""
        if self.dry_run:
            logger.info("DRY RUN: Cancel %d orders by cloid", len(cloids))
            return {"action": "cancel_bulk", "executed": True, "dry_run": True}

        try:
            requests = [{"coin": self.coin, "cloid": c} for c in cloids]
            result = self.exchange.bulk_cancel_by_cloid(requests)
            logger.info("Cancelled %d orders by cloid", len(cloids))
            return {"action": "cancel_bulk", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to cancel by cloids: %s", e)
            return {"action": "cancel_bulk", "executed": False, "error": str(e)}

    def cancel_all_orders(self) -> dict:
        """Cancel ALL open orders for the trading pair."""
        if self.dry_run:
            logger.info("DRY RUN: Cancel all %s orders", self.coin)
            self.tracker.clear()
            return {"action": "cancel_all", "executed": True, "dry_run": True}

        try:
            orders = self.exchange.bulk_cancel(self.coin)
            logger.info("Cancelled %d orders", len(orders) if orders else 0)
            self.tracker.clear()
            return {"action": "cancel_all", "executed": True, "result": orders}
        except Exception as e:
            logger.error("Failed to cancel all orders: %s", e)
            return {"action": "cancel_all", "executed": False, "error": str(e)}

    # ─── Modify Orders ───────────────────────────────────────────────────

    def modify_order(
        self, oid: int, is_buy: bool, size: float, limit_px: float,
        order_type: dict, reduce_only: bool = False,
    ) -> dict:
        """Modify any existing order."""
        if self.dry_run:
            logger.info("DRY RUN: Modify order #%d", oid)
            return {"action": "modify_order", "executed": True, "dry_run": True}

        try:
            result = self.exchange.modify_order(
                oid=oid, name=self.coin, is_buy=is_buy, sz=size,
                limit_px=limit_px, order_type=order_type,
                reduce_only=reduce_only,
            )
            logger.info("Order #%d modified", oid)
            return {"action": "modify_order", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to modify order #%d: %s", oid, e)
            return {"action": "modify_order", "executed": False, "error": str(e)}

    def modify_stop_loss(
        self, oid: int, new_price: float, is_buy: bool, size: float
    ) -> dict:
        """Move stop loss to new price."""
        px = _round_to_tick(new_price, self._tick_size)
        return self.modify_order(
            oid=oid, is_buy=is_buy, size=size, limit_px=px,
            order_type={"trigger": {"triggerPx": px, "isMarket": True, "tpsl": "sl"}},
            reduce_only=True,
        )

    def modify_take_profit(
        self, oid: int, new_price: float, is_buy: bool, size: float
    ) -> dict:
        """Move take profit to new price."""
        px = _round_to_tick(new_price, self._tick_size)
        return self.modify_order(
            oid=oid, is_buy=is_buy, size=size, limit_px=px,
            order_type={"trigger": {"triggerPx": px, "isMarket": True, "tpsl": "tp"}},
            reduce_only=True,
        )

    def modify_orders(self, modifications: list[dict]) -> dict:
        """Modify multiple orders atomically."""
        if not modifications:
            return {"action": "modify_bulk", "executed": False,
                    "reason": "Empty modification list"}
        if self.dry_run:
            logger.info("DRY RUN: Modify %d orders", len(modifications))
            return {"action": "modify_bulk", "executed": True, "dry_run": True}
        try:
            requests = [
                {"oid": m["oid"], "order": {
                    "coin": self.coin, "is_buy": m["order"]["is_buy"],
                    "sz": m["order"]["sz"], "limit_px": m["order"]["limit_px"],
                    "order_type": m["order"]["order_type"],
                    "reduce_only": m["order"].get("reduce_only", False),
                }}
                for m in modifications
            ]
            result = self.exchange.bulk_modify_orders_new(requests)
            logger.info("Modified %d orders in bulk", len(modifications))
            return {"action": "modify_bulk", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to bulk modify orders: %s", e)
            return {"action": "modify_bulk", "executed": False, "error": str(e)}

    # ─── LLM-triggered Stop Loss Modification ────────────────────────────

    def _handle_modify_sl(self, signal: TradingSignal) -> dict:
        """Move stop loss using tracked oid. Requires active position."""
        new_sl = signal.modify_sl_to or signal.stop_loss
        if new_sl is None:
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": "No new stop loss price provided"}

        if not self.tracker.has_position():
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": f"No active position (state={self.tracker.state})"}

        sl_oid = self.tracker.sl_oid
        new_sl = _round_to_tick(new_sl, self._tick_size)

        if self.dry_run:
            self.tracker.sl_price = new_sl
            return {
                "action": "MODIFY_SL", "executed": True, "dry_run": True,
                "sl_oid": sl_oid, "new_sl": new_sl,
                "reasoning": signal.reasoning,
            }

        if sl_oid is None:
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": "No tracked SL order ID."}

        logger.info("MODIFY_SL: moving SL #%d to $%.1f", sl_oid, new_sl)
        self.tracker.sl_price = new_sl  # track the new SL price for LLM visibility
        return self.modify_stop_loss(
            oid=sl_oid, new_price=new_sl,
            is_buy=(self.tracker.side == "short"),
            size=self.tracker.size,
        )

    def _handle_modify_tp(self, signal: TradingSignal) -> dict:
        """Move take profit using tracked oid. Requires active position."""
        new_tp = signal.modify_tp_to or signal.take_profit
        if new_tp is None:
            return {"action": "MODIFY_TP", "executed": False,
                    "reason": "No new take profit price provided"}

        if not self.tracker.has_position():
            return {"action": "MODIFY_TP", "executed": False,
                    "reason": f"No active position (state={self.tracker.state})"}

        tp_oid = self.tracker.tp_oid
        new_tp = _round_to_tick(new_tp, self._tick_size)

        if self.dry_run:
            self.tracker.tp_price = new_tp
            return {
                "action": "MODIFY_TP", "executed": True, "dry_run": True,
                "tp_oid": tp_oid, "new_tp": new_tp,
                "reasoning": signal.reasoning,
            }

        if tp_oid is None:
            return {"action": "MODIFY_TP", "executed": False,
                    "reason": "No tracked TP order ID."}

        logger.info("MODIFY_TP: moving TP #%d to $%.1f", tp_oid, new_tp)
        self.tracker.tp_price = new_tp  # track the new TP price for LLM visibility
        return self.modify_take_profit(
            oid=tp_oid, new_price=new_tp,
            is_buy=(self.tracker.side == "short"),
            size=self.tracker.size,
        )

    # ─── Margin Management ───────────────────────────────────────────────

    def update_margin(self, amount: float) -> dict:
        """Add or remove isolated margin."""
        if self.dry_run:
            logger.info("DRY RUN: Update margin by $%.2f", amount)
            return {"action": "update_margin", "executed": True, "dry_run": True}
        try:
            result = self.exchange.update_isolated_margin(amount, self.coin)
            logger.info("Margin updated by $%.2f", amount)
            return {"action": "update_margin", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to update margin: %s", e)
            return {"action": "update_margin", "executed": False, "error": str(e)}

    # ─── Dead Man's Switch ───────────────────────────────────────────────

    def schedule_cancel(self, timeout_seconds: int = 300) -> dict:
        """Schedule auto-cancel after timeout (dead man's switch)."""
        if self.dry_run:
            logger.info("DRY RUN: Dead man's switch set to %ds", timeout_seconds)
            return {"action": "schedule_cancel", "executed": True, "dry_run": True}

        now_ms = int(time.time() * 1000)
        cancel_time = now_ms + timeout_seconds * 1000
        try:
            result = self.exchange.schedule_cancel(cancel_time)
            logger.info("Dead man's switch: auto-cancel at +%ds", timeout_seconds)
            return {"action": "schedule_cancel", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to set dead man's switch: %s", e)
            return {"action": "schedule_cancel", "executed": False, "error": str(e)}
