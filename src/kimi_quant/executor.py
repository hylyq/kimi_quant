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
import time
from dataclasses import dataclass, field
from typing import Any

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


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

    # Order IDs
    entry_oid: int | None = None
    sl_oid: int | None = None
    tp_oid: int | None = None

    # How many cycles has the limit order been resting?
    resting_cycles: int = 0
    max_resting_cycles: int = 3  # cancel limit order after this many unfilled cycles

    def has_position(self) -> bool:
        return self.state == "active"

    def has_resting_order(self) -> bool:
        return self.state == "resting"

    def clear(self) -> None:
        """Reset tracker after position is closed."""
        self.side = "none"
        self.size = 0.0
        self.entry_price = 0.0
        self.state = "none"
        self.entry_oid = None
        self.sl_oid = None
        self.tp_oid = None
        self.resting_cycles = 0

    def update_from_open(
        self,
        side: str,
        size: float,
        entry_price: float | None,
        oids: dict[str, int | None],
    ) -> None:
        """Record a new position/order intent."""
        self.coin = config.trading_pair
        self.side = side
        self.size = size
        self.entry_price = entry_price or 0.0
        self.entry_oid = oids.get("entry")
        self.sl_oid = oids.get("sl")
        self.tp_oid = oids.get("tp")
        # Start as "resting" — will be promoted to "active" when chain confirms
        self.state = "resting"
        self.resting_cycles = 0

    def confirm_active(self, chain_entry_price: float, chain_size: float) -> None:
        """Chain confirms position exists — promote to active."""
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
        """Increment resting counter. Returns True if order should be cancelled."""
        if self.state == "resting":
            self.resting_cycles += 1

    def should_cancel_resting(self) -> bool:
        return self.state == "resting" and self.resting_cycles >= self.max_resting_cycles

    def to_summary(self) -> str:
        if self.state == "none":
            return "No position"
        state_label = "RESTING" if self.state == "resting" else "ACTIVE"
        return (
            f"[{state_label}] {self.side.upper()} {self.size:.4f} {self.coin} "
            f"@ ${self.entry_price:.1f} "
            f"(entry_oid={self.entry_oid}, sl_oid={self.sl_oid}, tp_oid={self.tp_oid})"
        )


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
            logger.info(
                "TradeExecutor initialized (address=%s, testnet=%s)",
                self.address,
                config.hl_testnet,
            )

            # Recover state from chain
            self._recover_state()

    # ─── Startup Recovery ────────────────────────────────────────────────

    def _recover_state(self) -> None:
        """Rebuild tracker from on-chain position and open orders.

        If the bot restarted while holding a position, this ensures we
        know about it and can manage (modify SL, close, etc.).
        """
        if self.dry_run:
            return

        try:
            # 1. Check for existing position
            user_state = self.info.user_state(self.address)
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

            # 2. Check for open orders (to recover SL/TP oids)
            open_orders = self.info.open_orders(self.address)
            for o in open_orders:
                if o["coin"] == self.coin:
                    oid = int(o["oid"])
                    is_trigger = "triggerPx" in str(o.get("orderType", ""))
                    is_tpsl = o.get("orderType", "") == "trigger"
                    # Hyperliquid doesn't label tpsl directly; we infer
                    limit_px = float(o.get("limitPx", 0))

                    if self.tracker.state == "active":
                        # Infer SL vs TP from price relative to entry
                        if limit_px < self.tracker.entry_price:
                            if self.tracker.side == "long":
                                self.tracker.sl_oid = oid  # below entry for long
                            else:
                                self.tracker.tp_oid = oid  # below entry for short
                        else:
                            if self.tracker.side == "long":
                                self.tracker.tp_oid = oid  # above entry for long
                            else:
                                self.tracker.sl_oid = oid  # above entry for short
                        logger.info(
                            "Recovered order #%d (price=$%.1f)", oid, limit_px
                        )

            if self.tracker.has_position():
                logger.info("Startup recovery: %s", self.tracker.to_summary())
            else:
                logger.info("Startup recovery: no active position found")

        except Exception as e:
            logger.warning("Startup recovery failed (non-fatal): %s", e)

    # ─── Main entry point ────────────────────────────────────────────────

    def execute(self, signal: TradingSignal) -> dict:
        """Execute a trading signal. Returns result with position summary."""
        result: dict = {}

        # Pre-check: if we have a resting limit order that's unfilled,
        # don't place new orders — check chain state first
        if signal.action in ("LONG", "SHORT") and self.tracker.has_resting_order():
            logger.warning(
                "Limit order #%d still resting — skip new entry",
                self.tracker.entry_oid,
            )
            return {
                "action": signal.action,
                "executed": False,
                "reason": "Previous limit order still resting",
                "position": self.tracker.to_summary(),
            }

        if signal.action == "HOLD":
            logger.info("HOLD — no action taken")
            result = {"action": "HOLD", "executed": False,
                      "reason": signal.reasoning}

        elif signal.action == "CLOSE":
            result = self._close_position()

        elif signal.action == "MODIFY_SL":
            result = self._handle_modify_sl(signal)

        elif signal.action in ("LONG", "SHORT"):
            result = self._open_position(
                signal, is_buy=(signal.action == "LONG")
            )

        else:
            result = {"action": signal.action, "executed": False,
                      "reason": f"Unknown action: {signal.action}"}

        result["position"] = self.tracker.to_summary()
        return result

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
            self.tracker.update_from_open(side, size, signal.entry_price, {})
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
                    orders, grouping="positionTpsl"
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

            oids = _parse_oids_from_result(result, num_orders)
            self.tracker.update_from_open(side, size, signal.entry_price, oids)

            # Determine if market order (should fill instantly) vs limit
            is_market = signal.entry_price is None
            if is_market:
                # Market/Ioc orders fill immediately — verify on next cycle
                logger.info("Market order placed; will confirm fill next cycle")
            else:
                logger.info(
                    "Limit order placed @ $%.1f; will monitor for fill",
                    signal.entry_price,
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

        For market orders: uses Ioc with a wide limit price (mid_price ± 2%)
        to ensure execution while protecting against extreme slippage.
        For limit orders: uses Gtc at the specified price.
        """
        orders: list[dict] = []

        # Order 0: Open position
        if signal.entry_price:
            # Limit order
            orders.append({
                "coin": self.coin,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": signal.entry_price,
                "order_type": {"limit": {"tif": "Gtc"}},
                "reduce_only": False,
            })
        else:
            # Market order via Ioc with a wide price bound
            # Get current mid for a reasonable limit_px
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
                "limit_px": round(px, 1),
                "order_type": {"limit": {"tif": "Ioc"}},
                "reduce_only": False,
            })

        # Order 1: Stop Loss (trigger, market execution)
        if signal.stop_loss:
            orders.append({
                "coin": self.coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": signal.stop_loss,
                "order_type": {
                    "trigger": {
                        "triggerPx": signal.stop_loss,
                        "isMarket": True,
                        "tpsl": "sl",
                    }
                },
                "reduce_only": True,
            })

        # Order 2: Take Profit (trigger, market execution)
        if signal.take_profit:
            orders.append({
                "coin": self.coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": signal.take_profit,
                "order_type": {
                    "trigger": {
                        "triggerPx": signal.take_profit,
                        "isMarket": True,
                        "tpsl": "tp",
                    }
                },
                "reduce_only": True,
            })

        logger.info(
            "Built %d orders: %s @%s %sSL %sTP",
            len(orders),
            "LIMIT" if signal.entry_price else "MARKET(Ioc)",
            f"${signal.entry_price:.0f}" if signal.entry_price else "market",
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
        return self.modify_order(
            oid=oid, is_buy=is_buy, size=size, limit_px=new_price,
            order_type={"trigger": {"triggerPx": new_price, "isMarket": True, "tpsl": "sl"}},
            reduce_only=True,
        )

    def modify_take_profit(
        self, oid: int, new_price: float, is_buy: bool, size: float
    ) -> dict:
        """Move take profit to new price."""
        return self.modify_order(
            oid=oid, is_buy=is_buy, size=size, limit_px=new_price,
            order_type={"trigger": {"triggerPx": new_price, "isMarket": True, "tpsl": "tp"}},
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

        if self.dry_run:
            return {
                "action": "MODIFY_SL", "executed": True, "dry_run": True,
                "sl_oid": sl_oid, "new_sl": new_sl,
                "reasoning": signal.reasoning,
            }

        if sl_oid is None:
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": "No tracked SL order ID."}

        logger.info("MODIFY_SL: moving SL #%d to $%.1f", sl_oid, new_sl)
        return self.modify_stop_loss(
            oid=sl_oid, new_price=new_sl,
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
