"""Trade execution via Hyperliquid Exchange API.

Wraps the full Hyperliquid SDK's order capabilities:
  - Market open/close (with slippage)
  - Limit orders (Gtc, Ioc, Alo)
  - Trigger orders (stop loss, take profit)
  - Bulk orders (atomic open + SL + TP in one request)
  - Order modification (move SL to breakeven, etc.)
  - Schedule cancel (dead man's switch)
  - Isolated margin management
  - Order state tracking (oids preserved across cycles)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


# ─── Order State Tracking ────────────────────────────────────────────────


@dataclass
class PositionTracker:
    """Tracks the current position and related order IDs across cycles.

    This is the bridge between LLM decisions and actual order modification:
      - After _open_position(), oids are parsed from the response and stored
      - When the LLM issues MODIFY_SL, _handle_modify_sl() uses the stored oid
      - On CLOSE or fill events, the tracker is cleared
    """

    coin: str = ""
    side: str = "none"  # "long" | "short" | "none"
    size: float = 0.0
    entry_price: float = 0.0

    # Order IDs from the last open — needed for modification
    entry_oid: int | None = None
    sl_oid: int | None = None
    tp_oid: int | None = None

    def has_position(self) -> bool:
        return self.side != "none" and self.size > 0

    def clear(self) -> None:
        """Reset tracker after position is closed."""
        self.side = "none"
        self.size = 0.0
        self.entry_price = 0.0
        self.entry_oid = None
        self.sl_oid = None
        self.tp_oid = None

    def update_from_open(
        self,
        side: str,
        size: float,
        entry_price: float | None,
        oids: dict[str, int | None],
    ) -> None:
        """Record a new position and its associated order IDs."""
        self.coin = config.trading_pair
        self.side = side
        self.size = size
        self.entry_price = entry_price or 0.0
        self.entry_oid = oids.get("entry")
        self.sl_oid = oids.get("sl")
        self.tp_oid = oids.get("tp")

    def to_summary(self) -> str:
        if not self.has_position():
            return "No position"
        return (
            f"{self.side.upper()} {self.size:.4f} {self.coin} "
            f"@ ${self.entry_price:.1f} "
            f"(entry_oid={self.entry_oid}, sl_oid={self.sl_oid}, tp_oid={self.tp_oid})"
        )


def _parse_oids_from_result(result: Any, num_orders: int) -> dict[str, int | None]:
    """Extract order IDs from a bulk_orders or order response.

    The Hyperliquid API returns statuses with oids in order:
      statuses[0] → entry order, statuses[1] → SL, statuses[2] → TP
    """
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
                    # "resting" = limit/trigger order placed, "filled" = market order
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
    """Executes trades on Hyperliquid based on validated signals.

    Maintains a PositionTracker to bridge LLM decisions and order
    modification — the tracker stores order IDs from _open_position()
    so that subsequent MODIFY_SL signals can target the right order.
    """

    def __init__(self):
        self.dry_run = config.dry_run
        self.coin = config.trading_pair
        self.tracker = PositionTracker()

        if not self.dry_run:
            if not config.hl_private_key:
                raise ValueError("Private key required for live trading")

            account: LocalAccount = Account.from_key(config.hl_private_key)
            base_url = (
                "https://api.hyperliquid-testnet.xyz"
                if config.hl_testnet
                else config.hl_base_url
            )
            self.exchange = Exchange(wallet=account, base_url=base_url)
            self.address = account.address
            logger.info(
                "TradeExecutor initialized (address=%s, testnet=%s)",
                self.address,
                config.hl_testnet,
            )
        else:
            self.exchange = None
            self.address = "0x_dry_run"
            logger.info("TradeExecutor initialized in DRY RUN mode")

    # ─── Main entry point ────────────────────────────────────────────────

    def execute(self, signal: TradingSignal) -> dict:
        """Execute a trading signal.

        Returns a dict with execution results including the updated
        PositionTracker summary.
        """
        result: dict = {}

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

        # Attach current tracker state to every result
        result["position"] = self.tracker.to_summary()
        return result

    # ─── Open Position ───────────────────────────────────────────────────

    def _open_position(self, signal: TradingSignal, is_buy: bool) -> dict:
        """Open a position with stop loss and take profit.

        Uses bulk_orders for atomic execution, then parses and stores
        the returned order IDs in the tracker — enabling later MODIFY_SL.
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
            # 1. Set leverage
            self.exchange.update_leverage(config.max_leverage, self.coin)

            # 2. Build orders & execute
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

            # 3. Parse order IDs from response and update tracker
            oids = _parse_oids_from_result(result, num_orders)
            self.tracker.update_from_open(side, size, signal.entry_price, oids)
            logger.info("Position tracker updated: %s", self.tracker.to_summary())

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
        """Build the order request list for atomic open + SL + TP."""
        orders: list[dict] = []

        # Order 0: Open position
        if signal.entry_price:
            orders.append({
                "coin": self.coin,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": signal.entry_price,
                "order_type": {"limit": {"tif": "Gtc"}},
                "reduce_only": False,
            })
        else:
            orders.append({
                "coin": self.coin,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": signal.entry_price or 0,
                "order_type": {"limit": {"tif": "Ioc"}},
                "reduce_only": False,
            })

        # Order 1: Stop Loss
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

        # Order 2: Take Profit
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
            "Built %d orders: open(%s) %s SL %s TP",
            len(orders),
            "limit" if signal.entry_price else "market",
            "✓" if signal.stop_loss else "✗",
            "✓" if signal.take_profit else "✗",
        )
        return orders

    # ─── Close Position ──────────────────────────────────────────────────

    def _close_position(self) -> dict:
        """Close the current position and clear the tracker."""
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
        """Cancel an order by client-assigned order ID (cloid)."""
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
        self,
        oid: int,
        is_buy: bool,
        size: float,
        limit_px: float,
        order_type: dict,
        reduce_only: bool = False,
    ) -> dict:
        """Modify any existing order — limit, trigger, or otherwise."""
        if self.dry_run:
            logger.info("DRY RUN: Modify order #%d", oid)
            return {"action": "modify_order", "executed": True, "dry_run": True}

        try:
            result = self.exchange.modify_order(
                oid=oid,
                name=self.coin,
                is_buy=is_buy,
                sz=size,
                limit_px=limit_px,
                order_type=order_type,
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
        """Move an existing stop loss to a new price."""
        return self.modify_order(
            oid=oid,
            is_buy=is_buy,
            size=size,
            limit_px=new_price,
            order_type={
                "trigger": {
                    "triggerPx": new_price,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            },
            reduce_only=True,
        )

    def modify_take_profit(
        self, oid: int, new_price: float, is_buy: bool, size: float
    ) -> dict:
        """Move an existing take profit to a new price."""
        return self.modify_order(
            oid=oid,
            is_buy=is_buy,
            size=size,
            limit_px=new_price,
            order_type={
                "trigger": {
                    "triggerPx": new_price,
                    "isMarket": True,
                    "tpsl": "tp",
                }
            },
            reduce_only=True,
        )

    def modify_orders(self, modifications: list[dict]) -> dict:
        """Modify multiple orders in a single atomic request."""
        if not modifications:
            return {"action": "modify_bulk", "executed": False,
                    "reason": "Empty modification list"}

        if self.dry_run:
            logger.info("DRY RUN: Modify %d orders", len(modifications))
            return {"action": "modify_bulk", "executed": True, "dry_run": True}

        try:
            requests = [
                {
                    "oid": m["oid"],
                    "order": {
                        "coin": self.coin,
                        "is_buy": m["order"]["is_buy"],
                        "sz": m["order"]["sz"],
                        "limit_px": m["order"]["limit_px"],
                        "order_type": m["order"]["order_type"],
                        "reduce_only": m["order"].get("reduce_only", False),
                    },
                }
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
        """Move stop loss to a new price — driven by LLM MODIFY_SL signal.

        Uses the tracked sl_oid from the last _open_position() call.
        This is the bridge: LLM says MODIFY_SL → we know which order to touch.
        """
        new_sl = signal.modify_sl_to or signal.stop_loss
        if new_sl is None:
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": "No new stop loss price provided"}

        if not self.tracker.has_position():
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": "No active position to modify"}

        sl_oid = self.tracker.sl_oid

        if self.dry_run:
            return {
                "action": "MODIFY_SL",
                "executed": True,
                "dry_run": True,
                "sl_oid": sl_oid,
                "new_sl": new_sl,
                "reasoning": signal.reasoning,
            }

        if sl_oid is None:
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": (
                        "No tracked SL order ID. "
                        "The position may have been opened without a stop loss, "
                        "or the SL order was already filled/cancelled."
                    )}

        logger.info(
            "MODIFY_SL: moving SL #%d to $%.1f (reason: %s)",
            sl_oid, new_sl, signal.reasoning[:80],
        )

        return self.modify_stop_loss(
            oid=sl_oid,
            new_price=new_sl,
            is_buy=(self.tracker.side == "short"),  # SL is opposite direction
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
            return {"action": "update_margin", "executed": True,
                    "result": result}
        except Exception as e:
            logger.error("Failed to update margin: %s", e)
            return {"action": "update_margin", "executed": False,
                    "error": str(e)}

    # ─── Dead Man's Switch ───────────────────────────────────────────────

    def schedule_cancel(self, timeout_seconds: int = 300) -> dict:
        """Schedule auto-cancel of all orders after timeout.

        If the bot crashes, all open orders are automatically cancelled
        after `timeout_seconds`. Call this periodically to reset the timer.
        """
        if self.dry_run:
            logger.info("DRY RUN: Dead man's switch set to %ds", timeout_seconds)
            return {"action": "schedule_cancel", "executed": True,
                    "dry_run": True}

        now_ms = int(__import__("time").time() * 1000)
        cancel_time = now_ms + timeout_seconds * 1000

        try:
            result = self.exchange.schedule_cancel(cancel_time)
            logger.info("Dead man's switch: auto-cancel at +%ds", timeout_seconds)
            return {"action": "schedule_cancel", "executed": True,
                    "result": result}
        except Exception as e:
            logger.error("Failed to set dead man's switch: %s", e)
            return {"action": "schedule_cancel", "executed": False,
                    "error": str(e)}
