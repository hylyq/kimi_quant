"""Trade execution via Hyperliquid Exchange API.

Wraps the full Hyperliquid SDK's order capabilities:
  - Market open/close (with slippage)
  - Limit orders (Gtc, Ioc, Alo)
  - Trigger orders (stop loss, take profit)
  - Bulk orders (atomic open + SL + TP in one request)
  - Order modification (move SL to breakeven, etc.)
  - Schedule cancel (dead man's switch)
  - Isolated margin management
"""

import logging
from typing import Any

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades on Hyperliquid based on validated signals.

    Supports the full order lifecycle:
      1. Open position (market or limit)
      2. Attach STOP LOSS + TAKE PROFIT (atomic via bulk_orders)
      3. Modify existing orders (e.g., move SL to breakeven)
      4. Close position (market or limit)
      5. Schedule cancel (dead man's switch)
      6. Margin management
    """

    def __init__(self):
        self.dry_run = config.dry_run
        self.coin = config.trading_pair

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
        """Execute a trading signal."""
        if signal.action == "HOLD":
            logger.info("HOLD — no action taken")
            return {"action": "HOLD", "executed": False,
                    "reason": signal.reasoning}

        if signal.action == "CLOSE":
            return self._close_position()

        if signal.action == "MODIFY_SL":
            return self._handle_modify_sl(signal)

        if signal.action in ("LONG", "SHORT"):
            return self._open_position(signal, is_buy=(signal.action == "LONG"))

        return {"action": signal.action, "executed": False,
                "reason": f"Unknown action: {signal.action}"}

    # ─── Open Position ───────────────────────────────────────────────────

    def _open_position(self, signal: TradingSignal, is_buy: bool) -> dict:
        """Open a position with stop loss and take profit.

        Uses bulk_orders for atomic execution:
          order_0: open position (market or limit)
          order_1: stop loss trigger
          order_2: take profit trigger
        """
        size = signal.size or config.max_position_size
        side = "BUY" if is_buy else "SELL"

        if self.dry_run:
            logger.info(
                "DRY RUN: %s %.4f %s | SL=$%.1f TP=$%.1f",
                side, size, self.coin,
                signal.stop_loss or 0,
                signal.take_profit or 0,
            )
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

            # 2. Build the orders list
            orders = self._build_entry_orders(signal, is_buy, size)

            # 3. Execute atomically with bulk_orders
            # grouping="positionTpsl" links SL/TP to the position
            if len(orders) > 1:
                result = self.exchange.bulk_orders(
                    orders, grouping="positionTpsl"
                )
                logger.info(
                    "Bulk order placed (%d orders): open + SL + TP", len(orders)
                )
            else:
                # Single order (no SL/TP)
                result = self.exchange.order(
                    name=orders[0]["coin"],
                    is_buy=orders[0]["is_buy"],
                    sz=orders[0]["sz"],
                    limit_px=orders[0]["limit_px"],
                    order_type=orders[0]["order_type"],
                )
                logger.info("Single order placed (no SL/TP)")

            logger.info("Result: %s", result)

            return {
                "action": signal.action,
                "executed": True,
                "dry_run": False,
                "side": side,
                "size": size,
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

        Returns a list of order dicts for bulk_orders.
        """
        orders: list[dict] = []

        # Order 0: Open position
        if signal.entry_price:
            open_order = {
                "coin": self.coin,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": signal.entry_price,
                "order_type": {"limit": {"tif": "Gtc"}},
                "reduce_only": False,
            }
        else:
            # Market orders use limit_px with a generous price
            open_order = {
                "coin": self.coin,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": signal.entry_price or 0,
                "order_type": {"limit": {"tif": "Ioc"}},
                "reduce_only": False,
            }
        orders.append(open_order)

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
            "Built %d orders: open(%s) %s SL %s TP",
            len(orders),
            "limit" if signal.entry_price else "market",
            "✓" if signal.stop_loss else "✗",
            "✓" if signal.take_profit else "✗",
        )
        return orders

    # ─── Close Position ──────────────────────────────────────────────────

    def _close_position(self) -> dict:
        """Close the current position (market close)."""
        if self.dry_run:
            logger.info("DRY RUN: Would close %s position", self.coin)
            return {"action": "CLOSE", "executed": True, "dry_run": True}

        try:
            result = self.exchange.market_close(self.coin)
            logger.info("Position closed: %s", result)
            return {
                "action": "CLOSE",
                "executed": True,
                "dry_run": False,
                "result": result,
            }
        except Exception as e:
            logger.error("Failed to close position: %s", e)
            return {"action": "CLOSE", "executed": False, "error": str(e)}

    # ─── Modify Orders ───────────────────────────────────────────────────

    def modify_stop_loss(
        self, oid: int, new_price: float, is_buy: bool, size: float
    ) -> dict:
        """Modify an existing stop loss order (e.g., move to breakeven).

        Args:
            oid: Order ID of the existing stop loss.
            new_price: New trigger price.
            is_buy: Direction of the stop loss order.
            size: Position size.
        """
        if self.dry_run:
            logger.info("DRY RUN: Move SL #%d to $%.1f", oid, new_price)
            return {"action": "modify_sl", "executed": True, "dry_run": True}

        try:
            result = self.exchange.modify_order(
                oid=oid,
                name=self.coin,
                is_buy=is_buy,
                sz=size,
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
            logger.info("Stop loss #%d moved to $%.1f", oid, new_price)
            return {"action": "modify_sl", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to modify stop loss #%d: %s", oid, e)
            return {"action": "modify_sl", "executed": False, "error": str(e)}

    def cancel_order(self, oid: int) -> dict:
        """Cancel a single order by ID."""
        if self.dry_run:
            logger.info("DRY RUN: Cancel order #%d", oid)
            return {"action": "cancel", "executed": True, "dry_run": True}

        try:
            result = self.exchange.cancel(self.coin, oid)
            logger.info("Order #%d cancelled", oid)
            return {"action": "cancel", "executed": True, "result": result}
        except Exception as e:
            logger.error("Failed to cancel order #%d: %s", oid, e)
            return {"action": "cancel", "executed": False, "error": str(e)}

    def _handle_modify_sl(self, signal: TradingSignal) -> dict:
        """Move stop loss to a new price.

        Used when the LLM wants to trail the stop or move to breakeven.
        Requires the current stop loss order ID — in a real implementation
        this would be tracked from the last _open_position result.

        For now, this is a best-effort operation that logs the intent.
        """
        new_sl = signal.modify_sl_to or signal.stop_loss
        if new_sl is None:
            return {"action": "MODIFY_SL", "executed": False,
                    "reason": "No new stop loss price provided"}

        logger.info(
            "MODIFY_SL: move stop loss to $%.1f (reason: %s)",
            new_sl, signal.reasoning[:80],
        )

        if self.dry_run:
            return {
                "action": "MODIFY_SL",
                "executed": True,
                "dry_run": True,
                "new_sl": new_sl,
                "reasoning": signal.reasoning,
            }

        return {
            "action": "MODIFY_SL",
            "executed": False,
            "reason": (
                "MODIFY_SL requires the existing stop loss order ID. "
                "Track oid from _open_position result and call modify_stop_loss()."
            ),
            "new_sl": new_sl,
        }

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
        """Schedule auto-cancel of all orders after timeout (dead man's switch).

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
