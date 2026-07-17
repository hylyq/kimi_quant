"""Trade execution via Hyperliquid Exchange API."""

import logging

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades on Hyperliquid based on validated signals."""

    def __init__(self):
        self.dry_run = config.dry_run
        self.coin = config.trading_pair

        if not self.dry_run:
            if not config.hl_private_key:
                raise ValueError(
                    "Private key required for live trading"
                )

            account: LocalAccount = Account.from_key(
                config.hl_private_key
            )
            base_url = (
                "https://api.hyperliquid-testnet.xyz"
                if config.hl_testnet
                else config.hl_base_url
            )
            self.exchange = Exchange(
                wallet=account, base_url=base_url
            )
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

    def execute(self, signal: TradingSignal) -> dict:
        """Execute a trading signal.

        Returns a dict with execution results.
        """
        if signal.action == "HOLD":
            logger.info("HOLD — no action taken")
            return {
                "action": "HOLD",
                "executed": False,
                "reason": signal.reasoning,
            }

        if signal.action == "CLOSE":
            return self._close_position()

        if signal.action == "LONG":
            return self._open_position(signal, is_buy=True)

        if signal.action == "SHORT":
            return self._open_position(signal, is_buy=False)

        return {"action": signal.action, "executed": False,
                "reason": f"Unknown action: {signal.action}"}

    def _open_position(
        self, signal: TradingSignal, is_buy: bool
    ) -> dict:
        """Open a long or short position."""
        size = signal.size or config.max_position_size
        side = "BUY" if is_buy else "SELL"

        if self.dry_run:
            logger.info(
                "DRY RUN: Would %s %s %s @ ~$%.1f (SL: $%.1f, TP: $%.1f)",
                side,
                size,
                self.coin,
                signal.entry_price or 0,
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
            # Set leverage first
            self.exchange.update_leverage(
                config.max_leverage, self.coin
            )

            if signal.entry_price:
                # Limit order
                order_type = {"limit": {"tif": "Gtc"}}
                result = self.exchange.order(
                    name=self.coin,
                    is_buy=is_buy,
                    sz=size,
                    limit_px=signal.entry_price,
                    order_type=order_type,
                )
            else:
                # Market order with 0.5% slippage
                result = self.exchange.market_open(
                    name=self.coin,
                    is_buy=is_buy,
                    sz=size,
                    slippage=0.005,
                )

            logger.info("Order placed: %s", result)

            # Place stop loss if enabled
            if signal.stop_loss:
                try:
                    sl_result = self.exchange.order(
                        name=self.coin,
                        is_buy=not is_buy,  # opposite direction
                        sz=size,
                        limit_px=signal.stop_loss,
                        order_type={"trigger": {
                            "triggerPx": signal.stop_loss,
                            "isMarket": True,
                            "tpsl": "sl",
                        }},
                    )
                    logger.info("Stop loss placed: %s", sl_result)
                except Exception as e:
                    logger.error("Failed to place stop loss: %s", e)

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

    def _close_position(self) -> dict:
        """Close the current position."""
        if self.dry_run:
            logger.info("DRY RUN: Would close %s position", self.coin)
            return {
                "action": "CLOSE",
                "executed": True,
                "dry_run": True,
            }

        try:
            result = self.exchange.close_position(self.coin)
            logger.info("Position closed: %s", result)
            return {
                "action": "CLOSE",
                "executed": True,
                "dry_run": False,
                "result": result,
            }
        except Exception as e:
            logger.error("Failed to close position: %s", e)
            return {
                "action": "CLOSE",
                "executed": False,
                "error": str(e),
            }
