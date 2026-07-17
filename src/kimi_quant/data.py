"""Market data fetching from Hyperliquid.

Provides structured market data for the LLM to analyze.
"""

import logging
from dataclasses import dataclass
from typing import Any

from hyperliquid.info import Info

from kimi_quant.config import config

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """A snapshot of current market conditions for the trading pair."""

    coin: str
    mid_price: float
    mark_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    spread_pct: float
    funding_rate: float
    open_interest: float
    prev_day_px: float
    day_change_pct: float
    oracle_price: float
    premium: float  # mark - oracle

    def to_summary(self) -> str:
        """Render as a human-readable summary for the LLM prompt."""
        return (
            f"Coin: {self.coin}\n"
            f"Mid Price: ${self.mid_price:.2f}\n"
            f"Mark Price: ${self.mark_price:.2f}\n"
            f"Oracle Price: ${self.oracle_price:.2f}\n"
            f"Premium (Mark-Oracle): ${self.premium:.2f}\n"
            f"Bid: ${self.bid_price:.2f} (size: {self.bid_size:.4f})\n"
            f"Ask: ${self.ask_price:.2f} (size: {self.ask_size:.4f})\n"
            f"Spread: ${self.spread:.2f} ({self.spread_pct:.4f}%)\n"
            f"Funding Rate: {self.funding_rate * 100:.4f}%\n"
            f"Open Interest: ${self.open_interest:,.0f}\n"
            f"24h Change: {self.day_change_pct:.2f}%\n"
        )


@dataclass
class OrderBookDepth:
    """Aggregated order book depth at multiple levels."""

    bids: list[dict[str, float]]  # [{price, size}, ...]
    asks: list[dict[str, float]]
    bid_total: float
    ask_total: float
    imbalance: float  # positive = bid-heavy, negative = ask-heavy

    def to_summary(self, levels: int = 5) -> str:
        """Render depth summary for LLM."""
        lines = [f"Order Book (top {levels} levels):"]
        lines.append("Bids:")
        for b in self.bids[:levels]:
            lines.append(f"  ${b['price']:.1f} — {b['size']:.4f}")
        lines.append("Asks:")
        for a in self.asks[:levels]:
            lines.append(f"  ${a['price']:.1f} — {a['size']:.4f}")
        lines.append(
            f"Bid/Ask Total: {self.bid_total:.2f} / {self.ask_total:.2f}"
        )
        lines.append(f"Imbalance: {self.imbalance:.4f}")
        return "\n".join(lines)


@dataclass
class AccountSnapshot:
    """Current account and position state."""

    balance: float
    position_size: float
    position_side: str  # "long", "short", "none"
    entry_price: float
    unrealized_pnl: float
    margin_used: float
    leverage: int

    def to_summary(self) -> str:
        return (
            f"Balance: ${self.balance:.2f}\n"
            f"Position: {self.position_size:.4f} {config.trading_pair} "
            f"({'LONG' if self.position_side == 'long' else 'SHORT' if self.position_side == 'short' else 'NONE'})\n"
            f"Entry Price: ${self.entry_price:.2f}\n"
            f"Unrealized PnL: ${self.unrealized_pnl:.2f}\n"
            f"Margin Used: ${self.margin_used:.2f}\n"
            f"Leverage: {self.leverage}x\n"
        )


class DataProvider:
    """Fetches and aggregates market data from Hyperliquid."""

    def __init__(self):
        if config.hl_testnet:
            base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            base_url = config.hl_base_url

        self.info = Info(base_url=base_url, skip_ws=True)
        self.coin = config.trading_pair
        logger.info(
            "DataProvider initialized (testnet=%s, coin=%s)",
            config.hl_testnet,
            self.coin,
        )

    def get_market_snapshot(self) -> MarketSnapshot:
        """Fetch current market snapshot for the trading pair."""
        meta = self.info.meta()
        all_mids = self.info.all_mids()

        # Find coin metadata
        coin_meta = None
        asset_index = None
        for i, asset in enumerate(meta["universe"]):
            if asset["name"] == self.coin:
                coin_meta = asset
                asset_index = i
                break

        if coin_meta is None:
            raise ValueError(f"Coin {self.coin} not found in universe")

        mid_price = float(all_mids.get(self.coin, 0))
        mark_price = float(coin_meta.get("markPx", 0))
        oracle_price = float(coin_meta.get("oraclePx", 0))
        funding_rate = float(coin_meta.get("funding", 0))
        open_interest = float(coin_meta.get("openInterest", 0))
        prev_day_px = float(coin_meta.get("prevDayPx", 0))

        # Get L2 order book for bid/ask
        l2 = self.info.l2_snapshot(self.coin)

        bid_price = float(l2["levels"][0][0]["px"]) if l2["levels"][0] else 0
        ask_price = float(l2["levels"][1][0]["px"]) if l2["levels"][1] else 0
        bid_size = float(l2["levels"][0][0]["sz"]) if l2["levels"][0] else 0
        ask_size = float(l2["levels"][1][0]["sz"]) if l2["levels"][1] else 0

        spread = ask_price - bid_price if ask_price and bid_price else 0
        spread_pct = (spread / mid_price * 100) if mid_price else 0
        day_change = (
            ((mid_price - prev_day_px) / prev_day_px * 100)
            if prev_day_px
            else 0
        )
        premium = mark_price - oracle_price

        return MarketSnapshot(
            coin=self.coin,
            mid_price=mid_price,
            mark_price=mark_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            spread_pct=spread_pct,
            funding_rate=funding_rate,
            open_interest=open_interest,
            prev_day_px=prev_day_px,
            day_change_pct=day_change,
            oracle_price=oracle_price,
            premium=premium,
        )

    def get_order_book_depth(self, levels: int = 10) -> OrderBookDepth:
        """Fetch order book with aggregated depth."""
        l2 = self.info.l2_snapshot(self.coin)

        bids = []
        asks = []
        bid_total = 0.0
        ask_total = 0.0

        for level in l2["levels"][0][:levels]:
            px, sz = float(level["px"]), float(level["sz"])
            bids.append({"price": px, "size": sz})
            bid_total += sz

        for level in l2["levels"][1][:levels]:
            px, sz = float(level["px"]), float(level["sz"])
            asks.append({"price": px, "size": sz})
            ask_total += sz

        imbalance = (
            (bid_total - ask_total) / (bid_total + ask_total)
            if (bid_total + ask_total) > 0
            else 0
        )

        return OrderBookDepth(
            bids=bids,
            asks=asks,
            bid_total=bid_total,
            ask_total=ask_total,
            imbalance=imbalance,
        )

    def get_account_snapshot(
        self, address: str
    ) -> AccountSnapshot | None:
        """Fetch account state. Returns None in dry-run mode."""
        if config.dry_run:
            return None

        try:
            user_state = self.info.user_state(address)
            positions = user_state.get("assetPositions", [])

            # Find position for our coin
            pos_data = None
            for pos in positions:
                if pos["position"]["coin"] == self.coin:
                    pos_data = pos["position"]
                    break

            if pos_data:
                size = float(pos_data.get("szi", 0))
                entry_px = float(pos_data.get("entryPx", 0))
                unrealized_pnl = float(pos_data.get("unrealizedPnl", 0))
                leverage_value = pos_data.get("leverage", {}).get("value", 1)
                if isinstance(leverage_value, str):
                    leverage = int(leverage_value)
                else:
                    leverage = int(leverage_value)

                if size > 0:
                    side = "long"
                elif size < 0:
                    side = "short"
                    size = abs(size)
                else:
                    side = "none"
            else:
                size = 0
                side = "none"
                entry_px = 0
                unrealized_pnl = 0
                leverage = 1

            margin_used = float(
                user_state.get("marginSummary", {}).get("totalMarginUsed", 0)
            )
            balance = float(
                user_state.get("marginSummary", {}).get("accountValue", 0)
            )

            return AccountSnapshot(
                balance=balance,
                position_size=size,
                position_side=side,
                entry_price=entry_px,
                unrealized_pnl=unrealized_pnl,
                margin_used=margin_used,
                leverage=leverage,
            )
        except Exception as e:
            logger.error("Failed to fetch account state: %s", e)
            return None

    def get_full_report(self, address: str | None = None) -> dict[str, Any]:
        """Get a complete market and account report for the LLM."""
        report: dict[str, Any] = {
            "market": self.get_market_snapshot(),
            "order_book": self.get_order_book_depth(),
        }

        if address:
            report["account"] = self.get_account_snapshot(address)

        return report
