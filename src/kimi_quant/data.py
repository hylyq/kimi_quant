"""Market data fetching from Hyperliquid.

Provides structured market data for the LLM to analyze:
  - Market snapshot (prices, funding, OI)
  - Order book depth with imbalance
  - Multi-timeframe candle analysis (5m/15m/1h/4h)
  - Funding rate trend history
  - Account position state
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from hyperliquid.info import Info

# Import curl_cffi if available (bypasses TLS fingerprint blocking on some servers)
try:
    from curl_cffi import requests as _cf_requests
except ImportError:
    _cf_requests = None

from kimi_quant.config import config

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────


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
        """Render as a compact summary for the LLM prompt."""
        return (
            f"BTC Mid=${self.mid_price:.1f} Mark=${self.mark_price:.1f} "
            f"Oracle=${self.oracle_price:.1f} Premium=${self.premium:.1f}\n"
            f"Bid=${self.bid_price:.1f}(sz={self.bid_size:.4f}) "
            f"Ask=${self.ask_price:.1f}(sz={self.ask_size:.4f}) "
            f"Spread=${self.spread:.1f}({self.spread_pct:.3f}%)\n"
            f"Funding={self.funding_rate*100:.4f}% OI=${self.open_interest:,.0f} "
            f"24h={self.day_change_pct:+.2f}%"
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
        lines = [f"OrderBook top{levels}:"]
        bids_str = " ".join(f"${b['price']:.0f}x{b['size']:.3f}" for b in self.bids[:levels])
        asks_str = " ".join(f"${a['price']:.0f}x{a['size']:.3f}" for a in self.asks[:levels])
        lines.append(f"Bids: {bids_str}")
        lines.append(f"Asks: {asks_str}")
        lines.append(
            f"Depth: {self.bid_total:.2f}B/{self.ask_total:.2f}A "
            f"Imb={self.imbalance:.3f}({'bid+' if self.imbalance > 0 else 'ask+'})"
        )
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
        side = self.position_side.upper() if self.position_side != 'none' else 'NONE'
        return (
            f"Balance=${self.balance:.0f} Pos={self.position_size:.4f}{config.trading_pair}({side}) "
            f"Entry=${self.entry_price:.1f} uPNL=${self.unrealized_pnl:.1f} "
            f"Margin=${self.margin_used:.0f} Lev={self.leverage}x"
        )


# ─── Candle Analysis ────────────────────────────────────────────────────


@dataclass
class TimeframeSummary:
    """Technical summary for a single timeframe — compact enough for LLM."""

    interval: str  # "5m", "15m", "1h", "4h"
    num_candles: int
    duration_hours: float

    # Trend
    trend: str  # "up", "down", "sideways"
    change_pct: float  # % change over the period
    current_close: float
    period_open: float

    # Range
    period_high: float
    period_low: float
    current_range_pct: float  # (high-low)/close of last candle

    # Volume
    total_volume: float
    avg_volume: float
    volume_trend: str  # "increasing", "decreasing", "steady"

    # ATR (Average True Range)
    atr: float = 0.0
    atr_pct: float = 0.0  # ATR as % of current price

    # Key levels (simple: recent swing highs/lows)
    resistance: float | None = None
    support: float | None = None

    def to_summary(self) -> str:
        """Compact one-line summary per timeframe."""
        base = (
            f"[{self.interval}] {self.trend.upper()} {self.change_pct:+.2f}% "
            f"${self.period_low:.0f}-${self.period_high:.0f} "
            f"ATR=${self.atr:.0f}({self.atr_pct:.2f}%) "
            f"Vol:{self.avg_volume:.0f}({self.volume_trend})"
        )
        if self.support and self.resistance:
            base += f" S/R=${self.support:.0f}/${self.resistance:.0f}"
        return base


@dataclass
class FundingTrend:
    """Funding rate change over time."""

    current: float
    avg_1h: float
    avg_8h: float
    trend: str  # "rising", "falling", "stable"
    interpretation: str

    def to_summary(self) -> str:
        return (
            f"Funding: now={self.current*100:.4f}% 1h={self.avg_1h*100:.4f}% "
            f"8h={self.avg_8h*100:.4f}% {self.trend} | {self.interpretation}"
        )


@dataclass
class MarketAnalysis:
    """Complete structured analysis combining all data sources."""

    snapshot: MarketSnapshot
    order_book: OrderBookDepth
    timeframes: list[TimeframeSummary]
    funding_trend: FundingTrend | None
    account: AccountSnapshot | None
    performance_context: str = ""

    def to_llm_prompt(self) -> str:
        """Build the full LLM prompt from all analysis components."""
        parts = ["# Market Data Snapshot\n"]

        if self.snapshot is not None:
            parts.append(self.snapshot.to_summary())
        else:
            parts.append("  (Market snapshot unavailable)")

        parts.append("\n# Multi-Timeframe Technical Analysis\n")
        if self.timeframes:
            for tf in self.timeframes:
                parts.append(tf.to_summary())
        else:
            parts.append("  (Candle data unavailable — testnet or API error)")

        parts.append("\n# Order Book Depth\n")
        if self.order_book is not None:
            parts.append(self.order_book.to_summary(levels=5))
        else:
            parts.append("  (Order book data unavailable)")

        if self.funding_trend:
            parts.append("\n# Funding Rate Trend\n")
            parts.append(self.funding_trend.to_summary())

        parts.append("\n# Account Status\n")
        if self.account:
            parts.append(self.account.to_summary())
        else:
            parts.append("Dry-run mode — no real position.")

        parts.append(
            f"\n# Instructions\n"
            f"Max size={config.max_position_size}BTC. "
            f"Higher TF (4h>1h>15m>5m) carry more weight. "
            f"Confluence → higher confidence. Divergence → follow higher TF, "
            f"reduce size, tighten SL. Stop loss min 0.5% from entry.\n"
        )

        if self.performance_context:
            parts.append(self.performance_context)

        return "\n".join(parts)


# ─── Data Provider ──────────────────────────────────────────────────────


class DataProvider:
    """Fetches and aggregates market data from Hyperliquid.

    Fetches from mainnet by default for candle data availability;
    switches to testnet if HYPERLIQUID_TESTNET=true (candles may be empty).
    """

    # Timeframes to analyze (interval, lookback candles, label, cache TTL seconds)
    TIMEFRAMES = [
        ("5m", 24, "1-hour micro structure", 60),     # cache 1 min
        ("15m", 32, "8-hour short-term trend", 180),   # cache 3 min
        ("1h", 48, "48-hour medium-term trend", 600),   # cache 10 min
        ("4h", 42, "7-day macro trend", 1200),          # cache 20 min
    ]

    def __init__(self):
        if config.hl_testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
            self.use_testnet = True
        else:
            self.base_url = config.hl_base_url
            self.use_testnet = False

        # Patch Hyperliquid SDK to use curl_cffi (Chrome TLS fingerprint).
        # Must happen BEFORE Info() is constructed because __init__ calls API.
        if _cf_requests is not None:
            import hyperliquid.api as _hl_api
            import hyperliquid.info as _hl_info
            _hl_api.requests = _cf_requests
            _hl_info.requests = _cf_requests

        self._info_local = Info(base_url=self.base_url, skip_ws=True)
        self.coin = config.trading_pair

        # Mainnet for candle data (testnet candles are empty).
        try:
            self._info_mainnet = Info(
                base_url="https://api.hyperliquid.xyz", skip_ws=True
            )
        except Exception as e:
            logger.warning(
                "Mainnet API unreachable (%s), using local for candles", e
            )
            self._info_mainnet = self._info_local

        # Candle cache: {interval: (timestamp, candles_list)}
        self._candle_cache: dict[str, tuple[float, list[dict]]] = {}

        logger.info(
            "DataProvider initialized (testnet=%s, coin=%s, curl_cffi=%s)",
            config.hl_testnet,
            self.coin,
            _cf_requests is not None,
        )

    # ─── Market Snapshot ─────────────────────────────────────────────────

    def get_market_snapshot(self) -> MarketSnapshot:
        """Fetch current market snapshot.

        Always uses mainnet for market data (testnet returns zeros for
        mark/oracle/OI). Uses meta_and_asset_ctxs() for full field coverage.
        """
        info = self._info_mainnet
        meta, asset_ctxs = info.meta_and_asset_ctxs()

        # Find BTC in both universe (for name lookup) and asset contexts
        coin_ctx = None
        for i, asset in enumerate(meta["universe"]):
            if asset["name"] == self.coin:
                coin_ctx = asset_ctxs[i]
                break

        if coin_ctx is None:
            raise ValueError(f"Coin {self.coin} not found in universe")

        mid_price = float(coin_ctx.get("midPx", 0))
        mark_price = float(coin_ctx.get("markPx", 0))
        oracle_price = float(coin_ctx.get("oraclePx", 0))
        funding_rate = float(coin_ctx.get("funding", 0))
        open_interest = float(coin_ctx.get("openInterest", 0))
        prev_day_px = float(coin_ctx.get("prevDayPx", 0))
        premium = float(coin_ctx.get("premium", 0))

        l2 = info.l2_snapshot(self.coin)
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
        """Fetch order book with aggregated depth.

        Always uses mainnet (testnet order books are too thin to analyze).
        """
        info = self._info_mainnet
        l2 = info.l2_snapshot(self.coin)

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

    # ─── Candle Analysis ─────────────────────────────────────────────────

    def _fetch_candles(
        self, interval: str, lookback_candles: int, cache_ttl: int = 60
    ) -> list[dict]:
        """Fetch candles for a given interval with TTL caching.

        Uses mainnet for candle data since testnet candles are empty.
        """
        now_s = time.time()

        # Check cache
        cached = self._candle_cache.get(interval)
        if cached is not None:
            cached_time, cached_data = cached
            if now_s - cached_time < cache_ttl:
                return cached_data

        now_ms = int(now_s * 1000)
        interval_minutes = {
            "5m": 5, "15m": 15, "1h": 60, "4h": 240,
        }
        mins = interval_minutes.get(interval, 5)
        start_ms = now_ms - lookback_candles * mins * 60 * 1000

        try:
            data = self._info_mainnet.candles_snapshot(
                self.coin, interval, start_ms, now_ms
            )
            self._candle_cache[interval] = (now_s, data)
            return data
        except Exception as e:
            logger.warning("Failed to fetch %s candles: %s", interval, e)
            # Return stale cache if available, otherwise empty
            if cached is not None:
                logger.info("Using stale cache for %s candles", interval)
                return cached[1]
            return []

    def _analyze_timeframe(
        self, interval: str, lookback_candles: int, cache_ttl: int = 60
    ) -> TimeframeSummary | None:
        """Analyze candles for one timeframe and produce a summary."""
        candles = self._fetch_candles(interval, lookback_candles, cache_ttl)
        if len(candles) < 3:
            return None

        # Split: recent half vs overall
        midpoint = len(candles) // 2
        recent = candles[-midpoint:]
        oldest_recent = recent[0]
        latest = candles[-1]
        first = candles[0]

        # Prices
        current_close = float(latest["c"])
        period_open = float(first["o"])
        period_high = max(float(c["h"]) for c in candles)
        period_low = min(float(c["l"]) for c in candles)
        change_pct = ((current_close - period_open) / period_open) * 100

        # Trend direction
        recent_open = float(oldest_recent["o"])
        recent_change = ((current_close - recent_open) / recent_open) * 100
        if recent_change > 0.5:
            trend = "up"
        elif recent_change < -0.5:
            trend = "down"
        else:
            trend = "sideways"

        # Current candle range
        current_range = (
            (float(latest["h"]) - float(latest["l"])) / float(latest["l"]) * 100
        ) if float(latest["l"]) > 0 else 0

        # Volume
        total_volume = sum(float(c["v"]) for c in candles)
        avg_volume = total_volume / len(candles)
        recent_vol = sum(float(c["v"]) for c in recent) / len(recent)
        older_vol = sum(float(c["v"]) for c in candles[:midpoint]) / max(midpoint, 1)
        if recent_vol > older_vol * 1.2:
            volume_trend = "increasing"
        elif recent_vol < older_vol * 0.8:
            volume_trend = "decreasing"
        else:
            volume_trend = "steady"

        # Simple support/resistance from swing points
        # Resistance: highest close in recent period
        resistance = max(float(c["h"]) for c in recent[-8:]) if len(recent) >= 8 else period_high
        support = min(float(c["l"]) for c in recent[-8:]) if len(recent) >= 8 else period_low

        # ATR: Average True Range over recent candles
        true_ranges = []
        for i in range(1, len(recent)):
            h, l = float(recent[i]["h"]), float(recent[i]["l"])
            prev_c = float(recent[i - 1]["c"])
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            true_ranges.append(tr)
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
        atr_pct = (atr / current_close * 100) if current_close > 0 else 0.0

        interval_minutes_map = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        duration = len(candles) * interval_minutes_map.get(interval, 5) / 60

        return TimeframeSummary(
            interval=interval,
            num_candles=len(candles),
            duration_hours=duration,
            trend=trend,
            change_pct=change_pct,
            current_close=current_close,
            period_open=period_open,
            period_high=period_high,
            period_low=period_low,
            current_range_pct=current_range,
            total_volume=total_volume,
            avg_volume=avg_volume,
            volume_trend=volume_trend,
            atr=atr,
            atr_pct=atr_pct,
            resistance=resistance,
            support=support,
        )

    def get_multi_timeframe_analysis(self) -> list[TimeframeSummary]:
        """Analyze all configured timeframes."""
        summaries = []
        for interval, lookback, _label, cache_ttl in self.TIMEFRAMES:
            tf = self._analyze_timeframe(interval, lookback, cache_ttl)
            if tf:
                summaries.append(tf)
            else:
                logger.warning("No candle data for %s timeframe", interval)
        return summaries

    # ─── Funding Trend ───────────────────────────────────────────────────

    def get_funding_trend(self) -> FundingTrend | None:
        """Analyze funding rate changes over time."""
        try:
            now_ms = int(time.time() * 1000)
            history = self._info_mainnet.funding_history(
                self.coin, now_ms - 24 * 3600 * 1000, now_ms
            )

            if len(history) < 2:
                return None

            rates = [float(h["fundingRate"]) for h in history]
            current = rates[-1]
            # Funding updates ~hourly on Hyperliquid
            recent_1h = rates[-2:] if len(rates) >= 2 else rates
            avg_1h = sum(recent_1h) / len(recent_1h)
            older = rates[:-4] if len(rates) > 4 else rates[:1]
            avg_8h = sum(older) / len(older) if older else current

            if current > avg_8h * 1.5:
                trend = "rising (longs paying more)"
                interpretation = (
                    "Funding increasing — longs are becoming more aggressive, "
                    "potential overcrowding on the long side."
                )
            elif current < avg_8h * 0.5 and current < 0:
                trend = "falling (shorts paying more)"
                interpretation = (
                    "Funding turning negative — shorts are becoming aggressive, "
                    "potential short squeeze setup."
                )
            elif abs(current - avg_8h) < avg_8h * 0.3:
                trend = "stable"
                interpretation = "Funding stable, no extreme positioning detected."
            else:
                trend = "shifting"
                interpretation = "Funding rate is in transition — monitor closely."

            return FundingTrend(
                current=current,
                avg_1h=avg_1h,
                avg_8h=avg_8h,
                trend=trend,
                interpretation=interpretation,
            )
        except Exception as e:
            logger.warning("Failed to fetch funding history: %s", e)
            return None

    # ─── Account ─────────────────────────────────────────────────────────

    def get_account_snapshot(self, address: str) -> AccountSnapshot | None:
        """Fetch account state. Returns None in dry-run mode."""
        if config.dry_run:
            return None

        info = self._info_local
        try:
            user_state = info.user_state(address)
            positions = user_state.get("assetPositions", [])

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

    # ─── Full Report ─────────────────────────────────────────────────────

    def get_full_report(self, address: str | None = None) -> dict[str, Any]:
        """Get a complete market report. Independent HTTP calls run in parallel."""
        snapshot = None
        order_book = None
        timeframes: list[TimeframeSummary] = []
        funding_trend = None
        account = None

        # Define fetch tasks — snapshot and order book are fast, candles are slow
        def _fetch_snapshot():
            return self.get_market_snapshot()

        def _fetch_order_book():
            return self.get_order_book_depth()

        def _fetch_timeframes():
            return self.get_multi_timeframe_analysis()

        def _fetch_funding():
            return self.get_funding_trend()

        def _fetch_account():
            return self.get_account_snapshot(address) if address else None

        # Run all 5 independent fetches in parallel (4 thread pool)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(_fetch_snapshot): "snapshot",
                pool.submit(_fetch_order_book): "order_book",
                pool.submit(_fetch_timeframes): "timeframes",
                pool.submit(_fetch_funding): "funding",
                pool.submit(_fetch_account): "account",
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result(timeout=30)
                    if key == "snapshot":
                        snapshot = result
                    elif key == "order_book":
                        order_book = result
                    elif key == "timeframes":
                        timeframes = result
                    elif key == "funding":
                        funding_trend = result
                    elif key == "account":
                        account = result
                except Exception as e:
                    logger.error("Parallel fetch [%s] failed: %s", key, e)

        # Fallback: if any failed, do sequential fetch
        if snapshot is None:
            snapshot = self.get_market_snapshot()
        if order_book is None:
            order_book = self.get_order_book_depth()
        if not timeframes:
            timeframes = self.get_multi_timeframe_analysis()
        if funding_trend is None:
            funding_trend = self.get_funding_trend()

        analysis = MarketAnalysis(
            snapshot=snapshot,
            order_book=order_book,
            timeframes=timeframes,
            funding_trend=funding_trend,
            account=account,
        )

        return {
            "analysis": analysis,
            "market": snapshot,
            "order_book": order_book,
            "account": account,
        }

    @staticmethod
    def build_llm_prompt(report: dict[str, Any]) -> str:
        """Build the full LLM prompt from a report dict.

        Uses MarketAnalysis.to_llm_prompt() if available,
        otherwise falls back to the old per-component method.
        """
        analysis = report.get("analysis")
        if analysis:
            prompt = analysis.to_llm_prompt()
            # Inject performance context if present
            perf = report.get("performance_context", "")
            if perf:
                prompt += "\n" + perf
            return prompt

        # Fallback: old-style prompt
        from kimi_quant.llm import build_market_prompt
        return build_market_prompt(report)
