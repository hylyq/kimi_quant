"""Trade recording, P&L analysis, and performance metrics.

Completes the decision chain:
  行情数据 → LLM分析 → 风控 → 执行 → 盈亏分析

TradeLogger:
  - Records every trade lifecycle (open → close) to a JSONL file
  - Computes running performance statistics
  - Provides a summary for LLM feedback (self-reflection loop)
"""

import fcntl
import json as _json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kimi_quant.config import config

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = str(
    Path(__file__).parent.parent.parent / "data" / "trades.jsonl"
)

# Hyperliquid standard fee rates (as of 2026-07).
# All entry orders use Ioc (market) execution → taker fee on both sides.
TAKER_FEE_RATE = 0.00035   # 0.035% per side
MAKER_FEE_RATE = 0.00010   # 0.010% per side (not used for Ioc entries)
ROUNDTRIP_TAKER_FEE = TAKER_FEE_RATE * 2  # 0.07% round-trip


# ─── Trade Record ────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    """A single completed trade from open to close."""

    # Entry
    opened_at: str  # ISO timestamp
    side: str  # "long" or "short"
    size: float  # BTC
    entry_price: float
    dry_run: bool = False  # True if this is a simulated trade

    # Exit
    closed_at: str | None = None
    exit_price: float = 0.0
    close_reason: str = ""  # "signal" | "stop_loss" | "take_profit" | "manual"

    # Computed
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees_est: float = 0.0  # estimated fees (0.02% maker + 0.05% taker)

    def close(
        self,
        exit_price: float,
        reason: str = "signal",
        closed_at: str | None = None,
    ) -> None:
        """Mark the trade as closed and compute P&L."""
        self.closed_at = closed_at or datetime.now(timezone.utc).isoformat()
        self.exit_price = exit_price
        self.close_reason = reason

        # P&L calculation
        if self.side == "long":
            self.pnl = (exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - exit_price) * self.size

        self.pnl_pct = (
            (self.pnl / (self.entry_price * self.size)) * 100
            if self.entry_price > 0 and self.size > 0
            else 0.0
        )

        # Estimated fees: taker on both sides (all entries use Ioc market orders).
        # Entry notional + exit notional, each at the taker rate.
        entry_notional = self.entry_price * self.size
        exit_notional = exit_price * self.size
        self.fees_est = (entry_notional + exit_notional) * TAKER_FEE_RATE

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.fees_est

    def to_dict(self) -> dict:
        return {
            "opened_at": self.opened_at,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "dry_run": self.dry_run,
            "closed_at": self.closed_at,
            "exit_price": self.exit_price,
            "close_reason": self.close_reason,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "fees_est": round(self.fees_est, 2),
            "net_pnl": round(self.net_pnl, 2),
            "is_win": self.is_win,
        }


# ─── Trade Logger ────────────────────────────────────────────────────────


@dataclass
class PerformanceStats:
    """Running performance statistics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0  # gross_profit / gross_loss
    largest_win: float = 0.0
    largest_loss: float = 0.0


class TradeLogger:
    """Records trades and computes performance statistics.

    Lifecycle:
      1. open_trade() — called when a position is opened
      2. close_trade() — called when a position is closed (by any reason)
      3. get_stats() — compute running performance metrics
      4. get_llm_context() — short summary for LLM self-reflection
    """

    def __init__(self, log_path: str | None = None):
        self.log_path = log_path or DEFAULT_LOG_PATH
        self._pending: TradeRecord | None = None  # currently open trade
        self._closed: list[TradeRecord] = []

        # Load existing trades from disk
        self._load()

        logger.info(
            "TradeLogger initialized: %d historical trades, path=%s",
            len(self._closed),
            self.log_path,
        )

    # ─── Lifecycle ───────────────────────────────────────────────────────

    def open_trade(
        self, side: str, size: float, entry_price: float,
        dry_run: bool = False,
    ) -> TradeRecord:
        """Record a new trade opening."""
        if self._pending is not None:
            logger.warning(
                "Opening new trade while one is pending — closing pending first"
            )
            # Don't force-close; just overwrite with a warning
            self._pending = None

        trade = TradeRecord(
            opened_at=datetime.now(timezone.utc).isoformat(),
            side=side,
            size=size,
            entry_price=entry_price,
            dry_run=dry_run,
        )
        self._pending = trade
        mode = " (dry-run)" if dry_run else ""
        logger.info(
            "Trade opened%s: %s %.4f @ $%.1f",
            mode, side.upper(), size, entry_price,
        )
        return trade

    def close_trade(
        self, exit_price: float, reason: str = "signal"
    ) -> TradeRecord | None:
        """Close the pending trade and persist it to disk.

        Args:
            exit_price: The price at which the position was closed.
            reason: "signal" (LLM decided), "stop_loss", "take_profit",
                    or "manual" (unknown/chain detected).
        """
        if self._pending is None:
            logger.warning("No pending trade to close")
            return None

        self._pending.close(exit_price, reason)
        trade = self._pending
        self._closed.append(trade)
        self._pending = None

        # Persist immediately
        self._append_to_file(trade)
        logger.info(
            "Trade closed: %s %.4f @ $%.1f → $%.1f | P&L=$%.2f (%.2f%%) | %s",
            trade.side.upper(),
            trade.size,
            trade.entry_price,
            trade.exit_price,
            trade.pnl,
            trade.pnl_pct,
            "WIN" if trade.is_win else "LOSS",
        )

        return trade

    def cancel_pending(self) -> None:
        """Clear a pending trade that never filled (limit order missed)."""
        if self._pending:
            logger.info(
                "Cancelling pending trade (limit order unfilled): %s %.4f @ $%.1f",
                self._pending.side,
                self._pending.size,
                self._pending.entry_price,
            )
            self._pending = None

    def recover_trade(
        self, side: str, size: float, entry_price: float
    ) -> TradeRecord | None:
        """Recover a pending trade from chain state after a restart.

        Called when the bot restarts and finds an existing position on chain
        that was not previously recorded in the trade log.
        """
        if self._pending is not None:
            logger.warning(
                "Pending trade already exists — not overwriting with recovered trade"
            )
            return None

        trade = TradeRecord(
            opened_at=datetime.now(timezone.utc).isoformat(),
            side=side,
            size=size,
            entry_price=entry_price,
        )
        self._pending = trade
        logger.info(
            "Trade recovered from chain: %s %.4f @ $%.1f",
            side.upper(), size, entry_price,
        )
        return trade

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    # ─── Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> PerformanceStats:
        """Compute running performance statistics from all closed trades."""
        return self._compute_stats(self._closed)

    @staticmethod
    def _compute_stats(trades: list[TradeRecord]) -> PerformanceStats:
        """Compute performance statistics from a list of trades."""
        stats = PerformanceStats()
        stats.total_trades = len(trades)

        if stats.total_trades == 0:
            return stats

        gross_profit = 0.0
        gross_loss = 0.0

        for t in trades:
            if t.is_win:
                stats.wins += 1
                gross_profit += t.pnl
                if t.pnl > stats.largest_win:
                    stats.largest_win = t.pnl
            else:
                stats.losses += 1
                gross_loss += abs(t.pnl)
                if abs(t.pnl) > abs(stats.largest_loss):
                    stats.largest_loss = t.pnl  # negative

            stats.total_pnl += t.pnl
            stats.total_fees += t.fees_est

        stats.win_rate = (stats.wins / stats.total_trades) * 100
        stats.net_pnl = stats.total_pnl - stats.total_fees
        stats.avg_win = gross_profit / stats.wins if stats.wins > 0 else 0.0
        stats.avg_loss = (
            -(gross_loss / stats.losses) if stats.losses > 0 else 0.0
        )
        stats.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        return stats

    def get_llm_context(self) -> str:
        """Build a short performance summary for LLM self-reflection.

        Append this to the market prompt so the LLM knows its own track record.
        Only includes real (non-simulated) trades.
        """
        stats = self.get_stats()
        real_trades = [t for t in self._closed if not t.dry_run]
        if len(real_trades) == 0:
            return ""

        # Compute real-trade-only stats
        real_stats = self._compute_stats(real_trades)

        lines = [
            "\n# Your Trading Performance (Historical — Real Trades Only)",
            f"Total Trades: {real_stats.total_trades}",
            f"Win Rate: {real_stats.win_rate:.1f}% ({real_stats.wins}W / {real_stats.losses}L)",
            f"Net P&L: ${real_stats.net_pnl:+.2f}",
            f"Gross P&L: ${real_stats.total_pnl:+.2f} (Fees: ${real_stats.total_fees:.2f})",
            f"Avg Win: ${real_stats.avg_win:+.2f} | Avg Loss: ${real_stats.avg_loss:+.2f}",
            f"Largest Win: ${real_stats.largest_win:+.2f} | Largest Loss: ${real_stats.largest_loss:+.2f}",
            f"Profit Factor: {real_stats.profit_factor:.2f}",
        ]

        # Add recent trades context (real only)
        recent = real_trades[-5:]
        if recent:
            lines.append("\n## Recent Trades")
            for t in recent:
                lines.append(
                    f"- {t.side.upper()} | in: ${t.entry_price:.1f} "
                    f"→ out: ${t.exit_price:.1f} | "
                    f"P&L: ${t.pnl:+.2f} ({t.pnl_pct:+.2f}%) | "
                    f"{t.close_reason}"
                )

        # Self-reflection guidance
        if real_stats.win_rate < 40:
            lines.append(
                "\n⚠️ Your win rate is below 40%. Consider being more "
                "conservative — only trade when confidence is very high."
            )
        elif real_stats.losses >= 3 and real_stats.avg_loss > real_stats.avg_win:
            lines.append(
                "\n⚠️ Your average loss exceeds your average win. "
                "Tighten stop losses or reduce position sizes."
            )

        return "\n".join(lines)

    def get_lessons_context(self) -> str:
        """Build a lessons-learned summary from recent closed trades.

        Analyzes patterns in the last 10 trades and produces actionable
        guidance for the LLM (e.g., repeated stop-outs at resistance,
        TP hit too early in trends).

        Returns empty string if < 3 trades exist.
        """
        real_trades = [t for t in self._closed if not t.dry_run]
        if len(real_trades) < 3:
            return ""

        recent = real_trades[-10:]
        lessons: list[str] = []

        # 1. Consecutive stop-outs
        sl_losses = [t for t in recent[-5:] if t.close_reason == "stop_loss"]
        if len(sl_losses) >= 2:
            lessons.append(
                f"⚠️  {len(sl_losses)} recent stop-outs. "
                f"SL may be too tight for current volatility — "
                f"consider widening SL or reducing position size."
            )

        # 2. TP hit early (market ran further)
        tp_wins = [t for t in recent[-5:] if t.close_reason == "take_profit"]
        if tp_wins and len(tp_wins) >= 2:
            lessons.append(
                "💡 Multiple TP hits — trend may be stronger than expected. "
                "Consider using trailing stops instead of fixed TP, or "
                "scaling out (close partial at TP, let remainder run)."
            )

        # 3. Win rate by side
        longs = [t for t in recent if t.side == "long"]
        shorts = [t for t in recent if t.side == "short"]
        if longs:
            lr = sum(1 for t in longs if t.is_win) / len(longs)
            if lr < 0.4 and len(longs) >= 2:
                lessons.append(
                    f"⚠️  Long win rate: {lr:.0%} ({sum(1 for t in longs if t.is_win)}/{len(longs)}). "
                    f"Reassess long entry criteria."
                )
        if shorts:
            sr = sum(1 for t in shorts if t.is_win) / len(shorts)
            if sr < 0.4 and len(shorts) >= 2:
                lessons.append(
                    f"⚠️  Short win rate: {sr:.0%} ({sum(1 for t in shorts if t.is_win)}/{len(shorts)}). "
                    f"Reassess short entry criteria."
                )

        # 4. Net P&L trend
        if len(recent) >= 5:
            recent_pnl = sum(t.net_pnl for t in recent[-5:])
            if recent_pnl < 0:
                lessons.append(
                    f"📉 Net P&L last 5 trades: ${recent_pnl:+.2f}. "
                    f"Be more selective — only trade when confidence is high."
                )

        if not lessons:
            return ""

        lines = ["\n# 📚 Recent Lessons (from closed trades)\n"]
        for i, lesson in enumerate(lessons, 1):
            lines.append(f"{i}. {lesson}")

        return "\n".join(lines)

    # ─── Persistence ─────────────────────────────────────────────────────

    def _append_to_file(self, trade: TradeRecord) -> None:
        """Append a single trade as a JSON line. Uses exclusive lock."""
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(_json.dumps(trade.to_dict(), default=str) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error("Failed to persist trade: %s", e)

    def _load(self) -> None:
        """Load existing trades from the JSONL file. Uses shared lock."""
        try:
            path = Path(self.log_path)
            if not path.exists():
                return
            with open(path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = _json.loads(line)
                            trade = TradeRecord(
                                opened_at=data["opened_at"],
                                side=data["side"],
                                size=data["size"],
                                entry_price=data["entry_price"],
                                dry_run=data.get("dry_run", False),
                                closed_at=data.get("closed_at"),
                                exit_price=data.get("exit_price", 0),
                                close_reason=data.get("close_reason", ""),
                                pnl=data.get("pnl", 0),
                                pnl_pct=data.get("pnl_pct", 0),
                                fees_est=data.get("fees_est", 0),
                            )
                            self._closed.append(trade)
                        except Exception:
                            pass
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            if self._closed:
                logger.info("Loaded %d trades from %s", len(self._closed), path)
        except Exception as e:
            logger.error("Failed to load trades: %s", e)

    def get_all_trades(self) -> list[TradeRecord]:
        """Return all closed trades."""
        return list(self._closed)
