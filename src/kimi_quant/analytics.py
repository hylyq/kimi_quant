"""Trade recording, P&L analysis, and performance metrics.

Completes the decision chain:
  行情数据 → LLM分析 → 风控 → 执行 → 盈亏分析

TradeLogger:
  - Records every trade lifecycle (open → close) to a JSONL file
  - Computes running performance statistics
  - Provides a summary for LLM feedback (self-reflection loop)
"""

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


# ─── Trade Record ────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    """A single completed trade from open to close."""

    # Entry
    opened_at: str  # ISO timestamp
    side: str  # "long" or "short"
    size: float  # BTC
    entry_price: float

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

        # Estimated fees: 0.02% maker (entry) + 0.05% taker (exit)
        notional = self.entry_price * self.size
        self.fees_est = notional * 0.0002 + exit_price * self.size * 0.0005

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
        self, side: str, size: float, entry_price: float
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
        )
        self._pending = trade
        logger.info(
            "Trade opened: %s %.4f @ $%.1f", side.upper(), size, entry_price
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

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    # ─── Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> PerformanceStats:
        """Compute running performance statistics from all closed trades."""
        trades = self._closed
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
        """
        stats = self.get_stats()
        if stats.total_trades == 0:
            return ""

        lines = [
            "\n# Your Trading Performance (Historical)",
            f"Total Trades: {stats.total_trades}",
            f"Win Rate: {stats.win_rate:.1f}% ({stats.wins}W / {stats.losses}L)",
            f"Net P&L: ${stats.net_pnl:+.2f}",
            f"Gross P&L: ${stats.total_pnl:+.2f} (Fees: ${stats.total_fees:.2f})",
            f"Avg Win: ${stats.avg_win:+.2f} | Avg Loss: ${stats.avg_loss:+.2f}",
            f"Largest Win: ${stats.largest_win:+.2f} | Largest Loss: ${stats.largest_loss:+.2f}",
            f"Profit Factor: {stats.profit_factor:.2f}",
        ]

        # Add recent trades context
        recent = self._closed[-5:]
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
        if stats.win_rate < 40:
            lines.append(
                "\n⚠️ Your win rate is below 40%. Consider being more "
                "conservative — only trade when confidence is very high."
            )
        elif stats.losses >= 3 and stats.avg_loss > stats.avg_win:
            lines.append(
                "\n⚠️ Your average loss exceeds your average win. "
                "Tighten stop losses or reduce position sizes."
            )

        return "\n".join(lines)

    # ─── Persistence ─────────────────────────────────────────────────────

    def _append_to_file(self, trade: TradeRecord) -> None:
        """Append a single trade as a JSON line."""
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(_json.dumps(trade.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error("Failed to persist trade: %s", e)

    def _load(self) -> None:
        """Load existing trades from the JSONL file."""
        try:
            path = Path(self.log_path)
            if not path.exists():
                return
            with open(path) as f:
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
            if self._closed:
                logger.info("Loaded %d trades from %s", len(self._closed), path)
        except Exception as e:
            logger.error("Failed to load trades: %s", e)

    def get_all_trades(self) -> list[TradeRecord]:
        """Return all closed trades."""
        return list(self._closed)
