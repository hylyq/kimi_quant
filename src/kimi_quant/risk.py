"""Risk management — validates trading signals before execution.

Checks:
  1. Confidence threshold
  2. Position size limits (with dynamic ATR-based sizing)
  3. Margin requirement (position notional / leverage ≤ 95% available balance)
  4. Risk amount (|entry - SL| × size ≤ 2% of balance)
  5. Direction validation (no redundant trades)
  6. Stop loss distance (minimum % from entry)
  7. Drawdown circuit breaker (pause after consecutive losses)
"""

import logging
from dataclasses import dataclass, field

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal
from kimi_quant.notify import notify

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk validation check."""
    passed: bool
    reason: str


class RiskManager:
    """Validates trading signals with multi-layer risk checks.

    Circuit breaker: after N consecutive losses, blocks new positions
    until a cooldown period passes or a win resets the counter.
    """

    # Minimum stop loss distance from entry (as fraction of price)
    MIN_SL_DISTANCE = 0.005  # 0.5% — BTC noise is ~0.3%

    # Circuit breaker
    MAX_CONSECUTIVE_LOSSES = 4
    COOLDOWN_CYCLES = 6  # wait 6 cycles (e.g. 30 min) before resuming
    MAX_DAILY_DRAWDOWN = -0.05  # -5% of account equity

    def __init__(self):
        self.max_position = config.max_position_size
        self.min_confidence = config.min_confidence
        self.max_leverage = config.max_leverage

        # Circuit breaker state (reset across sessions)
        self.consecutive_losses: int = 0
        self.cooldown_remaining: int = 0
        self.total_realized_pnl: float = 0.0
        self.initial_balance: float | None = None

    # ─── Main entry ──────────────────────────────────────────────────────

    def validate(
        self,
        signal: TradingSignal,
        current_position_size: float = 0.0,
        current_position_side: str = "none",
        mid_price: float = 0.0,
        account_balance: float | None = None,
    ) -> RiskCheck:
        """Run all risk checks. Returns first failure or success."""
        checks = [
            self._check_circuit_breaker(signal),
            self._check_confidence(signal),
            self._check_position_size(
                signal, current_position_size, current_position_side,
                mid_price, account_balance,
            ),
            self._check_direction(signal, current_position_side),
            self._check_stop_loss_distance(signal, mid_price),
        ]

        for check in checks:
            if not check.passed:
                logger.warning("Risk check FAILED: %s", check.reason)
                return check

        logger.info("All risk checks passed")
        return RiskCheck(passed=True, reason="All checks passed")

    # ─── Lifecycle (called from main loop) ────────────────────────────────

    def record_loss(self, pnl: float) -> None:
        """Called when a trade closes at a loss."""
        self.consecutive_losses += 1
        self.total_realized_pnl += pnl
        if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            # Only trigger if not already in cooldown (don't extend indefinitely)
            if self.cooldown_remaining <= 0:
                self.cooldown_remaining = self.COOLDOWN_CYCLES
                logger.warning(
                    "CIRCUIT BREAKER: %d consecutive losses. "
                    "Pausing new positions for %d cycles.",
                    self.consecutive_losses, self.COOLDOWN_CYCLES,
                )
                notify.send(
                    f"⚠️ Circuit breaker activated\n"
                    f"{self.consecutive_losses} consecutive losses\n"
                    f"Paused {self.COOLDOWN_CYCLES} cycles | P&L: ${self.total_realized_pnl:+.2f}"
                )
            else:
                logger.warning(
                    "Loss during active cooldown (%d remaining cycles) — "
                    "not extending cooldown. %d consecutive losses.",
                    self.cooldown_remaining, self.consecutive_losses,
                )

    def record_win(self, pnl: float) -> None:
        """Called when a trade closes at a profit."""
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.total_realized_pnl += pnl

    def tick_cooldown(self) -> None:
        """Decrement cooldown counter each cycle."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                logger.info("Circuit breaker cooldown expired — resuming trading.")
                self.consecutive_losses = 0

    def is_blocked(self) -> bool:
        return self.cooldown_remaining > 0

    def get_block_reason(self) -> str:
        return (
            f"Circuit breaker active: {self.consecutive_losses} consecutive losses, "
            f"{self.cooldown_remaining} cycles remaining in cooldown"
        )

    # ─── Individual Checks ────────────────────────────────────────────────

    def _check_circuit_breaker(self, signal: TradingSignal) -> RiskCheck:
        """Block new positions if circuit breaker is active.

        CLOSE and MODIFY_SL are always allowed (risk-reducing actions).
        HOLD is a no-op.
        """
        if signal.action in ("HOLD", "CLOSE", "MODIFY_SL"):
            return RiskCheck(passed=True, reason="Not a new position")

        if self.cooldown_remaining > 0:
            return RiskCheck(passed=False, reason=self.get_block_reason())

        # Check daily drawdown if we have a reference balance
        if self.initial_balance and self.total_realized_pnl < 0:
            drawdown = self.total_realized_pnl / self.initial_balance
            if drawdown < self.MAX_DAILY_DRAWDOWN:
                return RiskCheck(
                    passed=False,
                    reason=f"Max daily drawdown exceeded: {drawdown:.2%}",
                )

        return RiskCheck(passed=True, reason="Circuit breaker OK")

    def _check_confidence(self, signal: TradingSignal) -> RiskCheck:
        if signal.action in ("HOLD", "CLOSE", "MODIFY_SL"):
            return RiskCheck(passed=True, reason="Confidence not required")

        if signal.confidence < self.min_confidence:
            return RiskCheck(
                passed=False,
                reason=f"Confidence {signal.confidence:.2f} below "
                f"minimum {self.min_confidence}",
            )
        return RiskCheck(passed=True, reason="Confidence OK")

    def _check_position_size(
        self,
        signal: TradingSignal,
        current_size: float,
        current_side: str,
        mid_price: float,
        account_balance: float | None,
    ) -> RiskCheck:
        if signal.action in ("HOLD", "CLOSE", "MODIFY_SL"):
            return RiskCheck(passed=True, reason="No size needed")

        size = signal.size or self.max_position

        if size <= 0:
            return RiskCheck(passed=False, reason=f"Invalid position size: {size}")

        if size > self.max_position:
            return RiskCheck(
                passed=False,
                reason=f"Size {size} exceeds max {self.max_position}",
            )

        # Margin requirement check: ensure account can afford the position.
        # Without this, the LLM can order a position whose required margin
        # exceeds the available balance even though risk_amount passes below.
        if account_balance and account_balance > 0 and mid_price > 0:
            position_notional = size * mid_price
            margin_required = position_notional / self.max_leverage
            max_allowable_margin = account_balance * 0.95  # 95% of available balance

            if margin_required > max_allowable_margin:
                suggested_size = (max_allowable_margin * self.max_leverage) / mid_price
                return RiskCheck(
                    passed=False,
                    reason=(
                        f"Margin required ${margin_required:.2f} (${position_notional:.0f} "
                        f"notional / {self.max_leverage}x) exceeds 95% of available "
                        f"balance ${account_balance:.2f}. "
                        f"Suggested size: {suggested_size:.4f} (actual: {size:.4f})"
                    ),
                )

        # Dynamic risk-based sizing: don't risk more than 1% of balance per trade
        if account_balance and account_balance > 0 and signal.stop_loss and mid_price > 0:
            # Risk amount = |entry - SL| * size
            entry = signal.entry_price or mid_price
            risk_per_unit = abs(entry - signal.stop_loss)
            risk_amount = risk_per_unit * size
            max_risk_warn = account_balance * 0.01  # 1% risk: warn
            max_risk_hard = account_balance * 0.02  # 2% risk: reject
            if risk_amount > max_risk_hard and risk_per_unit > 0:
                suggested_size = max_risk_warn / risk_per_unit
                return RiskCheck(
                    passed=False,
                    reason=(
                        f"Position risk ${risk_amount:.2f} exceeds 2% "
                        f"hard cap of balance ${account_balance:.2f}. "
                        f"Suggested size: {suggested_size:.4f} (actual: {size:.4f})"
                    ),
                )
            elif risk_amount > max_risk_warn and risk_per_unit > 0:
                suggested_size = max_risk_warn / risk_per_unit
                logger.warning(
                    "Position risk $%.2f exceeds 1%% of balance $%.2f. "
                    "Suggested size: %.4f (actual: %.4f)",
                    risk_amount, account_balance, suggested_size, size,
                )

        return RiskCheck(passed=True, reason="Size OK")

    def _check_direction(
        self, signal: TradingSignal, current_side: str,
    ) -> RiskCheck:
        if signal.action == "LONG" and current_side == "long":
            return RiskCheck(passed=False, reason="Already holding a long position")
        if signal.action == "SHORT" and current_side == "short":
            return RiskCheck(passed=False, reason="Already holding a short position")
        if signal.action == "CLOSE" and current_side == "none":
            return RiskCheck(passed=False, reason="No position to close")
        if signal.action == "MODIFY_SL" and current_side == "none":
            return RiskCheck(passed=False, reason="No position to modify SL for")

        return RiskCheck(passed=True, reason="Direction OK")

    def _check_stop_loss_distance(
        self, signal: TradingSignal, mid_price: float,
    ) -> RiskCheck:
        """Validate stop loss is at a reasonable distance from entry.

        Too-tight stops get hit by market noise; too-wide stops risk too much.
        """
        if signal.action not in ("LONG", "SHORT"):
            return RiskCheck(passed=True, reason="No SL needed")

        if signal.stop_loss is None:
            return RiskCheck(
                passed=False,
                reason="Stop loss is required for directional trades",
            )

        entry = signal.entry_price or mid_price
        if entry <= 0:
            return RiskCheck(passed=True, reason="Cannot validate SL distance")

        sl_distance = abs(entry - signal.stop_loss) / entry

        if sl_distance < self.MIN_SL_DISTANCE:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Stop loss distance {sl_distance:.2%} is too tight "
                    f"(min {self.MIN_SL_DISTANCE:.2%}). "
                    f"Entry=${entry:.0f} SL=${signal.stop_loss:.0f}. "
                    f"BTC noise alone (~0.3%) can trigger this."
                ),
            )

        # Also warn if SL is excessively far (>10%)
        if sl_distance > 0.10:
            logger.warning(
                "Stop loss distance %.1f%% is very wide — risk/reward may be poor",
                sl_distance * 100,
            )

        return RiskCheck(passed=True, reason=f"SL distance OK ({sl_distance:.2%})")
