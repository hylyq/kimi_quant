"""Risk management — validates trading signals before execution.

Checks:
  1. Confidence threshold
  2. Position size limits (with dynamic ATR-based sizing)
  3. Margin requirement (position notional / leverage ≤ 95% available balance)
  4. Risk amount (|entry - SL| × size ≤ 2% of balance)
  5. Direction validation (no redundant or naked opposite trades)
  6. Stop loss distance + side (SL must sit on the losing side of entry)
  7. Take profit side + minimum risk/reward ratio
  8. Drawdown circuit breaker (pause after consecutive losses)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

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
    # Single source of truth lives in Config (MIN_SL_DISTANCE env var) —
    # data.py reads the same value for its prompt recap.
    MIN_SL_DISTANCE = config.min_sl_distance  # 0.5% default — BTC noise is ~0.3%

    # Minimum risk/reward ratio: |TP - entry| / |SL - entry| must be ≥ this.
    # Prevents trades where the potential loss dwarfs the potential gain.
    # TP is REQUIRED for LONG/SHORT (see _check_take_profit), so this gate
    # always applies to directional entries.
    MIN_RR_RATIO = 1.5  # reward must be at least 1.5× the risk

    # Circuit breaker
    MAX_CONSECUTIVE_LOSSES = 4
    COOLDOWN_CYCLES = 6  # wait 6 cycles (e.g. 30 min) before resuming
    MAX_DAILY_DRAWDOWN = -0.05  # -5% of account equity (UTC day)

    # Actions that never open risk — always allowed, never size/confidence checked.
    _RISK_NEUTRAL_ACTIONS = ("HOLD", "CLOSE", "MODIFY_SL", "MODIFY_TP", "CANCEL_STALE")

    def __init__(self):
        self.max_position = config.max_position_size
        self.min_confidence = config.min_confidence
        self.max_leverage = config.max_leverage

        # Circuit breaker state (consecutive losses are reset across sessions;
        # the daily-P&L baseline can be re-seeded from trade history on startup)
        self.consecutive_losses: int = 0
        self.cooldown_remaining: int = 0
        self.total_realized_pnl: float = 0.0
        self.initial_balance: float | None = None

        # Daily drawdown tracking — MAX_DAILY_DRAWDOWN applies to the
        # realized P&L of the CURRENT UTC day only (resets at midnight),
        # not to the lifetime cumulative P&L.
        self._daily_pnl: float = 0.0
        self._daily_date: str = ""  # "YYYY-MM-DD" (UTC)

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
            self._check_stop_loss_distance(signal, mid_price, current_position_side),
            self._check_take_profit(signal),
            self._check_risk_reward_ratio(signal, mid_price, current_position_side),
        ]

        for check in checks:
            if not check.passed:
                logger.warning("Risk check FAILED: %s", check.reason)
                return check

        logger.info("All risk checks passed")
        return RiskCheck(passed=True, reason="All checks passed")

    def validate_sequence(
        self,
        signal: TradingSignal,
        current_position_size: float = 0.0,
        current_position_side: str = "none",
        mid_price: float = 0.0,
        account_balance: float | None = None,
    ) -> RiskCheck:
        """Validate a multi-action sequence with simulated state transitions.

        Each action is validated against the state that would result from
        executing all prior actions. For example, ["CLOSE", "SHORT"] validates
        CLOSE against the current long position, then SHORT against a simulated
        "no position" state (since CLOSE would have cleared it).

        Falls back to single-action validate() for backward compatibility.
        """
        actions = signal.get_actions()
        sim_side = current_position_side
        sim_size = current_position_size

        for i, action in enumerate(actions):
            temp = TradingSignal(
                action=action,
                confidence=signal.confidence,
                reasoning=signal.reasoning,
                size=signal.size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                modify_sl_to=signal.modify_sl_to,
                modify_tp_to=signal.modify_tp_to,
                key_factors=signal.key_factors,
            )
            check = self.validate(
                temp, sim_size, sim_side, mid_price, account_balance,
            )
            if not check.passed:
                logger.warning(
                    "Risk check FAILED for action %d/%d (%s): %s",
                    i + 1, len(actions), action, check.reason,
                )
                return check

            # Simulate state change from this action for the next iteration
            if action == "CLOSE":
                sim_side = "none"
                sim_size = 0.0
            elif action in ("LONG", "SHORT"):
                sim_side = "long" if action == "LONG" else "short"
                sim_size = signal.size or self.max_position
            # MODIFY_SL, MODIFY_TP, HOLD: state unchanged

        logger.info("All risk checks passed (%d actions)", len(actions))
        return RiskCheck(passed=True, reason=f"All {len(actions)} actions passed")

    # ─── Lifecycle (called from main loop) ────────────────────────────────

    def record_loss(self, pnl: float) -> None:
        """Called when a trade closes at a loss."""
        self.consecutive_losses += 1
        self.total_realized_pnl += pnl
        self._roll_daily()
        self._daily_pnl += pnl
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
        self._roll_daily()
        self._daily_pnl += pnl

    def seed_daily_pnl(self, trades: list) -> None:
        """Seed today's realized P&L from closed trades (startup recovery).

        Called once at startup so the daily drawdown cap is not reset by a
        restart: trades closed earlier today count against today's cap.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_date = today
        self._daily_pnl = 0.0
        for t in trades:
            closed_at = getattr(t, "closed_at", None)
            if closed_at and str(closed_at)[:10] == today:
                self._daily_pnl += getattr(t, "net_pnl", 0.0) or 0.0
        if self._daily_pnl:
            logger.info(
                "Seeded daily P&L: $%.2f (UTC date %s)", self._daily_pnl, today,
            )

    def _roll_daily(self) -> None:
        """Reset the daily P&L counter when the UTC date changes."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_date != today:
            self._daily_date = today
            self._daily_pnl = 0.0

    def tick_cooldown(self) -> None:
        """Per-cycle housekeeping: cooldown countdown + UTC day rollover."""
        self._roll_daily()
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

    # ─── LLM Context ────────────────────────────────────────────────────

    def get_risk_context(
        self,
        mid_price: float = 0.0,
        account_balance: float | None = None,
        current_side: str = "none",
    ) -> str:
        """Build a risk-constraints summary for injection into the LLM prompt.

        This ensures the LLM knows the hard limits BEFORE proposing a trade,
        rather than wasting a cycle on a decision that will be rejected.

        Dynamic values (mid_price, account_balance) are computed from the
        current cycle's market data and plugged into the constraint formulas.
        """
        lines = ["# Risk Constraints (this cycle)\n"]

        # ── Circuit Breaker ──────────────────────────────────────────
        self._roll_daily()
        if self.cooldown_remaining > 0:
            daily_info = ""
            if self.initial_balance and self.initial_balance > 0:
                drawdown = self._daily_pnl / self.initial_balance
                daily_info = f" | Daily P&L: {drawdown:+.1%}"
            lines.append(
                f"⚠️  CIRCUIT BREAKER ACTIVE — {self.consecutive_losses} "
                f"consecutive losses, {self.cooldown_remaining} cycles remaining "
                f"in cooldown{daily_info}.\n"
                f"  → NEW POSITIONS BLOCKED (LONG/SHORT will be rejected).\n"
                f"  → ALLOWED: CLOSE, MODIFY_SL, MODIFY_TP, CANCEL_STALE, HOLD.\n"
                f"  → Do NOT propose LONG or SHORT — they WILL be rejected.\n"
            )
        elif self.consecutive_losses > 0:
            lines.append(
                f"⚡ Circuit breaker: {self.consecutive_losses}/"
                f"{self.MAX_CONSECUTIVE_LOSSES} consecutive losses. "
                f"Next loss triggers {self.COOLDOWN_CYCLES}-cycle cooldown. "
                f"Be more conservative.\n"
            )

        if self.initial_balance and self.initial_balance > 0 and self._daily_pnl < 0:
            drawdown = self._daily_pnl / self.initial_balance
            if drawdown < -0.02:  # approaching -5% hard cap
                lines.append(
                    f"⚠️  Daily drawdown: {drawdown:.1%} (hard cap: "
                    f"{self.MAX_DAILY_DRAWDOWN:.0%}, resets at UTC midnight). "
                    f"Reduce risk.\n"
                )

        # ── Static thresholds ────────────────────────────────────────
        lines.append("## Hard Limits")
        lines.append(f"- Min confidence: {self.min_confidence}")
        lines.append(f"- Max position size: {self.max_position} BTC")
        lines.append(f"- Max leverage: {self.max_leverage}x")
        lines.append(f"- SL min distance: {self.MIN_SL_DISTANCE:.1%} of entry "
                     f"(BTC noise is ~0.3%)")
        lines.append(f"- SL is REQUIRED for LONG/SHORT (no SL → rejected)")
        lines.append(f"- TP is REQUIRED for LONG/SHORT (no TP → rejected — "
                     f"prevents bypassing the R:R gate)")
        lines.append(f"- R:R minimum: {self.MIN_RR_RATIO}:1 "
                     f"(|TP-entry| / |SL-entry|) — always enforced because TP "
                     f"is mandatory")
        lines.append(f"- Taker fees: ~0.07% round-trip (entry+exit are Ioc market orders). "
                     f"Factor into P&L.")
        lines.append("")

        # ── Margin budget (dynamic) ──────────────────────────────────
        if account_balance and account_balance > 0:
            max_margin = account_balance * 0.95
            max_notional = max_margin * self.max_leverage
            lines.append("## Margin Budget (from your account)")
            lines.append(f"- Available balance: ${account_balance:.2f}")
            lines.append(f"- Max margin usable (95%): ${max_margin:.2f}")
            lines.append(f"- Max position notional: ${max_notional:.2f} "
                         f"({max_margin:.2f} × {self.max_leverage}x)")
            if mid_price > 0:
                max_size_margin = max_notional / mid_price
                capped_size = min(max_size_margin, self.max_position)
                lines.append(f"- At current price ${mid_price:.0f}: "
                             f"max size = {max_size_margin:.4f} BTC "
                             f"(capped at {capped_size:.4f} by position limit)")
            lines.append("")

            # ── Risk budget (dynamic) ────────────────────────────────
            max_risk_hard = account_balance * 0.02  # 2%
            max_risk_warn = account_balance * 0.01  # 1%
            lines.append("## Risk Budget (per trade, from your account)")
            lines.append(f"- Max risk per trade: ${max_risk_hard:.2f} "
                         f"(2% of ${account_balance:.2f})")
            lines.append(f"- Risk = |entry - SL| × size  (calculate this BEFORE proposing)")
            lines.append(f"- If risk > ${max_risk_hard:.2f} → HARD REJECT")
            lines.append(f"- If risk > ${max_risk_warn:.2f} → warning (but still allowed)")
            if mid_price > 0:
                # Show example: with a 1% SL distance, max size
                example_sl_dist = mid_price * 0.01
                example_max_size = max_risk_hard / example_sl_dist
                lines.append(f"- Example: at 1% SL distance (${example_sl_dist:.0f}), "
                             f"max size = ${max_risk_hard:.2f} / ${example_sl_dist:.0f} "
                             f"= {example_max_size:.4f} BTC")
            lines.append("")

        # ── Direction constraints ────────────────────────────────────
        lines.append("## Direction Constraints (this cycle)")
        if current_side == "long":
            lines.append("- You HOLD a LONG position.")
            lines.append("  → LONG → REJECTED (already long)")
            lines.append('  → SHORT alone → REJECTED (would corrupt position tracking)')
            lines.append('  → ["CLOSE", "SHORT"] → OK (flip to short)')
            lines.append("  → CLOSE → OK (flatten) | MODIFY_SL / MODIFY_TP / CANCEL_STALE → OK")
        elif current_side == "short":
            lines.append("- You HOLD a SHORT position.")
            lines.append("  → SHORT → REJECTED (already short)")
            lines.append('  → LONG alone → REJECTED (would corrupt position tracking)')
            lines.append('  → ["CLOSE", "LONG"] → OK (flip to long)')
            lines.append("  → CLOSE → OK (flatten) | MODIFY_SL / MODIFY_TP / CANCEL_STALE → OK")
        else:
            lines.append("- No position held.")
            lines.append("  → LONG / SHORT → OK")
            lines.append("  → CLOSE → REJECTED (nothing to close)")
            lines.append("  → MODIFY_SL / MODIFY_TP → REJECTED (no position)")
            lines.append("  → CANCEL_STALE → OK (clean up leftover orders)")

        # ── Expected Value Guidance ────────────────────────────────────
        lines.append("")
        lines.append("## Expected Value (EV) Check")
        lines.append(
            "Round-trip taker fees: ~0.07% of notional (entry + exit are "
            "both market/Ioc orders). Factor fee cost into your P&L estimate."
        )
        lines.append(f"Hard limit: R:R ≥ {self.MIN_RR_RATIO}:1 "
                     f"(|TP - entry| / |SL - entry|). "
                     f"Trades below this ratio are REJECTED.")
        lines.append("")
        lines.append("Before proposing a trade, compute the implied breakeven win-rate:")
        if mid_price > 0:
            # Show an example: assume 1% SL distance, pick a 2:1 reward:risk TP
            example_sl = mid_price * 0.99
            example_tp = mid_price * 1.02
            if current_side == "short":
                example_sl = mid_price * 1.01
                example_tp = mid_price * 0.98
            risk_pct = 1.0  # 1% SL
            reward_pct = 2.0  # 2% TP
            rr = reward_pct / risk_pct
            breakeven = 1.0 / (1.0 + rr)
            lines.append(
                f"Example: Entry=${mid_price:.0f}, SL=${example_sl:.0f} "
                f"({risk_pct:.0f}% risk), TP=${example_tp:.0f} "
                f"({reward_pct:.0f}% reward)"
            )
            lines.append(
                f"R:R = 1:{rr:.1f} → Breakeven win-rate = "
                f"1/(1+{rr:.1f}) = {breakeven:.1%}"
            )
            lines.append(
                f"Your confidence ({self.min_confidence:.2f} min) "
                f"must EXCEED {breakeven:.1%} for positive EV. "
                f"R:R below {self.MIN_RR_RATIO}:1 is HARD REJECTED — "
                f"widen TP or skip the trade."
            )
        lines.append(
            "Formula: breakeven = 1 / (1 + reward/risk). "
            "Your confidence MUST exceed this or the trade has negative EV.\n"
            "CHECK: |entry - SL| = risk per unit. |TP - entry| = reward per unit."
        )

        return "\n".join(lines)

    # ─── Individual Checks ────────────────────────────────────────────────

    def _check_circuit_breaker(self, signal: TradingSignal) -> RiskCheck:
        """Block new positions if circuit breaker is active.

        CLOSE, MODIFY_SL, MODIFY_TP, and CANCEL_STALE are always allowed
        (risk-reducing actions). HOLD is a no-op.
        """
        if signal.action in self._RISK_NEUTRAL_ACTIONS:
            return RiskCheck(passed=True, reason="Not a new position")

        if self.cooldown_remaining > 0:
            return RiskCheck(passed=False, reason=self.get_block_reason())

        # Check daily drawdown if we have a reference balance. The cap
        # applies to TODAY's realized P&L only (rolls over at UTC midnight).
        self._roll_daily()
        if self.initial_balance and self._daily_pnl < 0:
            drawdown = self._daily_pnl / self.initial_balance
            if drawdown < self.MAX_DAILY_DRAWDOWN:
                return RiskCheck(
                    passed=False,
                    reason=f"Max daily drawdown exceeded: {drawdown:.2%}",
                )

        return RiskCheck(passed=True, reason="Circuit breaker OK")

    def _check_confidence(self, signal: TradingSignal) -> RiskCheck:
        if signal.action in self._RISK_NEUTRAL_ACTIONS:
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
        if signal.action in self._RISK_NEUTRAL_ACTIONS:
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
        # Naked opposite entries are rejected: the executor treats LONG/SHORT as
        # a fresh open, which would orphan the existing position's SL/TP orders
        # and pending trade record. A lone ["SHORT"] while long would silently
        # REDUCE (or net out) the real position while the tracker believes it
        # is now short — tracker/chain divergence. Flips must go through
        # ["CLOSE", "..."].
        if signal.action == "SHORT" and current_side == "long":
            return RiskCheck(
                passed=False,
                reason='Already holding LONG — flip via actions ["CLOSE", "SHORT"]',
            )
        if signal.action == "LONG" and current_side == "short":
            return RiskCheck(
                passed=False,
                reason='Already holding SHORT — flip via actions ["CLOSE", "LONG"]',
            )
        if signal.action == "CLOSE" and current_side == "none":
            return RiskCheck(passed=False, reason="No position to close")
        if signal.action == "MODIFY_SL" and current_side == "none":
            return RiskCheck(passed=False, reason="No position to modify SL for")
        if signal.action == "MODIFY_TP" and current_side == "none":
            return RiskCheck(passed=False, reason="No position to modify TP for")

        return RiskCheck(passed=True, reason="Direction OK")

    def _check_stop_loss_distance(
        self,
        signal: TradingSignal,
        mid_price: float,
        current_side: str = "none",
    ) -> RiskCheck:
        """Validate stop loss is on the correct side and at a sane distance.

        Side check: a LONG's SL above entry (or a SHORT's SL below entry)
        would trigger instantly — abs()-only distance checks can't catch it.
        Too-tight stops get hit by market noise; too-wide stops risk too much.
        """
        if signal.action == "MODIFY_SL":
            return self._check_modify_sl_side(signal, mid_price, current_side)
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

        # Wrong-side SL: would trigger the moment it is placed.
        if signal.action == "LONG" and signal.stop_loss >= entry:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Stop loss ${signal.stop_loss:.0f} is ABOVE entry "
                    f"${entry:.0f} for a LONG — it must be below entry "
                    f"(at the invalidation level of your thesis)."
                ),
            )
        if signal.action == "SHORT" and signal.stop_loss <= entry:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Stop loss ${signal.stop_loss:.0f} is BELOW entry "
                    f"${entry:.0f} for a SHORT — it must be above entry "
                    f"(at the invalidation level of your thesis)."
                ),
            )

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

    def _check_modify_sl_side(
        self, signal: TradingSignal, mid_price: float, current_side: str,
    ) -> RiskCheck:
        """A modified SL must sit on the losing side of the CURRENT price.

        Moving a long's SL to/above the current mid (or a short's SL to/below
        it) arms a stop that fires immediately.
        """
        new_sl = signal.modify_sl_to or signal.stop_loss
        if new_sl is None or mid_price <= 0 or current_side == "none":
            return RiskCheck(passed=True, reason="Cannot validate modified SL")

        if current_side == "long" and new_sl >= mid_price:
            return RiskCheck(
                passed=False,
                reason=(
                    f"New SL ${new_sl:.0f} is at/above current price "
                    f"${mid_price:.0f} for a LONG — it would trigger instantly. "
                    f"SL must stay below the current price."
                ),
            )
        if current_side == "short" and new_sl <= mid_price:
            return RiskCheck(
                passed=False,
                reason=(
                    f"New SL ${new_sl:.0f} is at/below current price "
                    f"${mid_price:.0f} for a SHORT — it would trigger instantly. "
                    f"SL must stay above the current price."
                ),
            )
        return RiskCheck(passed=True, reason="Modified SL side OK")

    def _check_take_profit(self, signal: TradingSignal) -> RiskCheck:
        """Require a take profit for new directional trades.

        The R:R >= MIN_RR_RATIO gate is only computable when a TP exists —
        if TP were optional, an LLM could omit it to bypass the hard R:R
        rule. Requiring TP closes that loophole and matches the prompt
        contract (TP is a required field for directional trades).
        """
        if signal.action not in ("LONG", "SHORT"):
            return RiskCheck(passed=True, reason="No TP needed")

        if signal.take_profit is None:
            return RiskCheck(
                passed=False,
                reason=(
                    "Take profit is required for directional trades — "
                    "omitting it would bypass the minimum R:R check. "
                    "Set a realistic TP (R:R >= "
                    f"{self.MIN_RR_RATIO}:1) or output HOLD."
                ),
            )

        return RiskCheck(passed=True, reason="TP OK")

    def _check_risk_reward_ratio(
        self,
        signal: TradingSignal,
        mid_price: float,
        current_side: str = "none",
    ) -> RiskCheck:
        """Validate that the reward/risk ratio meets the minimum threshold.

        Also validates the take profit sits on the winning side of entry —
        a LONG's TP below entry is a guaranteed-loss structure that
        abs()-based R:R math alone would happily accept.

        Only enforced when both stop_loss and take_profit are provided
        (TP is REQUIRED for LONG/SHORT — see _check_take_profit; the
        guard below only fires for legacy/defensive paths).
        """
        if signal.action not in ("LONG", "SHORT"):
            return RiskCheck(passed=True, reason="No R:R needed")

        if signal.stop_loss is None or signal.take_profit is None:
            return RiskCheck(passed=True, reason="TP not set — R:R not enforced")

        entry = signal.entry_price or mid_price
        if entry <= 0 or entry == signal.stop_loss:
            return RiskCheck(passed=True, reason="Cannot validate R:R")

        # Wrong-side TP: breaks the reward premise of the trade.
        if signal.action == "LONG" and signal.take_profit <= entry:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Take profit ${signal.take_profit:.0f} is at/below entry "
                    f"${entry:.0f} for a LONG — TP must be above entry."
                ),
            )
        if signal.action == "SHORT" and signal.take_profit >= entry:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Take profit ${signal.take_profit:.0f} is at/above entry "
                    f"${entry:.0f} for a SHORT — TP must be below entry."
                ),
            )

        risk_distance = abs(entry - signal.stop_loss)
        reward_distance = abs(signal.take_profit - entry)

        if risk_distance <= 0:
            return RiskCheck(passed=True, reason="Zero risk distance")

        rr_ratio = reward_distance / risk_distance

        if rr_ratio < self.MIN_RR_RATIO:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Risk/reward ratio {rr_ratio:.1f}:1 is below minimum "
                    f"{self.MIN_RR_RATIO}:1. "
                    f"Entry=${entry:.0f} SL=${signal.stop_loss:.0f} "
                    f"(risk=${risk_distance:.0f}) "
                    f"TP=${signal.take_profit:.0f} "
                    f"(reward=${reward_distance:.0f}). "
                    f"Widen TP or tighten SL to improve R:R."
                ),
            )

        return RiskCheck(passed=True, reason=f"R:R OK ({rr_ratio:.1f}:1)")
