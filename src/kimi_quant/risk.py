"""Risk management — validates trading signals before execution.

Checks:
  1. Confidence threshold
  2. Position size limits (with dynamic ATR-based sizing)
  3. Margin requirement (position notional / leverage ≤ 95% available balance)
  4. Risk amount (|entry - SL| × size ≤ 2% of balance)
  5. Direction validation (no redundant or naked opposite trades)
  6. Close discipline (block noise CLOSEs churned within one cycle)
  7. Stop loss distance band + side (SL must sit on the losing side of entry)
  8. Take profit side + minimum risk/reward ratio
  9. Expected value (confidence must exceed the breakeven win rate;
     TP must clear round-trip fees)
 10. Trend alignment (counter-trend entries need higher confidence)
 11. Drawdown circuit breaker (pause after consecutive losses)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal
from kimi_quant.notify import notify

logger = logging.getLogger(__name__)

# Hyperliquid base-tier fees, verified against live fills (2026-07/08 CSV:
# every taker fill priced at exactly 4.5bp, maker fills at 1.5bp).
TAKER_FEE_RATE = 0.00045   # 0.045% per taker side
MAKER_FEE_RATE = 0.00015   # 0.015% per maker side


@dataclass
class PositionContext:
    """Snapshot of the currently open position (from PositionTracker).

    Used by the close-discipline gate: CLOSE is blocked while the trade is
    younger than MIN_HOLD_MINUTES AND price hasn't traveled a meaningful
    fraction of the SL distance — the live-data autopsy showed ~1/3 of round
    trips exiting within one 10-min cycle at 1-30bps moves (pure fee churn).
    """

    side: str = "none"        # "long" | "short" | "none"
    entry_price: float = 0.0
    sl_price: float = 0.0
    entry_time: str = ""      # ISO timestamp ("" = unknown, gate fails open)

    @property
    def known(self) -> bool:
        return (
            self.side in ("long", "short")
            and self.entry_price > 0
            and bool(self.entry_time)
        )

    def held_minutes(self) -> float:
        if not self.entry_time:
            return 0.0
        try:
            opened = datetime.fromisoformat(self.entry_time)
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - opened).total_seconds() / 60.0
        except (ValueError, TypeError):
            return 0.0


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

    # Maximum stop loss distance — live data showed realized adverse moves up
    # to 1.28% (worse than the best win at 1.20%): wide stops silently invert
    # the R:R the 1.5:1 gate is supposed to guarantee.
    MAX_SL_DISTANCE = config.max_sl_distance  # 1.5% default

    # Minimum risk/reward ratio: |TP - entry| / |SL - entry| must be ≥ this.
    # Prevents trades where the potential loss dwarfs the potential gain.
    # TP is REQUIRED for LONG/SHORT (see _check_take_profit), so this gate
    # always applies to directional entries.
    MIN_RR_RATIO = 1.5  # reward must be at least 1.5× the risk

    # Close discipline (anti-churn). Values from Config so they can be tuned
    # via env without touching code.
    MIN_HOLD_MINUTES = config.min_hold_minutes
    CLOSE_MIN_MOVE_FRAC = config.close_min_move_frac
    CLOSE_OVERRIDE_CONFIDENCE = config.close_override_confidence

    # Counter-trend entries (opposing BOTH 1h and 4h trend) need this much
    COUNTER_TREND_MIN_CONFIDENCE = config.counter_trend_min_confidence

    # A TP closer to entry than this multiple of the round-trip fee can never
    # pay for its own execution — reject regardless of R:R math.
    MIN_TP_FEE_MULTIPLE = 3.0  # × round-trip taker fee (0.09%) → TP ≥ 0.27%

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
        position_ctx: PositionContext | None = None,
        trend_1h: str = "",
        trend_4h: str = "",
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
            self._check_close_discipline(
                signal, mid_price, current_position_side, position_ctx,
            ),
            self._check_stop_loss_distance(signal, mid_price, current_position_side),
            self._check_take_profit(signal),
            self._check_risk_reward_ratio(signal, mid_price, current_position_side),
            self._check_expected_value(signal, mid_price),
            self._check_trend_alignment(signal, trend_1h, trend_4h),
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
        position_ctx: PositionContext | None = None,
        trend_1h: str = "",
        trend_4h: str = "",
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
                position_ctx=position_ctx,
                trend_1h=trend_1h, trend_4h=trend_4h,
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

    def clamp_size(
        self,
        signal: TradingSignal,
        mid_price: float,
        account_balance: float | None,
    ) -> float:
        """Return the signal's size clamped to every risk budget.

        Live data showed per-trade sizes varying 4× (0.0002–0.0008 BTC) with
        no corresponding risk normalization. Callers should mutate
        signal.size to the returned value BEFORE validation so every trade
        risks approximately the same fraction of the account.

        Budgets applied (most restrictive wins):
          - MAX_POSITION_SIZE (hard cap)
          - margin: notional / leverage ≤ 95% of available balance
          - risk: |entry - SL| × size ≤ 1% of balance (the warn level —
            the 2% hard cap stays a rejection in _check_position_size)
        """
        size = signal.size or self.max_position
        entry = signal.entry_price or mid_price
        if size <= 0 or entry <= 0:
            return max(0.0, min(size, self.max_position))

        size = min(size, self.max_position)

        if account_balance and account_balance > 0:
            max_margin = account_balance * 0.95
            size = min(size, (max_margin * self.max_leverage) / entry)

            if signal.stop_loss:
                risk_per_unit = abs(entry - signal.stop_loss)
                if risk_per_unit > 0:
                    size = min(
                        size, (account_balance * 0.01) / risk_per_unit
                    )

        return size

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
        lines.append(f"- SL distance band: {self.MIN_SL_DISTANCE:.1%}–"
                     f"{self.MAX_SL_DISTANCE:.1%} of entry "
                     f"(BTC noise is ~0.3%)")
        lines.append(f"- SL is REQUIRED for LONG/SHORT (no SL → rejected)")
        lines.append(f"- TP is REQUIRED for LONG/SHORT (no TP → rejected — "
                     f"prevents bypassing the R:R gate)")
        lines.append(f"- R:R minimum: {self.MIN_RR_RATIO}:1 "
                     f"(|TP-entry| / |SL-entry|) — always enforced because TP "
                     f"is mandatory")
        lines.append(f"- Counter-trend (vs BOTH 1h and 4h trend): confidence "
                     f"≥ {self.COUNTER_TREND_MIN_CONFIDENCE:.2f}")
        lines.append(f"- EV gate: confidence must EXCEED the breakeven win "
                     f"rate 1/(1+R:R) — enforced in code, not just requested")
        lines.append(f"- Fees: taker 0.09% round-trip (entry+exit market). "
                     f"With maker entries (passive limit) entry pays 0.015% "
                     f"→ 0.06% round-trip. Factor into P&L.")
        lines.append("")

        # ── Close discipline ─────────────────────────────────────────
        lines.append("## Close Discipline (if you hold a position)")
        lines.append(
            f"CLOSE is REJECTED while the position is younger than "
            f"{self.MIN_HOLD_MINUTES}min AND price has moved less than "
            f"{self.CLOSE_MIN_MOVE_FRAC:.0%} of the SL distance from entry. "
            f"Early CLOSE needs confidence ≥ "
            f"{self.CLOSE_OVERRIDE_CONFIDENCE:.2f}. "
            f"Exits belong to the exchange-side SL/TP orders — manage the "
            f"trade with MODIFY_SL / MODIFY_TP / HOLD instead of churning."
        )
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
            "Round-trip taker fees: ~0.09% of notional (entry + exit are "
            "both market/Ioc orders; maker entries pay ~0.06%). "
            "Factor fee cost into your P&L estimate."
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

    def _check_close_discipline(
        self,
        signal: TradingSignal,
        mid_price: float,
        current_side: str,
        position_ctx: PositionContext | None,
    ) -> RiskCheck:
        """Block noise CLOSEs — the single biggest fee leak in live data.

        A CLOSE within MIN_HOLD_MINUTES of entry is allowed only when:
          - price has traveled ≥ CLOSE_MIN_MOVE_FRAC of the SL distance
            (the trade has actually been tested), or
          - the signal's confidence ≥ CLOSE_OVERRIDE_CONFIDENCE (the LLM is
            emphatic that the thesis is invalidated).
        Exits via the exchange-side SL/TP trigger orders are unaffected —
        this gate only sees LLM-initiated CLOSE actions.

        Fails open when position context is unknown (recovered position,
        missing entry_time) — blocking an exit from a position we don't
        understand would be worse than the churn it prevents.
        """
        if signal.action != "CLOSE" or current_side == "none":
            return RiskCheck(passed=True, reason="No close to discipline-check")

        if position_ctx is None or not position_ctx.known:
            return RiskCheck(
                passed=True, reason="Position context unknown — close allowed"
            )

        held = position_ctx.held_minutes()
        if held >= self.MIN_HOLD_MINUTES:
            return RiskCheck(passed=True, reason=f"Held {held:.0f}min — close allowed")

        move_frac = 0.0
        if position_ctx.sl_price > 0 and mid_price > 0:
            sl_distance = abs(position_ctx.entry_price - position_ctx.sl_price)
            if sl_distance > 0:
                move_frac = abs(mid_price - position_ctx.entry_price) / sl_distance
        if move_frac >= self.CLOSE_MIN_MOVE_FRAC:
            return RiskCheck(
                passed=True,
                reason=f"Price moved {move_frac:.0%} of SL distance — close allowed",
            )

        if signal.confidence >= self.CLOSE_OVERRIDE_CONFIDENCE:
            return RiskCheck(
                passed=True,
                reason=f"High-confidence close ({signal.confidence:.2f}) allowed",
            )

        return RiskCheck(
            passed=False,
            reason=(
                f"CLOSE blocked by close discipline: held {held:.0f}min "
                f"(< {self.MIN_HOLD_MINUTES}min), price moved "
                f"{move_frac:.0%} of SL distance "
                f"(< {self.CLOSE_MIN_MOVE_FRAC:.0%}). "
                f"Let the exchange-side SL/TP do their job, or use "
                f"HOLD / MODIFY_SL. CLOSE now needs confidence ≥ "
                f"{self.CLOSE_OVERRIDE_CONFIDENCE:.2f}."
            ),
        )

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

        # Hard cap: a distant "invalidation" level looks safe in R:R math but
        # risks more per trade than the strategy's edge is worth. Live data:
        # losses up to 1.28% of price vs best win 1.20%.
        if sl_distance > self.MAX_SL_DISTANCE:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Stop loss distance {sl_distance:.2%} is too wide "
                    f"(max {self.MAX_SL_DISTANCE:.2%}). "
                    f"Entry=${entry:.0f} SL=${signal.stop_loss:.0f}. "
                    f"Reduce size and tighten SL, or skip the trade — "
                    f"if the invalidation level is that far away, the "
                    f"setup's R:R cannot survive the fees."
                ),
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

    def _check_expected_value(
        self, signal: TradingSignal, mid_price: float,
    ) -> RiskCheck:
        """Enforce in code what the prompt only used to suggest.

        1. Breakeven win rate: at R:R = reward/risk, a trader needs to win
           1/(1+RR) of the time to break even. The LLM's stated confidence
           is supposed to be that win probability — if it isn't higher than
           breakeven, the trade is negative-EV by the LLM's own numbers.
           (The prompt has asked for this since early versions; live data
           showed 26% realized win rate against a 41% breakeven, so it is
           now a hard gate.)
        2. Fee floor: a TP nearer than MIN_TP_FEE_MULTIPLE × the round-trip
           taker fee cannot pay for its own execution. With R:R ≥ 1.5 and
           SL ≥ 0.5% this rarely binds, but it guards configs that loosen
           those knobs.
        """
        if signal.action not in ("LONG", "SHORT"):
            return RiskCheck(passed=True, reason="No EV needed")

        if signal.stop_loss is None or signal.take_profit is None:
            return RiskCheck(passed=True, reason="TP not set — EV not enforced")

        entry = signal.entry_price or mid_price
        if entry <= 0:
            return RiskCheck(passed=True, reason="Cannot validate EV")

        risk_distance = abs(entry - signal.stop_loss)
        reward_distance = abs(signal.take_profit - entry)
        if risk_distance <= 0:
            return RiskCheck(passed=True, reason="Cannot validate EV")

        rr_ratio = reward_distance / risk_distance
        breakeven = 1.0 / (1.0 + rr_ratio)

        if signal.confidence <= breakeven:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Negative expected value: R:R {rr_ratio:.1f}:1 implies a "
                    f"breakeven win rate of {breakeven:.0%}, but your "
                    f"confidence is only {signal.confidence:.0%}. "
                    f"Either widen TP (higher R:R) or output HOLD."
                ),
            )

        reward_frac = reward_distance / entry
        min_reward = self.MIN_TP_FEE_MULTIPLE * TAKER_FEE_RATE * 2
        if reward_frac < min_reward:
            return RiskCheck(
                passed=False,
                reason=(
                    f"Take profit is only {reward_frac:.2%} from entry — below "
                    f"{min_reward:.2%} ({self.MIN_TP_FEE_MULTIPLE:.0f}× the "
                    f"round-trip taker fee). The trade cannot pay for its own "
                    f"execution. Widen TP or output HOLD."
                ),
            )

        return RiskCheck(
            passed=True,
            reason=f"EV OK (breakeven {breakeven:.0%} < confidence "
                   f"{signal.confidence:.0%})",
        )

    def _check_trend_alignment(
        self, signal: TradingSignal, trend_1h: str, trend_4h: str,
    ) -> RiskCheck:
        """Counter-trend entries must carry extra confidence.

        Live-data pattern: shorts placed into strong uptrends (e.g. shorting
        77,827 on 8/26 right before continuation to 84k+) at ordinary
        confidence. When BOTH the 1h and 4h trends oppose the entry, require
        confidence ≥ COUNTER_TREND_MIN_CONFIDENCE.
        """
        if signal.action not in ("LONG", "SHORT"):
            return RiskCheck(passed=True, reason="No trend check needed")

        if not trend_1h or not trend_4h:
            return RiskCheck(passed=True, reason="Trend data unavailable")

        t1, t4 = trend_1h.lower(), trend_4h.lower()
        if t1 not in ("up", "down") or t4 not in ("up", "down"):
            return RiskCheck(passed=True, reason="Trend neutral/unknown")

        counter = (
            (signal.action == "LONG" and t1 == "down" and t4 == "down")
            or (signal.action == "SHORT" and t1 == "up" and t4 == "up")
        )
        if not counter:
            return RiskCheck(passed=True, reason="Trend aligned/neutral")

        if signal.confidence >= self.COUNTER_TREND_MIN_CONFIDENCE:
            return RiskCheck(
                passed=True,
                reason=f"Counter-trend with high confidence "
                       f"({signal.confidence:.2f}) — allowed",
            )

        return RiskCheck(
            passed=False,
            reason=(
                f"Counter-trend entry: 1h={t1}, 4h={t4} both oppose a "
                f"{signal.action}. Requires confidence ≥ "
                f"{self.COUNTER_TREND_MIN_CONFIDENCE:.2f} "
                f"(got {signal.confidence:.2f}). Wait for the trend to break "
                f"or let this one go."
            ),
        )
