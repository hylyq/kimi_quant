"""Main entry point for Kimi Quant.

Orchestrates the trading loop: fetch data → analyze with LLM → validate risk →
execute trades → log results.

Supports two strategy modes:
- "single": single-agent analysis (KimiLLM)
- "debate": multi-agent debate (LangGraph, stateless graph + JSONL history)
"""

import argparse
import json as _json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any

# Configure logging BEFORE project imports — some modules (e.g. notify)
# run detection logic at import time and must be able to log.
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO"), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from kimi_quant.analytics import TradeLogger
from kimi_quant.config import config
from kimi_quant.data import DataProvider
from kimi_quant.executor import TradeExecutor
from kimi_quant.llm import KimiLLM
from kimi_quant.monitor import FlashReporter, OrderMonitor
from kimi_quant.notify import notify
from kimi_quant.risk import PositionContext, RiskManager
logger = logging.getLogger("kimi_quant")

# Graceful shutdown flag
_shutdown_requested = False


def handle_shutdown(signum: int, frame: object) -> None:
    """Handle SIGINT/SIGTERM gracefully."""
    global _shutdown_requested
    logger.info("Shutdown signal received, finishing current cycle...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# ─── Fail-safe: Unknown Account State ──────────────────────────────────────
# CRITICAL: in live mode, account=None is NOT the same as "no position".
# get_account_snapshot() returns None on API failure, and treating that as
# "position gone" would clear the tracker (sync Case 3) — the bot would then
# believe it is flat and could open a new position ON TOP of the real one.
# When account state is unknown we pause the whole cycle.

_account_outage_since: float | None = None
_ACCOUNT_OUTAGE_NOTIFY_INTERVAL = 900  # seconds — don't spam during outages

_last_sl_missing_notify: float = 0.0
_SL_MISSING_NOTIFY_INTERVAL = 900  # seconds


def _fail_safe_account_unavailable() -> dict:
    """Return a skipped-cycle result when account state is unknown in live mode.

    Without position state we cannot safely sync the tracker or validate
    signals. Trading is paused until the account API recovers.
    """
    global _account_outage_since
    now = time.monotonic()
    if _account_outage_since is None or (
        now - _account_outage_since >= _ACCOUNT_OUTAGE_NOTIFY_INTERVAL
    ):
        _account_outage_since = now
        notify.send(
            "⚠️ Account state unavailable (Hyperliquid API failure)\n"
            "Trading cycles paused for safety until account data recovers."
        )
    logger.critical(
        "Account state UNAVAILABLE — skipping cycle (fail-safe). "
        "No trading while position state is unknown."
    )
    return {
        "status": "skipped",
        "reason": "Account state unavailable (API failure)",
    }


def _account_ok() -> None:
    """Reset the outage notifier once account data is available again."""
    global _account_outage_since
    _account_outage_since = None


def _notify_sl_missing() -> None:
    """Notify about a missing stop loss, throttled (avoid notification spam)."""
    global _last_sl_missing_notify
    now = time.monotonic()
    if now - _last_sl_missing_notify >= _SL_MISSING_NOTIFY_INTERVAL:
        _last_sl_missing_notify = now
        notify.send(
            "⚠️ Stop Loss order missing from exchange\n"
            "Position is unprotected. LLM will attempt to restore."
        )


# ─── Last-Cycle Feedback ────────────────────────────────────────────────────


def _build_last_cycle_feedback(prev: dict | None) -> str:
    """Build a feedback section from the previous cycle's result for LLM context.

    Closes the decision→outcome loop: tells the LLM what it decided last time
    and whether the market validated or negated that decision.
    """
    if not prev:
        return ""

    action = prev.get("signal", "HOLD")
    confidence = prev.get("confidence", 0)
    reason = prev.get("reasoning", "no reasoning recorded")
    status = prev.get("status", "unknown")

    lines = [
        "# 📋 Last Cycle Feedback",
        f"Decision: {action} (confidence={confidence:.2f})",
        f"Reasoning: \"{reason[:200]}\"",
    ]

    if status == "executed":
        lines.append("Outcome: Signal EXECUTED — position is now active or modified.")
        lines.append(
            "Check the Position Memory section above to assess whether the "
            "thesis is playing out."
        )
    elif status == "rejected":
        lines.append(
            f"Outcome: Signal REJECTED by risk control. "
            f"Do NOT repeat the same proposal. Address the rejection reason."
        )
    elif status == "hold":
        lines.append(
            "Outcome: You chose to HOLD. Review the cycle-over-cycle diff above "
            "— has the situation changed enough to justify a new entry?"
        )
    elif status == "failed":
        lines.append("Outcome: Execution FAILED. Check error details and retry if appropriate.")
    else:
        lines.append(f"Outcome: {status}")

    return "\n".join(lines)


# ─── Shared: Cycle Report Preparation ───────────────────────────────────────


class _AccountStateUnavailable(Exception):
    """Internal: live mode could not fetch account state — cycle must skip.

    Raised by _prepare_cycle_report; the caller converts it into a
    fail-safe 'skipped' result via _fail_safe_account_unavailable().
    """


def _prepare_cycle_report(
    data: DataProvider,
    executor: TradeExecutor,
    trade_logger: TradeLogger,
    risk: RiskManager,
    report: dict,
    last_cycle: dict | None,
) -> dict:
    """Log market/account state and inject every LLM context section.

    Shared by the single-agent and debate strategies so both decision-makers
    see identical context: performance, lessons, open orders, SL/TP status,
    risk constraints, position memory, cycle-over-cycle diff and last-cycle
    feedback.

    Raises _AccountStateUnavailable when live mode cannot fetch account
    state — trading on unknown position state is unsafe, so the cycle must
    be skipped (see run_once_* callers).
    """
    market = report.get("market")
    if market:
        logger.info(
            "%s mid=%.1f | spread=%.1f(%.4f%%) | funding=%.4f%% | 24h=%.2f%%",
            config.trading_pair,
            market.mid_price,
            market.spread,
            market.spread_pct,
            market.funding_rate * 100,
            market.day_change_pct,
        )

    account = report.get("account")
    if account:
        logger.info("Account: %s", account.to_summary())
    elif config.dry_run:
        logger.info("Account: NOT AVAILABLE (dry-run — simulation only)")
    else:
        logger.warning("Account: NOT AVAILABLE (API error)")

    # Fail-safe: in live mode, unknown account state means we cannot safely
    # sync the tracker or validate signals — pause the cycle. A transient
    # API failure must never be interpreted as "position gone".
    if not config.dry_run and account is None:
        raise _AccountStateUnavailable()
    _account_ok()

    # Inject performance context for LLM self-reflection
    perf_ctx = trade_logger.get_llm_context()
    if perf_ctx:
        report["performance_context"] = perf_ctx

    # Inject lessons learned from past trades
    lessons_ctx = trade_logger.get_lessons_context()
    if lessons_ctx:
        report["lessons_context"] = lessons_ctx

    # Inject confidence calibration + exit attribution (real trades only)
    calib_ctx = trade_logger.get_calibration_context()
    if calib_ctx:
        report["calibration_context"] = calib_ctx

    # Inject open orders so the decision-maker knows existing SL/TP levels
    orders_summary = executor.tracker.to_orders_summary()
    if orders_summary:
        report["open_orders_summary"] = orders_summary

    # Verify tracked SL/TP orders exist on chain BEFORE analysis
    # so the prompt includes any missing-order warnings
    sl_tp_status = {"sl_missing": False, "tp_missing": False}
    if executor.tracker.has_position():
        open_orders_raw = report.get("open_orders_raw", [])
        sl_tp_status = executor.verify_tracked_orders(open_orders_raw)
        if sl_tp_status["sl_missing"]:
            _notify_sl_missing()
        if sl_tp_status["tp_missing"]:
            logger.warning("TP order missing from exchange — decision-maker will be notified")
    report["sl_tp_status"] = sl_tp_status

    # Inject risk constraints BEFORE analysis so the decision-maker knows
    # the hard limits. This prevents wasted cycles where a signal would be
    # rejected by risk checks — e.g. violating margin budget, risk cap,
    # or proposing a new position while circuit breaker is active.
    mid_price = market.mid_price if market else 0.0
    account_balance = account.available_balance if account else None
    current_side = account.position_side if account else "none"
    risk_context = risk.get_risk_context(
        mid_price=mid_price,
        account_balance=account_balance,
        current_side=current_side,
    )
    report["risk_context"] = risk_context

    # Inject position memory so the decision-maker can validate entry thesis (Step 0)
    if executor.tracker.has_position() and account:
        report["position_memory"] = executor.tracker.to_position_memory(
            unrealized_pnl=account.unrealized_pnl
        )

    # Inject cycle-over-cycle diff so the decision-maker sees what changed
    data.inject_cycle_diff(report)

    # Inject last-cycle feedback for decision→outcome loop
    fb = _build_last_cycle_feedback(last_cycle)
    if fb:
        report["last_cycle_feedback"] = fb

    return report


# ─── Single-Agent Strategy ──────────────────────────────────────────────────


def run_once_single(
    llm: KimiLLM,
    data: DataProvider,
    risk: RiskManager,
    executor: TradeExecutor,
    trade_logger: TradeLogger,
    last_cycle: dict | None = None,
) -> dict:
    """Single-agent analysis → risk check → execute."""
    logger.info("=" * 50)
    logger.info("Starting trading cycle [mode: single-agent]")

    logger.info("Fetching market data...")
    report = data.get_full_report(address=executor.address)

    try:
        report = _prepare_cycle_report(
            data, executor, trade_logger, risk, report, last_cycle
        )
    except _AccountStateUnavailable:
        return _fail_safe_account_unavailable()

    logger.info("Requesting single-agent LLM analysis...")
    signal_result = llm.analyze(report)

    if signal_result is None:
        logger.warning("LLM returned no signal, skipping cycle")
        return {"status": "skipped", "reason": "LLM returned None"}

    # Correction callback — gives the LLM one chance to fix a rejected signal
    def _correct_single(sig: Any, reason: str) -> Any:
        return llm.correct(report, sig, reason)

    return _validate_and_execute(
        signal_result, report, risk, executor, trade_logger,
        correct_fn=_correct_single,
    )


# ─── Multi-Agent Debate Strategy ────────────────────────────────────────────


def run_once_debate(
    strategy: "DebateStrategy",
    data: DataProvider,
    risk: RiskManager,
    executor: TradeExecutor,
    trade_logger: TradeLogger,
    last_cycle: dict | None = None,
) -> dict:
    """Multi-agent debate → risk check → execute."""
    logger.info("=" * 50)
    logger.info("Starting trading cycle [mode: multi-agent debate]")

    logger.info("Fetching market data...")
    report = data.get_full_report(address=executor.address)

    try:
        report = _prepare_cycle_report(
            data, executor, trade_logger, risk, report, last_cycle
        )
    except _AccountStateUnavailable:
        return _fail_safe_account_unavailable()

    logger.info("Launching multi-agent debate...")
    signal_result, transcript = strategy.analyze_sync(report)

    if signal_result is None:
        logger.warning("Debate produced no signal, skipping cycle")
        return {
            "status": "skipped",
            "reason": "Debate returned no signal",
            "transcript": transcript,
        }

    logger.info("Debate transcript (Bull): %s", transcript["bull"][:120])
    logger.info("Debate transcript (Bear): %s", transcript["bear"][:120])
    logger.info("Debate transcript (Hold): %s", transcript["hold"][:120])
    if transcript.get("bull_rebuttal"):
        logger.info("Rebuttal (Bull): %s", transcript["bull_rebuttal"][:120])
    if transcript.get("bear_rebuttal"):
        logger.info("Rebuttal (Bear): %s", transcript["bear_rebuttal"][:120])
    if transcript.get("hold_rebuttal"):
        logger.info("Rebuttal (Hold): %s", transcript["hold_rebuttal"][:120])

    # Correction callback — gives the Judge one chance to fix a rejected signal
    def _correct_debate(sig: Any, reason: str) -> Any:
        return strategy.correct_judge_sync(report, sig, reason, transcript)

    result = _validate_and_execute(
        signal_result, report, risk, executor, trade_logger,
        correct_fn=_correct_debate,
    )
    result["transcript"] = {
        "bull": transcript["bull"],
        "bear": transcript["bear"],
        "hold": transcript["hold"],
    }
    return result


# ─── Shared: Validate & Execute ──────────────────────────────────────────────


def _validate_and_execute(
    signal_result: Any,
    report: dict,
    risk: RiskManager,
    executor: TradeExecutor,
    trade_logger: TradeLogger,
    correct_fn: Any = None,
) -> dict:
    """Risk validation + trade execution — shared by all strategies.

    Args:
        correct_fn: Optional callable(signal, reason) -> corrected_signal | None.
            When set and risk_correction_enabled, a rejected signal gets one
            chance to be corrected by the original decision-maker.
    """

    account = report.get("account")
    market = report.get("market")
    mid_price = market.mid_price if market else 0.0

    # ── Phase 1: Sync chain state ──
    current_size = 0.0
    current_side = "none"
    chain_entry = 0.0
    account_balance = None

    if account:
        current_size = account.position_size
        current_side = account.position_side
        chain_entry = account.entry_price
        account_balance = account.available_balance  # free margin, not total value

    # Capture tracker state BEFORE sync (sync may clear it)
    tracker_was_resting = executor.tracker.has_resting_order()
    tracker_entry_before_sync = executor.tracker.entry_price
    tracker_side_before_sync = executor.tracker.side
    # Position context for the close-discipline gate (holds entry time/SL
    # of the position the LLM might want to CLOSE this cycle)
    position_ctx = PositionContext(
        side=tracker_side_before_sync,
        entry_price=tracker_entry_before_sync,
        sl_price=executor.tracker.sl_price,
        entry_time=executor.tracker.entry_time,
    )

    # Higher-timeframe trends for the counter-trend gate
    analysis = report.get("analysis")
    trend_1h = ""
    trend_4h = ""
    if analysis and getattr(analysis, "timeframes", None):
        for tf in analysis.timeframes:
            if tf.interval == "1h":
                trend_1h = tf.trend
            elif tf.interval == "4h":
                trend_4h = tf.trend

    # Use the smart sync: handles resting→active, active→gone, recovery.
    # CRITICAL: only sync when chain state is actually known. In live mode,
    # account=None (API failure) must NOT be interpreted as "position gone" —
    # that would clear the tracker and let a new entry double the real
    # position (run_once_* already skips the cycle in that case; this guard
    # is defense in depth). In dry-run there is no chain — the tracker is
    # the source of truth, so sync is skipped entirely and positions persist
    # until an explicit CLOSE signal.
    account_known = not config.dry_run and account is not None
    executor.sync_with_chain(
        current_side, current_size, chain_entry, account_known=account_known
    )
    if not account_known:
        logger.debug(
            "Skipping chain sync (dry_run=%s, account_known=%s)",
            config.dry_run, account is not None,
        )

    # If a resting limit order timed out and was cancelled, clean up the trade log
    if tracker_was_resting and not executor.tracker.has_resting_order() \
            and not executor.tracker.has_position():
        if trade_logger.has_pending:
            trade_logger.cancel_pending()
            logger.info("Limit order timed out — cancelled pending trade record")

    # If sync detected a close event, record it
    if not executor.tracker.has_position() and not executor.tracker.has_resting_order():
        # Check if trade_logger has a pending trade that was just closed by SL/TP
        if trade_logger.has_pending and current_side == "none":
            ws_reason, ws_price = executor.tracker.consume_ws_close()
            _record_close_from_chain(
                trade_logger, report,
                entry_price=tracker_entry_before_sync,
                side=tracker_side_before_sync,
                ws_reason=ws_reason,
                ws_price=ws_price,
            )
            # Record the P&L result for circuit breaker tracking
            stats = trade_logger.get_stats()
            if stats.total_trades > 0:
                last_trades = trade_logger.get_all_trades()
                if last_trades:
                    last = last_trades[-1]
                    if last.is_win:
                        risk.record_win(last.net_pnl)
                    else:
                        risk.record_loss(last.net_pnl)

    logger.info("Position: %s", executor.tracker.to_summary())

    # Protective-order safety net: maker entries defer SL/TP until the fill
    # is confirmed, and orders occasionally go missing on chain. This runs
    # every cycle so a confirmed fill gets protected within one (short)
    # cycle instead of waiting for the LLM to notice and emit MODIFY_SL.
    try:
        executor.ensure_protective_orders()
    except Exception as e:
        logger.error("ensure_protective_orders failed (non-fatal): %s", e)

    # Track MAE/MFE for position memory context
    if account and executor.tracker.has_position():
        executor.tracker.update_pnl_extremes(account.unrealized_pnl)

    # Set initial balance for drawdown tracking on first cycle.
    # Uses TOTAL equity — available balance shrinks whenever margin is
    # tied up in a position, which would distort drawdown percentages.
    total_equity = account.balance if account else None
    if risk.initial_balance is None and total_equity and total_equity > 0:
        risk.initial_balance = total_equity
        logger.info("Initial balance set: $%.2f", total_equity)

    # Tick circuit breaker cooldown
    risk.tick_cooldown()

    # ── Phase 2: Risk validation (multi-action aware) ──
    actions = signal_result.get_actions()

    # Size normalization BEFORE validation: clamp to the risk budgets so
    # every trade risks ~the same fraction of the account (live data showed
    # 4× size variance with no risk normalization). Floor to 5 decimals —
    # Hyperliquid's BTC sz precision — so we never round UP past a budget.
    if (any(a in ("LONG", "SHORT") for a in actions)
            and mid_price > 0):
        original_size = signal_result.size
        clamped = risk.clamp_size(signal_result, mid_price, account_balance)
        floored = int(clamped * 1e5) / 1e5  # BTC szDecimals = 5
        if floored > 0 and (original_size is None or floored < original_size):
            signal_result.size = floored
            logger.info(
                "Size normalized by risk budgets: %s → %.5f BTC",
                "max/default" if original_size is None else f"{original_size:.5f}",
                floored,
            )

    def _validate(sig: Any) -> Any:
        """Run risk validation and return the check result."""
        return risk.validate_sequence(
            sig, current_size, current_side,
            mid_price=mid_price, account_balance=account_balance,
            position_ctx=position_ctx,
            trend_1h=trend_1h, trend_4h=trend_4h,
        )

    risk_check = _validate(signal_result)
    if not risk_check.passed:
        logger.warning("Risk check failed: %s", risk_check.reason)

        # Attempt one-shot correction if enabled and a correction callback is provided
        if config.risk_correction_enabled and correct_fn is not None:
            logger.info("Attempting risk correction — asking LLM to fix signal...")
            notify.send(
                f"🔄 Risk correction: asking LLM to fix — {risk_check.reason[:100]}"
            )

            corrected_signal = correct_fn(signal_result, risk_check.reason)

            if corrected_signal is not None:
                corrected_actions = corrected_signal.get_actions()
                risk_check2 = _validate(corrected_signal)

                if risk_check2.passed:
                    logger.info("Risk correction accepted — executing corrected signal")
                    notify.send(
                        f"✅ Risk correction accepted — "
                        f"{'/'.join(corrected_actions)} "
                        f"(was {'/'.join(actions)})"
                    )
                    # Use the corrected signal going forward
                    signal_result = corrected_signal
                    actions = corrected_actions
                else:
                    logger.warning(
                        "Risk correction failed: %s — giving up", risk_check2.reason
                    )
                    notify.send(
                        f"❌ Risk correction failed: {risk_check2.reason}. Giving up."
                    )
                    return {
                        "status": "rejected",
                        "signal": "/".join(corrected_actions),
                        "confidence": corrected_signal.confidence,
                        "reason": risk_check2.reason,
                    }
            else:
                logger.warning("Risk correction returned None — giving up")
                notify.send("❌ Risk correction: LLM unavailable, giving up.")
                return {
                    "status": "rejected",
                    "signal": "/".join(actions),
                    "confidence": signal_result.confidence,
                    "reason": risk_check.reason,
                }
        else:
            # No correction available or disabled — reject immediately
            notify.send(f"🛡️ Risk rejected: {risk_check.reason}")
            return {
                "status": "rejected",
                "signal": "/".join(actions),
                "confidence": signal_result.confidence,
                "reason": risk_check.reason,
            }

    # ── Phase 3: Record trade open intent ──
    trade_opened_this_cycle = False
    has_close = "CLOSE" in actions
    has_open = any(a in ("LONG", "SHORT") for a in actions)
    is_flip = has_close and has_open  # e.g. ["CLOSE", "SHORT"]

    # Determine flip side early — needed in Phase 5 if execution succeeds
    flip_side = ""
    flip_size = 0.0
    flip_entry_price = 0.0
    if is_flip:
        open_action = next((a for a in reversed(actions)
                           if a in ("LONG", "SHORT")), None)
        flip_side = "long" if open_action == "LONG" else "short"
        flip_size = signal_result.size or config.max_position_size
        flip_entry_price = signal_result.entry_price or mid_price

    # Simple open (no CLOSE): record intent before execution so it exists
    # even if the bot restarts mid-cycle. Flips are deferred to Phase 5
    # because the old position must be closed first — opening a new pending
    # trade before the CLOSE executes would discard the old trade's P&L.
    if (has_open
            and not is_flip
            and not executor.tracker.has_resting_order()
            and (not executor.tracker.has_position() or has_close)):
        open_action = next((a for a in reversed(actions)
                           if a in ("LONG", "SHORT")), None)
        side = "long" if open_action == "LONG" else "short"
        size = signal_result.size or config.max_position_size
        entry_price = signal_result.entry_price or mid_price
        trade_logger.open_trade(
            side=side, size=size, entry_price=entry_price,
            dry_run=executor.dry_run,
            entry_confidence=signal_result.confidence,
            planned_sl=signal_result.stop_loss or 0.0,
            planned_tp=signal_result.take_profit or 0.0,
            entry_type=(
                "maker" if (config.entry_order_type == "maker"
                            and not executor.dry_run) else "market"
            ),
        )
        trade_opened_this_cycle = True
        notional = size * entry_price
        margin = notional / config.max_leverage
        total_balance = account.balance if account else 0.0
        action_label = " → ".join(actions)
        notify.send(
            f"📈 {action_label} {size} BTC @ ${entry_price:.0f}\n"
            f"Notional: ${notional:.0f} | Margin: ${margin:.2f} | Leverage: {config.max_leverage}x\n"
            f"SL: ${signal_result.stop_loss or 0:.0f} | TP: ${signal_result.take_profit or 0:.0f}\n"
            f"Confidence: {signal_result.confidence:.2f} | Balance: ${total_balance:.2f}"
        )

    # ── Phase 4: Execute ──
    result = executor.execute(signal_result)
    logger.info("Execution result: %s | Position: %s",
                result, executor.tracker.to_summary())

    # If execution failed after opening a pending trade this cycle, clean it up
    if has_open and trade_opened_this_cycle and not result.get("executed"):
        trade_logger.cancel_pending()
        logger.warning("Execution failed — cancelled pending trade record")

    # ── Phase 5: Post-execution recording ──
    # Check individual results for CLOSE (multi-action or single)
    results_list = result.get("results", [result])
    close_was_executed = any(
        r.get("action") == "CLOSE" and r.get("executed")
        for r in results_list
    )
    open_was_executed = any(
        r.get("action") in ("LONG", "SHORT") and r.get("executed")
        for r in results_list
    )

    if is_flip and close_was_executed:
        # Flip: close the OLD trade first, then open the new one.
        # Both operations happen AFTER execution succeeds, so a failed
        # CLOSE won't orphan a trade record.
        exit_price = _get_exit_price(report)
        closed_trade = trade_logger.close_trade(exit_price, reason="signal")
        if closed_trade:
            if closed_trade.is_win:
                risk.record_win(closed_trade.net_pnl)
            else:
                risk.record_loss(closed_trade.net_pnl)
            emoji = "🟢" if closed_trade.is_win else "🔴"
            total_balance = account.balance if account else 0.0
            notify.send(
                f"{emoji} Position closed: {closed_trade.side.upper()}\n"
                f"Entry: ${closed_trade.entry_price:.0f} → Exit: ${closed_trade.exit_price:.0f}\n"
                f"P&L: ${closed_trade.net_pnl:+.2f} ({closed_trade.pnl_pct:+.2f}%) | "
                f"Leverage: {config.max_leverage}x\n"
                f"Reason: {closed_trade.close_reason} | Balance: ${total_balance:.2f}"
            )

        if open_was_executed:
            trade_logger.open_trade(
                side=flip_side, size=flip_size,
                entry_price=flip_entry_price,
                dry_run=executor.dry_run,
                entry_confidence=signal_result.confidence,
                planned_sl=signal_result.stop_loss or 0.0,
                planned_tp=signal_result.take_profit or 0.0,
                entry_type=(
                    "maker" if (config.entry_order_type == "maker"
                                and not executor.dry_run) else "market"
                ),
            )
            trade_opened_this_cycle = True
            notional = flip_size * flip_entry_price
            margin = notional / config.max_leverage
            total_balance = account.balance if account else 0.0
            action_label = " → ".join(actions)
            notify.send(
                f"📈 {action_label} {flip_size} BTC @ ${flip_entry_price:.0f}\n"
                f"Notional: ${notional:.0f} | Margin: ${margin:.2f} | "
                f"Leverage: {config.max_leverage}x\n"
                f"SL: ${signal_result.stop_loss or 0:.0f} | TP: ${signal_result.take_profit or 0:.0f}\n"
                f"Confidence: {signal_result.confidence:.2f} | Balance: ${total_balance:.2f}"
            )

    elif close_was_executed:
        # Simple close (no flip): close the pending trade normally.
        exit_price = _get_exit_price(report)
        trade = trade_logger.close_trade(exit_price, reason="signal")
        if trade:
            if trade.is_win:
                risk.record_win(trade.net_pnl)
            else:
                risk.record_loss(trade.net_pnl)
            emoji = "🟢" if trade.is_win else "🔴"
            total_balance = account.balance if account else 0.0
            notify.send(
                f"{emoji} Position closed: {trade.side.upper()}\n"
                f"Entry: ${trade.entry_price:.0f} → Exit: ${trade.exit_price:.0f}\n"
                f"P&L: ${trade.net_pnl:+.2f} ({trade.pnl_pct:+.2f}%) | Leverage: {config.max_leverage}x\n"
                f"Reason: {trade.close_reason} | Balance: ${total_balance:.2f}"
            )

    # Entry price is already set in open_trade() from signal.entry_price
    # or approximated from mid_price. The next cycle's sync_with_chain()
    # will update it from actual chain fill data.

    # Derive a human-readable status
    if actions == ["HOLD"]:
        cycle_status = "hold"
    elif result.get("executed"):
        cycle_status = "executed"
    else:
        cycle_status = "failed"

    return {
        "status": cycle_status,
        "signal": "/".join(actions),
        "confidence": signal_result.confidence,
        "reasoning": signal_result.reasoning,
        "execution": result,
        "next_interval": getattr(signal_result, "next_interval", None),
    }


def _get_exit_price(report: dict) -> float:
    """Extract current mid price as exit price from the market report."""
    market = report.get("market")
    if market:
        return market.mid_price
    return 0.0


def _record_close_from_chain(
    trade_logger: TradeLogger,
    report: dict,
    entry_price: float = 0.0,
    side: str = "none",
    ws_reason: str | None = None,
    ws_price: float = 0.0,
) -> None:
    """Record a trade close detected from on-chain state (SL/TP filled).

    Args:
        trade_logger: The trade logger instance.
        report: Market data report (fallback exit price = cycle-start mid).
        entry_price: The entry price BEFORE the tracker was cleared.
        side: The position side BEFORE the tracker was cleared.
        ws_reason: "stop_loss" or "take_profit" from WebSocket event,
            if the WS cleared the tracker before this call. Takes
            priority over price-based inference.
        ws_price: Actual trigger fill price from the WebSocket event.
            Takes priority over the (stale) cycle-start mid — with adaptive
            intervals the cycle may run long after the SL actually fired.
    """
    market = report.get("market")
    exit_price = ws_price if ws_price > 0 else (market.mid_price if market else 0.0)

    # Priority: 1) WS event reason (most accurate) → 2) price inference
    if ws_reason:
        reason = ws_reason
    elif entry_price > 0 and exit_price > 0 and side != "none":
        if side == "long":
            reason = "take_profit" if exit_price > entry_price else "stop_loss"
        else:
            reason = "take_profit" if exit_price < entry_price else "stop_loss"
    else:
        reason = "manual"

    trade_logger.close_trade(exit_price, reason=reason)


# ─── Main Loop ───────────────────────────────────────────────────────────────


def run_loop():
    """Main trading loop — runs at configured intervals.

    Multi-layer error protection:
      - Startup errors: logged and raised (must fix config)
      - Per-cycle errors: caught, logged, loop continues
      - Sleep errors: caught, loop continues
      - Fatal (KeyboardInterrupt, SystemExit): clean shutdown
    """
    mode = config.strategy_mode
    logger.info("=" * 50)
    logger.info("Kimi Quant — %s Perpetual Contract Trading", config.trading_pair)
    logger.info("Model: %s | Mode: %s | Interval: %ds | Dry Run: %s",
                config.display_model, mode,
                config.trading_interval_seconds, config.dry_run)
    if mode == "debate" and config.judge_primary_llm:
        judge_model_desc = config.judge_model or (
            config.kimi_model if config.judge_primary_llm == "kimi"
            else config.deepseek_model
        )
        judge_effort = config.judge_reasoning_effort or config.reasoning_effort
        logger.info(
            "Judge: provider=%s model=%s effort=%s",
            config.judge_primary_llm, judge_model_desc, judge_effort,
        )
    logger.info("Risk: min_confidence=%.2f | max_position=%.4f BTC | max_leverage=%dx",
                config.min_confidence, config.max_position_size, config.max_leverage)
    logger.info("=" * 50)

    # ── Startup ──────────────────────────────────────────────────────────
    try:
        config.validate()
        data = DataProvider()
        risk = RiskManager()
        executor = TradeExecutor()
        trade_logger = TradeLogger()
    except Exception as e:
        logger.critical("Startup failed: %s", e, exc_info=True)
        notify.send(f"❌ Kimi Quant startup failed: {e}")
        raise  # can't recover from startup failures

    # ── Real-time Order Monitor (WebSocket + Flash LLM) ─────────────────
    monitor: OrderMonitor | None = None
    reporter: FlashReporter | None = None
    if config.monitor_enabled and not config.dry_run:
        try:
            base_url = (
                "https://api.hyperliquid-testnet.xyz"
                if config.hl_testnet
                else config.hl_base_url
            )
            monitor = OrderMonitor(
                base_url=base_url,
                address=executor.address,
                tracker=executor.tracker,
            )
            reporter = FlashReporter(
                event_queue=monitor.events,
                model=config.monitor_flash_model,
                api_key=config.monitor_flash_api_key or None,
                base_url=config.monitor_flash_base_url or None,
                tracker=executor.tracker,  # for SL/TP/entry classification
            )
            monitor.start()
            reporter.start()
            logger.info(
                "Order monitor active (flash_model=%s, llm=%s)",
                config.monitor_flash_model,
                "enabled" if config.monitor_flash_api_key or config.deepseek_api_key
                else "fallback-only",
            )
        except Exception as e:
            logger.warning("Failed to start order monitor (non-fatal): %s", e)

    notify.send(
        f"🚀 Kimi Quant started\n"
        f"Mode: {mode} | Dry Run: {config.dry_run}\n"
        f"Primary: {config.primary_llm} | {config.trading_interval_seconds}s interval"
    )

    # If executor recovered a position from chain that isn't in the trade log,
    # create a pending trade so the eventual close is recorded correctly
    if executor.tracker.has_position() and not trade_logger.has_pending:
        trade_logger.recover_trade(
            side=executor.tracker.side,
            size=executor.tracker.size,
            entry_price=executor.tracker.entry_price,
        )
        logger.info(
            "Recovered trade from chain for existing position: %s %.4f @ $%.1f",
            executor.tracker.side, executor.tracker.size,
            executor.tracker.entry_price,
        )

    # Seed circuit breaker from history
    stats = trade_logger.get_stats()
    if stats.total_trades > 0:
        logger.info(
            "Loaded trade history: %d trades | Win Rate: %.1f%% | Net P&L: $%.2f",
            stats.total_trades, stats.win_rate, stats.net_pnl,
        )
        all_trades = trade_logger.get_all_trades()
        recent = all_trades[-10:]
        consecutive = 0
        for t in reversed(recent):
            if not t.is_win:
                consecutive += 1
            else:
                break
        risk.consecutive_losses = consecutive
        risk.total_realized_pnl = stats.net_pnl
        # Seed TODAY's realized P&L so the daily drawdown cap survives restarts
        # (pre-restart losses today still count against today's cap)
        risk.seed_daily_pnl(trade_logger.get_all_trades())
        if consecutive > 0:
            logger.info("Seeded circuit breaker: %d consecutive losses from history",
                        consecutive)

    # Debate mode: create strategy ONCE (agents/LLM clients are reused;
    # the graph itself is stateless — history lives in debate.jsonl)
    strategy = None
    if mode == "debate":
        from kimi_quant.debate import DebateStrategy
        strategy = DebateStrategy()
        latest = strategy.get_latest_state()
        if latest:
            logger.info(
                "Recovered state from prior run (cycle_id=%s)",
                latest.get("cycle_id", "unknown"),
            )

    llm = KimiLLM() if mode == "single" else None

    # ── Main Loop ────────────────────────────────────────────────────────
    cycle_count = 0
    next_interval = config.trading_interval_seconds
    last_cycle: dict | None = None  # carries last cycle's result for feedback loop

    MIN_INTERVAL = config.min_interval      # default 300s (5 min) — cost control
    MAX_INTERVAL = config.max_interval      # default 10800s (3h) — don't drift too far

    while not _shutdown_requested:
        cycle_count += 1
        start_time = time.monotonic()

        # ═══ Layer 1: Per-cycle protection ═══
        try:
            if mode == "debate":
                assert strategy is not None
                result = run_once_debate(strategy, data, risk, executor, trade_logger,
                                         last_cycle=last_cycle)
            else:
                assert llm is not None
                result = run_once_single(llm, data, risk, executor, trade_logger,
                                         last_cycle=last_cycle)

            result["cycle"] = cycle_count
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["mode"] = mode

            status = result.get("status", "unknown")
            sig = result.get("signal", "N/A")
            conf = result.get("confidence", 0)
            logger.info(
                "Cycle %d complete: mode=%s status=%s signal=%s confidence=%.2f",
                cycle_count, mode, status, sig, conf,
            )

            # Store for next cycle's feedback loop
            last_cycle = result

            # Adaptive interval: LLM decides when to wake next
            llm_interval = result.get("next_interval")
            if llm_interval and isinstance(llm_interval, (int, float)):
                bounded = max(MIN_INTERVAL, min(MAX_INTERVAL, int(llm_interval)))
                if bounded != next_interval:
                    logger.info(
                        "LLM adjusted interval: %ds → %ds", next_interval, bounded
                    )
                next_interval = bounded
            else:
                next_interval = config.trading_interval_seconds

            # A resting maker entry needs a fast fill check: cap the sleep so
            # a fill is confirmed and SL/TP protection goes up within ~2min
            # instead of one full interval.
            if executor.tracker.has_resting_order() and next_interval > 120:
                logger.info(
                    "Resting entry outstanding — capping sleep at 120s "
                    "for fill confirmation"
                )
                next_interval = 120

        except Exception:
            logger.error("Cycle %d failed — continuing", cycle_count, exc_info=True)
            # Notify on first error, then every 10th to avoid spam
            if cycle_count == 1 or cycle_count % 10 == 0:
                notify.send(f"⚠️ Cycle {cycle_count} failed — check logs")

        # ═══ Layer 2: Sleep protection ═══
        try:
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, next_interval - elapsed)
            if not _shutdown_requested and sleep_time > 0:
                logger.info("Sleeping %.1fs until next cycle...", sleep_time)
                tick = min(10, sleep_time)
                while sleep_time > 0 and not _shutdown_requested:
                    time.sleep(min(tick, sleep_time))
                    sleep_time -= tick
        except Exception:
            logger.error("Sleep interrupted — continuing", exc_info=True)

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down. Total cycles: %d", cycle_count)

    # Stop order monitor (reporter first — it consumes the monitor's queue)
    if reporter is not None:
        reporter.stop()
    if monitor is not None:
        monitor.stop()

    stats = trade_logger.get_stats()
    if stats.total_trades > 0:
        logger.info(
            "Session P&L: %d trades | Win Rate: %.1f%% | Net P&L: $%.2f",
            stats.total_trades, stats.win_rate, stats.net_pnl,
        )
        notify.send(
            f"⏹️ Kimi Quant stopped\n"
            f"Cycles: {cycle_count} | Trades: {stats.total_trades}\n"
            f"Win: {stats.win_rate:.0f}% | P&L: ${stats.net_pnl:+.2f}"
        )
    else:
        notify.send(f"⏹️ Kimi Quant stopped — {cycle_count} cycles, no trades")

    if strategy is not None:
        strategy.close()
    notify.shutdown()
    logger.info("Session complete.")


def cmd_stats():
    """Print trade P&L statistics."""
    trade_logger = TradeLogger()
    all_trades = trade_logger.get_all_trades()
    real_trades = [t for t in all_trades if not t.dry_run]
    sim_trades = [t for t in all_trades if t.dry_run]

    if not all_trades:
        print("No trade history found.")
        return

    def _print_stats(label: str, trades: list):
        from kimi_quant.analytics import TradeLogger as TL
        stats = TL._compute_stats(trades)
        if stats.total_trades == 0:
            return
        print(f"=== {label} ===")
        print(f"Total Trades:    {stats.total_trades}")
        print(f"Wins:            {stats.wins}")
        print(f"Losses:          {stats.losses}")
        print(f"Win Rate:        {stats.win_rate:.1f}%")
        print(f"")
        print(f"Gross P&L:       ${stats.total_pnl:+.2f}")
        print(f"Total Fees:      ${stats.total_fees:.2f}")
        print(f"Net P&L:         ${stats.net_pnl:+.2f}")
        print(f"")
        print(f"Avg Win:         ${stats.avg_win:+.2f}")
        print(f"Avg Loss:        ${stats.avg_loss:+.2f}")
        print(f"Largest Win:     ${stats.largest_win:+.2f}")
        print(f"Largest Loss:    ${stats.largest_loss:+.2f}")
        print(f"Profit Factor:   {stats.profit_factor:.2f}")
        print()

    if real_trades:
        _print_stats("Trading Performance (Real)", real_trades)
    if sim_trades:
        _print_stats("Trading Performance (Simulated / Dry-Run)", sim_trades)

    # Recent trades
    print("=== Recent Trades ===")
    for t in all_trades[-10:]:
        tag = "[SIM]" if t.dry_run else "[LIVE]"
        conf = f"conf={t.entry_confidence:.2f}" if t.entry_confidence else "conf=N/A"
        print(
            f"{t.opened_at[:19]} {tag} | {t.side.upper():5s} | "
            f"in: ${t.entry_price:>8.1f} → out: ${t.exit_price:>8.1f} | "
            f"P&L: ${t.pnl:+7.2f} ({t.pnl_pct:+.2f}%) | "
            f"{t.close_reason} | {conf}"
        )

    # Confidence calibration + exit attribution (real trades only)
    calib = trade_logger.get_calibration_context()
    if calib:
        print()
        print("=== Confidence Calibration & Exit Attribution ===")
        print(calib.strip())


def cmd_history():
    """Print the full debate history from the checkpoint database."""
    from kimi_quant.debate import DebateStrategy

    strategy = DebateStrategy()
    try:
        history = strategy.get_history()

        if not history:
            print("No debate history found.")
            return

        print(f"=== Debate History ({len(history)} cycles) ===\n")
        for i, entry in enumerate(history, 1):
            print(f"--- Cycle {i}: {entry.get('cycle_id', '?')} ---")
            print(f"Account: {entry.get('account_summary', 'N/A')[:100]}")
            print(f"Bull: {entry.get('bull_argument', 'N/A')[:200]}")
            print(f"Bear: {entry.get('bear_argument', 'N/A')[:200]}")
            print(f"Hold: {entry.get('hold_argument', 'N/A')[:200]}")
            # Show rebuttals if present (debate rebuttal round enabled)
            if entry.get("bull_rebuttal"):
                print(f"Bull 🗣️: {entry['bull_rebuttal'][:200]}")
            if entry.get("bear_rebuttal"):
                print(f"Bear 🗣️: {entry['bear_rebuttal'][:200]}")
            if entry.get("hold_rebuttal"):
                print(f"Hold 🗣️: {entry['hold_rebuttal'][:200]}")
            if entry.get("final_signal_json"):
                try:
                    sig = _json.loads(entry["final_signal_json"])
                    print(f"Verdict: {sig['action']} (confidence={sig['confidence']})")
                    print(f"Reasoning: {sig['reasoning'][:200]}")
                except Exception:
                    print(f"Verdict (raw): {entry['final_signal_json'][:200]}")
            if entry.get("error"):
                print(f"ERROR: {entry['error']}")
            print()
    finally:
        strategy.close()


def _get_hl_info():
    """Create an Info instance for read-only chain queries."""
    from eth_account import Account
    from hyperliquid.info import Info

    config.validate()
    if not config.hl_private_key:
        raise ValueError(
            "HYPERLIQUID_PRIVATE_KEY is required. Set it in .env."
        )

    account = Account.from_key(config.hl_private_key)
    base_url = (
        "https://api.hyperliquid-testnet.xyz"
        if config.hl_testnet
        else config.hl_base_url
    )
    info = Info(base_url=base_url, skip_ws=True)
    return info, account.address, base_url


def _classify_order(
    is_buy: bool,
    limit_px: float,
    pos_side: str,
    pos_entry: float,
    mid_price: float,
) -> str:
    """Infer order type from context (openOrders API lacks orderType field).

    With an active position:
      - Opposite-side order above entry (long) / below entry (short) → Take Profit
      - Opposite-side order below entry (long) / above entry (short) → Stop Loss
      - Same-side order → Limit (add to position)

    Without a position:
      - BUY below market → Limit (entry long)
      - SELL above market → Limit (entry short)
      - BUY above market or SELL below market → Limit (aggressive)
    """
    if pos_side and pos_entry > 0:
        # We have a position — classify relative to entry
        is_opposite = (pos_side == "long" and not is_buy) or (pos_side == "short" and is_buy)

        if is_opposite:
            if pos_side == "long":
                if limit_px > pos_entry:
                    return "Take Profit 📈"
                else:
                    return "Stop Loss 🛑"
            else:  # short
                if limit_px < pos_entry:
                    return "Take Profit 📈"
                else:
                    return "Stop Loss 🛑"
        else:
            # Same-side order — adding to position or new entry
            return "Limit (entry)"

    # No position — classify relative to market
    if is_buy:
        if limit_px < mid_price:
            return "Limit (buy below mkt)"
        else:
            return "Limit (buy above mkt)"
    else:
        if limit_px > mid_price:
            return "Limit (sell above mkt)"
        else:
            return "Limit (sell below mkt)"


def cmd_status():
    """Print live account status: balance, position, open orders, market price."""
    from hyperliquid.info import Info

    try:
        info, address, base_url = _get_hl_info()
    except ValueError as e:
        print(f"Error: {e}")
        return

    coin = config.trading_pair
    testnet_label = "TESTNET" if config.hl_testnet else "Mainnet"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print()
    print("═" * 60)
    print(f"  Kimi Quant — Account Status")
    print(f"  Address: {address[:6]}...{address[-4:]}")
    print(f"  Network: {testnet_label} | {now}")
    print("═" * 60)

    # ── Fetch data ────────────────────────────────────────────────────
    try:
        from kimi_quant.data import retry_api_call

        user_state = retry_api_call(
            lambda: info.user_state(address),
            description="user_state",
        )
        open_orders = retry_api_call(
            lambda: info.open_orders(address),
            description="open_orders",
        )
        all_mids = retry_api_call(
            lambda: info.all_mids(),
            description="all_mids",
        )
    except Exception as e:
        print(f"\n  ❌ Failed to fetch data: {e}")
        return

    mid_price = float(all_mids.get(coin, 0))
    margin_summary = user_state.get("marginSummary", {})
    total_balance = float(margin_summary.get("accountValue", 0))
    margin_used = float(margin_summary.get("totalMarginUsed", 0))
    available = max(0.0, total_balance - margin_used)

    # ── Account Balance ───────────────────────────────────────────────
    print()
    print("  💰 Account Balance")
    print(f"  Total Value:     ${total_balance:,.2f}")
    print(f"  Available:       ${available:,.2f}")
    print(f"  Margin Used:     ${margin_used:,.2f}")
    if total_balance > 0:
        print(f"  Margin Ratio:    {margin_used/total_balance*100:.1f}%")

    # ── Position ──────────────────────────────────────────────────────
    positions = user_state.get("assetPositions", [])
    pos_data = None
    for p in positions:
        if p["position"]["coin"] == coin:
            pos_data = p["position"]
            break

    print()
    if pos_data:
        raw_size = float(pos_data.get("szi", 0))
        if raw_size > 0:
            side = "LONG"
            size = raw_size
        elif raw_size < 0:
            side = "SHORT"
            size = abs(raw_size)
        else:
            side = "NONE"
            size = 0.0

        entry_px = float(pos_data.get("entryPx", 0))
        u_pnl = float(pos_data.get("unrealizedPnl", 0))
        leverage_raw = pos_data.get("leverage", {})
        if isinstance(leverage_raw, dict):
            leverage = int(leverage_raw.get("value", 1))
        else:
            leverage = int(leverage_raw) if leverage_raw else 1

        notional = size * entry_px if entry_px > 0 else 0

        print("  📊 Position")
        print(f"  Side:            {side}")
        print(f"  Size:            {size:.4f} {coin}")
        if entry_px > 0:
            print(f"  Entry Price:     ${entry_px:,.2f}")
        if mid_price > 0:
            print(f"  Mark Price:      ${mid_price:,.2f}")
        print(f"  Unrealized PnL:  ${u_pnl:+,.2f}", end="")
        if notional > 0 and u_pnl != 0:
            print(f" ({u_pnl/notional*100:+.2f}%)", end="")
        print()
        print(f"  Leverage:        {leverage}x")
        if notional > 0:
            print(f"  Notional:        ${notional:,.2f}")
    else:
        print("  📊 Position:     No position")

    # ── Open Orders ───────────────────────────────────────────────────
    our_orders = [o for o in open_orders if o.get("coin") == coin]
    print()
    print(f"  📝 Open Orders ({len(our_orders)})")

    # Determine position side/entry for order classification
    pos_side = ""
    pos_entry = 0.0
    if pos_data:
        raw_sz = float(pos_data.get("szi", 0))
        pos_side = "long" if raw_sz > 0 else "short"
        pos_entry = float(pos_data.get("entryPx", 0))

    if our_orders:
        # Header
        print(f"  {'OID':<14} {'Type':<22} {'Side':<6} {'Size':<12} {'Price':<14}")
        print(f"  {'─'*14} {'─'*22} {'─'*6} {'─'*12} {'─'*14}")

        for o in our_orders:
            oid = str(o.get("oid", "?"))
            sz = float(o.get("sz", 0))
            limit_px = float(o.get("limitPx", 0))
            side_raw = o.get("side", "")
            is_buy = side_raw == "B"

            side_label = "BUY" if is_buy else ("SELL" if side_raw == "A" else side_raw)

            # Infer order type from context (openOrders API has no orderType field).
            # Strategy: compare side and price to position/market to guess intent.
            type_label = _classify_order(
                is_buy=is_buy,
                limit_px=limit_px,
                pos_side=pos_side,
                pos_entry=pos_entry,
                mid_price=mid_price,
            )

            print(
                f"  {oid:<14} {type_label:<22} {side_label:<6} "
                f"{sz:<12.4f} ${limit_px:<13,.2f}"
            )
    else:
        print("  (none)")

    # ── Market ────────────────────────────────────────────────────────
    print()
    print("  📈 Market")
    if mid_price > 0:
        print(f"  {coin} Mid Price:  ${mid_price:,.2f}")

    # 24h change / funding / OI from the full market snapshot
    try:
        from kimi_quant.data import DataProvider
        dp = DataProvider()
        snapshot = dp.get_market_snapshot()
        if snapshot:
            print(f"  24h Change:      {snapshot.day_change_pct:+.2f}%")
            print(f"  Funding Rate:    {snapshot.funding_rate*100:.4f}%")
            print(f"  Open Interest:   ${snapshot.open_interest:,.0f}")
    except Exception:
        pass

    print()
    print("═" * 60)
    print()


def main():
    """Parse arguments and launch the trading loop."""
    parser = argparse.ArgumentParser(
        description="Kimi Quant — LLM-based BTC Perpetual Trading"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis cycle and exit (no loop)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Trading interval in seconds (overrides env/config)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "debate"],
        default=None,
        help="Strategy mode: single-agent or multi-agent debate",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print persisted debate history and exit",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print trade P&L statistics and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print live account status: balance, position, open orders, market price",
    )
    parser.add_argument(
        "--deposit",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="Deposit USDC from Arbitrum to Hyperliquid (experimental, use official UI instead)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts (use with --deposit for scripting)",
    )
    parser.add_argument(
        "--spot-to-perp",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="Transfer USDC from spot to perp account and exit",
    )
    parser.add_argument(
        "--set-account-type",
        type=str,
        default=None,
        metavar="MODE",
        choices=["manual", "unified", "portfolio"],
        help="Change Hyperliquid account type (manual/unified/portfolio)",
    )
    parser.add_argument(
        "--arb-balance",
        action="store_true",
        help="Check Arbitrum USDC/ETH balances and exit",
    )
    args = parser.parse_args()

    if args.set_account_type is not None:
        from kimi_quant.deposit import cmd_set_account_type
        cmd_set_account_type(args.set_account_type, force=args.force)
        return

    if args.spot_to_perp is not None:
        from kimi_quant.deposit import cmd_spot_to_perp
        cmd_spot_to_perp(args.spot_to_perp, force=args.force)
        return

    if args.arb_balance:
        from kimi_quant.deposit import check_balance, _get_account
        usdc, eth = check_balance()
        acct = _get_account()
        print(f"Arbitrum balances for {acct.address}:")
        print(f"  USDC: {usdc:.2f}")
        print(f"  ETH:  {eth:.6f}")
        return

    if args.deposit is not None:
        from kimi_quant.deposit import cmd_deposit
        cmd_deposit(args.deposit, force=args.force)
        return

    if args.history:
        cmd_history()
        return

    if args.stats:
        cmd_stats()
        return

    if args.status:
        cmd_status()
        return

    if args.interval:
        config.trading_interval_seconds = args.interval
    if args.mode:
        config.strategy_mode = args.mode

    if args.once:
        config.validate()
        data = DataProvider()
        risk = RiskManager()
        executor = TradeExecutor()
        trade_logger = TradeLogger()

        if config.strategy_mode == "debate":
            from kimi_quant.debate import DebateStrategy
            strategy = DebateStrategy()
            try:
                result = run_once_debate(strategy, data, risk, executor, trade_logger)
            finally:
                strategy.close()
        else:
            llm = KimiLLM()
            result = run_once_single(llm, data, risk, executor, trade_logger)

        print(_json.dumps(result, indent=2, default=str))
    else:
        run_loop()


if __name__ == "__main__":
    main()
