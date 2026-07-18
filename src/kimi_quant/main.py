"""Main entry point for Kimi Quant.

Orchestrates the trading loop: fetch data → analyze with LLM → validate risk →
execute trades → log results.

Supports two strategy modes:
- "single": single-agent analysis (KimiLLM)
- "debate": multi-agent debate with LangGraph checkpointing
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
from kimi_quant.notify import notify
from kimi_quant.risk import RiskManager
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


# ─── Single-Agent Strategy ──────────────────────────────────────────────────


def run_once_single(
    llm: KimiLLM,
    data: DataProvider,
    risk: RiskManager,
    executor: TradeExecutor,
    trade_logger: TradeLogger,
) -> dict:
    """Single-agent analysis → risk check → execute."""
    logger.info("=" * 50)
    logger.info("Starting trading cycle [mode: single-agent]")

    logger.info("Fetching market data...")
    report = data.get_full_report(address=executor.address)

    market = report.get("market")
    if market:
        logger.info(
            "BTC mid=%.1f | spread=%.1f(%.4f%%) | funding=%.4f%% | 24h=%.2f%%",
            market.mid_price,
            market.spread,
            market.spread_pct,
            market.funding_rate * 100,
            market.day_change_pct,
        )

    # Inject performance context for LLM self-reflection
    perf_ctx = trade_logger.get_llm_context()
    if perf_ctx:
        report["performance_context"] = perf_ctx

    # Inject open orders so LLM knows existing SL/TP levels
    orders_summary = executor.tracker.to_orders_summary()
    if orders_summary:
        report["open_orders_summary"] = orders_summary

    logger.info("Requesting single-agent LLM analysis...")
    signal_result = llm.analyze(report)

    if signal_result is None:
        logger.warning("LLM returned no signal, skipping cycle")
        return {"status": "skipped", "reason": "LLM returned None"}

    return _validate_and_execute(signal_result, report, risk, executor, trade_logger)


# ─── Multi-Agent Debate Strategy ────────────────────────────────────────────


def run_once_debate(
    strategy: "DebateStrategy",
    data: DataProvider,
    risk: RiskManager,
    executor: TradeExecutor,
    trade_logger: TradeLogger,
) -> dict:
    """Multi-agent debate → risk check → execute."""
    logger.info("=" * 50)
    logger.info("Starting trading cycle [mode: multi-agent debate]")

    logger.info("Fetching market data...")
    report = data.get_full_report(address=executor.address)

    market = report.get("market")
    if market:
        logger.info(
            "BTC mid=%.1f | spread=%.1f(%.4f%%) | funding=%.4f%% | 24h=%.2f%%",
            market.mid_price,
            market.spread,
            market.spread_pct,
            market.funding_rate * 100,
            market.day_change_pct,
        )

    # Inject performance context
    perf_ctx = trade_logger.get_llm_context()
    if perf_ctx:
        report["performance_context"] = perf_ctx

    # Inject open orders so debaters know existing SL/TP levels
    orders_summary = executor.tracker.to_orders_summary()
    if orders_summary:
        report["open_orders_summary"] = orders_summary

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

    result = _validate_and_execute(signal_result, report, risk, executor, trade_logger)
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
) -> dict:
    """Risk validation + trade execution — shared by all strategies."""

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

    # Use the smart sync: handles resting→active, active→gone, recovery
    executor.sync_with_chain(current_side, current_size, chain_entry)

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
            _record_close_from_chain(
                trade_logger, report,
                entry_price=tracker_entry_before_sync,
                side=tracker_side_before_sync,
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

    # Set initial balance for drawdown tracking on first cycle
    if risk.initial_balance is None and account_balance and account_balance > 0:
        risk.initial_balance = account_balance
        logger.info("Initial balance set: $%.2f", account_balance)

    # Tick circuit breaker cooldown
    risk.tick_cooldown()

    # ── Phase 2: Risk validation ──
    risk_check = risk.validate(
        signal_result, current_size, current_side,
        mid_price=mid_price, account_balance=account_balance,
    )
    if not risk_check.passed:
        logger.warning("Risk check failed: %s", risk_check.reason)
        notify.send(f"🛡️ Risk rejected: {risk_check.reason}")
        return {
            "status": "rejected",
            "signal": signal_result.action,
            "confidence": signal_result.confidence,
            "reason": risk_check.reason,
        }

    # ── Phase 3: Record trade open intent ──
    trade_opened_this_cycle = False
    if (signal_result.action in ("LONG", "SHORT")
            and not executor.tracker.has_resting_order()
            and not executor.tracker.has_position()):
        side = "long" if signal_result.action == "LONG" else "short"
        size = signal_result.size or config.max_position_size
        entry_price = signal_result.entry_price or mid_price
        trade_logger.open_trade(
            side=side, size=size, entry_price=entry_price,
            dry_run=executor.dry_run,
        )
        trade_opened_this_cycle = True
        notify.send(
            f"📈 {signal_result.action} {size} BTC @ ${entry_price:.0f}\n"
            f"SL: ${signal_result.stop_loss:.0f} | TP: ${signal_result.take_profit:.0f}\n"
            f"Confidence: {signal_result.confidence:.2f}"
        )

    # ── Phase 4: Execute ──
    result = executor.execute(signal_result)
    logger.info("Execution result: %s | Position: %s",
                result, executor.tracker.to_summary())

    # If execution failed after opening a pending trade this cycle, clean it up
    if (signal_result.action in ("LONG", "SHORT")
            and trade_opened_this_cycle
            and not result.get("executed")):
        trade_logger.cancel_pending()
        logger.warning("Execution failed — cancelled pending trade record")

    # ── Phase 5: Post-execution recording ──
    if signal_result.action == "CLOSE" and result.get("executed"):
        exit_price = _get_exit_price(report)
        trade = trade_logger.close_trade(exit_price, reason="signal")
        if trade:
            if trade.is_win:
                risk.record_win(trade.net_pnl)
            else:
                risk.record_loss(trade.net_pnl)
            emoji = "🟢" if trade.is_win else "🔴"
            notify.send(
                f"{emoji} Position closed: {trade.side.upper()}\n"
                f"P&L: ${trade.net_pnl:+.2f} ({trade.pnl_pct:+.2f}%)\n"
                f"Reason: {trade.close_reason}"
            )

    # Entry price is already set in open_trade() from signal.entry_price
    # or approximated from mid_price. The next cycle's sync_with_chain()
    # will update it from actual chain fill data.

    # Derive a human-readable status
    if signal_result.action == "HOLD":
        cycle_status = "hold"
    elif result.get("executed"):
        cycle_status = "executed"
    else:
        cycle_status = "failed"

    return {
        "status": cycle_status,
        "signal": signal_result.action,
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
) -> None:
    """Record a trade close detected from on-chain state (SL/TP filled).

    Args:
        trade_logger: The trade logger instance.
        report: Market data report (for mid price as exit price).
        entry_price: The entry price BEFORE the tracker was cleared.
        side: The position side BEFORE the tracker was cleared.
    """
    market = report.get("market")
    exit_price = market.mid_price if market else 0.0

    # Determine reason from the captured pre-sync values
    if entry_price > 0 and exit_price > 0 and side != "none":
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
    logger.info("Kimi Quant — BTC Perpetual Contract Trading")
    logger.info("Model: %s | Mode: %s | Interval: %ds | Dry Run: %s",
                config.display_model, mode,
                config.trading_interval_seconds, config.dry_run)
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
        recent = trade_logger.get_all_trades()[-10:]
        consecutive = 0
        for t in reversed(recent):
            if not t.is_win:
                consecutive += 1
            else:
                break
        risk.consecutive_losses = consecutive
        risk.total_realized_pnl = stats.net_pnl
        if consecutive > 0:
            logger.info("Seeded circuit breaker: %d consecutive losses from history",
                        consecutive)

    # Debate mode: create strategy ONCE (checkpointer lazy-inits on first use)
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

    MIN_INTERVAL = config.min_interval      # default 300s (5 min) — cost control
    MAX_INTERVAL = config.max_interval      # default 10800s (3h) — don't drift too far

    while not _shutdown_requested:
        cycle_count += 1
        start_time = time.monotonic()

        # ═══ Layer 1: Per-cycle protection ═══
        try:
            if mode == "debate":
                assert strategy is not None
                result = run_once_debate(strategy, data, risk, executor, trade_logger)
            else:
                assert llm is not None
                result = run_once_single(llm, data, risk, executor, trade_logger)

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
        print(
            f"{t.opened_at[:19]} {tag} | {t.side.upper():5s} | "
            f"in: ${t.entry_price:>8.1f} → out: ${t.exit_price:>8.1f} | "
            f"P&L: ${t.pnl:+7.2f} ({t.pnl_pct:+.2f}%) | "
            f"{t.close_reason}"
        )


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
    args = parser.parse_args()

    if args.history:
        cmd_history()
        return

    if args.stats:
        cmd_stats()
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
