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
import signal
import time
from datetime import datetime, timezone
from typing import Any

from kimi_quant.analytics import TradeLogger
from kimi_quant.config import config
from kimi_quant.data import DataProvider
from kimi_quant.executor import TradeExecutor
from kimi_quant.llm import KimiLLM
from kimi_quant.risk import RiskManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
        account_balance = account.balance

    # Use the smart sync: handles resting→active, active→gone, recovery
    executor.sync_with_chain(current_side, current_size, chain_entry)

    # If sync detected a close event, record it
    was_active = executor.tracker.has_position() is False and (
        current_side != "none"  # was active before sync cleared it
    )
    if not executor.tracker.has_position() and not executor.tracker.has_resting_order():
        # Check if trade_logger has a pending trade that was just closed by SL/TP
        if trade_logger.has_pending and current_side == "none":
            _record_close_from_chain(executor, trade_logger, report)
            # Circuit breaker: record loss
            pending_trade = trade_logger._pending  # already moved to _closed
            # Actually, close_from_chain already calls close_trade, so check the
            # last closed trade from stats
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
        return {
            "status": "rejected",
            "signal": signal_result.action,
            "confidence": signal_result.confidence,
            "reason": risk_check.reason,
        }

    # ── Phase 3: Record trade open intent ──
    if signal_result.action in ("LONG", "SHORT") and not executor.dry_run:
        trade_logger.open_trade(
            side="long" if signal_result.action == "LONG" else "short",
            size=signal_result.size or config.max_position_size,
            entry_price=signal_result.entry_price or mid_price,
        )

    # ── Phase 4: Execute ──
    result = executor.execute(signal_result)
    logger.info("Execution result: %s | Position: %s",
                result, executor.tracker.to_summary())

    # ── Phase 5: Post-execution recording ──
    if signal_result.action == "CLOSE" and result.get("executed"):
        exit_price = _get_exit_price(report)
        trade = trade_logger.close_trade(exit_price, reason="signal")
        if trade:
            if trade.is_win:
                risk.record_win(trade.net_pnl)
            else:
                risk.record_loss(trade.net_pnl)

    # If position was opened with market order, update entry from chain
    if signal_result.action in ("LONG", "SHORT") and result.get("executed"):
        if account and account.entry_price > 0:
            pending = trade_logger._pending
            if pending and pending.entry_price == 0:
                pending.entry_price = account.entry_price

    return {
        "status": "executed" if result.get("executed") else "failed",
        "signal": signal_result.action,
        "confidence": signal_result.confidence,
        "reasoning": signal_result.reasoning,
        "execution": result,
    }


def _get_exit_price(report: dict) -> float:
    """Extract current mid price as exit price from the market report."""
    market = report.get("market")
    if market:
        return market.mid_price
    return 0.0


def _record_close_from_chain(
    executor: TradeExecutor, trade_logger: TradeLogger, report: dict
) -> None:
    """Record a trade close detected from on-chain state (SL/TP filled)."""
    market = report.get("market")
    exit_price = market.mid_price if market else 0.0

    # Determine reason from tracker state (before clear happened)
    entry = executor.tracker.entry_price
    side = executor.tracker.side
    if entry > 0 and exit_price > 0:
        if side == "long":
            reason = "take_profit" if exit_price > entry else "stop_loss"
        else:
            reason = "take_profit" if exit_price < entry else "stop_loss"
    else:
        reason = "manual"

    trade_logger.close_trade(exit_price, reason=reason)


# ─── Main Loop ───────────────────────────────────────────────────────────────


def run_loop():
    """Main trading loop — runs at configured intervals."""
    mode = config.strategy_mode
    logger.info("=" * 50)
    logger.info("Kimi Quant — BTC Perpetual Contract Trading")
    logger.info("Model: %s | Mode: %s | Interval: %ds | Dry Run: %s",
                config.kimi_model, mode,
                config.trading_interval_seconds, config.dry_run)
    logger.info("=" * 50)

    config.validate()

    data = DataProvider()
    risk = RiskManager()
    executor = TradeExecutor()
    trade_logger = TradeLogger()

    # Set initial balance for drawdown tracking (first report will have account)
    # We'll set it on the first cycle when account data is available

    # Log startup stats
    stats = trade_logger.get_stats()
    if stats.total_trades > 0:
        logger.info(
            "Loaded trade history: %d trades | Win Rate: %.1f%% | Net P&L: $%.2f",
            stats.total_trades, stats.win_rate, stats.net_pnl,
        )
        # Seed circuit breaker from history
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

    # Debate mode: create strategy ONCE with persistent checkpointer
    strategy = None
    if mode == "debate":
        from kimi_quant.debate import DebateStrategy, create_checkpointer
        handle = create_checkpointer()
        strategy = DebateStrategy(checkpointer_handle=handle)

        # Crash recovery: check if we have state from a prior run
        latest = strategy.get_latest_state()
        if latest:
            logger.info(
                "Recovered state from prior run (cycle_id=%s)",
                latest.get("cycle_id", "unknown"),
            )

    llm = KimiLLM() if mode == "single" else None

    cycle_count = 0
    signals_history: list[dict] = []

    while not _shutdown_requested:
        cycle_count += 1
        start_time = time.monotonic()

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
            signals_history.append(result)

            status = result.get("status", "unknown")
            sig = result.get("signal", "N/A")
            conf = result.get("confidence", 0)
            logger.info(
                "Cycle %d complete: mode=%s status=%s signal=%s confidence=%.2f",
                cycle_count, mode, status, sig, conf,
            )

        except Exception as e:
            logger.error("Cycle %d failed with error: %s", cycle_count, e,
                         exc_info=True)

        if len(signals_history) > 100:
            signals_history = signals_history[-100:]

        elapsed = time.monotonic() - start_time
        sleep_time = max(0, config.trading_interval_seconds - elapsed)
        if not _shutdown_requested and sleep_time > 0:
            logger.info("Sleeping %.1fs until next cycle...", sleep_time)
            while sleep_time > 0 and not _shutdown_requested:
                time.sleep(min(1, sleep_time))
                sleep_time -= 1

    logger.info("Shutting down. Total cycles: %d", cycle_count)

    # Final stats
    stats = trade_logger.get_stats()
    if stats.total_trades > 0:
        logger.info(
            "Session P&L: %d trades | Win Rate: %.1f%% | Net P&L: $%.2f",
            stats.total_trades, stats.win_rate, stats.net_pnl,
        )

    if strategy is not None:
        strategy.close()
        logger.info("Checkpointer closed. States persisted to debate.db")
    logger.info("Session complete.")


def cmd_stats():
    """Print trade P&L statistics."""
    trade_logger = TradeLogger()
    stats = trade_logger.get_stats()

    if stats.total_trades == 0:
        print("No trade history found.")
        return

    print(f"=== Trading Performance ===")
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

    # Recent trades
    trades = trade_logger.get_all_trades()
    print("=== Recent Trades ===")
    for t in trades[-10:]:
        print(
            f"{t.opened_at[:19]} | {t.side.upper():5s} | "
            f"in: ${t.entry_price:>8.1f} → out: ${t.exit_price:>8.1f} | "
            f"P&L: ${t.pnl:+7.2f} ({t.pnl_pct:+.2f}%) | "
            f"{t.close_reason}"
        )


def cmd_history():
    """Print the full debate history from the checkpoint database."""
    from kimi_quant.debate import DebateStrategy, create_checkpointer

    handle = create_checkpointer()
    strategy = DebateStrategy(checkpointer_handle=handle)
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
            from kimi_quant.debate import DebateStrategy, create_checkpointer
            handle = create_checkpointer()
            strategy = DebateStrategy(checkpointer_handle=handle)
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
