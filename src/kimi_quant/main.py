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

    logger.info("Requesting single-agent LLM analysis...")
    signal_result = llm.analyze(report)

    if signal_result is None:
        logger.warning("LLM returned no signal, skipping cycle")
        return {"status": "skipped", "reason": "LLM returned None"}

    return _validate_and_execute(signal_result, report, risk, executor)


# ─── Multi-Agent Debate Strategy ────────────────────────────────────────────


def run_once_debate(
    strategy: "DebateStrategy",
    data: DataProvider,
    risk: RiskManager,
    executor: TradeExecutor,
) -> dict:
    """Multi-agent debate → risk check → execute.

    The strategy object is persistent across cycles — its checkpointer
    automatically saves each cycle's full state (market data, all three
    arguments, judge verdict) to the database.
    """
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

    result = _validate_and_execute(signal_result, report, risk, executor)
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
) -> dict:
    """Risk validation + trade execution — shared by all strategies."""

    # Sync tracker with on-chain state:
    # If on-chain shows no position but tracker thinks we have one,
    # the position was closed (SL/TP filled or manual close) → clear tracker.
    account = report.get("account")
    current_size = 0.0
    current_side = "none"
    if account:
        current_size = account.position_size
        current_side = account.position_side
        if current_side == "none" and executor.tracker.has_position():
            logger.warning(
                "Tracker has position but on-chain shows none — "
                "SL/TP likely filled. Clearing tracker."
            )
            executor.tracker.clear()

    logger.info("Position: %s", executor.tracker.to_summary())

    risk_check = risk.validate(signal_result, current_size, current_side)
    if not risk_check.passed:
        logger.warning("Risk check failed, skipping execution")
        return {
            "status": "rejected",
            "signal": signal_result.action,
            "confidence": signal_result.confidence,
            "reason": risk_check.reason,
        }

    result = executor.execute(signal_result)
    logger.info("Execution result: %s | Position: %s",
                result, executor.tracker.to_summary())

    return {
        "status": "executed" if result.get("executed") else "failed",
        "signal": signal_result.action,
        "confidence": signal_result.confidence,
        "reasoning": signal_result.reasoning,
        "execution": result,
    }


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
                result = run_once_debate(strategy, data, risk, executor)
            else:
                assert llm is not None
                result = run_once_single(llm, data, risk, executor)

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
    if strategy is not None:
        strategy.close()
        logger.info("Checkpointer closed. States persisted to debate.db")
    logger.info("Session complete.")


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
    args = parser.parse_args()

    if args.history:
        cmd_history()
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

        if config.strategy_mode == "debate":
            from kimi_quant.debate import DebateStrategy, create_checkpointer
            handle = create_checkpointer()
            strategy = DebateStrategy(checkpointer_handle=handle)
            try:
                result = run_once_debate(strategy, data, risk, executor)
            finally:
                strategy.close()
        else:
            llm = KimiLLM()
            result = run_once_single(llm, data, risk, executor)

        print(_json.dumps(result, indent=2, default=str))
    else:
        run_loop()


if __name__ == "__main__":
    main()
