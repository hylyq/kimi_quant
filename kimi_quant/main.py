"""Main entry point for Kimi Quant.

Orchestrates the trading loop: fetch data → analyze with LLM → validate risk →
execute trades → log results.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

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


def run_once(
    llm: KimiLLM,
    data: DataProvider,
    risk: RiskManager,
    executor: TradeExecutor,
) -> dict:
    """Run a single analysis-execution cycle.

    Returns a dict summarizing what happened.
    """
    logger.info("=" * 50)
    logger.info("Starting trading cycle")

    # 1. Fetch market data
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

    # 2. Get LLM analysis
    logger.info("Requesting LLM analysis...")
    signal_result = llm.analyze(report)

    if signal_result is None:
        logger.warning("LLM returned no signal, skipping cycle")
        return {"status": "skipped", "reason": "LLM returned None"}

    # 3. Risk validation
    current_size = 0.0
    current_side = "none"
    account = report.get("account")
    if account:
        current_size = account.position_size
        current_side = account.position_side

    risk_check = risk.validate(signal_result, current_size, current_side)
    if not risk_check.passed:
        logger.warning("Risk check failed, skipping execution")
        return {
            "status": "rejected",
            "signal": signal_result.action,
            "confidence": signal_result.confidence,
            "reason": risk_check.reason,
        }

    # 4. Execute trade
    result = executor.execute(signal_result)
    logger.info("Execution result: %s", result)

    return {
        "status": "executed" if result.get("executed") else "failed",
        "signal": signal_result.action,
        "confidence": signal_result.confidence,
        "reasoning": signal_result.reasoning,
        "execution": result,
    }


def run_loop():
    """Main trading loop — runs at configured intervals."""
    logger.info("=" * 50)
    logger.info("Kimi Quant — BTC Perpetual Contract Trading")
    logger.info("Model: %s | Interval: %ds | Dry Run: %s",
                config.kimi_model,
                config.trading_interval_seconds,
                config.dry_run)
    logger.info("=" * 50)

    # Validate configuration
    config.validate()

    # Initialize components
    llm = KimiLLM()
    data = DataProvider()
    risk = RiskManager()
    executor = TradeExecutor()

    cycle_count = 0
    signals_history: list[dict] = []

    while not _shutdown_requested:
        cycle_count += 1
        start_time = time.monotonic()

        try:
            result = run_once(llm, data, risk, executor)
            result["cycle"] = cycle_count
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            signals_history.append(result)

            # Log summary
            status = result.get("status", "unknown")
            sig = result.get("signal", "N/A")
            conf = result.get("confidence", 0)
            logger.info(
                "Cycle %d complete: status=%s signal=%s confidence=%.2f",
                cycle_count, status, sig, conf,
            )

        except Exception as e:
            logger.error("Cycle %d failed with error: %s", cycle_count, e,
                         exc_info=True)

        # Keep only last 100 signals in memory
        if len(signals_history) > 100:
            signals_history = signals_history[-100:]

        # Sleep until next cycle (account for execution time)
        elapsed = time.monotonic() - start_time
        sleep_time = max(0, config.trading_interval_seconds - elapsed)
        if not _shutdown_requested and sleep_time > 0:
            logger.info(
                "Sleeping %.1fs until next cycle...", sleep_time
            )
            # Sleep in short intervals to respond to shutdown promptly
            while sleep_time > 0 and not _shutdown_requested:
                time.sleep(min(1, sleep_time))
                sleep_time -= 1

    logger.info("Shutting down. Total cycles: %d", cycle_count)
    logger.info("Session complete.")


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
    args = parser.parse_args()

    if args.interval:
        config.trading_interval_seconds = args.interval

    if args.once:
        # Single-shot mode
        config.validate()
        llm = KimiLLM()
        data = DataProvider()
        risk = RiskManager()
        executor = TradeExecutor()
        result = run_once(llm, data, risk, executor)

        # Pretty-print result
        import json as _json
        print(_json.dumps(result, indent=2, default=str))
    else:
        run_loop()


if __name__ == "__main__":
    main()
