"""Risk management — validates trading signals before execution."""

import logging
from dataclasses import dataclass

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk validation check."""

    passed: bool
    reason: str


class RiskManager:
    """Validates trading signals against risk parameters."""

    def __init__(self):
        self.max_position = config.max_position_size
        self.min_confidence = config.min_confidence
        self.max_leverage = config.max_leverage

    def validate(
        self,
        signal: TradingSignal,
        current_position_size: float = 0.0,
        current_position_side: str = "none",
    ) -> RiskCheck:
        """Run all risk checks on a trading signal.

        Returns a RiskCheck with passed=True only if all checks pass.
        """
        checks = [
            self._check_confidence(signal),
            self._check_size(signal, current_position_size, current_position_side),
            self._check_direction(signal, current_position_side),
            self._check_stop_loss(signal),
        ]

        for check in checks:
            if not check.passed:
                logger.warning("Risk check failed: %s", check.reason)
                return check

        logger.info("All risk checks passed")
        return RiskCheck(passed=True, reason="All checks passed")

    def _check_confidence(self, signal: TradingSignal) -> RiskCheck:
        if signal.confidence < self.min_confidence:
            return RiskCheck(
                passed=False,
                reason=f"Confidence {signal.confidence:.2f} below "
                f"minimum {self.min_confidence}",
            )
        return RiskCheck(passed=True, reason="Confidence OK")

    def _check_size(
        self,
        signal: TradingSignal,
        current_size: float,
        current_side: str,
    ) -> RiskCheck:
        # HOLD, CLOSE, and MODIFY_SL don't need size checks
        if signal.action in ("HOLD", "CLOSE", "MODIFY_SL"):
            return RiskCheck(passed=True, reason="No size needed")

        if signal.size is None or signal.size <= 0:
            return RiskCheck(
                passed=False,
                reason=f"Invalid position size: {signal.size}",
            )

        if signal.size > self.max_position:
            return RiskCheck(
                passed=False,
                reason=f"Size {signal.size} exceeds max {self.max_position}",
            )

        return RiskCheck(passed=True, reason="Size OK")

    def _check_direction(
        self,
        signal: TradingSignal,
        current_side: str,
    ) -> RiskCheck:
        # Prevent redundant trades
        if signal.action == "LONG" and current_side == "long":
            return RiskCheck(
                passed=False,
                reason="Already holding a long position",
            )
        if signal.action == "SHORT" and current_side == "short":
            return RiskCheck(
                passed=False,
                reason="Already holding a short position",
            )
        if signal.action == "CLOSE" and current_side == "none":
            return RiskCheck(
                passed=False,
                reason="No position to close",
            )
        # MODIFY_SL requires an existing position
        if signal.action == "MODIFY_SL" and current_side == "none":
            return RiskCheck(
                passed=False,
                reason="No position to modify stop loss for",
            )

        return RiskCheck(passed=True, reason="Direction OK")

    def _check_stop_loss(self, signal: TradingSignal) -> RiskCheck:
        if signal.action in ("LONG", "SHORT"):
            if signal.stop_loss is None:
                return RiskCheck(
                    passed=False,
                    reason="Stop loss is required for directional trades",
                )
        if signal.action == "MODIFY_SL":
            new_sl = signal.modify_sl_to or signal.stop_loss
            if new_sl is None:
                return RiskCheck(
                    passed=False,
                    reason="New stop loss price is required for MODIFY_SL",
                )
        return RiskCheck(passed=True, reason="Stop loss OK")
