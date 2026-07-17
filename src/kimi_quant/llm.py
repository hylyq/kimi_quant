"""Kimi K3 LLM integration via LangChain.

Uses OpenAI-compatible ChatOpenAI with the Moonshot API endpoint.
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from kimi_quant.config import config

logger = logging.getLogger(__name__)


class TradingSignal(BaseModel):
    """Structured trading signal from the LLM analysis.

    Supported actions:
      LONG  — open a long position (with SL + TP)
      SHORT — open a short position (with SL + TP)
      CLOSE — close the current position
      HOLD  — no action
      MODIFY_SL — move existing stop loss to a new price (trailing/breakeven)
    """

    action: str = Field(
        description="Trading action: LONG, SHORT, CLOSE, HOLD, or MODIFY_SL"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level of the signal, 0.0 to 1.0",
    )
    reasoning: str = Field(
        description="Brief reasoning behind the trading decision"
    )
    size: float | None = Field(
        default=None,
        description="Suggested position size in BTC (only for LONG/SHORT)",
    )
    entry_price: float | None = Field(
        default=None,
        description="Suggested entry price, None for market order",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Stop loss price (for LONG/SHORT) or new SL price (for MODIFY_SL)",
    )
    take_profit: float | None = Field(
        default=None,
        description="Suggested take profit price",
    )
    modify_sl_to: float | None = Field(
        default=None,
        description="New stop loss price when action=MODIFY_SL (e.g., move to breakeven)",
    )
    key_factors: list[str] = Field(
        default_factory=list,
        description="Key factors that influenced this decision",
    )


def build_market_prompt(market_data: dict[str, Any]) -> str:
    """Build a structured prompt from market data (shared by single & debate modes)."""
    from kimi_quant.config import config as _config

    market = market_data.get("market")
    order_book = market_data.get("order_book")
    account = market_data.get("account")

    prompt_parts = ["# Market Data Snapshot\n"]

    if market:
        prompt_parts.append(market.to_summary())
    else:
        prompt_parts.append("Market data unavailable.")

    prompt_parts.append("\n# Order Book Depth\n")
    if order_book:
        prompt_parts.append(order_book.to_summary(levels=5))
    else:
        prompt_parts.append("Order book data unavailable.")

    prompt_parts.append("\n# Account Status\n")
    if account:
        prompt_parts.append(account.to_summary())
    else:
        prompt_parts.append("Dry-run mode — no real position.")

    prompt_parts.append(
        f"\n# Instructions\n"
        f"Max position size: {_config.max_position_size} BTC.\n"
        f"Analyze the data above and produce a trading signal.\n"
    )

    # Append performance context if available (LLM self-reflection)
    perf_ctx = market_data.get("performance_context", "")
    if perf_ctx:
        prompt_parts.append(perf_ctx)

    return "\n".join(prompt_parts)


class KimiLLM:
    """LangChain wrapper for Kimi K3 via Moonshot API."""

    # System prompt that defines the trading analyst persona
    SYSTEM_PROMPT = """\
You are a BTC perpetual quant analyst on Hyperliquid. Analyze market data and \
output a TradingSignal.

Key principles:
1. Higher TF trend = anchor (4h > 1h > 15m > 5m). Don't fight it.
2. Order book: bid walls = support, ask walls = resistance. Thin books = noise.
3. Funding: very positive → crowded longs (reversal risk); negative → shorts paying (squeeze risk).
4. Multi-TF confluence → higher confidence. Divergence → follow higher TF, reduce size.
5. When uncertain, HOLD. Confidence < 0.7 → skip trade.

Output JSON only (no markdown):
- action: LONG|SHORT|CLOSE|HOLD|MODIFY_SL
- confidence: 0.0-1.0
- reasoning: brief synthesis
- size: BTC amount (null for CLOSE/HOLD)
- entry_price: limit price or null for market order
- stop_loss: mandatory for directional (min 0.5% from entry)
- take_profit: realistic target
- modify_sl_to: new SL price (MODIFY_SL only)
- key_factors: 2-4 items
"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=config.moonshot_api_key,
            base_url=config.moonshot_base_url,
            model=config.kimi_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )

        # Use LangChain's structured output for reliable JSON parsing
        self.structured_llm = self.llm.with_structured_output(
            TradingSignal, method="json_schema"
        )

        logger.info(
            "KimiLLM initialized (model=%s, base_url=%s)",
            config.kimi_model,
            config.moonshot_base_url,
        )

    @staticmethod
    def build_prompt(market_data: dict[str, Any]) -> str:
        """Build a structured prompt from market data.

        Uses the DataProvider's MarketAnalysis.to_llm_prompt() when available
        (includes multi-timeframe analysis), falls back to basic prompt.
        """
        from kimi_quant.data import DataProvider
        return DataProvider.build_llm_prompt(market_data)

    def analyze(self, market_data: dict[str, Any]) -> TradingSignal | None:
        """Analyze market data and return a trading signal.

        Returns None if analysis fails or the LLM is unavailable.
        """
        try:
            prompt = self.build_prompt(market_data)
            messages = [
                ("system", self.SYSTEM_PROMPT),
                ("user", prompt),
            ]

            logger.info("Requesting trading analysis from Kimi K3...")
            signal: TradingSignal = self.structured_llm.invoke(messages)

            logger.info(
                "Signal received: action=%s confidence=%.2f reasoning=%s",
                signal.action,
                signal.confidence,
                signal.reasoning[:80],
            )

            return signal

        except Exception as e:
            logger.error("LLM analysis failed: %s", e, exc_info=True)
            return None

    def analyze_raw(self, market_data: dict[str, Any]) -> str:
        """Fallback: get raw text response (when structured output fails)."""
        try:
            prompt = self.build_prompt(market_data)
            messages = [
                ("system", self.SYSTEM_PROMPT),
                ("user", prompt),
            ]

            response = self.llm.invoke(messages)
            return str(response.content)

        except Exception as e:
            logger.error("Raw LLM analysis failed: %s", e)
            return ""
