"""LLM integration with automatic fallback.

Primary: Kimi K3 via Moonshot API.
Fallback: DeepSeek V3 via DeepSeek API (auto-activated when Kimi fails).

Both use OpenAI-compatible ChatOpenAI.
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from kimi_quant.config import config

logger = logging.getLogger(__name__)


# ─── LLM Factory with Fallback ────────────────────────────────────────────


def _build_model_registry(
    temp: float, tokens: int, include_thinking: bool = True
) -> dict[str, ChatOpenAI]:
    """Build available LLM instances keyed by provider name.

    Only includes models whose API key is configured.
    Handles provider-specific reasoning/thinking parameters.

    Args:
        temp: LLM temperature.
        tokens: Max output tokens.
        include_thinking: If False, skip thinking/reasoning params.
            Structured output (json_schema) is incompatible with thinking
            on DeepSeek, so the Judge disables it.
    """
    registry: dict[str, ChatOpenAI] = {}
    effort = config.reasoning_effort.lower()

    # Kimi / Moonshot
    if config.moonshot_api_key:
        kimi_kwargs: dict[str, Any] = dict(
            api_key=config.moonshot_api_key,
            base_url=config.moonshot_base_url,
            model=config.kimi_model,
            temperature=temp,
            max_tokens=tokens,
        )
        # Kimi K3: reasoning_effort is a direct API param (only "max" supported)
        if include_thinking and effort == "max":
            kimi_kwargs["reasoning_effort"] = "max"
        registry["kimi"] = ChatOpenAI(**kimi_kwargs)

    # DeepSeek
    if config.deepseek_api_key:
        ds_kwargs: dict[str, Any] = dict(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model,
            temperature=temp,
            max_tokens=tokens,
        )
        # DeepSeek: thinking control via extra_body (OpenAI SDK passthrough).
        # Skip when structured output is active — json_schema + thinking
        # is incompatible on DeepSeek (returns 400).
        if include_thinking:
            if effort == "off":
                ds_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                ds_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        registry["deepseek"] = ChatOpenAI(**ds_kwargs)

    return registry


def _resolve_chain(
    registry: dict[str, Any],
    primary_name: str,
) -> Any:
    """Build a primary→fallback chain from the model registry.

    Args:
        registry: Available runnables keyed by provider name.
        primary_name: Which model to try first.

    Returns:
        A Runnable (possibly with_fallbacks) ready to use.
    """
    if primary_name not in registry:
        available = list(registry.keys())
        logger.warning(
            "Primary '%s' not available (missing API key?), using '%s'",
            primary_name, available[0],
        )
        primary_name = available[0]

    primary = registry[primary_name]
    fallbacks = [llm for name, llm in registry.items() if name != primary_name]

    if not fallbacks:
        logger.info("LLM: %s only (no fallback configured)", primary_name)
        return primary

    fb_names = ", ".join(n for n in registry if n != primary_name)
    logger.info("LLM: %s primary → fallback: %s", primary_name, fb_names)
    return primary.with_fallbacks(fallbacks)


def create_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI with automatic fallback chain.

    Primary model is determined by PRIMARY_LLM env var (default: "kimi").
    All other available models become automatic fallbacks.

    Args:
        temperature: Override default temperature.
        max_tokens: Override default max_tokens.

    Returns:
        A ChatOpenAI wrapped with fallback chain.
    """
    temp = temperature if temperature is not None else config.llm_temperature
    tokens = max_tokens if max_tokens is not None else config.llm_max_tokens
    registry = _build_model_registry(temp, tokens)
    return _resolve_chain(registry, config.primary_llm)


def create_structured_llm(
    schema: type[BaseModel],
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Create a structured-output LLM with automatic fallback chain.

    Primary model is determined by PRIMARY_LLM env var (default: "kimi").
    Each model gets with_structured_output(schema) applied, then chained.

    Args:
        schema: Pydantic model class for structured output.
        temperature: Override default temperature.
        max_tokens: Override default max_tokens.

    Returns:
        A Runnable[dict, schema] with fallback chain.
    """
    temp = temperature if temperature is not None else config.llm_temperature
    tokens = max_tokens if max_tokens is not None else config.llm_max_tokens
    # Structured output (json_schema) is incompatible with thinking on DeepSeek
    registry = _build_model_registry(temp, tokens, include_thinking=False)

    # Apply structured output to each model
    structured_registry = {
        name: llm.with_structured_output(schema, method="json_schema")
        for name, llm in registry.items()
    }
    return _resolve_chain(structured_registry, config.primary_llm)


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
    next_interval: int | None = Field(
        default=None,
        description=(
            "Suggested seconds until next analysis cycle (range 300-10800). "
            "Use 300-600 for active positions or imminent breakout confirmations. "
            "Use 1800-3600 for typical market conditions. "
            "Use 7200-10800 for quiet/sideways/weekend markets. "
            "Default recommendation: leave null unless you have a strong reason."
        ),
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
    """Single-agent LLM strategy with automatic Kimi→DeepSeek fallback."""

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
- next_interval: suggested seconds until next cycle (null=use default).
  Range 300-10800 (5min-3h). ONLY use 300-600 for active positions or confirmed
  breakout setups. Default to null or 1800+ for normal HOLD conditions.
  Longer (3600-10800) when market is quiet/sideways. When in doubt, go longer.
"""

    def __init__(self):
        self.llm = create_llm()
        self.structured_llm = create_structured_llm(TradingSignal)

        # Show the actual primary model, not hardcoded kimi_model
        primary = config.primary_llm.lower()
        if primary == "deepseek" and config.deepseek_api_key:
            display = config.deepseek_model
        else:
            display = config.kimi_model
        logger.info("KimiLLM initialized (primary=%s)", display)

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

            logger.info("Requesting trading analysis...")
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
