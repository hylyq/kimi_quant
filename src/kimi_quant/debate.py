"""Multi-Agent Debate Strategy via LangGraph.

Three specialized agents (Bull, Bear, Hold) independently analyze market data
and present their arguments. A Judge agent weighs all arguments and makes
the final trading decision.

Architecture (LangGraph StateGraph):

    START
      │
      ▼
  ┌─────────┐    asyncio.gather    ┌──────────────┐
  │ Prepare │───┬──────────────────▶│ Bull Agent   │──┐
  └─────────┘   │                  │ (论证做多)    │  │
                │                  └──────────────┘  │
                │                  ┌──────────────┐  │
                ├──────────────────▶│ Bear Agent   │──┤
                │                  │ (论证做空)    │  │
                │                  └──────────────┘  │
                │                  ┌──────────────┐  │
                └──────────────────▶│ Hold Agent   │──┘
                                   │ (论证观望)    │
                                   └──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Judge Agent  │
                                   │ (裁决)        │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   TradingSignal
                                          │
                                          ▼
                                         END
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)


# ─── Debate State ────────────────────────────────────────────────────────────


class DebateState(TypedDict):
    """State carried through the debate graph."""

    market_prompt: str
    account_summary: str
    bull_argument: str
    bear_argument: str
    hold_argument: str
    final_signal_json: str
    error: str


# ─── Agent Persona Prompts ───────────────────────────────────────────────────

BULL_SYSTEM_PROMPT = """\
You are an **aggressive bullish trader**. Your sole mission is to find and \
articulate the strongest possible case for going LONG on BTC right now.

Analyze the provided market data and build a compelling bull case:

1. **Bullish Order Book Signals**: Large bid walls? Ask walls being eaten? \
Buying pressure evident in the depth?
2. **Funding Rate Dynamics**: Negative or low funding? Shorts paying longs? \
This encourages going long.
3. **Price Action**: Is BTC at support? Oversold bounce territory? Bullish \
divergence forming?
4. **Market Structure**: Premium/discount analysis from a bull perspective. \
Any signs of accumulation?

**Rules:**
- Be SPECIFIC: reference actual price levels, sizes, funding rates from the data
- Be HONEST: don't fabricate evidence — if the bull case is weak, say so
- Be CONVINCING: present the best possible bull argument even if imperfect
- Keep it concise: 150-250 words

Output format: just your argument text, no JSON, no markdown headers."""

BEAR_SYSTEM_PROMPT = """\
You are a **skeptical bearish trader**. Your sole mission is to find and \
articulate the strongest possible case for going SHORT on BTC right now.

Analyze the provided market data and build a compelling bear case:

1. **Bearish Order Book Signals**: Large ask walls? Bid walls thinning? \
Selling pressure evident in the depth?
2. **Funding Rate Dynamics**: High positive funding? Longs paying shorts? \
This suggests overcrowded longs and potential reversal.
3. **Price Action**: Is BTC at resistance? Overbought? Bearish divergence?
4. **Market Structure**: Premium/discount from a bear perspective. \
Distribution signals?

**Rules:**
- Be SPECIFIC: reference actual price levels, sizes, funding rates from the data
- Be HONEST: don't fabricate evidence — if the bear case is weak, say so
- Be CONVINCING: present the best possible bear argument even if imperfect
- Keep it concise: 150-250 words

Output format: just your argument text, no JSON, no markdown headers."""

HOLD_SYSTEM_PROMPT = """\
You are a **cautious risk manager**. Your job is to find reasons to STAY OUT \
of the market right now.

Analyze the provided market data and build a case for HOLDING/WAITING:

1. **Conflicting Signals**: Do bulls and bears both have valid points? \
Is the direction unclear?
2. **Volatility & Spread**: Wide spreads? Choppy price action? \
Unfavorable risk/reward?
3. **Timing Concerns**: Are we between key levels? Is funding neutral? \
No clear edge?
4. **Risk/Reward Assessment**: Even if there's a slight directional bias, \
is the R:R ratio actually favorable right now?

**Rules:**
- Be SPECIFIC about what would need to change to justify entry
- Be HONEST: if the market actually has a clear direction, acknowledge it
- Be PRUDENT: when in doubt, waiting is the correct call
- Keep it concise: 150-250 words

Output format: just your argument text, no JSON, no markdown headers."""

JUDGE_SYSTEM_PROMPT = """\
You are the **Head Trader** with final decision authority. Your team of three \
analysts has just presented their arguments:

- **Bull Analyst**: argued for going LONG
- **Bear Analyst**: argued for going SHORT
- **Risk Manager**: argued for HOLDING/Waiting

Your job is to:

1. **Weigh the Evidence**: Which analyst presented the most data-backed, \
convincing case? Look for specific data references, not rhetoric.
2. **Identify the Weakest**: Which arguments are based on thin evidence \
or wishful thinking?
3. **Synthesize**: Combine insights from all three. Sometimes the best trade \
incorporates elements from multiple perspectives.
4. **Decide**: Make a final call — LONG, SHORT, CLOSE, or HOLD.

**Decision Guidelines:**
- If one analyst clearly has the strongest data-backed case → follow them
- If arguments are evenly matched → HOLD (uncertainty is a signal)
- If both Bull and Bear make weak cases → HOLD
- Be decisive when the evidence is clear, conservative when it's not
- Confidence 0.7+ only when the evidence is compelling

Output a JSON trading signal matching the TradingSignal schema:
- action: "LONG" | "SHORT" | "CLOSE" | "HOLD"
- confidence: 0.0-1.0
- reasoning: your synthesis of the debate
- size: position size in BTC (null for CLOSE/HOLD)
- entry_price: suggested entry (null = market order)
- stop_loss: mandatory for directional trades
- take_profit: realistic target
- key_factors: 2-4 factors that drove the decision
"""


# ─── Single-Turn Agent (non-streaming) ──────────────────────────────────────


class SingleTurnAgent:
    """A simple LLM agent that responds with a single message (no tool calling).

    Used for the debaters and judge — each gets one turn to produce their output.
    """

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            api_key=config.moonshot_api_key,
            base_url=config.moonshot_base_url,
            model=config.kimi_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )

    async def arun(self, user_prompt: str) -> str:
        """Run the agent asynchronously and return its text response."""
        try:
            messages = [
                ("system", self.system_prompt),
                ("user", user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Agent %s failed: %s", self.name, e)
            return f"[{self.name} failed to respond: {e}]"

    def run(self, user_prompt: str) -> str:
        """Synchronous wrapper for the agent."""
        try:
            messages = [
                ("system", self.system_prompt),
                ("user", user_prompt),
            ]
            response = self.llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Agent %s failed: %s", self.name, e)
            return f"[{self.name} failed to respond: {e}]"


# ─── Judge Agent with Structured Output ─────────────────────────────────────


class JudgeAgent:
    """The judge agent uses LangChain structured output to produce a TradingSignal."""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=config.moonshot_api_key,
            base_url=config.moonshot_base_url,
            model=config.kimi_model,
            temperature=0.05,  # lower temperature for the judge
            max_tokens=config.llm_max_tokens,
        )
        self.structured_llm = self.llm.with_structured_output(
            TradingSignal, method="json_schema"
        )

    async def ajudge(
        self, market_prompt: str, bull: str, bear: str, hold: str
    ) -> TradingSignal | None:
        """Asynchronously judge the debate and produce a TradingSignal."""
        try:
            debate_transcript = (
                "# Market Data\n"
                f"{market_prompt}\n\n"
                "# === DEBATE TRANSCRIPT ===\n\n"
                "## 🐂 BULL ANALYST (LONG Case)\n"
                f"{bull}\n\n"
                "## 🐻 BEAR ANALYST (SHORT Case)\n"
                f"{bear}\n\n"
                "## 😐 RISK MANAGER (HOLD Case)\n"
                f"{hold}\n\n"
                "# === YOUR DECISION ===\n"
                "Weigh the arguments above and produce the final trading signal."
            )
            messages = [
                ("system", JUDGE_SYSTEM_PROMPT),
                ("user", debate_transcript),
            ]

            logger.info("Judge deliberating on %d chars of debate...",
                        len(debate_transcript))
            signal: TradingSignal = await self.structured_llm.ainvoke(messages)
            return signal

        except Exception as e:
            logger.error("Judge failed: %s", e, exc_info=True)
            return None

    def judge(
        self, market_prompt: str, bull: str, bear: str, hold: str
    ) -> TradingSignal | None:
        """Synchronous judge."""
        try:
            debate_transcript = (
                "# Market Data\n"
                f"{market_prompt}\n\n"
                "# === DEBATE TRANSCRIPT ===\n\n"
                "## 🐂 BULL ANALYST (LONG Case)\n"
                f"{bull}\n\n"
                "## 🐻 BEAR ANALYST (SHORT Case)\n"
                f"{bear}\n\n"
                "## 😐 RISK MANAGER (HOLD Case)\n"
                f"{hold}\n\n"
                "# === YOUR DECISION ===\n"
                "Weigh the arguments above and produce the final trading signal."
            )
            messages = [
                ("system", JUDGE_SYSTEM_PROMPT),
                ("user", debate_transcript),
            ]

            logger.info("Judge deliberating on %d chars of debate...",
                        len(debate_transcript))
            signal: TradingSignal = self.structured_llm.invoke(messages)
            return signal

        except Exception as e:
            logger.error("Judge failed: %s", e, exc_info=True)
            return None


# ─── LangGraph Debate Graph ──────────────────────────────────────────────────


class DebateStrategy:
    """Multi-agent debate strategy orchestrated by LangGraph.

    Usage:
        strategy = DebateStrategy()
        signal = await strategy.analyze(market_data)
        # or synchronously:
        signal = strategy.analyze_sync(market_data)
    """

    def __init__(self):
        self.bull = SingleTurnAgent("Bull", BULL_SYSTEM_PROMPT)
        self.bear = SingleTurnAgent("Bear", BEAR_SYSTEM_PROMPT)
        self.hold = SingleTurnAgent("Hold", HOLD_SYSTEM_PROMPT)
        self.judge = JudgeAgent()

        # Build the LangGraph graph
        self.graph = self._build_graph()
        logger.info("DebateStrategy initialized with %d agents", 4)

    def _build_graph(self) -> StateGraph:
        """Construct the debate StateGraph.

        Nodes: debate (parallel bull/bear/hold) → adjudicate (judge)
        """
        builder = StateGraph(DebateState)

        builder.add_node("debate", self._debate_node)
        builder.add_node("adjudicate", self._adjudicate_node)

        builder.set_entry_point("debate")
        builder.add_edge("debate", "adjudicate")
        builder.add_edge("adjudicate", END)

        return builder.compile()

    async def _debate_node(self, state: DebateState) -> DebateState:
        """Run all three debaters in parallel."""
        prompt = state["market_prompt"]
        logger.info("Debate: launching 3 agents in parallel...")
        start = datetime.now(timezone.utc)

        bull_arg, bear_arg, hold_arg = await asyncio.gather(
            self.bull.arun(prompt),
            self.bear.arun(prompt),
            self.hold.arun(prompt),
        )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("Debate complete in %.1fs", elapsed)

        return {
            **state,
            "bull_argument": bull_arg,
            "bear_argument": bear_arg,
            "hold_argument": hold_arg,
        }

    async def _adjudicate_node(self, state: DebateState) -> DebateState:
        """Judge synthesizes all arguments into a TradingSignal."""
        signal = await self.judge.ajudge(
            state["market_prompt"],
            state["bull_argument"],
            state["bear_argument"],
            state["hold_argument"],
        )

        if signal is None:
            return {**state, "error": "Judge failed to produce a signal"}

        # Serialize to JSON for the state
        import json as _json
        return {
            **state,
            "final_signal_json": _json.dumps(signal.model_dump(), default=str),
        }

    def build_market_prompt(self, market_data: dict[str, Any]) -> str:
        """Build the market prompt from data."""
        from kimi_quant.llm import build_market_prompt

        return build_market_prompt(market_data)

    async def analyze(
        self, market_data: dict[str, Any]
    ) -> tuple[TradingSignal | None, dict[str, str]]:
        """Run the full debate asynchronously.

        Returns (signal, debate_transcript).
        """
        prompt = self.build_market_prompt(market_data)
        account = market_data.get("account")
        account_summary = account.to_summary() if account else "No position"

        initial_state: DebateState = {
            "market_prompt": prompt,
            "account_summary": account_summary,
            "bull_argument": "",
            "bear_argument": "",
            "hold_argument": "",
            "final_signal_json": "",
            "error": "",
        }

        logger.info("Starting multi-agent debate...")
        start = datetime.now(timezone.utc)

        final_state = await self.graph.ainvoke(initial_state)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("Debate + judgment complete in %.1fs", elapsed)

        if final_state.get("error"):
            logger.error("Debate error: %s", final_state["error"])
            return None, {
                "bull": final_state["bull_argument"],
                "bear": final_state["bear_argument"],
                "hold": final_state["hold_argument"],
            }

        # Deserialize the signal
        import json as _json

        try:
            signal_data = _json.loads(final_state["final_signal_json"])
            signal = TradingSignal(**signal_data)
        except Exception as e:
            logger.error("Failed to parse judge signal: %s", e)
            return None, {
                "bull": final_state["bull_argument"],
                "bear": final_state["bear_argument"],
                "hold": final_state["hold_argument"],
            }

        logger.info(
            "Final verdict: action=%s confidence=%.2f reasoning=%s",
            signal.action,
            signal.confidence,
            signal.reasoning[:100],
        )

        return signal, {
            "bull": final_state["bull_argument"],
            "bear": final_state["bear_argument"],
            "hold": final_state["hold_argument"],
        }

    def analyze_sync(
        self, market_data: dict[str, Any]
    ) -> tuple[TradingSignal | None, dict[str, str]]:
        """Synchronous wrapper for analyze()."""
        return asyncio.run(self.analyze(market_data))
