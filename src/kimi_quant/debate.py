"""Multi-Agent Debate Strategy via LangGraph with Checkpointing.

Three specialized agents (Bull, Bear, Hold) independently analyze market data
and present their arguments. A Judge agent weighs all arguments and makes
the final trading decision.

Each cycle's full state (market data → three arguments → verdict) is
automatically persisted via LangGraph's checkpointer, enabling:
  - Cycle history with thread_id
  - Crash recovery (resume from last checkpoint)
  - Full traceability for post-trade analysis

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
import json as _json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)

# Default checkpoint database path
DEFAULT_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "debate.db")


# ─── Debate State ────────────────────────────────────────────────────────────


class DebateState(TypedDict):
    """State carried through the debate graph and persisted to checkpointer."""

    market_prompt: str
    account_summary: str
    cycle_id: str  # ISO timestamp identifying this cycle
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


# ─── Single-Turn Agent ──────────────────────────────────────────────────────


class SingleTurnAgent:
    """A simple LLM agent that responds with a single message (no tool calling).

    Used for the debaters — each gets one turn to produce their output.
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


# ─── Judge Agent with Structured Output ─────────────────────────────────────


class JudgeAgent:
    """The judge agent uses LangChain structured output to produce a TradingSignal."""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=config.moonshot_api_key,
            base_url=config.moonshot_base_url,
            model=config.kimi_model,
            temperature=0.05,
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

            logger.info(
                "Judge deliberating on %d chars of debate...",
                len(debate_transcript),
            )
            signal: TradingSignal = await self.structured_llm.ainvoke(messages)
            return signal

        except Exception as e:
            logger.error("Judge failed: %s", e, exc_info=True)
            return None


# ─── Checkpointer Factory ───────────────────────────────────────────────────


class CheckpointerHandle:
    """Holds a checkpointer and manages its lifecycle.

    SqliteSaver.from_conn_string() is a context manager — the connection
    closes on __exit__. This handle keeps the context open for the
    application lifetime and closes it cleanly on shutdown.
    """

    def __init__(self, saver: BaseCheckpointSaver, ctx: Any | None = None):
        self.saver = saver
        self._ctx = ctx  # SqliteSaver context manager handle

    def close(self) -> None:
        """Release the checkpointer resources."""
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
                logger.info("Checkpointer connection closed")
            except Exception as e:
                logger.warning("Error closing checkpointer: %s", e)


def create_checkpointer(db_path: str | None = None) -> CheckpointerHandle:
    """Create the appropriate checkpointer.

    Uses SqliteSaver for production (persists to disk),
    MemorySaver as fallback.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        ctx = SqliteSaver.from_conn_string(db_path)
        saver = ctx.__enter__()
        logger.info("Checkpointer: SqliteSaver at %s", db_path)
        return CheckpointerHandle(saver, ctx)
    except Exception as e:
        logger.warning(
            "SqliteSaver unavailable (%s), falling back to MemorySaver", e
        )
        return CheckpointerHandle(MemorySaver())


# ─── LangGraph Debate Graph ──────────────────────────────────────────────────


class DebateStrategy:
    """Multi-agent debate strategy orchestrated by LangGraph.

    Uses LangGraph's built-in checkpointing to persist every cycle's
    full state: market data, each agent's argument, and the judge's verdict.

    Usage:
        strategy = DebateStrategy(checkpointer=SqliteSaver(...))
        signal, transcript = strategy.analyze_sync(market_data, thread_id="btc-1")
        # State is now persisted. Retrieve history:
        history = strategy.get_history("btc-1")
    """

    # Config key passed to graph.ainvoke to identify the trading session
    THREAD_ID = "btc-perpetual-trading"

    def __init__(self, checkpointer_handle: "CheckpointerHandle | None" = None):
        self.bull = SingleTurnAgent("Bull", BULL_SYSTEM_PROMPT)
        self.bear = SingleTurnAgent("Bear", BEAR_SYSTEM_PROMPT)
        self.hold = SingleTurnAgent("Hold", HOLD_SYSTEM_PROMPT)
        self.judge = JudgeAgent()

        if checkpointer_handle is None:
            checkpointer_handle = create_checkpointer()

        self._checkpointer_handle = checkpointer_handle
        self.checkpointer = checkpointer_handle.saver
        self.graph = self._build_graph()
        logger.info("DebateStrategy initialized with 4 agents + checkpointer")

    def close(self) -> None:
        """Release checkpointer resources."""
        self._checkpointer_handle.close()

    def _build_graph(self):
        """Construct the debate StateGraph with checkpointing.

        Nodes: debate (parallel bull/bear/hold) → adjudicate (judge)
        """
        builder = StateGraph(DebateState)

        builder.add_node("debate", self._debate_node)
        builder.add_node("adjudicate", self._adjudicate_node)

        builder.set_entry_point("debate")
        builder.add_edge("debate", "adjudicate")
        builder.add_edge("adjudicate", END)

        return builder.compile(checkpointer=self.checkpointer)

    async def _debate_node(self, state: DebateState) -> DebateState:
        """Run all three debaters in parallel."""
        prompt = state["market_prompt"]
        logger.info("Debate [%s]: launching 3 agents in parallel...",
                     state.get("cycle_id", "?"))
        start = datetime.now(timezone.utc)

        bull_arg, bear_arg, hold_arg = await asyncio.gather(
            self.bull.arun(prompt),
            self.bear.arun(prompt),
            self.hold.arun(prompt),
        )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("Debate [%s] complete in %.1fs",
                     state.get("cycle_id", "?"), elapsed)

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

        return {
            **state,
            "final_signal_json": _json.dumps(
                signal.model_dump(), default=str
            ),
        }

    def build_market_prompt(self, market_data: dict[str, Any]) -> str:
        """Build the market prompt from data."""
        from kimi_quant.llm import build_market_prompt

        return build_market_prompt(market_data)

    # ─── Public API ──────────────────────────────────────────────────────

    def _make_config(self, thread_id: str | None = None) -> dict:
        """Build the LangGraph config dict for checkpointing."""
        return {
            "configurable": {
                "thread_id": thread_id or self.THREAD_ID,
            }
        }

    async def analyze(
        self,
        market_data: dict[str, Any],
        thread_id: str | None = None,
    ) -> tuple[TradingSignal | None, dict[str, str]]:
        """Run the full debate asynchronously with checkpointing.

        Args:
            market_data: Market snapshot from DataProvider.
            thread_id: Checkpoint thread ID (default: 'btc-perpetual-trading').

        Returns:
            (signal, debate_transcript). Signal is None on failure.
        """
        prompt = self.build_market_prompt(market_data)
        account = market_data.get("account")
        account_summary = account.to_summary() if account else "No position"
        cycle_id = datetime.now(timezone.utc).isoformat()

        initial_state: DebateState = {
            "market_prompt": prompt,
            "account_summary": account_summary,
            "cycle_id": cycle_id,
            "bull_argument": "",
            "bear_argument": "",
            "hold_argument": "",
            "final_signal_json": "",
            "error": "",
        }

        graph_config = self._make_config(thread_id)
        logger.info("Starting debate [%s]...", cycle_id)
        start = datetime.now(timezone.utc)

        # State automatically checkpointed after each node execution
        final_state = await self.graph.ainvoke(initial_state, config=graph_config)

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
            "Final verdict [%s]: action=%s confidence=%.2f reasoning=%s",
            cycle_id,
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
        self,
        market_data: dict[str, Any],
        thread_id: str | None = None,
    ) -> tuple[TradingSignal | None, dict[str, str]]:
        """Synchronous wrapper for analyze()."""
        return asyncio.run(self.analyze(market_data, thread_id=thread_id))

    def get_history(
        self, thread_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve all checkpointed debate cycles from the database.

        Each entry contains the full DebateState for one cycle.
        Returns list ordered from oldest to newest.
        """
        graph_config = self._make_config(thread_id)
        history: list[dict] = []

        try:
            for checkpoint in self.graph.get_state_history(config=graph_config):
                # Filter out empty placeholder states
                if checkpoint.values.get("cycle_id"):
                    history.append(dict(checkpoint.values))
        except Exception as e:
            logger.error("Failed to retrieve history: %s", e)

        # get_state_history returns newest first, reverse to oldest first
        history.reverse()
        return history

    def get_latest_state(
        self, thread_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get the most recent checkpointed state (e.g. for crash recovery)."""
        graph_config = self._make_config(thread_id)
        try:
            snapshot = self.graph.get_state(config=graph_config)
            if snapshot and snapshot.values.get("cycle_id"):
                return dict(snapshot.values)
        except Exception as e:
            logger.error("Failed to get latest state: %s", e)
        return None
