"""Multi-Agent Debate Strategy via LangGraph.

Three specialized agents (Bull, Bear, Hold) independently analyze market data
and present their arguments. A Judge agent weighs all arguments and makes
the final trading decision.

Each cycle's result is persisted to a JSONL file for:
  - Cycle history
  - Crash recovery (resume from last checkpoint)
  - Full traceability for post-trade analysis

Architecture (LangGraph StateGraph):

    START
      │
      ▼
  ┌─────────┐
  │ Prepare │
  └────┬────┘
       │
       ▼
  ╔══════════════════════════════════════════════════╗
  ║  Phase 1: Hold Agent (cache warm-up)             ║
  ║  ┌──────────────┐                                ║
  ║  │ Hold Agent   │──▶ populates KV-cache          ║
  ║  │ (论证观望)    │    for shared market prefix    ║
  ║  └──────────────┘                                ║
  ╚══════════════════════════════════════════════════╝
       │
       ▼
  ╔══════════════════════════════════════════════════╗
  ║  Phase 2: Bull + Bear (cache hits)               ║
  ║  ┌──────────────┐  ┌──────────────┐              ║
  ║  │ Bull Agent   │  │ Bear Agent   │  parallel    ║
  ║  │ (论证做多)    │  │ (论证做空)    │  ~63% fewer  ║
  ║  └──────────────┘  └──────────────┘  input tokens║
  ╚══════════════════════════════════════════════════╝
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
import fcntl
import json as _json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from kimi_quant.config import config
from kimi_quant.llm import TradingSignal

logger = logging.getLogger(__name__)

# Default debate history path
DEFAULT_HISTORY_PATH = str(
    Path(__file__).parent.parent.parent / "data" / "debate.jsonl"
)


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
#
# Prefix-caching design: market data is placed in the system message (shared
# across all 3 agents) so DeepSeek V3's automatic KV-cache reuse kicks in.
# Persona instructions go in the user message (the varying suffix).
# Result: agent 2 and 3 only pay for ~50 new input tokens instead of ~1500.
#
# Shared system context — prepended to market data for all 3 agents.
DEBATE_SHARED_SYSTEM = (
    "You are a BTC perpetual quant analyst on Hyperliquid. "
    "Below is real-time market data. Analyze it from the specific perspective "
    "requested in the user message.\n"
    "Rules: Be specific — cite actual prices, sizes, funding rates. "
    "Be honest — if your side has weak evidence, say so. "
    "150-200 words. Plain text only, no JSON."
)

# Persona instructions — these are the varying suffix (NOT cached).
BULL_USER_PROMPT = (
    "Role: Bullish BTC Trader\n"
    "Find the strongest case for LONG: bid walls, buying pressure, "
    "low/negative funding, support levels, oversold bounces, accumulation signals."
)

BEAR_USER_PROMPT = (
    "Role: Bearish BTC Trader\n"
    "Find the strongest case for SHORT: ask walls, selling pressure, "
    "high positive funding (crowded longs), resistance levels, distribution "
    "signals, bearish divergences."
)

HOLD_USER_PROMPT = (
    "Role: Cautious Risk Manager\n"
    "Find reasons to STAY OUT: conflicting signals, wide spreads, choppy action, "
    "mid-range price, unclear multi-TF alignment, poor risk/reward. "
    "If market truly has clear direction, acknowledge it."
)

JUDGE_SYSTEM_PROMPT = """\
You are the Head Trader. Your team (Bull/Long, Bear/Short, Risk/Hold) debated. \
Weigh their arguments and decide: LONG, SHORT, CLOSE, HOLD, MODIFY_SL, or MODIFY_TP. \
For multi-step actions, use the `actions` array: ["CLOSE", "SHORT"] to flip, \
["MODIFY_SL", "MODIFY_TP"] to adjust both stops.

DECISION WORKFLOW — follow this order every cycle:

Step 0 — ASSESS EXISTING STATE FIRST (before weighing the debate):
  a. If there IS a position: is the original thesis still valid? If broken → CLOSE.
     If working → consider MODIFY_SL to lock in profit or move to breakeven.
  b. Check open orders: are SL/TP orders actually on the chain? If MISSING →
     position is UNPROTECTED → MODIFY_SL/MODIFY_TP immediately, or CLOSE.
     This takes priority over everything else.
  c. Are there stale orders on chain? Clean them up if needed.
  d. Are SL/TP levels appropriate for current ATR? Adjust if not.

Step 1 — WEIGH THE DEBATE (only after completing Step 0):
  Decision framework (higher TF = more weight: 4h > 1h > 15m > 5m):
  - 1h+4h aligned + strong argument → confidence 0.75+
  - Higher TF clear, lower TF diverging → follow higher TF, reduce size, \
    confidence 0.65-0.75
  - ALL timeframes sideways + all arguments weak → HOLD acceptable
  - DON'T default to HOLD just because timeframes diverge

Guidelines:
- Trust specific data references (prices, sizes) over rhetoric
- Divergence = smaller size + tighter stop, NOT automatic HOLD
- Confidence < 0.65 → skip trade
- stop_loss mandatory for LONG/SHORT, min 0.5% from entry
- If Step 0 found issues (missing SL/TP, stale orders), include the fix
  actions BEFORE any new entry actions from the debate.

Output TradingSignal JSON:
- actions: ordered list of actions (preferred). Use for flip ["CLOSE", "SHORT"],
  adjust both ["MODIFY_SL", "MODIFY_TP"], or single ["LONG"].
  For backward compatibility, may also output a single `action` string.
- confidence, reasoning, size (BTC), entry_price (null=market),
  stop_loss, take_profit, modify_sl_to, modify_tp_to, key_factors (2-4 items),
  next_interval (null=default, range 300-10800s): ONLY use 300-600 for
  active positions or confirmed breakout setups. Default to null or 1800+
  for normal HOLD conditions. When in doubt, go longer.
"""


# ─── Single-Turn Agent ──────────────────────────────────────────────────────


class SingleTurnAgent:
    """A simple LLM agent that responds with a single message (no tool calling).

    Used for the debaters — each gets one turn to produce their output.
    Automatically falls back to DeepSeek if Kimi fails.

    Prefix-caching: the market data is placed in the system message so all
    3 agents share an identical prompt prefix. DeepSeek V3's automatic
    KV-cache reuse means agents 2 and 3 only pay for the ~50-token user
    message instead of the full ~1500-token market prompt.
    """

    def __init__(self, name: str, user_prompt: str):
        from kimi_quant.llm import create_llm

        self.name = name
        self.user_prompt = user_prompt  # Persona instruction (short, varying)
        self.llm = create_llm()

    async def arun(self, market_prompt: str) -> str:
        """Run the agent asynchronously and return its text response.

        Args:
            market_prompt: The full market data text — placed in the system
                message as a shared prefix for all 3 agents (cacheable).
        """
        try:
            messages = [
                ("system", DEBATE_SHARED_SYSTEM + "\n\n" + market_prompt),
                ("user", self.user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Agent %s failed: %s", self.name, e)
            return f"[{self.name} failed to respond: {e}]"


# ─── Judge Agent with Structured Output ─────────────────────────────────────


class JudgeAgent:
    """The judge agent uses structured output via json_mode (response_format: json_object).

    json_mode is the only structured output method supported by DeepSeek
    (json_schema and function_calling both return 400). Kimi also supports it.
    """

    def __init__(self):
        from kimi_quant.llm import create_structured_llm

        self.structured_llm = create_structured_llm(
            TradingSignal,
            temperature=config.judge_temperature,
            max_tokens=4096,  # Judge needs more room for synthesizing 3 arguments
        )

    async def ajudge(
        self, account_summary: str, bull: str, bear: str, hold: str
    ) -> TradingSignal | None:
        """Asynchronously judge the debate and produce a TradingSignal.

        The account summary and trading constraints are included so the Judge
        knows the actual account state and position size limits — without this,
        the Judge can propose sizes far beyond what the account can afford.
        """
        try:
            size_limit = config.max_position_size
            max_lev = config.max_leverage
            constraints = (
                f"Max position size: {size_limit} BTC | Max leverage: {max_lev}x\n"
                f"Size must respect available balance (notional / {max_lev}x ≤ available).\n"
                f"When in doubt, use smaller size. Never exceed {size_limit} BTC."
            )
            debate_transcript = (
                "# === ACCOUNT CONTEXT ===\n"
                f"{account_summary}\n"
                f"Trading constraints: {constraints}\n\n"
                "# === DEBATE TRANSCRIPT ===\n\n"
                "## 🐂 BULL ANALYST (LONG Case)\n"
                f"{bull}\n\n"
                "## 🐻 BEAR ANALYST (SHORT Case)\n"
                f"{bear}\n\n"
                "## 😐 RISK MANAGER (HOLD Case)\n"
                f"{hold}\n\n"
                "# === YOUR DECISION ===\n"
                "Weigh the arguments above against the account context. "
                "The debaters have already referenced all relevant market data "
                "(prices, levels, funding, order book, multi-timeframe trends) "
                "in their arguments. "
                f"IMPORTANT: size must not exceed {size_limit} BTC. "
                "Produce the final trading signal."
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


# ─── Debate History Persistence ──────────────────────────────────────────────


def _read_history(history_path: str) -> list[dict[str, Any]]:
    """Read all debate history entries from the JSONL file."""
    entries: list[dict] = []
    try:
        path = Path(history_path)
        if not path.exists():
            return entries
        with open(path) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            pass
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error("Failed to read debate history: %s", e)
    return entries


def _append_history(history_path: str, entry: dict[str, Any]) -> None:
    """Append a single debate cycle entry to the JSONL file."""
    try:
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(_json.dumps(entry, default=str) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error("Failed to persist debate history: %s", e)


# ─── LangGraph Debate Graph ──────────────────────────────────────────────────


class DebateStrategy:
    """Multi-agent debate strategy orchestrated by LangGraph.

    Each cycle's debate is persisted to a JSONL file (debate.jsonl)
    alongside LangGraph's MemorySaver (intra-session checkpointing).

    Usage:
        strategy = DebateStrategy()
        signal, transcript = strategy.analyze_sync(market_data)
        history = strategy.get_history()
    """

    # Config key passed to graph.ainvoke to identify the trading session
    THREAD_ID = "btc-perpetual-trading"

    def __init__(self, history_path: str | None = None,
                 debate_timeout: int = 60):
        self.bull = SingleTurnAgent("Bull", BULL_USER_PROMPT)
        self.bear = SingleTurnAgent("Bear", BEAR_USER_PROMPT)
        self.hold = SingleTurnAgent("Hold", HOLD_USER_PROMPT)
        self.judge = JudgeAgent()
        self.debate_timeout = debate_timeout
        self._history_path = history_path or DEFAULT_HISTORY_PATH

        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("DebateStrategy initialized: 4 agents, timeout=%ds",
                     debate_timeout)

    def close(self) -> None:
        """No-op (MemorySaver needs no cleanup)."""
        pass

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
        """Two-phase debate with prefix-cache warmup.

        Phase 1 — Hold runs solo. Its prefill populates the KV-cache for
        the shared market-data prefix on the API backend.
        Phase 2 — Bull + Bear run in parallel. Both hit the warm cache,
        paying only for their ~50-token persona suffix (vs ~1500 tokens).

        Trade-off: adds ~Hold's latency to total wall time vs pure parallel.
        Hold is typically the fastest agent (simplest analysis), and the
        input-token savings (~63%) outweigh the latency cost for most users.
        """
        prompt = state["market_prompt"]
        cycle_id = state.get("cycle_id", "?")
        logger.info("Debate [%s]: Phase 1 — warming cache via Hold agent...",
                     cycle_id)
        start = datetime.now(timezone.utc)

        async def _run_with_timeout(agent, name: str) -> str:
            try:
                return await asyncio.wait_for(
                    agent.arun(prompt), timeout=self.debate_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Agent %s timed out after %ds", name, self.debate_timeout)
                return (
                    f"[{name} TIMEOUT after {self.debate_timeout}s — "
                    f"could not complete analysis in time. "
                    f"Proceed with available arguments from other agents.]"
                )

        # Phase 1: Hold warms the prefix cache
        hold_arg = await _run_with_timeout(self.hold, "Hold")

        # Phase 2: Bull + Bear in parallel, both hit the warm cache
        logger.info("Debate [%s]: Phase 2 — Bull + Bear (cache warm)...",
                     cycle_id)
        bull_arg, bear_arg = await asyncio.gather(
            _run_with_timeout(self.bull, "Bull"),
            _run_with_timeout(self.bear, "Bear"),
        )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("Debate [%s] complete in %.1fs", cycle_id, elapsed)

        return {
            **state,
            "bull_argument": bull_arg,
            "bear_argument": bear_arg,
            "hold_argument": hold_arg,
        }

    async def _adjudicate_node(self, state: DebateState) -> DebateState:
        """Judge synthesizes all arguments into a TradingSignal."""
        signal = await self.judge.ajudge(
            state["account_summary"],
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
        """Build the market prompt from data (with multi-timeframe analysis)."""
        from kimi_quant.data import DataProvider
        return DataProvider.build_llm_prompt(market_data)

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
        """Run the full debate asynchronously, persisting results to JSONL.

        Args:
            market_data: Market snapshot from DataProvider.
            thread_id: Checkpoint thread ID (default: 'btc-perpetual-trading').

        Returns:
            (signal, debate_transcript). Signal is None on failure.
        """
        prompt = self.build_market_prompt(market_data)
        account = market_data.get("account")
        account_summary = account.to_summary() if account else "No position"

        # Inject open orders context so the Judge can see SL/TP state
        orders_summary = market_data.get("open_orders_summary", "")
        if orders_summary:
            account_summary += "\n" + orders_summary
        sl_tp_status = market_data.get("sl_tp_status", {})
        if sl_tp_status.get("sl_missing"):
            account_summary += (
                "\n⚠️ STOP LOSS MISSING from exchange — position is unprotected!"
            )
        if sl_tp_status.get("tp_missing"):
            account_summary += (
                "\n⚠️ TAKE PROFIT MISSING from exchange!"
            )

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

        final_state = await self.graph.ainvoke(initial_state, config=graph_config)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("Debate + judgment complete in %.1fs", elapsed)

        # Persist to JSONL for cross-session history
        self._save_cycle(final_state)

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

    def _save_cycle(self, final_state: dict) -> None:
        """Persist a debate cycle to the JSONL history file."""
        entry = {
            "cycle_id": final_state.get("cycle_id", ""),
            "account_summary": final_state.get("account_summary", ""),
            "bull_argument": final_state.get("bull_argument", ""),
            "bear_argument": final_state.get("bear_argument", ""),
            "hold_argument": final_state.get("hold_argument", ""),
            "final_signal_json": final_state.get("final_signal_json", ""),
            "error": final_state.get("error", ""),
        }
        _append_history(self._history_path, entry)

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
        """Retrieve all debate cycles from the JSONL history file.

        Returns list ordered from oldest to newest.
        """
        return _read_history(self._history_path)

    def get_latest_state(
        self, thread_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get the most recent debate cycle (e.g. for crash recovery)."""
        history = _read_history(self._history_path)
        return history[-1] if history else None
