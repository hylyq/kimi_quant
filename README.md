# Kimi Quant 🚀

**BTC Perpetual Futures Quantitative Trading Bot** — An LLM-powered autonomous trading system running on Hyperliquid DEX. The LLM decides when to trade, how to trade, and when to rest. One command to start, no cron needed.

> 📖 **中文文档**: [README_CN.md](./README_CN.md)

### Decision & Strategy
- 🎯 **Mandatory Decision Workflow**: Step 0 Review Positions → Step 1 Market Analysis → Step 1.5 Counterfactual Checklist → Output Signal
- 💬 **Single/Dual Strategy Modes**: Single (fast & lightweight) / Debate (Bull+Bear+Hold 3-agent debate + optional rebuttal round + Judge cross-validation ruling)
- 📌 **Position Memory**: LLM sees original entry thesis, holding duration, MFE/MAE — truly validates whether the thesis still holds
- 🔄 **Decision Feedback Loop**: Previous cycle's decision→outcome auto-injected, "you said wait for pullback → now it's pulled back" autonomous learning

### Risk Control & Safety
- 🛡️ **Seven-Layer Risk Control**: Circuit breaker, confidence, position size, margin, risk amount, stop-loss distance, direction constraints
- 🔁 **Risk Correction**: Rejected signals get one LLM revision opportunity (e.g., widen SL, reduce size)
- 💰 **Tick Size Auto-Alignment**: All order prices auto-round to exchange minimum price unit, preventing batch order rejections
- 🔍 **SL/TP On-Chain Verification**: Cross-reference tracker vs on-chain orders every cycle; missing orders trigger immediate alert and recovery
- 🔐 **TLS Fingerprint Spoofing**: curl_cffi simulates Firefox fingerprint, bypassing TLS inspection on cloud servers (Alibaba Cloud, etc.)

### Data & Analysis
- ⏰ **Timezone/Calendar Awareness**: Auto-injects UTC/Beijing time, day of week, trading sessions (Asia/EU/US), weekend low-liquidity warnings
- 📊 **Multi-Timeframe Analysis**: 5m/15m/1h/4h K-lines + ATR + Order Book + Funding Rate
- 📈 **Inter-Cycle Diff + Volatility Classification**: Auto-compares with previous cycle data + HIGH/NORMAL/LOW volatility regime detection

### Cost & Performance
- 🧠 **Dual Model Failover**: Kimi K3 + DeepSeek V4, automatic fallback if primary fails; Judge can independently specify a stronger model
- 💸 **DeepSeek Context Caching**: 55-73% input token savings within Debate cycles; weak-debater strong-judge config saves 65% vs all-Kimi
- 🕐 **Adaptive Wake Interval**: LLM decides when to wake next (5min-3h), auto-lengthens during sideways markets to save costs

### Operations & Tooling
- 📱 **Push Notifications**: WeChat/Feishu real-time trade event notifications, auto-detection, no config needed
- 🔄 **WebSocket Real-Time Monitoring**: Order fill/SL/TP trigger detection at millisecond level; Flash model generates Chinese notifications
- 🔧 **CLI Account Tools**: One-click balance/position/order checks, account type switching, fund transfers
- 📝 **Complete Logging**: Trade P&L + debate history JSONL persistence with fcntl file locking for concurrent read/write

## Table of Contents

- [Quick Start](#quick-start)
- [Strategy Modes](#strategy-modes)
- [Architecture Overview](#architecture-overview)
- [LLM Model Configuration](#llm-model-configuration)
- [Adaptive Wake Interval](#adaptive-wake-interval)
- [DeepSeek Context Disk Caching](#deepseek-context-disk-caching)
- [Push Notifications](#push-notifications-wechat--feishu)
- [Real-Time Order Monitoring](#real-time-order-monitoring)
- [Phased Testing Guide](#phased-testing-guide)
- [Production Deployment](#production-deployment)
- [Configuration Reference](#configuration-reference)
- [CLI Commands](#cli-commands)
- [Account Management](#account-management)
- [TradingSignal](#tradingsignal)
- [Risk Control Rules](#risk-control-rules)
- [Prompt Enhancement System (v3.0)](#prompt-enhancement-system-v30)
- [Debate Mode Deep Dive](#debate-mode-deep-dive)
- [TradeExecutor SDK Coverage](#tradeexecutor--full-sdk-coverage)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [Risk Disclaimer](#risk-disclaimer)

## Strategy Modes

### Single-Agent Mode
```
Market Data → Kimi K3 Analysis → TradingSignal → Risk Control → Execution
```
1 LLM call, fast and lightweight. Suitable for most scenarios.

### Multi-Agent Debate Mode
```
              ┌── 🐂 Bull Agent (argues LONG) ──┐
Market Data ──┼── 🐻 Bear Agent (argues SHORT) ─┼── ⚖️ Judge ──→ TradingSignal ──→ Risk ──→ Execute
              └── 😐 Hold Agent (argues WAIT) ──┘        ↑
                                            Full market data + all 3 arguments
```
3 agents debate in parallel + 1 judge ruling, reducing single-model bias through adversarial validation. 4 LLM calls per cycle.
**All 4 agents see the same raw market data** (prices, multi-TF K-lines, order book, funding rate, risk constraints) — the Judge can cross-verify debater claims against the source data.

### Decision Workflow

Regardless of strategy mode, the LLM follows a **mandatory two-step decision process** each cycle:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0 — Review existing state (MUST run before analysis)     │
│                                                             │
│  a. Position Validity: Does the original entry thesis hold?  │
│     → Thesis broken → CLOSE                                 │
│     → Working well → consider MODIFY_SL to lock profit       │
│                                                             │
│  b. SL/TP On-Chain Verification: Are tracker SL/TPs on-chain?│
│     → Missing → MODIFY_SL/MODIFY_TP restore NOW (highest)    │
│     → Present → continue                                     │
│                                                             │
│  c. Zombie Order Cleanup: Any non-bot orders on-chain?       │
│                                                             │
│  d. SL/TP Distance Check: Do SL/TP match current ATR?        │
│     → Volatility decreased → tighten SL                      │
│     → Volatility increased → widen SL                        │
├─────────────────────────────────────────────────────────────┤
│ Step 1 — Market Analysis (only after Step 0 is complete)     │
│                                                             │
│  Multi-TF trend → Order Book → Funding Rate → Decision       │
│  Step 0 fix actions MUST be placed first in the actions array│
├─────────────────────────────────────────────────────────────┤
│ Step 1.5 — Mandatory Counterfactual Check (before directional)│
│                                                             │
│  1. Invalidation Price: "If BTC breaks $___, my thesis fails" │
│     → This price IS your stop-loss — don't set it arbitrarily │
│  2. ATR Validation: "1h ATR = X.X%, SL must ≥ 1.5× ATR"      │
│  3. 15-Min Reversal: "If 2% reversal in 15min, what signal?"  │
│  4. Bias Audit: Evidence (0-10) ___ vs Expectation (0-10) ___ │
│     → Expectation > evidence → confidence -0.15 or HOLD       │
│  Any serious doubt → HOLD is correct. There's always next.    │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: Step 0 fix actions (e.g., restore missing SL) take priority over Step 1 new positions. If Step 0 finds a missing SL and the market shows a LONG signal, the LLM should output `["MODIFY_SL", "LONG"]` not just `["LONG"]` — protect the position first, then open new.

This design is enforced through three-layer prompting (System Prompt + User Prompt Instructions + Judge Prompt) without extra code logic. In Debate mode, the Judge shares the same raw market data as debaters (via the `RAW MARKET DATA` block) for direct cross-verification. **Step 1.5** requires the LLM to answer four checklist items before directional decisions (invalidation→SL, ATR check, 15-min reversal scan, bias audit) rather than a vague "question yourself" — reducing overconfidence-driven oversized positions. **Inter-cycle diffs** automatically annotate actual elapsed time (since the LLM controls its own wake interval), letting the LLM focus on changes rather than re-analyzing all data.

**Prompt Enhancement System** (v3.0) injects seven categories of additional context before LLM decisions: ⏰ Timezone/Calendar/Trading Sessions, 📌 Position Memory (original entry thesis + MAE/MFE), 🔄 Previous Cycle Decision Feedback Loop, 📊 Volatility Regime Classification (HIGH/NORMAL/LOW) with SL width guidance, 🧮 Expected Value Framework (R:R → breakeven win rate), 📚 Recent Trade Lessons Summary, ⚠️ Hard Constraints Repeated at End (primacy + recency effects). See [Prompt Enhancement System](#prompt-enhancement-system-v30).

## Architecture Overview

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
│  DataProvider │───▶│ Strategy Engine  │───▶│  RiskManager │───▶│  TradeExecutor   │
│  (Hyperliquid)│    │ single / debate │    │  (7 layers)  │    │  (full lifecycle) │
└──────────────┘    └─────────────────┘    └──────────────┘    └──────────────────┘
      ▲                      │                                           │
      │                      ▼                                           │
      │              ┌──────────────┐                                    │
      └──────────────│ TradeLogger  │◀───────────────────────────────────┘
                     │ (P&L feedback)│
                     └──────────────┘
                           │
┌──────────────────┐       │       ┌──────────────────┐
│   OrderMonitor   │───────┼──────▶│  FlashReporter   │
│  (WebSocket RT)  │  tracker      │  (Flash LLM)      │
│  orders/fills/liq│   sync        │  CN notifications  │
└──────────────────┘               └──────────────────┘
         │                                  │
         └────── WeChat/Feishu Push ◀───────┘
```

### Data Flow

1. **DataProvider** — Market data via mainnet (`meta_and_asset_ctxs`, with mark/oracle/funding/OI), multi-TF K-line TTL cache. Syncs all on-chain open orders (`open_orders`). **Inter-cycle diff**: auto-compares with previous snapshot, injecting a `# 📊 Since Last Cycle (X.Xmin ago)` block — because the LLM controls its wake interval, the diff annotates actual elapsed time for correct change assessment
2. **Strategy Engine** — Single: one LLM analysis; Debate: 3-agent debate + Judge ruling (60s timeout), optional **rebuttal round** (debaters rebut each other before ruling). All agents receive identical market data (prices, multi-TF K-lines, order book, funding rate, risk constraints, inter-cycle diff); Judge additionally sees all three arguments for cross-verification. Both modes follow the Step 0→Step 1→Step 1.5 decision workflow
3. **RiskManager** — Seven-layer risk checks + `validate_sequence()` multi-operation state simulation (supports CLOSE+LONG/SHORT flip through risk). **Risk rejection → LLM one correction**: rejection reason fed back to the original decision-maker; corrected signal re-validated and executed if passed (can be disabled via `RISK_CORRECTION_ENABLED`)
4. **TradeExecutor** — Startup recovery + resting/active state machine + SL/TP price tracking + multi-operation sequential execution (fail-fast) + 15/15 SDK coverage
5. **TradeLogger** — P&L analysis + LLM performance feedback (introspection loop)
6. **OrderMonitor** — WebSocket real-time order state subscription (filled/partial/canceled/liquidated), millisecond-level sync to PositionTracker, giving the LLM the latest state next cycle
7. **FlashReporter** — Consumes WS events → Flash model generates Chinese natural language notifications → WeChat/Feishu push
8. **Context Injection** — Each cycle injects into the LLM prompt:
   - **Timezone/Calendar**: UTC/Beijing time, day of week, trading sessions (Asia/EU/US), weekend low-liquidity warning
   - **Position Memory**: Original entry thesis, holding duration, MAE/MFE
   - **Previous Cycle Feedback**: Last cycle's decision→outcome closed loop ("You said wait for pullback — has it pulled back?")
   - **Volatility Regime**: HIGH/NORMAL/LOW classification + SL width/position size guidance
   - **Expected Value Framework**: R:R → implied breakeven win rate, so the LLM knows what confidence threshold to beat
   - **Recent Lessons**: Patterns extracted from closed trades (repeated stop-outs, one-sided win rate, TP-then-trend-continues)
   - **Account/Risk**: Balance, positions, all on-chain orders (including orphans), tracker SL/TP prices, hard risk constraints (circuit breaker state, margin budget, risk budget, direction limits) — LLM knows the boundaries before deciding
   - **Hard Constraints Repeated**: Max position/min confidence/SL requirement/leverage repeated at prompt end (primacy + recency effects)
9. **Adaptive Interval** — LLM suggests next wake time (5min-3h), saving costs in sideways markets and tightening during key levels

### Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Kimi K3 (primary) + DeepSeek V4 (auto-fallback) |
| LLM Orchestration | LangChain + LangGraph StateGraph |
| State Persistence | LangGraph MemorySaver + JSONL files (fcntl locks) |
| Exchange | Hyperliquid (Perpetual DEX) |
| Real-Time Monitoring | Hyperliquid WebSocket + deepseek-v4-flash (lightweight reporting) |
| Structured Output | Pydantic + LangChain json_mode (response_format: json_object) |
| Trade Execution | hyperliquid-python-sdk (15/15 full coverage) |

## LLM Model Configuration

### Dual Model Failover Architecture

```
Each LLM call ──▶ Primary Model ──success──▶ Return
                    │
                    │ failure (timeout/429/5xx/balance)
                    ▼
                 Fallback Model ──success──▶ Return
                    │
                    │ also fails
                    ▼
               raise → upper-level handling
```

- Primary/fallback switching is fully transparent to strategy code — no business logic changes needed
- Startup logs clearly show the current chain: `LLM: kimi primary → fallback: deepseek`

### Model Selection

```bash
# .env

# Option 1: Kimi primary → DeepSeek fallback (default, recommended)
PRIMARY_LLM=kimi
MOONSHOT_API_KEY=sk-your-kimi-key
DEEPSEEK_API_KEY=sk-your-deepseek-key

# Option 2: DeepSeek primary → Kimi fallback (budget, ~1/10 of Kimi cost)
PRIMARY_LLM=deepseek

# Option 3: Debate mode — weak debaters + strong Judge (🏆 recommended)
# Debaters do directed search (just need to find evidence), Judge does synthesis (needs strong reasoning)
PRIMARY_LLM=deepseek        # Bull/Bear/Hold use cheap DeepSeek (3 calls)
JUDGE_PRIMARY_LLM=kimi       # Judge uses strong-reasoning Kimi K3 (1 call)

# Option 4: Kimi only (don't set DeepSeek key)
# Option 5: DeepSeek only (don't set Kimi key)
```

### Reasoning Effort Control

One `REASONING_EFFORT` variable, auto-converted to each model's native API format:

```bash
REASONING_EFFORT=max      # Strongest reasoning (default)
REASONING_EFFORT=high     # High reasoning
REASONING_EFFORT=medium   # Medium
REASONING_EFFORT=low      # Low reasoning
REASONING_EFFORT=minimal  # Minimal reasoning
REASONING_EFFORT=off      # Disable reasoning, significant token savings
```

Underlying conversion:

| Config | Kimi K3 | DeepSeek V3/V4 |
|--------|---------|----------------|
| `max` | `reasoning_effort: "max"` | `extra_body → thinking: enabled` |
| `high` ~ `minimal` | Not passed (K3 only supports max) | `extra_body → thinking: enabled` |
| `off` | Not passed | `extra_body → thinking: disabled` |

**Cost Impact**: Reasoning tokens account for ~80% of output. `REASONING_EFFORT=off` saves approximately **75% on output costs** per call. Suitable for high-frequency polling or low-cost modes.

> **Structured JSON Output**: Uses `response_format: json_object` (LangChain `json_mode`). This is the only structured output method officially supported by DeepSeek (`json_schema` and `function_calling` both return 400). Kimi is also compatible. Schema is passed to the model via system prompt.
>
> ⚠️ **Kimi K3 Temperature Limitation**: Kimi K3 is a reasoning model that **only accepts `temperature=1`**. The program auto-overrides `LLM_TEMPERATURE` to 1.0 for Kimi only — no manual config change needed. DeepSeek is unaffected.

### Debate Mode: Independent Judge Model (Weak Debaters, Strong Judge)

In Debate mode, the 4 agents have asymmetric cognitive loads:

| | 🐂🐻😐 Debaters (×3) | ⚖️ Judge (×1) |
|---|---|---|
| **Task** | Single-perspective directed search (only find bull/bear/risk signals) | Synthesize 3 arguments + cross-reference raw data + multi-timeframe trade-offs + account constraints → final decision |
| **Bias Sensitivity** | Intentionally biased (by role design), bias is a feature | Must identify and offset debater biases |
| **Error Cost** | Low — one weak debater, two others can compensate | High — Judge error = trade decision error |
| **Suitable Model** | Cheap, fast weak model | Strong reasoning model |

**Recommended Config**: Debaters use cheap DeepSeek, Judge uses strong-reasoning Kimi K3.

```bash
# .env
PRIMARY_LLM=deepseek        # Bull/Bear/Hold use DeepSeek (3 calls)
JUDGE_PRIMARY_LLM=kimi       # Judge uses Kimi K3 (1 call)
```

Failover: Judge's Kimi fails → auto-fallback to DeepSeek. Debaters unaffected.

Cost comparison (10-min interval, Debate mode monthly):

| Config | Debaters×3 | Judge×1 | Monthly Cost | Decision Quality |
|--------|-----------|---------|-------------|:--:|
| All Kimi | Kimi | Kimi | ~$62 | High |
| All DeepSeek | DeepSeek | DeepSeek | ~$8 | Medium |
| **Weak Debater Strong Judge** | DeepSeek | Kimi | **~$22** | **High** ✅ |

Saves **65%** vs all-Kimi, Judge decision quality unaffected — final rulings depend on reasoning synthesis ability, not debater eloquence.

Startup logs reflect independent config:

```
LLM: deepseek primary → fallback: kimi        ← Debaters
Judge LLM: kimi primary → fallback: deepseek   ← Judge
```

## Adaptive Wake Interval

**The LLM decides for itself when to wake up next.** No external cron. No manual `--once` repeated runs.

### How It Works

```
uv run kimi-quant
    │
    └── while True:
           ├── Fetch market data → LLM analysis → Risk → Execute
           ├── LLM returns: "next_interval": 900
           ├── sleep(900)           ← sleeps 15 minutes
           └── Wakes, repeats
```

The LLM dynamically adjusts based on market conditions:

| Market State | LLM Suggested Interval | Effect |
|-------------|:----------:|------|
| Position held, breakout confirming | 300-600s | Moderate monitoring |
| Normal market, no position | 600-1800s | Cost optimization priority |
| Sideways, no direction | 1800-3600s | Reduce costs |
| Weekend, low liquidity | 3600-10800s | Maximum savings |
| LLM leaves blank | Uses `TRADING_INTERVAL` default | Default behavior |

Each cycle's prompt auto-injects an **inter-cycle diff** (`# 📊 Since Last Cycle (X.Xmin ago)`) annotating actual elapsed time. Since intervals are LLM-decided — the same +0.26% means very different things at 5 minutes vs 2 hours — the diff lets the LLM correctly assess change significance.

### Boundary Protection

Program applies clamping: `[MIN_INTERVAL, MAX_INTERVAL]`, defaults `[300s, 10800s]` (5 min ~ 3 hours), adjustable via env vars. The LLM prompt is also guided to prefer longer intervals — only suggesting 300-600s when holding positions or confirming breakouts.

### Zero Configuration

No new parameters needed. The LLM knows the allowed range through the system prompt and auto-suggests appropriate intervals in each response. Startup logs show interval changes:

```
Cycle 5 complete: signal=HOLD confidence=0.65
LLM adjusted interval: 600s → 900s    ← Sideways, lengthened
Sleeping 900.0s until next cycle...
```

## DeepSeek Context Disk Caching

DeepSeek API has context disk caching enabled by default for all users (official docs), usable without code changes.

### Cache Mechanism

The system caches prefix units to disk via three mechanisms. Subsequent requests matching complete units get cache hits:

| Mechanism | Trigger | How We Use It |
|-----------|---------|---------------|
| **Request end position** | Each request completion | — (Debate debaters have different user messages, can't directly hit) |
| **Common prefix detection** | Multiple requests sharing prefix | Single mode: after 2 cycles, `SYSTEM_PROMPT` becomes common prefix |
| **Fixed token intervals** | Long inputs cached at intervals | **Core mechanism for intra-Debate-cycle hits**: Hold runs first → interval-cache units written → Bull/Bear hit system message portion |

### Debate Mode Cache Warmup

Debate mode's Phase 1 (Hold first) → Phase 2 (Bull + Bear parallel) design leverages fixed-interval caching:

```
Cycle N:
  Phase 1: Hold ──▶ Fixed-interval cache units written to disk (~2000 token system message)
                      │
                      ▼ (~2s write delay)
  Phase 2: Bull ──▶ Cache hit (only ~50 token user message billed)  ┐
           Bear ──▶ Cache hit (only ~50 token user message billed)  ┘ parallel
```

**Key Finding**: DeepSeek cache write delay is "seconds-level" (official docs). If Hold responds very fast (<1 second), Bull/Bear may send requests before cache is ready → all tokens billed at full price.

`CACHE_WARMUP_DELAY` (default 2.0s) waits after Hold completes to ensure stable Bull/Bear cache hits.

### Cache Hit Monitoring

The program logs cache hit rates after each LLM call for Debate debaters and rebuttal agents:

```
Cache [Bull]: hit=1950 miss=50 input=2000 rate=97.5%
Cache [Hold]: hit=0 miss=2050 input=2050 rate=0.0%
```

- `hit=0` → This request is a "cold start" (warmup phase), expected
- `hit` near input → Good cache hits
- Bull/Bear consistently hit=0 across multiple cycles → check if `CACHE_WARMUP_DELAY` is too small

> **Note**: Cache logging is only available for non-structured-output LLM calls (Debate debaters, rebuttal agents). Single mode and Judge (using `json_mode` structured output) cache data is not collected — but Single mode has only 1 call per cycle with no intra-cycle sharing need; Judge's system prompt auto-hits via common prefix detection from cycle 3 onward.

### Intra-Cycle Cache Benefits

| Mode | Calls/Cycle | Cache Hits | Effective Input Tokens | Cache Discount Tokens |
|------|-----------|---------|----------------------|----------------------|
| Single | 1 | 0 (no sharing target) | ~4000 | 0 |
| Debate (no rebuttal) | 4 | 2/4 (Bull + Bear hit) | ~7150 | ~3900 (55%) |
| Debate (with rebuttal) | 7 | 4/7 (Bull + Bear ×2) | ~10650 | ~7800 (73%) |

> **Intra-cycle sharing** (same market data used by multiple agents) is the main benefit source. **Cross-cycle sharing** (system prompt common prefix) has smaller benefits — since market data changes each time, only fixed instruction portions are cross-cycle cacheable.

## Push Notifications (WeChat / Feishu)

Publishes messages to Redis Pub/Sub via larky's `UnifiedClient.notify()`, delivered by larky's `UnifiedService` (standalone process `python -m larky`) to each platform (WeChat/Feishu/QQ). Multiple programs share the same infrastructure — no individual bot login management needed.

### Architecture

```
kimi_quant ──UnifiedClient──▶
                            │
cryptoguard ──UnifiedClient──▶── bot:outgoing ──▶ UnifiedService ──▶ WeChat/Feishu/QQ
                            │     (Redis Pub/Sub)     (larky standalone)
other apps  ──UnifiedClient──▶
```

### Dependencies

- `larky` (editable install, already in `pyproject.toml`)
- `redis` (transitive dependency of larky, explicitly declared to ensure installation)
- `UnifiedService` running independently (`python -m larky`), shared by all programs

### Push Events

| Event | Message | Source |
|-------|---------|--------|
| 🚀 Startup | Mode, model, interval | Main loop |
| ❌ Startup Failure | Error details | Main loop |
| 📈 Position Open | Direction, size, entry, SL/TP, confidence, balance | Main loop |
| 🟢/🔴 Position Close | P&L amount, percentage, close reason, balance | Main loop |
| 🛡️ Risk Rejected | Rejection reason (confidence insufficient/circuit breaker/SL too tight etc.) | Main loop |
| 🔄 Risk Correction | Requesting LLM to fix rejected signal + truncated rejection reason | Main loop |
| ✅ Correction Passed | Corrected signal passed risk (shows old vs new action) | Main loop |
| ❌ Correction Failed | Reason for failure after correction, giving up this cycle | Main loop |
| ⚠️ Circuit Breaker | Consecutive loss count, cooldown cycles, cumulative P&L | Main loop |
| ✅ Order Filled | Entry fill / Take Profit / Stop Loss triggered | **OrderMonitor (RT)** |
| ⏳ Partial Fill | Fill progress percentage, price | **OrderMonitor (RT)** |
| ❌ Order Canceled/Rejected | OID, reason | **OrderMonitor (RT)** |
| 💀 Position Liquidated | Liquidation price, quantity | **OrderMonitor (RT)** |
| ⚠️ Anomaly | First error + every 10 cycles (rate-limited) | Main loop |
| ⏹️ Stopped | Total cycles, trades, win rate, P&L | Main loop |

### Auto-Detection

```
larky importable → UnifiedClient sends (requires UnifiedService running)
Feishu APP_ID present → Feishu push (fallback)
Neither              → Silent operation

Redis config (optional, default localhost:6379):
  REDIS_HOST=localhost
  REDIS_PORT=6379
  REDIS_DB=0
```

The program auto-pings Redis on startup; if connected, routes through the notification channel. Send failures auto-reconnect — won't permanently go silent due to Redis temporary restart. `priority="high"` ensures offline messages aren't lost (Redis queue buffer, replay on recovery).

## Real-Time Order Monitoring

After the LLM places orders, they may fill at any time between cycles (especially market orders filling in seconds, SL/TP potentially triggering hours later). If relying solely on per-cycle (default 600s) on-chain sync, you might not know about fills for 10+ minutes.

**OrderMonitor + FlashReporter** solves this: real-time order state subscription via Hyperliquid WebSocket, millisecond-level sync to position tracker, and cheap Flash model for Chinese push notifications.

### Architecture

```
Hyperliquid WebSocket
  ├── orderUpdates (order state changes)
  └── userFills     (fill details)
         │
         ▼
   OrderMonitor (background thread)
    ├── Parse events → OrderEvent
    ├── apply_ws_event() → PositionTracker (instant state sync)
    └── Enqueue → queue.Queue (thread-safe)
         │
         ▼
   FlashReporter (background thread)
    ├── Consume events
    ├── Flash LLM generates Chinese notifications (deepseek-v4-flash)
    │    Falls back to deterministic formatting on failure
    └── Notifier → WeChat/Feishu push
```

### Coordination with Main Loop

```
Main Loop (every 600s)                Monitor (real-time)
     │                                    │
     ├── LLM decision                     │
     ├── risk.validate()                  ├── WS: order filled!
     ├── executor.execute() place order   │   ├── tracker.apply_ws_event()
     │                                    │   │   resting → active
     │                                    │   └── FlashReporter → push notification
     ├── sleep(600s)                      │
     │                                    ├── WS: SL triggered!
     │                                    │   ├── tracker.clear()
     │                                    │   └── FlashReporter → push notification
     ▼                                    │
  Next cycle                              ▼
  tracker state already latest → LLM sees real-time state
```

### Tracked Event Types

| WS Event | Tracker State Change | Push Notification (Flash LLM generated) |
|----------|---------------------|----------------------------------------|
| Entry order filled | `resting → active` | `✅ Order filled #12345 LONG 0.0100 BTC @ $67200` |
| Partial fill | Logged, awaiting full fill | `⏳ Partial fill 40% (0.004/0.01 BTC) @ $67150` |
| Stop Loss triggered | `clear tracker` | `🛑 Stop Loss triggered #12346 @ $66800` |
| Take Profit triggered | `clear tracker` | `🎯 Take Profit triggered #12347 @ $69100` |
| Order canceled | Clear corresponding oid | `❌ Order canceled #12348` |
| Order rejected | Clear tracker | `🚫 Order rejected #12349` |
| Position liquidated | Clear tracker | `💀 Position liquidated @ $70000` |

### Thread Safety

`PositionTracker` has built-in `threading.Lock`, mutually exclusive concurrent access between main loop and Monitor background thread:

- **Main thread**: `sync_with_chain()`, `execute()`, `to_summary()` etc. read/write
- **Monitor thread**: WebSocket callbacks call `apply_ws_event()` write

All public mutation methods (`clear()`, `update_from_open()`, `confirm_active()`, `tick_resting()`, `apply_ws_event()`) hold the lock.

### Flash Model Fallback Protection

If the Flash LLM API call fails (network timeout, insufficient balance, invalid key), `FlashReporter` immediately switches to **deterministic formatting** mode — generating notification text from fixed templates. Notifications are not lost, only losing natural language flexibility.

**Example fallback notification**:
```
✅ Order filled #12345
LONG 0.0100 BTC @ $67200.0
```

### Configuration

```bash
# .env
MONITOR_ENABLED=true                     # Enable real-time monitoring (default on)
MONITOR_FLASH_MODEL=deepseek-v4-flash    # Flash model (cheap & fast)
# MONITOR_FLASH_API_KEY=                 # Leave blank to reuse DEEPSEEK_API_KEY
# MONITOR_FLASH_BASE_URL=                # Leave blank to reuse DEEPSEEK_BASE_URL
```

| Config | Default | Description |
|--------|---------|-------------|
| `MONITOR_ENABLED` | `true` | Enable/disable real-time monitoring (auto-disabled in dry-run) |
| `MONITOR_FLASH_MODEL` | `deepseek-v4-flash` | Flash model name. ~$0.02/1M tokens, <1s latency |
| `MONITOR_FLASH_API_KEY` | Same as `DEEPSEEK_API_KEY` | Flash model API Key. Leave blank to reuse DeepSeek key |
| `MONITOR_FLASH_BASE_URL` | Same as `DEEPSEEK_BASE_URL` | Flash model API endpoint |

> **Extremely Low Cost**: One notification is ~100-200 input tokens + 30-50 output tokens. DeepSeek V4 Flash pricing ~$0.02/1M input. Even 100 notifications/day, monthly cost under **$0.01**.

### Startup Logs

```
OrderMonitor started (address=0xAeFB...)
WebSocket subscribed: orderUpdates(#1) + userFills(#2)
Order monitor active (flash_model=deepseek-v4-flash, llm=enabled)
FlashReporter LLM ready: deepseek-v4-flash
```

Runtime WS event sync logs:
```
WS sync: entry #12345 filled @ 67200.0 (state: resting→active)
WS → tracker synced: entry_filled oid=12345
FlashReporter sent: ✅ Order filled #12345...
```

## Quick Start

### Requirements

- Python >= 3.13
- Linux (WSL2 / Arch / Ubuntu) or macOS
- [uv](https://docs.astral.sh/uv/) package manager
- Kimi (Moonshot) API Key → [platform.moonshot.cn](https://platform.moonshot.cn)

### Installation

```bash
git clone <repo-url> && cd kimi_quant
uv sync
cp .env.example .env
```

### Minimal Configuration

Edit `.env`, only one API Key is required at minimum:

```bash
MOONSHOT_API_KEY=sk-your-key-here    # Required (at least one of Kimi or DeepSeek)
# DEEPSEEK_API_KEY=sk-...           # Optional, auto-used as fallback if set
```

All other configs keep defaults (dry-run mode, no real funds).

### Keeping .env in Sync with .env.example

`.env.example` gains new config items with version iterations. After `git pull`, quickly see what's new (auto-filters key lines):

```bash
diff <(grep -vE '^(MOONSHOT|DEEPSEEK|HYPERLIQUID_PRIVATE)_' .env) .env.example | grep '^>'
```

Output shows lines that are in `.env.example` but missing from your `.env`. Copy over what you need — most new configs have defaults, nothing breaks if you don't.

You can also add a shell alias for repeated use:

```bash
alias env-diff='diff <(grep -vE "^(MOONSHOT|DEEPSEEK|HYPERLIQUID_PRIVATE)_" .env) .env.example | grep "^>"'
```

### Running

```bash
# Single analysis (verify environment)
uv run kimi-quant --once

# Launch! One command, runs forever, no cron
uv run kimi-quant
#   ↓ Program internally loops:
#     Fetch market → Ask LLM → Risk → Execute → LLM decides sleep → wake repeat

# Another terminal, check status anytime
uv run kimi-quant --status     # Account balance, positions, orders, market
uv run kimi-quant --stats      # P&L
uv run kimi-quant --history    # Debate records
```

**You can close your screen after launching.** No crontab, no systemd timer, no repeated manual runs. The program is a `while True` loop internally; the LLM decides when to wake next.

## Phased Testing Guide

**Do NOT start with real money!** Follow these three phases progressively:

### 🧪 Phase 1: Dry-Run (Zero Cost, 1-3 Days)

**Goal**: Verify Kimi API connectivity, LLM decision quality, system stability.

```bash
# .env config
DRY_RUN=true                      # Simulation mode
STRATEGY_MODE=single              # Start with single mode
TRADING_INTERVAL=300              # 5-minute cycle
```

```bash
# Run once to see output
uv run kimi-quant --once

# Run continuously (recommend tmux/screen for half to full day)
uv run kimi-quant --interval 300

# Monitor simulated P&L in another terminal
watch -n 300 'uv run kimi-quant --stats'
```

**Checklist**:
- [ ] Each cycle outputs market data normally (price, spread, funding rate)
- [ ] LLM returns valid signals (action, confidence, reasoning)
- [ ] Risk checks work correctly (rejected/executed with valid reasons)
- [ ] `--stats` shows simulated trade records
- [ ] No abnormal crashes or API errors

### 🧪 Phase 2: Hyperliquid Testnet (Zero Cost, 1-3 Days)

**Goal**: Verify on-chain interactions (place/cancel orders, SL triggers) work in real environment.

#### 2.1 Get Wallet Private Key

Hyperliquid uses Ethereum-compatible addresses. You can use any EVM wallet (OKX Web3, MetaMask, Rabby, etc.):

```
OKX App → Wallet → Wallet Management → Export Private Key → Copy
MetaMask → Account Details → Export Private Key → Copy
```

You'll get a 64-character hex string starting with `0x`.

#### 2.2 Check Your Hyperliquid Address

```bash
uv run python -c "
from eth_account import Account
acct = Account.from_key('0xYourPrivateKey')
print('Hyperliquid Address:', acct.address)
"
```

#### 2.3 Get Testnet Funds

1. Visit [Hyperliquid Testnet](https://app.hyperliquid-testnet.xyz/trade)
2. Click "Deposit" in top right, use Arbitrum Sepolia testnet for test USDC
3. Or use the official [Faucet](https://hyperliquid.gitbook.io/hyperliquid-docs/onboarding/testnet-faucet)

#### 2.4 Configure Testnet

```bash
# Backup dry-run config
cp .env .env.dry-run

# Edit .env
MOONSHOT_API_KEY=sk-your-key-here
HYPERLIQUID_PRIVATE_KEY=0xYourPrivateKey
HYPERLIQUID_TESTNET=true            # Testnet
DRY_RUN=false                       # Enable live execution (testnet)
TRADING_INTERVAL=300
MAX_POSITION_SIZE=0.001             # Tiny position 0.001 BTC
MAX_LEVERAGE=1                      # 1x no leverage
MIN_CONFIDENCE=0.75                 # Higher confidence threshold (more conservative on testnet)
```

#### 2.5 Run

```bash
# Single run: confirm connectivity, order placement, results
uv run kimi-quant --once

# Continuous: observe several complete cycles
uv run kimi-quant --interval 120     # 2 minutes, accelerated testing

# Monitoring
uv run kimi-quant --status               # Real-time account status
uv run kimi-quant --stats                # Trade P&L
```

**Checklist**:
- [ ] Startup log shows `TradeExecutor initialized (address=0x... testnet=True)`
- [ ] Startup log shows `OrderMonitor started` + `WebSocket subscribed` (real-time monitoring active)
- [ ] Position tracking correct (`Position: [ACTIVE] LONG 0.0010 BTC @ $...`)
- [ ] SL/TP orders created normally
- [ ] CLOSE signal closes positions correctly
- [ ] Circuit breaker mechanism triggers properly
- [ ] `--status` shows on-chain-consistent position/order data
- [ ] `--stats` shows P&L records
- [ ] Check [Hyperliquid Testnet](https://app.hyperliquid-testnet.xyz/trade) to confirm positions/orders visible

### 🚀 Phase 3: Mainnet Live (Minimal Position Size)

Only proceed after **everything works on testnet**.

#### 3.1 Deposit to Hyperliquid

Fund path (choose one):

```
Path A (Recommended, via Arbitrum):
  Exchange buy USDC → Withdraw to Arbitrum chain address
  → app.hyperliquid.xyz/bridge → Bridge to Hyperliquid L1

Path B (Direct deposit):
  Exchange withdraw USDC to Arbitrum chain
  → app.hyperliquid.xyz → Deposit
```

**First deposit recommended $100-200 USDC. Don't put in too much at once.**

#### 3.2 Configure Mainnet

```bash
# Backup testnet config
cp .env .env.testnet

# .env
MOONSHOT_API_KEY=sk-your-key-here
HYPERLIQUID_PRIVATE_KEY=0xYourPrivateKey
HYPERLIQUID_TESTNET=false           # Mainnet!
HYPERLIQUID_BASE_URL=https://api.hyperliquid.xyz
DRY_RUN=false

# Extremely conservative launch parameters
MAX_POSITION_SIZE=0.001             # 0.001 BTC ≈ $87
MAX_LEVERAGE=1                      # 1x no leverage, no liquidation risk
MIN_CONFIDENCE=0.75                 # Only open with high confidence
TRADING_INTERVAL=300                # 5 minutes
```

#### 3.3 Run Under Human Supervision

```bash
# Run in foreground, watch 5-10 cycles
uv run kimi-quant --interval 300

# Another terminal for real-time monitoring
uv run kimi-quant --status                            # Account + positions
watch -n 60 'uv run kimi-quant --stats'               # P&L refresh
```

**First 24 hours**:
- Stay at the computer or check regularly
- Check `--stats` after each trade
- Verify [Hyperliquid App](https://app.hyperliquid.xyz/trade) position state matches program

#### 3.4 Gradual Scaling

Once system is confirmed stable and profitable, change one parameter at a time, observing at least 1-2 days:

```bash
# Evolution path (progressive, don't skip steps)
MAX_POSITION_SIZE=0.002   # → 0.005 → 0.01
MAX_LEVERAGE=2            # → 3
MIN_CONFIDENCE=0.70       # → 0.65
STRATEGY_MODE=debate      # Switch to debate mode (note: 4x LLM calls)
```

## Production Deployment

### Background (tmux)

```bash
# Create session
tmux new -s kimi

# Start trading bot
uv run kimi-quant --mode single --interval 300

# Detach (program continues running)
Ctrl+B, D

# Reattach
tmux attach -t kimi
```

### Background (systemd)

```bash
# Create service file
sudo tee /etc/systemd/system/kimi-quant.service << 'EOF'
[Unit]
Description=Kimi Quant Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=marvin
WorkingDirectory=/home/marvin/playground/kimi_quant
EnvironmentFile=/home/marvin/playground/kimi_quant/.env
ExecStart=/home/marvin/playground/kimi_quant/.venv/bin/kimi-quant --interval 300
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now kimi-quant

# View logs
journalctl -u kimi-quant -f
```

### Health Monitoring

```bash
# Set up periodic alerts (crontab)
*/10 * * * * cd /path/to/kimi_quant && uv run kimi-quant --stats 2>&1 | grep -q "Net P&L.*-[5-9][0-9]" && notify-send "Kimi Quant: Large drawdown warning"

# Check if process is alive
pgrep -f kimi-quant || echo "WARNING: Bot is not running!"
```

## Configuration Reference

### Complete Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **Required** | | |
| `MOONSHOT_API_KEY` | — | Kimi API Key ([Get it here](https://platform.moonshot.cn)) |
| **Kimi (Moonshot)** | | |
| `MOONSHOT_API_KEY` | — | Kimi API Key |
| `MOONSHOT_BASE_URL` | `https://api.moonshot.cn/v1` | API endpoint |
| `KIMI_MODEL` | `kimi-k3` | Model name |
| **DeepSeek (Optional Fallback)** | | Auto-fallback |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key (leave blank for Kimi only) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API endpoint |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Model name |
| **LLM Parameters** | | |
| `PRIMARY_LLM` | `kimi` | Primary model: `kimi` or `deepseek` |
| `REASONING_EFFORT` | `max` | Reasoning effort: `max`/`high`/`medium`/`low`/`minimal`/`off` |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature (0-2). **Note**: Kimi K3 only supports 1.0, auto-overridden |
| `LLM_MAX_TOKENS` | `2048` | Max output tokens (doesn't affect 1M context input) |
| `JUDGE_TEMPERATURE` | `0.05` | Debate mode Judge temperature |
| `JUDGE_PRIMARY_LLM` | (same as `PRIMARY_LLM`) | Judge-specific primary model: `kimi` or `deepseek`. Leave blank to match debaters. Recommend `kimi` (strong reasoning for rulings) |
| `DEBATE_REBUTTAL_ENABLED` | `false` | Enable rebuttal round: debaters rebut each other before Judge ruling (+3 LLM calls/cycle) |
| `CACHE_WARMUP_DELAY` | `2.0` | Debate mode cache write delay in seconds. Increase to ensure Bull/Bear cache hits, set to 0 to disable. Only affects timing, not decision quality |
| **Hyperliquid** | | |
| `HYPERLIQUID_PRIVATE_KEY` | — | Wallet private key (required for live) |
| `HYPERLIQUID_TESTNET` | `true` | `true`=testnet, `false`=mainnet |
| `HYPERLIQUID_BASE_URL` | `https://api.hyperliquid.xyz` | Mainnet API |
| **Trading Parameters** | | |
| `TRADING_PAIR` | `BTC` | Trading pair |
| `MAX_POSITION_SIZE` | `0.01` | Max position size (**Unit: BTC**, not USD) |
| `MIN_CONFIDENCE` | `0.7` | Minimum confidence threshold |
| `MAX_LEVERAGE` | `3` | Maximum leverage |
| **Strategy** | | |
| `STRATEGY_MODE` | `single` | `single` or `debate` |
| `TRADING_INTERVAL` | `600` | Default interval (seconds). LLM can dynamically override via `next_interval` |
| `MIN_INTERVAL` | `300` | Hard lower bound for LLM-suggested interval (seconds, default 5 min) |
| `MAX_INTERVAL` | `10800` | Hard upper bound for LLM-suggested interval (seconds, default 3 hours) |
| `DRY_RUN` | `true` | Simulation mode toggle |
| **Risk Correction** | | |
| `RISK_CORRECTION_ENABLED` | `true` | Give LLM one correction chance after risk rejection (set `false` to disable) |
| **Order Monitoring (RT WebSocket + Flash LLM)** | | |
| `MONITOR_ENABLED` | `true` | Enable real-time order state monitoring |
| `MONITOR_FLASH_MODEL` | `deepseek-v4-flash` | Flash model (lightweight reporting) |
| `MONITOR_FLASH_BASE_URL` | Same as `DEEPSEEK_BASE_URL` | Flash model API endpoint |
| `MONITOR_FLASH_API_KEY` | Same as `DEEPSEEK_API_KEY` | Flash model API Key (leave blank to reuse) |
| **Logging** | | |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Position Size Reference

`MAX_POSITION_SIZE` unit is **BTC quantity**, not USD:

| Setting | BTC Quantity | ~USD (BTC=$87k) | Suitable For |
|---------|-------------|-----------------|--------------|
| `0.001` | 0.001 BTC | ~$87 | First live test |
| `0.005` | 0.005 BTC | ~$435 | Cautious operation |
| `0.01` | 0.01 BTC | ~$870 | Regular operation |
| `0.05` | 0.05 BTC | ~$4,350 | Requires larger capital |

## CLI Commands

```bash
# Core commands
uv run kimi-quant                              # Launch continuous trading loop
uv run kimi-quant --once                       # Single analysis (returns JSON result)
uv run kimi-quant --once --mode debate         # Single debate mode analysis

# Parameters
uv run kimi-quant --interval 120               # Custom interval (seconds)
uv run kimi-quant --mode single                # Specify strategy mode
uv run kimi-quant --mode debate --interval 300 # Combined parameters

# Real-time status queries (safe to run concurrently with running bot)
uv run kimi-quant --status                     # Account balance, positions, orders, market
uv run kimi-quant --stats                      # View P&L stats (live + simulated)
uv run kimi-quant --history                    # View debate history
```

### `--status` Real-Time Account State

Directly queries Hyperliquid on-chain data, outputs four dimensions:

```
uv run kimi-quant --status

════════════════════════════════════════════════════════════
  Kimi Quant — Account Status
  Address: 0xAeFB...9eC3
  Network: Mainnet | 2026-07-18 10:28:12 UTC
════════════════════════════════════════════════════════════

  💰 Account Balance
  Total Value:     $5,234.56
  Available:       $4,892.10
  Margin Used:     $342.46
  Margin Ratio:    6.5%

  📊 Position
  Side:            LONG
  Size:            0.0100 BTC
  Entry Price:     $67,200.00
  Mark Price:      $67,450.00
  Unrealized PnL:  +$25.00 (+0.37%)
  Leverage:        3x
  Notional:        $672.00

  📝 Open Orders (2)
  OID          Type                 Side   Size (BTC)   Price          Status
  ──────────────────────────────────────────────────────────────────────────
  12345678     Stop Loss            SELL   0.0100       $66,800.00     Active
  12345679     Take Profit          SELL   0.0100       $69,100.00     Active

  📈 Market
  BTC Mid Price:  $67,450.00
  24h Change:     +1.36%
  Funding Rate:   0.0013%
  Open Interest:  $39,038
════════════════════════════════════════════════════════════
```

| Section | Data Source | Description |
|---------|------------|-------------|
| 💰 Balance | `user_state.marginSummary` | Total assets, available, margin used |
| 📊 Position | `user_state.assetPositions` | Direction, size, entry price, mark price, uPNL, leverage |
| 📝 Orders | `open_orders` | All open orders (limit, SL, TP) with OID, type, price |
| 📈 Market | `all_mids` + `meta_and_asset_ctxs` | Mid price, 24h change, funding rate, open interest |

All data queried directly from chain — no local tracker — objective ground truth.

> **Order Type Inference**: Hyperliquid's `openOrders` API doesn't return an `orderType` field. `--status` infers contextually: comparing order price vs position entry/market price to determine Limit Entry, Stop Loss, or Take Profit.

Refresh continuously in terminal:

```bash
watch -n 5 uv run kimi-quant --status
```

## Account Management

Hyperliquid account management tools covering deposits, transfers, and account type switching.

```bash
# Check Arbitrum chain USDC/ETH balance
uv run kimi-quant --arb-balance

# Set account type (manual confirmation, or add --force to skip)
uv run kimi-quant --set-account-type manual

# Spot → Perp account transfer (manual confirmation)
uv run kimi-quant --spot-to-perp 16

# Deposit from Arbitrum to Hyperliquid (⚠️ requires manual YES confirmation, web3 tooling not fully verified)
uv run kimi-quant --deposit 100
```

### Account Types

Hyperliquid supports three account modes:

| Mode | CLI Parameter | Characteristics | Recommended |
|------|--------------|-----------------|:--:|
| Manual | `--set-account-type manual` | spot/perp independent balances, SDK fully compatible | ✅ Recommended |
| Unified | `--set-account-type unified` | Unified balance management, but SDK doesn't support spot→perp transfers | |
| Portfolio Margin | `--set-account-type portfolio` | Cross-asset shared margin, high complexity | |

After switching to Manual, use `--spot-to-perp` for programmatic fund transfers.

### Full Deposit Flow

```
Exchange buy USDC → Withdraw to Arbitrum chain (your 0x... address)
  → uv run kimi-quant --arb-balance          # Confirm arrival
  → uv run kimi-quant --set-account-type manual  # Switch to Manual mode
  → uv run kimi-quant --spot-to-perp 100     # Transfer to perp account
  → uv run kimi-quant                        # Start trading
```

> **Note**: Confirm account type before first deposit. Unified Account requires switching via web or `--set-account-type manual` before SDK transfers work. Hyperliquid bridge minimum 5 USDC; amounts below this will be lost. `--deposit` tool not fully verified — recommend using official web UI for Arbitrum→Hyperliquid bridging.

## TradingSignal

### LLM Output Format (Recommended: multi-operation `actions`)

```json
{
  "actions": ["CLOSE", "SHORT"],
  "confidence": 0.85,
  "reasoning": "1h trend reversal confirmed, close long then open short...",
  "size": 0.005,
  "entry_price": 87000.0,
  "stop_loss": 88000.0,
  "take_profit": 85000.0,
  "modify_sl_to": null,
  "modify_tp_to": null,
  "key_factors": ["1h breakdown", "ask wall stacking", "funding positive"],
  "next_interval": 120
}
```

> **`actions` takes priority**: LLM should output the `actions` array. Single operation: `["LONG"]`, position flip: `["CLOSE", "SHORT"]`, adjust SL/TP: `["MODIFY_SL", "MODIFY_TP"]`. Legacy `action` string format still supported for backward compatibility.
>
> **`entry_price` is advisory only**: All entries execute as market orders (Ioc). `entry_price` is used only for risk calculations (SL distance, risk amount). Recommend setting to `null` to use current market price, or filling in estimated fill price for more precise risk calculations.

> `next_interval`: LLM-suggested next wake seconds (60-10800), null uses default. In the example, 120s means tight monitoring after opening.

### Action Reference

| Action | Trigger | Order Operation |
|--------|---------|----------------|
| `LONG` | Bullish signal | Open long + SL + TP (bulk_orders atomic) |
| `SHORT` | Bearish signal | Open short + SL + TP (bulk_orders atomic) |
| `CLOSE` | Close signal | Market close position |
| `HOLD` | Wait/uncertain | No operation |
| `MODIFY_SL` | Move stop loss | Move SL to new price (breakeven/trailing) |
| `MODIFY_TP` | Move take profit | Move TP to new price (adjust target) |

### Multi-Operation Combos (`actions` array)

A single cycle can execute an ordered sequence of operations. The executor runs sequentially, failing fast:

| Scenario | `actions` | Description |
|----------|-----------|-------------|
| Flip position | `["CLOSE", "SHORT"]` | Close long first, then open short |
| Adjust SL/TP | `["MODIFY_SL", "MODIFY_TP"]` | Move SL and TP simultaneously |
| Single operation | `["LONG"]` | Equivalent to legacy `action="LONG"` |

**Field Notes**:
- `actions`: **Recommended**. Ordered operation list, executed sequentially, stops on failure
- `action`: Legacy format (still supported), used when `actions` is null
- `entry_price`: **Advisory only** (for risk calculation) — all entries execute as market orders (Ioc), no limit. Set to `null` for current market price estimate, or fill in expected fill price for more accurate risk calculations
- `stop_loss`: **Mandatory field**, must be provided for LONG/SHORT, ≥ 0.5% from entry
- `size`: `null` auto-uses `MAX_POSITION_SIZE`
- `modify_sl_to`: Only used with MODIFY_SL, specifies new stop loss price
- `modify_tp_to`: Only used with MODIFY_TP, specifies new take profit price

### Cycle Status

Status output at each cycle end:

| Status | Meaning |
|--------|---------|
| `executed` | Order/close/modify SL succeeded |
| `hold` | LLM decided to wait, intentional no-op |
| `rejected` | Risk blocked (confidence insufficient/margin insufficient/SL too tight etc.) |
| `failed` | Execution exception (network error, API timeout etc.) |
| `skipped` | LLM returned no valid signal |

Startup logs print current risk parameters for easy config verification:

```
Risk: min_confidence=0.65 | max_position=0.0010 BTC | max_leverage=3x
```

## Risk Control Rules

### Proactive Risk Context Injection

**Problem**: In traditional architecture, risk rules are checked **after** the LLM outputs a decision. If the LLM proposes a doomed operation (e.g., opening during circuit breaker, insufficient margin, risk amount exceeded), that LLM call is wasted — spent tokens, incurred latency, missed the time window.

**Solution**: Before each LLM decision, dynamically inject current risk constraints into the prompt. The LLM already knows when generating TradingSignal:

```
# Risk Constraints (this cycle)

⚠️  CIRCUIT BREAKER ACTIVE — 4 consecutive losses, 3 cycles remaining
  → NEW POSITIONS BLOCKED. ALLOWED: CLOSE, MODIFY_SL, MODIFY_TP, HOLD.

## Hard Limits
- Min confidence: 0.7 | Max position: 0.001 BTC | Leverage: 3x
- SL min distance: 0.5% | SL REQUIRED for directional trades

## Margin Budget (from your account)
- Available: $5000.00 → Max notional: $14250.00 → Max size: 0.1583 BTC

## Risk Budget (per trade)
- Max risk/trade: $100.00 (2% of balance)
- Risk = |entry - SL| × size — calculate BEFORE proposing
- If risk > $100.00 → HARD REJECT

## Direction Constraints
- You HOLD a LONG position → LONG rejected, CLOSE/SHORT OK
```

**Effect**:

| Before | After |
|--------|-------|
| LLM unaware of circuit breaker, proposes LONG → rejected | LLM sees circuit breaker active, doesn't propose new positions |
| LLM unaware of margin formula, size too large → rejected | LLM calculates max notional itself, size within budget |
| LLM unaware of 2% risk cap, SL too wide → rejected | LLM computes `|entry-SL|×size`, ensures ≤ 2% |
| LLM unaware of direction constraints, duplicate open → rejected | LLM sees current position direction, avoids invalid operations |

**Implementation**: `RiskManager.get_risk_context()` dynamically generates constraint text based on current account balance, market price, and position direction, injected into `DataProvider.build_llm_prompt()`. All dynamic values (margin budget, risk budget, max position) are pre-calculated with actual numbers — the LLM doesn't need to derive formulas.

### Multi-Layer Protection

| Layer | Check | Rule |
|-------|-------|------|
| 1 | **Circuit Breaker** | 4 consecutive losses → pause 6 cycles; daily drawdown > 5% freeze; no cooldown extension during cooldown |
| 2 | **Confidence** | >= `MIN_CONFIDENCE` (default 0.7) to execute directional trades |
| 3 | **Position Cap** | Not exceeding `MAX_POSITION_SIZE` |
| 4 | **Margin Requirement** | `size × price / leverage` ≤ 95% available balance; reject and suggest appropriate size if exceeded |
| 5 | **Risk Amount** | Single trade SL loss > 1% account warning, > 2% reject (`\|entry - SL\| × size`) |
| 6 | **Stop Loss Distance** | ≥ 0.5% from entry (BTC noise ~0.3%, reject below this threshold) |
| 7 | **Direction** | Same-direction position rejected; CLOSE/MODIFY_SL/MODIFY_TP require existing position; flips (CLOSE+LONG/SHORT) pass through `validate_sequence()` state simulation |
| — | **SL/TP On-Chain Verification** | Each cycle before LLM call: cross-reference tracker oid with on-chain `open_orders`; if missing, prompt warning + push notification |
| — | **Multi-Operation Fail-Fast** | Any non-HOLD operation in sequence fails → immediately stop subsequent operations, prevent half-completed states |

### Risk Rejection Feedback Correction

**Problem**: After risk rejection, the signal is simply discarded. But if it's just parameter unsuitability (SL too tight, position slightly large), the LLM can easily adjust and resubmit — no need to wait for next cycle.

**Solution**: After risk rejection, the system feeds the **rejection reason** back to the LLM that made the decision (or the Debate Judge), giving one correction opportunity.

```
LLM Decision ──▶ Risk Check ──fail──▶ Notify "🔄 Requesting LLM correction..."
                │                      │
                │                      ▼
                │              Build correction prompt:
                │              - Original signal (action/entry/SL/TP/confidence)
                │              - Rejection reason (precise error description)
                │              - "This is your only correction chance this cycle"
                │                      │
                │                      ▼
                │              LLM re-decides ──▶ Risk second check
                │                                 │
                │                    ┌────────────┴────────────┐
                │                    ▼                         ▼
                │                  Pass                      Fail
                │              "✅ Correction passed"    "❌ Correction failed"
                │              Execute corrected          Give up this cycle
                │
                ▼
               Pass ──▶ Normal execution
```

**Prompt Design Principles**:

The correction prompt gives specific options rather than a vague "don't mess up":

```
🔧 Signal Adjustment Required

[Original signal details + Rejection reason]

Please adjust your signal. Here are your options (pick the ONE that applies):

  A. SIZE TOO LARGE → reduce size to fit within the margin or risk budget.
  B. SL TOO TIGHT → widen stop loss to ≥ 0.5% from entry, or ≥ 1.5× ATR.
  C. MARGIN EXCEEDED → reduce size so that notional / leverage ≤ 95% available.
  D. HARD BLOCK (circuit breaker, daily drawdown cap, or uncorrectable) →
     output HOLD with confidence=0.0. Do NOT try to work around the block.
  E. DIRECTION ERROR (LONG while long, CLOSE with no position) →
     use the correct action for the current position state.

Keep everything else the same — only fix what was rejected.
```

**LLM Response Guide by Rejection Type**:

| Rejection Type | Correctable? | LLM Should |
|---------------|:---:|------|
| SL too tight (< 0.5%) | ✅ | Widen SL distance |
| SL too far (> 10%, warning) | ✅ | Tighten SL distance |
| Position size exceeded | ✅ | Reduce size |
| Margin insufficient | ✅ | Reduce size |
| Single trade risk > 2% | ✅ | Reduce size or adjust SL |
| Confidence insufficient | ⚠️ | Re-evaluate evidence → boost confidence or HOLD |
| Circuit breaker active | ❌ | HOLD (new positions blocked) |
| Daily drawdown exceeded | ❌ | HOLD |
| Already hold same-direction | ❌ | HOLD |
| No position to close | ❌ | HOLD |

**Cache Friendliness**: Correction call reuses original market data prefix — Single mode ~1500 token market data fully hits cache, only ~100 token correction block is new. Debate mode is better: Judge's full debate transcript (~3000+ tokens) hits cache, only the correction block is added, and Bull/Bear/Hold re-debate is skipped (saving 3 LLM calls).

**Configuration**:

```bash
# .env
RISK_CORRECTION_ENABLED=true   # Default on
RISK_CORRECTION_ENABLED=false  # Off: risk rejections directly discarded (original behavior)
```

**Relationship with Risk Context Injection**: The two mechanisms complement each other —

| Mechanism | Timing | Function |
|-----------|--------|----------|
| Risk Context Injection | **Before** LLM decision | Inform hard constraints, reduce rejection probability |
| Risk Rejection Correction | **After** risk rejection | Give one correction chance, salvage adjustable parameter errors |

Context injection reduces correction frequency; correction acts as a safety net catching what slips through.

## Prompt Enhancement System (v3.0)

v3.0 systematically injects seven categories of additional context before LLM decisions, addressing issues where Step 0 (position review) lacked original entry thesis, the LLM didn't know the current trading session, and there was no decision→outcome closed loop.

### Enhancement Context Panorama

Before each LLM decision, the following information is injected into the prompt in order:

```
┌─────────────────────────────────────────────────────────┐
│ ⏰ Timezone/Calendar Context                            │
│   UTC + Beijing time, day of week, trading session      │
│   (Asia/EU/US/Weekend), auto low-liquidity warning      │
├─────────────────────────────────────────────────────────┤
│ 📊 Market Data (prices, multi-TF K-lines, OB, funding)  │
├─────────────────────────────────────────────────────────┤
│ 📊 Volatility Regime Classification                     │
│   HIGH (4h ATR > 0.7%) → wide SL, small size            │
│   NORMAL (0.3%–0.7%) → standard params                  │
│   LOW (≤ 0.3%) → tight SL ok but watch for breakout     │
│   Auto-calculates minimum SL distance (≥ max(1.5× ATR, 0.5%)) │
├─────────────────────────────────────────────────────────┤
│ 📌 Position Memory (when holding)                       │
│   Entry thesis, holding duration, entry confidence       │
│   Current uPNL + percentage                             │
│   Max Favorable Excursion (MFE) / Max Adverse Excursion (MAE) │
│   ⚠️ Step 0 thesis validation prompt                     │
├─────────────────────────────────────────────────────────┤
│ 📋 Previous Cycle Feedback                              │
│   Last cycle's decision, confidence, reasoning           │
│   Execution result (executed/rejected/hold/failed)       │
│   "You said wait for pullback confirmation — has it?"    │
├─────────────────────────────────────────────────────────┤
│ 🛡️ Risk Constraints                                     │
│   Circuit breaker state, margin budget, risk budget,     │
│   direction limits. EV framework: R:R → breakeven rate   │
├─────────────────────────────────────────────────────────┤
│ 📚 Recent Lessons (≥3 closed trades)                    │
│   Consecutive stop-outs, per-direction win rate,         │
│   net P&L trend. Pattern: SL too tight? TP then trend?   │
├─────────────────────────────────────────────────────────┤
│ 📊 Historical Performance (P&L stats + last 5 trades)    │
├─────────────────────────────────────────────────────────┤
│ ⚠️ Hard Constraints Repeated (end of prompt)             │
│   Max position / Min confidence / SL mandatory / Leverage│
│   Circuit breaker noted "LONG/SHORT WILL BE REJECTED"    │
│   (Recency effect — last thing LLM sees before output)    │
└─────────────────────────────────────────────────────────┘
```

### Enhancement Details

#### 1. Timezone/Calendar Context

**Problem**: The LLM doesn't know "when it is" — liquidity analysis at 2 AM EST is identical to EU session.

**Solution**: Auto-inject UTC time, Beijing time, day of week, trading session classification:

```
# ⏰ Time Context
UTC: 2026-07-20 14:30 | Beijing: 22:30 | Monday
Session: EU/US overlap — HIGH liquidity, strongest moves
```

Weekends auto-append warning: `⚠️ Weekend: lower liquidity, wider spreads, higher risk of false breakouts.`

**Trading Session Classification**: Asia (0-7 UTC) → EU morning (7-12) → EU/US overlap (12-15, HIGH) → US (15-21) → US close/Asia open (21-24)

#### 2. Position Memory

**Problem**: Step 0 requires the LLM to assess "does the original entry thesis still hold?" but the LLM has no idea what the original thesis was. Each cycle only sees current position state, no context.

**Solution**: `PositionTracker` has new fields tracking entry thesis, injected each cycle:

```
# 📌 Position Memory
Holding: LONG 0.0100 BTC @ $67,200
Opened: 2h35m ago | Entry confidence: 0.82
Entry thesis: "4h breakout above $67,000 resistance with volume confirmation"
Current uPNL: +$23.50 (+0.35%)
Best: +$45.00 | Worst: -$12.00
⚠️ Step 0 — THESIS VALIDATION: Has the original entry thesis held?
```

**MAE/MFE Tracking**: Updates max floating profit/loss extremes during position lifetime, showing the LLM where price has been.

#### 3. Previous Cycle Feedback Loop

**Problem**: The LLM said HOLD waiting for confirmation last cycle, doesn't know what it said this cycle. No decision→outcome learning loop.

**Solution**: Inject each cycle:

```
# 📋 Last Cycle Feedback
Decision: HOLD (confidence=0.65) — "waiting for 15m pullback to confirm breakout"
Outcome: 15m higher low formed → pullback confirmation ✅
Status: Breakout thesis still valid, entry opportunity approaching
```

**Four Feedback Paths**:
- `executed` → "Signal executed, check position memory thesis is materializing"
- `rejected` → "Rejected by risk, don't repeat the same mistake"
- `hold` → "You chose to wait — has the inter-cycle diff shown any change?"
- `failed` → "Execution failed, check error details"

#### 4. Volatility Regime Classification

**Problem**: ATR data exists but no volatility regime determination — high vol and low vol environments need completely different SL/position strategies.

**Solution**: Auto-classify based on 4h ATR% with specific guidance:

```
# 📊 Volatility Regime
4h ATR: 0.82% | 1h ATR: 0.60%
→ Regime: HIGH
Guidance: Wider stops required (≥ 1.5× ATR). Reduce position size to 50-75%.
Minimum SL distance: 0.72%
```

**Three Tiers**: HIGH (>0.7%) → NORMAL (0.3-0.7%) → LOW (≤0.3%)

#### 5. Expected Value Framework

**Problem**: LLM is required to have `confidence > 0.7` but doesn't consider R:R — a R:R=1:0.5 trade is negative EV even at confidence=0.8.

**Solution**: Inject R:R calculation example and breakeven formula:

```
## Expected Value (EV) Check
Example: Entry=$67,200, SL=$66,500 (1.0% risk), TP=$68,500 (1.9% reward)
R:R = 1:1.9 → Breakeven win-rate = 1/(1+1.9) = 34.5%
Your confidence must EXCEED 34.5% for positive EV.
```

#### 6. Recent Lessons

**Problem**: Repeating the same mistakes (e.g., consecutive resistance-level stop-outs) — the LLM isn't learning from trade history.

**Solution**: Analyze last 10 closed trades, extract patterns:

```
# 📚 Recent Lessons
1. ⚠️ 2 recent stop-outs. SL may be too tight for current volatility.
2. 💡 Multiple TP hits — trend may be stronger than expected. Consider trailing stops.
3. 📉 Net P&L last 5 trades: -$45.20. Be more selective.
```

**Detected Patterns**: Consecutive stops → suggest wider SL; multiple TPs → suggest trailing stops; one-sided low win rate → suggest reviewing entry criteria; recent net loss → suggest more conservative approach.

#### 7. Hard Constraints Repeated at End

**Problem**: LLM attention is uneven across long contexts — mid-prompt constraints may be overlooked.

**Solution**: Repeat key constraints at prompt end (last thing LLM sees before outputting JSON):

```
# ⚠️ HARD CONSTRAINTS (repeated from above)
- Max position: 0.01 BTC | Min confidence: 0.70
- SL REQUIRED for LONG/SHORT | Min SL distance: 0.5% of entry
- Max leverage: 3x
- ⛔ CIRCUIT BREAKER ACTIVE: NEW POSITIONS BLOCKED (use HOLD/CLOSE/MODIFY only)
```

#### Debate Mode Adaptation

All enhancement context works in both Single and Debate modes. Specifically:

- **Debaters aware of account state**: Bull/Bear/Hold's shared system message includes current position and risk constraints — debaters can make position-aware arguments ("add to existing 0.01 BTC long" rather than vague "go long")
- **Judge sees full enhancement**: Debate transcript + all above context + end constraint repeat

### Circuit Breaker State Machine

```
Normal ──(4 consecutive losses)──▶ Cooldown(6 cycles) ──(expires)──▶ Normal
                            │
                            │ (losses during cooldown don't extend it)
                            │ (profits during cooldown auto-clear)
                            │ (CLOSE/MODIFY_SL/MODIFY_TP always allowed)
                            ▼
                        Cooldown continues countdown
```

### Multi-Timeframe ATR Analysis

DataProvider auto-calculates ATR (Average True Range) for each timeframe:
- 5m / 15m / 1h / 4h ATR absolute values and percentages written to LLM prompt
- LLM sets more reasonable SL distances based on ATR
- Risk system uses ATR to assist SL reasonableness judgment

### Position State Tracking

```
PositionTracker three-state model:
  none ──(order placed)──▶ resting ──(on-chain confirmed)──▶ active
       │                 │                        │
       │                 │ (timeout 3 cycles)      │ (SL/TP triggered)
       │                 ▼                        ▼
       │            cancel_resting()           clear()
       │            + cancel_pending()         + record_close()
       └────────────────────────────────────────────┘
                     (crash recovery + trade backfill)
```

### Crash Recovery

TradeExecutor auto-queries on-chain state at startup:
- Recovers existing positions (`user_state` → positions)
- Recovers open order IDs and prices (`open_orders` → SL/TP oid + trigger price)
- Recovered positions auto-backfill TradeLogger pending trades
- After crash restart, immediately manage existing positions (close, move SL), and LLM sees current SL/TP prices

### Dry-Run Simulated P&L

Simulation mode supports complete trade logging:
- All trades (including simulated) persisted to `data/trades.jsonl`
- `--stats` separately displays `[LIVE]` / `[SIM]` trade statistics
- LLM performance feedback based only on real trades, avoiding simulation data contaminating the introspection loop

## Debate Mode Deep Dive

### Agent Roles

| Agent | Role | System Prompt Key Points |
|-------|------|-------------------------|
| 🐂 Bull | Aggressive bull analyst | Find long evidence: support levels, bid walls, negative funding |
| 🐻 Bear | Skeptical bear analyst | Find short evidence: resistance levels, ask walls, positive funding |
| 😐 Hold | Cautious risk officer | Find wait reasons: conflicting signals, excessive volatility, no clear direction |
| ⚖️ Judge | Chief trading officer | Synthesize multi-TF trends, cross-verify debater claims with raw data; 1h/4h weight > 5m/15m |

### Decision Flow

1. **Phase 1 — Cache Warmup**: Hold Agent runs first, DeepSeek backend writes its input as cache units at fixed token intervals
2. **Phase 2 — Parallel Hit**: Bull + Bear run concurrently, both requests' system messages (market data) hit Phase 1's fixed-interval cache units, only ~50 token role instructions are billed. Program waits `CACHE_WARMUP_DELAY` (default 2s) ensuring cache write completion
3. **Phase 3 (Optional) — Rebuttal Round**: Bull/Bear/Hold each see the other two's arguments and rebut, pointing out logical flaws or acknowledging strong points. Rebuttal phase also uses HoldRebut warmup → BullRebut+BearRebut parallel cache strategy
4. **Judge Receives Complete Info**: Judge's prompt contains four blocks — Account Context → Raw Market Data → Debate Transcript → **Rebuttal Round** (if enabled). Can cross-verify debater-cited prices, indicators, rates, and judge who prevailed in rebuttal
5. Judge synthesizes raw data and debate arguments to rule, outputs structured TradingSignal → Risk → Execute

### Rebuttal Round (Optional)

Before Judge ruling, add a round of cross-rebuttal. Each debater sees the other two's arguments and responds. **The Judge can see "whose arguments stood up under opponent rebuttal" to judge credibility** — rather than just comparing who wrote more eloquently.

```bash
# .env — Enable rebuttal round (default off)
DEBATE_REBUTTAL_ENABLED=true
```

Cost impact: 7 LLM calls per cycle when enabled (3 debaters + 3 rebuttals + 1 Judge), ~1.75× regular Debate. Recommend confirming regular Debate strategy works before using rebuttals to improve ruling quality.

```
debate (Bull/Bear/Hold) → rebuttal (cross-rebut) → adjudicate (Judge)
```

> **Prefix Caching**: All three debaters receive identical market data (placed in system message). Phase 1 Hold runs first → DeepSeek writes cache units at fixed token intervals → Phase 2 Bull/Bear hit, input token cost reduced ~95%, total Debate input tokens saved ~55%. Rebuttal round benefits similarly — rebuttal agents' system messages are the same market data. See [DeepSeek Context Disk Caching](#deepseek-context-disk-caching).

### Judge Decision Framework

The Judge receives complete information — account context (balance, available margin, current position, open SL/TP prices), trading constraints (max position, leverage), **raw market data (prices, K-lines, order book, funding rate — identical to debaters)**, all three debate arguments, and **rebuttal records** (if enabled). Can directly cross-verify debater-cited prices and indicators during ruling, and judge who was more convincing in rebuttal.

| Market State | Judge Decision Tendency |
|-------------|------------------------|
| 1h↑ + 4h↑ + Bull strong | LONG, confidence 0.75+ |
| 1h↓ + 4h↓ + Bear strong | SHORT, confidence 0.75+ |
| 4h↑ but 1h↓ | Favor LONG (higher TF dominates), reduce size + tight SL, confidence 0.65-0.75 |
| 4h↓ but short-term bounce | Favor SHORT, small size, confidence 0.65-0.75 |
| All TFs sideways | HOLD acceptable, but watch for breakouts |
| All 3 arguments weak + no clear trend | Only default HOLD in this case |

### LangGraph Checkpointing

Each cycle's complete debate results persisted to `data/debate.jsonl`:

```
One JSON object per line:
  ├── cycle_id             ← ISO timestamp
  ├── account_summary      ← Account summary
  ├── bull_argument        ← Bull Agent argument
  ├── bear_argument        ← Bear Agent argument
  ├── hold_argument        ← Hold Agent argument
  ├── bull_rebuttal        ← Bull rebuttal (when rebuttal round enabled)
  ├── bear_rebuttal        ← Bear rebuttal (when rebuttal round enabled)
  ├── hold_rebuttal        ← Hold rebuttal (when rebuttal round enabled)
  └── final_signal_json    ← Judge ruling
```

- **Resume**: `get_latest_state()` recovers after crash restart
- **History**: `--history` prints complete debate records
- **Auto-save**: LangGraph auto-writes checkpoint after each `ainvoke()`

### Cost Note

Debate mode calls LLM 4 times per cycle (3 Debaters + 1 Judge). API cost is ~4× single mode. Recommend verifying strategy works in single mode before switching to debate.

## TradeExecutor — Full SDK Coverage

All **15 trading-related methods** of the Hyperliquid Python SDK are wrapped:

```
Open Position
  market_open              ✅  Market open (Ioc)
  market_close             ✅  Market close
  order (limit/trigger)    ✅  Limit order / SL/TP trigger order
  bulk_orders              ✅  Open+SL+TP as one atomic transaction

Cancel
  cancel                   ✅  cancel_order(oid)
  cancel_by_cloid          ✅  cancel_by_cloid(cloid)
  bulk_cancel              ✅  cancel_all_orders()
  bulk_cancel_by_cloid     ✅  cancel_by_cloids([...])

Modify
  modify_order             ✅  modify_order (generic)
  bulk_modify_orders_new   ✅  modify_orders (batch)

Convenience Wrappers
  modify_stop_loss         ✅  Move stop loss
  modify_take_profit       ✅  Move take profit

Configuration
  update_leverage          ✅  Set leverage
  update_isolated_margin   ✅  Adjust margin

Safety
  schedule_cancel          ✅  Heartbeat protection (crash auto-cancel)
```

## Project Structure

```
kimi_quant/
├── src/kimi_quant/
│   ├── __init__.py      # Package definition
│   ├── config.py        # Configuration management (env + .env) + full validation
│   ├── tls.py           # curl_cffi Firefox TLS fingerprint spoofing (shared module)
│   ├── data.py          # Market data (Hyperliquid Info API + K-line cache + ATR + reconnect)
│   ├── llm.py           # TradingSignal + dual model failover (Kimi/DeepSeek)
│   ├── debate.py        # Multi-Agent debate + rebuttal round + LangGraph Checkpointing
│   ├── risk.py          # 7-layer risk checks + circuit breaker state machine
│   ├── executor.py      # 15/15 SDK coverage + startup recovery + PositionTracker
│   ├── monitor.py       # WebSocket order monitoring + crash recovery + Flash LLM
│   ├── analytics.py     # TradeLogger — P&L analysis + LLM introspection feedback
│   ├── notify.py        # WeChat/Feishu push notifications (optional, auto-detect)
│   ├── deposit.py       # Deposit/transfer/account type management (web3)
│   └── main.py          # CLI entry point + trading loop
├── data/
│   ├── debate.jsonl     # Debate history records (JSONL, fcntl file locks)
│   └── trades.jsonl     # Trade records JSONL (fcntl file locks, concurrent read/write)
├── .env                 # Actual config (gitignored, not committed)
├── .env.example         # Config template
├── .gitignore
├── pyproject.toml
└── README.md
```

## FAQ

### Q: Can't connect to Hyperliquid API from Alibaba Cloud?

Alibaba Cloud egress gateways perform TLS fingerprint inspection on Python's default SSL library and Reset connections (curl command line works but Python throws `ConnectionResetError` or `SSLError: curl: (35) Recv failure`). This project has built-in two-layer protection:

**Layer 1 — TLS Fingerprint Spoofing (curl_cffi)**: The `tls.py` shared module auto-patches Hyperliquid SDK's HTTP client with curl_cffi at import time, impersonating Firefox 147's JA3 TLS fingerprint. `data.py` and `executor.py` share the same patch logic, avoiding duplicate maintenance. Firefox is chosen over Chrome because anti-bot services most aggressively detect Chrome fingerprints (the most commonly impersonated browser); Firefox's TLS cipher suites and extension signals differ and aren't in the primary surveillance scope.

**Layer 2 — Retry + Rate Limiting**: Alibaba Cloud not only detects fingerprints but is also sensitive to concurrent request frequency. If too many TLS handshakes are initiated simultaneously, even with correct fingerprints you'll be temporarily blocked. Code has built-in:
- **Exponential backoff retry**: Auto-retry on transient errors like `Connection reset by peer` (max 3 retries, intervals 1.5s → 3s → 6s + random jitter)
- **Concurrency limit**: Parallel API request cap reduced from 5 to 2, avoiding frequency-triggered blocks
- **Staggered submission**: Each parallel task spaced 150ms apart to avoid TLS handshake bursts

Ensure `curl_cffi` is installed before running on server:
```bash
uv sync   # Auto-installs curl_cffi
```
Startup log showing `curl_cffi=True` means it's active:
```
DataProvider initialized (testnet=False, coin=BTC, curl_cffi=True)
```
During normal operation, occasional WARNING logs saying `retrying in X.Xs` indicate temporary blocks with automatic recovery — **no manual intervention needed**. Only ERROR when all 4 retries for a single call fail.

### Q: How much do API calls cost? How to save?

**Pricing Comparison** (per 1M tokens):

| Model | Input | Output | vs Kimi |
|-------|-------|--------|---------|
| Kimi K3 | ¥20 | ¥100 | — |
| DeepSeek V3 | ¥2 | ¥8 | **90%+ cheaper** |

**Monthly Cost Estimates** (10-min interval, Single mode):

| Config | Monthly Cost |
|--------|-------------|
| Kimi K3 | ~¥330 |
| DeepSeek V3 | ~¥40 |
| Kimi + DeepSeek fallback | ~¥330 (when Kimi available) |

**Debate Mode** (4 LLM calls/cycle, with prefix cache optimization):

| Config | Monthly Cost | vs Single |
|--------|-------------|-----------|
| All DeepSeek V3 | ~¥60 | 1.5× (cache saves ~55% input tokens) |
| Weak Debater Strong Judge (🏆) | ~¥160 | 1.6× (Judge uses Kimi for quality) |
| All Kimi K3 | ~¥450 | 1.4× |

**Three Money-Saving Levers**:

| Strategy | .env Config | Effect |
|----------|------------|--------|
| DeepSeek as primary | `PRIMARY_LLM=deepseek` | Monthly ¥330→¥40 |
| Weak Debater Strong Judge | `PRIMARY_LLM=deepseek` + `JUDGE_PRIMARY_LLM=kimi` | 65% cheaper vs all-Kimi, same quality |
| Disable reasoning | `REASONING_EFFORT=off` | Output cost down another 75% |
| Longer interval | `TRADING_INTERVAL=900` | Cost down 1/3 |
| Raise min interval | `MIN_INTERVAL=600` | Prevent LLM frequent wakeups (default 300s) |

**Extreme Budget Config**: `PRIMARY_LLM=deepseek + REASONING_EFFORT=off + TRADING_INTERVAL=900`, monthly ~**¥10**.

### Q: How to save tokens in Debate mode?

All three debate agents receive identical market data (~1500 tokens). Multiple optimizations are in place:

1. **Intra-cycle prefix caching**: Market data placed in system message (shared prefix), Hold runs first to warm cache, Bull + Bear hit in parallel. The latter two agents are billed for only ~50 token role instructions, saving ~55% intra-cycle input tokens (no rebuttal) or ~73% (with rebuttal). Rebuttal round uses same HoldRebut warmup strategy. Cache writes are guaranteed by `CACHE_WARMUP_DELAY` (default 2 seconds).
2. **Weak Debater Strong Judge**: Debaters use cheap DeepSeek (directed search), Judge uses Kimi K3 (synthesis). 65% cheaper than all-Kimi, decision quality unchanged. See [Debate Mode: Independent Judge Model](#debate-mode-independent-judge-model-weak-debaters-strong-judge).
3. **Inter-cycle diff**: Zero additional API calls, pure data comparison — LLM focuses on changes rather than re-analyzing all data.
4. **Cross-cycle common prefix**: Single mode system prompt, Debate debater shared system instructions auto-hit common prefix cache from cycle 3 onward.

Overall effect: Regular Debate's 4 calls equivalent to ~2.4 Single-mode input token volume. With rebuttal, 7 calls equivalent to ~3.5 Single-mode volume.

### Q: How to switch primary/fallback models?

```bash
# Global primary model
PRIMARY_LLM=kimi      # Kimi primary, DeepSeek fallback (default)
PRIMARY_LLM=deepseek  # DeepSeek primary, Kimi fallback (budget)

# Debate mode: Judge independent primary model (leave blank = same as PRIMARY_LLM)
JUDGE_PRIMARY_LLM=kimi      # Judge uses Kimi, debaters use PRIMARY_LLM
JUDGE_PRIMARY_LLM=deepseek  # Judge uses DeepSeek, debaters use PRIMARY_LLM
```

Just configure both model API Keys. Primary fails → auto fallback. No code changes needed. See [LLM Model Configuration](#llm-model-configuration).

### Q: Set `PRIMARY_LLM=deepseek` but HTTP requests still go to Kimi API?

When DeepSeek calls fail, they silently fall back to Kimi (LangChain `with_fallbacks` mechanism). `httpx` logs only record successful requests, so all you see is `api.moonshot.cn`. Common causes:

1. **`thinking` parameter format error** — must pass via `extra_body` (fixed in v2.2+)
2. **API key expired or insufficient balance**
3. **Network unreachable** (some datacenters have firewall blocks)

Quick diagnosis:
```bash
# Test DeepSeek API directly on server
curl -s https://api.deepseek.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -d '{"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}'
```
JSON response → API working; error/timeout → check network and key.

### Q: Kimi API returns HTTP 400 Bad Request?

The vast majority of cases are **temperature parameter** issues. Kimi K3 is a reasoning model that only accepts `temperature=1`. If `.env` has `LLM_TEMPERATURE=0.1` (or any non-1 value), the Moonshot API returns:
```
"invalid temperature: only 1 is allowed for this model"
```

**Auto-fixed** (v2.3+): Program auto-forces Kimi instances' temperature to 1.0, no manual `.env` change needed. Startup log shows: `Kimi K3 requires temperature=1 (got 0.10), forcing to 1.0`.

Other possible 400 causes:
- `reasoning_effort` and `response_format` sent simultaneously (auto-handled in v2.3+)
- API key format error (check if copied completely)
- Model name typo (should be `kimi-k3`)

### Q: What if Kimi API goes down?

Configure `DEEPSEEK_API_KEY`. Kimi calls auto-fallback to DeepSeek on failure, no manual intervention. Log shows `LLM: kimi primary → fallback: deepseek`. Without it configured, only Kimi is used.

### Q: How do I know when orders fill? Is there real-time notification?

Yes. **OrderMonitor** subscribes to order state changes via Hyperliquid WebSocket in real-time. **FlashReporter** uses a cheap Flash model to generate Chinese push notifications. Configuration:

```bash
# .env
MONITOR_ENABLED=true                  # Enable (default)
MONITOR_FLASH_MODEL=deepseek-v4-flash # Flash model
```

Notification events include: entry fills, SL/TP triggers, partial fills, order cancelation/rejection, position liquidation. If Flash model is unavailable, auto-fallback to formatted text notifications.

See [Real-Time Order Monitoring](#real-time-order-monitoring).

### Q: Can the LLM see the latest order/position state when making decisions?

Yes. Before each LLM decision, the system injects five categories of information:

1. **On-chain position**: `DataProvider` queries `user_state` for position direction, size, entry price, uPNL
2. **All on-chain open orders**: `DataProvider` queries `open_orders`, listing all unfilled orders (oid, direction, quantity, price). LLM sees **all** orders — including orphan orders from previous cycles
3. **Tracker-tracked SL/TP**: `PositionTracker.to_orders_summary()` outputs bot-created SL/TP oids and prices
4. **SL/TP on-chain verification results**: Program cross-references tracker oids with on-chain `open_orders` before LLM call; if missing, injects ⚠️ warning in prompt
5. **Hard risk constraints** (🆕): Circuit breaker state, margin budget ($ amount), risk budget ($ amount), direction limits — LLM knows **what it can and cannot do now**, won't propose doomed operations

More importantly, the LLM's **System Prompt mandatorily enforces Step 0** — review existing positions and orders before analyzing markets. This ensures the LLM doesn't skip current state assessment and jump to new trade decisions.

Plus **OrderMonitor**'s WebSocket real-time sync, `PositionTracker` updates the instant orders fill — the LLM sees the latest state in the next decision cycle.

### Q: What if the previous run left unfilled orders?

All entry orders execute as market orders (Ioc), normally filling instantly. In extreme cases (price jump >2% causing Ioc unfilled), orders are auto-canceled by the exchange. The program detects no on-chain position within 3 cycles and auto-cleans the tracker and pending trade records.

### Q: Can SL/TP orders be unexpectedly canceled by the exchange? How are they protected?

Yes, with three layers of protection:

1. **Programmatic verification**: Each cycle before LLM call, `verify_tracked_orders()` cross-references tracker `sl_oid`/`tp_oid` with on-chain `open_orders`. If missing → prompt injects ⚠️ warning + push notification
2. **LLM response**: Step 0-b requires LLM to treat missing SL/TP as highest priority, outputting `MODIFY_SL`/`MODIFY_TP` for immediate restore
3. **WebSocket real-time awareness**: If SL/TP triggers (fills) or gets canceled, `OrderMonitor` syncs to tracker at millisecond level

### Q: Will the program crash abnormally?

No. Three-layer error protection ensures the program can't crash through:

```
Layer 0 (Startup): Config errors → immediate error exit (must be manually fixed)
Layer 1 (Per-cycle): LLM down/API failure/network drop → logged, next cycle continues
Layer 2 (Sleep):    Sleep interrupted → logged, next cycle continues
```

Any post-startup exception — API rate limiting, LLM timeout, network jitter, disk full — only affects the current cycle. The program keeps running, next cycle auto-retries. With systemd's `Restart=on-failure`, even OOM-killed processes auto-restart.

### Q: How to receive trade notifications?

Prerequisites:
1. Server running `python -m larky` (UnifiedService, standalone process)
2. `redis` package installed (`uv sync` handles this)

With these met, the program auto-pushes via `UnifiedClient.notify()`. No additional config needed. Startup log shows:
```
Notification: larky UnifiedClient available (via Redis Pub/Sub)
```
If Redis isn't at `localhost:6379`, set `REDIS_HOST` / `REDIS_PORT` env vars.

### Q: How does the program auto-run? Need cron?

No. `uv run kimi-quant` launches into a `while True` loop. The program auto-repeats internally, with the LLM deciding how long to sleep each iteration. One command, runs forever, stops with `Ctrl+C`.

### Q: How is the next analysis time determined?

The LLM returns `next_interval` (seconds) in each response, telling the program how long to sleep before waking. Suggests 15-30 minutes during sideways markets to save costs, 1-2 minutes near key levels for tight monitoring. Program executes automatically — you need do nothing.

### Q: How to adjust Reasoning Effort (REASONING_EFFORT)?

```bash
REASONING_EFFORT=max      # Strongest analysis (default, recommended for trading)
REASONING_EFFORT=off      # No reasoning, fastest and cheapest
```

Reasoning tokens typically account for 80% of output. Setting `off` saves ~75% output cost. Recommended `off` for daily monitoring or low-cost modes, `max` for critical trades. See [Reasoning Effort Control](#reasoning-effort-control).

### Q: I only have an OKX Web3 wallet, can I use it?

Yes. OKX Web3 wallet is fundamentally a self-custodial wallet. After exporting the private key, it works identically to MetaMask/Rabby. See [Phase 2.1](#21-get-wallet-private-key).

### Q: Why is dry-run recommended first?

Dry-run involves zero on-chain operations, only validates LLM decision logic. You can observe at zero cost:
- What signals the LLM gives under what market conditions
- Rough signal win rate (via simulated P&L)
- System stability (any crashes, API errors)

### Q: What's the difference between testnet and mainnet?

- **Testnet**: USDC is free test tokens, unlimited faucet. Used to verify order placement/cancelation/SL triggers work correctly
- **Mainnet**: Real money. Only switch after everything works on testnet

### Q: What's a good cycle interval?

| Strategy | Recommended Interval | Notes |
|----------|---------------------|-------|
| Short-term/scalping | 120-300s (2-5min) | Leverage order book changes and short-term momentum |
| Mid-term/trend | 600-900s (10-15min) | More dependent on multi-TF K-line trend analysis |
| Long-term | 1800-3600s (30-60min) | Focus only on large-TF trends like 4h |

Default 300s (5min) is a balanced choice.

### Q: Will the LLM place random orders?

There's triple protection: seven-layer risk control + context pre-injection + **rejection correction**. Even if the LLM gives unreasonable signals, the risk layer rejects them. And the LLM already sees current risk constraints before deciding (circuit breaker state, margin budget, risk budget), significantly reducing invalid operation probability. If still rejected, the system gives the LLM one correction chance (see [Risk Rejection Feedback Correction](#risk-rejection-feedback-correction)). Common rejection cases:
- Confidence < 0.7
- Margin requirement > 95% available balance
- Single trade risk > 2% account
- Stop loss too tight (< 0.5%)
- Already hold same-direction position
- Circuit breaker active

### Q: What happens when risk rejects? Can the LLM self-correct?

Yes. After risk rejection, the system feeds the **rejection reason** (e.g., "SL distance 0.50% too tight, minimum 0.50%") back to the LLM, giving one correction chance. The LLM adjusts parameters based on the reason (widen SL, reduce size, etc.) and resubmits. If the correction passes risk, execute the corrected version; if still failing or hitting a hard block (circuit breaker, drawdown exceeded), give up this cycle.

Notifications clearly label the whole process:

```
🔄 Risk correction: asking LLM to fix — Stop loss distance 0.50% is too tight...
✅ Risk correction accepted — SHORT (was SHORT)     ← Correction successful
❌ Risk correction failed: ... Giving up.            ← Correction failed
```

Can disable via `RISK_CORRECTION_ENABLED=false`. See [Risk Rejection Feedback Correction](#risk-rejection-feedback-correction).

### Q: Why does account balance show $0?

Common causes:
1. **Funds in spot account**: Hyperliquid's spot and perp accounts are separate. Use `uv run kimi-quant --spot-to-perp <amount>` to transfer.
2. **Incompatible account type**: `usd_class_transfer` is disabled under Unified Account. First switch to Manual mode with `--set-account-type manual`.
3. **Not in perp account**: Use `--arb-balance` to check Arbitrum chain balance, confirm funds are bridged to Hyperliquid.

Startup logs print `Account:` line each cycle for real-time balance monitoring.

### Q: How to switch account types?

```bash
uv run kimi-quant --set-account-type manual    # Recommended: independent spot/perp balances
uv run kimi-quant --set-account-type unified   # Unified account
uv run kimi-quant --set-account-type portfolio # Portfolio margin
```

Wait a few seconds after switching, then use `--spot-to-perp` to transfer.

### Q: How to transfer USDC from Arbitrum to Hyperliquid?

Recommend using Hyperliquid's official web Deposit page. The CLI tool `--deposit` provides a web3 method but is not fully verified — requires manual `YES` confirmation when used.

### Q: How to check status while running?

```bash
# Real-time account + positions + orders (direct on-chain query)
uv run kimi-quant --status

# Auto-refresh (every 5 seconds)
watch -n 5 uv run kimi-quant --status

# P&L stats
uv run kimi-quant --stats

# Debate history
uv run kimi-quant --history

# View latest trades
tail -5 data/trades.jsonl | python -m json.tool
```

### Q: Where is debate history stored? Can it survive restart?

Debate results persist in `data/debate.jsonl` (JSONL format, using `fcntl` file locks for concurrent read/write). `--history` command views anytime, survives restarts.

### Q: How to stop the program?

- Foreground: `Ctrl+C` (graceful shutdown, completes current cycle)
- Inside tmux: `Ctrl+C` or `tmux kill-session -t kimi`
- systemd: `sudo systemctl stop kimi-quant`

## Risk Disclaimer

⚠️ **Please read carefully before use:**

1. **Quantitative trading carries significant loss risk**. Past performance does not indicate future returns. LLM (Kimi K3) judgments may be wrong.
2. **Do not commit funds you cannot afford to lose**. First live test recommended not exceeding $200.
3. **Private key security**: `.env` file stores private keys in plaintext. Ensure the runtime environment is secure. `.env` is gitignored and won't be committed to Git.
4. **API risk**: Kimi API or Hyperliquid API may experience latency, rate limiting, or temporary unavailability. The program includes exception handling but cannot eliminate risk.
5. **Liquidation risk**: Using leverage may result in liquidation. Recommended starting with `MAX_LEVERAGE=1` (no leverage).
6. **Funding rate risk**: Holding positions incurs funding payments every 8 hours. Rates may be very high in extreme market conditions.
7. **Disclaimer**: This software is provided for learning and research purposes only. Users assume all trading profit and loss responsibility. The author bears no responsibility for any trading losses.

## License

MIT
