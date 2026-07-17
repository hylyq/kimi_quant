# Kimi Quant 🚀

**BTC 永续合约量化交易程序** — 基于 Kimi K3 大模型的智能交易决策系统。

## 策略模式

### Single-Agent (单 Agent 模式)
```
市场数据 → Kimi K3 分析 → TradingSignal → 风控 → 执行
```
1 次 LLM 调用，快速轻量。

### Multi-Agent Debate (多 Agent 辩论模式)
```
              ┌── 🐂 Bull Agent (论证做多) ──┐
市场数据 ───┼── 🐻 Bear Agent (论证做空) ──┼── ⚖️ Judge ──→ TradingSignal ──→ 风控 ──→ 执行
              └── 😐 Hold Agent (论证观望) ──┘
```
3 个 Agent 并行辩论 + 1 个裁判裁决，通过对抗验证减少单一模型偏见。

## 架构概览

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
│  DataProvider │───▶│    策略引擎      │───▶│  RiskManager │───▶│  TradeExecutor   │
│  (Hyperliquid)│    │ single / debate │    │  (风控校验)   │    │  (全订单生命周期)  │
└──────────────┘    └─────────────────┘    └──────────────┘    └──────────────────┘
      ▲                                                               │
      │                      交易循环 (每 N 秒)                         │
      └───────────────────────────────────────────────────────────────┘
```

### 数据流

1. **DataProvider** — 从 Hyperliquid 获取 BTC 市场快照（价格、订单簿、资金费率、持仓）
2. **策略引擎** — Single：Kimi K3 单次分析；Debate：三 Agent 辩论 + 裁判裁决（LangGraph）
3. **RiskManager** — 四层风控校验（置信度、仓位、方向、止损）
4. **TradeExecutor** — 执行交易并管理完整订单生命周期

### 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | Kimi K3 (Moonshot API, OpenAI 兼容) |
| LLM 编排 | LangChain + LangGraph StateGraph |
| 状态持久化 | LangGraph SqliteSaver Checkpointing |
| 交易所 | Hyperliquid (Perpetual DEX) |
| 结构化输出 | Pydantic + LangChain json_schema |
| 交易执行 | hyperliquid-python-sdk (15/15 全覆盖) |

## 快速开始

### 环境要求

- Python >= 3.13
- WSL2 / Arch Linux
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone <repo-url> && cd kimi_quant
uv sync
cp .env.example .env
# 编辑 .env 填入 MOONSHOT_API_KEY
```

### 配置

```bash
# 必填
MOONSHOT_API_KEY=sk-your-key-here

# 策略模式
STRATEGY_MODE=single          # single | debate

# 交易参数
TRADING_PAIR=BTC
MAX_POSITION_SIZE=0.01
MIN_CONFIDENCE=0.7
MAX_LEVERAGE=3

# 运行模式
DRY_RUN=true                  # true=模拟, false=实盘
TRADING_INTERVAL=300          # 交易间隔 (秒)

# 实盘需要
HYPERLIQUID_PRIVATE_KEY=...
HYPERLIQUID_TESTNET=true
```

### 运行

```bash
# 单次分析
uv run kimi-quant --once                      # single 模式
uv run kimi-quant --once --mode debate         # debate 模式

# 交易循环
uv run kimi-quant                              # 默认 single
uv run kimi-quant --mode debate --interval 300

# 查看历史辩论记录
uv run kimi-quant --history

# 查看盈亏统计（可安全地与运行中的交易程序并发使用）
uv run kimi-quant --stats
```

## TradingSignal（LLM 输出格式）

```json
{
  "action": "LONG",              // LONG | SHORT | CLOSE | HOLD | MODIFY_SL
  "confidence": 0.85,
  "reasoning": "...",
  "size": 0.005,
  "entry_price": 85000.0,        // null = 市价单
  "stop_loss": 84500.0,
  "take_profit": 86000.0,
  "modify_sl_to": 63000.0,       // MODIFY_SL 时的新止损价
  "key_factors": ["..."]
}
```

### Action 说明

| Action | 触发条件 | 订单操作 |
|--------|---------|---------|
| `LONG` | 看多信号 | 开多仓 + 止损 + 止盈（bulk_orders 原子执行） |
| `SHORT` | 看空信号 | 开空仓 + 止损 + 止盈（bulk_orders 原子执行） |
| `CLOSE` | 平仓信号 | 市价平仓 |
| `HOLD` | 观望/不确定 | 无操作 |
| `MODIFY_SL` | 移动止损 | 将止损移至新价格（保本/追踪） |

## TradeExecutor — SDK 全覆盖

Hyperliquid Python SDK 的 **15 个交易相关方法全部封装**：

```
开仓
  market_open              ✅  市价开仓（Ioc）
  market_close             ✅  市价平仓
  order (limit/trigger)    ✅  限价单 / 止损止盈触发单
  bulk_orders              ✅  开仓+SL+TP 一笔原子交易

撤单
  cancel                   ✅  cancel_order(oid)
  cancel_by_cloid          ✅  cancel_by_cloid(cloid)
  bulk_cancel              ✅  cancel_all_orders()
  bulk_cancel_by_cloid     ✅  cancel_by_cloids([...])

改单
  modify_order             ✅  modify_order (通用)
  bulk_modify_orders_new   ✅  modify_orders (批量)

便捷封装
  modify_stop_loss         ✅  移动止损
  modify_take_profit       ✅  移动止盈

配置
  update_leverage          ✅  设置杠杆
  update_isolated_margin   ✅  调整保证金

安全
  schedule_cancel          ✅  心跳保护（崩溃自动撤单）
```

## Debate 模式详解

### Agent 分工

| Agent | 角色 | 职责 |
|-------|------|------|
| 🐂 Bull | 激进多头分析师 | 寻找做多证据：支撑位、bid wall、负资金费率 |
| 🐻 Bear | 怀疑派空头分析师 | 寻找做空证据：阻力位、ask wall、正资金费率 |
| 😐 Hold | 谨慎风控官 | 寻找观望理由：信号矛盾、波动过大、无明确方向 |
| ⚖️ Judge | 首席交易官 | 权衡三方论据，做出最终决策 |

### 决策流程

1. 三个 Debater **并行**接收同一份市场数据（`asyncio.gather`）
2. 每个 Debater 从自身角色出发提供 150-250 字论证
3. Judge 收到三方论据 + 原始市场数据，综合裁决
4. Judge 输出结构化 TradingSignal → 风控 → 执行

### LangGraph Checkpointing

每个 cycle 的完整状态自动持久化到 `data/debate.db`：

```
DebateState (每个 cycle 自动保存):
  ├── cycle_id             ← ISO 时间戳
  ├── market_prompt        ← 市场快照
  ├── bull_argument        ← Bull Agent 论据
  ├── bear_argument        ← Bear Agent 论据
  ├── hold_argument        ← Hold Agent 论据
  └── final_signal_json    ← Judge 裁决
```

- **断点续传**：崩溃重启后 `get_latest_state()` 恢复
- **历史回溯**：`--history` 打印完整辩论记录
- **零额外代码**：LangGraph 在每次 `ainvoke()` 后自动写 checkpoint

## 风控规则

| 检查项 | 规则 |
|--------|------|
| 置信度 | >= `MIN_CONFIDENCE` (默认 0.7) 才执行 |
| 仓位上限 | 不超过 `MAX_POSITION_SIZE` |
| 重复交易 | 已有同向仓位时拒绝 |
| 止损 | 方向性交易必须有止损价格 |
| MODIFY_SL | 需要已持仓 + 新止损价 |

## 项目结构

```
kimi_quant/
├── src/kimi_quant/
│   ├── __init__.py
│   ├── config.py      # 配置管理（env + .env）
│   ├── data.py        # 市场数据 (Hyperliquid Info API)
│   ├── llm.py         # TradingSignal + Kimi K3 单 Agent 策略
│   ├── debate.py      # Multi-Agent 辩论 + LangGraph Checkpointing
│   ├── risk.py        # 四层风控校验
│   ├── executor.py    # 15/15 SDK 全覆盖交易执行
│   └── main.py        # CLI 入口 + 交易循环
├── data/
│   └── debate.db      # SQLite checkpoint 数据库 (自动生成)
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## 风险声明

⚠️ **免责声明**：本程序仅供学习和研究使用。量化交易存在重大亏损风险，大模型判断可能出错。请勿投入无法承受损失的资金。作者不对任何交易亏损承担责任。

## License

MIT
