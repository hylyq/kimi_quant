# Kimi Quant 🚀

**BTC 永续合约量化交易程序** — 基于 Kimi K3 大模型的智能交易决策系统。

## 策略模式

### Single-Agent (单 Agent 模式)
```
市场数据 → Kimi K3 分析 → TradingSignal → 风控 → 执行
```
快速，1 次 LLM 调用。适合高频轮询。

### Multi-Agent Debate (多 Agent 辩论模式)
```
              ┌── 🐂 Bull Agent (论证做多) ──┐
市场数据 ───┼── 🐻 Bear Agent (论证做空) ──┼── ⚖️ Judge ──→ TradingSignal ──→ 风控 ──→ 执行
              └── 😐 Hold Agent (论证观望) ──┘
```
4 次 LLM 调用（3 debate + 1 judge）。通过对抗辩论减少单一模型偏见，决策更稳健。

## 架构概览

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DataProvider │───▶│   策略引擎    │───▶│  RiskManager │───▶│TradeExecutor │
│  (Hyperliquid)│    │ single/debate│    │  (风控校验)   │    │  (下单执行)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      ▲                                                          │
      │                    交易循环 (每 N 秒)                      │
      └──────────────────────────────────────────────────────────┘
```

### 数据流

1. **DataProvider** — 从 Hyperliquid 获取 BTC 市场快照（价格、订单簿、资金费率、持仓）
2. **策略引擎** — Single：Kimi K3 单次分析；Debate：三 Agent 辩论 + 裁判裁决
3. **RiskManager** — 校验信号：置信度阈值、仓位限制、方向检查、止损要求
4. **TradeExecutor** — 执行交易（市价/限价单），同时下止损单

### 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | Kimi K3 (Moonshot API, OpenAI 兼容) |
| LLM 框架 | LangChain + LangGraph |
| Multi-Agent | LangGraph StateGraph orchestration |
| 交易所 | Hyperliquid (Perpetual DEX) |
| 数据验证 | Pydantic 结构化输出 |
| 交易执行 | hyperliquid-python-sdk |

## 快速开始

### 环境要求

- Python >= 3.13
- WSL2 / Arch Linux
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd kimi_quant

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 配置

编辑 `.env` 文件：

```bash
# 必填：Kimi API Key
MOONSHOT_API_KEY=sk-your-key-here

# 策略模式：single（单Agent）或 debate（多Agent辩论）
STRATEGY_MODE=single

# 可选：Hyperliquid 私钥（仅实盘需要）
HYPERLIQUID_PRIVATE_KEY=your_private_key_hex

# 交易参数
TRADING_PAIR=BTC          # 交易对
MAX_POSITION_SIZE=0.01    # 最大仓位 (BTC)
MIN_CONFIDENCE=0.7        # 最低置信度阈值
MAX_LEVERAGE=3            # 最大杠杆

# 运行模式
DRY_RUN=true              # true=模拟交易, false=实盘
TRADING_INTERVAL=300      # 交易间隔 (秒)
```

### 运行

```bash
# 单次分析 — single 模式
uv run kimi-quant --once

# 单次分析 — debate 模式（3 Agent 辩论）
uv run kimi-quant --once --mode debate

# 启动交易循环
uv run kimi-quant

# 以 debate 模式启动交易循环
uv run kimi-quant --mode debate --interval 300
```

## Debate 模式详解

### Agent 分工

| Agent | 角色 | 职责 |
|-------|------|------|
| 🐂 Bull | 激进多头分析师 | 寻找做多证据：支撑位、bid wall、负资金费率 |
| 🐻 Bear | 怀疑派空头分析师 | 寻找做空证据：阻力位、ask wall、正资金费率 |
| 😐 Hold | 谨慎风控官 | 寻找观望理由：信号矛盾、波动过大、无明确方向 |
| ⚖️ Judge | 首席交易官 | 权衡三方论据，做出最终决策，输出 TradingSignal |

### 决策流程

1. 三个 Debater **并行**接收同一份市场数据（通过 `asyncio.gather`）
2. 每个 Debater 从自身角色出发提供 150-250 字的论证
3. Judge 收到三方论据 + 原始市场数据，综合裁决
4. Judge 输出结构化 TradingSignal（含置信度、仓位、止损止盈）

### 为什么 Debate 更稳健？

- **减少单一偏见**：单个模型容易被某个信号误导，三方辩论暴露多空分歧
- **对抗验证**：Bear 和 Bull 互相平衡，Hold 防止 FOMO
- **可追溯**：每次决策可回溯三方原始论据，便于事后复盘

### LangGraph Checkpointing（状态持久化）

Debate 模式下每个 cycle 的完整状态自动持久化到 SQLite 数据库：

```
LangGraph StateGraph
      │
      ├── checkpointer=SqliteSaver("data/debate.db")
      │
      ├── DebateState (每个 cycle 自动保存):
      │     ├── market_prompt     ← 市场快照
      │     ├── bull_argument     ← Bull Agent 论据
      │     ├── bear_argument     ← Bear Agent 论据
      │     ├── hold_argument     ← Hold Agent 论据
      │     └── final_signal_json ← Judge 裁决
      │
      └── thread_id="btc-perpetual-trading"
            └── 同一 thread 下形成完整时间线
```

**能力**：
- **断点续传**：崩溃重启后 `get_latest_state()` 恢复上次未完成的 cycle
- **历史回溯**：`--history` 打印所有历史 debate 完整记录
- **零额外代码**：LangGraph 在每次 `ainvoke()` 后自动写 checkpoint

```bash
# 查看所有历史辩论记录
uv run kimi-quant --history
```

## LLM 输出格式

Kimi K3 的响应通过 LangChain 结构化输出解析为：

```json
{
  "action": "LONG",           // LONG | SHORT | CLOSE | HOLD
  "confidence": 0.85,         // 0.0 - 1.0
  "reasoning": "...",         // 决策理由
  "size": 0.005,              // 建议仓位 (BTC)
  "entry_price": 85000.0,     // 入场价 (null = 市价单)
  "stop_loss": 84500.0,       // 止损价
  "take_profit": 86000.0,     // 止盈价
  "key_factors": ["..."]      // 关键决策因子
}
```

## 风控规则

| 检查项 | 规则 |
|--------|------|
| 置信度 | >= 0.7 才执行 |
| 仓位上限 | 不超过 MAX_POSITION_SIZE |
| 重复交易 | 已有同向仓位时拒绝 |
| 止损 | 方向性交易必须有止损 |

## 项目结构

```
kimi_quant/
├── src/kimi_quant/
│   ├── __init__.py
│   ├── config.py      # 配置管理
│   ├── data.py        # 市场数据 (Hyperliquid)
│   ├── llm.py         # 单 Agent 策略 + 共享工具
│   ├── debate.py      # Multi-Agent 辩论 + LangGraph Checkpointing
│   ├── risk.py        # 风控管理
│   ├── executor.py    # 交易执行
│   └── main.py        # 主程序入口
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
