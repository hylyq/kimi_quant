# Kimi Quant 🚀

**BTC 永续合约量化交易程序** — 基于 Kimi K3 大模型的智能交易决策系统。

## 架构概览

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DataProvider │───▶│   KimiLLM    │───▶│  RiskManager │───▶│TradeExecutor │
│  (Hyperliquid)│    │  (Kimi K3)   │    │  (风控校验)   │    │  (下单执行)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      ▲                                                          │
      │                    交易循环 (每 N 秒)                      │
      └──────────────────────────────────────────────────────────┘
```

### 数据流

1. **DataProvider** — 从 Hyperliquid 获取 BTC 市场快照（价格、订单簿、资金费率、持仓）
2. **KimiLLM** — 将市场数据构建为结构化 Prompt，发送给 Kimi K3 获取交易信号
3. **RiskManager** — 校验信号：置信度阈值、仓位限制、方向检查、止损要求
4. **TradeExecutor** — 执行交易（市价/限价单），同时下止损单

### 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | Kimi K3 (Moonshot API, OpenAI 兼容) |
| LLM 框架 | LangChain + langchain-openai |
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
# 单次分析（测试用，不执行交易）
uv run kimi-quant --once

# 启动交易循环
uv run kimi-quant

# 自定义间隔（60秒）
uv run kimi-quant --interval 60
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

## 策略逻辑

LLM 系统提示词要求分析以下维度：

1. **趋势分析** — 短期动量方向
2. **订单簿分析** — 买卖墙位置、盘口失衡度
3. **资金费率** — 多头/空头拥挤信号
4. **升贴水** — 标记价格 vs 预言机价格偏离
5. **风险意识** — 不确定时优先 HOLD

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
├── kimi_quant/
│   ├── __init__.py
│   ├── config.py      # 配置管理
│   ├── data.py        # 市场数据 (Hyperliquid)
│   ├── llm.py         # Kimi K3 集成 (LangChain)
│   ├── risk.py        # 风控管理
│   ├── executor.py    # 交易执行
│   └── main.py        # 主程序入口
├── .env.example       # 环境变量模板
├── .gitignore
├── pyproject.toml
└── README.md
```

## 风险声明

⚠️ **免责声明**：本程序仅供学习和研究使用。量化交易存在重大亏损风险，大模型判断可能出错。请勿投入无法承受损失的资金。作者不对任何交易亏损承担责任。

## License

MIT
