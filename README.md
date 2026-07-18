# Kimi Quant 🚀

**BTC 永续合约量化交易程序** — 基于大模型的智能交易决策系统，运行于 Hyperliquid 去中心化交易所。

- 🧠 **双模型容灾**：Kimi K3 + DeepSeek V3，一键切换主备，自动降级
- 💰 **成本优化**：前缀缓存省 63% Debate 输入 token、LLM 引导长间隔、可配置唤醒下限
- 📱 **消息推送**：飞书实时通知交易事件，自动检测无需配置
- 🛡️ **多层防护**：Firefox TLS 指纹伪装 + 六层风控 + 启动重试 + 异常熔断，炸不穿
- 🕐 **自适应唤醒**：LLM 自决下次分析时间，默认 ≥5min，横盘自动拉长省费
- 📊 **多周期分析**：5m/15m/1h/4h K 线 + ATR + 订单簿 + 资金费率
- 💬 **两种策略**：Single（单 Agent 快速分析） / Debate（三 Agent 辩论 + 前缀缓存 + Judge 裁决）
- 📝 **完整记录**：交易盈亏 + 辩论历史 JSONL 持久化，支持并发读写

## 目录

- [策略模式](#策略模式)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [分阶段测试指南](#分阶段测试指南)
- [实盘部署](#实盘部署)
- [配置参考](#配置参考)
- [CLI 命令](#cli-命令)
- [TradingSignal](#tradingsignal)
- [风控规则](#风控规则)
- [Debate 模式详解](#debate-模式详解)
- [TradeExecutor SDK 覆盖](#tradeexecutor--sdk-全覆盖)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [风险声明](#风险声明)

## 策略模式

### Single-Agent (单 Agent 模式)
```
市场数据 → Kimi K3 分析 → TradingSignal → 风控 → 执行
```
1 次 LLM 调用，快速轻量。适合大多数场景。

### Multi-Agent Debate (多 Agent 辩论模式)
```
              ┌── 🐂 Bull Agent (论证做多) ──┐
市场数据 ───┼── 🐻 Bear Agent (论证做空) ──┼── ⚖️ Judge ──→ TradingSignal ──→ 风控 ──→ 执行
              └── 😐 Hold Agent (论证观望) ──┘
```
3 个 Agent 并行辩论 + 1 个裁判裁决，通过对抗验证减少单一模型偏见。4 次 LLM 调用/周期。

## 架构概览

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
│  DataProvider │───▶│    策略引擎      │───▶│  RiskManager │───▶│  TradeExecutor   │
│  (Hyperliquid)│    │ single / debate │    │  (六层风控)   │    │  (全订单生命周期)  │
└──────────────┘    └─────────────────┘    └──────────────┘    └──────────────────┘
      ▲                      │                                           │
      │                      ▼                                           │
      │              ┌──────────────┐                                    │
      └──────────────│ TradeLogger  │◀───────────────────────────────────┘
                     │ (盈亏反馈)    │
                     └──────────────┘
```

### 数据流

1. **DataProvider** — 行情数据走主网（`meta_and_asset_ctxs`，含 mark/oracle/funding/OI），多周期 K 线 TTL 缓存
2. **策略引擎** — Single：单次 LLM 分析；Debate：三 Agent 辩论 + Judge 裁决（60s 超时）
3. **RiskManager** — 六层风控校验（熔断、置信度、仓位、止损距离、方向、降级）
4. **TradeExecutor** — 启动恢复 + resting/active 状态机 + 15/15 SDK 全覆盖
5. **TradeLogger** — 盈亏分析 + LLM 表现反馈（自省循环）
6. **自适应间隔** — LLM 建议下次唤醒时间（60s-3h），横盘省费/关键位盯紧

### 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | Kimi K3 (主) + DeepSeek V3/V4 (自动降级备份) |
| LLM 编排 | LangChain + LangGraph StateGraph |
| 状态持久化 | LangGraph MemorySaver + JSONL 文件（fcntl 锁） |
| 交易所 | Hyperliquid (Perpetual DEX) |
| 结构化输出 | Pydantic + LangChain json_schema |
| 交易执行 | hyperliquid-python-sdk (15/15 全覆盖) |

## LLM 模型配置

### 双模型容灾架构

```
每个 LLM 调用 ──▶ 主模型 ──成功──▶ 返回
                    │
                    │ 失败 (超时/429/5xx/余额)
                    ▼
                 备用模型 ──成功──▶ 返回
                    │
                    │ 也失败
                    ▼
               raise → 上层处理
```

- 主备切换对策略代码完全透明，无需改动任何业务逻辑
- 启动日志会明确显示当前链路：`LLM: kimi primary → fallback: deepseek`

### 模型选择

```bash
# .env

# 方案 1: Kimi 主 → DeepSeek 备 (默认，推荐)
PRIMARY_LLM=kimi
MOONSHOT_API_KEY=sk-your-kimi-key
DEEPSEEK_API_KEY=sk-your-deepseek-key

# 方案 2: DeepSeek 主 → Kimi 备 (省钱，费用约 Kimi 的 1/10)
PRIMARY_LLM=deepseek

# 方案 3: 仅 Kimi (不配 DeepSeek key 即可)
# 方案 4: 仅 DeepSeek (不配 Kimi key 即可)
```

### 推理强度控制

一个 `REASONING_EFFORT` 变量，自动转换为各模型的原生 API 格式：

```bash
REASONING_EFFORT=max      # 最强推理 (默认)
REASONING_EFFORT=high     # 高推理
REASONING_EFFORT=medium   # 中等
REASONING_EFFORT=low      # 低推理
REASONING_EFFORT=minimal  # 最小推理
REASONING_EFFORT=off      # 关闭推理，大幅节省 token
```

底层转换：

| 配置 | Kimi K3 | DeepSeek V3/V4 |
|------|---------|-------------|
| `max` | `reasoning_effort: "max"` | `extra_body → thinking: enabled` |
| `high` ~ `minimal` | 不传参（K3 仅支持 max） | `extra_body → thinking: enabled` |
| `off` | 不传参 | `extra_body → thinking: disabled` |

**费用影响**：推理 token 占输出 80%。`REASONING_EFFORT=off` 每次调用可节省约 **75% 输出费用**。适合高频轮询或低成本模式。

## 自适应唤醒间隔

**大模型自己决定下次什么时候醒。** 无需外部 cron。无需手动 `--once` 反复跑。

### 原理

```
uv run kimi-quant
    │
    └── while True:
           ├── 获取行情 → LLM 分析 → 风控 → 执行
           ├── LLM 返回: "next_interval": 900
           ├── sleep(900)           ← 睡 15 分钟
           └── 醒来，重复
```

LLM 根据市场状态动态调整：

| 市场状态 | LLM 建议间隔 | 效果 |
|----------|:----------:|------|
| 已持仓、突破确认中 | 300-600s | 适度盯盘 |
| 正常行情、无仓位 | 600-1800s | 控费优先 |
| 横盘、无方向 | 1800-3600s | 降低成本 |
| 周末、低流动性 | 3600-10800s | 省到极致 |
| LLM 不填 | 使用 `TRADING_INTERVAL` 默认值 | 默认行为 |

### 边界保护

程序内置钳制：`[MIN_INTERVAL, MAX_INTERVAL]`，默认 `[300s, 10800s]`（5 分钟 ~ 3 小时），可通过环境变量调整。LLM 提示词也已引导优先使用较长间隔——只有已持仓或确认突破时才建议 300-600s。

### 零配置

不需要任何新参数。LLM 通过系统提示词知道允许的范围，自动在每次响应中建议合适的间隔。启动日志会显示间隔变化：

```
Cycle 5 complete: signal=HOLD confidence=0.65
LLM adjusted interval: 600s → 900s    ← 横盘拉长
Sleeping 900.0s until next cycle...
```

## 消息推送（微信 / 飞书）

通过 larky 的 `WeChatClient.notify()` 将消息发布到 Redis Pub/Sub，由 larky 的 `WeChatService`（独立进程 `python -m larky`）投递到微信。多程序共享同一套基础设施，无需各自管理 Bot 登录。

### 架构

```
kimi_quant ──WeChatClient──▶
                            │
cryptoguard ──WeChatClient──▶── wechat:outgoing ──▶ WeChatService ──▶ 微信
                            │     (Redis Pub/Sub)     (larky 独立进程)
其他程序   ──WeChatClient──▶
```

### 依赖

- `larky`（可编辑安装，已在 `pyproject.toml` 中）
- `redis`（larky 的传递依赖，已显式声明确保安装）
- `WeChatService` 独立运行（`python -m larky`），各程序共享

### 推送事件

| 事件 | 消息 |
|------|------|
| 🚀 启动 | 模式、模型、间隔 |
| ❌ 启动失败 | 错误详情 |
| 📈 开仓 | 方向、仓位、入场价、SL/TP、置信度 |
| 🟢/🔴 平仓 | 盈亏金额、百分比、平仓原因 |
| 🛡️ 风控拒绝 | 拒绝原因（置信度不足/熔断/止损太近等） |
| ⚠️ 熔断 | 连续亏损次数、冷却周期、累计盈亏 |
| ⚠️ 异常 | 首个错误 + 每 10 轮（防刷屏） |
| ⏹️ 停止 | 总周期、交易数、胜率、盈亏 |

### 自动检测

```
larky 可导入 → WeChatClient 发送（需 WeChatService 运行中）
有飞书 APP_ID → 飞书推送（降级方案）
都没有        → 静默运行

Redis 配置（可选，默认 localhost:6379）：
  REDIS_HOST=localhost
  REDIS_PORT=6379
  REDIS_DB=0
```
```

程序启动时自动 ping Redis，连通即走微信通道。发送失败自动重连，不会因 Redis 临时重启而永久静默。`priority="high"` 确保离线消息不丢失（Redis 队列暂存，恢复后补发）。

## 快速开始

### 环境要求

- Python >= 3.13
- Linux (WSL2 / Arch / Ubuntu) 或 macOS
- [uv](https://docs.astral.sh/uv/) 包管理器
- Kimi (Moonshot) API Key → [platform.moonshot.cn](https://platform.moonshot.cn)

### 安装

```bash
git clone <repo-url> && cd kimi_quant
uv sync
cp .env.example .env
```

### 最小配置

编辑 `.env`，最少只需要填一个 API Key：

```bash
MOONSHOT_API_KEY=sk-your-key-here    # 必填（Kimi 或 DeepSeek 至少一个）
# DEEPSEEK_API_KEY=sk-...           # 可选，配置后自动作为降级备份
```

其他配置保持默认即可（dry-run 模式，不涉及真实资金）。

### 跑起来

```bash
# 单次分析（验证环境正常）
uv run kimi-quant --once

# 启动！一个命令，一直跑，无需 cron
uv run kimi-quant
#   ↓ 程序内部自己循环：
#     获取行情 → 问 LLM → 风控 → 执行 → LLM 说睡多久就睡多久 → 醒来重复

# 另一终端，随时查看状态
uv run kimi-quant --stats      # 盈亏
uv run kimi-quant --history    # 辩论记录
```

**启动后你就可以关屏幕了。** 不需要 crontab，不需要 systemd timer，不需要反复手动跑。程序内部是 `while True` 循环，LLM 自己决定下次什么时候醒来。

## 分阶段测试指南

**不要直接用真金白银跑！** 按以下三个阶段循序渐进：

### 🧪 阶段 1：Dry-Run（零成本，1-3 天）

**目的**：验证 Kimi API 通畅、LLM 决策质量、系统稳定性。

```bash
# .env 配置
DRY_RUN=true                      # 模拟模式
STRATEGY_MODE=single              # 先用 single 模式
TRADING_INTERVAL=300              # 5 分钟一个周期
```

```bash
# 跑一次看输出
uv run kimi-quant --once

# 连续跑（建议开 tmux/screen 后台跑半天到一天）
uv run kimi-quant --interval 300

# 另一终端监控模拟盈亏
watch -n 300 'uv run kimi-quant --stats'
```

**检查清单**：
- [ ] 每个 cycle 正常输出 market data（价格、价差、资金费率）
- [ ] LLM 返回有效信号（action、confidence、reasoning）
- [ ] 风控校验正常（rejected/executed 有合理理由）
- [ ] `--stats` 显示模拟交易记录
- [ ] 没有异常崩溃或 API 报错

### 🧪 阶段 2：Hyperliquid 测试网（零成本，1-3 天）

**目的**：验证链上交互（下单、撤单、止损触发）在真实环境正常工作。

#### 2.1 获取钱包私钥

Hyperliquid 使用以太坊兼容地址。你可以使用任何 EVM 钱包（OKX Web3、MetaMask、Rabby 等）：

```
OKX App → 钱包 → 钱包管理 → 导出私钥 → 复制
MetaMask → 账户详情 → 导出私钥 → 复制
```

你会得到一个 `0x` 开头的 64 位十六进制字符串。

#### 2.2 查看你的 Hyperliquid 地址

```bash
uv run python -c "
from eth_account import Account
acct = Account.from_key('0x你的私钥')
print('Hyperliquid 地址:', acct.address)
"
```

#### 2.3 领测试币

1. 访问 [Hyperliquid Testnet](https://app.hyperliquid-testnet.xyz/trade)
2. 在右上角点 "Deposit"，用 Arbitrum Sepolia 测试网领取测试 USDC
3. 或者用官方 [Faucet](https://hyperliquid.gitbook.io/hyperliquid-docs/onboarding/testnet-faucet)

#### 2.4 配置测试网

```bash
# 备份 dry-run 配置
cp .env .env.dry-run

# 编辑 .env
MOONSHOT_API_KEY=sk-your-key-here
HYPERLIQUID_PRIVATE_KEY=0x你的私钥
HYPERLIQUID_TESTNET=true            # 测试网
DRY_RUN=false                       # 开启实盘执行（测试网）
TRADING_INTERVAL=300
MAX_POSITION_SIZE=0.001             # 极小仓位 0.001 BTC
MAX_LEVERAGE=1                      # 1x 无杠杆
MIN_CONFIDENCE=0.75                 # 提高置信度门槛（测试网更保守）
```

#### 2.5 跑起来

```bash
# 单次：确认能连接、下单、返回结果
uv run kimi-quant --once

# 连续：观察几个完整周期
uv run kimi-quant --interval 120     # 2 分钟，加速测试

# 监控
uv run kimi-quant --stats
```

**检查清单**：
- [ ] 启动日志显示 `TradeExecutor initialized (address=0x... testnet=True)`
- [ ] 仓位追踪正确（`Position: [ACTIVE] LONG 0.0010 BTC @ $...`）
- [ ] SL/TP 订单正常创建
- [ ] CLOSE 信号正常平仓
- [ ] 风控熔断机制正常触发
- [ ] `--stats` 显示盈亏记录
- [ ] 去 [Hyperliquid Testnet](https://app.hyperliquid-testnet.xyz/trade) 确认仓位/订单可见

### 🚀 阶段 3：主网实盘（极小仓位）

**在测试网上一切正常后**才能进入此阶段。

#### 3.1 入金到 Hyperliquid

资金路径（选择其一）：

```
路径 A（推荐，通过 Arbitrum）：
  交易所买 USDC → 提币到 Arbitrum 链上你的地址
  → app.hyperliquid.xyz/bridge → 跨链存入 Hyperliquid L1

路径 B（直接充值）：
  从交易所提币 USDC 到 Arbitrum 链
  → app.hyperliquid.xyz → Deposit
```

**首次入金建议 $100-200 USDC 即可。不要一次放太多。**

#### 3.2 配置主网

```bash
# 备份测试网配置
cp .env .env.testnet

# .env
MOONSHOT_API_KEY=sk-your-key-here
HYPERLIQUID_PRIVATE_KEY=0x你的私钥
HYPERLIQUID_TESTNET=false           # 主网！
HYPERLIQUID_BASE_URL=https://api.hyperliquid.xyz
DRY_RUN=false

# 极度保守的启动参数
MAX_POSITION_SIZE=0.001             # 0.001 BTC ≈ $87
MAX_LEVERAGE=1                      # 1x 无杠杆，不会被清算
MIN_CONFIDENCE=0.75                 # 高置信度才开仓
TRADING_INTERVAL=300                # 5 分钟
```

#### 3.3 人工监控下运行

```bash
# 前台运行，盯着看 5-10 个周期
uv run kimi-quant --interval 300

# 另一终端实时监控
watch -n 60 'echo "=== $(date) ===" && uv run kimi-quant --stats'
```

**前 24 小时**：
- 确保人在电脑前或定期检查
- 每笔交易后看 `--stats`
- 去 [Hyperliquid App](https://app.hyperliquid.xyz/trade) 确认仓位状态与程序一致

#### 3.4 逐步放大

确认系统稳定盈利后，每次只改一个参数，至少观察 1-2 天：

```bash
# 演进路线（逐阶段，不要跳过）
MAX_POSITION_SIZE=0.002   # → 0.005 → 0.01
MAX_LEVERAGE=2            # → 3
MIN_CONFIDENCE=0.70       # → 0.65
STRATEGY_MODE=debate      # 切换到辩论模式（注意费用：4x LLM 调用）
```

## 实盘部署

### 后台运行 (tmux)

```bash
# 创建会话
tmux new -s kimi

# 启动交易程序
uv run kimi-quant --mode single --interval 300

# 断开会话（程序继续运行）
Ctrl+B, D

# 重新连接
tmux attach -t kimi
```

### 后台运行 (systemd)

```bash
# 创建 service 文件
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

# 查看日志
journalctl -u kimi-quant -f
```

### 健康监控

```bash
# 设置定时告警（crontab）
*/10 * * * * cd /path/to/kimi_quant && uv run kimi-quant --stats 2>&1 | grep -q "Net P&L.*-[5-9][0-9]" && notify-send "Kimi Quant: 大幅回撤警告"

# 检查进程是否存活
pgrep -f kimi-quant || echo "WARNING: Bot is not running!"
```

## 配置参考

### 完整环境变量列表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| **必填** | | |
| `MOONSHOT_API_KEY` | — | Kimi API Key ([获取地址](https://platform.moonshot.cn)) |
| **Kimi (Moonshot)** | | |
| `MOONSHOT_API_KEY` | — | Kimi API Key |
| `MOONSHOT_BASE_URL` | `https://api.moonshot.cn/v1` | API 端点 |
| `KIMI_MODEL` | `kimi-k3` | 模型名称 |
| **DeepSeek (可选备份)** | | 自动降级 |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（留空仅用 Kimi） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API 端点 |
| `DEEPSEEK_MODEL` | `deepseek-v3.1` | 模型名称 |
| **LLM 参数** | | |
| `PRIMARY_LLM` | `kimi` | 主模型：`kimi` 或 `deepseek` |
| `REASONING_EFFORT` | `max` | 推理强度：`max`/`high`/`medium`/`low`/`minimal`/`off` |
| `LLM_TEMPERATURE` | `0.1` | LLM 温度 (0-2) |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token（不影响 1M 上下文输入） |
| `JUDGE_TEMPERATURE` | `0.05` | Debate 模式 Judge 温度 |
| **Hyperliquid** | | |
| `HYPERLIQUID_PRIVATE_KEY` | — | 钱包私钥（实盘必填） |
| `HYPERLIQUID_TESTNET` | `true` | `true`=测试网, `false`=主网 |
| `HYPERLIQUID_BASE_URL` | `https://api.hyperliquid.xyz` | 主网 API |
| **交易参数** | | |
| `TRADING_PAIR` | `BTC` | 交易对 |
| `MAX_POSITION_SIZE` | `0.01` | 最大仓位 (**单位：BTC**，非 USD) |
| `MIN_CONFIDENCE` | `0.7` | 最低置信度阈值 |
| `MAX_LEVERAGE` | `3` | 最大杠杆倍数 |
| **策略** | | |
| `STRATEGY_MODE` | `single` | `single` 或 `debate` |
| `TRADING_INTERVAL` | `600` | 默认间隔（秒）。LLM 可通过 `next_interval` 动态覆盖 |
| `MIN_INTERVAL` | `300` | LLM 建议间隔的硬下限（秒，默认 5 分钟） |
| `MAX_INTERVAL` | `10800` | LLM 建议间隔的硬上限（秒，默认 3 小时） |
| `DRY_RUN` | `true` | 模拟模式开关 |
| **日志** | | |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### 仓位大小参考

`MAX_POSITION_SIZE` 的单位是 **BTC 数量**，不是 USD：

| 设置 | BTC 数量 | 约合 USD (BTC=$87k) | 适合场景 |
|------|----------|---------------------|----------|
| `0.001` | 0.001 BTC | ~$87 | 首次实盘测试 |
| `0.005` | 0.005 BTC | ~$435 | 谨慎运行 |
| `0.01` | 0.01 BTC | ~$870 | 常规运行 |
| `0.05` | 0.05 BTC | ~$4,350 | 需要较大本金 |

## CLI 命令

```bash
# 核心命令
uv run kimi-quant                              # 启动连续交易循环
uv run kimi-quant --once                       # 单次分析（返回 JSON 结果）
uv run kimi-quant --once --mode debate         # 单次辩论模式

# 参数
uv run kimi-quant --interval 120               # 自定义间隔（秒）
uv run kimi-quant --mode single                # 指定策略模式
uv run kimi-quant --mode debate --interval 300 # 组合参数

# 数据查询（可安全地与运行中的程序并发）
uv run kimi-quant --stats                      # 查看盈亏统计（真实+模拟）
uv run kimi-quant --history                    # 查看辩论历史记录
```

## TradingSignal

### LLM 输出格式

```json
{
  "action": "LONG",
  "confidence": 0.85,
  "reasoning": "1h 和 4h 均上涨，订单簿 bid wall 强劲...",
  "size": 0.005,
  "entry_price": 87000.0,
  "stop_loss": 86500.0,
  "take_profit": 88000.0,
  "modify_sl_to": null,
  "key_factors": ["4h uptrend", "bid wall at 86500", "funding neutral"],
  "next_interval": 120
}
```
> `next_interval`：LLM 建议的下次唤醒秒数（60-10800），null 则用默认值。上例中 120s 表示开仓后紧盯。

### Action 说明

| Action | 触发条件 | 订单操作 |
|--------|---------|---------|
| `LONG` | 看多信号 | 开多仓 + SL + TP（bulk_orders 原子执行） |
| `SHORT` | 看空信号 | 开空仓 + SL + TP（bulk_orders 原子执行） |
| `CLOSE` | 平仓信号 | 市价平仓 |
| `HOLD` | 观望/不确定 | 无操作 |
| `MODIFY_SL` | 移动止损 | 将 SL 移至新价格（保本/追踪止损） |

**字段说明**：
- `entry_price`: 设为具体价格 → 限价单；设为 `null` → 市价单（Ioc）
- `stop_loss`: **强制字段**，LONG/SHORT 时必须提供，且距入场价 ≥ 0.5%
- `size`: `null` 时自动使用 `MAX_POSITION_SIZE`
- `modify_sl_to`: 仅 MODIFY_SL 时使用，指定新的止损价格

## 风控规则

### 六层校验

| 层级 | 检查项 | 规则 |
|------|--------|------|
| 1 | **熔断机制** | 连续 4 笔亏损 → 暂停 6 个 cycle；日回撤 > 5% 冻结；cooldown 内不延长 |
| 2 | **置信度** | >= `MIN_CONFIDENCE` (默认 0.7) 才执行方向性交易 |
| 3 | **仓位上限** | 不超过 `MAX_POSITION_SIZE`；单笔风险 > 1% 账户警告，> 2% 拒绝 |
| 4 | **止损距离** | ≥ 0.5% 距入场价（BTC 噪音 ~0.3%，低于此阈值拒绝） |
| 5 | **方向** | 已有同向仓位拒绝；CLOSE/MODIFY_SL 需已持仓 |
| 6 | **降级保护** | CLOSE 和 MODIFY_SL 始终允许（降低风险的操作不受熔断限制） |

### 熔断状态机

```
正常交易 ──(连续4亏)──▶ Cooldown(6 cycles) ──(到期)──▶ 正常交易
                            │
                            │ (期间亏损不延长 cooldown)
                            │ (期间盈利自动清零)
                            ▼
                        Cooldown 继续倒数
```

### 多周期 ATR 分析

DataProvider 自动计算每个时间周期的 ATR（Average True Range）：
- 5m / 15m / 1h / 4h 各周期的 ATR 绝对值与百分比写入 LLM prompt
- LLM 根据 ATR 设定更合理的止损距离
- 风控系统使用 ATR 辅助判断止损是否合理

### 仓位状态追踪

```
PositionTracker 三态模型:
  none ──(下单)──▶ resting ──(链上确认)──▶ active
       │                 │                        │
       │                 │ (超时 3 cycle)          │ (SL/TP 触发)
       │                 ▼                        ▼
       │            cancel_resting()           clear()
       │            + cancel_pending()         + record_close()
       └────────────────────────────────────────────┘
                     (崩溃恢复 + 补录交易)
```

### 崩溃恢复

TradeExecutor 启动时自动查询链上状态：
- 恢复已有持仓（`user_state` → positions）
- 恢复挂单 ID（`open_orders` → SL/TP oid）
- 恢复仓位自动补录 TradeLogger pending trade
- 崩溃重启后可立即管理现有仓位（平仓、移止损）

### Dry-Run 模拟盈亏

模拟模式支持完整的交易记录：
- 所有交易（含模拟）持久化到 `data/trades.jsonl`
- `--stats` 分开展示 `[LIVE]` / `[SIM]` 交易统计
- LLM 表现反馈仅基于真实交易，避免模拟数据污染自省循环

## Debate 模式详解

### Agent 分工

| Agent | 角色 | 系统提示词要点 |
|-------|------|---------------|
| 🐂 Bull | 激进多头分析师 | 寻找做多证据：支撑位、bid wall、负资金费率 |
| 🐻 Bear | 怀疑派空头分析师 | 寻找做空证据：阻力位、ask wall、正资金费率 |
| 😐 Hold | 谨慎风控官 | 寻找观望理由：信号矛盾、波动过大、无明确方向 |
| ⚖️ Judge | 首席交易官 | 综合多周期趋势裁决，1h/4h 权重 > 5m/15m |

### 决策流程

1. **Phase 1 — 缓存预热**：Hold Agent 先跑，其 prefill 阶段将行情数据写入 DeepSeek KV-cache
2. **Phase 2 — 并行命中**：Bull + Bear 并发跑，两份请求的前缀（行情数据）命中缓存，仅对 ~50 token 的角色指令计费
3. Judge **综合多周期趋势**裁决（1h/4h 趋势权重 > 5m/15m，分歧时不默认 HOLD）
4. Judge 输出结构化 TradingSignal → 风控 → 执行

> **前缀缓存**：三个 Agent 接收完全相同的行情数据。将其置于 prompt 最前面（system message），DeepSeek V3 的后端自动复用第一个请求的 KV-cache。Phase 2 的两个 Agent 输入 token 费用降低 ~95%，整个 Debate 输入 token 省 ~63%。见[费用优化](#费用优化)。

### Judge 决策框架

| 市场状态 | Judge 决策倾向 |
|----------|---------------|
| 1h↑ + 4h↑ + Bull 强 | LONG，confidence 0.75+ |
| 1h↓ + 4h↓ + Bear 强 | SHORT，confidence 0.75+ |
| 4h↑ 但 1h↓ | 倾向 LONG（高 TF 主导），减仓+紧止损，confidence 0.65-0.75 |
| 4h↓ 但短线反弹 | 倾向 SHORT，小仓，confidence 0.65-0.75 |
| 所有 TF 横盘 | HOLD 可接受，但关注突破 |
| 三方论证都弱 + 无明确趋势 | 仅此时默认 HOLD |

### LangGraph Checkpointing

每个 cycle 的完整辩论结果持久化到 `data/debate.jsonl`：

```
每行一个 JSON 对象:
  ├── cycle_id             ← ISO 时间戳
  ├── account_summary      ← 账户摘要
  ├── bull_argument        ← Bull Agent 论据
  ├── bear_argument        ← Bear Agent 论据
  ├── hold_argument        ← Hold Agent 论据
  └── final_signal_json    ← Judge 裁决
```

- **断点续传**：崩溃重启后 `get_latest_state()` 恢复
- **历史回溯**：`--history` 打印完整辩论记录
- **自动保存**：LangGraph 在每次 `ainvoke()` 后自动写 checkpoint

### 费用注意

Debate 模式每个周期调用 4 次 LLM（3 Debater + 1 Judge），API 费用是 single 模式的约 4 倍。建议先在 single 模式验证策略有效，再切换到 debate。

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

## 项目结构

```
kimi_quant/
├── src/kimi_quant/
│   ├── __init__.py      # 包定义
│   ├── config.py        # 配置管理（env + .env）
│   ├── data.py          # 市场数据（Hyperliquid Info API + K线缓存 + ATR）
│   ├── llm.py           # TradingSignal + 双模型容灾 (Kimi/DeepSeek)
│   ├── debate.py        # Multi-Agent 辩论 + LangGraph Checkpointing
│   ├── risk.py          # 六层风控校验 + 熔断状态机
│   ├── executor.py      # 15/15 SDK 全覆盖 + PositionTracker 三态模型
│   ├── analytics.py     # TradeLogger — 盈亏分析 + LLM 自省反馈
│   ├── notify.py        # 微信/飞书消息推送（可选，自动检测）
│   └── main.py          # CLI 入口 + 交易循环
├── data/
│   ├── debate.jsonl   # 辩论历史记录 (JSONL, fcntl 文件锁)
│   └── trades.jsonl   # 交易记录 JSONL（fcntl 文件锁，支持并发读写）
├── .env                 # 实际配置（gitignore，不提交）
├── .env.example         # 配置模板
├── .gitignore
├── pyproject.toml
└── README.md
```

## 常见问题

### Q: 阿里云服务器无法连接 Hyperliquid API？

阿里云出口网关会对 Python 默认 SSL 库进行 TLS 指纹检测并 Reset 连接（`curl` 命令行正常但 Python 报 `ConnectionResetError`或 `SSLError: curl: (35) Recv failure`）。本项目已内置两层防护：

**第一层 — TLS 指纹伪装（curl_cffi）**：自动伪装成 Firefox 147 浏览器的 JA3 TLS 指纹。选择 Firefox 而非 Chrome 是因为反爬服务对 Chrome 指纹的检测最严格（Chrome 是最常被仿冒的浏览器），Firefox 的 TLS 密码套件和扩展信号不同，不在重点盯防范围。

**第二层 — 重试+限流保护**：阿里云不仅检测指纹，还会对并发请求频率敏感。如果同一时刻发起过多 TLS 握手（例如多线程并行请求），即使指纹正确也会被临时封锁。代码已内置：
- **指数退避重试**：遇到 `Connection reset by peer` 等瞬时错误时自动重试（最多 3 次，间隔 1.5s → 3s → 6s + 随机抖动）
- **并发限制**：并行 API 请求上限从 5 降到 2，避免触发频率封锁
- **交错提交**：每个并行任务间隔 150ms 提交，避免 TLS 握手瞬时爆量

服务器上运行前确保 `curl_cffi` 已安装：
```bash
uv sync   # 自动安装 curl_cffi
```
启动日志中显示 `curl_cffi=True` 表示已激活：
```
DataProvider initialized (testnet=False, coin=BTC, curl_cffi=True)
```
正常运行时如果偶尔看到 `retrying in X.Xs` 的 WARNING 日志，说明触发了临时封锁并正在自动恢复——**不需要人工干预**。只有当同一次调用 4 次重试全部失败时才会报 ERROR。

### Q: API 费用多少钱？怎么省钱？

**定价对比**（每 1M tokens）：

| 模型 | 输入 | 输出 | vs Kimi |
|------|------|------|---------|
| Kimi K3 | ¥20 | ¥100 | — |
| DeepSeek V3 | ¥2 | ¥8 | **便宜 90%+** |

**月成本估算**（10 分钟间隔，Single 模式）：

| 配置 | 月成本 |
|------|--------|
| Kimi K3 | ~¥330 |
| DeepSeek V3 | ~¥40 |
| Kimi + DeepSeek 备份 | ~¥330（仅 Kimi 可用时） |

**Debate 模式**（4 次 LLM 调用/周期，含前缀缓存优化）：

| 配置 | 月成本 | vs Single |
|------|--------|-----------|
| DeepSeek V3 | ~¥60 | 1.5x（缓存省 63% 输入 token） |
| Kimi K3 | ~¥450 | 1.4x |

**省钱三板斧**：

| 策略 | .env 配置 | 效果 |
|------|----------|------|
| 用 DeepSeek 主力 | `PRIMARY_LLM=deepseek` | 月成本 ¥330→¥40 |
| 关推理 | `REASONING_EFFORT=off` | 输出费用再降 75% |
| 加大间隔 | `TRADING_INTERVAL=900` | 成本再降 1/3 |
| 调高最低间隔 | `MIN_INTERVAL=600` | 防止 LLM 频繁唤醒（默认 300s） |

**极限省钱方案**：`PRIMARY_LLM=deepseek + REASONING_EFFORT=off + TRADING_INTERVAL=900`，月成本约 **¥10**。

### Q: Debate 模式怎么省 token？

三个辩论 Agent 接收相同的行情数据（~1500 tokens）。系统已做两项优化：

1. **前缀缓存**：行情数据放在 prompt 最前面，DeepSeek V3 自动复用 KV-cache。Hold 先跑预热缓存，Bull + Bear 并发命中——后两个 Agent 只对 ~50 token 角色指令计费。
2. **Judge 精简输入**：Judge 只收辩论摘要（不含原始行情），节省 ~450 token/周期。

整体效果：Debate 的 4 次调用等效 ~2.5 次 Single 调用的输入 token 量。

### Q: 如何切换主/备模型？

一个环境变量：

```bash
PRIMARY_LLM=kimi      # Kimi 主力，DeepSeek 备份（默认）
PRIMARY_LLM=deepseek  # DeepSeek 主力，Kimi 备份（省钱）
```

只需配好两个模型的 API Key，主模型挂了自动切备机。不需改任何代码。见 [LLM 模型配置](#llm-模型配置)。

### Q: 设了 `PRIMARY_LLM=deepseek`，但 HTTP 请求还是发到 Kimi API？

DeepSeek 调用失败时会静默降级到 Kimi（LangChain `with_fallbacks` 机制），`httpx` 日志只记录成功的请求，所以看到的全是 `api.moonshot.cn`。常见原因：

1. **`thinking` 参数格式错误** — 必须通过 `extra_body` 传递（v2.2+ 已修复）
2. **API key 过期或余额不足**
3. **网络不可达**（某些机房有墙）

快速诊断：
```bash
# 在服务器上直接测试 DeepSeek API
curl -s https://api.deepseek.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -d '{"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}'
```
正常返回 JSON → API 通；报错/超时 → 检查网络和 key。

### Q: Kimi API 挂了怎么办？

配置 `DEEPSEEK_API_KEY` 即可。Kimi 调用失败时自动切换到 DeepSeek，无需人工干预。日志会显示 `LLM: kimi primary → fallback: deepseek`。未配置则仅用 Kimi。

### Q: 程序会因为异常崩溃吗？

不会。三层错误防护确保程序炸不穿：

```
Layer 0 (启动):  配置错误 → 立即报错退出（必须人为修复）
Layer 1 (每轮):  LLM 挂/API 炸/网络断 → 记日志，下轮继续
Layer 2 (休眠):  sleep 中断 → 记日志，下轮继续
```

启动后的任何异常——API 限流、LLM 超时、网络抖动、磁盘满——都只影响当前轮次。程序持续运行，下个周期自动重试。加上 systemd 的 `Restart=on-failure`，即使进程被 OOM killer 杀掉也会自动拉起。

### Q: 怎么收到交易通知？

前提条件：
1. 服务器上运行着 `python -m larky`（WeChatService，独立进程）
2. `redis` 包已安装（`uv sync` 自动处理）

满足条件后程序自动通过 `WeChatClient.notify()` 推送。无需额外配置。启动日志会显示：
```
Notification: larky WeChatClient available (WeChat via Redis)
```
如果 Redis 不在 `localhost:6379`，设置 `REDIS_HOST` / `REDIS_PORT` 环境变量。

### Q: 程序怎么自动运行？需要 cron 吗？

不需要。`uv run kimi-quant` 启动后进入 `while True` 循环，程序内部自动反复执行，LLM 自己决定每次睡多久。一个命令，永远运行，直到 `Ctrl+C` 停止。

### Q: 怎么确定下次什么时候分析？

大模型在每次响应中返回 `next_interval`（秒），告诉程序睡多久再醒来。横盘时建议 15-30 分钟省费用，关键位附近建议 1-2 分钟盯紧。程序自动执行，你不需要任何操作。

### Q: 推理强度 (REASONING_EFFORT) 怎么调？

```bash
REASONING_EFFORT=max      # 最强分析（默认，推荐交易用）
REASONING_EFFORT=off      # 关闭推理，最快最省
```

推理 token 通常占输出 80%。设为 `off` 可节省约 75% 输出费用。日常监控或低成本模式推荐 `off`，关键交易建议 `max`。详见 [推理强度控制](#推理强度控制)。

### Q: 我只有 OKX Web3 钱包，能用吗？

能。OKX Web3 钱包本质是自托管钱包，导出私钥后与 MetaMask/Rabby 完全一样使用。见[阶段 2.1](#21-获取钱包私钥)。

### Q: 为什么推荐先跑 dry-run？

Dry-run 不涉及任何链上操作，只验证 LLM 决策逻辑。你可以在零成本下观察到：
- LLM 在什么市场条件下会给出什么信号
- 信号的胜率大概如何（通过模拟盈亏）
- 系统是否稳定（有无崩溃、API 报错）

### Q: 测试网和主网有什么区别？

- **测试网**：USDC 是免费的测试币，可以无限领。用来验证下单/撤单/止损等链上操作是否正常
- **主网**：真金白银。只有在测试网一切正常后才切换

### Q: 多久跑一个周期合适？

| 策略 | 推荐间隔 | 说明 |
|------|---------|------|
| 短线/震荡 | 120-300s (2-5min) | 利用订单簿变化和短期动量 |
| 中线/趋势 | 600-900s (10-15min) | 更依赖多周期 K 线趋势分析 |
| 长线 | 1800-3600s (30-60min) | 仅关注 4h 等大周期趋势 |

默认 300s (5min) 是较平衡的选择。

### Q: LLM 会不会乱下单？

有六层风控保护。即使 LLM 给出不合理信号，风控层也会拒绝。常见被拒绝的情况：
- 置信度 < 0.7
- 止损太近 (< 0.5%)
- 单笔风险 > 2% 账户
- 已有同向仓位
- 熔断激活中

### Q: 运行中怎么查看状态？

```bash
# 实时盈亏
uv run kimi-quant --stats

# 辩论历史
uv run kimi-quant --history

# 查看最新交易
tail -5 data/trades.jsonl | python -m json.tool
```

### Q: 辩论历史存在哪里？重启后能恢复吗？

辩论结果持久化在 `data/debate.jsonl`（JSONL 格式，使用 `fcntl` 文件锁支持并发读写）。`--history` 命令可随时查看，重启不丢失。

### Q: 怎么停止程序？

- 前台运行：`Ctrl+C`（优雅关闭，完成当前 cycle 后退出）
- tmux 内：`Ctrl+C` 或 `tmux kill-session -t kimi`
- systemd：`sudo systemctl stop kimi-quant`

## 风险声明

⚠️ **请在使用前仔细阅读：**

1. **量化交易存在重大亏损风险**。历史表现不代表未来收益。大模型（Kimi K3）的判断可能出错。
2. **请勿投入无法承受损失的资金**。首次实盘建议不超过 $200。
3. **私钥安全**：`.env` 文件明文存储私钥。确保运行环境安全，`.env` 已被 `.gitignore` 排除不会提交到 Git。
4. **API 风险**：Kimi API 或 Hyperliquid API 可能出现延迟、限流或临时不可用。程序已包含异常处理但无法消除风险。
5. **清算风险**：使用杠杆可能被清算。建议 `MAX_LEVERAGE=1`（无杠杆）开始。
6. **资金费率风险**：持有仓位每 8 小时支付/收取资金费。极端行情下费率可能很高。
7. **免责声明**：本程序仅供学习和研究使用。使用者自行承担所有交易盈亏责任。作者不对任何交易亏损承担责任。

## License

MIT
