# Kimi Quant 🚀

**BTC 永续合约量化交易程序** — 大模型驱动的自主交易决策系统，运行于 Hyperliquid 去中心化交易所。LLM 自己决定何时交易、如何交易、何时休息。一个命令启动，无需 cron。

### 决策与策略
- 🎯 **强制决策工作流**：Step 0 审视持仓 → Step 1 市场分析 → Step 1.5 反事实检查清单 → 输出信号
- 💬 **单/双策略模式**：Single（快速轻量）/ Debate（Bull+Bear+Hold 三 Agent 辩论 + 可选反驳轮 + Judge 交叉验证裁决）
- 📌 **持仓记忆**：LLM 可看到原始开仓逻辑、持仓时长、MFE/MAE，真正验证 thesis 是否仍然成立
- 🔄 **决策反馈闭环**：上周期决策→结果自动注入，"上次说等回踩 → 现在回踩了"的自主学习

### 风控与安全
- 🛡️ **七层风控**：熔断、置信度、仓位、保证金、风险金额、止损距离、方向限制
- 🔁 **风控拒绝修正**：信号被拒后给 LLM 一次调整机会（如放宽止损、减小仓位）
- 💰 **Tick Size 自动对齐**：所有订单价格自动取整至交易所最小价格单位，防止整组订单被拒
- 🔍 **SL/TP 链上验证**：每周期交叉对比 tracker 与链上挂单，丢失立即告警并恢复
- 🔐 **TLS 指纹伪装**：curl_cffi 模拟 Firefox 指纹，绕过阿里云等云服务器的 TLS 检测

### 数据与分析
- ⏰ **时区/日历感知**：自动注入 UTC/北京时间、星期、交易时段（亚/欧/美盘），周末低流动性警告
- 📊 **多周期分析**：5m/15m/1h/4h K 线 + ATR + 订单簿 + 资金费率
- 📈 **周期间 diff + 波动率分类**：自动对比上轮数据 + HIGH/NORMAL/LOW 波动率环境判定

### 成本与性能
- 🧠 **双模型容灾**：Kimi K3 + DeepSeek V4，主模型挂了自动切备机；Judge 可独立指定强模型
- 💸 **DeepSeek 上下文缓存**：Debate 周期内省 55-73% 输入 token；辩弱 Judge 强配置比全 Kimi 省 65%
- 🕐 **自适应唤醒间隔**：LLM 自己决定下次何时醒来（5min-3h），横盘自动拉长省费

### 运维与工具
- 📱 **消息推送**：微信/飞书实时通知交易事件，自动检测无需配置
- 🔄 **WebSocket 实时监控**：订单成交/SL/TP 触发毫秒级感知，Flash 模型生成中文通知
- 🔧 **CLI 账户工具**：一键查余额/持仓/挂单、切换账户类型、划转资金
- 📝 **完整记录**：交易盈亏 + 辩论历史 JSONL 持久化，fcntl 文件锁支持并发读写

## 目录

- [快速开始](#快速开始)
- [策略模式](#策略模式)
- [架构概览](#架构概览)
- [LLM 模型配置](#llm-模型配置)
- [自适应唤醒间隔](#自适应唤醒间隔)
- [DeepSeek 上下文硬盘缓存](#deepseek-上下文硬盘缓存)
- [消息推送](#消息推送微信--飞书)
- [订单实时监控](#订单实时监控)
- [分阶段测试指南](#分阶段测试指南)
- [实盘部署](#实盘部署)
- [配置参考](#配置参考)
- [CLI 命令](#cli-命令)
- [账户管理](#账户管理)
- [TradingSignal](#tradingsignal)
- [风控规则](#风控规则)
- [提示词增强系统（v3.0）](#提示词增强系统v30)
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
              └── 😐 Hold Agent (论证观望) ──┘        ↑
                                             完整市场数据 + 三方论证
```
3 个 Agent 并行辩论 + 1 个裁判裁决，通过对抗验证减少单一模型偏见。4 次 LLM 调用/周期。
**所有 4 个 Agent 看到相同的原始市场数据**（行情、K 线、订单簿、资金费率、风控约束）——Judge 可以在裁决时交叉验证辩手的论点。

### 决策工作流（Decision Workflow）

无论哪种策略模式，LLM 每周期都遵循**强制两步决策流程**：

```
┌─────────────────────────────────────────────────────┐
│ Step 0 — 审视已有状态（在分析市场之前强制执行）        │
│                                                     │
│  a. 持仓合理性：原始入场逻辑是否依然成立？             │
│     → 逻辑破坏 → CLOSE                              │
│     → 运行良好 → 考虑 MODIFY_SL 锁利润/移保本         │
│                                                     │
│  b. SL/TP 链上验证：tracker 记录的 SL/TP 是否在链上？  │
│     → 丢失 → MODIFY_SL/MODIFY_TP 立即恢复（最高优先级）│
│     → 存在 → 继续                                    │
│                                                     │
│  c. 僵尸订单清理：链上是否有非 bot 创建的挂单？        │
│                                                     │
│  d. SL/TP 距离检查：止损止盈距离是否匹配当前 ATR？     │
│     → 波动率下降 → 收紧止损                           │
│     → 波动率上升 → 放宽止损                           │
├─────────────────────────────────────────────────────┤
│ Step 1 — 市场分析（完成 Step 0 后才能进行）            │
│                                                     │
│  多周期趋势 → 订单簿 → 资金费率 → 决策                  │
│  Step 0 的修复动作必须放在 actions 数组最前面          │
├─────────────────────────────────────────────────────┤
│ Step 1.5 — 强制反事实检查（方向性决策前逐项回答）       │
│                                                     │
│  1. 失效价位: "如果 BTC 跌破 $___, 我的 thesis 就错了"  │
│     → 这个价位就是止损位，不要随意设置                  │
│  2. ATR 校验: "1h ATR = X.X%, SL 距离必须 ≥ 1.5× ATR"  │
│  3. 15 分钟反转: "如果 15 分钟内反向 2%, 我漏了什么信号?" │
│  4. 偏差审计: 数据证据(0-10) ___ vs 主观期望(0-10) ___   │
│     → 期望 > 证据 → confidence -0.15 或 HOLD            │
│  综合检查引发严重疑虑 → HOLD 是正确决策，永远有下一笔交易  │
└─────────────────────────────────────────────────────┘
```

**关键原则**：Step 0 的修复动作（如恢复丢失的 SL）优先级高于 Step 1 的新开仓动作。如果 Step 0 发现 SL 丢失且市场出现做多信号，LLM 应输出 `["MODIFY_SL", "LONG"]` 而非 `["LONG"]`——先保护仓位，再开新仓。

这项设计通过 **System Prompt + User Prompt Instructions + Judge Prompt** 三层提示词共同强制，不需要额外代码逻辑。在 Debate 模式下，Judge 与辩手共享相同的原始市场数据（通过 `RAW MARKET DATA` 区块），可直接验证辩手引用的价格和指标是否准确。**Step 1.5 强制反事实检查**要求 LLM 在方向性决策前逐项回答四个检查点（失效价位→SL、ATR 校验、15 分钟反转扫描、偏差审计），而非模糊的"反问自己"——降低过度自信导致的重仓。**周期间 diff** 自动注入每轮 prompt，标注实际经过的时间（因为 LLM 自主控制唤醒间隔），让 LLM 聚焦变化而非重新分析全量数据。

**提示词增强系统**（v3.0）在 LLM 决策前注入七类附加上下文：⏰ 时区/日历/交易时段、📌 持仓记忆（原始开仓逻辑 + MAE/MFE）、🔄 上周期决策反馈闭环、📊 波动率状态分类（HIGH/NORMAL/LOW）及止损宽度指导、🧮 期望值框架（R:R → 盈亏平衡胜率）、📚 近期交易教训总结、⚠️ 硬约束末尾重复（首因+近因效应）。详见 [提示词增强系统](#提示词增强系统)。

## 架构概览

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
│  DataProvider │───▶│    策略引擎      │───▶│  RiskManager │───▶│  TradeExecutor   │
│  (Hyperliquid)│    │ single / debate │    │  (七层风控)   │    │  (全订单生命周期)  │
└──────────────┘    └─────────────────┘    └──────────────┘    └──────────────────┘
      ▲                      │                                           │
      │                      ▼                                           │
      │              ┌──────────────┐                                    │
      └──────────────│ TradeLogger  │◀───────────────────────────────────┘
                     │ (盈亏反馈)    │
                     └──────────────┘
                           │
┌──────────────────┐       │       ┌──────────────────┐
│   OrderMonitor   │───────┼──────▶│  FlashReporter   │
│  (WebSocket 实时) │  tracker     │  (Flash LLM 通知) │
│  订单/成交/清算   │  同步        │  中文推送消息      │
└──────────────────┘               └──────────────────┘
         │                                  │
         └────── 微信/飞书推送 ◀────────────┘
```

### 数据流

1. **DataProvider** — 行情数据走主网（`meta_and_asset_ctxs`，含 mark/oracle/funding/OI），多周期 K 线 TTL 缓存。同步查询链上全部挂单（`open_orders`）。**周期间 diff**：自动对比上一轮快照，注入 `# 📊 Since Last Cycle (X.Xmin ago)` 区块——因为 LLM 自主控制唤醒间隔，diff 标注实际耗时让 LLM 正确评估变化幅度
2. **策略引擎** — Single：单次 LLM 分析；Debate：三 Agent 辩论 + Judge 裁决（60s 超时），可选**反驳轮**（辩手互驳后再裁决）。所有 Agent 接收相同的市场数据（行情、多周期 K 线、订单簿、资金费率、风控约束、周期间 diff）；Judge 额外看到三方论证后可交叉验证。两种模式均遵循 Step 0→Step 1→Step 1.5 决策工作流
3. **RiskManager** — 七层风控校验 + `validate_sequence()` 多操作状态模拟（支持翻转 CLOSE+LONG/SHORT 风控通过）。**风控拒绝 → LLM 一次更正**：拒绝原因反馈给原始决策者，修正后重新验证，通过则执行（可通过 `RISK_CORRECTION_ENABLED` 关闭）
4. **TradeExecutor** — 启动恢复 + resting/active 状态机 + SL/TP 价格追踪 + 多操作顺序执行（失败即停）+ 15/15 SDK 全覆盖
5. **TradeLogger** — 盈亏分析 + LLM 表现反馈（自省循环）
6. **OrderMonitor** — WebSocket 实时订阅订单状态变化（成交/部分成交/取消/清算），毫秒级同步到 PositionTracker，使 LLM 在下一周期看到最新状态
7. **FlashReporter** — 消费 WS 事件 → Flash 模型生成中文自然语言通知 → 微信/飞书推送
8. **上下文注入** — 每轮将以下信息注入 LLM prompt：
   - **时区/日历**：UTC/北京时间、星期、交易时段（亚/欧/美盘），周末低流动性警告
   - **持仓记忆**：原始开仓逻辑（entry thesis）、持仓时长、最大有利/不利偏移（MAE/MFE）
   - **上周期反馈**：上个周期的决策→结果闭环（"你上次说要等回踩——现在回踩了吗？"）
   - **波动率状态**：HIGH/NORMAL/LOW 分类 + 止损宽度/仓位大小指导
   - **期望值框架**：R:R → 隐含盈亏平衡胜率，LLM 知道 confidence 需要超过多少
   - **近期教训**：从已平仓交易中提取的模式（重复止损、单边胜率低、TP 后趋势延续）
   - **账户/风控**：账户余额、持仓、全部链上挂单（含孤儿订单）、tracker 追踪的 SL/TP 价格、风控硬约束（熔断状态、保证金预算、风险预算、方向限制）——LLM 做决策前就知道边界在哪
   - **硬约束末尾重复**：最大仓位/最低置信度/SL 要求/杠杆在 prompt 末尾再次出现（首因+近因效应）
9. **自适应间隔** — LLM 建议下次唤醒时间（5min-3h），横盘省费/关键位盯紧

### 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | Kimi K3 (主) + DeepSeek V4 (自动降级备份) |
| LLM 编排 | LangChain + LangGraph StateGraph |
| 状态持久化 | LangGraph MemorySaver + JSONL 文件（fcntl 锁） |
| 交易所 | Hyperliquid (Perpetual DEX) |
| 实时监控 | Hyperliquid WebSocket + deepseek-v4-flash (轻量汇报) |
| 结构化输出 | Pydantic + LangChain json_mode (response_format: json_object) |
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

# 方案 3: 辩论模式 — 辩手弱模型 + Judge 强模型 (🏆 推荐)
# 辩手执行定向搜索（只需找证据），Judge 做综合裁决（需要强推理）
PRIMARY_LLM=deepseek        # Bull/Bear/Hold 用便宜的 DeepSeek（3 次调用）
JUDGE_PRIMARY_LLM=kimi       # Judge 用强推理的 Kimi K3（1 次调用）

# 方案 4: 仅 Kimi (不配 DeepSeek key 即可)
# 方案 5: 仅 DeepSeek (不配 Kimi key 即可)
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

> **结构化 JSON 输出**：使用 `response_format: json_object`（LangChain `json_mode`）。这是 DeepSeek 官方唯一支持的结构化输出方式（`json_schema` 和 `function_calling` 均返回 400）。Kimi 也兼容。Schema 通过 system prompt 传递给模型。
>
> ⚠️ **Kimi K3 温度限制**：Kimi K3 是推理模型，**只接受 `temperature=1`**。程序会自动将 `LLM_TEMPERATURE` 覆盖为 1.0（仅对 Kimi），无需手动修改配置。DeepSeek 不受影响。

### Debate 模式：独立 Judge 模型（辩弱 Judge 强）

Debate 模式下，4 个 Agent 的认知负荷不对称：

| | 🐂🐻😐 辩手 (×3) | ⚖️ Judge (×1) |
|---|---|---|
| **任务** | 单一视角定向搜索（只看多/只看空/只看风险） | 综合三方论证 + 对照原始数据 + 多时间框架权衡 + 账户约束 → 最终决策 |
| **对偏差的敏感度** | 故意有偏（角色设定），偏差是 feature | 必须识别并抵消辩手偏差 |
| **出错代价** | 低 — 一个辩手弱，另两个可补充 | 高 — Judge 错 = 交易决策错 |
| **适合的模型** | 便宜快速的弱模型 | 强推理模型 |

**推荐配置**：辩手用便宜的 DeepSeek，Judge 用强推理的 Kimi K3。

```bash
# .env
PRIMARY_LLM=deepseek        # Bull/Bear/Hold 用 DeepSeek（3 次调用）
JUDGE_PRIMARY_LLM=kimi       # Judge 用 Kimi K3（1 次调用）
```

容灾：Judge 的 Kimi 挂了 → 自动降级到 DeepSeek。辩手不受影响。

费用对比（10min 间隔，Debate 模式月成本）：

| 方案 | 辩手×3 | Judge×1 | 月成本 | 决策质量 |
|------|--------|---------|--------|:--:|
| 全 Kimi | Kimi | Kimi | ~¥450 | 高 |
| 全 DeepSeek | DeepSeek | DeepSeek | ~¥60 | 中 |
| **辩弱 Judge 强** | DeepSeek | Kimi | **~¥160** | **高** ✅ |

比全 Kimi 省 **65%**，Judge 决策质量不受影响——最终裁决依赖推理综合能力而非辩手的文笔。

启动日志会体现独立配置：

```
LLM: deepseek primary → fallback: kimi        ← 辩手
Judge LLM: kimi primary → fallback: deepseek   ← Judge
```

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

每轮 prompt 自动注入**周期间 diff**（`# 📊 Since Last Cycle (X.Xmin ago)`），标注实际经过的时间。因为间隔由 LLM 自主决定——5 分钟前和 2 小时前的同一个 +0.26% 含义完全不同——diff 让 LLM 正确评估变化的显著性。

### 边界保护

程序内置钳制：`[MIN_INTERVAL, MAX_INTERVAL]`，默认 `[300s, 10800s]`（5 分钟 ~ 3 小时），可通过环境变量调整。LLM 提示词也已引导优先使用较长间隔——只有已持仓或确认突破时才建议 300-600s。

### 零配置

不需要任何新参数。LLM 通过系统提示词知道允许的范围，自动在每次响应中建议合适的间隔。启动日志会显示间隔变化：

```
Cycle 5 complete: signal=HOLD confidence=0.65
LLM adjusted interval: 600s → 900s    ← 横盘拉长
Sleeping 900.0s until next cycle...
```

## DeepSeek 上下文硬盘缓存

DeepSeek API 对所有用户默认开启**上下文硬盘缓存**（官方文档），无需修改代码即可享用。

### 缓存机制

系统通过三种方式落盘缓存前缀单元，后续请求匹配完整单元即可命中：

| 机制 | 触发条件 | 本项目利用方式 |
|------|---------|--------------|
| **请求结束位置** | 每次请求完成 | —（Debate 辩手 user message 不同，无法直接命中） |
| **公共前缀检测** | 多次请求共享前缀 | Single 模式：2 周期后 `SYSTEM_PROMPT` 成为公共前缀 |
| **固定 token 间隔** | 长输入按间隔截取 | **Debate 周期内命中核心机制**：Hold 先跑 → 固定间隔单元落盘 → Bull/Bear 命中 system message 部分 |

### Debate 模式缓存预热

Debate 模式的 Phase 1（Hold 先跑）→ Phase 2（Bull + Bear 并行）设计利用了固定间隔缓存：

```
Cycle N:
  Phase 1: Hold ──▶ 固定间隔缓存单元写入磁盘（~2000 token system message）
                      │
                      ▼ (~2s 落盘延迟)
  Phase 2: Bull ──▶ 命中缓存（仅 ~50 token user message 计费）  ┐
           Bear ──▶ 命中缓存（仅 ~50 token user message 计费）  ┘ 并行
```

**关键发现**：DeepSeek 缓存落盘耗时"秒级"（官方文档）。如果 Hold 响应很快（<1 秒），Bull/Bear 可能在缓存就绪前发出请求 → 全部 token 按全价计费。

`CACHE_WARMUP_DELAY`（默认 2.0 秒）在 Hold 完成后等待缓存落盘，确保 Bull/Bear 稳定命中。

### 缓存命中监控

程序在 Debate 辩手和反驳 Agent 的每次 LLM 调用后记录缓存命中率：

```
Cache [Bull]: hit=1950 miss=50 input=2000 rate=97.5%
Cache [Hold]: hit=0 miss=2050 input=2050 rate=0.0%
```

- `hit=0` → 该请求是"冷启动"（预热阶段），符合预期
- `hit` 接近 input → 缓存命中良好
- 连续多轮 Bull/Bear 的 hit=0 → 检查 `CACHE_WARMUP_DELAY` 是否太小

> **注意**：缓存日志仅在**非结构化输出**的 LLM 调用中可用（Debate 辩手、反驳 Agent）。Single 模式和 Judge（使用 `json_mode` 结构化输出）的缓存数据暂不采集——但 Single 模式每周期仅 1 次调用，无周期内共享需求；Judge 的 system prompt 在第 3 周期起通过公共前缀检测自动命中。

### 周期内缓存收益

| 模式 | 周期内调用 | 缓存命中 | 有效输入 token | 缓存折扣 token |
|------|----------|---------|---------------|---------------|
| Single | 1 次 | 0（无共享对象） | ~4000 | 0 |
| Debate（无反驳） | 4 次 | 2/4（Bull + Bear 命中） | ~7150 | ~3900 (55%) |
| Debate（有反驳） | 7 次 | 4/7（Bull + Bear ×2） | ~10650 | ~7800 (73%) |

> **周期内共享**（同一 market data 被多个 Agent 使用）是主要收益来源。**跨周期共享**（system prompt 公共前缀）收益较小——因为行情数据每次都变，只有固定指令部分可以跨周期缓存。

## 消息推送（微信 / 飞书）

通过 larky 的 `UnifiedClient.notify()` 将消息发布到 Redis Pub/Sub，由 larky 的 `UnifiedService`（独立进程 `python -m larky`）投递到各平台（微信/飞书/QQ）。多程序共享同一套基础设施，无需各自管理 Bot 登录。

### 架构

```
kimi_quant ──UnifiedClient──▶
                            │
cryptoguard ──UnifiedClient──▶── bot:outgoing ──▶ UnifiedService ──▶ 微信/飞书/QQ
                            │     (Redis Pub/Sub)     (larky 独立进程)
其他程序   ──UnifiedClient──▶
```

### 依赖

- `larky`（可编辑安装，已在 `pyproject.toml` 中）
- `redis`（larky 的传递依赖，已显式声明确保安装）
- `UnifiedService` 独立运行（`python -m larky`），各程序共享

### 推送事件

| 事件 | 消息 | 来源 |
|------|------|------|
| 🚀 启动 | 模式、模型、间隔 | 主循环 |
| ❌ 启动失败 | 错误详情 | 主循环 |
| 📈 开仓 | 方向、仓位、入场价、SL/TP、置信度、账户余额 | 主循环 |
| 🟢/🔴 平仓 | 盈亏金额、百分比、平仓原因、账户余额 | 主循环 |
| 🛡️ 风控拒绝 | 拒绝原因（置信度不足/熔断/止损太近等） | 主循环 |
| 🔄 风控修正 | 正在要求 LLM 修正被拒信号 + 截断的拒绝原因 | 主循环 |
| ✅ 修正通过 | 修正后信号通过风控（显示新旧 action 对比） | 主循环 |
| ❌ 修正失败 | 修正后仍被拒的原因，放弃本轮 | 主循环 |
| ⚠️ 熔断 | 连续亏损次数、冷却周期、累计盈亏 | 主循环 |
| ✅ 订单成交 | 入场成交 / 止盈止损触发 | **OrderMonitor (实时)** |
| ⏳ 部分成交 | 成交进度百分比、价格 | **OrderMonitor (实时)** |
| ❌ 订单取消/被拒 | OID、原因 | **OrderMonitor (实时)** |
| 💀 仓位清算 | 清算价格、数量 | **OrderMonitor (实时)** |
| ⚠️ 异常 | 首个错误 + 每 10 轮（防刷屏） | 主循环 |
| ⏹️ 停止 | 总周期、交易数、胜率、盈亏 | 主循环 |

### 自动检测

```
larky 可导入 → UnifiedClient 发送（需 UnifiedService 运行中）
有飞书 APP_ID → 飞书推送（降级方案）
都没有        → 静默运行

Redis 配置（可选，默认 localhost:6379）：
  REDIS_HOST=localhost
  REDIS_PORT=6379
  REDIS_DB=0
```
```

程序启动时自动 ping Redis，连通即走通知通道。发送失败自动重连，不会因 Redis 临时重启而永久静默。`priority="high"` 确保离线消息不丢失（Redis 队列暂存，恢复后补发）。

## 订单实时监控

大模型下单后，订单可能在周期之间的任意时刻成交（尤其是市价单几秒内即成交，SL/TP 可能在数小时后触发）。如果只依赖每周期（默认 600s）的链上同步，你可能在 10 分钟后才知道订单已成交。

**OrderMonitor + FlashReporter** 解决了这个问题：通过 Hyperliquid WebSocket **实时**订阅订单状态变化，毫秒级同步到持仓追踪器，并用便宜的 Flash 模型生成中文推送通知。

### 架构

```
Hyperliquid WebSocket
  ├── orderUpdates (订单状态变化)
  └── userFills     (成交明细)
         │
         ▼
   OrderMonitor (后台线程)
    ├── 解析事件 → OrderEvent
    ├── apply_ws_event() → PositionTracker (即时同步状态)
    └── 入队 → queue.Queue (线程安全)
         │
         ▼
   FlashReporter (后台线程)
    ├── 消费事件
    ├── Flash LLM 生成中文通知 (deepseek-v4-flash)
    │    失败时降级为确定性格式化
    └── Notifier → 微信/飞书推送
```

### 与主循环的协作

```
主循环 (每 600s)                     Monitor (实时)
     │                                    │
     ├── LLM 决策                          │
     ├── risk.validate()                   ├── WS: 订单成交!
     ├── executor.execute() 下单           │   ├── tracker.apply_ws_event()
     │                                    │   │   resting → active
     │                                    │   └── FlashReporter → 推送通知
     ├── sleep(600s)                       │
     │                                    ├── WS: SL 触发!
     │                                    │   ├── tracker.clear()
     │                                    │   └── FlashReporter → 推送通知
     ▼                                    │
  下一周期                                 ▼
  tracker state 已是最新 → LLM 看到实时状态
```

### 追踪的事件类型

| WS 事件 | tracker 状态变化 | 推送通知（Flash LLM 生成） |
|---------|-----------------|--------------------------|
| 入场订单成交 | `resting → active` | `✅ 订单成交 #12345 多 0.0100 BTC @ $67200` |
| 部分成交 | 记录日志，等待完全成交 | `⏳ 部分成交 40% (0.004/0.01 BTC) @ $67150` |
| 止损触发 | `clear tracker` | `🛑 止损触发 #12346 @ $66800` |
| 止盈触发 | `clear tracker` | `🎯 止盈触发 #12347 @ $69100` |
| 订单取消 | 清除对应 oid | `❌ 订单已取消 #12348` |
| 订单被拒 | 清除 tracker | `🚫 订单被拒 #12349` |
| 仓位清算 | 清除 tracker | `💀 仓位被清算 @ $70000` |

### 线程安全

`PositionTracker` 内置 `threading.Lock`，主循环和 Monitor 后台线程并发访问互斥：

- **主线程**：`sync_with_chain()`、`execute()`、`to_summary()` 等读写
- **Monitor 线程**：WebSocket 回调中调用 `apply_ws_event()` 写入

所有公开的 mutation 方法（`clear()`、`update_from_open()`、`confirm_active()`、`tick_resting()`、`apply_ws_event()`）均持有锁。

### Flash 模型降级保护

如果 Flash LLM API 调用失败（网络超时、余额不足、key 失效），`FlashReporter` 立即切换到**确定性格式化**模式——按照固定模板生成通知文本。通知不会丢失，只是失去了自然语言的灵活性。

**降级后的通知示例**：
```
✅ 订单成交 #12345
多 0.0100 BTC @ $67200.0
```

### 配置

```bash
# .env
MONITOR_ENABLED=true                     # 开启实时监控（默认开启）
MONITOR_FLASH_MODEL=deepseek-v4-flash    # Flash 模型（便宜快速）
# MONITOR_FLASH_API_KEY=                 # 留空则复用 DEEPSEEK_API_KEY
# MONITOR_FLASH_BASE_URL=                # 留空则复用 DEEPSEEK_BASE_URL
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MONITOR_ENABLED` | `true` | 开启/关闭实时监控（dry-run 模式下自动禁用） |
| `MONITOR_FLASH_MODEL` | `deepseek-v4-flash` | Flash 模型名称。约 ¥0.14/1M tokens，<1s 延迟 |
| `MONITOR_FLASH_API_KEY` | 同 `DEEPSEEK_API_KEY` | Flash 模型 API Key。不配则复用 DeepSeek key |
| `MONITOR_FLASH_BASE_URL` | 同 `DEEPSEEK_BASE_URL` | Flash 模型 API 端点 |

> **费用极低**：一次通知约 100-200 input tokens + 30-50 output tokens。DeepSeek V4 Flash 定价约 ¥0.14/1M input。即使每天 100 次通知，月成本不到 **¥0.01**。

### 启动日志

```
OrderMonitor started (address=0xAeFB...)
WebSocket subscribed: orderUpdates(#1) + userFills(#2)
Order monitor active (flash_model=deepseek-v4-flash, llm=enabled)
FlashReporter LLM ready: deepseek-v4-flash
```

运行时 WS 事件同步日志：
```
WS sync: entry #12345 filled @ 67200.0 (state: resting→active)
WS → tracker synced: entry_filled oid=12345
FlashReporter sent: ✅ 订单成交 #12345...
```

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

### 保持 .env 与 .env.example 同步

`.env.example` 会随版本迭代增加新配置项。`git pull` 后发现多了东西，用这条命令快速看差异（自动过滤密钥行）：

```bash
diff <(grep -vE '^(MOONSHOT|DEEPSEEK|HYPERLIQUID_PRIVATE)_' .env) .env.example | grep '^>'
```

输出的是 `.env.example` 有、你的 `.env` 没有的行。看到需要的就复制过去——大部分新配置有默认值，不追也不会出错。

也可以加到 shell alias 方便反复用：

```bash
alias env-diff='diff <(grep -vE "^(MOONSHOT|DEEPSEEK|HYPERLIQUID_PRIVATE)_" .env) .env.example | grep "^>"'
```

### 跑起来

```bash
# 单次分析（验证环境正常）
uv run kimi-quant --once

# 启动！一个命令，一直跑，无需 cron
uv run kimi-quant
#   ↓ 程序内部自己循环：
#     获取行情 → 问 LLM → 风控 → 执行 → LLM 说睡多久就睡多久 → 醒来重复

# 另一终端，随时查看状态
uv run kimi-quant --status     # 账户余额、持仓、挂单、行情
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
uv run kimi-quant --status               # 实时账户状态
uv run kimi-quant --stats                # 交易盈亏
```

**检查清单**：
- [ ] 启动日志显示 `TradeExecutor initialized (address=0x... testnet=True)`
- [ ] 启动日志显示 `OrderMonitor started` + `WebSocket subscribed`（实时监控已激活）
- [ ] 仓位追踪正确（`Position: [ACTIVE] LONG 0.0010 BTC @ $...`）
- [ ] SL/TP 订单正常创建
- [ ] CLOSE 信号正常平仓
- [ ] 风控熔断机制正常触发
- [ ] `--status` 显示与链上一致的持仓/挂单数据
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
uv run kimi-quant --status                            # 账户+持仓
watch -n 60 'uv run kimi-quant --stats'               # 盈亏刷新
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
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | 模型名称 |
| **LLM 参数** | | |
| `PRIMARY_LLM` | `kimi` | 主模型：`kimi` 或 `deepseek` |
| `REASONING_EFFORT` | `max` | 推理强度：`max`/`high`/`medium`/`low`/`minimal`/`off` |
| `LLM_TEMPERATURE` | `0.1` | LLM 温度 (0-2)。**注意**：Kimi K3 只支持 1.0，程序自动覆盖 |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token（不影响 1M 上下文输入） |
| `JUDGE_TEMPERATURE` | `0.05` | Debate 模式 Judge 温度 |
| `JUDGE_PRIMARY_LLM` | (同 `PRIMARY_LLM`) | Judge 专用主模型：`kimi` 或 `deepseek`。留空则与辩手相同。推荐 `kimi`（强推理裁决） |
| `DEBATE_REBUTTAL_ENABLED` | `false` | 开启反驳轮：辩手互相反驳后再由 Judge 裁决（+3 次 LLM 调用/周期） |
| `CACHE_WARMUP_DELAY` | `2.0` | Debate 模式缓存落盘等待秒数。增大确保 Bull/Bear 命中缓存，设 0 关闭。仅影响时序，不影响决策质量 |
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
| **风控修正** | | |
| `RISK_CORRECTION_ENABLED` | `true` | 风控拒绝后给 LLM 一次更正机会（设为 `false` 关闭） |
| **订单监控 (实时 WebSocket + Flash LLM)** | | |
| `MONITOR_ENABLED` | `true` | 开启实时订单状态监控 |
| `MONITOR_FLASH_MODEL` | `deepseek-v4-flash` | Flash 模型（轻量汇报） |
| `MONITOR_FLASH_BASE_URL` | 同 `DEEPSEEK_BASE_URL` | Flash 模型 API 端点 |
| `MONITOR_FLASH_API_KEY` | 同 `DEEPSEEK_API_KEY` | Flash 模型 API Key（留空复用） |
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

# 实时状态查询（可安全地与运行中的程序并发）
uv run kimi-quant --status                     # 账户余额、持仓、挂单、行情
uv run kimi-quant --stats                      # 查看盈亏统计（真实+模拟）
uv run kimi-quant --history                    # 查看辩论历史记录
```

### `--status` 实时账户状态

直接查询 Hyperliquid 链上数据，输出四个维度的信息：

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

| 区域 | 数据来源 | 说明 |
|------|---------|------|
| 💰 Balance | `user_state.marginSummary` | 总资产、可用余额、保证金占用 |
| 📊 Position | `user_state.assetPositions` | 多空方向、仓位、开仓价、标记价、浮动盈亏、杠杆 |
| 📝 Orders | `open_orders` | 所有挂单（限价单、SL、TP），含 OID、类型、价格 |
| 📈 Market | `all_mids` + `meta_and_asset_ctxs` | 中间价、24h 涨跌、资金费率、未平仓量 |

所有数据直接从链上查询，不经本地 tracker——是客观真实数据。

> **订单类型推断**：Hyperliquid 的 `openOrders` API 不返回 `orderType` 字段。`--status` 通过上下文自动推断：比较订单价格与持仓入场价/市场价，判断是 Limit Entry、Stop Loss 还是 Take Profit。

可在终端持续刷新：

```bash
watch -n 5 uv run kimi-quant --status
```

## 账户管理

Hyperliquid 账户管理工具，覆盖入金、划转、账户类型切换全流程。

```bash
# 查询 Arbitrum 链上 USDC/ETH 余额
uv run kimi-quant --arb-balance

# 设置账户类型（手动确认，或加 --force 跳过）
uv run kimi-quant --set-account-type manual

# 现货 → 合约账户划转（手动确认）
uv run kimi-quant --spot-to-perp 16

# 从 Arbitrum 入金到 Hyperliquid（⚠️ 需手动输入 YES 确认，web3 工具尚未充分验证）
uv run kimi-quant --deposit 100
```

### 账户类型

Hyperliquid 支持三种账户模式：

| 模式 | CLI 参数 | 特点 | 推荐 |
|------|---------|------|:--:|
| Manual | `--set-account-type manual` | spot/perp 独立余额，SDK 完全兼容 | ✅ 推荐 |
| Unified | `--set-account-type unified` | 统一余额管理，但 SDK 不支持 spot→perp 划转 | |
| Portfolio Margin | `--set-account-type portfolio` | 跨资产共用保证金，复杂度高 | |

切换到 Manual 后即可用 `--spot-to-perp` 程序化划转资金。

### 完整入金流程

```
交易所买 USDC → 提币到 Arbitrum 链（你的 0x... 地址）
  → uv run kimi-quant --arb-balance          # 确认到账
  → uv run kimi-quant --set-account-type manual  # 切到 Manual 模式
  → uv run kimi-quant --spot-to-perp 100     # 划转到合约账户
  → uv run kimi-quant                        # 启动交易
```

> **注意**：首次入金前确认账户类型。Unified Account 需要用网页或 `--set-account-type manual` 切换后才能用 SDK 划转。Hyperliquid 桥接最低 5 USDC，低于这个金额会丢失。`--deposit` 工具尚未经过充分验证，推荐使用官方网页进行 Arbitrum→Hyperliquid 跨链操作。

## TradingSignal

### LLM 输出格式（推荐：多操作 `actions`）

```json
{
  "actions": ["CLOSE", "SHORT"],
  "confidence": 0.85,
  "reasoning": "1h 趋势反转确认，先平多仓再开空...",
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

> **`actions` 优先**：LLM 应输出 `actions` 数组。单个操作用 `["LONG"]`，翻转仓位用 `["CLOSE", "SHORT"]`，调整止盈止损用 `["MODIFY_SL", "MODIFY_TP"]`。仍支持旧格式 `action` 字符串以保证向后兼容。

> `next_interval`：LLM 建议的下次唤醒秒数（60-10800），null 则用默认值。上例中 120s 表示开仓后紧盯。

### Action 说明

| Action | 触发条件 | 订单操作 |
|--------|---------|---------|
| `LONG` | 看多信号 | 开多仓 + SL + TP（bulk_orders 原子执行） |
| `SHORT` | 看空信号 | 开空仓 + SL + TP（bulk_orders 原子执行） |
| `CLOSE` | 平仓信号 | 市价平仓 |
| `HOLD` | 观望/不确定 | 无操作 |
| `MODIFY_SL` | 移动止损 | 将 SL 移至新价格（保本/追踪止损） |
| `MODIFY_TP` | 移动止盈 | 将 TP 移至新价格（调整目标） |

### 多操作组合（`actions` 数组）

单个周期可执行有序操作序列，执行器按顺序执行，失败即停：

| 场景 | `actions` | 说明 |
|------|-----------|------|
| 翻转仓位 | `["CLOSE", "SHORT"]` | 先平多仓，再开空仓 |
| 调整止盈止损 | `["MODIFY_SL", "MODIFY_TP"]` | 同时移动 SL 和 TP |
| 单操作 | `["LONG"]` | 等价于旧格式 `action="LONG"` |

**字段说明**：
- `actions`: **推荐使用**。有序操作列表，执行器按序执行，遇失败停止后续操作
- `action`: 旧格式（仍支持），当 `actions` 为 null 时使用
- `entry_price`: 设为具体价格 → 限价单；设为 `null` → 市价单（Ioc）
- `stop_loss`: **强制字段**，LONG/SHORT 时必须提供，且距入场价 ≥ 0.5%
- `size`: `null` 时自动使用 `MAX_POSITION_SIZE`
- `modify_sl_to`: 仅 MODIFY_SL 时使用，指定新的止损价格
- `modify_tp_to`: 仅 MODIFY_TP 时使用，指定新的止盈价格

### Cycle Status

每个 cycle 结束时输出状态：

| Status | 含义 |
|--------|------|
| `executed` | 下单/平仓/移止损成功 |
| `hold` | LLM 决定观望，有意不操作 |
| `rejected` | 风控拦截（置信度不足/保证金不够/止损太近等） |
| `failed` | 执行异常（网络错误、API 超时等） |
| `skipped` | LLM 未返回有效信号 |

启动日志中会打印当前风控参数，方便确认配置是否正确加载：

```
Risk: min_confidence=0.65 | max_position=0.0010 BTC | max_leverage=3x
```

## 风控规则

### 风控上下文注入（Proactive Risk Context）

**问题**：传统架构中，风控规则在 LLM 输出决策**之后**才检查。如果 LLM 提议了一个注定被拒的操作（如熔断期间开仓、保证金不足、风险金额超限），这一轮 LLM 调用就白费了——花了 token、等了延迟、错过了时间窗口。

**解决方案**：每轮 LLM 决策**之前**，将当前风控约束动态注入 prompt。LLM 在生成 TradingSignal 时已经知道：

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

**效果**：

| 之前 | 之后 |
|------|------|
| LLM 不知道熔断状态，提议 LONG → 被拒 | LLM 看到熔断激活，不提议新开仓 |
| LLM 不知道保证金公式，size 过大 → 被拒 | LLM 自行计算 max notional，size 在预算内 |
| LLM 不知道 2% 风险上限，SL 太宽 → 被拒 | LLM 自己算 `|entry-SL|×size`，确保 ≤ 2% |
| LLM 不知道方向限制，重复开仓 → 被拒 | LLM 看到当前持仓方向，避免无效操作 |

**实现**：`RiskManager.get_risk_context()` 根据当前账户余额、市价、持仓方向动态生成约束文本，注入到 `DataProvider.build_llm_prompt()` 的 prompt 中。所有动态值（保证金预算、风险预算、最大仓位）都用实际数字计算好，LLM 不需要自己推导公式。

### 多层防护

| 层级 | 检查项 | 规则 |
|------|--------|------|
| 1 | **熔断机制** | 连续 4 笔亏损 → 暂停 6 个 cycle；日回撤 > 5% 冻结；cooldown 内不延长 |
| 2 | **置信度** | >= `MIN_CONFIDENCE` (默认 0.7) 才执行方向性交易 |
| 3 | **仓位上限** | 不超过 `MAX_POSITION_SIZE` |
| 4 | **保证金需求** | `size × price / leverage` ≤ 可用余额的 95%，超出则拒绝并建议合理 size |
| 5 | **风险金额** | 单笔止损亏损 > 1% 账户警告，> 2% 拒绝（`\|entry - SL\| × size`） |
| 6 | **止损距离** | ≥ 0.5% 距入场价（BTC 噪音 ~0.3%，低于此阈值拒绝） |
| 7 | **方向** | 已有同向仓位拒绝；CLOSE/MODIFY_SL/MODIFY_TP 需已持仓；翻转（CLOSE+LONG/SHORT）通过 `validate_sequence()` 模拟状态转换 |
| — | **SL/TP 链上验证** | 每周期 LLM 调用前交叉对比 tracker oid 与链上 `open_orders`，丢失时 prompt 告警 + 推送通知 |
| — | **多操作失败即停** | 序列中任一非 HOLD 操作失败，立即停止后续操作，防止半完成状态 |

### 风控拒绝反馈修正（Risk Correction）

**问题**：风控拒绝后，信号被直接丢弃。但如果只是参数不合适（止损太近、仓位稍大），LLM 完全可以调整后重新提交——不需要等到下一个周期。

**解决方案**：风控拒绝后，系统将**拒绝原因**反馈给做出该决策的 LLM（或 Debate 的 Judge），给一次更正机会。

```
LLM 决策 ──▶ 风控校验 ──失败──▶ 通知 "🔄 正在要求 LLM 修正..."
                │                      │
                │                      ▼
                │              构建修正提示词：
                │              - 原始信号（action/entry/SL/TP/confidence）
                │              - 拒绝原因（精确的错误描述）
                │              - "这是本轮唯一一次更正机会"
                │                      │
                │                      ▼
                │              LLM 重新决策 ──▶ 风控二次校验
                │                                 │
                │                    ┌────────────┴────────────┐
                │                    ▼                         ▼
                │                 通过                      失败
                │              "✅ 修正通过"            "❌ 修正失败"
                │              执行修正版                放弃本轮
                │
                ▼
              通过 ──▶ 正常执行
```

**提示词设计原则**：

修正提示词给出具体选项而非笼统的"别犯错"：

```
🔧 Signal Adjustment Required

[原始信号详情 + 拒绝原因]

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

**各类拒绝的 LLM 应对方针**：

| 拒绝类型 | 可修正？ | LLM 应做 |
|---------|:---:|------|
| 止损太近（< 0.5%） | ✅ | 放宽 SL 距离 |
| 止损太远（> 10%，警告） | ✅ | 收紧 SL 距离 |
| 仓位超限 | ✅ | 减小 size |
| 保证金不足 | ✅ | 减小 size |
| 单笔风险 > 2% | ✅ | 减小 size 或调整 SL |
| 置信度不足 | ⚠️ | 重新评估证据 → 提升信心或 HOLD |
| 熔断激活中 | ❌ | HOLD（新开仓被阻止） |
| 日内回撤超限 | ❌ | HOLD |
| 已持有同向仓位 | ❌ | HOLD |
| 无仓位可平 | ❌ | HOLD |

**缓存友好性**：修正调用复用原始 market data 前缀——Single 模式 ~1500 token 市场数据完全命中缓存，仅 ~100 token 修正块是新增的。Debate 模式更优：Judge 的完整辩论记录（~3000+ token）命中缓存，只新增修正块，且 Skip 了 Bull/Bear/Hold 的重新辩论（省 3 次 LLM 调用）。

**配置**：

```bash
# .env
RISK_CORRECTION_ENABLED=true   # 默认开启
RISK_CORRECTION_ENABLED=false  # 关闭：风控拒绝直接丢弃（原行为）
```

**与风控上下文注入的关系**：两个机制互补——

| 机制 | 时机 | 作用 |
|------|------|------|
| 风控上下文注入 | LLM 决策**前** | 告知硬约束，降低被拒概率 |
| 风控拒绝修正 | 风控拒绝**后** | 给一次修正机会，挽救可调参数错误 |

上下文注入减少了修正的发生频率，修正作为兜底捕获漏网之鱼。

## 提示词增强系统（v3.0）

v3.0 在 LLM 决策前系统性地注入七类附加上下文，解决此前 Step 0（审视持仓）缺乏原始开仓逻辑、LLM 不知道当前交易时段、没有决策→结果闭环等问题。

### 增强上下文全景

每轮 LLM 决策前，以下信息按顺序注入 prompt：

```
┌─────────────────────────────────────────────────────────┐
│ ⏰ 时区/日历上下文                                       │
│   UTC + 北京时间、星期几、交易时段（亚/欧/美盘/周末）      │
│   周末自动出现低流动性警告                                │
├─────────────────────────────────────────────────────────┤
│ 📊 市场数据（行情、多周期 K 线、订单簿、资金费率）          │
├─────────────────────────────────────────────────────────┤
│ 📊 波动率状态分类                                        │
│   HIGH（4h ATR > 0.7%）→ 宽止损、小仓位                  │
│   NORMAL（0.3%–0.7%）→ 标准参数                          │
│   LOW（≤ 0.3%）→ 可紧止损但警惕突破扩张                   │
│   自动计算最低 SL 距离（≥ max(1.5× ATR, 0.5%)）           │
├─────────────────────────────────────────────────────────┤
│ 📌 持仓记忆（有持仓时）                                   │
│   开仓逻辑（原始 thesis）、持仓时长、入场置信度             │
│   当前浮动盈亏 + 百分比                                   │
│   最大有利偏移 (MFE) / 最大不利偏移 (MAE)                  │
│   ⚠️ Step 0 thesis 验证提示                              │
├─────────────────────────────────────────────────────────┤
│ 📋 上周期反馈                                            │
│   上个周期的决策、置信度、理由                             │
│   执行结果（已执行/被拒/观望/失败）                        │
│   "你上次说等回踩确认——现在回踩成立了吗？"                 │
├─────────────────────────────────────────────────────────┤
│ 🛡️ 风控约束                                              │
│   熔断状态、保证金预算、风险预算、方向限制                  │
│   期望值框架：R:R → 盈亏平衡胜率                          │
├─────────────────────────────────────────────────────────┤
│ 📚 近期教训（≥3 笔已平仓交易时）                          │
│   连续止损次数、各方向胜率、净盈亏趋势                     │
│   模式识别：SL 太紧？TP 后趋势延续？                      │
├─────────────────────────────────────────────────────────┤
│ 📊 历史交易表现（盈亏统计 + 最近 5 笔）                    │
├─────────────────────────────────────────────────────────┤
│ ⚠️ 硬约束重复（prompt 末尾）                              │
│   最大仓位 / 最低置信度 / SL 必须设 / 杠杆限制             │
│   熔断激活时标注"LONG/SHORT 将被拒绝"                     │
│   （近因效应——LLM 输出 JSON 前最后看到的约束）              │
└─────────────────────────────────────────────────────────┘
```

### 各增强项详解

#### 1. 时区/日历上下文

**问题**：LLM 不知道"现在是什么时候"——美东凌晨 2 点的流动性分析和欧盘时段完全相同。

**方案**：自动注入 UTC 时间、北京时间、星期几、交易时段分类：

```
# ⏰ Time Context
UTC: 2026-07-20 14:30 | Beijing: 22:30 | Monday
Session: EU/US overlap — HIGH liquidity, strongest moves
```

周末自动追加警告：`⚠️ Weekend: lower liquidity, wider spreads, higher risk of false breakouts.`

**交易时段分类**：Asia (0-7 UTC) → EU morning (7-12) → EU/US overlap (12-15, HIGH) → US (15-21) → US close/Asia open (21-24)

#### 2. 持仓记忆（Position Memory）

**问题**：Step 0 要求 LLM 评估"原始入场逻辑是否依然成立"，但 LLM 根本不知道原始入场逻辑是什么。每个周期只看到当前持仓状态，没有上下文。

**方案**：`PositionTracker` 新增字段追踪开仓时的 thesis，每周期注入：

```
# 📌 Position Memory
Holding: LONG 0.0100 BTC @ $67,200
Opened: 2h35m ago | Entry confidence: 0.82
Entry thesis: "4h breakout above $67,000 resistance with volume confirmation"
Current uPNL: +$23.50 (+0.35%)
Best: +$45.00 | Worst: -$12.00
⚠️ Step 0 — THESIS VALIDATION: Has the original entry thesis held?
```

**MAE/MFE 追踪**：每周期更新持仓期间的最大浮盈/浮亏极值，让 LLM 看到价格曾走到哪。

#### 3. 上周期反馈闭环

**问题**：LLM 上周期说 HOLD 等待确认，本周期不知道上次说了什么。没有决策→结果的学习循环。

**方案**：每周期注入：

```
# 📋 Last Cycle Feedback
Decision: HOLD (confidence=0.65) — "waiting for 15m pullback to confirm breakout"
Outcome: 15m higher low formed → pullback confirmation ✅
Status: Breakout thesis still valid, entry opportunity approaching
```

**四种反馈路径**：
- `executed` → "信号已执行，检查持仓记忆中的 thesis 是否在兑现"
- `rejected` → "被风控拒绝，不要重复相同错误"
- `hold` → "你选择观望——周期间 diff 显示情况有无变化？"
- `failed` → "执行失败，检查错误详情"

#### 4. 波动率状态分类

**问题**：虽然有 ATR 数据，但没有帮 LLM 做波动率状态判定——高波动环境和低波动环境的止损/仓位策略完全不同。

**方案**：基于 4h ATR% 自动分类并给出具体指导：

```
# 📊 Volatility Regime
4h ATR: 0.82% | 1h ATR: 0.60%
→ Regime: HIGH
Guidance: Wider stops required (≥ 1.5× ATR). Reduce position size to 50-75%.
Minimum SL distance: 0.72%
```

**三档分类**：HIGH (>0.7%) → NORMAL (0.3-0.7%) → LOW (≤0.3%)

#### 5. 期望值框架

**问题**：LLM 被要求 `confidence > 0.7`，但没有考虑 R:R——一个 R:R=1:0.5 的交易即使 confidence=0.8 也是负期望。

**方案**：注入 R:R 计算示例和盈亏平衡公式：

```
## Expected Value (EV) Check
Example: Entry=$67,200, SL=$66,500 (1.0% risk), TP=$68,500 (1.9% reward)
R:R = 1:1.9 → Breakeven win-rate = 1/(1+1.9) = 34.5%
Your confidence must EXCEED 34.5% for positive EV.
```

#### 6. 近期教训

**问题**：重复犯同样的错误（如连续在阻力位止损）——LLM 没有从历史交易中学习。

**方案**：分析最近 10 笔已平仓交易，提取模式：

```
# 📚 Recent Lessons
1. ⚠️ 2 recent stop-outs. SL may be too tight for current volatility.
2. 💡 Multiple TP hits — trend may be stronger than expected. Consider trailing stops.
3. 📉 Net P&L last 5 trades: -$45.20. Be more selective.
```

**检测模式**：连续止损 → 建议放宽 SL；多次 TP → 建议追踪止损；单边胜率低 → 建议重审入场标准；近期净亏 → 建议更保守。

#### 7. 硬约束末尾重复

**问题**：LLM 的注意力在长上下文中不均匀——中段的约束可能被忽视。

**方案**：prompt 末尾（LLM 输出 JSON 前最后看到的位置）重复关键约束：

```
# ⚠️ HARD CONSTRAINTS (repeated from above)
- Max position: 0.01 BTC | Min confidence: 0.70
- SL REQUIRED for LONG/SHORT | Min SL distance: 0.5% of entry
- Max leverage: 3x
- ⛔ CIRCUIT BREAKER ACTIVE: NEW POSITIONS BLOCKED (use HOLD/CLOSE/MODIFY only)
```

#### Debate 模式适配

所有增强上下文在 Single 和 Debate 模式中均生效。特别地：

- **辩手感知账户状态**：Bull/Bear/Hold 的共享 system message 中包含当前持仓和风控约束，辩手可做出仓位感知的论证（"在当前 0.01 BTC 多头基础上加仓"而非模糊的"做多"）
- **Judge 看到完整增强**：辩论记录 + 全部上述上下文 + 末尾约束重复

### 熔断状态机

```
正常交易 ──(连续4亏)──▶ Cooldown(6 cycles) ──(到期)──▶ 正常交易
                            │
                            │ (期间亏损不延长 cooldown)
                            │ (期间盈利自动清零)
                            │ (CLOSE/MODIFY_SL/MODIFY_TP 始终允许)
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
- 恢复挂单 ID 和价格（`open_orders` → SL/TP oid + trigger price）
- 恢复仓位自动补录 TradeLogger pending trade
- 崩溃重启后可立即管理现有仓位（平仓、移止损），且 LLM 能看到当前 SL/TP 价格

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
| ⚖️ Judge | 首席交易官 | 综合多周期趋势裁决，可交叉验证辩手论点与原始数据；1h/4h 权重 > 5m/15m |

### 决策流程

1. **Phase 1 — 缓存预热**：Hold Agent 先跑，DeepSeek 后端将其输入按固定 token 间隔落盘为缓存单元
2. **Phase 2 — 并行命中**：Bull + Bear 并发跑，两份请求的 system message（行情数据）命中 Phase 1 的固定间隔缓存单元，仅对 ~50 token 的角色指令计费。程序等待 `CACHE_WARMUP_DELAY`（默认 2s）确保缓存落盘完成
3. **Phase 3（可选）— 反驳轮**：Bull/Bear/Hold 各自看到另外两方的论证后进行反驳，指出对手逻辑漏洞或承认强论点。反驳阶段同样使用 HoldRebut 预热 → BullRebut+BearRebut 并行的缓存策略
4. **Judge 收到完整信息**：Judge 的 prompt 包含四个区块 — Account Context → Raw Market Data → Debate Transcript → **Rebuttal Round**（如有）。可交叉验证辩手引用的价格、指标、费率，并判断反驳轮中谁占了上风
5. Judge 综合原始数据与辩论论证裁决，输出结构化 TradingSignal → 风控 → 执行

### 反驳轮（可选功能）

辩论结束后、Judge 裁决前，增加一轮交叉反驳。每个辩手看到另外两方的论证后进行回应。**Judge 可以看"谁的论点在对方反驳下站住了脚"来判断哪方更可信**——而不只是比谁写得更漂亮。

```bash
# .env — 开启反驳轮（默认关闭）
DEBATE_REBUTTAL_ENABLED=true
```

费用影响：开启后每周期 7 次 LLM 调用（3 辩手 + 3 反驳 + 1 Judge），约是普通 Debate 的 1.75 倍。建议在先确定普通 Debate 策略有效后，再用反驳轮提升裁决质量。

```
debate (Bull/Bear/Hold) → rebuttal (互相反驳) → adjudicate (Judge)
```

> **前缀缓存**：三个辩手 Agent 接收完全相同的行情数据（置于 system message）。Phase 1 Hold 先跑 → DeepSeek 按固定 token 间隔落盘缓存单元 → Phase 2 Bull/Bear 命中，输入 token 费用降低 ~95%，整个 Debate 输入 token 省 ~55%。反驳轮同样受益——反驳 Agent 的 system message 也是相同的行情数据。详见 [DeepSeek 上下文硬盘缓存](#deepseek-上下文硬盘缓存)。

### Judge 决策框架

Judge 接收完整信息——账户上下文（余额、可用保证金、当前持仓、挂单 SL/TP 价格）、交易约束（最大仓位、杠杆）、**原始市场数据（行情、K 线、订单簿、资金费率——与辩手完全相同）**、三方辩论论证、以及**反驳记录**（如开启）。可在裁决时直接交叉验证辩手引用的价格和指标，并判断哪方在反驳中更站得住脚。

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
  ├── bull_rebuttal        ← Bull 反驳 (开启反驳轮时)
  ├── bear_rebuttal        ← Bear 反驳 (开启反驳轮时)
  ├── hold_rebuttal        ← Hold 反驳 (开启反驳轮时)
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
│   ├── config.py        # 配置管理（env + .env）+ 完整校验
│   ├── tls.py           # curl_cffi Firefox TLS 指纹伪装（共享模块）
│   ├── data.py          # 市场数据（Hyperliquid Info API + K线缓存 + ATR + 断线重试）
│   ├── llm.py           # TradingSignal + 双模型容灾 (Kimi/DeepSeek)
│   ├── debate.py        # Multi-Agent 辩论 + 反驳轮 + LangGraph Checkpointing
│   ├── risk.py          # 七层风控校验 + 熔断状态机
│   ├── executor.py      # 15/15 SDK 全覆盖 + 启动恢复 + PositionTracker
│   ├── monitor.py       # WebSocket 订单监控 + 崩溃自恢复 + Flash LLM
│   ├── analytics.py     # TradeLogger — 盈亏分析 + LLM 自省反馈
│   ├── notify.py        # 微信/飞书消息推送（可选，自动检测）
│   ├── deposit.py       # 入金/划转/账户类型管理（web3）
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

**第一层 — TLS 指纹伪装（curl_cffi）**：通过 `tls.py` 共享模块在导入时自动将 Hyperliquid SDK 的 HTTP 客户端替换为 curl_cffi，伪装成 Firefox 147 浏览器的 JA3 TLS 指纹。`data.py` 和 `executor.py` 共用同一套补丁逻辑，避免重复维护。选择 Firefox 而非 Chrome 是因为反爬服务对 Chrome 指纹的检测最严格（Chrome 是最常被仿冒的浏览器），Firefox 的 TLS 密码套件和扩展信号不同，不在重点盯防范围。

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
| 全 DeepSeek V3 | ~¥60 | 1.5x（缓存省 ~55% 输入 token） |
| 辩弱 Judge 强 (🏆) | ~¥160 | 1.6x（Judge 用 Kimi 保证质量） |
| 全 Kimi K3 | ~¥450 | 1.4x |

**省钱三板斧**：

| 策略 | .env 配置 | 效果 |
|------|----------|------|
| 用 DeepSeek 主力 | `PRIMARY_LLM=deepseek` | 月成本 ¥330→¥40 |
| 辩弱 Judge 强 | `PRIMARY_LLM=deepseek` + `JUDGE_PRIMARY_LLM=kimi` | 比全 Kimi 省 65%，质量不变 |
| 关推理 | `REASONING_EFFORT=off` | 输出费用再降 75% |
| 加大间隔 | `TRADING_INTERVAL=900` | 成本再降 1/3 |
| 调高最低间隔 | `MIN_INTERVAL=600` | 防止 LLM 频繁唤醒（默认 300s） |

**极限省钱方案**：`PRIMARY_LLM=deepseek + REASONING_EFFORT=off + TRADING_INTERVAL=900`，月成本约 **¥10**。

### Q: Debate 模式怎么省 token？

三个辩论 Agent 接收相同的行情数据（~1500 tokens）。系统已做多项优化：

1. **周期内前缀缓存**：行情数据放在 system message（共享前缀），Hold 先跑预热，Bull + Bear 并行命中。后两个 Agent 仅对 ~50 token 角色指令计费，周期内省 ~55% 输入 token（无反驳）或 ~73%（有反驳）。反驳轮使用相同的 HoldRebut 预热策略。缓存落盘有 `CACHE_WARMUP_DELAY` 保证（默认 2 秒）。
2. **辩弱 Judge 强**：辩手用便宜的 DeepSeek（定向搜索），Judge 用 Kimi K3（综合裁决）。比全 Kimi 省 65%，决策质量不变。见 [Debate 模式：独立 Judge 模型](#debate-模式独立-judge-模型辩弱-judge-强)。
3. **周期间 diff**：零额外 API 调用，纯数据对比——LLM 聚焦变化量而非重新分析全量数据。
4. **跨周期公共前缀**：Single 模式的 system prompt、Debate 辩手的共享系统指令在第 3 周期起自动命中公共前缀缓存。

整体效果：普通 Debate 的 4 次调用等效 ~2.4 次 Single 的输入 token 量。开启反驳轮后 7 次调用等效 ~3.5 次 Single 量。

### Q: 如何切换主/备模型？

```bash
# 全局主模型
PRIMARY_LLM=kimi      # Kimi 主力，DeepSeek 备份（默认）
PRIMARY_LLM=deepseek  # DeepSeek 主力，Kimi 备份（省钱）

# Debate 模式：Judge 独立主模型（留空则同 PRIMARY_LLM）
JUDGE_PRIMARY_LLM=kimi      # Judge 用 Kimi，辩手用 PRIMARY_LLM
JUDGE_PRIMARY_LLM=deepseek  # Judge 用 DeepSeek，辩手用 PRIMARY_LLM
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

### Q: Kimi API 返回 HTTP 400 Bad Request 怎么办？

绝大多数情况是 **温度（temperature）参数**导致的。Kimi K3 是推理模型，只接受 `temperature=1`。如果 `.env` 中设置了 `LLM_TEMPERATURE=0.1`（或其他非 1 的值），Moonshot API 会返回：
```
"invalid temperature: only 1 is allowed for this model"
```

**已自动修复**（v2.3+）：程序自动将 Kimi 实例的 temperature 强制设为 1.0，无需手动改 `.env`。启动日志会提示：`Kimi K3 requires temperature=1 (got 0.10), forcing to 1.0`。

其他可能的 400 原因：
- `reasoning_effort` 与 `response_format` 同时发送（v2.3+ 已自动处理）
- API key 格式错误（检查是否完整复制）
- 模型名拼写错误（应为 `kimi-k3`）

### Q: Kimi API 挂了怎么办？

配置 `DEEPSEEK_API_KEY` 即可。Kimi 调用失败时自动切换到 DeepSeek，无需人工干预。日志会显示 `LLM: kimi primary → fallback: deepseek`。未配置则仅用 Kimi。

### Q: 订单成交了怎么知道？有实时通知吗？

有。**OrderMonitor** 通过 Hyperliquid WebSocket 实时订阅订单状态变化，**FlashReporter** 用便宜的 Flash 模型生成中文推送通知。配置：

```bash
# .env
MONITOR_ENABLED=true                  # 开启（默认）
MONITOR_FLASH_MODEL=deepseek-v4-flash # Flash 模型
```

通知事件包括：入场成交、止损/止盈触发、部分成交、订单取消/被拒、仓位清算。如果 Flash 模型不可用，自动降级为格式化文本通知。

参见 [订单实时监控](#订单实时监控)。

### Q: 大模型做决策时能看到最新的订单/持仓状态吗？

能。每次 LLM 决策前，系统注入五类信息：

1. **链上仓位**：`DataProvider` 查询 `user_state`，获取持仓方向、大小、入场价、浮动盈亏
2. **全部链上挂单**：`DataProvider` 查询 `open_orders`，列出所有未成交订单（oid、方向、数量、价格）。LLM 能看到**所有**订单——包括上一轮遗留的孤儿订单
3. **Tracker 追踪的 SL/TP**：`PositionTracker.to_orders_summary()` 输出 bot 自己创建的 SL/TP 的 oid 和价格
4. **SL/TP 链上验证结果**：程序在 LLM 调用前交叉对比 tracker oid 与链上 `open_orders`，若发现丢失，在 prompt 中注入 ⚠️ 告警
5. **风控硬约束**（🆕）：熔断状态、保证金预算（$ 金额）、风险预算（$ 金额）、方向限制——LLM 知道**当前能做什么、不能做什么**，不会提出注定被拒的操作

更重要的是，LLM 的 **System Prompt 强制执行 Step 0**——在分析市场之前先审视已有持仓和挂单。这确保 LLM 不会跳过现状评估直接做新交易决策。

加上 **OrderMonitor** 的 WebSocket 实时同步，`PositionTracker` 在订单成交瞬间更新——大模型在下一个决策周期就能看到最新状态。

### Q: 如果上一轮运行留下了未成交的限价单怎么办？

两层保护：

1. **程序化**：限价单 3 个周期未成交自动取消（`max_resting_cycles=3`）
2. **LLM 判断**：每个周期 prompt 列出**全部链上挂单**，且 Step 0-c 要求 LLM 检查僵尸订单。LLM 可以判断这些订单是否还有效，通过 `CLOSE` 清理或在下个 `actions` 中覆盖。你也可以用 `--status` 手动查看。

### Q: SL/TP 订单会被交易所意外取消吗？如何防护？

有。三层保护：

1. **程序化验证**：每周期 LLM 调用前，`verify_tracked_orders()` 交叉对比 tracker 记录的 `sl_oid`/`tp_oid` 与链上 `open_orders`。若发现丢失 → prompt 注入 ⚠️ 告警 + 推送通知
2. **LLM 响应**：Step 0-b 要求 LLM 将 SL/TP 丢失视为最高优先级，输出 `MODIFY_SL`/`MODIFY_TP` 立即恢复
3. **WebSocket 实时感知**：若 SL/TP 被执行（成交）或被取消，`OrderMonitor` 毫秒级同步到 tracker

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
1. 服务器上运行着 `python -m larky`（UnifiedService，独立进程）
2. `redis` 包已安装（`uv sync` 自动处理）

满足条件后程序自动通过 `UnifiedClient.notify()` 推送。无需额外配置。启动日志会显示：
```
Notification: larky UnifiedClient available (via Redis Pub/Sub)
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

有七层风控 + 上下文预注入 + **拒绝修正**三重保护。即使 LLM 给出不合理信号，风控层也会拒绝。且 LLM 在决策前已经看到了当前的风控约束（熔断状态、保证金预算、风险预算），大幅降低了提出无效操作的概率。万一仍被拒绝，系统还会给 LLM 一次修正机会（见 [风控拒绝反馈修正](#风控拒绝反馈修正risk-correction)）。常见被拒绝的情况：
- 置信度 < 0.7
- 保证金需求超过可用余额的 95%
- 单笔风险 > 2% 账户
- 止损太近 (< 0.5%)
- 已有同向仓位
- 熔断激活中

### Q: 风控拒绝了会怎样？LLM 能自己修正吗？

能。风控拒绝后，系统会把**拒绝原因**（如"止损距离 0.50% 太近，至少需要 0.50%"）反馈给 LLM，给一次更正机会。LLM 根据原因调整参数（放宽止损、减小仓位等）重新提交。如果修正后通过风控，直接执行修正版；如果仍不通过或遇到硬阻断（熔断、回撤超限），放弃本轮。

通知消息会清楚标注整个过程：

```
🔄 Risk correction: asking LLM to fix — Stop loss distance 0.50% is too tight...
✅ Risk correction accepted — SHORT (was SHORT)     ← 修正成功
❌ Risk correction failed: ... Giving up.            ← 修正失败
```

可通过 `RISK_CORRECTION_ENABLED=false` 关闭此功能。详见 [风控拒绝反馈修正](#风控拒绝反馈修正risk-correction)。

### Q: 账户余额为什么显示 $0？

常见原因：
1. **资金在现货账户**：Hyperliquid 的 spot 和 perp 账户分开。用 `uv run kimi-quant --spot-to-perp <金额>` 划转。
2. **账户类型不兼容**：Unified Account 下 `usd_class_transfer` 被禁用。先用 `--set-account-type manual` 切换到 Manual 模式。
3. **不在永续合约账户**：用 `--arb-balance` 查 Arbitrum 链上余额，确认资金已跨链到 Hyperliquid。

启动日志每个 cycle 都会打印 `Account:` 行，可实时监控余额。

### Q: 如何切换账户类型？

```bash
uv run kimi-quant --set-account-type manual    # 推荐：独立 spot/perp 余额
uv run kimi-quant --set-account-type unified   # 统一账户
uv run kimi-quant --set-account-type portfolio # 组合保证金
```

切换后等几秒生效，再用 `--spot-to-perp` 划转。

### Q: 如何把 USDC 从 Arbitrum 划转到 Hyperliquid？

推荐用 Hyperliquid 官方网页 Deposit。CLI 工具 `--deposit` 提供了 web3 方式但尚未充分验证，使用时需要手动输入 `YES` 确认。

### Q: 运行中怎么查看状态？

```bash
# 实时账户 + 持仓 + 挂单（直接查询链上数据）
uv run kimi-quant --status

# 实时刷新（每 5 秒）
watch -n 5 uv run kimi-quant --status

# 盈亏统计
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
