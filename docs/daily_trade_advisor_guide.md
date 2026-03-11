# Daily Trade Advisor 使用指南

## 📋 概述

`Daily Trade Advisor` 是一套集成"量化模型评分 + 双引擎 AI 审计 + 大盘择时"的实战决策系统，严格执行"宁缺毋滥"原则，为 5 万元小额本金提供每日操作指令。

## 🏗️ 系统架构

### 三层过滤系统

```
┌─────────────────────────────────────────────────────────────┐
│                    Daily Trade Advisor                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 量化门槛与 Top 10 提取                               │
│  - 加载模型并推理                                            │
│  - 筛选 P >= 70% 的股票                                       │
│  - 取 Top 10 进入审计环节                                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 混合 AI 分级审计                                    │
│  - Step A: Qwen/Kimi 初筛（新闻总结）                        │
│  - Step B: DeepSeek 精审（风控审计）                         │
│  - 判定：PASS 或 REJECT                                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 大盘环境择时                                       │
│  - 计算中证 500 指数 20 日均线                                  │
│  - 价格在均线下 → 防守模式（预算减半）                        │
│  - 价格在均线上 → 正常模式                                   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 安装依赖

```bash
# 安装新增依赖
pip install akshare httpx

# 或更新全部依赖
pip install -r requirements.txt
```

## ⚙️ 配置说明

### 1. 配置文件 `config/settings.yaml`

```yaml
# 核心交易参数
trading:
  capital: 50000          # 总本金（元）
  max_positions: 3        # 最大持仓数
  min_prob: 0.70          # 量化概率门槛（70%）
  regime_ma: 20           # 大盘均线择时天数
  
  # 交易成本
  commission_rate: 0.0003     # 佣金率（万分之三）
  commission_min: 5.0         # 最低佣金（元）
  stamp_duty_rate: 0.001      # 印花税率（千分之一）

# 混合 AI 审计配置
ai_agents:
  # 第一级：免费模型初筛
  pre_filter:
    enabled: true
    provider: "qwen"
    api_key: "sk-sp-f47e1715e4bf4239bda9f14effd9aca4"
    base_url: "https://coding.dashscope.aliyuncs.com/v1"
    model: "qwen3.5-plus"
  
  # 第二级：DeepSeek 核心精审
  deep_audit:
    enabled: true
    provider: "deepseek"
    api_key: "sk-fcd8121f14114ef6865a3c193dad193b"
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"

# 股票池与标的
assets:
  index_code: "000905.SH"   # 中证 500（用于大盘择时）
  bond_etf: "511010"        # 国债 ETF（空仓时的避险标的）
```

### 2. 环境变量配置

在 `.env` 文件中配置 API 密钥：

```bash
# AI API 密钥
QWEN_API_KEY=your_qwen_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Mock 模式（无需 API 密钥测试）
AI_MOCK_MODE=true
```

## 🚀 使用方法

### 方式一：直接运行

```bash
# 使用 Mock 模式测试（无需 API 密钥）
set AI_MOCK_MODE=true && python src/daily_trade_advisor.py

# Linux/Mac
export AI_MOCK_MODE=true && python src/daily_trade_advisor.py

# 使用真实 API（需配置密钥）
python src/daily_trade_advisor.py
```

### 方式二：作为模块导入

```python
from src.daily_trade_advisor import DailyTradeAdvisor

# 创建顾问实例
advisor = DailyTradeAdvisor("config/settings.yaml")

# 执行决策流程
report = advisor.run()

# 打印报表
print(report)
```

## 📊 输出报表示例

### 有买入标的

```markdown
# 📊 明日决策清单
**交易日期**: 2026-03-11
**报告生成时间**: 2026-03-11 20:00:00

## 📈 买入标的

| 代码 | 名称 | 预测概率 | AI 审计摘要 | 建议股数 | 预估金额 |
|------|------|----------|-----------|----------|----------|
| 000001.SZ | 平安银行 | 78.5% | Mock 审计通过，无风险 | 300 | 3300.00 |

## 🛡️ 避险配置（国债 ETF）
- **代码**: 511010
- **建议股数**: 460
- **预估金额**: 46000.00

---

## 🔍 决策链条全回溯

### AI 审计层
- **No.1 [000001.SZ]** ✅ PASS: Mock 审计通过，无风险

---

## 🛡️ 风控看板

| 指标 | 数值 |
|------|------|
| **市场模式** | NORMAL |
| **API 调用消耗** | DeepSeek: 100 tokens |

---

## ⚠️ 风险提示
- **预估总换手率**: 6.6%
- **交易总成本**: 13.90 元
- **剩余现金**: 690.10 元
```

### 空仓避险

```markdown
# 📊 明日决策清单
**交易日期**: 2026-03-11

## 🚫 当前环境不佳，建议全仓避险（国债 ETF）

> **市场模式**: DEFENSIVE
> **中证 500 价格**: 5800.00 | **20 日均线**: 5900.00

## 🛡️ 避险配置（国债 ETF）
- **代码**: 511010
- **建议股数**: 495
- **预估金额**: 49500.00
```

## 💡 核心逻辑说明

### 1. 资金分配规则

| 场景 | 每只预算 | 说明 |
|------|---------|------|
| 正常模式，N 只 PASS | 50000/3 | 最多 3 只 |
| 防守模式，N 只 PASS | 50000/3/2 | 预算减半 |
| 无 PASS | 全仓国债 ETF | 空仓避险 |

### 2. 碎股取整规则

```python
# 严格执行 100 股整数倍向下取整
raw_shares = int(per_stock_budget / price)
shares = (raw_shares // 100) * 100
```

### 3. 交易成本计算

```python
# 佣金（最低 5 元）
commission = max(amount * 0.0003, 5.0)

# 印花税（卖出时收取，预留）
stamp_duty = amount * 0.001
```

### 4. AI 审计关键词

**负面关键词**（触发 REJECT）：
- 立案调查、违规担保、财务造假、面值退市
- 实控人变更、信披违规、被证监会处罚
- 重大违法强制退市、资金占用、关联交易

**"小作文"识别**：
- 传闻、消息称、据悉、网传、社交媒体
- 高送转、送转、蹭热点、蹭概念

## 🧪 Mock 模式测试

Mock 模式用于在没有真实 API 密钥的情况下测试系统：

```bash
# 设置 Mock 模式
set AI_MOCK_MODE=true

# 运行测试
python src/daily_trade_advisor.py
```

Mock 模式特点：
- Qwen Agent：返回固定摘要 "Mock 摘要：共 N 条新闻，无重大负面信息"
- DeepSeek Agent：默认 PASS，但 ST 股票会被 REJECT

## 📝 日志文件

日志文件位于 `logs/trade_advisor.log`，包含详细的执行过程：

```
2026-03-11 20:00:00 | INFO     | Layer 1: 量化门槛与 Top 10 提取
2026-03-11 20:00:01 | INFO     | Loaded 5000 rows of data
2026-03-11 20:00:02 | INFO     | Qualified stocks (P >= 70%): 15
2026-03-11 20:00:02 | INFO     | Top 10 candidates selected for AI audit
2026-03-11 20:00:03 | INFO     | Layer 2: 混合 AI 分级审计
...
```

## ⚠️ 注意事项

1. **数据库依赖**：系统需要以下数据表：
   - `stock_daily`：股票日线数据
   - `index_daily`：指数日线数据（用于大盘择时）
   - `etf_daily`：ETF 日线数据（用于国债 ETF 配置）

2. **模型文件**：需要预先训练好的模型 `data/models/stock_model.txt`

3. **API 密钥**：真实交易需要配置 Qwen 和 DeepSeek 的 API 密钥

4. **免责声明**：本报告仅供参考，不构成投资建议。量化模型和 AI 审计均存在局限性，投资需谨慎。

## 🔧 故障排除

### 问题 1：API 调用失败

```
Qwen API call failed: 401 Unauthorized
```

**解决方案**：
- 检查 `.env` 文件中的 API 密钥是否正确
- 确认 API 密钥格式正确（无多余空格）
- 如使用 Mock 模式，设置 `AI_MOCK_MODE=true`

### 问题 2：数据库连接失败

```
Database connection error: Can't connect to MySQL server
```

**解决方案**：
- 检查 `.env` 文件中的数据库配置
- 确认 MySQL 服务已启动
- 检查网络连接

### 问题 3：模型文件不存在

```
Model not found: data/models/stock_model.txt
```

**解决方案**：
- 运行模型训练：`python src/model_trainer.py`
- 或使用 Mock 模式（无模型时使用默认概率）