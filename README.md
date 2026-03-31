# Quantitative Trading System - V101 Alpha Prediction

**Version**: 1.0.1 | **Status**: Production Ready

---

## 📋 概述

V101 是一个基于多因子模型的 A 股量化交易系统，专注于 Alpha 预测能力的提升。

**核心特点**:
- ✅ **统一架构**: 三个核心模块，代码清晰易维护
- ✅ **量价非线性交互**: 核心预测算法，捕捉市场非线性特征
- ✅ **T+1 收益预测**: 以 Spearman Rank IC 为损失函数的预测目标
- ✅ **自检机制**: IC < 0.03 自动触发 AlphaWeakWarning 并分析因子贡献度
- ✅ **数据防御**: 自动检查 2024 年 total_mv 数据完整性

---

## 🏗️ 架构设计

```
src/
├── data_loader.py          # 数据加载模块
│   ├── DataLoader          # 从 Tushare/数据库加载数据
│   ├── check_2024_total_mv # 数据防御检查
│   └── fetch_*             # 获取日线、复权因子、daily_basic 数据
│
├── alpha_research.py       # Alpha 预测核心（唯一存放预测算法）
│   ├── AlphaResearch       # 因子计算引擎
│   ├── compute_*           # 量价非线性交互因子
│   ├── calculate_rank_ic   # Spearman Rank IC 计算
│   └── run_alpha_analysis  # 完整分析流程
│
└── backtest_accounting.py  # 回测会计模块
    ├── BacktestAccounting  # 回测引擎
    ├── generate_signals    # 交易信号生成
    ├── calculate_transaction_cost  # 交易成本计算
    └── generate_report     # 审计报告生成
```

### 模块职责

| 模块 | 职责 | 禁止行为 |
|------|------|----------|
| `data_loader.py` | 数据拉取、补全、校验 | 不进行任何预测计算 |
| `alpha_research.py` | 因子计算、预测评分、IC 评估 | 不修改回测参数 |
| `backtest_accounting.py` | 回测、扣费、报告 | 不修改调仓频率、初始资金 |

---

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 TUSHARE_TOKEN 和 DATABASE_URL
```

### 运行回测

```bash
# 运行单一年份审计
python run_v101.py --year 2019
python run_v101.py --year 2021
python run_v101.py --year 2024

# 运行所有年份审计
python run_v101.py --all

# 使用 Parquet 数据文件
python run_v101.py --year 2024 --parquet data/parquet/features.parquet
```

### 代码调用

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from alpha_research import AlphaResearch
from backtest_accounting import BacktestAccounting

# 1. 加载数据
loader = DataLoader()
df = loader.load_data("000001.SZ", "20240101", "20241231")

# 2. 运行 Alpha 分析
alpha = AlphaResearch()
result = alpha.run_alpha_analysis(df)

# 3. 运行回测
backtest = BacktestAccounting()
audit_result = backtest.run_full_audit(df, result)
```

---

## 📊 验收指标

| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| **T+1 Rank IC** | > 0.05 | 核心指标：低于此值直接视为失败 |
| **IC IR (稳定性)** | > 0.6 | 跨年度预测能力的稳定性 |
| **Top Factor IC** | > 0.04 | 必须有至少一个核心因子具备独立战斗力 |
| 回测净收益 | 仅作输出参考 | 不作为优化目标 |

---

## 🔬 预测算法详解

### 核心逻辑

**预测目标**: T+1 收益的截面排名

**损失函数**: Spearman Rank IC

```python
# T+1 收益计算
T+1_Return = Close_{t+1} / Close_t - 1

# 截面排名（按日期分组）
Rank_Norm = (Rank - Min_Rank) / (Max_Rank - Min_Rank)

# Rank IC（Spearman 相关系数）
IC = Corr(Rank(Predict_Score), Rank(T+1_Return))
```

### 量价非线性交互因子

| 因子 | 公式 | 金融逻辑 |
|------|------|----------|
| `volume_price_divergence` | Price_Change - Volume_Change | 捕捉价量背离信号 |
| `volume_price_health` | 非线性映射（4 象限） | 价涨量增健康，价跌量增危险 |
| `vcp_score` | (波动率收缩 × 成交量萎缩) | VCP 整理形态识别 |
| `volume_entropy` | -Σ(p × ln(p)) | 成交量分布熵值 |

### 因子权重配置

```yaml
# config/factors.yaml
factors:
  - name: momentum_5
    expression: "close / close.shift(5) - 1"
    window: 5
  - name: momentum_10
    expression: "close / close.shift(10) - 1"
    window: 10
  # ... 更多因子
```

---

## 🛡️ 自检机制

### AlphaWeakWarning

当 T+1 IC < 0.03 时，系统自动触发警告并分析因子贡献度：

```
[AlphaWeakWarning] T+1 IC = 0.0215 < 0.03
[因子贡献度分析] 开始分析各因子 IC 贡献...
[因子贡献度分析] 结果:
  1. volume_price_health: IC=0.0421, Weight=0.10, Contribution=0.0042 ✓
  2. vcp_score: IC=0.0385, Weight=0.12, Contribution=0.0046 ✓
  3. momentum_5: IC=0.0125, Weight=0.15, Contribution=0.0019 ✗
  ...
[因子贡献度分析] 发现 3 个失效因子:
  - momentum_20: IC=0.0052
  - volatility_20: IC=0.0031
  - rsi_14: IC=0.0018
```

### 数据防御

运行前自动检查 2024 年 total_mv 数据：

```
[数据防御] 检查 000001.SZ 的 2024 年 total_mv 数据...
[数据防御] 000001.SZ 的 2024 年 total_mv 数据完整 (250 条)
```

---

## 📁 目录结构

```
Quantitative-Trading/
├── src/
│   ├── data_loader.py          # 数据加载模块
│   ├── alpha_research.py       # Alpha 预测核心
│   ├── backtest_accounting.py  # 回测会计模块
│   └── core/
│       └── __init__.py         # 统一导出
├── config/
│   ├── factors.yaml            # 因子配置
│   └── settings.yaml           # 系统设置
├── run_v101.py                 # 统一运行脚本
├── reports/                    # 审计报告输出
├── data/
│   ├── parquet/                # Parquet 数据缓存
│   └── raw/                    # 原始数据
└── README.md                   # 本文档
```

---

## 📝 更新日志

### V101 (当前版本)

- ✅ **项目清零行动**: 删除 v1_ 到 v100_ 过时代码
- ✅ **统一架构**: 三个核心模块 (data_loader, alpha_research, backtest_accounting)
- ✅ **预测算法优化**: 回归 V90 基础，引入量价非线性交互逻辑
- ✅ **自检机制**: IC < 0.03 触发警告并分析因子贡献
- ✅ **数据防御**: 自动检查 2024 年 total_mv 数据

---

## 🔧 配置说明

### 环境变量 (.env)

```bash
# Tushare API Token
TUSHARE_TOKEN=your_token_here

# 数据库连接 URL
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/dbname
```

### 因子配置 (config/factors.yaml)

```yaml
factors:
  - name: momentum_5
    expression: "close / close.shift(5) - 1"
    window: 5

label:
  name: sharpe_label
  expression: "close.shift(-5) / close - 1"
```

---

## 📄 许可证

MIT License

---

*Last Updated: 2026-03-31*