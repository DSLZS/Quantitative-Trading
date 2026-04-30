# A股日频Alpha策略研发框架

> 本项目是一个A股日频Alpha策略研发框架，专注于因子组合优化与信号合成。
> 当前版本 **V218** 采用市场状态适配器 + 特征解耦架构，通过动态权重门控机制
> 在不同市场状态下自动调整反转与动量的权重配比。

---

## 目录结构

```
Quantitative-Trading/
├── config/                     # 配置文件目录
│   ├── factors.yaml            # 因子表达式配置
│   └── settings.yaml           # 系统参数配置
├── data/                       # 数据目录（原始数据 + 缓存）
│   ├── cache/                  # 数据缓存（Parquet格式）
│   └── models/                 # 训练好的模型文件
├── logs/                       # 运行日志目录
├── reports/                    # 回测报告目录（仅保留V218最新报告）
├── scripts/                    # 辅助脚本
│   └── diagnose_ic.py          # IC诊断分析脚本
├── src/                        # 核心源代码
│   ├── alpha_model_v218.py     # V218 Alpha模型（因子加权与信号生成）
│   ├── backtest_engine.py      # 回测引擎（数据加载、校验、报告生成）
│   └── engine/                 # 引擎模块
│       ├── __init__.py
│       └── backtest_referee.py # 不可变裁判引擎（T+1 IC计算、回测执行）
├── run_v218.py                 # V218 回测运行入口
├── requirements.txt            # Python依赖
├── .env                        # 环境变量（数据库连接等）
├── .gitignore                  # Git忽略规则
├── ALPHA_HISTORY.md            # 因子研发历史日志（永不丢失）
├── CURRENT_STATE.md            # 当前代码库快照摘要
├── TODOS.md                    # 下一步改进方向
└── PROJECT_ARCHITECTURE.md     # 项目架构详细文档
```

### 目录说明

| 目录 | 用途 |
|------|------|
| `config/` | 因子表达式和系统参数配置 |
| `data/` | 原始数据（Parquet缓存）和模型文件 |
| `logs/` | 运行日志输出 |
| `reports/` | 回测审计报告（仅保留V218） |
| `scripts/` | 辅助诊断和分析脚本 |
| `src/` | 核心源代码（Alpha模型、回测引擎） |

---

## 数据流向

```
┌─────────────────────────────────────────────────────────────────┐
│                    A股日频Alpha策略数据流                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 原始数据  │───▶│ 因子计算  │───▶│ 信号合成  │───▶│ 回测引擎  │  │
│  │ MySQL DB │    │ Alpha    │    │ Score    │    │ Referee  │  │
│  │ (价格/   │    │ Model    │    │ 加权     │    │ (IC/IR/  │  │
│  │  财务)   │    │          │    │          │    │  收益)   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  stock_daily    momentum_5      score =     T+1 IC      │
│  stock_fund_flow volatility_20  W_t*Rev    IR           │
│  (行业/市值)    reversal_5d    +(1-W_t)*Mom  年化收益    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流说明

1. **原始数据**：从MySQL数据库加载 `stock_daily`（日行情）和 `stock_fund_flow`（资金流）
2. **因子计算**：`AlphaModelV218` 计算反转（reversal）和动量（momentum）因子
3. **信号合成**：根据市场状态（CRISIS/TREND/NORMAL）动态加权反转与动量得分
4. **回测引擎**：`BacktestReferee` 执行T+1交易，计算IC、IR、年化收益等指标
5. **绩效评估**：生成审计报告（Markdown + JSON）
6. **日志记录**：所有运行日志输出到 `logs/` 和 `reports/`

---

## 核心模块说明

### `src/alpha_model_v218.py` - Alpha模型（Player）

- **职责**：因子计算与信号生成
- **当前模式**：线性组合（反转 + 动量）
- **核心方法**：`compute_score(df)` → 返回含 `score` 列的DataFrame
- **市场状态门控**：根据大盘波动率和趋势动态调整权重

### `src/backtest_engine.py` - 回测引擎（Referee）

- **职责**：数据加载、校验、报告生成
- **合规锁定**：
  - 初始资金：100,000
  - 费率：1.3‰（佣金0.3‰ + 印花税1‰ + 滑点0.5‰）
  - 无未来函数：所有计算仅使用T-1日及之前数据
- **核心方法**：`run_cross_year_audit(df, alpha_model, years)`

### `src/engine/backtest_referee.py` - 不可变裁判引擎

- **职责**：T+1 IC计算、信号生成、回测执行
- **不可变参数**：佣金率、印花税、滑点、持仓数量
- **验收标准**：T+1 Rank IC > 0.05，IC Decay 单调递减

### `run_v218.py` - 运行入口

- **职责**：环境清理、模型初始化、回测执行、日志更新
- **使用方法**：`python run_v218.py --years 2020 2022 2024`

---

## 运行方式

### 执行V218回测

```bash
# 默认回测年份：2020, 2022, 2024
python run_v218.py

# 指定年份
python run_v218.py --years 2020 2022 2024

# 指定输出目录
python run_v218.py --output-dir reports

# 指定数据库连接
python run_v218.py --db-url "mysql+pymysql://user:pass@host/db"
```

### IC诊断分析

```bash
python scripts/diagnose_ic.py
```


---

## 禁止规则

1. **严禁T+0交易**：所有交易在T+1日执行
2. **严禁使用未来信息**：禁止 `shift(-1)` 或任何T+1数据访问（`iloc[-1]` 除外）
3. **严禁 `fillna(0)`**：数据缺失必须使用 `data_healer` 模块处理
4. **严禁修改历史起始日期**：回测起始日期不可更改
5. **严禁修改裁判引擎参数**：`backtest_referee.py` 中的费率和资金参数不可变

---

## 技术栈

- **Python**: 3.13.x
- **数据处理**: Polars / Pandas
- **数据库**: MySQL + SQLAlchemy 2.0+
- **日志**: loguru
- **回测**: 自研引擎（T+1 IC计算 + 会计核算）

---

*最后更新: 2026-04-27 | 当前版本: V218*