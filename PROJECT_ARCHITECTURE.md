# PROJECT_ARCHITECTURE.md - 项目架构详细文档

> 展示项目核心模块的类图、流程图和调用关系。

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Quantitative Trading System                      │
│                           A股日频Alpha策略                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  数据层      │    │  策略层      │    │  回测层      │              │
│  │  Data Layer  │───▶│ Strategy     │───▶│ Backtest     │              │
│  │              │    │ Layer        │    │ Layer        │              │
│  │ MySQL DB     │    │ AlphaModel   │    │ Referee      │              │
│  │ stock_daily  │    │ V218         │    │ Immutable    │              │
│  │ stock_fund_  │    │              │    │              │              │
│  │ flow         │    │              │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                     │                     │                   │
│         ▼                     ▼                     ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ 数据加载器    │    │ 因子计算器    │    │ IC计算器     │              │
│  │ BacktestEng. │    │ compute_score│    │ Rank IC      │              │
│  │ load_data()  │    │ ()           │    │ calculate_   │              │
│  │              │    │              │    │ t1_ic()      │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 类图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BacktestEngine                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - output_dir: Path                                           │  │
│  │ - db_url: str                                                │  │
│  │ - _engine: Engine                                            │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ + __init__(output_dir, db_url)                               │  │
│  │ + load_data(years, warmup_year, warmup_days) -> DataFrame    │  │
│  │ + validate_data(df, years) -> dict                           │  │
│  │ + run_cross_year_audit(df, alpha_model, years) -> dict       │  │
│  │ + generate_cross_year_report(results, years) -> str          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                │ calls
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AlphaModelV218                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - VERSION: str = "V218"                                      │  │
│  │ - market_state: str                                          │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ + __init__()                                                 │  │
│  │ + compute_score(df) -> DataFrame                             │  │
│  │ + _detect_market_state(df) -> str                            │  │
│  │ + _compute_reversal_score(df) -> Series                      │  │
│  │ + _compute_momentum_score(df) -> Series                      │  │
│  │ + _compute_dynamic_weight(volatility, trend) -> float        │  │
│  │ + get_factor_ics(df) -> dict                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                │ calls
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BacktestReferee                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ # 不可变参数                                                  │  │
│  │ COMMISSION_RATE = 0.0003                                     │  │
│  │ STAMP_DUTY_RATE = 0.001                                      │  │
│  │ SLIPPAGE_RATE = 0.0005                                       │  │
│  │ TOP_N = 50                                                   │  │
│  │ POSITION_PER_STOCK = 0.02                                    │  │
│  │ INITIAL_CAPITAL = 100_000                                    │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ + __init__(alpha_module, output_dir)                         │  │
│  │ + validate_signal_input(df) -> bool                          │  │
│  │ + generate_signals(df, score_column) -> DataFrame            │  │
│  │ + calculate_transaction_cost(buy, sell) -> dict              │  │
│  │ + run_backtest(signals, returns) -> dict                     │  │
│  │ + calculate_rank_ic(factor, label) -> float                  │  │
│  │ + calculate_t1_ic(df, score_column) -> dict                  │  │
│  │ + calculate_ic_decay(df, score_column) -> dict               │  │
│  │ + run_audit(df) -> dict                                      │  │
│  │ + generate_report(...) -> str                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 调用流程图

```
run_v218.py
    │
    ├── Phase 1: clean_old_versions()
    │       └── 删除旧版本文件和过期报告
    │
    ├── Phase 2: get_alpha_model()
    │       └── 创建 AlphaModelV218 实例
    │
    ├── Phase 3: get_backtest_engine()
    │       └── 创建 BacktestEngine 实例
    │
    └── Phase 4: 加载数据并运行回测
            │
            ├── engine.load_data(years, warmup_year, warmup_days)
            │       │
            │       ├── 查询 stock_daily 表
            │       ├── 查询 stock_fund_flow 表
            │       └── 合并数据
            │
            ├── engine.validate_data(df, years)
            │       └── 检查每日股票数 > 5000
            │
            └── engine.run_cross_year_audit(df, alpha_model, years)
                    │
                    └── for each year:
                            │
                            ├── alpha_model.compute_score(year_data)
                            │       │
                            │       ├── _detect_market_state(df)
                            │       │       └── 基于波动率和趋势判定状态
                            │       │
                            │       ├── _compute_reversal_score(df)
                            │       │       └── 5d/10d/20d 反转因子加权
                            │       │
                            │       ├── _compute_momentum_score(df)
                            │       │       └── 20d 动量 + 中期动量
                            │       │
                            │       └── 动态权重合成
                            │               └── score = W_t*Rev + (1-W_t)*Mom
                            │
                            ├── referee.run_audit(score_df)
                            │       │
                            │       ├── validate_signal_input(df)
                            │       ├── _compute_t1_returns(df)
                            │       ├── calculate_t1_ic(df)
                            │       ├── calculate_ic_decay(df)
                            │       ├── generate_signals(df)
                            │       ├── run_backtest(signals, returns)
                            │       │       │
                            │       │       ├── 按日期遍历
                            │       │       ├── 执行调仓
                            │       │       ├── 计算交易成本
                            │       │       └── 更新组合价值
                            │       │
                            │       └── generate_report(...)
                            │
                            └── 汇总结果
```

---

## 数据流详细图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MySQL Database                               │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  stock_daily    │    │ stock_fund_flow │                         │
│  │ ─────────────── │    │ ─────────────── │                         │
│  │ trade_date      │    │ trade_date      │                         │
│  │ symbol          │    │ symbol          │                         │
│  │ open            │    │ net_main_amount │                         │
│  │ high            │    │ net_main_rate   │                         │
│  │ low             │    └─────────────────┘                         │
│  │ close           │                                                │
│  │ pre_close       │                                                │
│  │ pct_chg         │                                                │
│  │ volume          │                                                │
│  │ amount          │                                                │
│  │ turnover_rate   │                                                │
│  │ industry_code   │                                                │
│  │ total_mv        │                                                │
│  │ is_st           │                                                │
│  └─────────────────┘                                                │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       │ pd.read_sql()
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BacktestEngine.load_data()                        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 合并后的 DataFrame                                            │   │
│  │                                                              │   │
│  │ trade_date | symbol | close | volume | pct_chg | ...        │   │
│  │ 20200102   | 000001 | 10.5  | 10000  | 0.01    | ...        │   │
│  │ 20200103   | 000001 | 10.8  | 12000  | 0.02    | ...        │   │
│  │ ...        | ...   | ...   | ...   | ...     | ...        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       │ alpha_model.compute_score()
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    AlphaModelV218.compute_score()                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 因子计算                                                      │   │
│  │                                                              │   │
│  │ reversal_5d  = close / close.shift(5) - 1                    │   │
│  │ reversal_10d = close / close.shift(10) - 1                   │   │
│  │ reversal_20d = close / close.shift(20) - 1                   │   │
│  │ momentum_20  = close / close.shift(20) - 1                   │   │
│  │ volatility_20 = close.rolling_std(20)                        │   │
│  │                                                              │   │
│  │ 截面Rank标准化                                                │   │
│  │ reversal_5d_rank = reversal_5d.rank(method='dense')          │   │
│  │ ...                                                          │   │
│  │                                                              │   │
│  │ 市场状态判定                                                  │   │
│  │ volatility_20 = std(ret_20)                                  │   │
│  │ trend_20 = ma(close, 20) / close                             │   │
│  │                                                              │   │
│  │ 动态权重                                                      │   │
│  │ W_t = sigmoid(1.5 * vol_norm + 1.0 * trend)                  │   │
│  │                                                              │   │
│  │ 最终得分                                                      │   │
│  │ score = W_t * score_rev + (1 - W_t) * score_mom              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       │ referee.run_audit()
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BacktestReferee.run_audit()                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 审计流程                                                      │   │
│  │                                                              │   │
│  │ 1. validate_signal_input(df)    # 验证格式                    │   │
│  │ 2. _compute_t1_returns(df)      # 计算T+1收益                │   │
│  │ 3. calculate_t1_ic(df)          # 计算T+1 IC                 │   │
│  │ 4. calculate_ic_decay(df)       # 计算IC Decay               │   │
│  │ 5. generate_signals(df)         # 生成交易信号               │   │
│  │ 6. run_backtest(signals, ret)   # 执行回测                   │   │
│  │ 7. generate_report(...)         # 生成报告                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         输出报告                                      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ reports/                                                     │   │
│  │ ├── V218_Cross_Year_Report_YYYYMMDD.md                      │   │
│  │ ├── V218_Cross_Year_Report_YYYYMMDD.json                    │   │
│  │ ├── v218_year2020_audit_YYYYMMDD.md                         │   │
│  │ ├── v218_year2022_audit_YYYYMMDD.md                         │   │
│  │ └── v218_year2024_audit_YYYYMMDD.md                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 模块依赖关系

```
run_v218.py
├── src.alpha_model_v218
│   └── (因子计算、信号生成)
├── src.backtest_engine
│   ├── src.alpha_model_v218 (调用Player)
│   └── src.engine.backtest_referee (调用Referee)
└── src.engine.backtest_referee
    └── (不可变裁判引擎)

config/
├── factors.yaml (因子表达式)
└── settings.yaml (系统参数)

data/
├── cache/ (Parquet缓存)
└── models/ (模型文件)
```

---

## 架构设计原则

### 1. Referee-Player 架构
- **Player (AlphaModel)**: 负责因子计算和信号生成
- **Referee (BacktestReferee)**: 负责回测执行和结果验证
- **隔离原则**: Player 不可接触回测逻辑，Referee 不可修改因子计算

### 2. 不可变性原则
- `backtest_referee.py` 中的费率和资金参数一旦设定，不可修改
- 任何修改必须通过创建新版本实现

### 3. 无未来函数原则
- 所有因子计算仅使用 T-1 日及之前数据
- T+1 收益由 Referee 独立计算，Player 不可访问

### 4. 可扩展性原则
- 新增因子只需修改 `alpha_model_v218.py`
- 新增市场状态只需修改 `_detect_market_state()` 方法
- 回测引擎保持稳定，不随策略变化而修改

---

*最后更新: 2026-04-27 | 维护者: AI Assistant*