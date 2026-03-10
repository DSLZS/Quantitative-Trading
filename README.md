# Quantitative Trading System | 量化交易系统

基于 Polars + LightGBM 的 A 股量化交易系统，支持数据获取、因子计算、模型训练、策略回测全流程。

## 📋 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [已实现功能](#已实现功能)
- [使用指南](#使用指南)
- [模块详解](#模块详解)
- [故障排除](#故障排除)

---

## 🚀 快速开始

### 环境要求

- Python 3.13.x
- MySQL 8.0+
- 虚拟环境 (venv)

### 安装依赖

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env` 文件并填写配置：

```ini
# MySQL 数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=quantitative_trading

# Tushare API Token (获取：https://tushare.pro/user/token)
TUSHARE_TOKEN=your_tushare_token_here
```

---

## 🏗️ 系统架构

### 数据流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Quantitative Trading System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据同步流程：                                                              │
│  Tushare API → TushareLoader → DatabaseManager → MySQL                      │
│                                                                             │
│  特征计算流程：                                                              │
│  MySQL → FeaturePipeline → FactorEngine → Parquet                           │
│                                                                             │
│  模型训练流程：                                                              │
│  Parquet → ModelTrainer → LightGBM → Model File                             │
│                                                                             │
│  回测流程：                                                                  │
│  Parquet + Model → Backtester → Performance Metrics → Visualizer → Plots   │
│                                                                             │
│  预测流程：                                                                  │
│  MySQL → NextDayPredictor → Model → Trading Signals                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ 已实现功能

### 1. 数据获取模块

| 脚本/模块 | 功能 | 描述 |
|-----------|------|------|
| `run_sync.py` | **统一数据同步** | 同步股票和 ETF 基金数据，支持 --asset-type 参数 |
| `src/data_loader.py` | 数据加载器 | 从 Tushare API 获取日线、复权因子数据 |
| `src/db_manager.py` | 数据库管理 | MySQL 连接池、数据读写 |

**支持的数据字段:**
- 基础行情：open, high, low, close, pre_close, change, pct_chg
- 成交量：volume, amount
- 复权数据：adj_factor, adj_open, adj_high, adj_low, adj_close
- 扩展字段：**turnover_rate** (换手率), **vol_ratio** (量比)

**run_sync.py 参数说明:**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--asset-type` | 资产类型：stock/fund/auto | auto |
| `--index` | 指数代码（如 000300.SH） | None |
| `--all-stocks` | 同步全部 A 股 | False |
| `--single-stock` | 同步单只股票 | None |
| `--start` | 开始日期 (YYYYMMDD) | 20240101 |
| `--end` | 结束日期 (YYYYMMDD) | 今天 |
| `--funds` | 指定基金代码（逗号分隔） | 全部目标基金 |

### 2. 特征工程模块

| 模块 | 功能 | 描述 |
|------|------|------|
| `src/feature_pipeline.py` | FeaturePipeline | 从数据库读取数据，计算因子，保存 Parquet |
| `src/factor_engine.py` | FactorEngine | 从 YAML 加载因子配置，向量化计算因子 |
| `config/factors.yaml` | 因子配置 | 定义技术因子和标签 |

**支持的因子类型:**
- 动量因子：momentum_5, momentum_10, momentum_20
- 波动率因子：volatility_5, volatility_20
- 成交量因子：volume_ma_ratio_5, volume_ma_ratio_20
- 价格位置因子：price_position_20, price_position_60
- 均线偏离因子：ma_deviation_5, ma_deviation_20
- **技术指标因子：RSI, MFI, Turnover_Bias**

### 3. 模型训练模块

| 模块 | 功能 | 描述 |
|------|------|------|
| `src/model_trainer.py` | ModelTrainer | LightGBM 模型训练、交叉验证、预测 |

### 4. 回测引擎

| 模块 | 功能 | 描述 |
|------|------|------|
| `src/backtester.py` | Backtester | 策略回测引擎，模拟交易 |
| `run_backtest.py` | 回测脚本 | 运行完整回测流程 |

**回测特性:**
- 滚动训练：每日使用过去 60 天数据重新训练模型
- **组合选股：选取预测分最高的前 N 只股票等权重配置**
- 交易信号：预测收益率 > 阈值时买入
- 次日卖出：次日开盘价卖出所有持仓
- **智能风控：支持信号失效止损**
- 交易成本：佣金万分之五 (双边) + 印花税千分之一 (卖出)

### 5. 可视化模块

| 模块 | 功能 | 描述 |
|------|------|------|
| `src/visualizer.py` | Visualizer | 绩效评估与可视化 |

**核心指标:**
- 年化收益率 (Annualized Return)
- 最大回撤 (Max Drawdown)
- 夏普比率 (Sharpe Ratio)
- 卡玛比率 (Calmar Ratio)
- 年化波动率 (Volatility)
- **胜率 (Win Rate)**
- **盈亏比 (Profit Factor)**
- **日均换手率**

**图表类型:**
- 资金曲线图 (Equity Curve)
- 回撤曲线图 (Drawdown Curve)
- 收益分布图 (Returns Distribution)
- **基准对比图 (Benchmark Comparison)**
- 综合报告图 (Summary Report)

### 6. 预测模块

| 模块 | 功能 | 描述 |
|------|------|------|
| `src/predict_next_day.py` | NextDayPredictor | 次日交易信号预测 |

**输出信号:**
- BUY: 预测收益率 > 0.5%
- SELL: 预测收益率 < -0.5%
- HOLD: 其他情况

---

## 📖 使用指南

### Step 1: 同步数据

```bash
# 同步所有 A 股股票数据 (从 2024-01-01 至今)
python run_sync.py

# 同步沪深 300 成分股
python run_sync.py --index 000300.SH --start 20240101

# 同步基金数据 (510300.SH 沪深 300ETF, 159915.SZ 创业板 ETF)
python run_sync.py --asset-type fund

# 同步股票和基金 (默认 auto 模式)
python run_sync.py --asset-type auto --start 20230101

# 同步单只股票
python run_sync.py --single-stock 000001.SZ --start 20230101
```

### Step 2: 生成特征

```bash
# 运行特征管道，生成 Parquet 文件
python src/feature_pipeline.py
```

### Step 3: 训练模型

```bash
# 运行模型训练
python src/model_trainer.py
```

### Step 4: 运行回测

```bash
# 使用默认参数运行回测
python run_backtest.py

# 自定义参数
python run_backtest.py --threshold 0.01 --capital 500000 --max-positions 5

# 不生成图表
python run_backtest.py --no-plot
```

**回测参数说明:**
- `--parquet`: 特征文件路径 (默认：data/parquet/features_latest.parquet)
- `--model`: 模型文件路径 (默认：data/models/stock_model.txt)
- `--threshold`: 买入阈值 (默认：0.005 = 0.5%)
- `--capital`: 初始资金 (默认：1,000,000)
- `--max-positions`: 最大持仓数 (默认：10)
- `--position-size`: 单仓位占比 (默认：0.1 = 10%)
- `--output`: 输出目录 (默认：data/plots)

### Step 5: 查看回测结果

回测完成后，结果保存在 `data/plots/` 目录：
- `backtest_result.png` - 综合报告图
- `equity_curve.png` - 资金曲线
- `drawdown_curve.png` - 回撤曲线
- `returns_distribution.png` - 收益分布
- `backtest_metrics.txt` - 绩效指标文本

### Step 6: 生成预测信号

```bash
# 运行次日预测
python src/predict_next_day.py
```

---

## 📖 模块详解

### Backtester 类 (回测引擎)

```python
from src.backtester import Backtester

# 初始化回测引擎
backtester = Backtester(
    initial_capital=1_000_000,      # 初始资金
    prediction_threshold=0.005,     # 买入阈值
    max_positions=10,               # 最大持仓数
    position_size_pct=0.1,          # 单仓位占比
)

# 运行回测
results = backtester.run(
    parquet_path="data/parquet/features_latest.parquet",
    model_path="data/models/stock_model.txt",
)

# 查看结果
print(f"总收益率：{results['metrics']['total_return']:.2%}")
print(f"年化收益：{results['metrics']['annualized_return']:.2%}")
print(f"最大回撤：{results['metrics']['max_drawdown']:.2%}")
print(f"夏普比率：{results['metrics']['sharpe_ratio']:.2f}")
```

### Visualizer 类 (可视化器)

```python
from src.visualizer import Visualizer

# 初始化可视化器
viz = Visualizer()

# 生成完整报告
report = viz.generate_report(
    equity_curve=results['equity_curve'],
    trade_records=results['records'],
    initial_capital=1_000_000,
    save_dir="data/plots",
)

# 查看生成的图表路径
print(report['plot_paths'])
# 输出：
# {
#     'equity_curve': 'data/plots/equity_curve.png',
#     'drawdown_curve': 'data/plots/drawdown_curve.png',
#     'returns_distribution': 'data/plots/returns_distribution.png',
#     'summary': 'data/plots/backtest_result.png'
# }
```

### NextDayPredictor 类 (预测器)

```python
from src.predict_next_day import NextDayPredictor

# 初始化预测器
predictor = NextDayPredictor(
    config_path="config/factors.yaml",
    model_path="data/models/stock_model.txt",
    lookback_days=60,
)

# 执行预测
result = predictor.predict()

# 获取最强买入信号
top_signals = predictor.get_top_signals(result, top_n=10)

# 查看信号
for row in top_signals.iter_rows():
    print(f"{row[0]}: 预测收益 {row[3]:.2%}")
```

---

## 📁 项目结构

```
Quantitative-Trading/
├── config/
│   └── factors.yaml              # 因子配置文件
├── data/
│   ├── parquet/                  # Parquet 格式因子数据
│   │   └── features_latest.parquet
│   ├── models/                   # 训练好的模型
│   │   └── stock_model.txt
│   └── plots/                    # 回测图表
│       ├── backtest_result.png
│       ├── equity_curve.png
│       ├── drawdown_curve.png
│       └── returns_distribution.png
├── src/
│   ├── __init__.py
│   ├── db_manager.py             # 数据库管理器
│   ├── data_loader.py            # Tushare 数据加载器
│   ├── factor_engine.py          # 因子计算引擎
│   ├── feature_pipeline.py       # 特征管道
│   ├── model_trainer.py          # 模型训练器
│   ├── backtester.py             # 回测引擎
│   ├── visualizer.py             # 可视化器
│   └── predict_next_day.py       # 预测模块
├── logs/                         # 日志文件
├── run_sync.py                   # 数据同步脚本 (统一入口)
├── run_backtest.py               # 回测运行脚本
├── requirements.txt              # 依赖
├── .env                          # 环境变量
└── README.md                     # 本文档
```

---

## 🔧 故障排除

### TUSHARE_TOKEN 未配置

```
ValueError: TUSHARE_TOKEN is required
```

**解决**: 在 `.env` 文件中设置有效的 Tushare Token

### MySQL 连接失败

```
sqlalchemy.exc.OperationalError: (2003, "Can't connect to MySQL server")
```

**解决**: 
1. 确保 MySQL 服务已启动
2. 检查 `.env` 中的连接配置
3. 确认数据库 `quantitative_trading` 已创建

### 模型文件不存在

```
FileNotFoundError: Model not found: data/models/stock_model.txt
```

**解决**: 先运行模型训练 `python src/model_trainer.py`

### 特征文件不存在

```
FileNotFoundError: Features file not found: data/parquet/features_latest.parquet
```

**解决**: 先运行特征管道 `python src/feature_pipeline.py`

### 绘图时字体问题

```
Warning: Font family not found
```

**解决**: 系统已配置 `matplotlib.use('Agg')` 确保无 GUI 环境也能保存图表，字体警告不影响功能

---

## 📊 交易成本说明

回测中考虑的交易成本：

| 成本类型 | 费率 | 收取方式 |
|----------|------|----------|
| 佣金 | 0.05% (万分之五) | 买卖双边收取，最低 5 元 |
| 印花税 | 0.1% (千分之一) | 仅卖出时收取 |

**计算公式:**
```
买入成本 = max(买入金额 × 0.0005, 5)
卖出成本 = max(卖出金额 × 0.0005, 5) + 卖出金额 × 0.001
总成本 = 买入成本 + 卖出成本
```

---

## 📄 许可证

MIT License

## 📬 联系方式

如有问题请提交 Issue 或联系开发者。