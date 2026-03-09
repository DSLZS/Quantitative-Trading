# Quantitative Trading System | 量化交易系统

基于 Polars + LightGBM 的 A 股量化交易系统，支持数据获取、因子计算、模型训练全流程。

## 📋 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [已实现功能](#已实现功能)
- [TODO](#todo)
- [模块详解](#模块详解)
- [使用示例](#使用示例)

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

```bash
cp .env.example .env
```

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

### 同步数据

```bash
# 同步沪深 300 成分股数据 (2024-01-01 至今)
python run_sync.py

# 同步指定指数
python run_sync.py --index 000905.SH --start 20230101

# 同步单只股票
python run_sync.py --single-stock 000001.SZ
```

---

## 🏗️ 系统架构

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Quantitative Trading System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Application Layer                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │   run_sync.py   │  │  backtest.py    │  │   train.py      │     │   │
│  │  │   数据同步脚本   │  │   回测脚本       │  │   训练脚本       │     │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│         ┌───────────────────────────┼───────────────────────────┐           │
│         │                           │                           │           │
│         ▼                           ▼                           ▼           │
│  ┌─────────────────┐       ┌─────────────────┐         ┌─────────────────┐ │
│  │   Data Layer    │       │ Compute Layer   │         │    Model Layer  │ │
│  │   ───────────   │       │   ───────────   │         │    ───────────  │ │
│  │                 │       │                 │         │                 │ │
│  │ TushareLoader   │       │ FactorEngine    │         │ ModelTrainer    │ │
│  │ (数据加载器)    │       │ (因子引擎)      │         │ (模型训练器)    │ │
│  │                 │       │                 │         │                 │ │
│  │ • 日线数据获取   │       │ • 因子计算       │         │ • LightGBM 训练  │ │
│  │ • 复权因子获取   │       │ • 标签生成       │         │ • 交叉验证       │ │
│  │ • 指数成分股    │       │ • 表达式解析     │         │ • 特征重要性     │ │
│  └────────┬────────┘       └────────┬────────┘         └────────┬────────┘ │
│           │                         │                           │           │
│           └─────────────────────────┼───────────────────────────┘           │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DatabaseManager (数据库管理器)                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  • 连接池管理 (SQLAlchemy QueuePool)                         │   │   │
│  │  │  • SQL 查询 → Polars DataFrame                                │   │   │
│  │  │  • Polars DataFrame → MySQL 写入                              │   │   │
│  │  │  • Upsert 操作 (处理主键冲突)                                  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│           ┌─────────────────────────┴───────────────────────────┐           │
│           │                         │                           │           │
│           ▼                         ▼                           ▼           │
│  ┌─────────────────┐       ┌─────────────────┐         ┌─────────────────┐ │
│  │   MySQL DB      │       │  Parquet Files  │         │  Model Files    │ │
│  │  quantitative_  │       │  data/parquet/  │         │  data/models/   │ │
│  │  trading        │       │                 │         │                 │ │
│  │  • stock_daily  │       │  • factors_*.   │         │  • *.txt        │ │
│  │  • stock_list   │       │  • features_*.  │         │  • *.pkl        │ │
│  └─────────────────┘       └─────────────────┘         └─────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 各层职责

| 层级 | 文件 | 类/功能 | 职责描述 |
|------|------|---------|----------|
| **应用层** | `run_sync.py` | - | 数据同步脚本，协调各模块完成数据获取 |
| | `backtest.py` (TODO) | - | 回测脚本，执行策略回测 |
| | `train.py` (TODO) | - | 模型训练脚本 |
| **计算层** | `src/factor_engine.py` | `FactorEngine` | 从 YAML 加载因子配置，使用 Polars 向量化计算因子 |
| **模型层** | `src/model_trainer.py` | `ModelTrainer` | LightGBM 模型训练、交叉验证、预测 |
| **数据层** | `src/data_loader.py` | `TushareLoader` | 从 Tushare API 获取股票数据 |
| | `src/db_manager.py` | `DatabaseManager` | MySQL 连接池管理、数据读写 |
| **存储层** | MySQL | `stock_daily` 表 | 存储日线行情数据 |
| | Parquet | `data/parquet/` | 存储因子计算结果 |
| | 文本 | `data/models/` | 存储训练好的模型 |

---

## ✅ 已实现功能

### 1. 数据获取模块 (`src/data_loader.py`)

| 功能 | 方法 | 描述 |
|------|------|------|
| 获取日线数据 | `_fetch_daily_data()` | 从 Tushare 获取 OHLCV 数据 |
| 获取复权因子 | `_fetch_adj_factor()` | 获取复权因子用于计算复权价格 |
| 获取指数成分股 | `_fetch_index_members()` | 获取沪深 300 等指数成分股 |
| 数据转换 | `_transform_data()` | 字段映射、日期转换、复权价格计算 |
| 单只股票同步 | `sync_stock_data()` | 同步单只股票到数据库 |
| 指数成分股同步 | `sync_index_constituents()` | 批量同步指数所有成分股 |
| 频率限制 | `_rate_limit()` | 自动控制 API 请求频率 |

### 2. 数据库管理模块 (`src/db_manager.py`)

| 功能 | 方法 | 描述 |
|------|------|------|
| 连接池管理 | `connect()`, `close()` | SQLAlchemy QueuePool 连接池 |
| SQL 查询 | `read_sql()` | 执行 SELECT 返回 Polars DataFrame |
| 数据写入 | `to_sql()` | Polars DataFrame 写入 MySQL |
| Upsert 操作 | `upsert()` | 处理主键冲突 (DELETE + INSERT) |
| 表检查 | `table_exists()` | 检查表是否存在 |
| 直接执行 | `execute()` | 执行 INSERT/UPDATE/DELETE |

### 3. 因子计算模块 (`src/factor_engine.py`)

| 功能 | 方法 | 描述 |
|------|------|------|
| 配置加载 | `_load_config()` | 从 YAML 加载因子定义 |
| 因子计算 | `compute_factors()` | 向量化计算所有因子 |
| 标签计算 | `compute_label()` | 计算预测目标 (未来收益率) |
| 因子列表 | `get_factor_names()` | 获取所有因子名称 |
| 特征列 | `get_feature_columns()` | 获取因子 + 标签列名 |

**支持的因子类型:**
- 动量因子：`momentum_5`, `momentum_10`, `momentum_20`
- 波动率因子：`volatility_5`, `volatility_20`
- 成交量因子：`volume_ma_ratio_5`, `volume_ma_ratio_20`
- 价格位置因子：`price_position_20`, `price_position_60`
- 均线偏离因子：`ma_deviation_5`, `ma_deviation_20`

### 4. 模型训练模块 (`src/model_trainer.py`)

| 功能 | 方法 | 描述 |
|------|------|------|
| 模型训练 | `train()` | LightGBM 训练，支持早停 |
| 交叉验证 | `cross_validate()` | 时间序列交叉验证 |
| 预测 | `predict()` | 在新数据上预测 |
| 特征重要性 | `get_top_features()` | 获取最重要特征 |
| 模型保存 | `save_model()` | 保存模型到文件 |
| 模型加载 | `load_model()` | 从文件加载模型 |

### 5. 同步脚本 (`run_sync.py`)

| 功能 | 参数 | 描述 |
|------|------|------|
| 同步指数成分股 | `--index INDEX` | 默认 000300.SH (沪深 300) |
| 设置日期范围 | `--start`, `--end` | 格式 YYYYMMDD |
| 同步单只股票 | `--single-stock CODE` | 同步指定股票 |
| 指定表名 | `--table NAME` | 默认 stock_daily |

---

## 🔜 TODO

### 短期计划

- [ ] **回测引擎**: 实现基于因子信号的回测框架
- [ ] **因子分析**: 添加因子 IC 分析、分层回测
- [ ] **数据验证**: 添加数据质量检查 (缺失值、异常值)
- [ ] **增量同步**: 优化同步逻辑，只获取新增数据

### 中期计划

- [ ] **更多数据源**: 支持 Baostock、AKShare 等
- [ ] **分钟线支持**: 支持高频数据获取和因子计算
- [ ] **特征工程**: 添加自动特征选择、特征构造
- [ ] **模型优化**: 支持 XGBoost、CatBoost 等更多模型

### 长期计划

- [ ] **实时数据**: 接入实时行情，支持盘中计算
- [ ] **策略库**: 实现多种经典量化策略
- [ ] **风险控制**: 添加仓位管理、止损止盈模块
- [ ] **可视化**: 添加收益曲线、因子分布等图表

---

## 📖 模块详解

### TushareLoader 类

```python
from src.data_loader import TushareLoader

# 初始化 (从.env 读取 TUSHARE_TOKEN)
loader = TushareLoader()

# 同步单只股票
rows = loader.sync_stock_data(
    ts_code="000001.SZ",
    start_date="20240101",
    end_date="20241231"
)

# 同步指数成分股
stats = loader.sync_index_constituents(
    index_code="000300.SH",
    start_date="20240101"
)
print(f"成功：{stats['successful_stocks']}/{stats['total_stocks']}")
```

### DatabaseManager 类

```python
from src.db_manager import DatabaseManager

# 获取单例实例
db = DatabaseManager()

# 查询数据 (返回 Polars DataFrame)
df = db.read_sql("SELECT * FROM stock_daily WHERE symbol = %s", 
                 params={"symbol": "000001.SZ"})

# 写入数据
rows = db.to_sql(df, "stock_daily", if_exists="append")

# Upsert (处理主键冲突)
db.upsert(df, "stock_daily", key_columns=["symbol", "Date"])
```

### FactorEngine 类

```python
from src.factor_engine import FactorEngine
import polars as pl

# 初始化 (加载 config/factors.yaml)
engine = FactorEngine("config/factors.yaml")

# 准备数据
df = pl.DataFrame({
    "close": [10.0, 10.5, 11.0, 10.8, 11.2],
    "volume": [1000, 1200, 1100, 1300, 1400],
    "pct_change": [0.0, 0.05, 0.048, -0.018, 0.037]
})

# 计算因子
result = engine.compute_factors(df)
print(result.columns)  # 包含所有因子列

# 计算标签
result = engine.compute_label(result)
```

### ModelTrainer 类

```python
from src.model_trainer import ModelTrainer

# 初始化
trainer = ModelTrainer(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

# 训练模型
model = trainer.train(X_train, y_train, X_val, y_val)

# 交叉验证
scores = trainer.cross_validate(X, y, n_splits=5)
print(f"平均 MSE: {sum(scores['valid'])/len(scores['valid']):.6f}")

# 获取重要特征
top_features = trainer.get_top_features(n=10)
for name, importance in top_features:
    print(f"{name}: {importance:.2f}")

# 保存模型
trainer.save_model("data/models/stock_model.txt")
```

---

## 📁 项目结构

```
Quantitative-Trading/
├── config/
│   └── factors.yaml          # 因子配置文件
├── data/
│   ├── parquet/              # Parquet 格式因子数据
│   ├── models/               # 训练好的模型
│   └── raw/                  # 原始数据
├── src/
│   ├── __init__.py
│   ├── db_manager.py         # 数据库管理器
│   ├── data_loader.py        # Tushare 数据加载器
│   ├── factor_engine.py      # 因子计算引擎
│   └── model_trainer.py      # 模型训练器
├── tests/
│   ├── test_db_manager.py
│   ├── test_factor_engine.py
│   └── ...
├── logs/                     # 日志文件
├── run_sync.py              # 数据同步脚本
├── requirements.txt         # 依赖
├── .env                     # 环境变量
└── README.md                # 本文档
```

---

## 📝 使用示例

### 完整流程示例

```python
# 1. 同步数据
python run_sync.py --index 000300.SH --start 20240101

# 2. 读取数据
from src.db_manager import DatabaseManager
db = DatabaseManager()
df = db.read_sql("SELECT * FROM stock_daily WHERE symbol = '000001.SZ'")

# 3. 计算因子
from src.factor_engine import FactorEngine
engine = FactorEngine("config/factors.yaml")
df = engine.compute_factors(df)
df = engine.compute_label(df)

# 4. 准备训练数据
# (去除 null 值，划分训练/验证集)
df_clean = df.drop_nulls()
feature_cols = engine.get_factor_names()
X = df_clean.select(feature_cols)
y = df_clean["future_return_5"]

# 5. 训练模型
from src.model_trainer import ModelTrainer
trainer = ModelTrainer()
trainer.train(X[:800], y[:800], X[800:], y[800:])

# 6. 查看重要特征
for name, importance in trainer.get_top_features(5):
    print(f"{name}: {importance:.2f}")

# 7. 保存模型
trainer.save_model("data/models/stock_model.txt")
```

---

## 📊 数据库表结构

### stock_daily 表

```sql
CREATE TABLE `stock_daily` (
    `symbol` varchar(10) NOT NULL COMMENT '股票代码',
    `trade_date` date NOT NULL COMMENT '交易日期',
    `open` decimal(10,2) NOT NULL,
    `high` decimal(10,2) NOT NULL,
    `low` decimal(10,2) NOT NULL,
    `close` decimal(10,2) NOT NULL,
    `volume` bigint(20) NOT NULL COMMENT '原始成交量',
    `amount` decimal(18,2) NOT NULL COMMENT '成交额',
    `adj_factor` decimal(12,6) NOT NULL DEFAULT '1.000000' COMMENT '复权因子(当日)',
    `turnover_rate` decimal(8,4) DEFAULT NULL COMMENT '换手率(基础非计算指标,建议保留)',
    PRIMARY KEY (`symbol`, `trade_date`),
    KEY `idx_date` (`trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY RANGE COLUMNS(`trade_date`) (
    PARTITION p_hist VALUES LESS THAN ('2023-01-01'),
    PARTITION p_2023 VALUES LESS THAN ('2024-01-01'),
    PARTITION p_2024 VALUES LESS THAN ('2025-01-01'),
    PARTITION p_2025 VALUES LESS THAN ('2026-01-01'),
    PARTITION p_future VALUES LESS THAN (MAXVALUE)
);
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

### ADBC 驱动不可用

```
WARNING: ADBC driver not available. Falling back to SQLAlchemy.
```

**说明**: Python 3.13 暂不支持 adbc-driver-mysql，系统会自动使用 SQLAlchemy 方案，不影响功能。

---

## 📄 许可证

MIT License

## 📬 联系方式

如有问题请提交 Issue 或联系开发者。