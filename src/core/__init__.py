"""
V101 Core Module - Unified Alpha Prediction System.

核心模块导出:
    - AlphaResearch: Alpha 预测核心引擎
    - BacktestAccounting: 回测会计引擎
    - DataLoader: 数据加载器

【架构说明】
- src/data_loader.py: 负责数据拉取、补全和校验
- src/alpha_research.py: 唯一存放预测算法的地方
- src/backtest_accounting.py: 负责回测、计算扣费、生成报告
"""

# 从 alpha_research 模块导入
from alpha_research import (
    AlphaResearch,
    AlphaWeakWarning,
    get_alpha_research,
)

# 从 backtest_accounting 模块导入
from backtest_accounting import (
    BacktestAccounting,
    get_backtest_accounting,
)

# 从 data_loader 模块导入
from data_loader import (
    DataLoader,
    DataLoaderError,
    get_loader,
)

__version__ = "1.0.1"
__all__ = [
    # Alpha Research
    "AlphaResearch",
    "AlphaWeakWarning",
    "get_alpha_research",
    # Backtest Accounting
    "BacktestAccounting",
    "get_backtest_accounting",
    # Data Loader
    "DataLoader",
    "DataLoaderError",
    "get_loader",
]