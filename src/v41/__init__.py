"""
V41 Module - 模块化量化交易系统

【核心模块】
1. DataLoader - 数据加载模块
2. RiskManager - 风险管理模块
3. FactorLibrary - 因子库模块

【V41 核心改进】
1. 二阶导动量因子（Momentum of Momentum）
2. 板块中性化逻辑
3. 低波动时提升风险暴露（0.5% -> 0.8%）

作者：量化系统
版本：V41.0
日期：2026-03-19
"""

from .data_loader import DataLoader, DataLoaderConfig, FALLBACK_STOCKS
from .risk_manager import RiskManager, RiskManagerConfig, Position, Trade, TradeAudit
from .factor_library import FactorLibrary, FactorConfig, DEFAULT_FACTOR_WEIGHTS

__all__ = [
    'DataLoader',
    'DataLoaderConfig',
    'RiskManager',
    'RiskManagerConfig',
    'Position',
    'Trade',
    'TradeAudit',
    'FactorLibrary',
    'FactorConfig',
    'DEFAULT_FACTOR_WEIGHTS',
    'FALLBACK_STOCKS',
]

__version__ = '41.0.0'