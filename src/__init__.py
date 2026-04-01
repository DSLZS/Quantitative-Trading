"""
Quantitative Trading System - Source Package
"""

__version__ = "0.1.0"

# V114: 延迟导入以避免循环依赖
try:
    from .db_manager import DatabaseManager
    __all__ = ["DatabaseManager"]
except ImportError:
    # db_manager 可能不存在，使用延迟导入
    __all__ = []
