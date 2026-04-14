#!/usr/bin/env python
"""
V192 工业进化与 2025 OOS 压力测试运行脚本

运行命令：python run_v192.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alpha_research_v192 import V192BacktestRunner
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("V192 工业进化与 2025 OOS 压力测试")
    logger.info("=" * 70)
    
    # 创建回测运行器
    runner = V192BacktestRunner(
        output_dir='reports',
        initial_capital=100000.0,  # 锁定：100,000
        commission_rate=0.0013,     # 锁定：1.3‰ (佣金 0.3‰ + 印花税 1‰)
        slippage_rate=0.001,        # 锁定：0.1%
    )
    
    # 运行全量回测（2023/2024/2025）
    results = runner.run_full_backtest(years=[2023, 2024, 2025])
    
    if results:
        logger.info("=" * 70)
        logger.info("V192 回测完成！")
        logger.info("=" * 70)
    else:
        logger.error("V192 回测失败！")
        sys.exit(1)