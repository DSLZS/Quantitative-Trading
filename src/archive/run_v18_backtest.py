#!/usr/bin/env python
"""
Run V18 Backtest - 非线性集成与回撤熔断对比测试

功能:
    - 运行 V18 策略回测
    - 生成 V18 报告
    - 对比 V17 与 V18 的性能差异

使用示例:
    >>> python src/run_v18_backtest.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.final_strategy_v1_18 import FinalStrategyV18


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    # 添加文件日志
    log_file = Path("logs/v18_backtest.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
    )
    
    logger.info("=" * 70)
    logger.info("V18 BACKTEST - 非线性集成与回撤熔断")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建策略实例
    strategy = FinalStrategyV18(
        config_path="config/production_params.yaml",
    )
    
    # 运行完整分析
    results = strategy.run_full_analysis(
        start_date="2023-01-01",
        end_date="2026-03-31",
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("V18 BACKTEST COMPLETE")
    logger.info("=" * 70)
    
    if "error" not in results:
        backtest = results["backtest_result"]
        regime = results["regime_result"]
        
        logger.info("\n📊 KEY METRICS:")
        logger.info(f"  Total Return: {backtest['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown: {backtest['max_drawdown']:.2%}")
        logger.info(f"  Total Trades: {backtest['total_trades']:,}")
        logger.info(f"  Q5-Q1 Spread: {backtest['q1_q5_spread']:.2%}")
        
        logger.info("\n🛡️ MARKET REGIME:")
        logger.info(f"  Current Status: {regime.regime_status}")
        logger.info(f"  Position Ratio: {regime.position_ratio:.1%}")
        logger.info(f"  Trigger Days: {len(regime.trigger_dates)}")
        
        logger.info("\n📄 Reports:")
        logger.info(f"  Report Path: {results['report_path']}")
        logger.info(f"  Cache Path: {results['cache_path']}")
        
        # 生成对比摘要
        logger.info("\n" + "=" * 70)
        logger.info("V17 vs V18 COMPARISON")
        logger.info("=" * 70)
        logger.info(f"{'Metric':<20} {'V17':<15} {'V18':<15} {'Improvement'}")
        logger.info("-" * 70)
        logger.info(f"{'Max Drawdown':<20} {'83.19%':<15} {backtest['max_drawdown']:.2%}  {'✅' if backtest['max_drawdown'] < 0.30 else '⚠️'}")
        logger.info(f"{'Total Trades':<20} {'0':<15} {backtest['total_trades']:,}  {'✅' if backtest['total_trades'] > 0 else '⚠️'}")
        logger.info(f"{'Sharpe Ratio':<20} {'1.808 (bug)':<15} {backtest['sharpe_ratio']:.3f}  {'✅' if 0 < backtest['sharpe_ratio'] < 5 else '⚠️'}")
        
    else:
        logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()