#!/usr/bin/env python3
"""
V17 Backtest Runner - 双重中性化与全周期实战回测

使用方法:
    python src/run_v17_backtest.py

输出:
    - reports/V17_Double_Neutralization_Report_YYYYMMDD_HHMMSS.md
    - data/features_v17.parquet (缓存文件)
    - data/plots/v17_quintile_cumulative_returns.png
    - data/plots/v17_mv_correlation_history.png
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.final_strategy_v1_17 import FinalStrategyV17
from loguru import logger


def main():
    """主入口函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V17 BACKTEST RUNNER")
    logger.info("双重中性化与全周期实战回测")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 全周期回测参数
    START_DATE = "2023-01-01"
    END_DATE = "2026-03-31"
    
    logger.info(f"Backtest Period: {START_DATE} to {END_DATE}")
    logger.info(f"Total Period: ~3 years 3 months")
    
    # 创建策略实例
    strategy = FinalStrategyV17(
        config_path="config/production_params.yaml",
    )
    
    # 运行完整分析
    results = strategy.run_full_analysis(
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("V17 BACKTEST COMPLETE")
    logger.info("=" * 70)
    
    if "error" not in results:
        logger.info(f"Report Path: {results['report_path']}")
        logger.info(f"Cache Path: {results['cache_path']}")
        
        # 打印关键结果摘要
        logger.info("\n" + "=" * 60)
        logger.info("KEY RESULTS SUMMARY")
        logger.info("=" * 60)
        
        # IC 结果
        logger.info("\nFactor IC Analysis:")
        for ic_result in results.get('ic_results', []):
            logger.info(f"  {ic_result.factor_name}: Mean IC = {ic_result.mean_ic:.4f}, IC IR = {ic_result.ic_ir:.2f}")
        
        # 中性化验证
        logger.info("\nMarket Cap Neutralization:")
        for factor, stats in results.get('mv_neutral_results', {}).items():
            logger.info(f"  {factor}: Mean Corr with Log(MV) = {stats['mean_corr']:.4f}")
        
        # 回测结果
        backtest = results.get('backtest_result', {})
        logger.info("\nBacktest Performance (V13 Standard):")
        logger.info(f"  Total Return: {backtest.get('total_return', 0):.2%}")
        logger.info(f"  Annual Return: {backtest.get('annual_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown: {backtest.get('max_drawdown', 0):.2%}")
        logger.info(f"  Q5-Q1 Spread: {backtest.get('q1_q5_spread', 0):.2%}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ V17 backtest completed successfully!")
        logger.info("=" * 70)
        
        return 0
    else:
        logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())