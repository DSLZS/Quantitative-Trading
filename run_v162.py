"""
V162 Alpha 绝地反击计划 - 运行脚本.

使用方法:
    python run_v162.py --year 2024
    python run_v162.py --years 2024 2023 2022
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

from src.alpha_research_v162 import V162Runner


def main():
    parser = argparse.ArgumentParser(description="V162 Alpha 绝地反击计划 - 回测运行器")
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="回测年份 (默认：2024)"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="回测年份列表 (例如：--years 2024 2023 2022)"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Parquet 数据文件路径 (可选)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="输出目录 (默认：reports)"
    )
    
    args = parser.parse_args()
    
    # 初始化运行器
    runner = V162Runner(
        parquet_path=args.parquet,
        output_dir=args.output,
    )
    
    # 运行审计
    if args.years:
        # 多年份审计
        summary = runner.run_multi_year_audit(args.years)
        
        logger.info("=" * 70)
        logger.info("V162 Multi-Year Audit Summary")
        logger.info("=" * 70)
        logger.info(f"Years: {summary['years']}")
        logger.info(f"Passed: {summary['passed_count']}/{summary['total_count']}")
        logger.info(f"Cross-Year IC Mean: {summary['cross_year_ic_mean']:.4f}")
        logger.info(f"Cross-Year IC IR: {summary['cross_year_ic_ir']:.2f}")
        
        # 判断是否达标
        ic_target = 0.095
        ir_target = 0.60
        ic_passed = summary['cross_year_ic_mean'] > ic_target
        ir_passed = summary['cross_year_ic_ir'] > ir_target
        
        logger.info("=" * 70)
        logger.info(f"IC Target (> {ic_target}): {'PASSED ✓' if ic_passed else 'FAILED ✗'}")
        logger.info(f"IR Target (> {ir_target}): {'PASSED ✓' if ir_passed else 'FAILED ✗'}")
        logger.info("=" * 70)
        
    else:
        # 单一年份审计
        result = runner.run_audit(args.year)
        
        logger.info("=" * 70)
        logger.info(f"V162 Audit Summary - Year {args.year}")
        logger.info("=" * 70)
        
        t1_ic = result.get('t1_ic', {})
        ic_decay = result.get('ic_decay', {})
        backtest_result = result.get('backtest_result', {})
        passed = result.get('passed', False)
        
        logger.info(f"T+1 Rank IC: {t1_ic.get('mean_ic', 0):.4f} (Target: > 0.095)")
        logger.info(f"IC IR: {t1_ic.get('ic_ir', 0):.2f} (Target: > 0.60)")
        logger.info(f"IC Decay: {ic_decay.get('decay_pattern', 'N/A')}")
        logger.info(f"Overall: {'PASSED ✓' if passed else 'FAILED ✗'}")
        
        if backtest_result:
            logger.info(f"Total Return: {backtest_result.get('total_return', 0):.2%}")
            logger.info(f"Sharpe Ratio: {backtest_result.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {backtest_result.get('max_drawdown', 0):.2%}")
        
        logger.info("=" * 70)


if __name__ == "__main__":
    main()