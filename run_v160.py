"""
V160 回测运行脚本 - Feature Engineering Breakthrough.

【使用方法】
python run_v160.py --year 2024
python run_v160.py --years 2024 2025

【目标指标】
- T+1 Rank IC > 0.10
- IC_IR > 0.7
- IC Decay 单调递减
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.alpha_research_v160 import V160Runner
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='V160 Backtest Runner')
    parser.add_argument('--year', type=int, default=2024, help='Backtest year')
    parser.add_argument('--years', nargs='+', type=int, default=None, help='Multiple years')
    parser.add_argument('--parquet', type=str, default=None, help='Parquet file path')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("V160 Backtest Runner - Feature Engineering Breakthrough")
    logger.info("=" * 70)
    
    runner = V160Runner(parquet_path=args.parquet, output_dir=args.output)
    
    if args.years:
        result = runner.run_multi_year_audit(args.years)
    else:
        result = runner.run_audit(args.year)
    
    logger.info("=" * 70)
    logger.info("V160 Backtest Complete")
    logger.info("=" * 70)
    
    if 'cross_year_ic_mean' in result:
        logger.info(f"Cross-Year IC Mean: {result['cross_year_ic_mean']:.4f}")
        logger.info(f"Cross-Year IC IR: {result['cross_year_ic_ir']:.2f}")
        logger.info(f"Passed: {result['passed_count']}/{result['total_count']}")
    
    return result

if __name__ == "__main__":
    main()