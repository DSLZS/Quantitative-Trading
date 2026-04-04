#!/usr/bin/env python3
"""
V150 回测运行脚本 - PCE (Polarity-Corrected-Ensemble).

【使用说明】
运行 2021 和 2024 年回测，输出以 IC 为核心的详细审计报告。

使用示例:
    python run_v150.py --year 2021
    python run_v150.py --year 2024
    python run_v150.py --all

【V150 核心改进】
1. Polarity Auto-Correction (PAC): 因子极性自动校正
2. Partial Industry Neutralization (PIN): 0.3 软中性化
3. EMA Signal Smoothing: 简化的指数平滑
4. 回归 V147 核心算子：volume_price_contradiction 和 liquidity_alpha

【目标指标】
- T+1 Rank IC > 0.05
- IC_IR > 0.50 (V149: ~0.40)
- IC Std < 0.08
- 400 Error: 0
- 日度信号换手率降低 15%+
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="V150 PCE Backtest Runner")
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Year to run audit (e.g., 2021, 2024)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run audit for all years (2021, 2024)'
    )
    parser.add_argument(
        '--parquet',
        type=str,
        default=None,
        help='Path to Parquet data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    # 使用 main.py 的统一入口
    sys.argv = [
        'main.py',
        '--version', '150',
        '--output', args.output,
    ]
    
    if args.parquet:
        sys.argv.extend(['--parquet', args.parquet])
    
    if args.all:
        sys.argv.append('--all')
    elif args.year:
        sys.argv.extend(['--year', str(args.year)])
    else:
        parser.print_help()
        logger.warning("Please specify --year or --all")
        sys.exit(1)
    
    # 导入并运行 main
    from main import main as main_func
    main_func()


if __name__ == '__main__':
    main()