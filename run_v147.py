#!/usr/bin/env python3
"""
V147 回测运行脚本 - Multi-Resolution Entropy Fusion (MREF).

【使用说明】
运行 2021/2024 年回测，输出以 IC 为核心的详细审计报告。

使用示例:
    python run_v147.py --year 2024
    python run_v147.py --all
    python run_v147.py --year 2021 --parquet data/parquet/stock_data_2024_2026.parquet
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
    parser = argparse.ArgumentParser(description="V147 Backtest Runner - Multi-Resolution Entropy Fusion")
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Year to run backtest (e.g., 2021, 2024)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run backtest for all years (2021, 2024)'
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
    
    # Call main.py with version 147
    sys.argv = [
        'main.py',
        '--version', '147',
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
    
    # Import and run main
    from main import main as main_entry
    main_entry()


if __name__ == '__main__':
    main()