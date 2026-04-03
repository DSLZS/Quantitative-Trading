#!/usr/bin/env python3
"""
V149 回测运行脚本 - SIE (Spectral-Inertia-Enhancement).

【使用说明】
运行 2021 和 2024 年回测，输出以 IC 为核心的详细审计报告。

使用示例:
    python run_v149.py --year 2021
    python run_v149.py --year 2024
    python run_v149.py --all

【V149 核心改进】
1. 修复 400 报错：日志截断，禁止 Dump 全量数据
2. Dynamic Signal Inertia Kernel: α 根据自相关性动态调整
3. Enhanced Gram-Schmidt: 每日截面因子正交化
4. Strict Industry Neutralization: 行业均值减法

【目标指标】
- T+1 Rank IC > 0.055
- IC_IR > 0.55 (V148: ~0.40)
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
    parser = argparse.ArgumentParser(description="V149 SIE Backtest Runner")
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
        '--version', '149',
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