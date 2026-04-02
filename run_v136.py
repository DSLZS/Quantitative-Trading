#!/usr/bin/env python3
"""
V136 回测运行脚本 - 高维非线性空间拓展.

【使用说明】
    python run_v136.py --year 2024
    python run_v136.py --all

【验收指标】
- IC > 0.05
- IR > 0.3
- 二阶合成因子数量 >= 2
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alpha_research_v136 import run_v136_backtest


def main():
    parser = argparse.ArgumentParser(description="V136 Backtest Runner")
    parser.add_argument(
        '--data-path',
        type=str,
        default="data/parquet/features_latest.parquet",
        help='Path to Parquet data file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--ic-threshold',
        type=float,
        default=0.023,
        help='IC threshold for factor selection'
    )
    parser.add_argument(
        '--n-factors',
        type=int,
        default=5,
        help='Number of factors to select'
    )
    
    args = parser.parse_args()
    
    result = run_v136_backtest(
        data_path=args.data_path,
        output_dir=args.output_dir,
        ic_threshold=args.ic_threshold,
        n_factors=args.n_factors,
    )
    
    print(f"V136 Backtest Complete!")
    print(f"Result: {result}")


if __name__ == '__main__':
    main()