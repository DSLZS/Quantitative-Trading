#!/usr/bin/env python3
"""
V139 运行脚本 - 非线性场景切换与极端 Alpha 挖掘.

【使用说明】
运行 2024 年 V139 审计:
    python run_v139.py --year 2024

运行多年份审计:
    python run_v139.py --all

使用 Parquet 数据:
    python run_v139.py --year 2024 --parquet data/parquet/stock_2024.parquet

【V139 核心改进】
1. TailRiskPerception: 尾部风险感知算子 (Skewness + Tail_Risk_Indicator)
2. RegimeAdaptiveGate: 场景自适应门控 (高波动→防御因子，低波动→进攻因子)
3. SignalDelta Orthogonalization: 信号变化量单独正交化
4. DataHealing: 增强数据自愈 (NaN 修复)
5. 目标指标：IC > 0.05, IR > 0.6
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import V139Runner


def main():
    parser = argparse.ArgumentParser(description="V139 运行脚本 - 非线性场景切换与极端 Alpha 挖掘")
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Year to run audit (e.g., 2019, 2021, 2024)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run audit for all years (2019, 2021, 2024)'
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
    
    runner = V139Runner(
        parquet_path=args.parquet,
        output_dir=args.output,
    )
    
    if args.all:
        years = [2019, 2021, 2024]
        summary = runner.run_multi_year_audit(years)
        
        print("\n" + "=" * 70)
        print("V139 Multi-Year Audit Complete!")
        print("=" * 70)
        print(f"  Years: {years}")
        print(f"  Passed: {summary['passed_count']}/{summary['total_count']}")
        print(f"  Cross-Year IC: {summary['cross_year_ic_mean']:.4f} ± {summary['cross_year_ic_std']:.4f}")
        print(f"  Cross-Year IC IR: {summary['cross_year_ic_ir']:.2f}")
        print("=" * 70)
        
    elif args.year:
        result = runner.run_audit(args.year)
        
        print("\n" + "=" * 70)
        print("V139 Audit Complete!")
        print("=" * 70)
        print(f"  Year: {args.year}")
        print(f"  Status: {'PASSED ✓' if result.get('passed', False) else 'FAILED ✗'}")
        print(f"  Report: {result.get('custom_report_path', 'N/A')}")
        print("=" * 70)
        
    else:
        parser.print_help()
        print("\nPlease specify --year or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()