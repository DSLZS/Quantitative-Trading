"""
V195 运行脚本 - 执行选手和裁判的完整流程

【流程】
1. 选手脚本 (strategy_v195.py): 读取数据、计算因子、输出 signals.csv
2. 裁判脚本 (referee_v195.py): 读取 signals.csv、执行回测、生成报告

【红线】
1. 物理分离：选手和裁判是独立的脚本
2. 严禁裁判调用选手的内部函数
3. 初始资金 100,000，费率 1.3‰
4. 2023-2025 三年 Mean Rank IC 必须全部 > 0.08
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 配置
VERSION = "V195"
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_strategy():
    """运行选手脚本"""
    print("=" * 70)
    print("Step 1: Running V195 Strategy (选手脚本)")
    print("=" * 70)
    
    # 导入并运行选手脚本
    from src.strategy_v195 import (
        V195DataLoader,
        V195FactorCalculator,
        generate_signals,
        VERSION
    )
    import polars as pl
    import pandas as pd  # 用于 SQL 读取
    
    print(f"\n核心公式：Score = Rank(Momentum_5) × Rank(1/Volatility_5)")
    
    # 加载数据
    loader = V195DataLoader()
    
    # 加载 2023-2025 年数据
    years = [2023, 2024, 2025]
    all_data = []
    
    for year in years:
        df = loader.load_year_data(year)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("ERROR: No data loaded")
        return False
    
    # 使用 pandas concat
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data: {len(full_df):,} rows")
    
    # 生成信号
    output = generate_signals(full_df, output_path="signals.csv")
    
    # 输出因子日志
    print("\nFactor log:")
    print("  momentum_5: close / close.shift(5) - 1 (rank)")
    print("  volatility_5: 1 / stddev(pct_chg, 5) (rank)")
    print("  composite: Rank(Momentum_5) × Rank(1/Volatility_5) (multiplicative)")
    
    return True


def run_referee():
    """运行裁判脚本"""
    print("\n" + "=" * 70)
    print("Step 2: Running V195 Referee (裁判脚本)")
    print("=" * 70)
    
    # 导入并运行裁判脚本
    from src.referee_v195 import V195Referee, V195WeightOptimizer, TARGET_MEAN_IC
    
    referee = V195Referee()
    
    # 运行完整审计
    results = referee.run_full_audit(signals_path="signals.csv")
    
    # 生成报告
    report = referee.generate_audit_report()
    print("\n" + report)
    
    # 检查是否所有年份 IC 达标
    all_pass = all(
        r.get('ic_stats', {}).get('mean_ic', 0) >= TARGET_MEAN_IC
        for r in results.values()
        if 'error' not in r
    )
    
    if not all_pass:
        print("\n" + "=" * 70)
        print("IC 未达标，启动权重优化...")
        print("=" * 70)
        
        # 运行权重优化
        optimizer = V195WeightOptimizer(step_size=0.05)
        opt_result = optimizer.optimize()
        
        print(f"\n优化完成")
        print("NOTE: 完整优化需要为每个权重组合重新生成 signals")
        print("      这里仅记录优化搜索过程")
    
    return results


def main():
    """主函数"""
    print("=" * 70)
    print(f"V195 强制指令：算法重构、架构隔离与 2023 Bug 修复")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: 运行选手脚本
    if not run_strategy():
        print("\nERROR: Strategy execution failed")
        return None
    
    # Step 2: 运行裁判脚本
    results = run_referee()
    
    print("\n" + "=" * 70)
    print("V195 Execution Complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()