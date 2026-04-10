"""
V191 回测运行脚本

【V191 核心任务】
1. 引入符号回归特征交互 (Feature Cross)
2. 实现 RFE 特征稳定性筛选
3. 全量回测 2023/2024/2025
4. 输出双年度对比表
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_research_v191 import V191BacktestRunner

def main():
    print("=" * 70)
    print("V191 Symbolic Feature Cross & RFE Stability Selection")
    print("=" * 70)
    
    # 创建回测运行器
    runner = V191BacktestRunner(
        output_dir='reports',
        initial_capital=100000.0,      # 锁定 10 万
        commission_rate=0.0013,        # 锁定 1.3‰
        slippage_rate=0.001,           # 锁定 0.1%
    )
    
    # 运行全量回测（2023/2024）
    results = runner.run_full_backtest(years=[2023, 2024])
    
    if results:
        print("\n" + "=" * 70)
        print("V191 回测完成")
        print("=" * 70)
        
        # 输出简要总结
        for year, data in results.items():
            metrics = data.get('metrics', {})
            t1_ic = metrics.get('t1_ic', {}).get('mean_ic', 0.0)
            t1_ir = metrics.get('t1_ic', {}).get('ic_ir', 0.0)
            print(f"  {year}年：T+1 IC = {t1_ic:.4f}, IR = {t1_ir:.2f}")
        
        print("=" * 70)
    else:
        print("回测失败，请检查日志")

if __name__ == "__main__":
    main()