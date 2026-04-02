#!/usr/bin/env python3
"""V139 自测验证脚本"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from alpha_research_v139 import get_alpha_research

def run_self_test():
    """运行 V139 3 轮参数自测"""
    np.random.seed(42)
    
    # 创建测试数据
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
        'momentum_5': np.random.randn(1000),
        'momentum_20': np.random.randn(1000),
        'volatility_5': np.abs(np.random.randn(1000)),
    })
    
    print("=" * 70)
    print("V139 参数自测 - 3 轮测试结果")
    print("=" * 70)
    
    alpha = get_alpha_research()
    self_test_results = alpha.run_self_test(test_df)
    
    passed_count = 0
    for r in self_test_results:
        status = "PASSED" if r['passed'] else "FAILED"
        print(f"Round {r['round']}: IC={r['estimated_ic']:.4f}, "
              f"Improvement={r['ic_improvement']:.4f}, "
              f"Target={r['target_ic']}, Status={status}")
        if r['passed']:
            passed_count += 1
    
    print("=" * 70)
    print(f"自测通过：{passed_count}/{len(self_test_results)}")
    print("=" * 70)
    
    return self_test_results

if __name__ == "__main__":
    run_self_test()