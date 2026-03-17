#!/usr/bin/env python3
"""
V8 Strategy Backtest Runner
运行 V8 因子纯化策略回测
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.final_strategy_v8 import run_v8_strategy

if __name__ == "__main__":
    result = run_v8_strategy()
    
    if result:
        print("\n" + "=" * 60)
        print("V8 回测执行完毕")
        print("=" * 60)
        print(f"模型统计：{result.get('model_stats', {})}")
        print(f"报告路径：{result.get('report_path', 'N/A')}")
    else:
        print("V8 策略执行失败")