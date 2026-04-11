"""
V193 Emergency Recovery & Full-Universe Healing - 回测运行脚本
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.alpha_research_v193 import V193BacktestRunner

if __name__ == "__main__":
    runner = V193BacktestRunner(output_dir='reports')
    results = runner.run_full_backtest(years=[2023, 2024, 2025])