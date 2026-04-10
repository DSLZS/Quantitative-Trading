"""
V188 回测运行脚本 - V172 精确复制
"""
import sys
sys.path.insert(0, '.')

from src.alpha_research_v188 import V188BacktestRunner

if __name__ == "__main__":
    runner = V188BacktestRunner(output_dir='reports')
    runner.run_cross_cycle_audit(years=[2023, 2024])