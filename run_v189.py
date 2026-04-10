"""
V189 回测运行脚本 - V172 精确复制 v2（简单 ORM）
"""
import sys
sys.path.insert(0, '.')

from src.alpha_research_v189 import V189BacktestRunner

if __name__ == "__main__":
    runner = V189BacktestRunner(output_dir='reports')
    runner.run_cross_cycle_audit(years=[2023, 2024])