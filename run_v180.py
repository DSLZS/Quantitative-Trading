"""
V180 策略优化与回测主轴运行器

【运行方式】
python run_v180.py

【功能】
1. 执行 2023-2024 跨周期审计
2. 实时输出 Sharpe Ratio 和 Max Drawdown
3. 自动执行自省逻辑（若不达标）
4. 生成 Acceptance Report
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_research_v180 import main

if __name__ == "__main__":
    exit(main())