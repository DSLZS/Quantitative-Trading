"""
V32 回测运行脚本

运行 V32 Alpha Strike 策略并生成完整审计报告
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from v32_alpha_strike import main

if __name__ == "__main__":
    main()