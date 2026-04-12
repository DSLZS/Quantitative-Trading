"""
V194 回测运行脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_research_v194 import main

if __name__ == "__main__":
    main()