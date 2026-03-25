"""
项目清理与架构规范化脚本

【功能】
1. 创建 archive/ 文件夹
2. 移动 V1-V63 旧版本文件到 archive/
3. 保留 V64, V67, V68, V69, V70, V71 及核心文件
4. 创建新的目录结构 src/core/, src/engine/, src/loaders/
5. 更新 README.md

作者：量化系统
版本：V72.0
日期：2026-03-25
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Set


# ===========================================
# 配置
# ===========================================

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
ARCHIVE_DIR = SRC_DIR / "archive"

# 保留的文件（不归档）
KEEP_FILES = {
    # 核心底层文件
    "__init__.py",
    "db_manager.py",
    "data_loader.py",
    "backtest_engine.py",
    "backtester.py",
    "visualizer.py",
    "predict_next_day.py",
    "daily_trade_advisor.py",
    "preflight_check.py",
    "final_stress_tester.py",
    "execution_optimizer.py",
    "walk_forward_backtester.py",
    "walk_forward_backtester_v2.py",
    "generate_optimization_report.py",
    "generate_wfa_report.py",
    # V64+ 保留版本
    "data_fetcher_v64.py",
    "v64_core.py",
    "v64_engine.py",
    "run_v64_backtest.py",
    "v67_core.py",
    "v67_engine.py",
    "v67_data_filler.py",
    "run_v67_backtest.py",
    "v68_core.py",
    "v68_engine.py",
    "v68_data_force_filler.py",
    "run_v68_backtest.py",
    "v69_core.py",
    "v69_engine.py",
    "v69_data_boot.py",
    "run_v69_backtest.py",
    "v70_core.py",
    "v70_engine.py",
    "v70_data_loader.py",
    "v71_raw_fetcher.py",
    "v71_db_validator.py",
    "v71_tushare_boot.py",
    # 比较报告脚本
    "v62_vs_v70_comparison.py",
    # 数据同步脚本
    "sync_all_stocks.py",
    "sync_csi800.py",
    "sync_etf_data.py",
    "sync_index_data.py",
    "sync_industry_and_market_cap.py",
    "sync_stock_metadata.py",
}


def should_archive(filename: str) -> bool:
    """判断文件是否应该归档"""
    # 保留的文件不移位
    if filename in KEEP_FILES:
        return False
    
    # 目录不移位
    if filename.endswith('/'):
        return False
    
    # 检查版本号
    # 匹配 vXX_*.py 模式
    match = re.match(r'v(\d+)_', filename.lower())
    if match:
        version = int(match.group(1))
        if version < 64:
            return True
    
    # 匹配 final_strategy_vX*.py 模式
    match = re.match(r'final_strategy_v(\d+)', filename.lower())
    if match:
        version = int(match.group(1))
        if version < 64:
            return True
    
    # 匹配 run_vX_*.py 模式
    match = re.match(r'run_v(\d+)_', filename.lower())
    if match:
        version = int(match.group(1))
        if version < 64:
            return True
    
    return False


def get_files_to_archive(src_dir: Path) -> List[Path]:
    """获取需要归档的文件列表"""
    files_to_archive = []
    
    for item in src_dir.iterdir():
        if item.is_dir():
            continue
        
        if should_archive(item.name):
            files_to_archive.append(item)
    
    return files_to_archive


def create_archive_directory():
    """创建归档目录"""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 创建归档目录：{ARCHIVE_DIR}")


def move_files_to_archive(files: List[Path]):
    """移动文件到归档目录"""
    moved_count = 0
    
    for file_path in files:
        try:
            dest = ARCHIVE_DIR / file_path.name
            shutil.move(str(file_path), str(dest))
            moved_count += 1
        except Exception as e:
            print(f"  ✗ 归档失败 {file_path.name}: {e}")
    
    print(f"✓ 已归档 {moved_count} 个文件到 archive/")


def create_new_structure():
    """创建新的目录结构"""
    new_dirs = {
        "core": "存放核心算法逻辑（SNR、Rank IC 计算）",
        "engine": "存放回测执行引擎",
        "loaders": "存放 Tushare/AkShare 数据同步脚本",
    }
    
    for dir_name, description in new_dirs.items():
        dir_path = SRC_DIR / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            # 创建 __init__.py
            (dir_path / "__init__.py").write_text(f'"""\n{description}\n"""\n')
            print(f"✓ 创建目录：src/{dir_name}/ - {description}")
        else:
            print(f"✓ 目录已存在：src/{dir_name}/")


def move_data_loaders():
    """将数据加载器移动到新位置"""
    loaders_to_move = [
        "v71_tushare_boot.py",
        "v71_raw_fetcher.py",
        "v70_data_loader.py",
        "v69_data_boot.py",
        "v68_data_force_filler.py",
        "v67_data_filler.py",
        "data_fetcher_v64.py",
        "sync_all_stocks.py",
        "sync_csi800.py",
        "sync_etf_data.py",
        "sync_index_data.py",
        "sync_industry_and_market_cap.py",
        "sync_stock_metadata.py",
    ]
    
    loaders_dir = SRC_DIR / "loaders"
    loaders_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for loader in loaders_to_move:
        src = SRC_DIR / loader
        if src.exists():
            dest = loaders_dir / loader
            shutil.move(str(src), str(dest))
            moved_count += 1
            print(f"  移动：{loader} -> loaders/")
    
    print(f"✓ 已移动 {moved_count} 个数据加载器到 loaders/")


def move_core_files():
    """将核心算法文件移动到新位置"""
    core_files = [
        "factor_engine.py",
        "ic_calculator.py",
        "feature_pipeline.py",
        "model_trainer.py",
        "factor_validator.py",
        "oos_validator.py",
        "parameter_scan.py",
    ]
    
    core_dir = SRC_DIR / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for core_file in core_files:
        src = SRC_DIR / core_file
        if src.exists():
            dest = core_dir / core_file
            shutil.move(str(src), str(dest))
            moved_count += 1
            print(f"  移动：{core_file} -> core/")
    
    print(f"✓ 已移动 {moved_count} 个核心文件到 core/")


def move_engine_files():
    """将引擎文件移动到新位置"""
    engine_files = [
        "backtest_engine.py",
        "backtester.py",
        "walk_forward_backtester.py",
        "walk_forward_backtester_v2.py",
        "visualizer.py",
        "execution_optimizer.py",
        "generate_optimization_report.py",
        "generate_wfa_report.py",
    ]
    
    engine_dir = SRC_DIR / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for engine_file in engine_files:
        src = SRC_DIR / engine_file
        if src.exists():
            dest = engine_dir / engine_file
            shutil.move(str(src), str(dest))
            moved_count += 1
            print(f"  移动：{engine_file} -> engine/")
    
    print(f"✓ 已移动 {moved_count} 个引擎文件到 engine/")


def update_readme():
    """更新 README.md"""
    readme_path = PROJECT_ROOT / "README.md"
    
    readme_content = """# 量化交易系统 V72

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📊 数据版本

- **数据源**: Tushare Pro
- **数据覆盖**: 2024 年全年（2024-01-01 至 2024-12-31）
- **个股资金流**: 1,233,188 条记录
- **行业日线**: 106,238 条记录

## 🎯 核心审计指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| SNR (信噪比) | > 0.15 | 信号质量核心指标 |
| Rank IC | > 0.03 | 因子预测能力 |
| 最大回撤 | < 15% | 风险控制指标 |
| 夏普比率 | > 1.0 | 风险调整后收益 |

## 📁 项目结构

```
src/
├── core/              # 核心算法逻辑
│   ├── factor_engine.py      # 因子引擎
│   ├── ic_calculator.py      # IC 计算
│   ├── feature_pipeline.py   # 特征管道
│   ├── model_trainer.py      # 模型训练
│   ├── factor_validator.py   # 因子验证
│   ├── oos_validator.py      # 样本外验证
│   └── parameter_scan.py     # 参数扫描
│
├── engine/            # 回测执行引擎
│   ├── backtest_engine.py    # 回测引擎核心
│   ├── backtester.py         # 回测器
│   ├── visualizer.py         # 可视化
│   ├── execution_optimizer.py # 执行优化
│   └── walk_forward_backtester.py  # 滚动回测
│
├── loaders/           # 数据同步脚本
│   ├── v71_tushare_boot.py   # Tushare 数据补完
│   ├── v70_data_loader.py    # V70 数据加载器
│   └── sync_*.py             # 各类数据同步
│
├── db_manager.py       # 数据库管理器
├── data_loader.py      # 数据加载接口
├── predict_next_day.py # 次日预测
└── daily_trade_advisor.py  # 每日交易顾问
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 Tushare

编辑 `config/settings.yaml`，配置 Tushare Token：

```yaml
TUSHARE_CONFIG = {
    'token': 'your_token_here',
    'rate_limit': 10,
}
```

### 3. 运行回测

```bash
# 运行 V70 回测（推荐）
python src/run_v70_backtest.py

# 运行 V68 回测（稳定版本）
python src/run_v68_backtest.py

# 运行滚动回测
python src/engine/walk_forward_backtester.py
```

### 4. 数据同步

```bash
# 同步 Tushare 数据（2024 年全年）
python src/loaders/v71_tushare_boot.py

# 同步股票元数据
python src/loaders/sync_stock_metadata.py

# 同步行业数据
python src/loaders/sync_industry_and_market_cap.py
```

## 📈 性能报告

生成回测报告后，查看 `reports/` 目录下的最新报告：

```bash
# 查看最新报告
ls -lt reports/*.md | head -1
```

## 🔧 工具脚本

### 项目维护

```bash
# 清理归档旧版本
python src/cleanup_and_refactor.py

# 运行预检查
python src/preflight_check.py

# 因子验证
python src/core/factor_validator.py
```

### 压力测试

```bash
# 运行压力测试
python src/final_stress_tester.py
```

## 📝 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| V72 | 2026-03-25 | 架构规范化，项目瘦身 |
| V71 | 2026-03-25 | Tushare 数据强制补完计划 |
| V70 | 2026-03-24 | 数据加载器优化 |
| V69 | 2026-03-24 | 数据补全与验证 |
| V68 | 2026-03-24 | 稳定版本交付 |

## 📄 License

MIT License
"""
    
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"✓ 已更新 README.md")


def run_cleanup():
    """运行清理任务"""
    print("=" * 60)
    print("项目清理与架构规范化脚本")
    print("=" * 60)
    
    # 1. 获取需要归档的文件
    print("\n步骤 1: 扫描需要归档的文件...")
    files_to_archive = get_files_to_archive(SRC_DIR)
    print(f"  找到 {len(files_to_archive)} 个需要归档的文件")
    
    # 2. 创建归档目录
    print("\n步骤 2: 创建归档目录...")
    create_archive_directory()
    
    # 3. 移动文件到归档目录
    print("\n步骤 3: 归档旧版本文件...")
    move_files_to_archive(files_to_archive)
    
    # 4. 创建新目录结构
    print("\n步骤 4: 创建新目录结构...")
    create_new_structure()
    
    # 5. 移动文件到新位置
    print("\n步骤 5: 移动文件到新位置...")
    print("  移动核心算法文件...")
    move_core_files()
    print("  移动引擎文件...")
    move_engine_files()
    print("  移动数据加载器...")
    move_data_loaders()
    
    # 6. 更新 README
    print("\n步骤 6: 更新 README.md...")
    update_readme()
    
    print("\n" + "=" * 60)
    print("✓ 清理与架构规范化完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_cleanup()