"""
V197 主运行脚本 - 三年全量回测 (2023-2025)

【V197 任务目标】
1. 基于市场状态识别的 Alpha 进化
2. 解决 2024 年因子失效问题
3. 目标：2024 年 Rank IC > 0.08

【合规锁定】
- 初始资金：100,000
- 费率：1.3‰
- 无未来函数
"""

import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 导入 V197 Alpha 模型
from alpha_model_v197 import get_alpha_model, AlphaModel, VERSION

# 直接导入 engine.py
import importlib.util
engine_path = project_root / "src" / "engine.py"
spec = importlib.util.spec_from_file_location("engine_core", str(engine_path))
engine_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_core)
get_backtest_engine = engine_core.get_backtest_engine
BacktestEngine = engine_core.BacktestEngine


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info(f"V197 三年全量回测 (2023-2025) - {VERSION}")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化组件
    logger.info("\n[Init] Initializing V197 components...")
    alpha_model = get_alpha_model(
        n_factors=6,
        enable_orm=True,
        enable_gated_residual=True,
        enable_nag=True,
        enable_regime_detection=True,  # V197 新增
    )
    engine = get_backtest_engine(output_dir="reports")
    
    # 加载数据
    years = [2023, 2024, 2025]
    logger.info(f"\n[Data] Loading data for years {years}...")
    df = engine.load_data(years=years, warmup_year=2022, warmup_days=60)
    
    if df.empty:
        logger.error("[Error] Failed to load data. Exiting.")
        return
    
    logger.info(f"[Data] Loaded {len(df)} rows")
    
    # 数据校验
    logger.info("\n[Validate] Validating data...")
    validation = engine.validate_data(df, years)
    
    if not validation['passed']:
        logger.warning(f"[Validate] {len(validation['missing_dates'])} dates need healing")
        df = engine.heal_data(df, validation['missing_dates'])
        validation = engine.validate_data(df, years)
        logger.info(f"[Validate] Data healing complete")
    
    # 执行跨年度审计
    logger.info("\n[Audit] Running V197 cross-year audit...")
    results = engine.run_cross_year_audit(df, alpha_model, years)
    
    # 输出摘要
    logger.info("\n" + "=" * 70)
    logger.info(f"V197 Cross-Year Audit Summary")
    logger.info("=" * 70)
    
    summary_data = []
    for year in years:
        if year not in results['results']:
            continue
        
        r = results['results'][year]
        t1_ic = r['t1_ic']['mean_ic']
        ic_ir = r['t1_ic']['ic_ir']
        ann_ret = r['backtest_result'].get('annual_return', 0)
        sharpe = r['backtest_result'].get('sharpe_ratio', 0)
        
        if year == 2023:
            passed = t1_ic >= 0.08
        elif year == 2024:
            passed = t1_ic >= 0.08  # V197 目标
        else:
            passed = t1_ic > 0.05
        
        status = '✓ PASS' if passed else '✗ FAIL'
        summary_data.append((year, t1_ic, ic_ir, ann_ret, sharpe, status))
    
    logger.info("\n| Year | T+1 IC | IC IR | Ann Return | Sharpe | Status |")
    logger.info("|------|--------|-------|------------|--------|--------|")
    for year, ic, ir, ret, sharpe, status in summary_data:
        logger.info(f"| {year} | {ic:.4f} | {ir:.2f} | {ret:.2%} | {sharpe:.2f} | {status} |")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Report saved to: {results['report_path']}")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    main()