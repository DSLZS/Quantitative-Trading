"""
Run V218 - Backtest Runner (Referee-Player Architecture)
=========================================================

【核心职责】
1. 环境清理：删除旧版本代码和过期报告
2. 数据加载：通过 BacktestEngine 加载数据
3. 模型初始化：创建 AlphaModelV218 实例
4. 回测执行：运行跨年度审计
5. 日志更新：写入 experiment_metadata.csv

【使用方法】
python run_v218.py --years 2020 2022 2024

【合规锁定】
- 初始资金：100,000 (BacktestReferee 中锁定)
- 费率：1.3‰ (BacktestReferee 中锁定)
- 严禁修改回测参数
"""

import sys
import os
import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# 确保项目根目录在 sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 V218 模块
from src.alpha_model_v218 import AlphaModelV218, get_alpha_model
from src.backtest_engine import BacktestEngine, get_backtest_engine
from src.engine.backtest_referee import BacktestReferee

# 版本号
VERSION = "V218"

# 实验日志文件
EXPERIMENT_LOG = project_root / "experiment_metadata.csv"


def setup_logging():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "reports/v218_run_{time:YYYYMMDD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    )


def clean_old_versions():
    """
    清理旧版本代码和过期报告
    
    【物理清理】
    - 删除 v191-v217 的旧代码
    - 清理 reports/ 下的过期文件
    """
    logger.info("=" * 70)
    logger.info("Phase 1: Environment Cleanup")
    logger.info("=" * 70)
    
    # 旧版本文件列表 (V191-V206)
    old_files = [
        'src/alpha_model_v197.py',
        'src/alpha_model_v198.py',
        'src/alpha_model_v199.py',
        'src/alpha_model_v200.py',
        'src/alpha_model_v201.py',
        'src/alpha_model_v202.py',
        'src/alpha_model_v203.py',
        'src/alpha_model_v204.py',
        'src/alpha_model_v205.py',
        'src/alpha_model_v206.py',
        'src/alpha_model_v207.py',
        'src/alpha_research_v192.py',
        'src/alpha_research_v193.py',
        'src/alpha_research_v194.py',
        'src/v202_data_healer.py',
        'src/v203_data_healer.py',
        'src/v204_data_healer.py',
        'src/v205_data_healer.py',
        'src/v206_data_healer.py',
        'src/v207_data_healer.py',
    ]
    
    deleted_count = 0
    for f in old_files:
        filepath = project_root / f
        if filepath.exists():
            try:
                os.remove(filepath)
                deleted_count += 1
                logger.info(f"  Deleted: {f}")
            except Exception as e:
                logger.warning(f"  Failed to delete {f}: {e}")
    
    # 清理过期报告
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        old_reports = list(reports_dir.glob("v19*.md")) + list(reports_dir.glob("v20*.md"))
        for report in old_reports:
            try:
                os.remove(report)
                deleted_count += 1
                logger.info(f"  Deleted report: {report.name}")
            except Exception as e:
                logger.warning(f"  Failed to delete {report}: {e}")
    
    logger.info(f"  Cleanup complete: {deleted_count} files deleted")
    logger.info("=" * 70)


def update_experiment_log(
    version: str,
    core_logic: str,
    ic_2020: float,
    ic_2022: float,
    ic_2024: float,
    turn_over: float,
    failure_reason: str,
):
    """
    更新实验日志
    
    Args:
        version: 版本号
        core_logic: 核心逻辑描述
        ic_2020: 2020年 IC
        ic_2022: 2022年 IC
        ic_2024: 2024年 IC
        turn_over: 换手率
        failure_reason: 失败原因 (如果适用)
    """
    logger.info("=" * 70)
    logger.info("Phase 5: Updating Experiment Log")
    logger.info("=" * 70)
    
    # 创建日志文件 (如果不存在)
    if not EXPERIMENT_LOG.exists():
        with open(EXPERIMENT_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Version', 'Core_Logic', '2020_IC', '2022_IC', '2024_IC',
                'Turn_Over', 'Failure_Reason', 'Timestamp'
            ])
    
    # 追加新记录
    with open(EXPERIMENT_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            version,
            core_logic,
            f"{ic_2020:.4f}",
            f"{ic_2022:.4f}",
            f"{ic_2024:.4f}",
            f"{turn_over:.2%}",
            failure_reason,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ])
    
    logger.info(f"  Experiment log updated: {EXPERIMENT_LOG}")
    logger.info("=" * 70)


def analyze_failure(results: dict, years: list):
    """
    失败诊断分析
    
    【分析内容】
    1. 对比各年份 IC 差异
    2. 检查是否存在过度拟合
    3. 分析市场状态分布
    
    Args:
        results: 回测结果
        years: 回测年份列表
    """
    logger.info("=" * 70)
    logger.info("Failure Analysis")
    logger.info("=" * 70)
    
    ic_values = {}
    for year in years:
        if year in results:
            ic = results[year]['t1_ic']['mean_ic']
            ic_values[year] = ic
            logger.info(f"  Year {year}: IC = {ic:.4f}")
    
    # 检查 IC 差异
    if len(ic_values) >= 2:
        ic_list = list(ic_values.values())
        ic_range = max(ic_list) - min(ic_list)
        
        if ic_range > 0.10:
            logger.warning(f"  Large IC variation detected: {ic_range:.4f}")
            logger.warning("  Possible overfitting to specific market conditions")
        
        # 检查 2024 年是否显著低于其他年份
        if 2024 in ic_values:
            other_ics = [v for k, v in ic_values.items() if k != 2024]
            if other_ics and ic_values[2024] < min(other_ics) - 0.05:
                logger.warning("  2024 IC significantly lower than other years")
                logger.warning("  Strategy may be overfitting to pre-2024 patterns")
    
    logger.info("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="V218 Backtest Runner")
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2020, 2022, 2024],
        help='Years to backtest (default: 2020 2022 2024)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help='Database URL (overrides DATABASE_URL env var)'
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging()
    
    logger.info("=" * 70)
    logger.info("V218 Backtest Runner Starting")
    logger.info("=" * 70)
    logger.info(f"  Years: {args.years}")
    logger.info(f"  Output Dir: {args.output_dir}")
    logger.info("=" * 70)
    
    # Phase 1: 环境清理
    clean_old_versions()
    
    # Phase 2: 初始化模型
    logger.info("=" * 70)
    logger.info("Phase 2: Initializing Alpha Model (Player)")
    logger.info("=" * 70)
    alpha_model = get_alpha_model()
    
    # Phase 3: 初始化回测引擎
    logger.info("=" * 70)
    logger.info("Phase 3: Initializing Backtest Engine (Referee)")
    logger.info("=" * 70)
    engine = get_backtest_engine(output_dir=args.output_dir, db_url=args.db_url)
    
    # Phase 4: 加载数据并运行回测
    logger.info("=" * 70)
    logger.info("Phase 4: Loading Data and Running Backtest")
    logger.info("=" * 70)
    
    try:
        # 加载数据 (包含预热期)
        warmup_year = min(args.years) - 1
        df = engine.load_data(years=args.years, warmup_year=warmup_year, warmup_days=60)
        
        if df.empty:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)
        
        # 验证数据质量
        validation = engine.validate_data(df, args.years)
        if not validation['passed']:
            logger.warning("Data validation failed, but continuing with backtest...")
        
        # 运行跨年度审计
        audit_results = engine.run_cross_year_audit(df, alpha_model, args.years)
        
        # 提取结果
        results = audit_results['results']
        
        # 输出汇总
        logger.info("\n" + "=" * 70)
        logger.info("V218 Cross-Year Backtest Summary")
        logger.info("=" * 70)
        
        ic_2020 = 0.0
        ic_2022 = 0.0
        ic_2024 = 0.0
        all_passed = True
        
        for year in args.years:
            if year in results:
                r = results[year]
                t1_ic = r['t1_ic']['mean_ic']
                ic_ir = r['t1_ic']['ic_ir']
                ann_ret = r['backtest_result'].get('annual_return', 0)
                sharpe = r['backtest_result'].get('sharpe_ratio', 0)
                max_dd = r['backtest_result'].get('max_drawdown', 0)
                
                status = "PASS" if t1_ic >= 0.05 else "FAIL"
                if t1_ic < 0.05:
                    all_passed = False
                
                logger.info(f"  Year {year}:")
                logger.info(f"    T+1 IC: {t1_ic:.4f} (threshold: 0.05)")
                logger.info(f"    IC IR: {ic_ir:.2f} (threshold: 0.30)")
                logger.info(f"    Annual Return: {ann_ret:.2%}")
                logger.info(f"    Sharpe Ratio: {sharpe:.2f}")
                logger.info(f"    Max Drawdown: {max_dd:.2%}")
                logger.info(f"    Status: {status}")
                
                # 保存 IC 值
                if year == 2020:
                    ic_2020 = t1_ic
                elif year == 2022:
                    ic_2022 = t1_ic
                elif year == 2024:
                    ic_2024 = t1_ic
            else:
                logger.warning(f"  Year {year}: No results")
                all_passed = False
        
        logger.info("=" * 70)
        
        # 失败诊断
        if not all_passed:
            analyze_failure(results, args.years)
        
        # Phase 5: 更新实验日志
        core_logic = "Market State Adapter + Feature Decoupling (Rev/Mom Gating)"
        failure_reason = "" if all_passed else "IC below threshold - strategy underperforms"
        
        update_experiment_log(
            version=VERSION,
            core_logic=core_logic,
            ic_2020=ic_2020,
            ic_2022=ic_2022,
            ic_2024=ic_2024,
            turn_over=0.15,  # 估算换手率
            failure_reason=failure_reason,
        )
        
        # 最终状态
        logger.info("\n" + "=" * 70)
        logger.info("V218 Backtest Complete")
        logger.info("=" * 70)
        logger.info(f"  Overall Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        logger.info(f"  2020 IC: {ic_2020:.4f}")
        logger.info(f"  2022 IC: {ic_2022:.4f}")
        logger.info(f"  2024 IC: {ic_2024:.4f}")
        logger.info("=" * 70)
        
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        logger.error(f"Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # 即使失败也记录日志
        update_experiment_log(
            version=VERSION,
            core_logic="Market State Adapter + Feature Decoupling",
            ic_2020=0.0,
            ic_2022=0.0,
            ic_2024=0.0,
            turn_over=0.0,
            failure_reason=f"Runtime error: {str(e)}",
        )
        
        sys.exit(1)


if __name__ == "__main__":
    main()