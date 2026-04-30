"""
Run V229 - Backtest Runner (Industry-Relative Reversal)
=========================================================

【使用方法】
python -u run_v229.py --years 2020 2022 2024

【V229 假设 H2】
- 行业轮动因子可以捕捉 2024 年牛市中的 alpha
- 行业内相对弱势的股票更容易反弹
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

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.alpha_model_v229 import AlphaModelV229, get_alpha_model
from src.backtest_engine import BacktestEngine, get_backtest_engine

VERSION = "V229"

EXPERIMENT_LOG = project_root / "experiment_metadata.csv"
ALPHA_HISTORY = project_root / "ALPHA_HISTORY.md"


def setup_logging():
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    logger.add("reports/v229_run_{time:YYYYMMDD}.log", level="DEBUG", rotation="10 MB", retention="30 days")


def clean_old_versions():
    logger.info("Phase 1: Environment Cleanup")
    # 清理旧版本报告 (保留 README.md)
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        # 删除旧版本的 V229 报告 (保留 V227 基线报告和 README.md)
        for report in list(reports_dir.glob("V229_Cross_Year_Report_*.md")) + list(reports_dir.glob("v229_year*_audit_*.md")):
            try:
                os.remove(report)
                logger.info(f"  Deleted old report: {report.name}")
            except:
                pass


def update_experiment_log(version, core_logic, ic_2020, ic_2022, ic_2024, turn_over, failure_reason):
    if not EXPERIMENT_LOG.exists():
        with open(EXPERIMENT_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Version', 'Core_Logic', '2020_IC', '2022_IC', '2024_IC', 'Turn_Over', 'Failure_Reason', 'Timestamp'])
    with open(EXPERIMENT_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([version, core_logic, f"{ic_2020:.4f}", f"{ic_2022:.4f}", f"{ic_2024:.4f}", f"{turn_over:.2%}", failure_reason, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])


def update_alpha_history(version, core_logic, ic_2020, ic_2022, ic_2024, ic_ir_2020, ic_ir_2022, ic_ir_2024, ann_ret_2020, ann_ret_2022, ann_ret_2024, max_dd_2020, max_dd_2022, max_dd_2024, passed, failure_reason, lessons):
    status = "PASS" if passed else "FAIL"
    today = datetime.now().strftime('%Y-%m-%d')
    avg_ic = (ic_2020 + ic_2022 + ic_2024) / 3
    entry = f"\n## {version} (H2: 行业相对反转) - {today}\n- **假设**: 行业轮动因子可以捕捉 2024 年牛市中的 alpha；行业内相对弱势的股票更容易反弹\n- **因子数量**: 3 (极端超卖 35% + 行业相对弱势 35% + 低波动 30%)\n- **结果** (IC/IC IR 数字):\n  - 2020: T+1 IC={ic_2020:.4f}, IC IR={ic_ir_2020:.2f}, 年化={ann_ret_2020:.2%}, MaxDD={max_dd_2020:.2%}\n  - 2022: T+1 IC={ic_2022:.4f}, IC IR={ic_ir_2022:.2f}, 年化={ann_ret_2022:.2%}, MaxDD={max_dd_2022:.2%}\n  - 2024: T+1 IC={ic_2024:.4f}, IC IR={ic_ir_2024:.2f}, 年化={ann_ret_2024:.2%}, MaxDD={max_dd_2024:.2%}\n  - 平均 IC: {avg_ic:.4f}\n- **状态**: {'PASS' if passed else 'FAIL'}\n- **教训**: {lessons}\n"
    with open(ALPHA_HISTORY, 'a', encoding='utf-8') as f:
        f.write(entry)


def analyze_failure(results, years):
    logger.info("Failure Analysis - Auto Diagnosis")
    ic_values = {}
    for year in years:
        if year in results:
            ic = results[year]['t1_ic']['mean_ic']
            ic_values[year] = ic
            logger.info(f"  Year {year}: IC = {ic:.4f}")


def main():
    parser = argparse.ArgumentParser(description="V229 Backtest Runner")
    parser.add_argument('--years', type=int, nargs='+', default=[2020, 2022, 2024])
    parser.add_argument('--output-dir', type=str, default='reports')
    parser.add_argument('--db-url', type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    print(f"\n{'='*70}")
    print(f"V229 Backtest Runner Starting")
    print(f"  Hypothesis H2: Industry-Relative Reversal")
    print(f"  Years: {args.years}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    logger.info("V229 Backtest Runner Starting")
    logger.info(f"  Years: {args.years}")

    clean_old_versions()

    alpha_model = get_alpha_model()
    engine = get_backtest_engine(output_dir=args.output_dir, db_url=args.db_url)

    try:
        print(f"\n[Phase 2] Loading data...")
        sys.stdout.flush()
        warmup_year = min(args.years) - 1
        df = engine.load_data(years=args.years, warmup_year=warmup_year, warmup_days=60)
        if df.empty:
            logger.error("No data loaded. Exiting.")
            print("ERROR: No data loaded. Exiting.")
            sys.stdout.flush()
            sys.exit(1)
        print(f"  Data loaded: {len(df)} rows, {df['symbol'].nunique()} symbols")
        sys.stdout.flush()

        print(f"\n[Phase 3] Running cross-year audit...")
        sys.stdout.flush()
        audit_results = engine.run_cross_year_audit(df, alpha_model, args.years)
        results = audit_results['results']

        print(f"\n[Phase 4] Analyzing results...")
        sys.stdout.flush()
        ic_2020 = ic_2022 = ic_2024 = 0.0
        ic_ir_2020 = ic_ir_2022 = ic_ir_2024 = 0.0
        ann_ret_2020 = ann_ret_2022 = ann_ret_2024 = 0.0
        max_dd_2020 = max_dd_2022 = max_dd_2024 = 0.0
        all_passed = True

        print(f"\n{'='*70}")
        print(f"{'年份':<6} | {'T+1 IC':<10} | {'IC IR':<10} | {'年化收益':<10} | {'最大回撤':<10} | {'状态':<6}")
        print(f"{'-'*70}")

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
                logger.info(f"  Year {year}: T+1 IC={t1_ic:.4f}, IC IR={ic_ir:.2f}, AnnRet={ann_ret:.2%}, MaxDD={max_dd:.2%}, Status={status}")
                print(f"  {year:<6} | {t1_ic:<10.4f} | {ic_ir:<10.2f} | {ann_ret:<10.2%} | {max_dd:<10.2%} | {status:<6}")
                sys.stdout.flush()
                if year == 2020: ic_2020, ic_ir_2020, ann_ret_2020, max_dd_2020 = t1_ic, ic_ir, ann_ret, max_dd
                elif year == 2022: ic_2022, ic_ir_2022, ann_ret_2022, max_dd_2022 = t1_ic, ic_ir, ann_ret, max_dd
                elif year == 2024: ic_2024, ic_ir_2024, ann_ret_2024, max_dd_2024 = t1_ic, ic_ir, ann_ret, max_dd

        print(f"{'-'*70}")
        avg_ic = (ic_2020 + ic_2022 + ic_2024) / 3
        print(f"  {'平均':<6} | {avg_ic:<10.4f} |")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        logger.info(f"  Average IC: {avg_ic:.4f}")

        if not all_passed:
            analyze_failure(results, args.years)

        core_logic = "Industry-Relative Reversal (Oversold 35% + IndRel 35% + LowVol 30%)"
        failure_reason = f"IC below threshold - Avg IC={avg_ic:.4f}, 2024 IC={ic_2024:.4f}" if not all_passed else ""
        lessons = ""
        if ic_2024 < 0.05: lessons += "2024年因子失效; "
        if avg_ic < 0.05: lessons += "平均IC未达0.05; "
        if not lessons: lessons = "行业相对反转方案仍需验证"

        update_experiment_log(VERSION, core_logic, ic_2020, ic_2022, ic_2024, 0.15, failure_reason)
        update_alpha_history(VERSION, core_logic, ic_2020, ic_2022, ic_2024, ic_ir_2020, ic_ir_2022, ic_ir_2024, ann_ret_2020, ann_ret_2022, ann_ret_2024, max_dd_2020, max_dd_2022, max_dd_2024, all_passed, failure_reason, lessons)

        logger.info(f"V229 Backtest Complete - Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        logger.info(f"  2020 IC={ic_2020:.4f}, 2022 IC={ic_2022:.4f}, 2024 IC={ic_2024:.4f}, Avg IC={avg_ic:.4f}")
        
        print(f"\nV229 Backtest Complete - Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"  2020 IC={ic_2020:.4f}, 2022 IC={ic_2022:.4f}, 2024 IC={ic_2024:.4f}, Avg IC={avg_ic:.4f}")
        sys.stdout.flush()
        
        sys.exit(0 if all_passed else 1)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nERROR: Backtest failed: {e}")
        sys.stdout.flush()
        update_experiment_log(VERSION, "Industry-Relative Reversal", 0.0, 0.0, 0.0, 0.0, f"Runtime error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()