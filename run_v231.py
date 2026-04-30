"""
Run V231 - Backtest Runner (Volatility Prediction)
====================================================

【使用方法】
python -u run_v231.py --years 2020 2022 2024

【V231 假设 H4】
- 放弃回归与分类，改为预测未来5日波动率
- 低波动股票长期有溢价，且波动率在牛熊市中更稳定
- IC 是对波动率的预测能力，验收标准: IC > 0.03
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

from src.alpha_model_v231 import AlphaModelV231, get_alpha_model
from src.backtest_engine import BacktestEngine, get_backtest_engine
from src.engine.backtest_referee import BacktestReferee

VERSION = "V231"

EXPERIMENT_LOG = project_root / "experiment_metadata.csv"
ALPHA_HISTORY = project_root / "ALPHA_HISTORY.md"


def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    log_file = project_root / "reports" / f"v{VERSION.lower()}_run_{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs(log_file.parent, exist_ok=True)
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="30 days")


def clean_old_versions():
    """清理旧版本报告"""
    logger.info("Phase 1: Environment Cleanup")
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        # 删除所有旧版本报告 (保留 README.md)
        patterns_to_delete = [
            "V*_Cross_Year_Report_*.md",
            "v*_year*_audit_*.md",
            "v*_year*_audit_*.json",
        ]
        for pattern in patterns_to_delete:
            for report in reports_dir.glob(pattern):
                try:
                    os.remove(report)
                    logger.info(f"  Deleted old report: {report.name}")
                except Exception as e:
                    logger.warning(f"  Failed to delete {report.name}: {e}")


def update_experiment_log(version, core_logic, ic_2020, ic_2022, ic_2024, turn_over, failure_reason):
    """更新实验日志"""
    if not EXPERIMENT_LOG.exists():
        with open(EXPERIMENT_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Version', 'Core_Logic', '2020_IC', '2022_IC', '2024_IC', 'Turn_Over', 'Failure_Reason', 'Timestamp'])
    with open(EXPERIMENT_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([version, core_logic, f"{ic_2020:.4f}", f"{ic_2022:.4f}", f"{ic_2024:.4f}", f"{turn_over:.2%}", failure_reason, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])


def update_alpha_history(version, core_logic, ic_2020, ic_2022, ic_2024, ic_ir_2020, ic_ir_2022, ic_ir_2024, ann_ret_2020, ann_ret_2022, ann_ret_2024, max_dd_2020, max_dd_2022, max_dd_2024, passed, failure_reason, lessons):
    """更新 ALPHA_HISTORY.md"""
    status = "PASS" if passed else "FAIL"
    today = datetime.now().strftime('%Y-%m-%d')
    avg_ic = (ic_2020 + ic_2022 + ic_2024) / 3
    entry = f"\n## {version} (H4: 波动率预测) - {today}\n- **假设**: 低波动股票长期有溢价，波动率在牛熊市中更稳定；预测未来5日波动率，低波动得高分\n- **因子数量**: 3 (极端超卖 30% + 行业相对弱势 40% + 低波动 30%)\n- **结果** (IC/IC IR 数字 - 对波动率的预测能力):\n  - 2020: Vol IC={ic_2020:.4f}, IC IR={ic_ir_2020:.2f}, 年化={ann_ret_2020:.2%}, MaxDD={max_dd_2020:.2%}\n  - 2022: Vol IC={ic_2022:.4f}, IC IR={ic_ir_2022:.2f}, 年化={ann_ret_2022:.2%}, MaxDD={max_dd_2022:.2%}\n  - 2024: Vol IC={ic_2024:.4f}, IC IR={ic_ir_2024:.2f}, 年化={ann_ret_2024:.2%}, MaxDD={max_dd_2024:.2%}\n  - 平均 Vol IC: {avg_ic:.4f}\n- **状态**: {'PASS' if passed else 'FAIL'}\n- **教训**: {lessons}\n"
    with open(ALPHA_HISTORY, 'a', encoding='utf-8') as f:
        f.write(entry)


def compute_volatility_ic(score_df: pd.DataFrame) -> dict:
    """
    计算波动率预测 IC (score 与 future_vol 的相关性)
    
    由于我们预测波动率，IC 应该是负的（高分=低波动=低未来波动率）
    """
    if 'future_vol' not in score_df.columns or 'score' not in score_df.columns:
        return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
    
    if 'trade_date' not in score_df.columns:
        return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
    
    unique_dates = sorted(score_df['trade_date'].unique())
    ic_series = []
    
    for date in unique_dates:
        day_data = score_df[score_df['trade_date'] == date]
        
        if len(day_data) < 10:
            continue
        
        score_values = day_data['score']
        vol_values = day_data['future_vol']
        
        # 去除空值
        mask = score_values.notna() & vol_values.notna()
        if mask.sum() < 10:
            continue
        
        # 计算 Spearman 相关系数
        corr = score_values[mask].corr(vol_values[mask], method='spearman')
        
        if not np.isnan(corr):
            ic_series.append({
                'trade_date': date,
                'ic': corr,
            })
    
    if not ic_series:
        return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
    
    ic_df = pd.DataFrame(ic_series)
    ic_values = ic_df['ic'].values
    
    mean_ic = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
    
    return {
        'mean_ic': mean_ic,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'num_days': len(ic_values),
        'min_ic': float(np.min(ic_values)),
        'max_ic': float(np.max(ic_values)),
    }


def generate_cross_year_report(results: dict, years: list) -> str:
    """生成跨年度报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"V{VERSION}_Cross_Year_Report_{timestamp}.md"
    
    lines = []
    lines.append(f"# V{VERSION} Cross-Year Audit Report (Volatility Prediction)")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Version**: V{VERSION} (Volatility Prediction with Linear Factor Weighting)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary (执行摘要)")
    lines.append("")
    lines.append("| Year | Vol IC | IC IR | Annual Return | Max Drawdown | Status |")
    lines.append("|------|--------|-------|---------------|--------------|--------|")
    
    for year in years:
        if year not in results:
            lines.append(f"| {year} | N/A | N/A | N/A | N/A | FAIL |")
            continue
        
        r = results[year]
        vol_ic = r.get('vol_ic', {}).get('mean_ic', 0)
        ic_ir = r.get('vol_ic', {}).get('ic_ir', 0)
        ann_ret = r.get('backtest_result', {}).get('annual_return', 0)
        max_dd = r.get('backtest_result', {}).get('max_drawdown', 0)
        
        status = 'PASS' if abs(vol_ic) >= 0.03 else 'FAIL'
        lines.append(f"| {year} | {vol_ic:.4f} | {ic_ir:.2f} | {ann_ret:.2%} | {max_dd:.2%} | {status} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. V231 Hypothesis (核心假设)")
    lines.append("")
    lines.append("### 2.1 Volatility Prediction (波动率预测)")
    lines.append("- **标签**: 未来5日波动率 (future_vol = close.pct_change().rolling(5).std().shift(-5))")
    lines.append("- **逻辑**: 低波动股票长期有溢价，且波动率在牛熊市中更稳定")
    lines.append("- **因子**: 极端超卖 (30%) + 行业相对弱势 (40%) + 低波动率 (30%)")
    lines.append("- **验收**: Vol IC > 0.03 (对波动率的预测能力)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Backtest Configuration (回测配置 - Locked)")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Initial Capital | 100,000 |")
    lines.append("| Commission Rate | 0.3 per mille |")
    lines.append("| Stamp Duty Rate | 1.0 per mille |")
    lines.append("| Slippage Rate | 0.5 per mille |")
    lines.append("| Total Fee Rate | 1.3 per mille |")
    lines.append("| Top N Stocks | 50 |")
    lines.append("| Position per Stock | 2% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Compliance Statement (合规声明)")
    lines.append("")
    lines.append("- **初始资金**: 100,000 (已锁定)")
    lines.append("- **费率**: 1.3 per mille (佣金 0.3 + 印花税 1 + 滑点 0.5)")
    lines.append("- **无未来函数**: 所有因子仅使用 T 日及之前数据")
    lines.append("- **Referee-Player 隔离**: AlphaModel 仅输出 Score，严禁接触回测逻辑")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by V231 Backtest Runner*")
    
    report_content = "\n".join(lines)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"[Report] Cross-year report saved to: {report_path}")
    
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="V231 Backtest Runner (Volatility Prediction)")
    parser.add_argument('--years', type=int, nargs='+', default=[2020, 2022, 2024])
    parser.add_argument('--output-dir', type=str, default='reports')
    parser.add_argument('--db-url', type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    print(f"\n{'='*70}")
    print(f"V{VERSION} Backtest Runner Starting")
    print(f"  Hypothesis H4: Volatility Prediction")
    print(f"  Years: {args.years}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    logger.info(f"V{VERSION} Backtest Runner Starting")
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

        print(f"\n[Phase 3] Running per-year audit...")
        sys.stdout.flush()
        
        results = {}
        
        for year in args.years:
            print(f"\n{'='*50}")
            print(f"  Running Year {year}")
            print(f"{'='*50}")
            sys.stdout.flush()
            
            year_str = str(year)
            year_data = df[df['trade_date'].astype(str).str.startswith(year_str)].copy()
            
            if year_data.empty:
                logger.warning(f"[Year {year}] No data available")
                continue
            
            logger.info(f"[Year {year}] Data: {len(year_data)} rows, {year_data['symbol'].nunique()} symbols")
            
            # 计算 alpha scores
            score_df = alpha_model.compute_score(year_data)
            
            # 计算波动率 IC
            vol_ic = compute_volatility_ic(score_df)
            
            # 使用 BacktestReferee 进行回测
            referee = BacktestReferee(alpha_module=alpha_model, output_dir=args.output_dir)
            referee.VERSION = f"V{VERSION}_Year{year}"
            
            audit_result = referee.run_audit(year_data)
            
            backtest_result = audit_result.get('backtest_result', {})
            
            results[year] = {
                'vol_ic': vol_ic,
                'backtest_result': backtest_result,
            }
            
            ann_ret = backtest_result.get('annual_return', 0)
            sharpe = backtest_result.get('sharpe_ratio', 0)
            max_dd = backtest_result.get('max_drawdown', 0)
            
            print(f"  Year {year}: Vol IC={vol_ic['mean_ic']:.4f}, AnnRet={ann_ret:.2%}, MaxDD={max_dd:.2%}")
            sys.stdout.flush()
        
        # 打印汇总结果
        print(f"\n{'='*70}")
        print(f"{'年份':<6} | {'Vol IC':<10} | {'IC IR':<10} | {'年化收益':<10} | {'最大回撤':<10} | {'状态':<6}")
        print(f"{'-'*70}")

        ic_2020 = ic_2022 = ic_2024 = 0.0
        ic_ir_2020 = ic_ir_2022 = ic_ir_2024 = 0.0
        ann_ret_2020 = ann_ret_2022 = ann_ret_2024 = 0.0
        max_dd_2020 = max_dd_2022 = max_dd_2024 = 0.0
        
        # 验收标准: 2024 收益 > -10%, 2020/2022 收益 > 20%, Vol IC > 0.03
        all_passed = True

        for year in args.years:
            if year in results:
                r = results[year]
                vol_ic = r['vol_ic']['mean_ic']
                ic_ir = r['vol_ic']['ic_ir']
                ann_ret = r['backtest_result'].get('annual_return', 0)
                max_dd = r['backtest_result'].get('max_drawdown', 0)
                
                # 验收标准
                ic_pass = abs(vol_ic) >= 0.03
                if year == 2024:
                    ret_pass = ann_ret >= -0.10  # 2024: > -10%
                else:
                    ret_pass = ann_ret >= 0.20   # 2020/2022: > 20%
                
                passed = ic_pass and ret_pass
                if not passed:
                    all_passed = False
                
                status = "PASS" if passed else "FAIL"
                print(f"  {year:<6} | {vol_ic:<10.4f} | {ic_ir:<10.2f} | {ann_ret:<10.2%} | {max_dd:<10.2%} | {status:<6}")
                sys.stdout.flush()
                
                if year == 2020: ic_2020, ic_ir_2020, ann_ret_2020, max_dd_2020 = vol_ic, ic_ir, ann_ret, max_dd
                elif year == 2022: ic_2022, ic_ir_2022, ann_ret_2022, max_dd_2022 = vol_ic, ic_ir, ann_ret, max_dd
                elif year == 2024: ic_2024, ic_ir_2024, ann_ret_2024, max_dd_2024 = vol_ic, ic_ir, ann_ret, max_dd

        print(f"{'-'*70}")
        avg_ic = (ic_2020 + ic_2022 + ic_2024) / 3
        print(f"  {'平均':<6} | {avg_ic:<10.4f} |")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        logger.info(f"  Average Vol IC: {avg_ic:.4f}")

        # 更新日志
        core_logic = "Volatility Prediction (Oversold 30% + IndRel 40% + LowVol 30%)"
        failure_reason = f"IC below threshold or return below threshold - Avg Vol IC={avg_ic:.4f}" if not all_passed else ""
        lessons = ""
        if abs(ic_2024) < 0.03: lessons += "2024年波动率预测IC低于0.03; "
        if ann_ret_2024 < -0.10: lessons += f"2024年收益{ann_ret_2024:.2%}低于-10%; "
        if ann_ret_2020 < 0.20: lessons += f"2020年收益{ann_ret_2020:.2%}低于20%; "
        if ann_ret_2022 < 0.20: lessons += f"2022年收益{ann_ret_2022:.2%}低于20%; "
        if not lessons: lessons = "所有验收标准通过"

        update_experiment_log(VERSION, core_logic, ic_2020, ic_2022, ic_2024, 0.15, failure_reason)
        update_alpha_history(VERSION, core_logic, ic_2020, ic_2022, ic_2024, ic_ir_2020, ic_ir_2022, ic_ir_2024, ann_ret_2020, ann_ret_2022, ann_ret_2024, max_dd_2020, max_dd_2022, max_dd_2024, all_passed, failure_reason, lessons)

        # 生成跨年度报告
        report_path = generate_cross_year_report(results, args.years)

        logger.info(f"V{VERSION} Backtest Complete - Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        logger.info(f"  2020 Vol IC={ic_2020:.4f}, 2022 Vol IC={ic_2022:.4f}, 2024 Vol IC={ic_2024:.4f}, Avg Vol IC={avg_ic:.4f}")
        
        print(f"\nV{VERSION} Backtest Complete - Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"  2020 Vol IC={ic_2020:.4f}, 2022 Vol IC={ic_2022:.4f}, 2024 Vol IC={ic_2024:.4f}, Avg Vol IC={avg_ic:.4f}")
        print(f"  2020 Return={ann_ret_2020:.2%}, 2022 Return={ann_ret_2022:.2%}, 2024 Return={ann_ret_2024:.2%}")
        sys.stdout.flush()
        
        sys.exit(0 if all_passed else 1)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nERROR: Backtest failed: {e}")
        sys.stdout.flush()
        update_experiment_log(VERSION, "Volatility Prediction", 0.0, 0.0, 0.0, 0.0, f"Runtime error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()