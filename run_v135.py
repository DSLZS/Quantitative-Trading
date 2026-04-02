"""V135 回测运行脚本 - 低阈值增强版"""
import json
from src.alpha_research_v135 import run_v135_backtest
from loguru import logger

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("V135 Backtest - Starting (Low Threshold)")
    logger.info("=" * 80)
    
    result = run_v135_backtest(
        data_path="data/parquet/features_latest.parquet",
        output_dir="reports",
        ic_threshold=0.023,
        n_factors=5
    )
    
    print("\n" + "=" * 80)
    print("V135 BACKTEST RESULTS")
    print("=" * 80)
    
    if 't1_ic' in result:
        print(f"\nT+1 Rank IC Metrics:")
        print(f"  Mean IC: {result['t1_ic'].get('mean_ic', 0):.4f}")
        print(f"  IC Std:  {result['t1_ic'].get('ic_std', 0):.4f}")
        print(f"  IC IR:   {result['t1_ic'].get('ic_ir', 0):.2f}")
        print(f"  Target:  IC>0.05, IR>0.6")
    
    if 'factor_ics' in result:
        print(f"\nFactor ICs (Top 15):")
        for factor_name, ic in sorted(result['factor_ics'].items(), key=lambda x: abs(x[1]), reverse=True)[:15]:
            status = '+' if ic > 0 else '-'
            print(f"  {factor_name}: {ic:.4f} {status}")
    
    print(f"\nOverall Status: {'PASSED' if result.get('passed', False) else 'FAILED'}")
    print(f"Report: {result.get('report_path', 'N/A')}")
    print("=" * 80)