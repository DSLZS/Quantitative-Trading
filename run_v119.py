"""
V119 回测运行脚本 - 特征多样性喷发.
"""

import json
from src.alpha_research_v119 import run_v119_backtest, DataHealingError
from loguru import logger

if __name__ == "__main__":
    try:
        logger.info("=" * 80)
        logger.info("V119 Backtest - Starting")
        logger.info("=" * 80)
        
        result = run_v119_backtest(
            data_path="data/parquet/features_latest.parquet",
            output_dir="reports"
        )
        
        # 输出关键指标
        print("\n" + "=" * 80)
        print("V119 BACKTEST RESULTS")
        print("=" * 80)
        
        if 't1_ic' in result:
            print(f"\nT+1 Rank IC Metrics:")
            print(f"  Mean IC: {result['t1_ic'].get('mean_ic', 0):.4f}")
            print(f"  IC Std:  {result['t1_ic'].get('ic_std', 0):.4f}")
            print(f"  IC IR:   {result['t1_ic'].get('ic_ir', 0):.2f}")
            print(f"  Target:  > 0.05")
        
        if 'ic_decay' in result:
            print(f"\nIC Decay Analysis:")
            print(f"  {result['ic_decay'].get('decay_pattern', 'N/A')}")
            print(f"  Monotonic: {result['ic_decay'].get('is_monotonic', False)}")
        
        if 'backtest_result' in result:
            print(f"\nBacktest Performance:")
            print(f"  Total Return:  {result['backtest_result'].get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio:  {result['backtest_result'].get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown:  {result['backtest_result'].get('max_drawdown', 0):.2%}")
        
        if 'factor_ics' in result:
            print(f"\nGenetic Factor ICs:")
            for factor_name, ic in sorted(result['factor_ics'].items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                status = '✓' if abs(ic) > 0.02 else '✗'
                print(f"  {factor_name}: {ic:.4f} {status}")
        
        print(f"\nOverall Status: {'PASSED ✓' if result.get('passed', False) else 'FAILED ✗'}")
        print(f"Report saved to: {result.get('report_path', 'N/A')}")
        print("=" * 80)
        
        # 保存 JSON 结果
        json_path = "reports/v119_backtest_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"JSON result saved to: {json_path}")
        
    except DataHealingError as e:
        logger.error(f"V119 requires real data: {e}")
        logger.info("Please configure DATABASE_URL or add Parquet files to data/parquet/")
    except Exception as e:
        logger.error(f"V119 Backtest failed: {e}")
        raise