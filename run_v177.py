#!/usr/bin/env python3
"""
V177 统一回测运行器 - 暴力数据补全与跨周期对齐.

【使用说明】
- 数据补全：python run_v177.py --heal
- 回测运行：python run_v177.py --all
- 检查数据：python run_v177.py --check

【V177 强制目标】
- 2023 年 stock_daily > 500,000 行
- 2023/2024 双年份 IC 均 > 0.08
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger
import pandas as pd

from alpha_research_v177 import V177Runner, TushareHealerV177, SQL_HEALER_MIN_ROWS_2023

load_dotenv()

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def check_data_count():
    """检查数据库中的数据行数"""
    logger.info("=" * 70)
    logger.info("V177 Data Count Check")
    logger.info("=" * 70)
    
    try:
        from sqlalchemy import create_engine, text
        import os
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL not configured!")
            return
        
        engine = create_engine(db_url)
        
        # 检查 2023 年
        query_2023 = text("""
            SELECT COUNT(*) as cnt FROM stock_daily
            WHERE trade_date BETWEEN '20230101' AND '20231231'
        """)
        result_2023 = pd.read_sql_query(query_2023, engine)
        count_2023 = result_2023['cnt'].values[0]
        
        # 检查 2024 年
        query_2024 = text("""
            SELECT COUNT(*) as cnt FROM stock_daily
            WHERE trade_date BETWEEN '20240101' AND '20241231'
        """)
        result_2024 = pd.read_sql_query(query_2024, engine)
        count_2024 = result_2024['cnt'].values[0]
        
        # 检查资金流数据
        query_fund = text("""
            SELECT COUNT(*) as cnt FROM stock_fund_flow
            WHERE trade_date BETWEEN '20230101' AND '20241231'
        """)
        result_fund = pd.read_sql_query(query_fund, engine)
        count_fund = result_fund['cnt'].values[0]
        
        logger.info(f"2023 stock_daily rows: {count_2023:,} (target: >{SQL_HEALER_MIN_ROWS_2023:,})")
        logger.info(f"2024 stock_daily rows: {count_2024:,}")
        logger.info(f"stock_fund_flow rows (2023-2024): {count_fund:,}")
        
        status_2023 = "✓ PASSED" if count_2023 >= SQL_HEALER_MIN_ROWS_2023 else "✗ FAILED"
        logger.info(f"2023 Data Status: {status_2023}")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to check data count: {e}")


def run_healing():
    """运行数据补全"""
    logger.info("=" * 70)
    logger.info("V177 Data Healing - Violent Batch Fetching")
    logger.info("=" * 70)
    
    try:
        import os
        db_url = os.getenv("DATABASE_URL")
        tushare_token = os.getenv("TUSHARE_TOKEN")
        
        if not db_url:
            logger.error("DATABASE_URL not configured!")
            return
        
        if not tushare_token:
            logger.error("TUSHARE_TOKEN not configured!")
            return
        
        healer = TushareHealerV177(db_url=db_url, tushare_token=tushare_token)
        
        # 补全 2023 年数据
        logger.info("\n[Step 1] Healing 2023 data...")
        df_2023 = healer.fetch_and_heal_year(2023)
        logger.info(f"2023 data rows after healing: {len(df_2023):,}")
        
        # 补全 2024 年数据
        logger.info("\n[Step 2] Healing 2024 data...")
        df_2024 = healer.fetch_and_heal_year(2024)
        logger.info(f"2024 data rows after healing: {len(df_2024):,}")
        
        # 验证结果
        logger.info("\n[Step 3] Verifying data...")
        check_data_count()
        
        logger.info("=" * 70)
        logger.info("V177 Data Healing Complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Data healing failed: {e}")


def run_backtest():
    """运行回测"""
    logger.info("=" * 70)
    logger.info("V177 Backtest - Cross-Cycle OOS Validation")
    logger.info("=" * 70)
    
    try:
        runner = V177Runner(output_dir='reports')
        
        # 运行跨周期审计
        summary = runner.run_cross_cycle_audit(years=[2023, 2024])
        
        # 输出验证结果
        validation = summary.get('validation_passed', {})
        
        logger.info("\n" + "=" * 70)
        logger.info("V177 Cross-Cycle Audit Complete!")
        logger.info("=" * 70)
        logger.info(f"2023 IC Target (>0.08): {'MET ✓' if validation.get('2023', {}).get('passed', False) else 'NOT MET ✗'}")
        logger.info(f"2023 Data Rows Target (>{SQL_HEALER_MIN_ROWS_2023:,}): {'MET ✓' if validation.get('2023', {}).get('min_rows_actual', 0) >= SQL_HEALER_MIN_ROWS_2023 else 'NOT MET ✗'}")
        logger.info(f"2024 IC Target (>0.08): {'MET ✓' if validation.get('2024', {}).get('passed', False) else 'NOT MET ✗'}")
        logger.info(f"Overall Validation: {'PASSED ✓' if validation.get('overall_passed', False) else 'FAILED ✗'}")
        logger.info("=" * 70)
        
    except SystemExit as e:
        logger.error(f"Backtest exited with code {e.code}")
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="V177 Unified Runner")
    parser.add_argument('--heal', action='store_true', help='Run data healing')
    parser.add_argument('--all', action='store_true', help='Run backtest for all years')
    parser.add_argument('--check', action='store_true', help='Check data count only')
    
    args = parser.parse_args()
    
    if args.check:
        check_data_count()
    elif args.heal:
        run_healing()
    elif args.all:
        run_backtest()
    else:
        parser.print_help()
        logger.warning("Please specify --heal, --all, or --check")
        sys.exit(1)


if __name__ == '__main__':
    main()