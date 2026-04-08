# V178: DATA FIX MODE - STRICT EXECUTION
"""
V178 数据补全运行脚本 - 2023 年数据硬通关

【运行指令】
python run_v178.py --fix --year 2023

【强制执行】
- 按交易日循环，每天 1 次 API 调用
- 实时反馈打印
- 数据库事务强制提交
- 超时重试机制
- 数据校验
- 断点续传
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.alpha_research_v178 import V178DataFixer, MIN_ROWS_2023, YEAR_TARGET

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/v178_data_fix_{time:YYYYMMDD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


def check_data_count(year: int) -> tuple:
    """检查指定年份的数据行数"""
    from sqlalchemy import create_engine, text
    import os
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not configured!")
        return 0, True
    
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        query = text("""
            SELECT COUNT(*) FROM stock_daily
            WHERE trade_date BETWEEN :start_date AND :end_date
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"start_date": start_date, "end_date": end_date})
            count = result.scalar()
        
        count = int(count) if count else 0
        needs_fix = count < MIN_ROWS_2023 if year == 2023 else count == 0
        
        status = "✓ PASSED" if count >= MIN_ROWS_2023 else "✗ FAILED"
        
        logger.info("=" * 80)
        logger.info(f"[V178] Data Count Check - Year {year}")
        logger.info(f"  Total Rows: {count:,}")
        logger.info(f"  Target: >= {MIN_ROWS_2023:,}")
        logger.info(f"  Status: {status}")
        logger.info("=" * 80)
        
        return count, needs_fix
        
    except Exception as e:
        logger.error(f"[V178] Failed to check data count: {e}")
        return 0, True


def run_fix(year: int, force: bool = False):
    """运行数据修复"""
    logger.info("=" * 80)
    logger.info(f"[V178] Starting DATA FIX")
    logger.info(f"  Year: {year}")
    logger.info(f"  Force Mode: {force}")
    logger.info(f"  Target: > {MIN_ROWS_2023:,} rows")
    logger.info("=" * 80)
    
    # 检查是否已存在足够数据
    if not force:
        count, needs_fix = check_data_count(year)
        if not needs_fix:
            logger.info(f"[V178] Data already sufficient ({count:,} >= {MIN_ROWS_2023:,}), skipping fix")
            return count
    
    # 创建修复器
    fixer = V178DataFixer()
    
    if not fixer.ts_pro:
        logger.error("[V178] Tushare API not available! Please set TUSHARE_TOKEN in .env")
        return 0
    
    if not fixer.engine:
        logger.error("[V178] Database connection not available! Please set DATABASE_URL in .env")
        return 0
    
    # 执行修复
    try:
        success = fixer.fix_year(year)
        
        if success:
            logger.info("=" * 80)
            logger.info(f"[V178] DATA FIX SUCCESSFUL")
            logger.info(f"  Final count verification: PASSED")
            logger.info("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error(f"[V178] DATA FIX FAILED")
            logger.error("=" * 80)
            
    except RuntimeError as e:
        logger.error("=" * 80)
        logger.error(f"[V178] DATA FIX FAILED: {e}")
        logger.error("=" * 80)
        raise
    
    # 生成报告
    final_count = check_data_count(year)[0]
    
    report_path = Path("reports") / f"v178_data_fix_report_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': 'V178',
        'year': year,
        'final_row_count': final_count,
        'target_row_count': MIN_ROWS_2023,
        'passed': final_count >= MIN_ROWS_2023,
        'checkpoint': fixer.checkpoint,
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"[V178] Report saved to: {report_path}")
    
    return final_count


def main():
    parser = argparse.ArgumentParser(
        description="V178 Data Fix - 2023 Data Hard Pass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check data count for 2023
  python run_v178.py --check --year 2023
  
  # Fix 2023 data from Tushare
  python run_v178.py --fix --year 2023
  
  # Force re-fix even if data exists
  python run_v178.py --fix --year 2023 --force
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check data count for specified year'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Run data fix from Tushare'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=YEAR_TARGET,
        help='Year to process (default: 2023)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fix even if data exists'
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_data_count(args.year)
    
    elif args.fix:
        run_fix(args.year, force=args.force)
    
    else:
        # Default: check data count first, then fix if needed
        logger.info("[V178] No mode specified, running default workflow...")
        
        count, needs_fix = check_data_count(args.year)
        
        if needs_fix:
            logger.info(f"[V178] Data insufficient, starting fix...")
            run_fix(args.year)
        else:
            logger.info(f"[V178] Data sufficient, no fix needed.")


if __name__ == "__main__":
    main()