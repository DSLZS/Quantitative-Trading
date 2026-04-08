#!/usr/bin/env python3
"""
V178 暴力数据管道 - 2023 年数据硬通关入口脚本

【Mission】
停止所有关于 Alpha、NAG、Regime Switching 的口头分析。
现在！立刻！把 2023 年的数据给我凿进 MySQL！

【运行指令】
- 数据愈合：python run_v178.py --heal --year 2023
- 回测审计：python run_v178.py --audit --year 2023
- 跨周期验证：python run_v178.py --cross-cycle

【V178 核心改进】
1. 硬性拦截：main.py 入口校验，2023 年数据 < 500,000 行则强制 heal
2. 实时审计：每 1000 行打印进度 [DATA_PROGRESS]
3. 频控陷阱修复：HTTP 403 强制 sleep 60 秒
4. 小步快跑：按【交易日】拉取，每天 5000 只股票
5. 字段强制对齐：open, high, low, close, vol, amount, pct_chg, adj_factor 全部入库
6. 断点恢复：heal_checkpoint.json 记录断点

【反欺诈条款】
- 禁止修改初始资金：始终锁定 100,000
- 禁止美化结果：如果数据拉不下来，严禁使用 2024 年的平均值填充 2023 年！
- 禁止偷看未来：因子计算必须严格使用滚动窗口
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.alpha_research_v178 import (
    V178Runner,
    TushareHealerV178,
    SQL_HEALER_MIN_ROWS_2023,
    VERSION
)

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
    "logs/v178_{time:YYYYMMDD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


def check_data_count(year: int) -> tuple[int, bool]:
    """检查指定年份的数据行数"""
    healer = TushareHealerV178()
    if not healer.engine:
        logger.error("Database connection not available!")
        return 0, True
    
    count, needs_healing = healer.check_data_count(year)
    
    status = "✓ PASSED" if count >= SQL_HEALER_MIN_ROWS_2023 else "✗ FAILED"
    target_str = f">= {SQL_HEALER_MIN_ROWS_2023}" if year == 2023 else "> 0"
    
    logger.info("=" * 70)
    logger.info(f"[V178] Data Count Check - Year {year}")
    logger.info(f"  Total Rows: {count:,}")
    logger.info(f"  Target: {target_str}")
    logger.info(f"  Status: {status}")
    logger.info("=" * 70)
    
    return count, needs_healing


def run_heal(year: int, force: bool = False):
    """运行数据愈合"""
    logger.info("=" * 70)
    logger.info(f"[V178] Starting VIOLENT Data Healing")
    logger.info(f"  Year: {year}")
    logger.info(f"  Force Mode: {force}")
    logger.info(f"  Target (2023): > {SQL_HEALER_MIN_ROWS_2023:,} rows")
    logger.info("=" * 70)
    
    # 检查是否已存在足够数据
    if not force:
        count, needs_healing = check_data_count(year)
        if not needs_healing:
            logger.info(f"[V178] Data already sufficient, skipping heal")
            return count
    
    healer = TushareHealerV178()
    
    if not healer.ts_pro:
        logger.error("[V178] Tushare API not available! Please set TUSHARE_TOKEN in .env")
        return 0
    
    if not healer.engine:
        logger.error("[V178] Database connection not available! Please set DATABASE_URL in .env")
        return 0
    
    # 执行愈合
    df = healer.heal_year(year)
    
    # 验证结果
    final_count = len(df)
    logger.info("=" * 70)
    logger.info(f"[V178] Data Healing Complete")
    logger.info(f"  Final Row Count: {final_count:,}")
    logger.info(f"  Target (2023): {SQL_HEALER_MIN_ROWS_2023:,}")
    logger.info(f"  Status: {'✓ PASSED' if final_count >= SQL_HEALER_MIN_ROWS_2023 else '✗ FAILED'}")
    logger.info("=" * 70)
    
    # 生成愈合报告
    report_path = Path("reports") / f"v178_heal_report_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': VERSION,
        'year': year,
        'final_row_count': final_count,
        'target_row_count': SQL_HEALER_MIN_ROWS_2023,
        'passed': final_count >= SQL_HEALER_MIN_ROWS_2023,
        'healing_log': healer.get_healing_log(),
        'checkpoint': healer.checkpoint_data,
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"[V178] Healing report saved to: {report_path}")
    
    return final_count


def run_audit(year: int, parquet_path: Optional[str] = None):
    """运行回测审计"""
    logger.info("=" * 70)
    logger.info(f"[V178] Running Backtest Audit")
    logger.info(f"  Year: {year}")
    logger.info(f"  Parquet Path: {parquet_path}")
    logger.info("=" * 70)
    
    runner = V178Runner(parquet_path=parquet_path)
    result = runner.run_audit(year)
    
    # 生成审计报告
    report_path = Path("reports") / f"v178_audit_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"[V178] Audit report saved to: {report_path}")
    
    # 打印审计摘要
    logger.info("=" * 70)
    logger.info("[V178] Audit Summary")
    logger.info(f"  T+1 Rank IC: {result.get('t1_ic', {}).get('mean_ic', 'N/A'):.4f}")
    logger.info(f"  IC IR: {result.get('t1_ic', {}).get('ic_ir', 'N/A'):.2f}")
    logger.info(f"  Data Rows: {result.get('data_rows', 'N/A'):,}")
    logger.info(f"  Selected Factors: {result.get('selected_factors', [])}")
    logger.info(f"  Status: {'✓ PASSED' if result.get('passed', False) else '✗ FAILED'}")
    logger.info("=" * 70)
    
    return result


def run_cross_cycle(years: list[int] = None):
    """运行跨周期 OOS 验证"""
    if years is None:
        years = [2023, 2024]
    
    logger.info("=" * 70)
    logger.info(f"[V178] Running Cross-Cycle OOS Validation")
    logger.info(f"  Years: {years}")
    logger.info(f"  Target 2023: IC > 0.08, Data Rows > {SQL_HEALER_MIN_ROWS_2023:,}")
    logger.info(f"  Target 2024: IC > 0.08, IR > 0.40")
    logger.info("=" * 70)
    
    runner = V178Runner()
    result = runner.run_cross_cycle_audit(years)
    
    # 生成跨周期报告
    report_path = Path("reports") / f"v178_cross_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"[V178] Cross-cycle report saved to: {report_path}")
    
    # 打印验证结果
    validation = result.get('validation_passed', {})
    logger.info("=" * 70)
    logger.info("[V178] Cross-Cycle Validation Result")
    logger.info(f"  2023 Passed: {'✓ YES' if validation.get('2023', {}).get('passed', False) else '✗ NO'}")
    logger.info(f"  2024 Passed: {'✓ YES' if validation.get('2024', {}).get('passed', False) else '✗ NO'}")
    logger.info(f"  Overall Passed: {'✓ YES' if validation.get('overall_passed', False) else '✗ NO'}")
    logger.info("=" * 70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="V178 Violent Data Pipeline - 2023 Data Hard Pass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check data count for 2023
  python run_v178.py --check --year 2023
  
  # Heal 2023 data from Tushare
  python run_v178.py --heal --year 2023
  
  # Force re-heal even if data exists
  python run_v178.py --heal --year 2023 --force
  
  # Run backtest audit for 2023
  python run_v178.py --audit --year 2023
  
  # Run cross-cycle OOS validation (2023 + 2024)
  python run_v178.py --cross-cycle
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check data count for specified year'
    )
    parser.add_argument(
        '--heal',
        action='store_true',
        help='Run data healing from Tushare'
    )
    parser.add_argument(
        '--audit',
        action='store_true',
        help='Run backtest audit'
    )
    parser.add_argument(
        '--cross-cycle',
        action='store_true',
        help='Run cross-cycle OOS validation'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2023,
        help='Year to process (default: 2023)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-heal even if data exists'
    )
    parser.add_argument(
        '--parquet',
        type=str,
        default=None,
        help='Path to Parquet data file'
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_data_count(args.year)
    
    elif args.heal:
        run_heal(args.year, force=args.force)
    
    elif args.audit:
        run_audit(args.year, parquet_path=args.parquet)
    
    elif args.cross_cycle:
        run_cross_cycle()
    
    else:
        # Default: check data count first, then heal if needed
        logger.info("[V178] No mode specified, running default workflow...")
        
        count, needs_healing = check_data_count(args.year)
        
        if needs_healing:
            logger.info(f"[V178] Data insufficient, starting heal...")
            run_heal(args.year)
        else:
            logger.info(f"[V178] Data sufficient, running audit...")
            run_audit(args.year, parquet_path=args.parquet)


if __name__ == "__main__":
    main()