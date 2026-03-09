#!/usr/bin/env python3
"""
Stock Data Synchronization Script

This script synchronizes stock data from Tushare API to the local database.
It supports syncing CSI 300 index constituents with configurable date ranges.

Features:
    - Sync specific stocks or index constituents
    - Handle duplicate key conflicts (ON DUPLICATE KEY UPDATE)
    - Progress tracking and statistics
    - Rate limiting to avoid API restrictions

Usage:
    python run_sync.py [--index INDEX_CODE] [--start START_DATE] [--end END_DATE]
    
Examples:
    python run_sync.py  # Sync CSI 300 from 2024-01-01 to today
    python run_sync.py --index 000905.SH --start 20230101 --end 20231231
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from dotenv import load_dotenv
from loguru import logger

from db_manager import DatabaseManager
from data_loader import TushareLoader

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/sync_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
)


def check_table_schema(db: DatabaseManager, table_name: str) -> bool:
    """
    Check if the target table exists and has the correct schema.
    
    Args:
        db: Database manager instance
        table_name: Table name to check
        
    Returns:
        True if table exists with correct schema, False otherwise
    """
    try:
        # Check if table exists
        if not db.table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist. Creating...")
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `symbol` VARCHAR(20) NOT NULL COMMENT 'Stock code',
                `trade_date` DATE NOT NULL COMMENT 'Trading date',
                `open` DECIMAL(20, 4) COMMENT 'Opening price',
                `high` DECIMAL(20, 4) COMMENT 'Highest price',
                `low` DECIMAL(20, 4) COMMENT 'Lowest price',
                `close` DECIMAL(20, 4) COMMENT 'Closing price',
                `pre_close` DECIMAL(20, 4) COMMENT 'Previous close',
                `change` DECIMAL(20, 4) COMMENT 'Change',
                `pct_chg` DECIMAL(20, 4) COMMENT 'Change percentage',
                `volume` DECIMAL(30, 4) COMMENT 'Volume',
                `amount` DECIMAL(30, 4) COMMENT 'Amount',
                `adj_close` DECIMAL(20, 4) COMMENT 'Adjusted close',
                `adj_open` DECIMAL(20, 4) COMMENT 'Adjusted open',
                `adj_high` DECIMAL(20, 4) COMMENT 'Adjusted high',
                `adj_low` DECIMAL(20, 4) COMMENT 'Adjusted low',
                `adj_factor` DECIMAL(20, 4) COMMENT 'Adjustment factor',
                UNIQUE KEY `uk_symbol_date` (`symbol`, `trade_date`),
                INDEX `idx_symbol` (`symbol`),
                INDEX `idx_date` (`trade_date`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Daily stock data';
            """
            db.execute(create_table_sql)
            logger.info(f"Table {table_name} created successfully")
            return True
        
        logger.info(f"Table {table_name} exists")
        return True
        
    except Exception as e:
        logger.error(f"Failed to check/create table schema: {e}")
        return False


def handle_duplicates_with_upsert(
    db: DatabaseManager,
    df: pl.DataFrame,
    table_name: str,
    key_columns: list[str],
) -> int:
    """
    Handle duplicate keys using DELETE + INSERT approach.
    
    This is needed because ADBC driver doesn't support
    ON DUPLICATE KEY UPDATE directly.
    
    Args:
        db: Database manager instance
        df: DataFrame to insert
        table_name: Target table name
        key_columns: Columns forming the unique key
        
    Returns:
        Number of rows inserted
    """
    if df.is_empty():
        return 0
    
    # Get unique key combinations from the DataFrame
    keys_df = df.select(key_columns).unique()
    
    if keys_df.is_empty():
        return 0
    
    # Build DELETE condition using parameterized approach
    with db.get_connection() as conn:
        from sqlalchemy import text
        
        # Build WHERE clause for existing records
        conditions = []
        params = {}
        param_idx = 0
        
        for row in keys_df.iter_rows():
            condition_parts = []
            for col, val in zip(key_columns, row):
                param_name = f"param_{param_idx}"
                condition_parts.append(f"`{col}` = :{param_name}")
                params[param_name] = val
                param_idx += 1
            conditions.append(f"({' AND '.join(condition_parts)})")
        
        if conditions:
            delete_sql = f"""
                DELETE FROM `{table_name}`
                WHERE {' OR '.join(conditions)}
            """
            try:
                conn.execute(text(delete_sql), params)
                conn.commit()
                logger.debug(f"Deleted {len(keys_df)} existing records for upsert")
            except Exception as e:
                logger.warning(f"Delete failed, will attempt insert anyway: {e}")
    
    # Insert new records
    rows = db.to_sql(df, table_name, if_exists="append")
    return rows


def sync_with_conflict_handling(
    loader: TushareLoader,
    ts_code: str,
    start_date: str,
    end_date: str,
    table_name: str = "stock_daily",
) -> int:
    """
    Sync stock data with duplicate key handling.
    
    Args:
        loader: TushareLoader instance
        ts_code: Stock code
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        table_name: Target table name
        
    Returns:
        Number of rows synced
    """
    from db_manager import DatabaseManager
    
    db = DatabaseManager.get_instance()
    
    # Fetch data using loader's internal methods
    daily_df = loader._fetch_daily_data(ts_code, start_date, end_date)
    if daily_df is None or daily_df.is_empty():
        logger.warning(f"No data for {ts_code}")
        return 0
    
    adj_factor_df = loader._fetch_adj_factor(ts_code, start_date, end_date)
    if adj_factor_df is None or adj_factor_df.is_empty():
        adj_factor_df = daily_df.select(["ts_code", "trade_date"]).with_columns(
            pl.lit(1000).alias("adj_factor")
        )
    
    # Transform data
    transformed_df = loader._transform_data(daily_df, adj_factor_df)
    
    if transformed_df.is_empty():
        return 0
    
    # Use upsert with conflict handling
    # Note: Database column is 'trade_date', not 'Date'
    rows = handle_duplicates_with_upsert(
        db=db,
        df=transformed_df,
        table_name=table_name,
        key_columns=["symbol", "trade_date"],
    )
    
    return rows


def fetch_stock_list_basic(loader: TushareLoader) -> list[str]:
    """
    使用 stock_basic 接口获取所有上市股票（不需要高积分）。
    
    Args:
        loader: TushareLoader 实例
        
    Returns:
        list[str]: 股票代码列表
    """
    try:
        # 使用 tushare 直接调用 stock_basic 接口
        df = loader.pro.stock_basic(
            exchange='',  # 所有交易所
            list_status='L',  # 正常上市
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        if df is None or df.empty:
            logger.warning("No stocks found from stock_basic")
            return []
        
        # 过滤掉一些特殊股票（ST、*ST 等）
        pl_df = pl.from_pandas(df)
        stock_codes = pl_df.filter(
            ~pl.col("name").str.contains("ST|退|停牌", case_insensitive=True)
        )["ts_code"].to_list()
        
        logger.info(f"Found {len(stock_codes)} active stocks (excluding ST stocks)")
        return stock_codes
        
    except Exception as e:
        logger.error(f"Failed to fetch stock list: {e}")
        return []


def main() -> None:
    """Main entry point for the sync script."""
    parser = argparse.ArgumentParser(
        description="Sync stock data from Tushare to local database"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Index code (default: None, use --all-stocks for all stocks)",
    )
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="Sync all A-share stocks (uses stock_basic interface)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="20240101",
        help="Start date in YYYYMMDD format (default: 20240101)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYYMMDD format (default: today)",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="stock_daily",
        help="Target table name (default: stock_daily)",
    )
    parser.add_argument(
        "--single-stock",
        type=str,
        default=None,
        help="Sync a single stock instead of index constituents",
    )
    
    args = parser.parse_args()
    
    # Set end date to today if not specified
    if args.end is None:
        args.end = datetime.now().strftime("%Y%m%d")
    
    logger.info("=" * 60)
    logger.info("Stock Data Synchronization")
    logger.info("=" * 60)
    logger.info(f"Index: {args.index}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Target table: {args.table}")
    
    # Initialize components
    try:
        db = DatabaseManager.get_instance()
        loader = TushareLoader()
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    # Check/create table schema
    if not check_table_schema(db, args.table):
        logger.error("Failed to setup table schema")
        sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    # Sync data
    if args.single_stock:
        # Sync single stock
        logger.info(f"Syncing single stock: {args.single_stock}")
        rows = sync_with_conflict_handling(
            loader=loader,
            ts_code=args.single_stock,
            start_date=args.start,
            end_date=args.end,
            table_name=args.table,
        )
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("Sync Complete")
        logger.info("=" * 60)
        logger.info(f"Stocks synced: 1")
        logger.info(f"Total rows: {rows}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
    else:
        # Sync index constituents or all stocks
        stock_codes = []
        
        if args.all_stocks:
            # Use stock_basic interface to get all A-share stocks
            logger.info("Fetching all A-share stocks using stock_basic interface...")
            stock_codes = fetch_stock_list_basic(loader)
        elif args.index:
            # Try to fetch index constituents (requires high API access)
            logger.info(f"Fetching constituents of {args.index}...")
            stock_codes = loader._fetch_index_members(args.index)
        else:
            # Default: fetch all stocks if no index specified
            logger.info("No index specified, fetching all A-share stocks...")
            stock_codes = fetch_stock_list_basic(loader)
        
        if not stock_codes:
            logger.error("No stocks found to sync")
            sys.exit(1)
        
        total_stocks = len(stock_codes)
        successful_stocks = 0
        failed_stocks = []
        total_rows = 0
        
        for i, ts_code in enumerate(stock_codes):
            try:
                rows = sync_with_conflict_handling(
                    loader=loader,
                    ts_code=ts_code,
                    start_date=args.start,
                    end_date=args.end,
                    table_name=args.table,
                )
                
                if rows > 0:
                    successful_stocks += 1
                    total_rows += rows
                
                # Progress logging
                if (i + 1) % 10 == 0 or (i + 1) == total_stocks:
                    logger.info(f"Progress: {i + 1}/{total_stocks} stocks processed")
                
            except Exception as e:
                logger.error(f"Failed to sync {ts_code}: {e}")
                failed_stocks.append(ts_code)
        
        elapsed = time.time() - start_time
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Sync Complete")
        logger.info("=" * 60)
        logger.info(f"Total stocks in index: {total_stocks}")
        logger.info(f"Successfully synced: {successful_stocks}")
        logger.info(f"Failed: {len(failed_stocks)}")
        if failed_stocks:
            logger.warning(f"Failed stocks: {', '.join(failed_stocks[:10])}")
            if len(failed_stocks) > 10:
                logger.warning(f"... and {len(failed_stocks) - 10} more")
        logger.info(f"Total rows inserted/updated: {total_rows}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"Average per stock: {total_rows / max(successful_stocks, 1):.1f} rows")
        logger.info(f"Rate: {total_stocks / max(elapsed/60, 0.1):.1f} stocks/minute")


if __name__ == "__main__":
    main()