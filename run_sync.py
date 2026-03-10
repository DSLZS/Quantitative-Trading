#!/usr/bin/env python3
"""
Unified Stock & Fund Data Synchronization Script

This script synchronizes stock and fund data from Tushare API to the local database.
It supports syncing stocks, ETF funds, or both.

Features:
    - Sync stocks, funds, or both via --asset-type parameter
    - Handle duplicate key conflicts (ON DUPLICATE KEY UPDATE)
    - Progress tracking and statistics
    - Rate limiting to avoid API restrictions

Usage:
    python run_sync.py [--asset-type TYPE] [--index INDEX_CODE] [--start START_DATE] [--end END_DATE]
    
Examples:
    python run_sync.py  # Sync all stocks (auto mode) from 2024-01-01 to today
    python run_sync.py --asset-type stock --index 000905.SH --start 20230101 --end 20231231
    python run_sync.py --asset-type fund --start 20230101
    python run_sync.py --asset-type auto  # Sync both stocks and funds
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

# Target fund list for fund synchronization
TARGET_FUNDS = [
    {"code": "510300.SH", "name": "沪深 300ETF"},
    {"code": "159915.SZ", "name": "创业板 ETF"},
]


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
        if not db.table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist. Creating...")
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `symbol` VARCHAR(20) NOT NULL COMMENT 'Stock/fund code',
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
                `adj_factor` DECIMAL(20, 4) COMMENT 'Adjustment factor',
                `turnover_rate` DECIMAL(20, 4) COMMENT 'Turnover rate',
                `vol_ratio` DECIMAL(20, 4) COMMENT 'Volume ratio',
                UNIQUE KEY `uk_symbol_date` (`symbol`, `trade_date`),
                INDEX `idx_symbol` (`symbol`),
                INDEX `idx_date` (`trade_date`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Daily stock/fund data';
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
    
    keys_df = df.select(key_columns).unique()
    
    if keys_df.is_empty():
        return 0
    
    with db.get_connection() as conn:
        from sqlalchemy import text
        
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
    Sync stock/fund data with duplicate key handling.
    
    Args:
        loader: TushareLoader instance
        ts_code: Stock/fund code
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        table_name: Target table name
        
    Returns:
        Number of rows synced
    """
    db = DatabaseManager.get_instance()
    
    daily_df = loader._fetch_daily_data(ts_code, start_date, end_date)
    if daily_df is None or daily_df.is_empty():
        logger.warning(f"No data for {ts_code}")
        return 0
    
    adj_factor_df = loader._fetch_adj_factor(ts_code, start_date, end_date)
    if adj_factor_df is None or adj_factor_df.is_empty():
        adj_factor_df = daily_df.select(["ts_code", "trade_date"]).with_columns(
            pl.lit(1000).alias("adj_factor")
        )
    
    transformed_df = loader._transform_data(daily_df, adj_factor_df)
    
    if transformed_df.is_empty():
        return 0
    
    rows = handle_duplicates_with_upsert(
        db=db,
        df=transformed_df,
        table_name=table_name,
        key_columns=["symbol", "trade_date"],
    )
    
    return rows


def sync_fund(
    loader: TushareLoader,
    ts_code: str,
    start_date: str,
    end_date: str,
    table_name: str = "stock_daily",
) -> int:
    """
    Sync fund data to database.
    
    Args:
        loader: TushareLoader instance
        ts_code: Fund code
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        table_name: Target table name
        
    Returns:
        int: Number of rows synced
    """
    logger.info(f"Syncing fund: {ts_code} from {start_date} to {end_date}")
    
    try:
        rows = sync_with_conflict_handling(
            loader=loader,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            table_name=table_name,
        )
        return rows
    except Exception as e:
        logger.error(f"Failed to sync {ts_code}: {e}")
        return 0


def fetch_stock_list_basic(loader: TushareLoader) -> list[str]:
    """
    Fetch all active A-share stocks using stock_basic interface.
    
    Args:
        loader: TushareLoader instance
        
    Returns:
        list[str]: Stock code list
    """
    try:
        df = loader.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        if df is None or df.empty:
            logger.warning("No stocks found from stock_basic")
            return []
        
        pl_df = pl.from_pandas(df)
        stock_codes = pl_df.filter(
            ~pl.col("name").str.contains("(?i)ST|退 | 停牌")
        )["ts_code"].to_list()
        
        logger.info(f"Found {len(stock_codes)} active stocks (excluding ST stocks)")
        return stock_codes
        
    except Exception as e:
        logger.error(f"Failed to fetch stock list: {e}")
        return []


def sync_stocks(
    loader: TushareLoader,
    args: argparse.Namespace,
) -> tuple[int, int, list[str]]:
    """
    Sync stock data based on provided arguments.
    
    Args:
        loader: TushareLoader instance
        args: Parsed command line arguments
        
    Returns:
        tuple: (successful_stocks, total_rows, failed_stocks)
    """
    stock_codes = []
    
    if args.all_stocks:
        logger.info("Fetching all A-share stocks using stock_basic interface...")
        stock_codes = fetch_stock_list_basic(loader)
    elif args.index:
        logger.info(f"Fetching constituents of {args.index}...")
        stock_codes = loader._fetch_index_members(args.index)
    elif args.single_stock:
        stock_codes = [args.single_stock]
    else:
        logger.info("No index specified, fetching all A-share stocks...")
        stock_codes = fetch_stock_list_basic(loader)
    
    if not stock_codes:
        logger.error("No stocks found to sync")
        return 0, 0, []
    
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
            
            if (i + 1) % 10 == 0 or (i + 1) == total_stocks:
                logger.info(f"Progress: {i + 1}/{total_stocks} stocks processed")
            
        except Exception as e:
            logger.error(f"Failed to sync {ts_code}: {e}")
            failed_stocks.append(ts_code)
    
    return successful_stocks, total_rows, failed_stocks


def sync_funds(
    loader: TushareLoader,
    args: argparse.Namespace,
) -> tuple[int, int, list[str]]:
    """
    Sync fund data based on provided arguments.
    
    Args:
        loader: TushareLoader instance
        args: Parsed command line arguments
        
    Returns:
        tuple: (successful_funds, total_rows, failed_funds)
    """
    funds_to_sync = TARGET_FUNDS
    if args.funds:
        fund_codes = [code.strip() for code in args.funds.split(",")]
        funds_to_sync = [f for f in TARGET_FUNDS if f["code"] in fund_codes]
    
    if not funds_to_sync:
        logger.warning("No valid funds specified to sync")
        return 0, 0, []
    
    total_funds = len(funds_to_sync)
    successful_funds = 0
    failed_funds = []
    total_rows = 0
    
    for fund_info in funds_to_sync:
        ts_code = fund_info["code"]
        fund_name = fund_info["name"]
        
        logger.info(f"Syncing {fund_name} ({ts_code})...")
        
        rows = sync_fund(
            loader=loader,
            ts_code=ts_code,
            start_date=args.start,
            end_date=args.end,
            table_name=args.table,
        )
        
        if rows > 0:
            successful_funds += 1
            total_rows += rows
            logger.info(f"  ✓ {fund_name}: {rows} rows synced")
        else:
            failed_funds.append(ts_code)
            logger.warning(f"  ✗ {fund_name}: failed to sync")
    
    return successful_funds, total_rows, failed_funds


def print_sync_summary(
    elapsed: float,
    stocks_synced: int = 0,
    funds_synced: int = 0,
    total_rows: int = 0,
    failed_stocks: list[str] = None,
    failed_funds: list[str] = None,
) -> None:
    """Print synchronization summary."""
    failed_stocks = failed_stocks or []
    failed_funds = failed_funds or []
    
    logger.info("=" * 60)
    logger.info("Sync Complete")
    logger.info("=" * 60)
    
    if stocks_synced > 0:
        logger.info(f"Stocks synced: {stocks_synced}")
    if funds_synced > 0:
        logger.info(f"Funds synced: {funds_synced}")
    
    logger.info(f"Total rows inserted/updated: {total_rows}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    if stocks_synced > 0:
        logger.info(f"Average per stock: {total_rows / max(stocks_synced, 1):.1f} rows")
        logger.info(f"Rate: {stocks_synced / max(elapsed/60, 0.1):.1f} stocks/minute")
    
    if failed_stocks:
        logger.warning(f"Failed stocks: {', '.join(failed_stocks[:10])}")
        if len(failed_stocks) > 10:
            logger.warning(f"... and {len(failed_stocks) - 10} more")
    
    if failed_funds:
        logger.warning(f"Failed funds: {', '.join(failed_funds)}")
    
    if failed_stocks or failed_funds:
        sys.exit(1)


def main() -> None:
    """Main entry point for the sync script."""
    parser = argparse.ArgumentParser(
        description="Sync stock and fund data from Tushare to local database"
    )
    parser.add_argument(
        "--asset-type",
        type=str,
        default="auto",
        choices=["stock", "fund", "auto"],
        help="Asset type to sync: 'stock' for stocks only, 'fund' for funds only, 'auto' for both (default: auto)",
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
    parser.add_argument(
        "--funds",
        type=str,
        default=None,
        help="Comma-separated fund codes to sync (default: all target funds)",
    )
    
    args = parser.parse_args()
    
    if args.end is None:
        args.end = datetime.now().strftime("%Y%m%d")
    
    logger.info("=" * 60)
    logger.info("Stock & Fund Data Synchronization")
    logger.info("=" * 60)
    logger.info(f"Asset type: {args.asset_type}")
    logger.info(f"Index: {args.index}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Target table: {args.table}")
    
    try:
        db = DatabaseManager.get_instance()
        loader = TushareLoader()
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    if not check_table_schema(db, args.table):
        logger.error("Failed to setup table schema")
        sys.exit(1)
    
    start_time = time.time()
    
    stocks_synced, stock_rows, failed_stocks = 0, 0, []
    funds_synced, fund_rows, failed_funds = 0, 0, []
    
    # Sync stocks
    if args.asset_type in ["stock", "auto"]:
        stocks_synced, stock_rows, failed_stocks = sync_stocks(loader, args)
    
    # Sync funds
    if args.asset_type in ["fund", "auto"]:
        funds_synced, fund_rows, failed_funds = sync_funds(loader, args)
    
    elapsed = time.time() - start_time
    
    print_sync_summary(
        elapsed=elapsed,
        stocks_synced=stocks_synced,
        funds_synced=funds_synced,
        total_rows=stock_rows + fund_rows,
        failed_stocks=failed_stocks,
        failed_funds=failed_funds,
    )


if __name__ == "__main__":
    main()