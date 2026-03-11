#!/usr/bin/env python3
"""
Index Constituents Syncer - Sync index constituent stocks using index_weight interface.

This script fetches index constituent stocks from Tushare's index_weight interface
and syncs their historical daily data to the database.

Usage:
    python sync_index_constituents.py --index 000905.SH --start 20230101
    
Examples:
    # Sync CSI 500 (中证 500) constituents
    python sync_index_constituents.py --index 000905.SH --start 20230101
    
    # Sync CSI 300 (沪深 300) constituents
    python sync_index_constituents.py --index 000300.SH --start 20230101
    
    # Sync with resume mode (skip already synced stocks)
    python sync_index_constituents.py --index 000905.SH --start 20230101 --resume
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


class IndexConstituentSyncer:
    """
    指数成分股同步器，使用 index_weight 接口获取成分股。
    
    功能特性:
        - 通过 index_weight 接口获取指数成分股权重数据
        - 自动提取所有唯一股票代码
        - 支持增量同步和断点续传
        - 同步成分股历史日线数据到数据库
    """
    
    def __init__(
        self,
        index_code: str,
        start_date: str,
        end_date: str | None = None,
        resume: bool = False,
    ) -> None:
        """
        初始化指数成分股同步器。
        
        Args:
            index_code (str): 指数代码，如 000905.SH (中证 500)
            start_date (str): 开始日期，格式 YYYYMMDD
            end_date (str, optional): 结束日期，格式 YYYYMMDD，默认为今天
            resume (bool): 是否启用断点续传模式，默认 False
        """
        self.index_code = index_code
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")
        self.resume = resume
        
        self.db = DatabaseManager.get_instance()
        self.loader = TushareLoader()
        
        logger.info(f"IndexConstituentSyncer initialized for {index_code}")
        logger.info(f"Date range: {start_date} to {self.end_date}")
        logger.info(f"Resume mode: {resume}")
    
    def _fetch_index_constituents(self) -> list[str]:
        """
        通过 index_weight 接口获取指数成分股列表。
        
        使用 Tushare 的 index_weight 接口获取指定指数在指定日期范围内的
        所有成分股及其权重数据，返回唯一的股票代码列表。
        
        Returns:
            list[str]: 股票代码列表
        """
        logger.info(f"Fetching constituents for {self.index_code}...")
        
        try:
            # 使用 index_weight 接口获取成分股权重数据
            # 注意：接口返回字段为 con_code (成分股代码)
            df = self.loader.pro.index_weight(
                index_code=self.index_code,
                start=self.start_date,
                end=self.end_date,
            )
            
            if df is None or df.empty:
                logger.warning(f"No constituents found for {self.index_code}")
                return []
            
            # 转换为 Polars DataFrame
            pl_df = pl.from_pandas(df)
            
            # 获取唯一的股票代码列表 (接口返回字段为 con_code)
            unique_codes = pl_df["con_code"].unique().to_list()
            
            logger.info(f"Found {len(unique_codes)} unique constituents in {self.index_code}")
            
            # 打印前 10 只股票作为示例
            logger.info(f"Sample constituents (first 10): {unique_codes[:10]}")
            
            return unique_codes
            
        except Exception as e:
            logger.error(f"Failed to fetch index constituents: {e}")
            return []
    
    def _sync_single_stock(
        self,
        ts_code: str,
        table_name: str = "stock_daily",
    ) -> int:
        """
        同步单只股票的历史日线数据。
        
        Args:
            ts_code (str): 股票代码
            table_name (str): 目标表名
            
        Returns:
            int: 同步的行数
        """
        try:
            # 检查是否已同步（断点续传模式）
            if self.resume:
                query = f"""
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE symbol = '{ts_code}' 
                    AND trade_date >= '{self.start_date}'
                """
                result = self.db.read_sql(query)
                if not result.is_empty() and result[0, 0] > 0:
                    logger.debug(f"Skipping {ts_code} - already synced")
                    return 0
            
            # 同步股票数据
            rows = self.loader.sync_stock_data(
                ts_code=ts_code,
                start_date=self.start_date,
                end_date=self.end_date,
                table_name=table_name,
            )
            
            return rows
            
        except Exception as e:
            logger.error(f"Failed to sync {ts_code}: {e}")
            return 0
    
    def sync(
        self,
        table_name: str = "stock_daily",
        delay: float = 0.2,
    ) -> dict:
        """
        执行成分股数据同步。
        
        Args:
            table_name (str): 目标表名
            delay (float): 每只股票之间的延迟时间（秒），默认 0.2 秒
            
        Returns:
            dict: 同步统计信息
        """
        logger.info("=" * 60)
        logger.info(f"Syncing Index Constituents: {self.index_code}")
        logger.info("=" * 60)
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Resume mode: {self.resume}")
        
        # Step 1: 获取成分股列表
        constituents = self._fetch_index_constituents()
        
        if not constituents:
            logger.warning("No constituents to sync")
            return {"total": 0, "synced": 0, "failed": 0, "rows": 0}
        
        total_constituents = len(constituents)
        logger.info(f"Total constituents: {total_constituents}")
        logger.info(f"Stocks to sync: {total_constituents}")
        
        # Step 2: 同步每只股票的数据
        synced = 0
        failed = 0
        total_rows = 0
        failed_stocks = []
        
        for i, ts_code in enumerate(constituents):
            rows = self._sync_single_stock(ts_code, table_name)
            
            if rows > 0:
                synced += 1
                total_rows += rows
                logger.info(f"Synced {ts_code}: {rows} rows")
            elif rows == 0 and not self.resume:
                # 非 resume 模式下，0 行表示失败
                failed += 1
                failed_stocks.append(ts_code)
                logger.warning(f"Failed to sync {ts_code}")
            else:
                # resume 模式下，0 行可能表示已存在
                logger.debug(f"Skipped {ts_code} (already synced)")
            
            # 显示进度
            if (i + 1) % 50 == 0 or (i + 1) == total_constituents:
                logger.info(f"Progress: {i + 1}/{total_constituents} stocks processed")
                logger.info(f"  Synced: {synced}, Failed: {failed}, Total rows: {total_rows}")
            
            # 延迟，避免 API 限流
            if delay > 0 and i < len(constituents) - 1:
                time.sleep(delay)
        
        # 统计结果
        stats = {
            "total": total_constituents,
            "synced": synced,
            "failed": failed,
            "rows": total_rows,
            "failed_stocks": failed_stocks,
        }
        
        logger.info("=" * 60)
        logger.info("Sync Complete")
        logger.info("=" * 60)
        logger.info(f"Total constituents: {total_constituents}")
        logger.info(f"Successfully synced: {synced}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total rows inserted/updated: {total_rows}")
        
        if failed_stocks:
            logger.warning(f"Failed stocks: {', '.join(failed_stocks[:20])}")
            if len(failed_stocks) > 20:
                logger.warning(f"... and {len(failed_stocks) - 20} more")
        
        return stats


def main() -> None:
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="Sync index constituent stocks using index_weight interface"
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Index code (e.g., 000905.SH for CSI 500, 000300.SH for CSI 300)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYYMMDD format",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYYMMDD format (default: today)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip already synced stocks",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="stock_daily",
        help="Target table name (default: stock_daily)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between stocks in seconds (default: 0.2)",
    )
    
    args = parser.parse_args()
    
    # 创建同步器并执行同步
    syncer = IndexConstituentSyncer(
        index_code=args.index,
        start_date=args.start,
        end_date=args.end,
        resume=args.resume,
    )
    
    stats = syncer.sync(
        table_name=args.table,
        delay=args.delay,
    )
    
    # 验证数据库中的股票数量
    logger.info("=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)
    
    try:
        # 数据库表使用 symbol 字段存储股票代码
        query = f"SELECT COUNT(DISTINCT symbol) as unique_stocks FROM {args.table}"
        result = syncer.db.read_sql(query)
        unique_stocks = result[0, 0] if not result.is_empty() else 0
        logger.info(f"Unique stocks (symbol) in database: {unique_stocks}")
        
        query = f"SELECT COUNT(*) as total_rows FROM {args.table}"
        result = syncer.db.read_sql(query)
        total_rows = result[0, 0] if not result.is_empty() else 0
        logger.info(f"Total rows in database: {total_rows}")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")


if __name__ == "__main__":
    main()