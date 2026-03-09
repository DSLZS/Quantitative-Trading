"""
Data Loader Module - Tushare data fetching and loading.

This module provides a TushareLoader class for fetching stock data
from Tushare API and loading it into the database using Polars.

核心功能:
    - 从 Tushare API 获取 A 股股票数据
    - 获取日线数据和复权因子
    - 频率限制控制，避免触发 API 限制
    - 将 Pandas DataFrame 转换为 Polars DataFrame
    - 字段映射和数据转换，适配数据库 schema
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import polars as pl
import tushare as ts
from dotenv import load_dotenv
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager

# Load environment variables
load_dotenv()


class TushareLoader:
    """
    Tushare 数据加载器，用于获取 A 股股票数据。
    
    功能特性:
        - 获取日线价格数据和复权因子
        - 频率限制控制，避免触发 API 限制
        - 将 Pandas DataFrame 转换为 Polars DataFrame
        - 字段映射，适配数据库 schema
    
    使用示例:
        >>> loader = TushareLoader()  # 初始化加载器
        >>> loader.sync_stock_data("000001.SZ", "20240101", "20241231")  # 同步单只股票
        >>> loader.sync_index_constituents("000300.SH")  # 同步指数成分股
    """
    
    # Tushare API 频率限制配置
    # 根据 Tushare 会员等级调整：
    # - 基础用户：100 次/分钟
    # - 高级用户：根据积分不同
    REQUESTS_PER_MINUTE = 60  # 每分钟请求数限制
    SLEEP_BETWEEN_REQUESTS = 0.5  # 请求间隔时间（秒）
    
    def __init__(self, token: Optional[str] = None) -> None:
        """
        使用 API token 初始化 Tushare 加载器。
        
        Args:
            token (str, optional): Tushare API token。
                如果为 None，则从 TUSHARE_TOKEN 环境变量读取。
                获取 token: https://tushare.pro/user/token
                
        Raises:
            ValueError: 当 TUSHARE_TOKEN 未配置时抛出
            
        使用示例:
            >>> # 从环境变量读取 token
            >>> loader = TushareLoader()
            >>> # 或者手动指定 token
            >>> loader = TushareLoader(token="your_token_here")
        """
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token or self.token == "your_tushare_token_here":
            logger.warning("TUSHARE_TOKEN not configured. Please set it in .env file.")
            raise ValueError("TUSHARE_TOKEN is required")
        
        # 初始化 Tushare API 客户端
        ts.set_token(self.token)
        self.pro = ts.pro_api()  # Tushare Pro API 客户端
        
        # 获取数据库管理器单例
        self.db = DatabaseManager.get_instance()
        
        # 频率限流计数器
        self.request_count = 0  # 当前请求计数
        self.last_request_time = time.time()  # 上次请求时间
        
        logger.info("TushareLoader initialized")
    
    def _rate_limit(self) -> None:
        """
        执行 API 请求的频率限制。
        
        Tushare 根据用户积分有不同的频率限制：
        - 基础用户：100 次/分钟
        - 积分用户：根据积分等级不同
        
        此方法确保不会超过限制，通过在请求间添加延迟实现。
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # 确保请求间有最小时间间隔
        if time_since_last < self.SLEEP_BETWEEN_REQUESTS:
            sleep_time = self.SLEEP_BETWEEN_REQUESTS - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # 每 10 次请求记录一次日志
        if self.request_count % 10 == 0:
            logger.debug(f"Tushare API requests: {self.request_count}")
    
    def _fetch_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        从 Tushare 获取日线价格数据。
        
        调用 Tushare 的 daily 接口获取股票日线数据，
        包含开盘价、最高价、最低价、收盘价、成交量等。
        
        Args:
            ts_code (str): Tushare 股票代码，格式为 "代码。交易所"
                示例："000001.SZ" (平安银行), "600000.SH" (浦发银行)
            start_date (str): 开始日期，格式 YYYYMMDD
                示例："20240101"
            end_date (str): 结束日期，格式 YYYYMMDD
                示例："20241231"
            
        Returns:
            Optional[pl.DataFrame]: 包含日线数据的 Polars DataFrame，
                如果获取失败返回 None。
                列包括：ts_code, trade_date, open, high, low, close,
                      pre_close, change, pct_chg, vol, amount
            
        Tushare daily 接口返回字段说明:
            - ts_code: 股票代码
            - trade_date: 交易日期
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - pre_close: 昨收价
            - change: 涨跌额
            - pct_chg: 涨跌幅 (%)
            - vol: 成交量 (手)
            - amount: 成交额 (千元)
        """
        try:
            self._rate_limit()  # 执行频率限制
            
            # 调用 Tushare API 获取日线数据
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            
            if df is None or df.empty:
                logger.warning(f"No daily data for {ts_code}")
                return None
            
            # 将 Pandas DataFrame 转换为 Polars DataFrame
            pl_df = pl.from_pandas(df)
            
            logger.debug(f"Fetched {len(pl_df)} rows of daily data for {ts_code}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch daily data for {ts_code}: {e}")
            return None
    
    def _fetch_adj_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        从 Tushare 获取复权因子数据。
        
        复权因子用于计算复权价格，消除分红、配股等事件对价格的影响。
        Tushare 的复权因子是累积的，可以直接用于计算后复权价格。
        
        Args:
            ts_code (str): Tushare 股票代码
            start_date (str): 开始日期，格式 YYYYMMDD
            end_date (str): 结束日期，格式 YYYYMMDD
            
        Returns:
            Optional[pl.DataFrame]: 包含复权因子的 Polars DataFrame，
                如果获取失败返回 None。
                列包括：ts_code, trade_date, adj_factor
            
        Tushare adj_factor 接口返回字段说明:
            - ts_code: 股票代码
            - trade_date: 交易日期
            - adj_factor: 复权因子 (累积值)
            
        复权价格计算公式:
            后复权收盘价 = 收盘价 × 复权因子 / 基准值 (通常为 1000)
        """
        try:
            self._rate_limit()  # 执行频率限制
            
            # 调用 Tushare API 获取复权因子
            df = self.pro.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            
            if df is None or df.empty:
                logger.warning(f"No adj_factor data for {ts_code}")
                return None
            
            # 将 Pandas DataFrame 转换为 Polars DataFrame
            pl_df = pl.from_pandas(df)
            
            logger.debug(f"Fetched {len(pl_df)} rows of adj_factor data for {ts_code}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch adj_factor data for {ts_code}: {e}")
            return None
    
    def _fetch_index_members(self, index_code: str = "000300.SH") -> list[str]:
        """
        获取指数成分股列表。
        
        Args:
            index_code (str): 指数代码，默认 "000300.SH" (沪深 300)
                常用指数:
                - 000300.SH: 沪深 300
                - 000001.SH: 上证指数
                - 000016.SH: 上证 50
                - 000905.SH: 中证 500
            
        Returns:
            list[str]: 成分股股票代码列表
                示例：["000001.SZ", "000002.SZ", ...]
            
        Tushare index_member 接口返回字段说明:
            - index_code: 指数代码
            - con_code: 成分股代码
            - in_date: 纳入日期
            - out_date: 剔除日期
        """
        try:
            self._rate_limit()  # 执行频率限制
            
            # 调用 Tushare API 获取指数成分股
            df = self.pro.index_member(ts_code=index_code)
            
            if df is None or df.empty:
                logger.warning(f"No constituents found for {index_code}")
                return []
            
            # 将 Pandas DataFrame 转换为 Polars 并获取成分股代码列表
            pl_df = pl.from_pandas(df)
            stock_codes = pl_df["con_code"].to_list()
            
            logger.info(f"Found {len(stock_codes)} constituents in {index_code}")
            return stock_codes
            
        except Exception as e:
            logger.error(f"Failed to fetch index constituents: {e}")
            return []
    
    def _transform_data(
        self,
        daily_df: pl.DataFrame,
        adj_factor_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        转换并合并日线数据与复权因子。
        
        此方法执行以下操作:
        1. 重命名字段：ts_code -> symbol (适配数据库 schema)
        2. 日期转换：trade_date (YYYYMMDD 字符串) -> Date (ISO 8601 YYYY-MM-DD)
        3. 左连接合并：日线数据 LEFT JOIN 复权因子
        4. 计算复权价格：使用复权因子计算复权开高低收
        5. 选择并重命名列：适配数据库字段名
        
        Args:
            daily_df (pl.DataFrame): 日线价格 DataFrame
            adj_factor_df (pl.DataFrame): 复权因子 DataFrame
            
        Returns:
            pl.DataFrame: 转换后的 DataFrame，包含以下列:
                - symbol: 股票代码
                - Date: 交易日期 (Date 类型，ISO 8601 格式 YYYY-MM-DD)
                - open, high, low, close: 原始价格
                - pre_close, change, pct_chg: 价格变化
                - volume, amount: 成交量和成交额
                - adj_open, adj_high, adj_low, adj_close: 复权价格
                - adj_factor: 复权因子
        
        注意:
            - 日期格式必须为 ISO 8601 (YYYY-MM-DD) 以支持 MySQL 分区表
            - 转换后日期类型可直接用于 PARTITION BY RANGE 逻辑
        """
        # 重命名 ts_code 为 symbol (数据库字段名)
        daily_df = daily_df.with_columns(
            pl.col("ts_code").alias("symbol")
        )
        
        adj_factor_df = adj_factor_df.with_columns(
            pl.col("ts_code").alias("symbol")
        )
        
        # 将 trade_date 从 YYYYMMDD 字符串转换为 Date 类型 (ISO 8601 YYYY-MM-DD)
        # 使用 str.strptime 确保正确的日期解析和存储
        daily_df = daily_df.with_columns(
            pl.col("trade_date")
            .str.strptime(pl.Date, "%Y%m%d")
            .alias("Date")
        )
        
        adj_factor_df = adj_factor_df.with_columns(
            pl.col("trade_date")
            .str.strptime(pl.Date, "%Y%m%d")
            .alias("Date")
        )
        
        # 合并日线数据与复权因子 (左连接)
        merged_df = daily_df.join(
            adj_factor_df.select(["symbol", "Date", "adj_factor"]),
            on=["symbol", "Date"],
            how="left",
        )
        
        # 计算复权价格
        # adj_factor 是累积值，基准值通常为 1000
        if "adj_factor" in merged_df.columns:
            merged_df = merged_df.with_columns(
                (pl.col("close") * pl.col("adj_factor") / 1000).alias("adj_close"),
                (pl.col("open") * pl.col("adj_factor") / 1000).alias("adj_open"),
                (pl.col("high") * pl.col("adj_factor") / 1000).alias("adj_high"),
                (pl.col("low") * pl.col("adj_factor") / 1000).alias("adj_low"),
                (pl.col("pre_close") * pl.col("adj_factor") / 1000).alias("adj_pre_close"),
            )
        else:
            # 如果没有复权因子，使用原始价格
            merged_df = merged_df.with_columns(
                pl.col("close").alias("adj_close"),
                pl.col("open").alias("adj_open"),
                pl.col("high").alias("adj_high"),
                pl.col("low").alias("adj_low"),
                pl.col("pre_close").alias("adj_pre_close"),
            )
        
        # 选择并重命名列以适配数据库 schema
        # 注意：数据库字段名为 trade_date (DATE 类型)，不是 Date
        # 现有表结构：symbol, trade_date, open, high, low, close, volume, amount, adj_factor, turnover_rate
        result_df = merged_df.select([
            pl.col("symbol"),
            pl.col("Date").alias("trade_date"),  # 重命名为 trade_date
            pl.col("open").alias("open"),
            pl.col("high").alias("high"),
            pl.col("low").alias("low"),
            pl.col("close").alias("close"),
            pl.col("vol").alias("volume"),
            pl.col("amount").alias("amount"),
            pl.col("adj_factor").fill_null(1000).alias("adj_factor"),
            # turnover_rate 需要额外计算，这里用 null 填充
            pl.lit(None).cast(pl.Float64).alias("turnover_rate"),
        ])
        
        # 按 symbol 和 trade_date 排序
        result_df = result_df.sort(["symbol", "trade_date"])
        
        return result_df
    
    def sync_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        table_name: str = "stock_daily",
    ) -> int:
        """
        同步单只股票数据到数据库。
        
        此方法执行完整的 ETL 流程:
        1. 获取日线数据
        2. 获取复权因子
        3. 转换和合并数据
        4. 写入数据库
        
        Args:
            ts_code (str): Tushare 股票代码
                示例："000001.SZ"
            start_date (str): 开始日期，格式 YYYYMMDD
                示例："20240101"
            end_date (str): 结束日期，格式 YYYYMMDD
                示例："20241231"
            table_name (str): 目标数据库表名，默认 "stock_daily"
            
        Returns:
            int: 同步的行数
            
        使用示例:
            >>> loader = TushareLoader()
            >>> rows = loader.sync_stock_data(
            ...     ts_code="000001.SZ",
            ...     start_date="20240101",
            ...     end_date="20241231"
            ... )
            >>> print(f"Synced {rows} rows")
        """
        logger.info(f"Syncing data for {ts_code} from {start_date} to {end_date}")
        
        # 获取日线数据
        daily_df = self._fetch_daily_data(ts_code, start_date, end_date)
        if daily_df is None or daily_df.is_empty():
            logger.warning(f"No data to sync for {ts_code}")
            return 0
        
        # 获取复权因子
        adj_factor_df = self._fetch_adj_factor(ts_code, start_date, end_date)
        if adj_factor_df is None or adj_factor_df.is_empty():
            logger.warning(f"No adj_factor data for {ts_code}, using daily data only")
            # 如果没有复权因子，使用默认值 1000
            adj_factor_df = daily_df.select(["ts_code", "trade_date"]).with_columns(
                pl.lit(1000).alias("adj_factor")
            )
        
        # 转换和合并数据
        transformed_df = self._transform_data(daily_df, adj_factor_df)
        
        if transformed_df.is_empty():
            logger.warning(f"Transformed data is empty for {ts_code}")
            return 0
        
        # 写入数据库
        rows = self.db.to_sql(
            df=transformed_df,
            table_name=table_name,
            if_exists="append",
        )
        
        logger.info(f"Synced {rows} rows for {ts_code}")
        return rows
    
    def sync_index_constituents(
        self,
        index_code: str = "000300.SH",
        start_date: str = "20240101",
        end_date: str = None,
        table_name: str = "stock_daily",
    ) -> dict[str, Any]:
        """
        同步指数所有成分股数据到数据库。
        
        此方法会:
        1. 获取指数成分股列表
        2. 遍历每只股票，调用 sync_stock_data 同步数据
        3. 返回同步统计信息
        
        Args:
            index_code (str): 指数代码，默认 "000300.SH" (沪深 300)
            start_date (str): 开始日期，格式 YYYYMMDD，默认 "20240101"
            end_date (str, optional): 结束日期，格式 YYYYMMDD，默认今天
            table_name (str): 目标数据库表名，默认 "stock_daily"
            
        Returns:
            dict[str, Any]: 同步统计信息字典，包含:
                - success (bool): 是否成功
                - total_stocks (int): 总股票数
                - successful_stocks (int): 成功同步的股票数
                - failed_stocks (list[str]): 失败的股票代码列表
                - total_rows (int): 总行数
            
        使用示例:
            >>> loader = TushareLoader()
            >>> stats = loader.sync_index_constituents(
            ...     index_code="000300.SH",
            ...     start_date="20240101"
            ... )
            >>> print(f"成功：{stats['successful_stocks']}/{stats['total_stocks']}")
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        # 获取指数成分股
        stock_codes = self._fetch_index_members(index_code)
        
        if not stock_codes:
            logger.error(f"No constituents found for {index_code}")
            return {
                "success": False,
                "error": "No constituents found",
            }
        
        total_rows = 0
        successful_stocks = 0
        failed_stocks = []
        
        logger.info(f"Starting to sync {len(stock_codes)} stocks")
        
        # 遍历成分股，逐个同步数据
        for i, ts_code in enumerate(stock_codes):
            try:
                rows = self.sync_stock_data(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    table_name=table_name,
                )
                
                if rows > 0:
                    successful_stocks += 1
                    total_rows += rows
                
                # 每 10 只股票记录一次进度
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(stock_codes)} stocks processed")
                
            except Exception as e:
                logger.error(f"Failed to sync {ts_code}: {e}")
                failed_stocks.append(ts_code)
        
        stats = {
            "success": True,
            "total_stocks": len(stock_codes),
            "successful_stocks": successful_stocks,
            "failed_stocks": failed_stocks,
            "total_rows": total_rows,
        }
        
        logger.info(
            f"Sync complete: {successful_stocks}/{len(stock_codes)} stocks, "
            f"{total_rows} total rows"
        )
        
        return stats


# 便捷函数
def get_loader(token: Optional[str] = None) -> TushareLoader:
    """
    获取 TushareLoader 实例。
    
    Args:
        token (str, optional): Tushare API token
        
    Returns:
        TushareLoader: Tushare 数据加载器实例
    """
    return TushareLoader(token)