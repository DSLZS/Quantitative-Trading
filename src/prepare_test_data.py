"""
Prepare Test Data Script - 为 daily_trade_advisor.py 测试模式准备数据

功能:
    1. 从 AKShare 获取贵州茅台 (600519.SH) 和宁德时代 (300750.SZ) 最近 30 天的日线数据
    2. 存入 stock_daily 表
    3. 确保测试模式能够正常运行

使用示例:
    >>> python src/prepare_test_data.py
    # 同步贵州茅台和宁德时代最近 30 天的数据到 stock_daily 表
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# 测试股票列表
TEST_STOCKS = {
    "600519.SH": "贵州茅台",
    "300750.SZ": "宁德时代",
}


def _convert_symbol_to_ak_code(symbol: str) -> str:
    """
    将标准 symbol 格式转换为 AKShare 需要的代码格式。
    
    标准格式：600519.SH -> AKShare 格式：600519
    标准格式：300750.SZ -> AKShare 格式：300750
    
    Args:
        symbol: 标准股票代码 (如 600519.SH)
        
    Returns:
        AKShare 代码格式 (如 600519)
    """
    if symbol.endswith(".SH") or symbol.endswith(".SZ"):
        return symbol.split(".")[0]
    return symbol


def _normalize_date_to_standard(date_str: str) -> str:
    """
    将日期标准化为 YYYY-MM-DD 格式。
    
    Args:
        date_str: 日期字符串
        
    Returns:
        标准化后的日期字符串 (YYYY-MM-DD)
    """
    if date_str is None:
        return None
    
    date_str = str(date_str)
    
    # 如果已经是 YYYY-MM-DD 格式，直接返回
    if len(date_str) == 10 and date_str.count('-') == 2:
        return date_str
    
    # 处理 YYYYMMDD 格式
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # 其他格式尝试解析
    try:
        for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass
    
    return date_str


def fetch_stock_daily(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 30,
) -> pl.DataFrame:
    """
    使用 akshare 获取股票日线数据。
    
    Args:
        symbol (str): 股票代码，如 "600519.SH"
        start_date (str, optional): 开始日期 (YYYY-MM-DD)
        end_date (str, optional): 结束日期 (YYYY-MM-DD)
        lookback_days (int): 回溯天数，默认 30 天
    
    Returns:
        pl.DataFrame: 包含股票日线数据的 DataFrame
            列：symbol, trade_date, open, high, low, close, volume, amount,
                 adj_factor, turnover_rate, pre_close, change, pct_chg
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed. Please install it: pip install akshare")
        raise
    
    stock_name = TEST_STOCKS.get(symbol, symbol)
    logger.info(f"Fetching data for {symbol} ({stock_name})...")
    
    # 获取 AKShare 代码格式（去掉市场后缀）
    ak_code = _convert_symbol_to_ak_code(symbol)
    logger.debug(f"AKShare code: {ak_code}")
    
    # ========== 日期参数处理 ==========
    today = datetime.now()
    
    if end_date is None:
        end_dt = today
        end_date_str = today.strftime("%Y%m%d")
    else:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_date_str = end_dt.strftime("%Y%m%d")
        except ValueError:
            end_dt = today
            end_date_str = today.strftime("%Y%m%d")
    
    if start_date is None:
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
    
    start_date_str = start_date.replace("-", "")
    
    # 容错：如果 start_date 晚于 end_date，交换它们
    if start_dt > end_dt:
        logger.warning(f"start_date ({start_date}) is after end_date, swapping...")
        start_date, end_date = end_date, start_date
        start_date_str, end_date_str = end_date_str, start_date_str
        start_dt, end_dt = end_dt, start_dt
    
    logger.info(f"Date range: {start_date} to {end_date} ({lookback_days} days)")
    
    try:
        # ========== 使用 AKShare 获取股票历史数据 ==========
        # stock_zh_a_hist 是最稳定的 A 股数据接口
        logger.info(f"Calling ak.stock_zh_a_hist(symbol={ak_code}, period='daily', start_date={start_date_str}, end_date={end_date_str})")
        
        df = ak.stock_zh_a_hist(
            symbol=ak_code,
            period="daily",
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq"  # 使用前复权
        )
        
        # ========== 空数据检查 ==========
        if df is None:
            logger.warning(f"Stock data for {symbol} is None (API returned None)")
            return pl.DataFrame()
        
        if len(df) == 0:
            logger.warning(f"No data returned for {symbol} (empty DataFrame)")
            return pl.DataFrame()
        
        logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
        
        # ========== 转换为 Polars DataFrame ==========
        df_pl = pl.from_pandas(df)
        
        # ========== 列名映射 (中文 -> 英文) ==========
        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }
        
        for old_name, new_name in rename_map.items():
            if old_name in df_pl.columns:
                df_pl = df_pl.rename({old_name: new_name})
        
        # ========== 添加 symbol 列 ==========
        df_pl = df_pl.with_columns(
            pl.lit(symbol).alias("symbol")
        )
        
        # ========== 日期格式处理 ==========
        if "trade_date" in df_pl.columns:
            trade_date_type = df_pl.schema["trade_date"]
            
            if trade_date_type == pl.Utf8:
                df_pl = df_pl.with_columns(
                    pl.col("trade_date")
                    .cast(pl.Utf8)
                    .map_elements(_normalize_date_to_standard, return_dtype=pl.Utf8)
                    .alias("trade_date")
                )
            elif trade_date_type == pl.Datetime:
                df_pl = df_pl.with_columns(
                    pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("trade_date")
                )
            elif trade_date_type == pl.Date:
                df_pl = df_pl.with_columns(
                    pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("trade_date")
                )
        
        # ========== 计算昨收 (pre_close) ==========
        if "pre_close" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col("close").shift(1).over("symbol").alias("pre_close")
            )
        
        # ========== 添加复权因子 (默认为 1) ==========
        if "adj_factor" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.lit(1.0).alias("adj_factor")
            )
        
        # ========== 排序 ==========
        df_pl = df_pl.sort(["symbol", "trade_date"])
        
        # ========== 选择需要的列（对齐 stock_daily 表结构） ==========
        target_columns = [
            "symbol", "trade_date", "open", "high", "low", "close",
            "volume", "amount", "adj_factor", "turnover_rate",
            "pre_close", "change", "pct_chg"
        ]
        
        available_columns = [c for c in target_columns if c in df_pl.columns]
        df_pl = df_pl.select(available_columns)
        
        # ========== 数值列类型转换 ==========
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", 
                          "adj_factor", "turnover_rate", "pre_close", "change", "pct_chg"]
        for col in numeric_columns:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        logger.info(f"Stock data processing complete for {symbol}: {len(df_pl)} rows")
        
        return df_pl
        
    except Exception as e:
        logger.error(f"Failed to fetch stock data for {symbol}: {e}", exc_info=True)
        return pl.DataFrame()


def sync_stocks_to_db(
    symbols: list[str] = None,
    db: DatabaseManager = None,
    table_name: str = "stock_daily",
    lookback_days: int = 30,
) -> dict[str, int]:
    """
    同步测试股票数据到数据库。
    
    Args:
        symbols (list[str]): 要同步的股票代码列表
        db (DatabaseManager): 数据库管理器
        table_name (str): 目标表名
        lookback_days (int): 回溯天数
    
    Returns:
        dict[str, int]: 每个股票同步的行数
    """
    if symbols is None:
        symbols = list(TEST_STOCKS.keys())
    
    if db is None:
        db = DatabaseManager.get_instance()
    
    results = {}
    
    for symbol in symbols:
        stock_name = TEST_STOCKS.get(symbol, symbol)
        logger.info(f"Syncing {symbol} ({stock_name})...")
        
        # 获取数据
        df = fetch_stock_daily(symbol, lookback_days=lookback_days)
        
        if df.is_empty():
            logger.warning(f"No data for {symbol}, skipping")
            results[symbol] = 0
            continue
        
        # 写入数据库
        try:
            # 先删除已有数据（避免主键冲突）
            dates = df["trade_date"].unique().to_list()
            if dates:
                dates_str = "', '".join([str(d) for d in dates])
                
                delete_query = f"""
                    DELETE FROM {table_name}
                    WHERE symbol = '{symbol}'
                    AND trade_date IN ('{dates_str}')
                """
                
                logger.debug(f"Executing delete query: {delete_query}")
                db.execute(delete_query)
            
            # 插入新数据
            rows = db.to_sql(df, table_name, if_exists="append")
            results[symbol] = rows
            
            logger.info(f"Synced {rows} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to write {symbol} to database: {e}")
            results[symbol] = 0
    
    return results


def create_stock_daily_table(db: DatabaseManager = None) -> None:
    """
    创建 stock_daily 表。
    
    表结构对齐新规范:
    - symbol (varchar(20)): 股票代码
    - trade_date (DATE): 交易日期
    - 价格字段 (decimal(18,4))
    - 成交量/额 (double)
    
    Args:
        db (DatabaseManager): 数据库管理器
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    create_sql = """
    CREATE TABLE IF NOT EXISTS `stock_daily` (
      `symbol` varchar(20) NOT NULL COMMENT '代码 (如 600519.SH)',
      `trade_date` DATE NOT NULL COMMENT '交易日期 (YYYY-MM-DD)',
      `open` decimal(18,4) DEFAULT NULL,
      `high` decimal(18,4) DEFAULT NULL,
      `low` decimal(18,4) DEFAULT NULL,
      `close` decimal(18,4) DEFAULT NULL,
      `volume` double DEFAULT NULL COMMENT '成交量',
      `amount` double DEFAULT NULL COMMENT '成交额',
      `adj_factor` decimal(18,6) DEFAULT '1.000000' COMMENT '复权因子',
      `turnover_rate` decimal(18,4) DEFAULT NULL COMMENT '换手率',
      `pre_close` decimal(18,4) DEFAULT NULL,
      `change` decimal(18,4) DEFAULT NULL,
      `pct_chg` decimal(18,4) DEFAULT NULL,
      PRIMARY KEY (`symbol`,`trade_date`),
      KEY `idx_date` (`trade_date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='股票日线行情表';
    """
    
    try:
        db.execute(create_sql)
        logger.info("stock_daily table created successfully (aligned with new schema)")
    except Exception as e:
        logger.error(f"Failed to create stock_daily table: {e}")
        raise


def prepare_test_data(lookback_days: int = 30) -> None:
    """
    准备测试数据的主函数。
    
    Args:
        lookback_days (int): 回溯天数，默认 30 天
    """
    logger.info("=" * 60)
    logger.info("PREPARE TEST DATA - Starting")
    logger.info(f"Target stocks: {list(TEST_STOCKS.keys())}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info("=" * 60)
    
    db = DatabaseManager.get_instance()
    
    # 创建表（如果不存在）
    create_stock_daily_table(db)
    
    # 同步数据
    results = sync_stocks_to_db(db=db, lookback_days=lookback_days)
    
    # 输出结果
    logger.info("=" * 60)
    logger.info("PREPARE TEST DATA COMPLETE")
    logger.info("=" * 60)
    
    total_rows = 0
    for symbol, rows in results.items():
        stock_name = TEST_STOCKS.get(symbol, symbol)
        status = "✓" if rows > 0 else "✗"
        logger.info(f"  {status} {symbol} ({stock_name}): {rows} rows")
        total_rows += rows
    
    logger.info(f"Total: {total_rows} rows synced")
    
    if total_rows > 0:
        logger.info("\n✅ Test data preparation complete!")
        logger.info("You can now run: python src/daily_trade_advisor.py --test --mode draft")
    else:
        logger.warning("\n⚠️ No data was synced. Please check the logs for errors.")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行准备测试数据
    prepare_test_data(lookback_days=30)