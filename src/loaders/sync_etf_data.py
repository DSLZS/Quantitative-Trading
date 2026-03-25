"""
ETF Data Sync Module - Sync ETF data using AKShare.

This module syncs:
- 国债 ETF (511010) - Government Bond ETF
- 十年国债 ETF (511260)
- 黄金 ETF (518880)

核心功能:
    - 使用 akshare 的 fund_etf_hist_em (东方财富接口) 抓取 ETF 历史日线数据
    - 日期格式严格校验 (YYYY-MM-DD)
    - 增量更新，避免重复插入
    - 存入 MySQL 数据库 etf_daily 表

表结构对齐:
    etf_daily 表字段:
    - symbol (varchar(20)): ETF 代码 (如 511010)
    - trade_date (date): 交易日期
    - open, high, low, close (decimal(18,4))
    - volume (double): 成交量
    - amount (double): 成交额

使用示例:
    >>> python src/sync_etf_data.py
    # 同步所有配置的 ETF 数据
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


# ETF 代码列表
ETF_CODES = {
    "511010": "国债 ETF",
    "511260": "十年国债 ETF",
    "518880": "黄金 ETF",
}


def _normalize_date(date_str: str) -> str:
    """
    标准化日期格式为 YYYYMMDD（去掉横杠）。
    
    Args:
        date_str: 日期字符串，支持 YYYY-MM-DD 或 YYYYMMDD 格式
        
    Returns:
        标准化后的日期字符串 (YYYYMMDD)
    """
    if date_str is None:
        return None
    # 去掉横杠
    return str(date_str).replace("-", "")


def fetch_etf_daily(
    etf_code: str,
    start_date: str = None,
    end_date: str = None,
    adjust: str = "qfq",
) -> pl.DataFrame:
    """
    使用 akshare 获取 ETF 日线数据（东方财富接口）。
    
    Args:
        etf_code (str): ETF 代码，如 "511010"
        start_date (str, optional): 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
                                   默认为 "20200101"
        end_date (str, optional): 结束日期 (YYYYMMDD 或 YYYY-MM-DD)
                                 默认为当天
        adjust (str, optional): 复权类型，可选值：
            - "qfq": 前复权 (默认，推荐用于回测和策略)
            - "hfq": 后复权
            - "": 不复权
    
    Returns:
        pl.DataFrame: 包含 ETF 日线数据的 DataFrame
            列：symbol, trade_date, open, high, low, close, volume, amount
            
    注意:
        - 使用 akshare 的 fund_etf_hist_em 接口（东方财富，最稳定）
        - 日期格式必须为 YYYYMMDD（不带横杠）写入，但输出为 YYYY-MM-DD
        - 参数为空时自动设置默认值
        - 数据会自动按日期排序
        - 字段严格对齐现有 etf_daily 表结构
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed. Please install it: pip install akshare")
        raise
    
    logger.info(f"Fetching ETF data for {etf_code}...")
    
    # ========== 日期参数处理：强制设置默认值 ==========
    # 如果 start_date 为空，默认从 20200101 开始
    if not start_date:
        start_date = "20200101"
    else:
        start_date = _normalize_date(start_date)
    
    # 如果 end_date 为空，默认为当天
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    else:
        end_date = _normalize_date(end_date)
    
    logger.debug(f"Date range: {start_date} to {end_date}, adjust={adjust}")
    
    try:
        # ========== 使用东方财富接口获取 ETF 历史数据 ==========
        # fund_etf_hist_em 是最稳定的 ETF 数据接口
        # 参数说明:
        #   symbol: ETF 代码 (如 511010)
        #   period: 周期 (daily=日线，weekly=周线，monthly=月线)
        #   start_date: 开始日期 (YYYYMMDD)
        #   end_date: 结束日期 (YYYYMMDD)
        #   adjust: 复权类型 ("qfq"=前复权，"hfq"=后复权，""=不复权)
        logger.info(f"Calling ak.fund_etf_hist_em(symbol={etf_code}, period='daily', start_date={start_date}, end_date={end_date}, adjust='{adjust}')")
        
        df = ak.fund_etf_hist_em(
            symbol=etf_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        # ========== 空数据检查 ==========
        if df is None:
            logger.warning(f"ETF data for {etf_code} is None (API returned None)")
            logger.warning("Possible causes: 1) Invalid ETF code, 2) Network issue, 3) API rate limit")
            return pl.DataFrame()
        
        if len(df) == 0:
            logger.warning(f"No data returned for {etf_code} (empty DataFrame)")
            logger.warning(f"Possible reasons: 1) ETF not found, 2) Date range issue, 3) Market closed")
            return pl.DataFrame()
        
        logger.info(f"Successfully fetched {len(df)} rows for {etf_code}")
        
        # ========== 转换为 Polars DataFrame ==========
        df_pl = pl.from_pandas(df)
        
        # ========== 列名映射 (中文 -> 英文) ==========
        # 东方财富接口返回的列名是中文，需要映射为数据库字段名
        # 只保留 etf_daily 表中存在的字段：symbol, trade_date, open, high, low, close, volume, amount
        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        
        for old_name, new_name in rename_map.items():
            if old_name in df_pl.columns:
                df_pl = df_pl.rename({old_name: new_name})
        
        # ========== 添加 symbol 列 ==========
        # 注意：使用 symbol 而非 ts_code，不添加交易所后缀
        df_pl = df_pl.with_columns(
            pl.lit(etf_code).alias("symbol")
        )
        
        # ========== 日期格式处理 ==========
        # 确保日期列为 YYYY-MM-DD 格式（Polars Date 类型）
        if "trade_date" in df_pl.columns:
            trade_date_type = df_pl.schema["trade_date"]
            if trade_date_type == pl.Utf8:
                # 如果是字符串，移除可能的分隔符后重新格式化为 YYYY-MM-DD
                df_pl = df_pl.with_columns(
                    pl.col("trade_date")
                    .str.replace_all("-", "")
                    .str.to_date("%Y%m%d")
                    .alias("trade_date")
                )
            elif trade_date_type == pl.Datetime:
                # 如果是 datetime 类型，转换为 Date
                df_pl = df_pl.with_columns(
                    pl.col("trade_date").dt.date().alias("trade_date")
                )
        
        # ========== 排序 ==========
        df_pl = df_pl.sort(["symbol", "trade_date"])
        
        # ========== 选择需要的列（严格对齐表结构） ==========
        # etf_daily 表字段：symbol, trade_date, open, high, low, close, volume, amount
        target_columns = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        
        available_columns = [c for c in target_columns if c in df_pl.columns]
        df_pl = df_pl.select(available_columns)
        
        # ========== 数值列类型转换 ==========
        numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_columns:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        logger.info(f"ETF data processing complete for {etf_code}: {len(df_pl)} rows, columns: {df_pl.columns}")
        logger.debug(f"DataFrame schema: {df_pl.schema}")
        
        return df_pl
        
    except Exception as e:
        logger.error(f"Failed to fetch ETF data for {etf_code}: {e}", exc_info=True)
        return pl.DataFrame()


def sync_etf_to_db(
    etf_codes: list[str] = None,
    db: DatabaseManager = None,
    table_name: str = "etf_daily",
    incremental: bool = True,
) -> dict[str, int]:
    """
    同步 ETF 数据到数据库。
    
    Args:
        etf_codes (list[str]): 要同步的 ETF 代码列表
        db (DatabaseManager): 数据库管理器
        table_name (str): 目标表名
        incremental (bool): 是否增量更新
    
    Returns:
        dict[str, int]: 每个 ETF 同步的行数
    """
    if etf_codes is None:
        etf_codes = list(ETF_CODES.keys())
    
    if db is None:
        db = DatabaseManager.get_instance()
    
    results = {}
    
    for code in etf_codes:
        logger.info(f"Syncing {code} ({ETF_CODES.get(code, 'Unknown')})...")
        
        # 获取数据 - 使用前复权模式
        df = fetch_etf_daily(code, adjust="qfq")
        
        if df.is_empty():
            logger.warning(f"No data for {code}, skipping")
            results[code] = 0
            continue
        
        # ========== 调试日志：输出 DataFrame 列名 ==========
        logger.debug(f"Inserting into {table_name}: {df.columns}")
        
        # 写入数据库
        try:
            if incremental:
                # 增量更新：先删除已有数据（避免主键冲突）
                # 注意：使用 symbol 字段进行匹配
                dates = df["trade_date"].unique().to_list()
                if dates:
                    # 将日期转换为字符串格式用于 SQL
                    dates_str = "', '".join([str(d) for d in dates])
                    
                    delete_query = f"""
                        DELETE FROM {table_name}
                        WHERE symbol = '{code}'
                        AND trade_date IN ('{dates_str}')
                    """
                    
                    logger.debug(f"Executing delete query: {delete_query}")
                    db.execute(delete_query)
            
            # 插入新数据
            rows = db.to_sql(df, table_name, if_exists="append")
            results[code] = rows
            
            logger.info(f"Synced {rows} rows for {code}")
            
        except Exception as e:
            logger.error(f"Failed to write {code} to database: {e}", exc_info=True)
            results[code] = 0
    
    return results


def create_etf_daily_table(db: DatabaseManager = None) -> None:
    """
    创建 etf_daily 表。
    
    表结构完全对齐现有数据库:
    - symbol (varchar(20)): ETF 代码 (如 511010)
    - trade_date (date): 交易日期
    - open, high, low, close (decimal(18,4))
    - volume (double): 成交量
    - amount (double): 成交额
    
    Args:
        db (DatabaseManager): 数据库管理器
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    # 完全对齐现有数据库表结构
    create_sql = """
    CREATE TABLE IF NOT EXISTS `etf_daily` (
      `symbol` varchar(20) NOT NULL,
      `trade_date` date NOT NULL,
      `open` decimal(18,4),
      `high` decimal(18,4),
      `low` decimal(18,4),
      `close` decimal(18,4),
      `volume` double,
      `amount` double,
      PRIMARY KEY (`symbol`,`trade_date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ETF 日线行情表';
    """
    
    try:
        db.execute(create_sql)
        logger.info("etf_daily table created successfully (aligned with existing schema)")
    except Exception as e:
        logger.error(f"Failed to create etf_daily table: {e}", exc_info=True)
        raise


def sync_all_etfs() -> None:
    """
    同步所有 ETF 数据。
    
    这是主入口函数，会:
    1. 创建表结构（如果不存在）
    2. 获取所有 ETF 数据（前复权模式）
    3. 写入数据库
    """
    logger.info("=" * 60)
    logger.info("ETF DATA SYNC - Starting")
    logger.info("=" * 60)
    
    db = DatabaseManager.get_instance()
    
    # 创建表（如果不存在）
    create_etf_daily_table(db)
    
    # 同步数据（使用前复权）
    results = sync_etf_to_db(db=db)
    
    # 输出结果
    logger.info("=" * 60)
    logger.info("SYNC COMPLETE")
    logger.info("=" * 60)
    
    total_rows = 0
    for code, rows in results.items():
        status = "✓" if rows > 0 else "✗"
        logger.info(f"  {status} {code} ({ETF_CODES.get(code, 'Unknown')}): {rows} rows")
        total_rows += rows
    
    logger.info(f"Total: {total_rows} rows synced (adjusted: qfq)")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行同步
    sync_all_etfs()