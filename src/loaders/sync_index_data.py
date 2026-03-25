"""
Index Data Sync Module - Sync major market indices using AKShare.

This module syncs:
- 中证 500 (000905.SH) - CSI 500 Index
- 上证指数 (000001.SH) - Shanghai Composite
- 深证成指 (399001.SZ) - Shenzhen Component

核心功能:
    - 使用 akshare 抓取指数历史日线数据
    - 计算 MA20 均线
    - 存入 MySQL 数据库 index_daily 表

表结构对齐:
    index_daily 表字段:
    - symbol (varchar(20)): 指数代码 (如 000905.SH)
    - trade_date (DATE): 交易日期 (YYYY-MM-DD)
    - open, high, low, close (decimal(18,4))
    - pre_close, change, pct_chg (decimal(18,4))
    - volume (double): 成交量
    - amount (double): 成交额
    - ma20 (decimal(18,4)): 20 日均线
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


# 主要指数代码列表
# 注意：AKShare 指数代码格式为小写 + 市场前缀
# 上证指数：sh000001, 深证成指：sz399001, 中证 500：sh000905
INDEX_CODES = {
    "000905.SH": {"name": "中证 500", "ak_code": "sh000905"},
    "000001.SH": {"name": "上证指数", "ak_code": "sh000001"},
    "399001.SZ": {"name": "深证成指", "ak_code": "sz399001"},
    "000300.SH": {"name": "沪深 300", "ak_code": "sh000300"},
    "000016.SH": {"name": "上证 50", "ak_code": "sh000016"},
}


def _normalize_date_to_standard(date_str: str) -> str:
    """
    将日期标准化为 YYYY-MM-DD 格式。
    
    支持多种输入格式:
    - YYYYMMDD (如 20260310)
    - YYYY-MM-DD (如 2026-03-10)
    - YYYY/MM/DD (如 2026/03/10)
    
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
    
    # 处理 YYYY/MM/DD 格式
    if len(date_str) == 10 and date_str.count('/') == 2:
        return date_str.replace('/', '-')
    
    # 其他格式尝试解析
    try:
        # 尝试多种格式
        for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass
    
    return date_str


def _convert_symbol_to_ak_code(symbol: str) -> str:
    """
    将标准 symbol 格式转换为 AKShare 需要的代码格式。
    
    标准格式：000001.SH -> AKShare 格式：sh000001
    标准格式：399001.SZ -> AKShare 格式：sz399001
    
    Args:
        symbol: 标准股票代码 (如 000001.SH)
        
    Returns:
        AKShare 代码格式 (如 sh000001)
    """
    if symbol.endswith(".SH"):
        code = symbol.replace(".SH", "")
        return f"sh{code}"
    elif symbol.endswith(".SZ"):
        code = symbol.replace(".SZ", "")
        return f"sz{code}"
    else:
        # 如果没有市场后缀，尝试根据代码判断
        if symbol.startswith("0") or symbol.startswith("6"):
            return f"sh{symbol}"
        elif symbol.startswith("3") or symbol.startswith("2"):
            return f"sz{symbol}"
        return symbol


def fetch_index_daily(
    index_code: str,
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 60,
) -> pl.DataFrame:
    """
    使用 akshare 获取指数日线数据。
    
    Args:
        index_code (str): 指数代码，如 "000905.SH"
        start_date (str, optional): 开始日期 (YYYY-MM-DD)
        end_date (str, optional): 结束日期 (YYYY-MM-DD)
        lookback_days (int): 回溯天数，默认 60 天
    
    Returns:
        pl.DataFrame: 包含指数日线数据的 DataFrame
            列：symbol, trade_date, open, high, low, close, pre_close, 
                 change, pct_chg, volume, amount
    
    注意:
        - 使用 akshare 的 stock_zh_index_hist_em 接口
        - 数据会自动按日期排序
        - 日期格式统一为 YYYY-MM-DD
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed. Please install it: pip install akshare")
        raise
    
    logger.info(f"Fetching index data for {index_code}...")
    
    # 获取 AKShare 代码格式
    ak_code = _convert_symbol_to_ak_code(index_code)
    logger.debug(f"AKShare code: {ak_code}")
    
    # ========== 日期参数处理 ==========
    # 计算回溯日期（容错处理）
    today = datetime.now()
    
    if end_date is None:
        # 默认使用今天，但如果今天还没收盘，使用最近一个交易日
        end_dt = today
        end_date = today.strftime("%Y-%m-%d")
    else:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            end_dt = today
            end_date = today.strftime("%Y-%m-%d")
    
    if start_date is None:
        # 默认回溯 lookback_days 天
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
    
    # 容错：如果 start_date 晚于 end_date，交换它们
    if start_dt > end_dt:
        logger.warning(f"start_date ({start_date}) is after end_date, swapping...")
        start_date, end_date = end_date, start_date
        start_dt, end_dt = end_dt, start_dt
    
    # 确保 end_date 是字符串格式用于 API 调用
    end_date_str = end_dt.strftime("%Y%m%d")
    start_date_str = start_date.replace("-", "")
    
    logger.info(f"Date range: {start_date} to {end_date} ({lookback_days} days)")
    
    try:
        # ========== 使用 AKShare 获取指数历史数据 ==========
        # 尝试多个接口，带重试机制
        df = None
        
        # 接口列表（按优先级）
        api_funcs = [
            # 接口 1: stock_zh_index_hist_csindex (中证指数官方接口)
            lambda: ak.stock_zh_index_hist_csindex(symbol=ak_code),
            # 接口 2: index_zh_a_hist (东方财富接口)
            lambda: ak.index_zh_a_hist(symbol=ak_code, period="daily", start_date=start_date_str, end_date=end_date_str),
            # 接口 3: stock_zh_index_daily (通用接口)
            lambda: ak.stock_zh_index_daily(symbol=ak_code),
        ]
        
        for i, api_func in enumerate(api_funcs):
            try:
                logger.info(f"Trying API {i+1}/{len(api_funcs)}: {api_func.__name__ if hasattr(api_func, '__name__') else 'lambda'}")
                df = api_func()
                if df is not None and len(df) > 0:
                    logger.info(f"Successfully fetched data using API {i+1}")
                    break
                else:
                    logger.warning(f"API {i+1} returned empty data, trying next...")
            except Exception as api_error:
                logger.warning(f"API {i+1} failed: {api_error}, trying next...")
                continue
        
        # 如果所有接口都失败，返回空 DataFrame
        if df is None or len(df) == 0:
            logger.warning(f"All APIs failed for {index_code}")
            return pl.DataFrame()
        
        # ========== 空数据检查 ==========
        if df is None:
            logger.warning(f"Index data for {index_code} is None (API returned None)")
            return pl.DataFrame()
        
        if len(df) == 0:
            logger.warning(f"No data returned for {index_code} (empty DataFrame)")
            logger.warning(f"Possible reasons: 1) Invalid index code, 2) Network issue, 3) Date range issue")
            return pl.DataFrame()
        
        logger.info(f"Successfully fetched {len(df)} rows for {index_code}")
        
        # ========== 转换为 Polars DataFrame ==========
        df_pl = pl.from_pandas(df)
        
        # ========== 【修复 1】统一列名映射 - 检查并统一 date/日期 -> trade_date ==========
        # 首先检查是否存在 date 列（AKShare 接口返回的列名）
        if "date" in df_pl.columns and "trade_date" not in df_pl.columns:
            logger.info("Renaming 'date' column to 'trade_date'")
            df_pl = df_pl.rename({"date": "trade_date"})
        
        # 检查是否存在 日期 列（中文列名）
        if "日期" in df_pl.columns and "trade_date" not in df_pl.columns:
            logger.info("Renaming '日期' column to 'trade_date'")
            df_pl = df_pl.rename({"日期": "trade_date"})
        
        # 标准列名映射 (中文 -> 英文)
        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "昨收": "pre_close",
            "涨跌额": "change",
            "涨跌幅": "pct_chg",
        }
        
        for old_name, new_name in rename_map.items():
            if old_name in df_pl.columns:
                df_pl = df_pl.rename({old_name: new_name})
        
        # ========== 添加 symbol 列 ==========
        df_pl = df_pl.with_columns(
            pl.lit(index_code).alias("symbol")
        )
        
        # ========== 【修复 2】补齐字段：pre_close, change, pct_chg ==========
        # 检查并计算 pre_close（昨收）
        if "pre_close" not in df_pl.columns:
            logger.info("Computing pre_close from close.shift(1)")
            df_pl = df_pl.with_columns(
                pl.col("close").shift(1).over("symbol").alias("pre_close")
            )
        
        # 检查并计算 change（涨跌额）
        if "change" not in df_pl.columns:
            logger.info("Computing change = close - pre_close")
            df_pl = df_pl.with_columns(
                (pl.col("close") - pl.col("pre_close")).alias("change")
            )
        
        # 检查并计算 pct_chg（涨跌幅）
        if "pct_chg" not in df_pl.columns:
            logger.info("Computing pct_chg = (close / pre_close - 1) * 100")
            df_pl = df_pl.with_columns(
                ((pl.col("close") / pl.col("pre_close") - 1) * 100).alias("pct_chg")
            )
        
        # ========== 日期格式处理 ==========
        if "trade_date" in df_pl.columns:
            trade_date_type = df_pl.schema["trade_date"]
            
            if trade_date_type == pl.Utf8:
                # 字符串类型，标准化为 YYYY-MM-DD
                df_pl = df_pl.with_columns(
                    pl.col("trade_date")
                    .cast(pl.Utf8)
                    .map_elements(_normalize_date_to_standard, return_dtype=pl.Utf8)
                    .alias("trade_date")
                )
            elif trade_date_type == pl.Datetime:
                # 日期时间类型，转换为日期字符串
                df_pl = df_pl.with_columns(
                    pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("trade_date")
                )
            elif trade_date_type == pl.Date:
                # 日期类型，转换为字符串
                df_pl = df_pl.with_columns(
                    pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("trade_date")
                )
        
        # ========== 排序 ==========
        df_pl = df_pl.sort(["symbol", "trade_date"])
        
        # ========== 【修复 3】字段对齐 - 严格对齐数据库 index_daily 表结构 ==========
        # index_daily 表字段：symbol, trade_date, open, high, low, close, 
        #                     pre_close, change, pct_chg, volume, amount, ma20
        target_columns = [
            "symbol", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "volume", "amount"
        ]
        
        available_columns = [c for c in target_columns if c in df_pl.columns]
        logger.info(f"Available columns after mapping: {available_columns}")
        df_pl = df_pl.select(available_columns)
        
        # ========== 数值列类型转换 ==========
        numeric_columns = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]
        for col in numeric_columns:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        logger.info(f"Index data processing complete for {index_code}: {len(df_pl)} rows, columns: {df_pl.columns}")
        
        return df_pl
        
    except Exception as e:
        logger.error(f"Failed to fetch index data for {index_code}: {e}", exc_info=True)
        return pl.DataFrame()


def calculate_ma20(df: pl.DataFrame) -> pl.DataFrame:
    """
    计算 20 日均线。
    
    Args:
        df (pl.DataFrame): 包含 close 列的 DataFrame
    
    Returns:
        pl.DataFrame: 添加了 ma20 列的 DataFrame
    """
    logger.info("Calculating MA20...")
    
    # ========== 【修复 4】增加容错：检查 DataFrame 是否为空 ==========
    if df.is_empty():
        logger.warning("DataFrame is empty, skipping MA20 calculation")
        return df
    
    if "close" not in df.columns:
        logger.error("close column not found")
        return df
    
    # 检查数据量是否足够计算 MA20
    if len(df) < 20:
        logger.warning(f"Insufficient data ({len(df)} rows) for MA20 calculation, but proceeding anyway")
    
    # 确保数据已排序
    df = df.sort(["symbol", "trade_date"])
    
    try:
        df = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).over("symbol").alias("ma20")
        )
        logger.info("MA20 calculation completed")
    except Exception as e:
        logger.error(f"Failed to calculate MA20: {e}")
        # 容错：添加空的 ma20 列
        df = df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("ma20")
        )
    
    return df


def sync_index_to_db(
    index_codes: list[str] = None,
    db: DatabaseManager = None,
    table_name: str = "index_daily",
    lookback_days: int = 60,
) -> dict[str, int]:
    """
    同步指数数据到数据库。
    
    Args:
        index_codes (list[str]): 要同步的指数代码列表
        db (DatabaseManager): 数据库管理器
        table_name (str): 目标表名
        lookback_days (int): 回溯天数
    
    Returns:
        dict[str, int]: 每个指数同步的行数
    """
    if index_codes is None:
        index_codes = list(INDEX_CODES.keys())
    
    if db is None:
        db = DatabaseManager.get_instance()
    
    results = {}
    
    for code in index_codes:
        index_name = INDEX_CODES.get(code, {}).get("name", "Unknown")
        logger.info(f"Syncing {code} ({index_name})...")
        
        # 获取数据
        df = fetch_index_daily(code, lookback_days=lookback_days)
        
        if df.is_empty():
            logger.warning(f"No data for {code}, skipping")
            results[code] = 0
            continue
        
        # 计算 MA20
        df = calculate_ma20(df)
        
        # 转换数值列为 Float64
        numeric_columns = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "ma20"]
        for col in numeric_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        # 写入数据库
        try:
            # 先删除已有数据（避免主键冲突）
            dates = df["trade_date"].unique().to_list()
            if dates:
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
            logger.error(f"Failed to write {code} to database: {e}")
            results[code] = 0
    
    return results


def create_index_daily_table(db: DatabaseManager = None) -> None:
    """
    创建 index_daily 表。
    
    表结构对齐新规范:
    - symbol (varchar(20)): 指数代码
    - trade_date (DATE): 交易日期
    - 价格字段 (decimal(18,4))
    - 成交量/额 (double)
    
    Args:
        db (DatabaseManager): 数据库管理器
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    create_sql = """
    CREATE TABLE IF NOT EXISTS `index_daily` (
      `symbol` varchar(20) NOT NULL COMMENT '指数代码 (如 000905.SH)',
      `trade_date` DATE NOT NULL COMMENT '交易日期 (YYYY-MM-DD)',
      `open` decimal(18,4) DEFAULT NULL,
      `high` decimal(18,4) DEFAULT NULL,
      `low` decimal(18,4) DEFAULT NULL,
      `close` decimal(18,4) DEFAULT NULL,
      `pre_close` decimal(18,4) DEFAULT NULL,
      `change` decimal(18,4) DEFAULT NULL,
      `pct_chg` decimal(18,4) DEFAULT NULL,
      `volume` double DEFAULT NULL COMMENT '成交量',
      `amount` double DEFAULT NULL COMMENT '成交额',
      `ma20` decimal(18,4) DEFAULT NULL COMMENT '20 日均线',
      PRIMARY KEY (`symbol`,`trade_date`),
      KEY `idx_date` (`trade_date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='指数日线行情表';
    """
    
    try:
        db.execute(create_sql)
        logger.info("index_daily table created successfully (aligned with new schema)")
    except Exception as e:
        logger.error(f"Failed to create index_daily table: {e}")
        raise


def sync_all_indices(lookback_days: int = 60) -> None:
    """
    同步所有主要指数数据。
    
    这是主入口函数，会:
    1. 创建表结构（如果不存在）
    2. 获取所有指数数据（默认回溯 60 天）
    3. 计算 MA20
    4. 写入数据库
    
    Args:
        lookback_days (int): 回溯天数，默认 60 天
    """
    logger.info("=" * 60)
    logger.info("INDEX DATA SYNC - Starting")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info("=" * 60)
    
    db = DatabaseManager.get_instance()
    
    # 创建表
    create_index_daily_table(db)
    
    # 同步数据
    results = sync_index_to_db(db=db, lookback_days=lookback_days)
    
    # 输出结果
    logger.info("=" * 60)
    logger.info("SYNC COMPLETE")
    logger.info("=" * 60)
    
    total_rows = 0
    for code, rows in results.items():
        index_name = INDEX_CODES.get(code, {}).get("name", "Unknown")
        status = "✓" if rows > 0 else "✗"
        logger.info(f"  {status} {code} ({index_name}): {rows} rows")
        total_rows += rows
    
    logger.info(f"Total: {total_rows} rows synced")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行同步（默认回溯 60 天）
    sync_all_indices(lookback_days=60)