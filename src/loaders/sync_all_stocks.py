"""
All Stocks Data Sync Module - 同步沪深 300+ 中证 500 成分股历史日线数据。

核心功能:
    - 多数据源冗余获取沪深 300 (000300.SH) 和 中证 500 (000905.SH) 成分股列表
    - 同步每只成分股最近 2 年的历史日线数据
    - 增量更新 stock_daily 表，避免重复插入
    - 重试机制：每只股票失败后自动重试 3 次
    - 每 50 只股票 commit 一次，防止连接超时
    - 随机延迟 (0.2-0.5 秒) 防止 IP 被封禁

使用示例:
    >>> python src/sync_all_stocks.py
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量
# ===========================================
DEFAULT_LOOKBACK_DAYS = 730  # 2 年
COMMIT_INTERVAL = 50  # 每 50 只股票 commit 一次
MAX_RETRIES = 3  # 最大重试次数
RANDOM_DELAY_MIN = 0.2  # 随机延迟最小值 (秒)
RANDOM_DELAY_MAX = 0.5  # 随机延迟最大值 (秒)


# ===========================================
# 内置备用成分股列表（20 只蓝筹股）
# ===========================================
FALLBACK_STOCKS = [
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "300750.SZ", "name": "宁德时代"},
    {"symbol": "000858.SZ", "name": "五粮液"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "600036.SH", "name": "招商银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "002415.SZ", "name": "海康威视"},
    {"symbol": "601888.SH", "name": "中国中免"},
    {"symbol": "600276.SH", "name": "恒瑞医药"},
    {"symbol": "601166.SH", "name": "兴业银行"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000002.SZ", "name": "万科 A"},
    {"symbol": "600030.SH", "name": "中信证券"},
    {"symbol": "000651.SZ", "name": "格力电器"},
    {"symbol": "000725.SZ", "name": "京东方 A"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "300059.SZ", "name": "东方财富"},
    {"symbol": "601398.SH", "name": "工商银行"},
    {"symbol": "601988.SH", "name": "中国银行"},
    {"symbol": "601857.SH", "name": "中国石油"},
]


def _add_market_suffix(symbol: str) -> str:
    """
    为股票代码添加市场后缀 (.SH/.SZ)。
    
    Args:
        symbol: 纯数字股票代码
        
    Returns:
        带市场后缀的代码
    """
    if "." in symbol:
        return symbol
    
    symbol = str(symbol).strip()
    
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    elif symbol.startswith("0") or symbol.startswith("3"):
        return f"{symbol}.SZ"
    elif symbol.startswith("4") or symbol.startswith("8"):
        return f"{symbol}.BJ"
    
    return symbol


def fetch_constituents_from_csindex(index_code: str) -> List[Dict[str, str]]:
    """
    从中证指数官网获取成分股列表。
    
    Args:
        index_code: 指数代码（如 000300, 000905）
        
    Returns:
        成分股列表
    """
    try:
        import akshare as ak
        
        logger.info(f"Trying csindex API for {index_code}...")
        
        # 方法 1: index_stock_cons_csindex
        df = ak.index_stock_cons_csindex(symbol=index_code)
        
        if df is not None and len(df) > 0:
            result = []
            df_pl = pl.from_pandas(df)
            
            # 尝试不同的列名
            symbol_col = None
            name_col = None
            
            for col in df_pl.columns:
                if "代码" in col or "code" in col.lower() or "symbol" in col.lower():
                    symbol_col = col
                if "名称" in col or "name" in col.lower():
                    name_col = col
            
            if symbol_col:
                for row in df_pl.iter_rows(named=True):
                    symbol = str(row.get(symbol_col, "")).strip()
                    name = str(row.get(name_col, "")).strip() if name_col else ""
                    
                    if symbol and symbol != "nan":
                        result.append({
                            "symbol": _add_market_suffix(symbol),
                            "name": name
                        })
            
            logger.info(f"csindex API returned {len(result)} stocks for {index_code}")
            return result
            
    except Exception as e:
        logger.warning(f"csindex API failed for {index_code}: {e}")
    
    return []


def fetch_constituents_from_akshare(index_code: str) -> List[Dict[str, str]]:
    """
    从 AKShare 其他接口获取成分股列表。
    
    Args:
        index_code: 指数代码
        
    Returns:
        成分股列表
    """
    try:
        import akshare as ak
        
        logger.info(f"Trying akshare API for {index_code}...")
        
        # 尝试不同的 API
        apis_to_try = [
            ("index_stock_cons", {"symbol": index_code}),
            ("index_stock_info", {}),
        ]
        
        for api_name, kwargs in apis_to_try:
            try:
                if hasattr(ak, api_name):
                    df = getattr(ak, api_name)(**kwargs)
                    
                    if df is not None and len(df) > 0:
                        # 过滤出目标指数的成分股
                        df_pl = pl.from_pandas(df)
                        
                        # 尝试找到指数代码列
                        for col in df_pl.columns:
                            if "指数" in col or "index" in col.lower():
                                if index_code in str(df_pl[col][0]):
                                    # 找到成分股列
                                    symbol_col = None
                                    name_col = None
                                    
                                    for c in df_pl.columns:
                                        if "代码" in c or "code" in c.lower():
                                            symbol_col = c
                                        if "名称" in c or "name" in c.lower():
                                            name_col = c
                                    
                                    if symbol_col:
                                        result = []
                                        for row in df_pl.iter_rows(named=True):
                                            symbol = str(row.get(symbol_col, "")).strip()
                                            name = str(row.get(name_col, "")).strip() if name_col else ""
                                            
                                            if symbol and symbol != "nan":
                                                result.append({
                                                    "symbol": _add_market_suffix(symbol),
                                                    "name": name
                                                })
                                        
                                        logger.info(f"{api_name} returned {len(result)} stocks for {index_code}")
                                        return result
                                        
            except Exception:
                continue
                
    except Exception as e:
        logger.warning(f"akshare API failed for {index_code}: {e}")
    
    return []


def fetch_constituents_from_etf(index_code: str) -> List[Dict[str, str]]:
    """
    从 ETF 成分股角度获取指数成分股。
    
    Args:
        index_code: 指数代码
        
    Returns:
        成分股列表
    """
    try:
        import akshare as ak
        
        logger.info(f"Trying ETF-based approach for {index_code}...")
        
        # 尝试获取跟踪该指数的 ETF
        # 沪深 300 ETF: 510300, 中证 500 ETF: 510500
        etf_map = {
            "000300": ["510300", "159919", "510330"],
            "000905": ["510500", "512500", "159922"],
        }
        
        etf_codes = etf_map.get(index_code, [])
        
        all_stocks = set()
        for etf_code in etf_codes:
            try:
                df = ak.fund_etf_em(symbol=etf_code, period="daily")
                if df is not None:
                    # 获取 ETF 持仓
                    holdings_df = ak.fund_etf_holdings_em(symbol=etf_code)
                    if holdings_df is not None:
                        holdings_pl = pl.from_pandas(holdings_df)
                        
                        for row in holdings_pl.iter_rows(named=True):
                            symbol = str(row.get("股票代码", "")).strip()
                            name = str(row.get("股票名称", "")).strip()
                            
                            if symbol and symbol != "nan":
                                all_stocks.add(_add_market_suffix(symbol))
                                
            except Exception:
                continue
        
        if all_stocks:
            result = [{"symbol": s, "name": ""} for s in all_stocks]
            logger.info(f"ETF-based approach returned {len(result)} stocks for {index_code}")
            return result
            
    except Exception as e:
        logger.warning(f"ETF-based approach failed for {index_code}: {e}")
    
    return []


def get_index_constituents(index_code: str) -> List[Dict[str, str]]:
    """
    获取指定指数成分股列表（多数据源冗余）。
    
    尝试顺序：
    1. 中证指数官网 (csindex)
    2. AKShare 其他接口
    3. ETF 成分股方式
    
    Args:
        index_code: 指数代码（如 000300, 000905）
        
    Returns:
        成分股列表
    """
    # 尝试多个数据源
    sources = [
        ("中证指数官网", lambda: fetch_constituents_from_csindex(index_code)),
        ("AKShare", lambda: fetch_constituents_from_akshare(index_code)),
        ("ETF 方式", lambda: fetch_constituents_from_etf(index_code)),
    ]
    
    for source_name, fetch_func in sources:
        try:
            result = fetch_func()
            if result and len(result) > 0:
                logger.success(f"Successfully fetched {len(result)} stocks from {source_name}")
                return result
        except Exception as e:
            logger.warning(f"{source_name} failed: {e}")
    
    logger.warning(f"All sources failed for index {index_code}")
    return []


def get_combined_constituents() -> List[Dict[str, str]]:
    """
    获取沪深 300 + 中证 500 合并后的成分股列表（去重）。
    
    Returns:
        合并后的成分股列表
    """
    logger.info("=" * 60)
    logger.info("Fetching index constituents...")
    logger.info("=" * 60)
    
    # 获取两个指数的成分股
    hs300_stocks = get_index_constituents("000300")
    zz500_stocks = get_index_constituents("000905")
    
    # 合并并去重
    seen: Set[str] = set()
    unique_stocks: List[Dict[str, str]] = []
    
    for stock in hs300_stocks + zz500_stocks:
        symbol = stock["symbol"]
        if symbol not in seen:
            seen.add(symbol)
            unique_stocks.append(stock)
    
    logger.info(f"Combined constituents: {len(unique_stocks)} unique stocks")
    logger.info(f"  - HS300 (000300): {len(hs300_stocks)} stocks")
    logger.info(f"  - ZZ500 (000905): {len(zz500_stocks)} stocks")
    logger.info(f"  - Overlap removed: {len(hs300_stocks) + len(zz500_stocks) - len(unique_stocks)} stocks")
    
    return unique_stocks


def get_constituents_with_fallback() -> List[Dict[str, str]]:
    """
    获取成分股列表，如果所有 API 失败则回退到内置列表。
    
    Returns:
        成分股列表
    """
    constituents = get_combined_constituents()
    
    if not constituents or len(constituents) < 50:
        logger.warning(f"API returned only {len(constituents) if constituents else 0} stocks, using fallback")
        logger.info(f"Fallback list contains {len(FALLBACK_STOCKS)} blue-chip stocks")
        return FALLBACK_STOCKS.copy()
    
    return constituents


def get_existing_dates_for_symbol(
    symbol: str,
    db: DatabaseManager,
    table_name: str = "stock_daily"
) -> Set[str]:
    """
    获取数据库中某只股票已存在的日期集合。
    """
    try:
        query = f"SELECT trade_date FROM {table_name} WHERE symbol = '{symbol}'"
        result = db.read_sql(query)
        
        if result.is_empty():
            return set()
        
        return set(str(d) for d in result["trade_date"].to_list())
    
    except Exception as e:
        logger.warning(f"Failed to get existing dates for {symbol}: {e}")
        return set()


def get_latest_date_for_symbol(
    symbol: str,
    db: DatabaseManager,
    table_name: str = "stock_daily"
) -> Optional[str]:
    """
    获取数据库中某只股票的最新日期（用于增量更新）。
    """
    try:
        query = f"""
            SELECT MAX(trade_date) as max_date 
            FROM {table_name} 
            WHERE symbol = '{symbol}'
        """
        result = db.read_sql(query)
        
        if result.is_empty() or result["max_date"][0] is None:
            return None
        
        return str(result["max_date"][0])
    
    except Exception as e:
        logger.warning(f"Failed to get latest date for {symbol}: {e}")
        return None


def fetch_stock_daily_with_retry(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[pl.DataFrame]:
    """
    获取单只股票的历史日线数据（带重试机制）。
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        max_retries: 最大重试次数
        
    Returns:
        DataFrame 或 None（如果失败）
    """
    import akshare as ak
    
    # AKShare 日期格式（YYYYMMDD）
    start_date_ak = start_date.replace("-", "")
    end_date_ak = end_date.replace("-", "")
    
    # 提取纯代码（去掉.SZ/.SH 后缀）
    pure_code = symbol.split(".")[0]
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # 随机延迟
            if attempt > 1:
                delay = random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX)
                time.sleep(delay)
            
            # 使用 ak.stock_zh_a_hist 获取 A 股历史数据
            df = ak.stock_zh_a_hist(
                symbol=pure_code,
                period="daily",
                start_date=start_date_ak,
                end_date=end_date_ak,
                adjust="qfq"  # 前复权
            )
            
            if df is None or len(df) == 0:
                logger.warning(f"No data for {symbol}")
                return None
            
            # 转换为 Polars
            df_pl = pl.from_pandas(df)
            
            # 列名映射
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
            
            for old, new in rename_map.items():
                if old in df_pl.columns:
                    df_pl = df_pl.rename({old: new})
            
            # 添加 symbol 列
            df_pl = df_pl.with_columns(pl.lit(symbol).alias("symbol"))
            
            # 日期格式标准化 (YYYYMMDD -> YYYY-MM-DD)
            if "trade_date" in df_pl.columns:
                def format_date(x):
                    """将日期转换为标准 YYYY-MM-DD 格式"""
                    if x is None:
                        return None
                    s = str(x).strip()
                    # 移除可能存在的横杠
                    s = s.replace('-', '')
                    # 确保是 8 位数字
                    if len(s) >= 8 and s[:8].isdigit():
                        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                    return s
                
                df_pl = df_pl.with_columns(
                    pl.col("trade_date")
                    .cast(pl.Utf8, strict=False)
                    .map_elements(
                        format_date,
                        return_dtype=pl.Utf8
                    )
                    .alias("trade_date")
                )
            
            # 计算 pre_close（如果不存在）
            if "pre_close" not in df_pl.columns and "close" in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col("close").shift(1).over("symbol").alias("pre_close")
                )
            
            # 计算 change（如果不存在）
            if "change" not in df_pl.columns and "close" in df_pl.columns and "pre_close" in df_pl.columns:
                df_pl = df_pl.with_columns(
                    (pl.col("close") - pl.col("pre_close")).alias("change")
                )
            
            # 选择目标列
            target_columns = [
                "symbol", "trade_date", "open", "high", "low", "close",
                "pre_close", "change", "pct_chg", "volume", "amount",
                "turnover_rate"
            ]
            
            available_columns = [c for c in target_columns if c in df_pl.columns]
            df_pl = df_pl.select(available_columns)
            
            # 数值列类型转换
            numeric_columns = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]
            for col in numeric_columns:
                if col in df_pl.columns:
                    df_pl = df_pl.with_columns(
                        pl.col(col).cast(pl.Float64, strict=False)
                    )
            
            # 排序
            df_pl = df_pl.sort(["symbol", "trade_date"])
            
            logger.debug(f"Fetched {len(df_pl)} rows for {symbol} (attempt {attempt})")
            return df_pl
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {symbol}: {e}")
            continue
    
    logger.error(f"All {max_retries} attempts failed for {symbol}")
    return None


def sync_stock_to_db(
    symbol: str,
    db: DatabaseManager,
    table_name: str = "stock_daily",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> int:
    """
    同步单只股票数据到数据库（增量更新 + 重试机制）。
    
    Returns:
        int: 同步的行数
    """
    # 增量更新：获取数据库中该股票的最新日期
    latest_db_date = get_latest_date_for_symbol(symbol, db, table_name)
    
    # 计算需要获取的日期范围
    today = datetime.now()
    end_date = today
    end_date_str = today.strftime("%Y-%m-%d")
    start_date_str: str
    
    if latest_db_date:
        # 增量更新：从最新日期之后开始
        try:
            latest_dt = datetime.strptime(latest_db_date, "%Y-%m-%d")
            start_dt = latest_dt + timedelta(days=1)
            
            # 如果最新日期已经很近（7 天内），则跳过
            days_diff = (today - latest_dt).days
            if days_diff <= 7:
                logger.debug(f"{symbol}: Data is recent ({days_diff} days ago), skipping")
                return 0
            
            # 但也要确保有足够的数据（至少 lookback_days 天）
            if days_diff < lookback_days:
                start_date_str = start_dt.strftime("%Y-%m-%d")
            else:
                # 如果数据太少，重新获取全部
                start_dt = end_date - timedelta(days=lookback_days)
                start_date_str = start_dt.strftime("%Y-%m-%d")
        except ValueError:
            start_dt = end_date - timedelta(days=lookback_days)
            start_date_str = start_dt.strftime("%Y-%m-%d")
    else:
        # 新股票：获取全部历史数据
        start_dt = end_date - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")
    
    # 获取数据
    df = fetch_stock_daily_with_retry(symbol, start_date_str, end_date_str)
    
    if df is None or df.is_empty():
        return 0
    
    # 增量更新：过滤掉已存在的日期
    if latest_db_date:
        existing_dates = get_existing_dates_for_symbol(symbol, db, table_name)
        if existing_dates:
            original_count = len(df)
            df = df.filter(~pl.col("trade_date").is_in(existing_dates))
            if len(df) < original_count:
                logger.debug(f"{symbol}: Skipped {original_count - len(df)} existing rows")
    
    if df.is_empty():
        logger.debug(f"{symbol}: No new data to sync")
        return 0
    
    try:
        # 插入新数据
        rows = db.to_sql(df, table_name, if_exists="append")
        logger.debug(f"Synced {rows} rows for {symbol}")
        return rows
        
    except Exception as e:
        logger.error(f"Failed to write {symbol} to database: {e}")
        return 0


def sync_all_stocks(
    constituents: List[Dict[str, str]] = None,
    db: DatabaseManager = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, int]:
    """
    同步所有成分股数据到数据库。
    
    Args:
        constituents: 成分股列表
        db: 数据库管理器
        lookback_days: 回溯天数
        
    Returns:
        dict: 每只股票同步的行数
    """
    if constituents is None:
        constituents = get_constituents_with_fallback()
    
    if not constituents:
        logger.warning("No constituents to sync, using fallback")
        constituents = FALLBACK_STOCKS.copy()
    
    if db is None:
        db = DatabaseManager.get_instance()
    
    results: Dict[str, int] = {}
    total_rows = 0
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    logger.info("=" * 60)
    logger.info("ALL STOCKS DATA SYNC - Starting")
    logger.info(f"Constituents: {len(constituents)} stocks")
    logger.info(f"Lookback days: {lookback_days} ({lookback_days // 365} years)")
    logger.info(f"Commit interval: every {COMMIT_INTERVAL} stocks")
    logger.info(f"Max retries per stock: {MAX_RETRIES}")
    logger.info("=" * 60)
    
    # 创建表（如果不存在）
    create_stock_daily_table(db)
    
    # 进度条显示
    total = len(constituents)
    
    for i, stock in enumerate(constituents, 1):
        symbol = stock["symbol"]
        name = stock.get("name", "Unknown")
        
        # 进度显示
        progress = i / total * 100
        bar_length = 40
        filled = int(bar_length * i / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        logger.info(f"[{i}/{total}] [{bar}] Syncing {symbol} ({name})...")
        
        try:
            rows = sync_stock_to_db(symbol, db, lookback_days=lookback_days)
            results[symbol] = rows
            total_rows += rows
            
            if rows > 0:
                success_count += 1
                logger.info(f"  ✅ Synced {rows} rows")
            else:
                skipped_count += 1
                logger.debug(f"  ⏭ Skipped (no new data)")
                
        except Exception as e:
            logger.error(f"  ❌ Failed to sync {symbol}: {e}")
            results[symbol] = 0
            fail_count += 1
        
        # 每 50 只股票 commit 一次
        if i % COMMIT_INTERVAL == 0:
            logger.info(f"📌 Checkpoint: {i}/{total} stocks synced, committing...")
            try:
                db.execute("COMMIT")
                logger.info(f"✅ Commit successful after {i} stocks")
            except Exception as e:
                logger.warning(f"Commit failed: {e}, will retry later")
        
        # 随机延迟（防止被封 IP）
        if i < total:  # 最后一只股票不需要延迟
            delay = random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX)
            time.sleep(delay)
    
    # 最终 commit
    try:
        db.execute("COMMIT")
        logger.info("✅ Final commit successful")
    except Exception as e:
        logger.warning(f"Final commit failed: {e}")
    
    # 统计输出
    logger.info("=" * 60)
    logger.info("SYNC COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total stocks: {total}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Skipped (no new data): {skipped_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Total rows synced: {total_rows}")
    logger.info("=" * 60)
    
    return results


def create_stock_daily_table(db: DatabaseManager = None) -> None:
    """
    创建 stock_daily 表。
    
    【V8 增强 - 因子纯化】
    表结构对齐新规范:
    - symbol (varchar(20)): 股票代码
    - trade_date (DATE): 交易日期
    - 价格字段 (decimal(18,4))
    - 成交量/额 (double)
    - industry_code (varchar(20)): 申万一级行业分类【V8 新增】
    - total_mv (double): 总市值（亿元）【V8 新增】
    - is_st (tinyint): 是否 ST 股票【V8 新增】
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
      `industry_code` varchar(20) DEFAULT NULL COMMENT '申万一级行业代码【V8 新增】',
      `total_mv` double DEFAULT NULL COMMENT '总市值 (亿元)【V8 新增】',
      `is_st` tinyint DEFAULT 0 COMMENT '是否 ST 股票 (1=ST, 0=非 ST)【V8 新增】',
      PRIMARY KEY (`symbol`,`trade_date`),
      KEY `idx_date` (`trade_date`),
      KEY `idx_industry` (`industry_code`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """
    
    try:
        db.execute(create_sql)
        logger.info("stock_daily table ready (aligned with V8 schema)")
    except Exception as e:
        logger.error(f"Failed to create stock_daily table: {e}")
        raise


def add_v8_columns_to_table(db: DatabaseManager = None) -> None:
    """
    【V8 增强】为现有表添加新字段（如果不存在）。
    
    用于增量升级现有数据库表结构。
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    columns_to_add = [
        ("industry_code", "varchar(20) DEFAULT NULL COMMENT '申万一级行业代码'"),
        ("total_mv", "double DEFAULT NULL COMMENT '总市值 (亿元)'"),
        ("is_st", "tinyint DEFAULT 0 COMMENT '是否 ST 股票'"),
    ]
    
    for col_name, col_def in columns_to_add:
        try:
            # 检查列是否已存在
            check_query = f"""
                SELECT COUNT(*) as cnt 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'stock_daily' 
                AND COLUMN_NAME = '{col_name}'
            """
            result = db.read_sql(check_query)
            
            if result.is_empty() or result["cnt"][0] == 0:
                # 列不存在，添加
                alter_sql = f"ALTER TABLE `stock_daily` ADD COLUMN {col_name} {col_def}"
                db.execute(alter_sql)
                logger.info(f"Added column '{col_name}' to stock_daily table")
            else:
                logger.debug(f"Column '{col_name}' already exists")
        except Exception as e:
            logger.warning(f"Failed to add column '{col_name}': {e}")


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行同步
    sync_all_stocks(lookback_days=DEFAULT_LOOKBACK_DAYS)


if __name__ == "__main__":
    main()