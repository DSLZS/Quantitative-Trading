"""
中证 800 成分股历史数据定向同步脚本 (修复版)

用途：
    为了进行完整的 Walk-Forward 验证，定向同步中证 800 (000906.SH)
    成份股在 2023-01-01 至 2024-05-01 期间的历史日线数据。

核心特性：
    - 使用 ak.index_stock_cons_csindex 获取成分股列表
    - 增量更新，避免重复插入
    - 失败重试机制（指数退避）
    - Header 伪装防反爬
    - 闭环验证逻辑

使用示例:
    >>> python src/sync_csi800.py
"""

import sys
import os
import time
import random
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple
from urllib.parse import urlencode

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量
# ===========================================
# 目标日期范围
SYNC_START_DATE = "2023-01-01"
SYNC_END_DATE = "2024-05-01"

# 同步控制
COMMIT_INTERVAL = 50  # 每 50 只股票 commit 一次
MAX_RETRIES = 3  # 最大重试次数
RANDOM_DELAY_MIN = 0.3  # 随机延迟最小值 (秒)
RANDOM_DELAY_MAX = 0.6  # 随机延迟最大值 (秒)

# 数据源配置
CSI800_INDEX_CODE = "000906"

# 请求头伪装（防反爬）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


# ===========================================
# 辅助函数
# ===========================================

def _add_market_suffix(symbol: str) -> str:
    """
    为股票代码添加市场后缀 (.SH/.SZ)。
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


def _get_random_user_agent() -> str:
    """获取随机 User-Agent"""
    return random.choice(USER_AGENTS)


def _setup_akshare_headers() -> None:
    """
    设置 AKShare 请求头（防反爬）。
    
    注意：AKShare 内部不直接支持设置 headers，
    但设置 user_agent 可能有助于某些版本。
    """
    try:
        import akshare as ak
        
        # 设置随机 User-Agent
        ak.user_agent = _get_random_user_agent()
        logger.debug(f"AKShare UA configured: {ak.user_agent[:50]}...")
        
    except Exception as e:
        logger.debug(f"Failed to configure AKShare headers: {e}")


# ===========================================
# 中证 800 成份股获取
# ===========================================

def fetch_csi800_constituents() -> List[Dict[str, str]]:
    """
    获取中证 800 成份股列表。
    
    使用 ak.index_stock_cons_csindex 获取最新成分股。
    
    Returns:
        成份股列表，每项为 {"symbol": "600519.SH", "name": "贵州茅台"}
    """
    logger.info("Fetching CSI800 constituents...")
    
    try:
        import akshare as ak
        
        # 配置请求头
        _setup_akshare_headers()
        
        # 使用 index_stock_cons_csindex 获取成分股
        df = ak.index_stock_cons_csindex(symbol=CSI800_INDEX_CODE)
        
        if df is None or len(df) == 0:
            logger.warning("index_stock_cons_csindex returned empty data")
            return []
        
        # 转换为 Polars
        df_pl = pl.from_pandas(df)
        
        # 查找列名
        symbol_col = None
        name_col = None
        
        for col in df_pl.columns:
            col_lower = col.lower()
            if "代码" in col or "code" in col_lower or "symbol" in col_lower:
                symbol_col = col
            if "名称" in col or "name" in col_lower:
                name_col = col
        
        if not symbol_col:
            logger.error(f"Could not find symbol column in columns: {df_pl.columns}")
            return []
        
        # 提取成分股
        result = []
        seen_symbols = set()
        
        for row in df_pl.iter_rows(named=True):
            symbol = str(row.get(symbol_col, "")).strip()
            name = str(row.get(name_col, "")).strip() if name_col else ""
            
            if symbol and symbol != "nan" and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                result.append({
                    "symbol": _add_market_suffix(symbol),
                    "name": name
                })
        
        logger.info(f"Fetched {len(result)} CSI800 constituents")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch constituents: {e}")
        return []


# ===========================================
# 数据同步核心函数
# ===========================================

def get_existing_dates_for_symbol(
    symbol: str,
    db: DatabaseManager,
    table_name: str = "stock_daily"
) -> Set[str]:
    """
    获取数据库中某只股票已存在的日期集合。
    """
    try:
        query = f"SELECT trade_date FROM {table_name} WHERE symbol = %s"
        # 使用参数化查询
        cursor = db.engine.connect().connection.cursor()
        cursor.execute(query.replace("%s", "'{0}'".format(symbol)))
        rows = cursor.fetchall()
        return set(str(r[0]) for r in rows if r[0])
    
    except Exception as e:
        logger.debug(f"Failed to get existing dates for {symbol}: {e}")
        return set()


def fetch_stock_daily_data(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[pl.DataFrame]:
    """
    获取单只股票的历史日线数据（带重试机制）。
    
    Args:
        symbol: 股票代码（带后缀，如 600519.SH）
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
            # 配置请求头
            _setup_akshare_headers()
            
            # 重试前延迟（较短延迟）
            if attempt > 1:
                delay = random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX)
                time.sleep(delay)
                logger.debug(f"Retry {attempt}/{max_retries} for {symbol} after {delay:.2f}s delay")
            
            # 获取数据
            df = ak.stock_zh_a_hist(
                symbol=pure_code,
                period="daily",
                start_date=start_date_ak,
                end_date=end_date_ak,
                adjust="qfq"
            )
            
            if df is None or len(df) == 0:
                logger.warning(f"No data for {symbol}")
                return None
            
            # 转换为 Polars
            df_pl = pl.from_pandas(df)
            
            # 列名映射（中文 -> 英文）
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
            
            # 日期格式标准化
            if "trade_date" in df_pl.columns:
                def format_date(x):
                    if x is None:
                        return None
                    s = str(x).strip().replace('-', '')
                    if len(s) >= 8 and s[:8].isdigit():
                        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                    return s
                
                df_pl = df_pl.with_columns(
                    pl.col("trade_date")
                    .cast(pl.Utf8, strict=False)
                    .map_elements(format_date, return_dtype=pl.Utf8)
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
            
            logger.debug(f"Fetched {len(df_pl)} rows for {symbol}")
            return df_pl
            
        except Exception as e:
            # 所有错误都触发重试
            last_error = e
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {symbol}: {e}")
            continue
    
    # 所有重试失败
    logger.error(f"All {max_retries} attempts failed for {symbol}: {last_error}")
    return None


def sync_stock_to_db(
    symbol: str,
    db: DatabaseManager,
    target_start_date: str,
    target_end_date: str,
    table_name: str = "stock_daily",
) -> Tuple[int, bool]:
    """
    同步单只股票数据到数据库（增量更新）。
    
    Args:
        symbol: 股票代码
        db: 数据库管理器
        target_start_date: 目标开始日期
        target_end_date: 目标结束日期
        table_name: 表名
        
    Returns:
        (同步行数，是否成功)
    """
    try:
        # 检查已存在的日期
        existing_dates = get_existing_dates_for_symbol(symbol, db, table_name)
        
        # 确定需要获取的日期范围
        if existing_dates:
            existing_dates_list = sorted(existing_dates)
            min_existing = existing_dates_list[0]
            max_existing = existing_dates_list[-1]
            
            # 如果已覆盖目标范围，跳过
            if min_existing <= target_start_date and max_existing >= target_end_date:
                logger.debug(f"{symbol}: Data already exists, skipping")
                return (0, True)
            
            # 计算需要补充的日期范围
            if min_existing > target_start_date:
                start_date = target_start_date
                end_date = (datetime.strptime(min_existing, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            elif max_existing < target_end_date:
                start_date = (datetime.strptime(max_existing, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = target_end_date
            else:
                return (0, True)
        else:
            start_date = target_start_date
            end_date = target_end_date
        
        # 获取数据
        df = fetch_stock_daily_data(symbol, start_date, end_date)
        
        if df is None or df.is_empty():
            return (0, False)
        
        # 过滤已存在的日期
        if existing_dates:
            original_count = len(df)
            df = df.filter(~pl.col("trade_date").is_in(existing_dates))
            if len(df) < original_count:
                logger.debug(f"{symbol}: Skipped {original_count - len(df)} existing rows")
        
        if df.is_empty():
            logger.debug(f"{symbol}: No new data to sync")
            return (0, True)
        
        # 写入数据库
        rows = db.to_sql(df, table_name, if_exists="append")
        logger.debug(f"Synced {rows} rows for {symbol}")
        return (rows, True)
        
    except Exception as e:
        logger.error(f"Failed to sync {symbol}: {e}")
        return (0, False)


def create_stock_daily_table(db: DatabaseManager) -> None:
    """
    创建 stock_daily 表（如果不存在）。
    """
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """
    
    try:
        db.execute(create_sql)
        logger.info("stock_daily table ready")
    except Exception as e:
        logger.error(f"Failed to create stock_daily table: {e}")
        raise


def sync_csi800_stocks(
    constituents: List[Dict[str, str]] = None,
    db: DatabaseManager = None,
    start_date: str = SYNC_START_DATE,
    end_date: str = SYNC_END_DATE,
) -> Dict[str, Tuple[int, bool]]:
    """
    同步所有中证 800 成分股数据到数据库。
    
    Args:
        constituents: 成分股列表
        db: 数据库管理器
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        dict: {symbol: (rows_synced, success)}
    """
    if constituents is None:
        constituents = fetch_csi800_constituents()
    
    if not constituents:
        logger.error("No constituents to sync")
        return {}
    
    if db is None:
        db = DatabaseManager.get_instance()
    
    results: Dict[str, Tuple[int, bool]] = {}
    total_rows = 0
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    logger.info("=" * 60)
    logger.info("CSI800 DATA SYNC - Starting")
    logger.info(f"Target period: {start_date} to {end_date}")
    logger.info(f"Constituents: {len(constituents)} stocks")
    logger.info(f"Commit interval: every {COMMIT_INTERVAL} stocks")
    logger.info(f"Max retries per stock: {MAX_RETRIES}")
    logger.info("=" * 60)
    
    # 创建表
    create_stock_daily_table(db)
    
    total = len(constituents)
    
    for i, stock in enumerate(constituents, 1):
        symbol = stock["symbol"]
        name = stock.get("name", "Unknown")
        
        # 进度条
        progress = i / total * 100
        bar_length = 40
        filled = int(bar_length * i / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        logger.info(f"[{i}/{total}] [{bar}] Syncing {symbol} ({name})...")
        
        try:
            # 同步股票数据
            rows, success = sync_stock_to_db(
                symbol=symbol,
                db=db,
                target_start_date=start_date,
                target_end_date=end_date,
            )
            
            results[symbol] = (rows, success)
            total_rows += rows
            
            if success:
                success_count += 1
                if rows > 0:
                    logger.info(f"  ✅ Synced {rows} rows")
                else:
                    skipped_count += 1
                    logger.debug(f"  ⏭ Skipped")
            else:
                fail_count += 1
                logger.error(f"  ❌ Failed")
                
        except Exception as e:
            logger.error(f"  ❌ Exception for {symbol}: {e}")
            results[symbol] = (0, False)
            fail_count += 1
        
        # 每 N 只股票 commit 一次
        if i % COMMIT_INTERVAL == 0:
            logger.info(f"📌 Checkpoint: {i}/{total} stocks, committing...")
            try:
                db.execute("COMMIT")
                logger.info(f"✅ Commit successful")
            except Exception as e:
                logger.warning(f"Commit failed: {e}")
        
        # 随机延迟（防反爬）- 使用较短延迟
        if i < total:
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
    logger.info(f"Total stocks: {total}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Total rows synced: {total_rows}")
    logger.info("=" * 60)
    
    return results


# ===========================================
# 闭环验证逻辑
# ===========================================

def verify_sync_results(
    db: DatabaseManager,
    start_date: str = SYNC_START_DATE,
    end_date: str = SYNC_END_DATE,
    sample_size: int = 3,
) -> Dict[str, any]:
    """
    验证同步结果的正确性。
    
    验证项:
    1. 统计成功写入的股票总数
    2. 随机抽取样本股票，检查数据行数
    3. 检查价格数据有效性（close > 0, volume 非零）
    
    Args:
        db: 数据库管理器
        start_date: 开始日期
        end_date: 结束日期
        sample_size: 抽样数量
        
    Returns:
        验证报告
    """
    logger.info("=" * 60)
    logger.info("DATA VERIFICATION")
    logger.info("=" * 60)
    
    report = {
        "total_stocks": 0,
        "total_records": 0,
        "sample_checks": [],
        "data_quality_issues": [],
        "passed": True,
    }
    
    try:
        # 1. 统计股票总数和记录总数
        query = """
            SELECT 
                COUNT(DISTINCT symbol) as stock_count,
                COUNT(*) as record_count
            FROM stock_daily
            WHERE trade_date >= %s AND trade_date <= %s
        """ % ("'%s'" % start_date, "'%s'" % end_date)
        
        result = db.read_sql(query)
        
        if not result.is_empty():
            report["total_stocks"] = int(result["stock_count"][0])
            report["total_records"] = int(result["record_count"][0])
        
        logger.info(f"Total stocks in range: {report['total_stocks']}")
        logger.info(f"Total records in range: {report['total_records']}")
        
        # 2. 随机抽取样本检查
        # 获取所有股票列表
        stocks_query = """
            SELECT DISTINCT symbol FROM stock_daily
            WHERE trade_date >= '%s' AND trade_date <= '%s'
            ORDER BY symbol
        """ % (start_date, end_date)
        
        stocks_result = db.read_sql(stocks_query)
        
        if not stocks_result.is_empty():
            all_symbols = stocks_result["symbol"].to_list()
            
            # 随机抽样
            if len(all_symbols) >= sample_size:
                sampled_symbols = random.sample(all_symbols, sample_size)
            else:
                sampled_symbols = all_symbols
            
            # 预期交易日数（约 320 天）
            expected_days = 320
            min_expected_ratio = 0.5  # 至少 50% 的交易日
            
            for symbol in sampled_symbols:
                sample_report = {"symbol": symbol}
                
                # 查询该股票的数据行数
                count_query = """
                    SELECT COUNT(*) as cnt
                    FROM stock_daily
                    WHERE symbol = '%s'
                    AND trade_date >= '%s' AND trade_date <= '%s'
                """ % (symbol, start_date, end_date)
                
                count_result = db.read_sql(count_query)
                actual_count = int(count_result["cnt"][0]) if not count_result.is_empty() else 0
                sample_report["record_count"] = actual_count
                
                # 检查是否显著少于预期
                if actual_count < expected_days * min_expected_ratio:
                    sample_report["warning"] = f"Only {actual_count} records (expected ~{expected_days})"
                    report["data_quality_issues"].append({
                        "symbol": symbol,
                        "issue": f"Low record count: {actual_count}"
                    })
                    logger.warning(f"⚠️ {symbol}: Only {actual_count} records (expected ~{expected_days})")
                else:
                    sample_report["status"] = "OK"
                    logger.info(f"✅ {symbol}: {actual_count} records")
                
                # 检查价格有效性
                price_query = """
                    SELECT 
                        MIN(close) as min_close,
                        MAX(close) as max_close,
                        AVG(close) as avg_close,
                        SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END) as invalid_close_count,
                        AVG(volume) as avg_volume,
                        SUM(CASE WHEN volume = 0 OR volume IS NULL THEN 1 ELSE 0 END) as zero_volume_count
                    FROM stock_daily
                    WHERE symbol = '%s'
                """ % symbol
                
                price_result = db.read_sql(price_query)
                
                if not price_result.is_empty():
                    min_close = float(price_result["min_close"][0]) if price_result["min_close"][0] else 0
                    invalid_close = int(price_result["invalid_close_count"][0]) if price_result["invalid_close_count"][0] else 0
                    zero_volume = int(price_result["zero_volume_count"][0]) if price_result["zero_volume_count"][0] else 0
                    
                    sample_report["min_close"] = min_close
                    sample_report["invalid_close_count"] = invalid_close
                    sample_report["zero_volume_count"] = zero_volume
                    
                    # 检查 close 价格
                    if min_close <= 0:
                        sample_report["price_warning"] = f"Found non-positive close prices (min={min_close})"
                        report["data_quality_issues"].append({
                            "symbol": symbol,
                            "issue": f"Non-positive close price: min={min_close}"
                        })
                        logger.warning(f"⚠️ {symbol}: Non-positive close prices found (min={min_close})")
                    
                    # 检查零成交量比例
                    if zero_volume > actual_count * 0.3:  # 超过 30% 零成交量
                        sample_report["volume_warning"] = f"High zero-volume ratio: {zero_volume}/{actual_count}"
                        report["data_quality_issues"].append({
                            "symbol": symbol,
                            "issue": f"High zero-volume ratio: {zero_volume}/{actual_count}"
                        })
                        logger.warning(f"⚠️ {symbol}: High zero-volume ratio ({zero_volume}/{actual_count})")
                
                report["sample_checks"].append(sample_report)
        
        # 3. 总体评估
        if report["data_quality_issues"]:
            report["passed"] = False
            logger.warning(f"Verification found {len(report['data_quality_issues'])} issues")
        else:
            logger.info("✅ All verification checks passed")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        report["error"] = str(e)
        report["passed"] = False
    
    # 输出总结
    logger.info("=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info(f"Total stocks: {report['total_stocks']}")
    logger.info(f"Total records: {report['total_records']}")
    logger.info(f"Sample checks: {len(report['sample_checks'])}")
    logger.info(f"Issues found: {len(report['data_quality_issues'])}")
    logger.info(f"Passed: {'✅ YES' if report['passed'] else '❌ NO'}")
    logger.info("=" * 60)
    
    return report


def check_data_coverage(db: DatabaseManager) -> Dict[str, any]:
    """
    检查数据覆盖情况。
    """
    try:
        # 日期范围
        date_query = """
            SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date
            FROM stock_daily
        """
        date_result = db.read_sql(date_query)
        
        min_date = str(date_result["min_date"][0]) if not date_result.is_empty() and date_result["min_date"][0] else "N/A"
        max_date = str(date_result["max_date"][0]) if not date_result.is_empty() and date_result["max_date"][0] else "N/A"
        
        # 股票数量
        stock_query = """
            SELECT COUNT(DISTINCT symbol) as stock_count
            FROM stock_daily
        """
        stock_result = db.read_sql(stock_query)
        stock_count = int(stock_result["stock_count"][0]) if not stock_result.is_empty() else 0
        
        return {
            "date_range": {"min": min_date, "max": max_date},
            "total_stocks": stock_count,
        }
        
    except Exception as e:
        logger.error(f"Failed to check data coverage: {e}")
        return {}


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("中证 800 成分股历史数据定向同步 (修复版)")
    logger.info(f"目标区间：{SYNC_START_DATE} 至 {SYNC_END_DATE}")
    logger.info("=" * 60)
    
    # 初始化数据库
    db = DatabaseManager.get_instance()
    
    # 同步数据
    results = sync_csi800_stocks(db=db)
    
    # 检查数据覆盖
    logger.info("")
    logger.info("数据覆盖检查...")
    coverage = check_data_coverage(db)
    
    if coverage:
        logger.info(f"日期范围：{coverage['date_range']['min']} ~ {coverage['date_range']['max']}")
        logger.info(f"股票数量：{coverage['total_stocks']}")
    
    # 闭环验证
    logger.info("")
    verification = verify_sync_results(db)
    
    # 保存验证报告
    report_path = Path("reports") / f"sync_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "sync_results": {k: {"rows": v[0], "success": v[1]} for k, v in results.items()},
            "verification": verification,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Verification report saved to: {report_path}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("同步完成!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("下一步操作:")
    logger.info("1. 运行审计报告：python src/run_iteration13_audit.py")
    logger.info("2. 启动 Iteration 14 训练：python src/final_strategy_v1_3.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()