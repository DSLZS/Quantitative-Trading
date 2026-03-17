"""
Stock Metadata Sync Module - 同步股票元数据（行业分类、市值、ST 状态）。

核心功能:
    - 从 AKShare 获取 A 股上市公司的行业分类（申万一级）
    - 获取总市值数据
    - 获取 ST 状态
    - 更新 stock_daily 表的 industry_code, total_mv, is_st 字段

使用示例:
    >>> python src/sync_stock_metadata.py
"""

import sys
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from pathlib import Path

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量
# ===========================================
COMMIT_INTERVAL = 100  # 每 100 只股票 commit 一次
MAX_RETRIES = 3  # 最大重试次数
RANDOM_DELAY_MIN = 0.3  # 随机延迟最小值 (秒)
RANDOM_DELAY_MAX = 0.8  # 随机延迟最大值 (秒)


def add_missing_columns(db: DatabaseManager) -> None:
    """
    为 stock_daily 表添加缺失的字段。
    
    Args:
        db: 数据库管理器实例
    """
    columns_to_add = [
        ("industry_code", "varchar(50) DEFAULT NULL COMMENT '申万一级行业代码'"),
        ("total_mv", "DOUBLE DEFAULT NULL COMMENT '总市值 (亿元)'"),
        ("is_st", "TINYINT DEFAULT 0 COMMENT '是否 ST 股票 (1=ST, 0=非 ST)'"),
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
                alter_sql = f"ALTER TABLE `stock_daily` ADD COLUMN `{col_name}` {col_def}"
                db.execute(alter_sql)
                logger.success(f"✅ 添加列 '{col_name}' 成功")
            else:
                logger.info(f"列 '{col_name}' 已存在")
        except Exception as e:
            logger.error(f"❌ 添加列 '{col_name}' 失败：{e}")
            raise


def get_all_symbols_from_db(db: DatabaseManager) -> List[str]:
    """
    获取数据库中所有唯一的股票代码。
    
    Args:
        db: 数据库管理器实例
        
    Returns:
        股票代码列表
    """
    query = "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol"
    result = db.read_sql(query)
    
    if result.is_empty():
        return []
    
    return result["symbol"].to_list()


def fetch_stock_info_from_tushare(symbol: str, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """
    从 Tushare 获取单只股票的基本信息（行业、市值、ST 状态）。
    
    Args:
        symbol: 股票代码（如 600519.SH）
        max_retries: 最大重试次数
        
    Returns:
        包含行业、市值、ST 状态的字典，失败返回 None
    """
    import tushare as ts
    
    # 提取纯代码（去掉.SZ/.SH 后缀）
    pure_code = symbol.split(".")[0]
    
    # 确定 ts_code 格式
    if symbol.endswith(".SH"):
        ts_code = f"{pure_code}.SH"
    elif symbol.endswith(".SZ"):
        ts_code = f"{pure_code}.SZ"
    else:
        # 根据代码前缀判断
        if pure_code.startswith("6"):
            ts_code = f"{pure_code}.SH"
        else:
            ts_code = f"{pure_code}.SZ"
    
    for attempt in range(1, max_retries + 1):
        try:
            # 随机延迟
            if attempt > 1:
                delay = random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX)
                time.sleep(delay)
            
            # 方法 1: 使用 tushare 的 stock_basic 获取基本信息
            try:
                pro = ts.pro_api()
                
                # 获取股票基本信息
                basic_df = pro.stock_basic(
                    exchange='' if not ts_code else ts_code.split('.')[1],
                    list_status='L',
                    fields='ts_code,symbol,name,area,industry,market,list_date,act_shr,total_mv'
                )
                
                if basic_df is not None and len(basic_df) > 0:
                    basic_pl = pl.from_pandas(basic_df)
                    
                    # 查找匹配的股票
                    matched = basic_pl.filter(
                        (pl.col("ts_code") == ts_code) |
                        (pl.col("symbol") == pure_code)
                    )
                    
                    if len(matched) > 0:
                        row = matched.row(0, named=True)
                        
                        # 提取行业信息（申万行业）
                        industry = row.get("industry", "")
                        
                        # 提取总市值（单位：亿元）
                        total_mv = row.get("total_mv")
                        try:
                            total_mv_value = float(total_mv) if total_mv else None
                        except (ValueError, TypeError):
                            total_mv_value = None
                        
                        # 检查是否 ST（名称中包含 ST）
                        name = row.get("name", "")
                        is_st = 1 if "ST" in str(name) else 0
                        
                        return {
                            "industry_code": industry if industry else None,
                            "total_mv": total_mv_value,
                            "is_st": is_st,
                        }
                        
            except Exception as e:
                logger.debug(f"Tushare method 1 failed for {symbol}: {e}")
            
            # 方法 2: 使用 tushare 的 daily 接口获取市值数据
            try:
                pro = ts.pro_api()
                
                # 获取最新交易日的市值数据
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                
                mv_df = pro.daily_basic(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,total_mv'
                )
                
                if mv_df is not None and len(mv_df) > 0:
                    mv_pl = pl.from_pandas(mv_df)
                    # 获取最新市值
                    latest_mv = mv_pl.sort("trade_date", descending=True)["total_mv"][0]
                    try:
                        total_mv_value = float(latest_mv) / 100000000  # 转换为亿元
                    except (ValueError, TypeError):
                        total_mv_value = None
                    
                    return {
                        "industry_code": None,
                        "total_mv": total_mv_value,
                        "is_st": 0,
                    }
                    
            except Exception as e:
                logger.debug(f"Tushare method 2 failed for {symbol}: {e}")
            
            # 所有方法都失败
            return None
            
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {symbol}: {e}")
            continue
    
    return None


def fetch_stock_info_from_akshare(symbol: str, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """
    从 AKShare 获取单只股票的基本信息（行业、市值、ST 状态）。
    作为 Tushare 的备用方案。
    
    Args:
        symbol: 股票代码（如 600519.SH）
        max_retries: 最大重试次数
        
    Returns:
        包含行业、市值、ST 状态的字典，失败返回 None
    """
    import akshare as ak
    
    # 提取纯代码（去掉.SZ/.SH 后缀）
    pure_code = symbol.split(".")[0]
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # 随机延迟
            if attempt > 1:
                delay = random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX)
                time.sleep(delay)
            
            # 方法 1: 使用 ak.stock_individual_info_em 获取个股信息
            try:
                info_df = ak.stock_individual_info_em(symbol=pure_code)
                
                if info_df is not None and len(info_df) > 0:
                    info_pl = pl.from_pandas(info_df)
                    
                    # 转换为字典方便查找
                    info_dict = dict(zip(info_pl["item"].to_list(), info_pl["value"].to_list()))
                    
                    # 提取行业信息
                    industry = info_dict.get("行业", "")
                    
                    # 提取市值信息（总市值）
                    market_cap = info_dict.get("总市值", "")
                    try:
                        # 解析市值（可能是 "1.23 万亿" 或 "456.78 亿" 格式）
                        if market_cap:
                            market_cap_str = str(market_cap)
                            if "万亿" in market_cap_str:
                                market_cap_value = float(market_cap_str.replace("万亿", "").replace(",", "")) * 10000
                            elif "亿" in market_cap_str:
                                market_cap_value = float(market_cap_str.replace("亿", "").replace(",", ""))
                            elif "万" in market_cap_str:
                                market_cap_value = float(market_cap_str.replace("万", "").replace(",", "")) / 10000
                            else:
                                market_cap_value = float(market_cap_str.replace(",", ""))
                        else:
                            market_cap_value = None
                    except (ValueError, TypeError):
                        market_cap_value = None
                    
                    # 检查是否 ST
                    stock_name = info_dict.get("股票简称", "")
                    is_st = 1 if "ST" in str(stock_name) else 0
                    
                    return {
                        "industry_code": industry if industry else None,
                        "total_mv": market_cap_value,
                        "is_st": is_st,
                    }
                    
            except Exception as e:
                logger.debug(f"Method 1 failed for {symbol}: {e}")
            
            # 所有方法都失败
            return None
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {symbol}: {e}")
            continue
    
    if last_error:
        logger.error(f"All {max_retries} attempts failed for {symbol}")
    
    return None


def fetch_market_cap_from_sina(symbol: str) -> Optional[float]:
    """
    从新浪财经获取总市值。
    
    Args:
        symbol: 股票代码
        
    Returns:
        总市值（亿元），失败返回 None
    """
    try:
        import akshare as ak
        
        # 使用 ak.stock_financial_analysis_indicator 获取财务指标
        df = ak.stock_financial_analysis_indicator(symbol=symbol.split(".")[0])
        
        if df is not None and len(df) > 0:
            df_pl = pl.from_pandas(df)
            
            # 查找市值相关列
            for col in df_pl.columns:
                if "总市值" in col or "市值" in col:
                    # 获取最新值
                    value = df_pl[col][0]
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        continue
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to fetch market cap from sina for {symbol}: {e}")
        return None


def update_stock_metadata(
    symbol: str,
    industry_code: Optional[str],
    total_mv: Optional[float],
    is_st: int,
    db: DatabaseManager
) -> int:
    """
    更新单只股票的元数据到所有历史交易日期。
    
    Args:
        symbol: 股票代码
        industry_code: 行业代码
        total_mv: 总市值
        is_st: 是否 ST
        db: 数据库管理器
        
    Returns:
        更新的行数
    """
    try:
        # 构建 UPDATE 语句
        set_parts = []
        
        if industry_code is not None:
            set_parts.append(f"`industry_code` = '{industry_code}'")
        else:
            set_parts.append("`industry_code` = NULL")
        
        if total_mv is not None:
            set_parts.append(f"`total_mv` = {total_mv}")
        else:
            set_parts.append("`total_mv` = NULL")
        
        set_parts.append(f"`is_st` = {is_st}")
        
        update_sql = f"""
            UPDATE `stock_daily`
            SET {', '.join(set_parts)}
            WHERE `symbol` = '{symbol}'
        """
        
        rows = db.execute(update_sql)
        return rows
        
    except Exception as e:
        logger.error(f"Failed to update metadata for {symbol}: {e}")
        return 0


def sync_all_stock_metadata(
    symbols: List[str] = None,
    db: DatabaseManager = None,
) -> Dict[str, int]:
    """
    同步所有股票的元数据。
    
    Args:
        symbols: 股票代码列表，如果为 None 则从数据库获取
        db: 数据库管理器
        
    Returns:
        每只股票更新的行数
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    if symbols is None:
        symbols = get_all_symbols_from_db(db)
    
    if not symbols:
        logger.warning("No symbols to sync")
        return {}
    
    results: Dict[str, int] = {}
    total_rows = 0
    success_count = 0
    fail_count = 0
    no_data_count = 0
    
    logger.info("=" * 60)
    logger.info("STOCK METADATA SYNC - 同步股票元数据")
    logger.info(f"Total symbols: {len(symbols)}")
    logger.info("=" * 60)
    
    # 首先添加缺失的列
    logger.info("Step 1: Adding missing columns...")
    add_missing_columns(db)
    
    # 然后更新数据
    logger.info("Step 2: Fetching and updating metadata...")
    
    for i, symbol in enumerate(symbols, 1):
        # 进度显示
        progress = i / len(symbols) * 100
        
        logger.info(f"[{i}/{len(symbols)}] ({progress:.1f}%) Syncing {symbol}...")
        
        try:
            # 获取股票信息
            stock_info = fetch_stock_info_from_akshare(symbol)
            
            if stock_info is None:
                logger.warning(f"  ⚠️ 无法获取 {symbol} 的基本信息，使用默认值")
                stock_info = {
                    "industry_code": None,
                    "total_mv": None,
                    "is_st": 0,
                }
                no_data_count += 1
            
            # 更新数据库
            rows = update_stock_metadata(
                symbol=symbol,
                industry_code=stock_info["industry_code"],
                total_mv=stock_info["total_mv"],
                is_st=stock_info["is_st"],
                db=db
            )
            
            results[symbol] = rows
            total_rows += rows
            
            if rows > 0:
                industry = stock_info["industry_code"] or "N/A"
                mv = f"{stock_info['total_mv']:.2f}" if stock_info["total_mv"] else "N/A"
                st_status = "ST" if stock_info["is_st"] else "Normal"
                logger.info(f"  ✅ Updated {rows} rows | Industry: {industry} | MV: {mv}亿 | Status: {st_status}")
                success_count += 1
            else:
                logger.debug(f"  ⏭ No rows updated")
                
        except Exception as e:
            logger.error(f"  ❌ Failed to sync {symbol}: {e}")
            results[symbol] = 0
            fail_count += 1
        
        # 每 100 只股票 commit 一次
        if i % COMMIT_INTERVAL == 0:
            logger.info(f"📌 Checkpoint: {i}/{len(symbols)} stocks synced, committing...")
            try:
                db.execute("COMMIT")
                logger.info(f"✅ Commit successful after {i} stocks")
            except Exception as e:
                logger.warning(f"Commit failed: {e}")
        
        # 随机延迟（防止被封 IP）
        if i < len(symbols):
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
    logger.info(f"Total stocks: {len(symbols)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"No data (used default): {no_data_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Total rows updated: {total_rows}")
    logger.info("=" * 60)
    
    return results


def batch_update_industry_from_sector_data(db: DatabaseManager = None) -> int:
    """
    批量更新行业数据（使用 AKShare 的行业成分股数据）。
    
    这是一种更高效的方法，通过获取每个行业的成分股列表，
    然后批量更新这些股票的行业分类。
    
    Args:
        db: 数据库管理器
        
    Returns:
        更新的行数
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    import akshare as ak
    
    logger.info("=" * 60)
    logger.info("BATCH INDUSTRY UPDATE - 批量更新行业数据")
    logger.info("=" * 60)
    
    total_updated = 0
    
    try:
        # 获取申万一级行业列表
        sector_df = ak.stock_board_industry_name_em()
        
        if sector_df is None or len(sector_df) == 0:
            logger.warning("Failed to fetch sector list")
            return 0
        
        sector_pl = pl.from_pandas(sector_df)
        sectors = sector_pl["板块名称"].to_list() if "板块名称" in sector_pl.columns else []
        
        logger.info(f"Found {len(sectors)} sectors")
        
        for i, sector_name in enumerate(sectors, 1):
            logger.info(f"[{i}/{len(sectors)}] Processing sector: {sector_name}")
            
            try:
                # 获取该行业的成分股
                constituents_df = ak.stock_board_industry_cons_em(symbol=sector_name)
                
                if constituents_df is None or len(constituents_df) == 0:
                    continue
                
                constituents_pl = pl.from_pandas(constituents_df)
                
                # 提取股票代码列
                symbol_col = None
                for col in constituents_pl.columns:
                    if "代码" in col or "code" in col.lower():
                        symbol_col = col
                        break
                
                if symbol_col is None:
                    continue
                
                # 为每只股票添加市场后缀
                symbols = []
                for code in constituents_pl[symbol_col].cast(pl.Utf8, strict=False).to_list():
                    code = str(code).strip()
                    if code:
                        # 添加市场后缀
                        if code.startswith("6"):
                            symbols.append(f"{code}.SH")
                        elif code.startswith("0") or code.startswith("3"):
                            symbols.append(f"{code}.SZ")
                        else:
                            symbols.append(code)
                
                if not symbols:
                    continue
                
                # 批量更新
                symbols_str = "','".join(symbols)
                update_sql = f"""
                    UPDATE `stock_daily`
                    SET `industry_code` = '{sector_name}'
                    WHERE `symbol` IN ('{symbols_str}')
                """
                
                rows = db.execute(update_sql)
                total_updated += rows
                
                logger.info(f"  ✅ Updated {rows} rows for sector: {sector_name}")
                
                # 延迟
                time.sleep(random.uniform(0.3, 0.6))
                
            except Exception as e:
                logger.warning(f"Failed to process sector {sector_name}: {e}")
                continue
        
        logger.info("=" * 60)
        logger.info(f"BATCH UPDATE COMPLETE - Total rows updated: {total_updated}")
        logger.info("=" * 60)
        
        return total_updated
        
    except Exception as e:
        logger.error(f"Batch update failed: {e}")
        return 0


def verify_metadata_completeness(db: DatabaseManager = None) -> Dict:
    """
    验证元数据完整性。
    
    Args:
        db: 数据库管理器
        
    Returns:
        验证结果字典
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    logger.info("=" * 60)
    logger.info("METADATA COMPLETENESS CHECK")
    logger.info("=" * 60)
    
    # 检查每个字段的缺失率
    checks = {}
    
    # industry_code 和 total_mv: NULL 或空字符串视为缺失
    for field in ["industry_code", "total_mv"]:
        query = f"""
            SELECT 
                COUNT(*) as total_count,
                SUM(CASE WHEN `{field}` IS NULL OR `{field}` = '' THEN 1 ELSE 0 END) as missing_count
            FROM stock_daily
        """
        result = db.read_sql(query)
        
        if result.is_empty():
            checks[field] = {"total": 0, "missing": 0, "rate": 100.0}
            continue
        
        total = result["total_count"][0]
        missing = result["missing_count"][0]
        rate = (missing / total * 100) if total > 0 else 100.0
        
        checks[field] = {
            "total": int(total),
            "missing": int(missing),
            "rate": round(rate, 2)
        }
        
        status = "✅" if rate < 10 else "⚠️" if rate < 50 else "❌"
        logger.info(f"{status} {field}: {rate:.2f}% missing ({missing}/{total})")
    
    # is_st: 0 是有效值，只有 NULL 视为缺失
    query = """
        SELECT 
            COUNT(*) as total_count,
            SUM(CASE WHEN `is_st` IS NULL THEN 1 ELSE 0 END) as missing_count
        FROM stock_daily
    """
    result = db.read_sql(query)
    
    if result.is_empty():
        checks["is_st"] = {"total": 0, "missing": 0, "rate": 100.0}
    else:
        total = result["total_count"][0]
        missing = result["missing_count"][0]
        rate = (missing / total * 100) if total > 0 else 100.0
        
        checks["is_st"] = {
            "total": int(total),
            "missing": int(missing),
            "rate": round(rate, 2)
        }
        
        status = "✅" if rate < 10 else "⚠️" if rate < 50 else "❌"
        logger.info(f"{status} is_st: {rate:.2f}% missing ({missing}/{total})")
    
    return checks


def update_is_st_from_name(db: DatabaseManager = None) -> int:
    """
    从股票名称更新 is_st 字段。
    
    通过分析股票名称中的 ST 标记来更新 is_st 字段。
    
    Args:
        db: 数据库管理器
        
    Returns:
        更新的行数
    """
    if db is None:
        db = DatabaseManager.get_instance()
    
    logger.info("=" * 60)
    logger.info("UPDATE IS_ST FROM STOCK NAME - 从股票名称更新 ST 状态")
    logger.info("=" * 60)
    
    try:
        # 获取所有唯一的股票名称
        query = """
            SELECT DISTINCT 
                SUBSTRING_INDEX(SUBSTRING_INDEX(symbol, '.', 1), '.', -1) as stock_code,
                symbol
            FROM stock_daily
            WHERE trade_date = (SELECT MAX(trade_date) FROM stock_daily)
        """
        result = db.read_sql(query)
        
        if result.is_empty():
            logger.warning("No stocks found")
            return 0
        
        total_updated = 0
        
        for row in result.iter_rows(named=True):
            stock_code = str(row.get("stock_code", ""))
            symbol = row.get("symbol", "")
            
            # 确定 ts_code 格式
            if symbol.endswith(".SH"):
                ts_code = f"{stock_code}.SH"
            elif symbol.endswith(".SZ"):
                ts_code = f"{stock_code}.SZ"
            else:
                ts_code = symbol
            
            try:
                import tushare as ts
                pro = ts.pro_api()
                
                # 获取股票基本信息
                basic_df = pro.stock_basic(
                    ts_code=ts_code,
                    fields='ts_code,name'
                )
                
                if basic_df is not None and len(basic_df) > 0:
                    name = str(basic_df.iloc[0]["name"])
                    is_st = 1 if "ST" in name else 0
                    
                    # 更新数据库
                    update_sql = f"""
                        UPDATE stock_daily
                        SET is_st = {is_st}
                        WHERE symbol = '{symbol}'
                    """
                    rows = db.execute(update_sql)
                    total_updated += rows
                    
                    if is_st:
                        logger.info(f"  ✅ {ts_code}: {name} -> ST")
                    
            except Exception as e:
                logger.debug(f"Failed to fetch info for {ts_code}: {e}")
                continue
        
        logger.info(f"BATCH UPDATE COMPLETE - Total rows updated: {total_updated}")
        return total_updated
        
    except Exception as e:
        logger.error(f"Update failed: {e}")
        return 0


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    db = DatabaseManager.get_instance()
    
    # 步骤 1: 先添加缺失的列
    logger.info("Step 1: Adding missing columns...")
    add_missing_columns(db)
    
    # 步骤 2: 尝试从 Tushare 获取行业数据
    logger.info("Step 2: Trying to fetch industry data from Tushare...")
    try:
        # 获取前 50 只股票进行测试
        symbols = get_all_symbols_from_db(db)[:50]
        if symbols:
            sync_all_stock_metadata(symbols=symbols, db=db)
    except Exception as e:
        logger.warning(f"Tushare sync failed: {e}")
    
    # 步骤 3: 使用 AKShare 批量更新行业数据
    logger.info("Step 3: Trying batch update from AKShare...")
    try:
        batch_update_industry_from_sector_data(db)
    except Exception as e:
        logger.warning(f"Batch update failed: {e}")
    
    # 步骤 4: 验证完整性
    logger.info("Step 4: Verifying metadata completeness...")
    verify_metadata_completeness(db)
    
    # 输出建议
    logger.info("=" * 60)
    logger.info("NOTES")
    logger.info("=" * 60)
    logger.info("如果 industry_code 和 total_mv 仍然缺失，可以考虑:")
    logger.info("  1. 使用 Tushare Pro API (需要 token)")
    logger.info("  2. 在因子计算时使用替代数据源")
    logger.info("  3. 使用行业中性化时跳过行业字段")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()