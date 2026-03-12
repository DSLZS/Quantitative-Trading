"""
All Stocks Data Sync Module - 同步沪深 300 成分股历史日线数据。

核心功能:
    - 获取沪深 300 指数成分股列表
    - 同步每只成分股最近 1 年的历史日线数据
    - 增量更新 stock_daily 表，避免重复插入
    - 支持断点续传和错误重试

使用示例:
    >>> python src/sync_all_stocks.py
    # 同步所有沪深 300 成分股最近 1 年数据
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine


# 沪深 300 成分股获取方式：
# 1. 使用 AKShare 的 index_stock_cons_csindex 接口
# 2. 或使用本地配置文件
def get_hs300_constituents() -> list[dict[str, str]]:
    """
    获取沪深 300 指数成分股列表。
    
    Returns:
        list[dict]: 成分股列表，每项包含 {"symbol": "600519.SH", "name": "贵州茅台"}
    """
    try:
        import akshare as ak
        
        # 使用中证指数官网接口获取沪深 300 成分股
        logger.info("Fetching CSI 300 constituents from AKShare...")
        
        # 获取沪深 300 成分股（000300 是沪深 300 的指数代码）
        # 接口返回：symbol, name, market
        try:
            # 方法 1: 使用 index_stock_cons_csindex（最稳定）
            constituents_df = ak.index_stock_cons_csindex(symbol="000300")
        except Exception as e1:
            logger.warning(f"First API failed: {e1}, trying alternative...")
            try:
                # 方法 2: 使用 index_stock_cons
                constituents_df = ak.index_stock_cons(symbol="000300")
            except Exception as e2:
                logger.error(f"Alternative API also failed: {e2}")
                # 降级：使用备用成分股列表
                return _get_fallback_constituents()
        
        if constituents_df is None or len(constituents_df) == 0:
            logger.warning("Empty constituents list, using fallback")
            return _get_fallback_constituents()
        
        # 转换为 Polars DataFrame
        df_pl = pl.from_pandas(constituents_df)
        
        # 列名映射（根据实际接口返回调整）
        rename_map = {}
        if "品种代码" in df_pl.columns:
            rename_map["品种代码"] = "symbol"
        if "品种名称" in df_pl.columns:
            rename_map["品种名称"] = "name"
        if "market" in df_pl.columns and "symbol" in df_pl.columns:
            # 需要添加市场后缀
            pass
        
        for old, new in rename_map.items():
            if old in df_pl.columns:
                df_pl = df_pl.rename({old: new})
        
        # 确保有 symbol 和 name 列
        result = []
        for row in df_pl.iter_rows(named=True):
            symbol = row.get("symbol", "")
            name = row.get("name", "")
            
            # 添加市场后缀（如果还没有）
            if symbol and "." not in symbol:
                if symbol.startswith("6"):
                    symbol = f"{symbol}.SH"
                elif symbol.startswith("0") or symbol.startswith("3"):
                    symbol = f"{symbol}.SZ"
            
            if symbol:
                result.append({"symbol": symbol, "name": name})
        
        logger.info(f"Fetched {len(result)} constituents from CSI 300")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch constituents: {e}")
        return _get_fallback_constituents()


def _get_fallback_constituents() -> list[dict[str, str]]:
    """
    降级方案：返回备用成分股列表（主要蓝筹股）。
    
    Returns:
        list[dict]: 备用成分股列表
    """
    fallback_stocks = [
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
        {"symbol": "600000.SH", "name": "浦发银行"},
        {"symbol": "600016.SH", "name": "民生银行"},
        {"symbol": "600030.SH", "name": "中信证券"},
        {"symbol": "600048.SH", "name": "保利发展"},
        {"symbol": "600104.SH", "name": "上汽集团"},
        {"symbol": "600276.SH", "name": "恒瑞医药"},
        {"symbol": "600346.SH", "name": "恒力石化"},
        {"symbol": "600519.SH", "name": "贵州茅台"},
        {"symbol": "600585.SH", "name": "海螺水泥"},
        {"symbol": "600588.SH", "name": "用友网络"},
        {"symbol": "600690.SH", "name": "海尔智家"},
        {"symbol": "600745.SH", "name": "闻泰科技"},
        {"symbol": "600809.SH", "name": "山西汾酒"},
        {"symbol": "600887.SH", "name": "伊利股份"},
        {"symbol": "601012.SH", "name": "隆基股份"},
        {"symbol": "601066.SH", "name": "中信建投"},
        {"symbol": "601088.SH", "name": "中国神华"},
        {"symbol": "601138.SH", "name": "工业富联"},
        {"symbol": "601211.SH", "name": "国泰君安"},
        {"symbol": "601229.SH", "name": "上海银行"},
        {"symbol": "601288.SH", "name": "农业银行"},
        {"symbol": "601318.SH", "name": "中国平安"},
        {"symbol": "601328.SH", "name": "交通银行"},
        {"symbol": "601336.SH", "name": "新华保险"},
        {"symbol": "601390.SH", "name": "中国中铁"},
        {"symbol": "601398.SH", "name": "工商银行"},
        {"symbol": "601601.SH", "name": "中国太保"},
        {"symbol": "601628.SH", "name": "中国人寿"},
        {"symbol": "601633.SH", "name": "长城汽车"},
        {"symbol": "601668.SH", "name": "中国建筑"},
        {"symbol": "601688.SH", "name": "华泰证券"},
        {"symbol": "601728.SH", "name": "中国电信"},
        {"symbol": "601766.SH", "name": "中国中车"},
        {"symbol": "601800.SH", "name": "中国交建"},
        {"symbol": "601816.SH", "name": "京沪高铁"},
        {"symbol": "601818.SH", "name": "光大银行"},
        {"symbol": "601857.SH", "name": "中国石油"},
        {"symbol": "601878.SH", "name": "浙商证券"},
        {"symbol": "601881.SH", "name": "中国银河"},
        {"symbol": "601888.SH", "name": "中国中免"},
        {"symbol": "601898.SH", "name": "中煤能源"},
        {"symbol": "601899.SH", "name": "紫金矿业"},
        {"symbol": "601919.SH", "name": "中远海控"},
        {"symbol": "601939.SH", "name": "建设银行"},
        {"symbol": "601985.SH", "name": "中国核电"},
        {"symbol": "601988.SH", "name": "中国银行"},
        {"symbol": "601995.SH", "name": "中金公司"},
        {"symbol": "601998.SH", "name": "中信银行"},
        {"symbol": "603259.SH", "name": "药明康德"},
        {"symbol": "603288.SH", "name": "海天味业"},
        {"symbol": "603501.SH", "name": "韦尔股份"},
        {"symbol": "603799.SH", "name": "华友钴业"},
        {"symbol": "603986.SH", "name": "兆易创新"},
        {"symbol": "000001.SZ", "name": "平安银行"},
        {"symbol": "000002.SZ", "name": "万科 A"},
        {"symbol": "000063.SZ", "name": "中兴通讯"},
        {"symbol": "000100.SZ", "name": "TCL 科技"},
        {"symbol": "000157.SZ", "name": "中联重科"},
        {"symbol": "000333.SZ", "name": "美的集团"},
        {"symbol": "000538.SZ", "name": "云南白药"},
        {"symbol": "000568.SZ", "name": "泸州老窖"},
        {"symbol": "000596.SZ", "name": "古井贡酒"},
        {"symbol": "000625.SZ", "name": "长安汽车"},
        {"symbol": "000651.SZ", "name": "格力电器"},
        {"symbol": "000661.SZ", "name": "长春高新"},
        {"symbol": "000725.SZ", "name": "京东方 A"},
        {"symbol": "000776.SZ", "name": "广发证券"},
        {"symbol": "000858.SZ", "name": "五粮液"},
        {"symbol": "000895.SZ", "name": "双汇发展"},
        {"symbol": "002001.SZ", "name": "新和成"},
        {"symbol": "002007.SZ", "name": "华兰生物"},
        {"symbol": "002027.SZ", "name": "分众传媒"},
        {"symbol": "002049.SZ", "name": "紫光国微"},
        {"symbol": "002129.SZ", "name": "TCL 中环"},
        {"symbol": "002142.SZ", "name": "宁波银行"},
        {"symbol": "002179.SZ", "name": "中航光电"},
        {"symbol": "002230.SZ", "name": "科大讯飞"},
        {"symbol": "002236.SZ", "name": "大华股份"},
        {"symbol": "002241.SZ", "name": "歌尔股份"},
        {"symbol": "002252.SZ", "name": "上海莱士"},
        {"symbol": "002304.SZ", "name": "洋河股份"},
        {"symbol": "002311.SZ", "name": "海大集团"},
        {"symbol": "002352.SZ", "name": "顺丰控股"},
        {"symbol": "002371.SZ", "name": "北方华创"},
        {"symbol": "002410.SZ", "name": "广联达"},
        {"symbol": "002415.SZ", "name": "海康威视"},
        {"symbol": "002422.SZ", "name": "科伦药业"},
        {"symbol": "002459.SZ", "name": "晶澳科技"},
        {"symbol": "002460.SZ", "name": "赣锋锂业"},
        {"symbol": "002466.SZ", "name": "天齐锂业"},
        {"symbol": "002475.SZ", "name": "立讯精密"},
        {"symbol": "002493.SZ", "name": "荣盛石化"},
        {"symbol": "002507.SZ", "name": "涪陵榨菜"},
        {"symbol": "002555.SZ", "name": "三七互娱"},
        {"symbol": "002568.SZ", "name": "百润股份"},
        {"symbol": "002594.SZ", "name": "比亚迪"},
        {"symbol": "002601.SZ", "name": "龙佰集团"},
        {"symbol": "002648.SZ", "name": "卫星化学"},
        {"symbol": "002709.SZ", "name": "天赐材料"},
        {"symbol": "002714.SZ", "name": "牧原股份"},
        {"symbol": "002812.SZ", "name": "恩捷股份"},
        {"symbol": "002821.SZ", "name": "凯莱英"},
        {"symbol": "002850.SZ", "name": "科达利"},
        {"symbol": "002916.SZ", "name": "深南电路"},
        {"symbol": "002920.SZ", "name": "德赛西威"},
        {"symbol": "002938.SZ", "name": "鹏鼎控股"},
        {"symbol": "003816.SZ", "name": "中国广核"},
        {"symbol": "300012.SZ", "name": "华测检测"},
        {"symbol": "300014.SZ", "name": "亿纬锂能"},
        {"symbol": "300033.SZ", "name": "同花顺"},
        {"symbol": "300059.SZ", "name": "东方财富"},
        {"symbol": "300122.SZ", "name": "智飞生物"},
        {"symbol": "300124.SZ", "name": "汇川技术"},
        {"symbol": "300142.SZ", "name": "沃森生物"},
        {"symbol": "300274.SZ", "name": "阳光电源"},
        {"symbol": "300316.SZ", "name": "晶盛机电"},
        {"symbol": "300347.SZ", "name": "泰格医药"},
        {"symbol": "300408.SZ", "name": "三环集团"},
        {"symbol": "300413.SZ", "name": "芒果超媒"},
        {"symbol": "300433.SZ", "name": "蓝思科技"},
        {"symbol": "300450.SZ", "name": "先导智能"},
        {"symbol": "300454.SZ", "name": "深信服"},
        {"symbol": "300496.SZ", "name": "中科创达"},
        {"symbol": "300498.SZ", "name": "温氏股份"},
        {"symbol": "300601.SZ", "name": "康泰生物"},
        {"symbol": "300628.SZ", "name": "亿联网络"},
        {"symbol": "300724.SZ", "name": "捷佳伟创"},
        {"symbol": "300725.SZ", "name": "药石科技"},
        {"symbol": "300750.SZ", "name": "宁德时代"},
        {"symbol": "300759.SZ", "name": "康龙化成"},
        {"symbol": "300760.SZ", "name": "迈瑞医疗"},
        {"symbol": "300763.SZ", "name": "锦浪科技"},
        {"symbol": "300782.SZ", "name": "卓胜微"},
        {"symbol": "300896.SZ", "name": "爱美客"},
        {"symbol": "300957.SZ", "name": "贝泰妮"},
        {"symbol": "300979.SZ", "name": "华利集团"},
        {"symbol": "300999.SZ", "name": "金龙鱼"},
    ]
    
    # 去重
    seen = set()
    unique_stocks = []
    for stock in fallback_stocks:
        if stock["symbol"] not in seen:
            seen.add(stock["symbol"])
            unique_stocks.append(stock)
    
    logger.info(f"Using fallback constituents list: {len(unique_stocks)} stocks")
    return unique_stocks


def fetch_stock_daily(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 365,
) -> pl.DataFrame:
    """
    获取单只股票的历史日线数据。
    
    Args:
        symbol: 股票代码（如 600519.SH）
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        lookback_days: 回溯天数，默认 365 天（1 年）
        
    Returns:
        pl.DataFrame: 包含股票日线数据的 DataFrame
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed")
        raise
    
    logger.debug(f"Fetching data for {symbol}...")
    
    # 日期参数处理
    today = datetime.now()
    
    if end_date is None:
        end_dt = today
        end_date = today.strftime("%Y-%m-%d")
    else:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            end_dt = today
            end_date = today.strftime("%Y-%m-%d")
    
    if start_date is None:
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
    
    # AKShare 日期格式（YYYYMMDD）
    start_date_ak = start_date.replace("-", "")
    end_date_ak = end_date.replace("-", "")
    
    # 提取纯代码（去掉.SZ/.SH 后缀）
    pure_code = symbol.split(".")[0]
    
    try:
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
            return pl.DataFrame()
        
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
        
        # 日期格式标准化
        if "trade_date" in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col("trade_date")
                .cast(pl.Utf8)
                .map_elements(
                    lambda x: f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}" if len(str(x)) >= 8 else str(x),
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
        
        logger.debug(f"Fetched {len(df_pl)} rows for {symbol}")
        return df_pl
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return pl.DataFrame()


def sync_stock_to_db(
    symbol: str,
    db: DatabaseManager,
    table_name: str = "stock_daily",
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 365,
) -> int:
    """
    同步单只股票数据到数据库（增量更新）。
    
    Args:
        symbol: 股票代码
        db: 数据库管理器
        table_name: 表名
        start_date: 开始日期
        end_date: 结束日期
        lookback_days: 回溯天数
        
    Returns:
        int: 同步的行数
    """
    # 获取数据
    df = fetch_stock_daily(symbol, start_date, end_date, lookback_days)
    
    if df.is_empty():
        return 0
    
    try:
        # 增量更新：先删除已有数据
        dates = df["trade_date"].unique().to_list()
        if dates:
            dates_str = "', '".join([str(d) for d in dates])
            
            delete_query = f"""
                DELETE FROM {table_name}
                WHERE symbol = '{symbol}'
                AND trade_date IN ('{dates_str}')
            """
            
            db.execute(delete_query)
        
        # 插入新数据
        rows = db.to_sql(df, table_name, if_exists="append")
        logger.info(f"Synced {rows} rows for {symbol}")
        return rows
        
    except Exception as e:
        logger.error(f"Failed to write {symbol} to database: {e}")
        return 0


def sync_all_stocks(
    constituents: list[dict[str, str]] = None,
    db: DatabaseManager = None,
    lookback_days: int = 365,
    max_workers: int = 4,
) -> dict[str, int]:
    """
    同步所有成分股数据到数据库。
    
    Args:
        constituents: 成分股列表
        db: 数据库管理器
        lookback_days: 回溯天数，默认 365 天
        max_workers: 最大并发数
        
    Returns:
        dict[str, int]: 每只股票同步的行数
    """
    if constituents is None:
        constituents = get_hs300_constituents()
    
    if db is None:
        db = DatabaseManager.get_instance()
    
    results = {}
    total_rows = 0
    success_count = 0
    fail_count = 0
    
    logger.info("=" * 60)
    logger.info("ALL STOCKS DATA SYNC - Starting")
    logger.info(f"Constituents: {len(constituents)} stocks")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info("=" * 60)
    
    # 创建表（如果不存在）
    create_stock_daily_table(db)
    
    for i, stock in enumerate(constituents, 1):
        symbol = stock["symbol"]
        name = stock.get("name", "Unknown")
        
        logger.info(f"[{i}/{len(constituents)}] Syncing {symbol} ({name})...")
        
        try:
            rows = sync_stock_to_db(symbol, db, lookback_days=lookback_days)
            results[symbol] = rows
            total_rows += rows
            if rows > 0:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Failed to sync {symbol}: {e}")
            results[symbol] = 0
            fail_count += 1
    
    logger.info("=" * 60)
    logger.info("SYNC COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Success: {success_count}, Failed: {fail_count}")
    logger.info(f"Total rows synced: {total_rows}")
    
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
        db: 数据库管理器
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """
    
    try:
        db.execute(create_sql)
        logger.info("stock_daily table created successfully (aligned with new schema)")
    except Exception as e:
        logger.error(f"Failed to create stock_daily table: {e}")
        raise


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
    sync_all_stocks(lookback_days=365)


if __name__ == "__main__":
    main()