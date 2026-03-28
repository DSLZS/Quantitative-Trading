"""
V82 数据补齐模块 - 强制补齐 2019/2021/2024 年数据

【V82 数据要求】
1. 强制测试年份：2019、2021、2024
2. 数据表：stock_daily（股票日线数据）
3. 每只股票必须包含：open, high, low, close, volume, amount, pct_chg

作者：量化系统
版本：V82.0
日期：2026-03-26
"""

import sys
import os
import time
import random
import traceback
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import polars as pl
from loguru import logger

# 尝试导入 akshare
try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False
    logger.error("V82: akshare 库未安装，请运行：pip install akshare")

# 尝试导入数据库管理器
try:
    from src.db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    try:
        from db_manager import DatabaseManager, get_db
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False
        logger.error("V82: db_manager 模块未找到")


# ===========================================
# V82 配置常量
# ===========================================

# V82 强制测试年份
V82_REQUIRED_YEARS = ['2019', '2021', '2024']

# 数据表配置
V82_STOCK_DAILY_TABLE = "stock_daily"

# 随机延迟配置
V82_RANDOM_DELAY_MIN = 1.0
V82_RANDOM_DELAY_MAX = 3.0

# 写入批次大小
V82_WRITE_BATCH_SIZE = 5000


# ===========================================
# V82 数据获取器
# ===========================================

class V82DataFetcher:
    """V82 数据获取器 - 使用 AkShare 获取历史数据"""
    
    def __init__(self):
        self._last_error = ""
    
    def _random_delay(self):
        """随机延迟"""
        delay = random.uniform(V82_RANDOM_DELAY_MIN, V82_RANDOM_DELAY_MAX)
        time.sleep(delay)
    
    def fetch_stock_daily(self, symbol: str, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """
        获取个股日线数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        start_date : str
            开始日期，格式：YYYYMMDD
        end_date : str
            结束日期，格式：YYYYMMDD
            
        Returns
        -------
        Optional[pl.DataFrame]
            日线数据
        """
        if not AK_AVAILABLE:
            raise ImportError("V82: akshare 库未安装")
        
        try:
            # 使用 ak.stock_zh_a_hist 获取历史数据
            self._random_delay()
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df is None or df.empty:
                return None
            
            # 转换为 Polars DataFrame
            df = pl.from_pandas(df)
            
            # 标准化列名
            columns_mapping = {
                '日期': 'trade_date',
                '收盘': 'close',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
                '换手率': 'turnover',
            }
            
            # 重命名存在的列
            available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
            if available_cols:
                df = df.rename(available_cols)
            
            # 添加股票代码
            df = df.with_columns(pl.lit(symbol).alias('symbol'))
            
            # 格式化日期
            if 'trade_date' in df.columns:
                df = df.with_columns(pl.col('trade_date').cast(pl.Utf8).alias('trade_date'))
            
            return df
            
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"V82: 获取 {symbol} 数据失败 - {e}")
            return None


# ===========================================
# V82 数据填充器
# ===========================================

class V82DataForceFiller:
    """V82 数据填充器 - 强制补齐指定年份数据"""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V82: 数据库连接失败：{e}")
                self.db = None
        else:
            self.db = db
        
        self.fetcher = V82DataFetcher()
        self.total_inserted_rows = 0
    
    def create_table(self):
        """建表校验"""
        if self.db is None:
            logger.error("V82: 数据库连接未初始化，无法建表")
            return
        
        logger.info("V82: 开始建表校验...")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS `stock_daily` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `symbol` VARCHAR(20) NOT NULL,
            `trade_date` VARCHAR(20) NOT NULL,
            `open` DECIMAL(20, 4) DEFAULT 0,
            `high` DECIMAL(20, 4) DEFAULT 0,
            `low` DECIMAL(20, 4) DEFAULT 0,
            `close` DECIMAL(20, 4) DEFAULT 0,
            `volume` DECIMAL(20, 2) DEFAULT 0,
            `amount` DECIMAL(20, 2) DEFAULT 0,
            `pct_chg` DECIMAL(10, 4) DEFAULT 0,
            `industry_code` VARCHAR(20) DEFAULT '',
            `total_mv` DECIMAL(20, 2) DEFAULT 0,
            `is_st` TINYINT DEFAULT 0,
            INDEX `idx_symbol` (`symbol`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_symbol_date` (`symbol`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票日线数据表'
        """
        
        try:
            self.db.execute(create_table_sql)
            logger.info("V82: stock_daily 表创建成功")
        except Exception as e:
            logger.error(f"V82: 创建 stock_daily 表失败：{e}")
            raise
    
    def _get_table_count(self, table_name: str) -> int:
        """获取表行数"""
        if self.db is None:
            return 0
        
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return 0
            
            return int(result['cnt'][0])
            
        except Exception as e:
            logger.error(f"V82: 查询表 {table_name} 行数失败：{e}")
            return 0
    
    def _get_existing_symbols(self, year: str) -> set:
        """获取指定年份已存在的股票代码"""
        if self.db is None:
            return set()
        
        try:
            query = f"""
            SELECT DISTINCT symbol 
            FROM stock_daily 
            WHERE trade_date >= '{year}-01-01' AND trade_date <= '{year}-12-31'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return set()
            
            return set(result['symbol'].to_list())
            
        except Exception as e:
            logger.warning(f"V82: 查询 {year} 年已存在股票失败：{e}")
            return set()
    
    def _get_stock_list(self) -> List[str]:
        """获取 A 股股票列表"""
        if not AK_AVAILABLE:
            raise ImportError("V82: akshare 库未安装")
        
        try:
            # 获取 A 股股票列表
            df = ak.stock_info_a_code_name()
            
            if df is None or df.empty:
                return []
            
            # 转换为 Polars
            df = pl.from_pandas(df)
            
            # 获取股票代码
            symbols = df['code'].to_list() if 'code' in df.columns else []
            
            logger.info(f"V82: 获取到 {len(symbols)} 只 A 股股票")
            return symbols
            
        except Exception as e:
            logger.error(f"V82: 获取股票列表失败：{e}")
            return []
    
    def _write_to_db(self, df: pl.DataFrame) -> int:
        """写入数据到数据库"""
        if self.db is None:
            logger.error("V82: 数据库连接未初始化")
            return 0
        
        try:
            rows = len(df)
            
            # 分批写入
            for start_idx in range(0, rows, V82_WRITE_BATCH_SIZE):
                end_idx = min(start_idx + V82_WRITE_BATCH_SIZE, rows)
                batch_df = df.slice(start_idx, end_idx - start_idx)
                
                self.db.to_sql(batch_df, V82_STOCK_DAILY_TABLE, if_exists='append')
            
            return rows
            
        except Exception as e:
            logger.error(f"V82: 写入数据失败：{e}")
            return 0
    
    def fill_year_data(self, year: str, limit_symbols: Optional[int] = None) -> Dict[str, int]:
        """
        填充指定年份数据
        
        Parameters
        ----------
        year : str
            年份
        limit_symbols : Optional[int]
            限制股票数量（用于测试）
            
        Returns
        -------
        Dict[str, int]
            填充结果统计
        """
        logger.info("=" * 60)
        logger.info(f"V82: 开始填充 {year} 年数据")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V82: 数据库连接未初始化")
            return {'success_symbols': 0, 'failure_symbols': 0, 'total_rows': 0}
        
        # 建表校验
        self.create_table()
        
        # 检查已存在数据
        existing_symbols = self._get_existing_symbols(year)
        logger.info(f"V82: {year} 年已存在 {len(existing_symbols)} 只股票数据")
        
        # 获取股票列表
        all_symbols = self._get_stock_list()
        
        # 过滤掉已存在的股票
        new_symbols = [s for s in all_symbols if s not in existing_symbols]
        
        if limit_symbols:
            new_symbols = new_symbols[:limit_symbols]
        
        logger.info(f"V82: 需要抓取 {len(new_symbols)} 只新股票")
        
        if not new_symbols:
            logger.info(f"V82: {year} 年数据已完整，无需抓取")
            return {'success_symbols': 0, 'failure_symbols': 0, 'total_rows': 0}
        
        # 日期范围
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        result = {'success_symbols': 0, 'failure_symbols': 0, 'total_rows': 0}
        
        # 抓取数据
        for i, symbol in enumerate(new_symbols):
            try:
                logger.info(f"[PROGRESS] {year} [{i+1}/{len(new_symbols)}]: 抓取 {symbol}...")
                
                df = self.fetcher.fetch_stock_daily(symbol, start_date, end_date)
                
                if df is not None and not df.is_empty():
                    rows = self._write_to_db(df)
                    
                    if rows > 0:
                        result['success_symbols'] += 1
                        result['total_rows'] += rows
                        logger.info(f"[PROGRESS] {year} {symbol}: Inserted {rows} rows.")
                    else:
                        result['failure_symbols'] += 1
                else:
                    logger.warning(f"V82: {symbol} 无数据")
                    result['failure_symbols'] += 1
                    
            except Exception as e:
                logger.error(f"V82: {symbol} 处理失败：{e}")
                result['failure_symbols'] += 1
        
        # 打印最终统计
        final_count = self._get_table_count(V82_STOCK_DAILY_TABLE)
        
        logger.info("=" * 60)
        logger.info(f"V82: {year} 年数据抓取完成")
        logger.info(f"[FINAL COUNT] stock_daily: {final_count:,} rows")
        logger.info(f"成功：{result['success_symbols']}只，失败：{result['failure_symbols']}只")
        logger.info(f"新增行数：{result['total_rows']:,}")
        logger.info("=" * 60)
        
        return result
    
    def fill_all_required_years(self, limit_symbols: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """
        填充所有必需年份数据
        
        Parameters
        ----------
        limit_symbols : Optional[int]
            限制股票数量（用于测试）
            
        Returns
        -------
        Dict[str, Dict[str, int]]
            各年份填充结果
        """
        results = {}
        
        for year in V82_REQUIRED_YEARS:
            result = self.fill_year_data(year, limit_symbols)
            results[year] = result
        
        return results
    
    def verify_data(self, year: str) -> Tuple[bool, str]:
        """
        验证指定年份数据
        
        Parameters
        ----------
        year : str
            年份
            
        Returns
        -------
        Tuple[bool, str]
            (是否通过，消息)
        """
        logger.info("=" * 60)
        logger.info(f"V82: 验证 {year} 年数据")
        logger.info("=" * 60)
        
        if self.db is None:
            return (False, "数据库连接未初始化")
        
        try:
            # 查询股票数量和总行数
            query = f"""
            SELECT COUNT(DISTINCT symbol) as stock_count, 
                   COUNT(*) as total_rows,
                   MIN(trade_date) as min_date,
                   MAX(trade_date) as max_date
            FROM stock_daily
            WHERE trade_date >= '{year}-01-01' AND trade_date <= '{year}-12-31'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return (False, f"{year}年：无数据")
            
            stock_count = int(result['stock_count'][0])
            total_rows = int(result['total_rows'][0])
            min_date = result['min_date'][0]
            max_date = result['max_date'][0]
            
            logger.info(f"{year}年：股票数={stock_count}, 总行数={total_rows}, 日期范围={min_date} 至 {max_date}")
            
            # 验证阈值
            min_stocks = 100  # 至少 100 只股票
            min_rows = 20000  # 至少 2 万行
            
            if stock_count < min_stocks:
                return (False, f"股票数量不足：{stock_count} < {min_stocks}")
            
            if total_rows < min_rows:
                return (False, f"总行数不足：{total_rows} < {min_rows}")
            
            logger.info(f"V82: {year} 年数据验证通过")
            return (True, f"数据完整 (stocks={stock_count}, rows={total_rows})")
            
        except Exception as e:
            return (False, f"验证失败：{e}")


# ===========================================
# 便捷函数
# ===========================================

def fill_v82_data(years: Optional[List[str]] = None, 
                  limit_symbols: Optional[int] = None,
                  db: Optional[DatabaseManager] = None) -> Dict[str, Dict[str, int]]:
    """
    便捷函数：填充 V82 数据
    
    Parameters
    ----------
    years : Optional[List[str]]
        年份列表
    limit_symbols : Optional[int]
        限制股票数量
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    Dict[str, Dict[str, int]]
        各年份填充结果
    """
    years = years or V82_REQUIRED_YEARS
    filler = V82DataForceFiller(db=db)
    
    results = {}
    for year in years:
        result = filler.fill_year_data(year, limit_symbols)
        results[year] = result
    
    return results


def verify_v82_data(years: Optional[List[str]] = None,
                    db: Optional[DatabaseManager] = None) -> Dict[str, Tuple[bool, str]]:
    """
    便捷函数：验证 V82 数据
    
    Parameters
    ----------
    years : Optional[List[str]]
        年份列表
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    Dict[str, Tuple[bool, str]]
        各年份验证结果
    """
    years = years or V82_REQUIRED_YEARS
    filler = V82DataForceFiller(db=db)
    
    results = {}
    for year in years:
        passed, message = filler.verify_data(year)
        results[year] = (passed, message)
    
    return results


# ===========================================
# 主程序
# ===========================================

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V82 数据补齐模块 - 启动")
    logger.info("=" * 60)
    
    # 检查 akshare 是否可用
    if not AK_AVAILABLE:
        logger.error("V82: akshare 库未安装，请运行：pip install akshare")
        sys.exit(1)
    
    # 检查数据库是否可用
    if not DB_AVAILABLE:
        logger.error("V82: db_manager 模块未找到")
        sys.exit(1)
    
    # 初始化数据库
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V82: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 填充数据
    try:
        results = fill_v82_data(db=db, limit_symbols=None)
        
        logger.info("=" * 60)
        logger.info("V82: 数据抓取完成")
        for year, result in results.items():
            logger.info(f"{year}年：成功={result['success_symbols']}只，失败={result['failure_symbols']}只，新增={result['total_rows']:,}行")
        logger.info("=" * 60)
        
        # 验证数据
        logger.info("")
        logger.info("=" * 60)
        logger.info("V82: 数据验证")
        logger.info("=" * 60)
        
        verify_results = verify_v82_data(db=db)
        for year, (passed, message) in verify_results.items():
            status = "✓" if passed else "✗"
            logger.info(f"{year}年：{status} {message}")
        
    except Exception as e:
        logger.error(f"V82: 数据抓取失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V82DataFetcher',
    'V82DataForceFiller',
    'fill_v82_data',
    'verify_v82_data',
    'V82_REQUIRED_YEARS',
]