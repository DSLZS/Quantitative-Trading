"""
V83 数据修复器 - 带指数退避重试机制的强制数据补齐

【V83 核心理念】
1. Exponential Backoff：指数退避重试机制
2. Connection Aborted 处理：自动 sleep(60) 并断点续传
3. 失败符号追踪：跳过失败股票并记录，全部运行完后自动重试
4. 数据完整性校验：2019/2021/2024 三年，每年 stock_daily 记录必须 > 50 万条

【V83 数据要求】
- 2019 年：stock_daily 记录 > 500,000 条
- 2021 年：stock_daily 记录 > 500,000 条
- 2024 年：stock_daily 记录 > 500,000 条
- 数据抓取完整率 >= 99%

作者：量化系统
版本：V83.0
日期：2026-03-28
"""

import sys
import os
import time
import random
import traceback
import math
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

# 尝试导入 akshare
try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False
    logger.error("V83: akshare 库未安装，请运行：pip install akshare")

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
        logger.error("V83: db_manager 模块未找到")


# ===========================================
# V83 配置常量
# ===========================================

# V83 强制测试年份
V83_REQUIRED_YEARS = ['2019', '2021', '2024']

# 数据表配置
V83_STOCK_DAILY_TABLE = "stock_daily"

# 数据完整性阈值（硬性要求）
V83_MIN_ROWS_PER_YEAR = 500000  # 每年至少 50 万条记录

# 重试配置 - Exponential Backoff
V83_MAX_RETRY_ATTEMPTS = 5
V83_BASE_RETRY_DELAY = 1.0  # 基础延迟（秒）
V83_MAX_RETRY_DELAY = 300.0  # 最大延迟（秒）
V83_CONNECTION_ABORTED_DELAY = 60.0  # Connection aborted 时的强制延迟（秒）

# 随机抖动配置（避免重试风暴）
V83_JITTER_FACTOR = 0.1

# 写入批次大小
V83_WRITE_BATCH_SIZE = 5000

# 失败符号文件
V83_FAILED_SYMBOLS_FILE = "logs/failed_symbols.txt"
V83_RETRY_LOG_FILE = "logs/v83_retry_log.txt"

# 进度报告文件
V83_PROGRESS_FILE = "logs/v83_progress.json"


# ===========================================
# V83 工具函数
# ===========================================

def exponential_backoff(attempt: int, base_delay: float = V83_BASE_RETRY_DELAY,
                        max_delay: float = V83_MAX_RETRY_DELAY,
                        jitter: float = V83_JITTER_FACTOR) -> float:
    """
    计算指数退避延迟时间
    
    公式：delay = min(base_delay * 2^attempt + random_jitter, max_delay)
    
    Parameters
    ----------
    attempt : int
        当前重试次数（从 0 开始）
    base_delay : float
        基础延迟（秒）
    max_delay : float
        最大延迟（秒）
    jitter : float
        随机抖动因子
        
    Returns
    -------
    float
        延迟时间（秒）
    """
    # 指数增长
    delay = base_delay * (2 ** attempt)
    
    # 添加随机抖动
    jitter_range = delay * jitter
    jitter_value = random.uniform(-jitter_range, jitter_range)
    delay += jitter_value
    
    # 限制最大延迟
    delay = min(delay, max_delay)
    
    return max(0.1, delay)  # 至少 0.1 秒


def is_connection_aborted_error(error_msg: str) -> bool:
    """
    判断是否为连接中止错误
    
    Parameters
    ----------
    error_msg : str
        错误消息
        
    Returns
    -------
    bool
        是否为连接中止错误
    """
    connection_keywords = [
        'Connection aborted',
        'connection aborted',
        'Connection reset',
        'connection reset',
        'Connection refused',
        'connection refused',
        'Timeout',
        'timeout',
        'NetworkError',
        'network error',
        'EOFError',
        'Read timed out',
        'read timed out',
    ]
    
    return any(keyword in error_msg for keyword in connection_keywords)


def ensure_directory(file_path: str) -> None:
    """确保文件所在目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# ===========================================
# V83 失败符号管理器
# ===========================================

class V83FailedSymbolManager:
    """V83 失败符号管理器 - 追踪和重试失败的股票"""
    
    def __init__(self, failed_file: str = V83_FAILED_SYMBOLS_FILE):
        self.failed_file = failed_file
        self.failed_symbols: Dict[str, List[Dict[str, Any]]] = {}
        ensure_directory(failed_file)
        self._load_existing_failures()
    
    def _load_existing_failures(self) -> None:
        """加载已存在的失败记录"""
        if os.path.exists(self.failed_file):
            try:
                with open(self.failed_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('|')
                            if len(parts) >= 2:
                                symbol = parts[0].strip()
                                year = parts[1].strip()
                                error = parts[2].strip() if len(parts) > 2 else ""
                                timestamp = parts[3].strip() if len(parts) > 3 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                if year not in self.failed_symbols:
                                    self.failed_symbols[year] = []
                                
                                self.failed_symbols[year].append({
                                    'symbol': symbol,
                                    'error': error,
                                    'timestamp': timestamp,
                                    'retry_count': 0
                                })
                logger.info(f"V83: 从 {self.failed_file} 加载了 {sum(len(v) for v in self.failed_symbols.values())} 条失败记录")
            except Exception as e:
                logger.warning(f"V83: 加载失败记录失败：{e}")
    
    def add_failure(self, symbol: str, year: str, error: str) -> None:
        """添加失败记录"""
        if year not in self.failed_symbols:
            self.failed_symbols[year] = []
        
        self.failed_symbols[year].append({
            'symbol': symbol,
            'error': error,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'retry_count': 0
        })
        self._save_failures()
    
    def get_failed_symbols(self, year: str) -> List[str]:
        """获取指定年份的失败股票列表"""
        if year not in self.failed_symbols:
            return []
        
        # 去重
        seen = set()
        unique_symbols = []
        for record in self.failed_symbols[year]:
            if record['symbol'] not in seen:
                seen.add(record['symbol'])
                unique_symbols.append(record['symbol'])
        
        return unique_symbols
    
    def clear_failures(self, year: str) -> None:
        """清除指定年份的失败记录"""
        if year in self.failed_symbols:
            del self.failed_symbols[year]
        self._save_failures()
    
    def _save_failures(self) -> None:
        """保存失败记录到文件"""
        try:
            ensure_directory(self.failed_file)
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                f.write(f"# V83 Failed Symbols Log\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Format: symbol|year|error|timestamp\n\n")
                
                for year, symbols in sorted(self.failed_symbols.items()):
                    f.write(f"\n# === {year} ===\n")
                    for record in symbols:
                        line = f"{record['symbol']}|{year}|{record['error']}|{record['timestamp']}\n"
                        f.write(line)
            
            logger.debug(f"V83: 失败记录已保存到 {self.failed_file}")
        except Exception as e:
            logger.error(f"V83: 保存失败记录失败：{e}")
    
    def get_total_failure_count(self) -> int:
        """获取总失败数量"""
        return sum(len(v) for v in self.failed_symbols.values())
    
    def get_unique_failure_count(self) -> int:
        """获取去重后的失败股票数量"""
        all_symbols = set()
        for year, symbols in self.failed_symbols.items():
            for record in symbols:
                all_symbols.add(record['symbol'])
        return len(all_symbols)


# ===========================================
# V83 数据获取器（带重试机制）
# ===========================================

class V83DataFetcher:
    """
    V83 数据获取器 - 带指数退避重试机制
    
    【核心特性】
    1. Exponential Backoff：指数退避重试
    2. Connection Aborted 处理：自动 sleep(60) 并断点续传
    3. 失败符号追踪：记录失败股票
    """
    
    def __init__(self, failed_manager: Optional[V83FailedSymbolManager] = None):
        self._last_error = ""
        self._retry_count = 0
        self.failed_manager = failed_manager or V83FailedSymbolManager()
    
    def fetch_with_retry(self, symbol: str, year: str,
                         start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """
        带重试机制的数据获取
        
        Parameters
        ----------
        symbol : str
            股票代码
        year : str
            年份
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Optional[pl.DataFrame]
            获取到的数据
        """
        last_exception = None
        
        for attempt in range(V83_MAX_RETRY_ATTEMPTS):
            try:
                # 获取数据
                df = self._fetch_stock_daily(symbol, start_date, end_date)
                
                if df is not None and not df.is_empty():
                    self._retry_count = 0  # 重置重试计数
                    return df
                else:
                    logger.warning(f"V83: {symbol} 无数据")
                    return None
                    
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                self._last_error = error_msg
                
                # 判断错误类型
                is_aborted = is_connection_aborted_error(error_msg)
                
                if is_aborted:
                    # Connection Aborted：强制延迟 60 秒
                    logger.warning(f"V83: {symbol} 连接中止，强制延迟 {V83_CONNECTION_ABORTED_DELAY}秒...")
                    time.sleep(V83_CONNECTION_ABORTED_DELAY)
                    # 记录失败
                    self.failed_manager.add_failure(symbol, year, f"Connection Aborted: {error_msg}")
                else:
                    # 其他错误：指数退避
                    if attempt < V83_MAX_RETRY_ATTEMPTS - 1:
                        delay = exponential_backoff(attempt)
                        logger.warning(f"V83: {symbol} 第{attempt + 1}次重试，延迟{delay:.2f}秒...")
                        time.sleep(delay)
                    else:
                        # 最后一次重试失败
                        logger.error(f"V83: {symbol} 重试{V83_MAX_RETRY_ATTEMPTS}次后仍失败：{error_msg}")
                        self.failed_manager.add_failure(symbol, year, error_msg)
        
        # 所有重试都失败
        logger.error(f"V83: {symbol} 所有重试失败，已记录到失败列表")
        return None
    
    def _fetch_stock_daily(self, symbol: str, start_date: str,
                           end_date: str) -> Optional[pl.DataFrame]:
        """
        获取个股日线数据（内部方法）
        
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
            raise ImportError("V83: akshare 库未安装")
        
        try:
            # 使用 ak.stock_zh_a_hist 获取历史数据
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
            
            # 标准化列名 - 完整映射（包含所有可能的列）
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
                '股票代码': 'symbol',  # 新增：处理股票代码列
            }
            
            # 重命名存在的列
            available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
            if available_cols:
                df = df.rename(available_cols)
            
            # 确保 symbol 列存在（如果没有股票代码列，则使用传入的 symbol）
            if 'symbol' not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias('symbol'))
            
            # 格式化日期
            if 'trade_date' in df.columns:
                df = df.with_columns(pl.col('trade_date').cast(pl.Utf8).alias('trade_date'))
            
            # 选择需要的列（确保与数据库表结构匹配）
            required_cols = ['symbol', 'trade_date', 'open', 'close', 'high', 'low', 
                            'volume', 'amount', 'pct_chg']
            available_required = [c for c in required_cols if c in df.columns]
            
            if len(available_required) >= 4:  # 至少有 symbol, trade_date 和价格数据
                df = df.select(available_required)
            
            return df
            
        except Exception as e:
            raise e


# ===========================================
# V83 数据修复器
# ===========================================

class V83DataRepairer:
    """
    V83 数据修复器 - 强制补齐数据并满足完整性要求
    
    【V83 核心特性】
    1. 数据完整性校验：每年 > 50 万条记录
    2. Exponential Backoff：指数退避重试
    3. Connection Aborted 处理：自动 sleep(60) 并断点续传
    4. 失败符号追踪：跳过失败股票，全部运行完后自动重试
    5. 断点续传：支持从中断点继续
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V83: 数据库连接失败：{e}")
                self.db = None
        else:
            self.db = db
        
        self.failed_manager = V83FailedSymbolManager()
        self.fetcher = V83DataFetcher(self.failed_manager)
        self.total_inserted_rows = 0
        self.total_success_symbols = 0
        self.total_failure_symbols = 0
    
    def create_table(self) -> None:
        """建表校验"""
        if self.db is None:
            logger.error("V83: 数据库连接未初始化，无法建表")
            return
        
        logger.info("V83: 开始建表校验...")
        
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
            logger.info("V83: stock_daily 表创建成功")
        except Exception as e:
            logger.error(f"V83: 创建 stock_daily 表失败：{e}")
            raise
    
    def check_data_integrity(self, year: str) -> Tuple[bool, int, str]:
        """
        检查指定年份的数据完整性
        
        Parameters
        ----------
        year : str
            年份
            
        Returns
        -------
        Tuple[bool, int, str]
            (是否达标，记录数，消息)
        """
        if self.db is None:
            return (False, 0, "数据库连接未初始化")
        
        try:
            query = f"""
            SELECT COUNT(*) as cnt 
            FROM stock_daily
            WHERE trade_date >= '{year}-01-01' 
              AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return (False, 0, "无法查询 stock_daily 表")
            
            daily_count = int(df['cnt'][0])
            
            if daily_count >= V83_MIN_ROWS_PER_YEAR:
                return (True, daily_count, f"数据完整 (rows={daily_count:,} >= {V83_MIN_ROWS_PER_YEAR:,})")
            else:
                deficit = V83_MIN_ROWS_PER_YEAR - daily_count
                return (False, daily_count, f"数据不完整 (rows={daily_count:,} < {V83_MIN_ROWS_PER_YEAR:,}, 缺{deficit:,})")
                
        except Exception as e:
            return (False, 0, f"检查失败：{e}")
    
    def _get_existing_symbols(self, year: str) -> Set[str]:
        """获取指定年份已存在的股票代码"""
        if self.db is None:
            return set()
        
        try:
            query = f"""
            SELECT DISTINCT symbol 
            FROM stock_daily 
            WHERE trade_date >= '{year}-01-01' 
              AND trade_date <= '{year}-12-31'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return set()
            
            return set(result['symbol'].to_list())
            
        except Exception as e:
            logger.warning(f"V83: 查询 {year} 年已存在股票失败：{e}")
            return set()
    
    def _get_stock_list(self) -> List[str]:
        """获取 A 股股票列表"""
        if not AK_AVAILABLE:
            raise ImportError("V83: akshare 库未安装")
        
        try:
            # 获取 A 股股票列表
            df = ak.stock_info_a_code_name()
            
            if df is None or df.empty:
                return []
            
            # 转换为 Polars
            df = pl.from_pandas(df)
            
            # 获取股票代码
            symbols = df['code'].to_list() if 'code' in df.columns else []
            
            logger.info(f"V83: 获取到 {len(symbols)} 只 A 股股票")
            return symbols
            
        except Exception as e:
            logger.error(f"V83: 获取股票列表失败：{e}")
            return []
    
    def _write_to_db(self, df: pl.DataFrame) -> int:
        """写入数据到数据库"""
        if self.db is None:
            logger.error("V83: 数据库连接未初始化")
            return 0
        
        try:
            rows = len(df)
            
            # 数据库表结构列（只包含这些列）
            db_columns = ['symbol', 'trade_date', 'open', 'close', 'high', 'low', 
                         'volume', 'amount', 'pct_chg', 'industry_code', 'total_mv', 'is_st']
            
            # 只选择 DataFrame 中存在的列
            available_cols = [c for c in db_columns if c in df.columns]
            
            if len(available_cols) < 4:
                logger.warning(f"V83: DataFrame 列不足，跳过写入。当前列：{df.columns}")
                return 0
            
            # 选择需要的列
            df_to_write = df.select(available_cols)
            
            # 确保缺失的列有默认值
            if 'industry_code' not in available_cols:
                df_to_write = df_to_write.with_columns(pl.lit('').alias('industry_code'))
            if 'total_mv' not in available_cols:
                df_to_write = df_to_write.with_columns(pl.lit(0.0).alias('total_mv'))
            if 'is_st' not in available_cols:
                df_to_write = df_to_write.with_columns(pl.lit(0).alias('is_st'))
            
            # 分批写入
            for start_idx in range(0, rows, V83_WRITE_BATCH_SIZE):
                end_idx = min(start_idx + V83_WRITE_BATCH_SIZE, rows)
                batch_df = df_to_write.slice(start_idx, end_idx - start_idx)
                
                self.db.to_sql(batch_df, V83_STOCK_DAILY_TABLE, if_exists='append')
            
            return rows
            
        except Exception as e:
            logger.error(f"V83: 写入数据失败：{e}")
            return 0
    
    def repair_year_data(self, year: str) -> Dict[str, Any]:
        """
        修复指定年份数据
        
        Parameters
        ----------
        year : str
            年份
            
        Returns
        -------
        Dict[str, Any]
            修复结果
        """
        logger.info("=" * 60)
        logger.info(f"V83: 开始修复 {year} 年数据")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V83: 数据库连接未初始化")
            return {'success': False, 'error': 'Database not initialized'}
        
        # 建表校验
        self.create_table()
        
        # 检查当前数据完整性
        passed, current_count, message = self.check_data_integrity(year)
        logger.info(f"V83: {year} 年当前状态：{message}")
        
        if passed:
            logger.info(f"V83: {year} 年数据已达标，跳过")
            return {'success': True, 'skipped': True, 'count': current_count}
        
        # 计算需要抓取的数据量
        deficit = V83_MIN_ROWS_PER_YEAR - current_count
        logger.info(f"V83: {year} 年需要补充至少 {deficit:,} 条记录")
        
        # 获取已存在的股票
        existing_symbols = self._get_existing_symbols(year)
        logger.info(f"V83: {year} 年已存在 {len(existing_symbols)} 只股票")
        
        # 获取股票列表
        all_symbols = self._get_stock_list()
        
        # 过滤掉已存在的股票
        new_symbols = [s for s in all_symbols if s not in existing_symbols]
        
        # 也过滤掉失败列表中的股票（稍后重试）
        failed_for_year = self.failed_manager.get_failed_symbols(year)
        new_symbols = [s for s in new_symbols if s not in failed_for_year]
        
        logger.info(f"V83: 需要抓取 {len(new_symbols)} 只新股票")
        
        if not new_symbols:
            logger.info(f"V83: {year} 年没有新股票可抓取")
            # 检查是否需要重试失败列表
            if failed_for_year:
                logger.info(f"V83: 将重试 {len(failed_for_year)} 只失败的股票")
            else:
                return {'success': True, 'skipped': True, 'count': current_count}
        
        # 日期范围
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        # 抓取统计
        success_symbols = 0
        failure_symbols = 0
        total_rows = 0
        
        # 抓取数据
        for i, symbol in enumerate(new_symbols):
            try:
                logger.info(f"[PROGRESS] {year} [{i+1}/{len(new_symbols)}]: 抓取 {symbol}...")
                
                df = self.fetcher.fetch_with_retry(symbol, year, start_date, end_date)
                
                if df is not None and not df.is_empty():
                    rows = self._write_to_db(df)
                    
                    if rows > 0:
                        success_symbols += 1
                        total_rows += rows
                        self.total_inserted_rows += rows
                        self.total_success_symbols += 1
                        logger.info(f"[PROGRESS] {year} {symbol}: Inserted {rows} rows.")
                    else:
                        failure_symbols += 1
                        self.total_failure_symbols += 1
                else:
                    logger.warning(f"V83: {symbol} 无数据")
                    failure_symbols += 1
                    self.total_failure_symbols += 1
                    
            except Exception as e:
                logger.error(f"V83: {symbol} 处理失败：{e}")
                failure_symbols += 1
                self.total_failure_symbols += 1
                self.failed_manager.add_failure(symbol, year, str(e))
        
        # 重试失败列表
        if failed_for_year:
            logger.info("")
            logger.info(f"V83: 开始重试 {len(failed_for_year)} 只失败的股票...")
            retry_success, retry_failure, retry_rows = self._retry_failed_symbols(year, failed_for_year, start_date, end_date)
            success_symbols += retry_success
            failure_symbols += retry_failure
            total_rows += retry_rows
        
        # 打印最终统计
        final_count = self._get_table_count(V83_STOCK_DAILY_TABLE)
        passed, final_count, message = self.check_data_integrity(year)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"V83: {year} 年数据修复完成")
        logger.info(f"[FINAL COUNT] stock_daily: {final_count:,} rows")
        logger.info(f"本次抓取：成功={success_symbols}只，失败={failure_symbols}只")
        logger.info(f"本次新增：{total_rows:,} 行")
        logger.info(f"数据状态：{message}")
        logger.info("=" * 60)
        
        return {
            'success': passed,
            'count': final_count,
            'success_symbols': success_symbols,
            'failure_symbols': failure_symbols,
            'total_rows': total_rows
        }
    
    def _retry_failed_symbols(self, year: str, failed_symbols: List[str],
                               start_date: str, end_date: str) -> Tuple[int, int, int]:
        """
        重试失败的股票
        
        Parameters
        ----------
        year : str
            年份
        failed_symbols : List[str]
            失败股票列表
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Tuple[int, int, int]
            (成功数，失败数，行数)
        """
        success_count = 0
        failure_count = 0
        total_rows = 0
        
        for i, symbol in enumerate(failed_symbols):
            try:
                logger.info(f"[RETRY] {year} [{i+1}/{len(failed_symbols)}]: 重试 {symbol}...")
                
                df = self.fetcher.fetch_with_retry(symbol, year, start_date, end_date)
                
                if df is not None and not df.is_empty():
                    rows = self._write_to_db(df)
                    
                    if rows > 0:
                        success_count += 1
                        total_rows += rows
                        logger.info(f"[RETRY] {year} {symbol}: 成功插入 {rows} rows.")
                    else:
                        failure_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                logger.error(f"V83: {symbol} 重试失败：{e}")
                failure_count += 1
        
        logger.info(f"V83: 重试完成：成功={success_count}, 失败={failure_count}")
        return (success_count, failure_count, total_rows)
    
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
            logger.error(f"V83: 查询表 {table_name} 行数失败：{e}")
            return 0
    
    def repair_all_required_years(self) -> Dict[str, Dict[str, Any]]:
        """
        修复所有必需年份数据
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            各年份修复结果
        """
        results = {}
        
        for year in V83_REQUIRED_YEARS:
            result = self.repair_year_data(year)
            results[year] = result
        
        return results
    
    def verify_all_years(self) -> Dict[str, Tuple[bool, str]]:
        """
        验证所有年份数据
        
        Returns
        -------
        Dict[str, Tuple[bool, str]]
            各年份验证结果
        """
        results = {}
        
        for year in V83_REQUIRED_YEARS:
            passed, count, message = self.check_data_integrity(year)
            results[year] = (passed, message)
        
        return results
    
    def get_completion_rate(self) -> float:
        """
        计算数据抓取完整率
        
        Returns
        -------
        float
            完整率（0-1 之间）
        """
        total = self.total_success_symbols + self.total_failure_symbols
        if total == 0:
            return 1.0
        return self.total_success_symbols / total
    
    def generate_report(self) -> str:
        """生成修复报告"""
        lines = [
            "=" * 60,
            "V83 数据修复报告",
            "=" * 60,
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "【抓取统计】",
            f"  总成功股票数：{self.total_success_symbols}",
            f"  总失败股票数：{self.total_failure_symbols}",
            f"  总新增行数：{self.total_inserted_rows:,}",
            f"  抓取完整率：{self.get_completion_rate():.2%}",
            "",
            "【年份验证】",
        ]
        
        verify_results = self.verify_all_years()
        for year, (passed, message) in verify_results.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {year}年：{status} {message}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ===========================================
# 主程序
# ===========================================

def setup_logging() -> None:
    """配置日志"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/v83_data_repairer_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def main() -> int:
    """主函数"""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("V83 数据修复器 - 启动")
    logger.info("=" * 60)
    logger.info(f"数据完整性要求：每年 > {V83_MIN_ROWS_PER_YEAR:,} 条记录")
    logger.info(f"重试机制：Exponential Backoff (max={V83_MAX_RETRY_ATTEMPTS})")
    logger.info(f"Connection Aborted 延迟：{V83_CONNECTION_ABORTED_DELAY}秒")
    logger.info("=" * 60)
    
    # 检查 akshare 是否可用
    if not AK_AVAILABLE:
        logger.error("V83: akshare 库未安装，请运行：pip install akshare")
        return 1
    
    # 检查数据库是否可用
    if not DB_AVAILABLE:
        logger.error("V83: db_manager 模块未找到")
        return 1
    
    # 初始化数据库
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V83: 数据库连接失败：{e}")
        return 1
    
    # 初始化修复器
    repairer = V83DataRepairer(db=db)
    
    # 首先检查数据完整性
    logger.info("")
    logger.info("=" * 60)
    logger.info("V83: 数据完整性检查")
    logger.info("=" * 60)
    
    needs_repair = []
    for year in V83_REQUIRED_YEARS:
        passed, count, message = repairer.check_data_integrity(year)
        status = "✓" if passed else "✗"
        logger.info(f"{year}年：{status} {message}")
        if not passed:
            needs_repair.append(year)
    
    if not needs_repair:
        logger.info("")
        logger.info("V83: 所有年份数据已达标，无需修复")
        return 0
    
    logger.info("")
    logger.info(f"V83: 需要修复的年份：{', '.join(needs_repair)}")
    
    # 执行修复
    try:
        results = {}
        for year in needs_repair:
            result = repairer.repair_year_data(year)
            results[year] = result
        
        # 打印最终报告
        logger.info("")
        logger.info(repairer.generate_report())
        
        # 验证最终结果
        logger.info("")
        logger.info("=" * 60)
        logger.info("V83: 最终验证")
        logger.info("=" * 60)
        
        all_passed = True
        for year in V83_REQUIRED_YEARS:
            passed, count, message = repairer.check_data_integrity(year)
            status = "✓" if passed else "✗"
            logger.info(f"{year}年：{status} {message}")
            if not passed:
                all_passed = False
        
        # 检查抓取完整率
        completion_rate = repairer.get_completion_rate()
        logger.info("")
        logger.info(f"V83: 抓取完整率：{completion_rate:.2%}")
        
        if completion_rate < 0.99:
            logger.warning(f"V83: 抓取完整率低于 99%，请检查失败列表")
        
        if all_passed and completion_rate >= 0.99:
            logger.info("")
            logger.info("V83: 数据修复成功！")
            return 0
        else:
            logger.warning("")
            logger.warning("V83: 数据修复未完全成功，请检查日志")
            return 1
            
    except Exception as e:
        logger.error(f"V83: 数据修复失败：{e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())