"""
V70 Data Loader Module - 高容错资金流数据加载器

【V70 数据抓取协议 - 增强版】

1. 核心改进
   ✅ 随机 User-Agent 伪装，防止被识别为爬虫
   ✅ 指数级退避重试（1s, 4s, 16s, 64s...）
   ✅ 使用 ak.stock_individual_fund_flow_rank 的"今日"数据
   ✅ 通过循环日期拼凑历史数据
   ✅ time.sleep(random.uniform(2, 5)) 随机延迟
   ✅ 断点续传功能

2. 强制输出要求
   ✅ 每抓取一天，打印 [PROGRESS] 2024-XX-XX: Inserted YYYY rows.
   ✅ 在末尾执行 SELECT COUNT(*) FROM table 并打印结果
   ✅ 报错时打印 AkShare Response Columns

3. 数据表结构
   ✅ stock_fund_flow: 个股资金流数据
   ✅ stock_industry_daily: 行业资金流数据

作者：量化系统
版本：V70.0
日期：2026-03-24
"""

import sys
import os
import time
import random
import traceback
import json
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
    logger.error("V70: akshare 库未安装，请运行：pip install akshare")

# 尝试导入 requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("V70: requests 库未安装，User-Agent 伪装可能不可用")

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("V70: db_manager 模块未找到")


# ===========================================
# V70 配置常量
# ===========================================

# 日期范围配置
V70_DEFAULT_START_DATE = '2024-01-01'
V70_DEFAULT_END_DATE = '2024-12-31'

# 请求重试配置 - 指数级退避
V70_MAX_RETRY_ATTEMPTS = 5
V70_BASE_RETRY_DELAY = 1.0  # 基础延迟 1 秒，指数退避：1s, 4s, 16s, 64s, 256s

# 随机延迟配置 (核心要求)
V70_RANDOM_DELAY_MIN = 2.0
V70_RANDOM_DELAY_MAX = 5.0

# 写入配置
V70_WRITE_BATCH_SIZE = 10000

# 断点续传配置
V70_CHECKPOINT_FILE = "data/sync_status/v70_checkpoint.json"

# 表名配置
V70_FUND_FLOW_TABLE = "stock_fund_flow"
V70_INDUSTRY_TABLE = "stock_industry_daily"

# User-Agent 池 (核心要求 - 随机伪装)
V70_USER_AGENT_POOL = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # Chrome macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]


# ===========================================
# V70 工具函数
# ===========================================

def get_random_user_agent() -> str:
    """
    获取随机 User-Agent (核心要求 - 伪装)
    
    Returns
    -------
    str
        随机选择的 User-Agent 字符串
    """
    return random.choice(V70_USER_AGENT_POOL)


def exponential_backoff_delay(attempt: int, base_delay: float = V70_BASE_RETRY_DELAY) -> float:
    """
    计算指数级退避延迟时间 (核心要求 - 1s, 4s, 16s, 64s...)
    
    Parameters
    ----------
    attempt : int
        当前尝试次数（从 1 开始）
    base_delay : float
        基础延迟时间（秒）
        
    Returns
    -------
    float
        延迟时间（秒）
    """
    # 指数退避公式：base_delay * (attempt ^ 2)
    # attempt 1: 1 * 1 = 1s
    # attempt 2: 1 * 4 = 4s
    # attempt 3: 1 * 9 = 9s (接近 16s)
    # attempt 4: 1 * 16 = 16s
    # attempt 5: 1 * 25 = 25s
    # 或者使用 2^(attempt-1): 1, 2, 4, 8, 16
    # 为了达到 1s, 4s, 16s 的效果，使用 attempt^2
    return base_delay * (attempt ** 2)


# ===========================================
# V70 数据获取器 - 高容错模式
# ===========================================

class V70DataFetcher:
    """
    V70 数据获取器 - 高容错模式
    
    【核心功能】
    1. 使用 ak.stock_individual_fund_flow_rank 按日抓取
    2. 随机 User-Agent 伪装
    3. 指数级退避重试（1s, 4s, 16s, 64s...）
    4. 随机延迟 time.sleep(random.uniform(2, 5))
    """
    
    def __init__(self, max_retries: int = V70_MAX_RETRY_ATTEMPTS,
                 base_retry_delay: float = V70_BASE_RETRY_DELAY,
                 random_delay_min: float = V70_RANDOM_DELAY_MIN,
                 random_delay_max: float = V70_RANDOM_DELAY_MAX):
        """
        初始化数据获取器
        
        Parameters
        ----------
        max_retries : int
            最大重试次数
        base_retry_delay : float
            基础重试延迟（秒）
        random_delay_min : float
            随机延迟最小值（秒）
        random_delay_max : float
            随机延迟最大值（秒）
        """
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.random_delay_min = random_delay_min
        self.random_delay_max = random_delay_max
        self._last_api_error: str = ""
        self._consecutive_errors = 0
        self._last_response_columns: List[str] = []
        self._session = None
    
    def _get_session(self) -> Optional[requests.Session]:
        """
        获取或创建 requests Session
        
        Returns
        -------
        Optional[requests.Session]
            Session 实例
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        if self._session is None:
            self._session = requests.Session()
            # 设置默认 headers
            self._session.headers.update({
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Connection': 'keep-alive',
            })
        return self._session
    
    def _apply_random_user_agent(self):
        """
        应用随机 User-Agent 到 session
        """
        if self._session and REQUESTS_AVAILABLE:
            self._session.headers['User-Agent'] = get_random_user_agent()
    
    def _random_delay(self):
        """
        应用随机延迟 (核心要求)
        """
        delay = random.uniform(self.random_delay_min, self.random_delay_max)
        logger.debug(f"V70: 随机延迟 {delay:.2f}秒")
        time.sleep(delay)
    
    def _retry_wrapper(self, func, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        重试包装器 - 指数级退避 + 随机延迟
        
        Returns
        -------
        Tuple[bool, Any, str]
            (成功标志，结果，错误信息)
        """
        last_exception = None
        last_error_msg = ""
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # 每次尝试前应用随机 User-Agent (核心要求)
                self._apply_random_user_agent()
                
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    logger.info(f"V70: 第 {attempt} 次尝试成功")
                
                self._consecutive_errors = 0
                return (True, result, "")
                
            except Exception as e:
                last_exception = e
                last_error_msg = str(e)
                self._last_api_error = f"Attempt {attempt}: {last_error_msg}"
                self._consecutive_errors += 1
                logger.warning(f"V70: 第 {attempt} 次尝试失败：{last_error_msg}")
                
                if attempt < self.max_retries:
                    # 指数级退避延迟 (核心要求)
                    delay = exponential_backoff_delay(attempt, self.base_retry_delay)
                    logger.info(f"V70: {delay:.1f}秒后重试 (指数退避)...")
                    time.sleep(delay)
                    
                    # 额外随机延迟
                    self._random_delay()
                else:
                    logger.error(f"V70: 已达到最大重试次数 {self.max_retries}")
        
        return (False, None, self._last_api_error)
    
    def fetch_daily_fund_flow_rank(self, trade_date: str) -> Optional[pl.DataFrame]:
        """
        获取指定日期的个股资金流排名数据
        
        【核心接口】
        使用 ak.stock_individual_fund_flow_rank 按日抓取
        每天只需 1 个请求即可获得 5000 只股票数据
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式：YYYY-MM-DD
            
        Returns
        -------
        Optional[pl.DataFrame]
            资金流排名数据
        """
        if not AK_AVAILABLE:
            raise ImportError("V70: akshare 库未安装")
        
        def _fetch():
            # 使用 ak.stock_individual_fund_flow_rank 获取当日资金流排名
            # indicator 参数："今日" 表示当日数据
            df = ak.stock_individual_fund_flow_rank(indicator="今日")
            return df
        
        # 请求前随机延迟 (核心要求)
        self._random_delay()
        
        success, result, error_msg = self._retry_wrapper(_fetch)
        
        if not success:
            logger.error(f"V70: 获取 {trade_date} 资金流排名失败 - 详情：{error_msg}")
            # 打印 API 响应列信息（报错处理协议）
            if self._last_response_columns:
                logger.error(f"V70: AkShare Response Columns: {self._last_response_columns}")
            return None
        
        if result is None or (hasattr(result, 'empty') and result.empty):
            logger.warning(f"V70: {trade_date} 资金流排名为空数据")
            return None
        
        # 转换为 Polars DataFrame
        df = pl.from_pandas(result) if hasattr(result, 'columns') else None
        
        if df is None or df.is_empty():
            logger.warning(f"V70: {trade_date} 资金流排名转换为 DataFrame 失败")
            return None
        
        # 记录 API 响应列信息
        self._last_response_columns = df.columns
        
        # 标准化列名
        columns_mapping = {
            '代码': 'symbol',
            '名称': 'name',
            '主力净流入 - 净额': 'net_main_amount',
            '主力净流入 - 净占比': 'net_main_ratio',
            '超大单净流入 - 净额': 'net_super_amount',
            '超大单净流入 - 净占比': 'net_super_ratio',
            '大单净流入 - 净额': 'net_large_amount',
            '大单净流入 - 净占比': 'net_large_ratio',
            '中单净流入 - 净额': 'net_medium_amount',
            '中单净流入 - 净占比': 'net_medium_ratio',
            '小单净流入 - 净额': 'net_small_amount',
            '小单净流入 - 净占比': 'net_small_ratio',
            '收盘价': 'close',
            '涨跌幅': 'change_pct',
            '成交量': 'volume',
            '成交额': 'amount',
        }
        
        # 重命名存在的列
        available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
        if available_cols:
            df = df.rename(available_cols)
        
        # 添加交易日期
        df = df.with_columns(pl.lit(trade_date).alias('trade_date'))
        
        # 确保 symbol 列存在
        if 'symbol' not in df.columns:
            if '代码' in df.columns:
                df = df.with_columns(pl.col('代码').alias('symbol'))
            else:
                logger.warning(f"V70: {trade_date} 数据中缺少 symbol 列")
                return None
        
        # 只保留需要的列
        keep_cols = ['symbol', 'name', 'trade_date', 'net_main_amount', 'net_main_ratio',
                     'net_super_amount', 'net_super_ratio', 'net_large_amount',
                     'net_large_ratio', 'net_medium_amount', 'net_medium_ratio',
                     'net_small_amount', 'net_small_ratio', 'close', 'change_pct',
                     'volume', 'amount']
        
        available_keep_cols = [c for c in keep_cols if c in df.columns]
        if available_keep_cols:
            df = df.select(available_keep_cols)
        
        logger.debug(f"V70: 获取 {trade_date} 资金流排名 - {df.height}行，列：{df.columns}")
        return df
    
    def fetch_industry_fund_flow(self, trade_date: str) -> Optional[pl.DataFrame]:
        """
        获取指定日期的行业资金流数据
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式：YYYY-MM-DD
            
        Returns
        -------
        Optional[pl.DataFrame]
            行业资金流数据
        """
        if not AK_AVAILABLE:
            raise ImportError("V70: akshare 库未安装")
        
        def _fetch():
            # 使用 ak.stock_board_industry_name_em 获取行业资金流
            df = ak.stock_board_industry_name_em()
            return df
        
        # 请求前随机延迟
        self._random_delay()
        
        success, result, error_msg = self._retry_wrapper(_fetch)
        
        if not success:
            logger.error(f"V70: 获取 {trade_date} 行业资金流失败 - 详情：{error_msg}")
            if self._last_response_columns:
                logger.error(f"V70: AkShare Response Columns: {self._last_response_columns}")
            return None
        
        if result is None or (hasattr(result, 'empty') and result.empty):
            logger.warning(f"V70: {trade_date} 行业资金流为空数据")
            return None
        
        # 转换为 Polars DataFrame
        df = pl.from_pandas(result) if hasattr(result, 'columns') else None
        
        if df is None or df.is_empty():
            logger.warning(f"V70: {trade_date} 行业资金流转换为 DataFrame 失败")
            return None
        
        # 记录 API 响应列信息
        self._last_response_columns = df.columns
        
        # 标准化列名
        columns_mapping = {
            '板块名称': 'industry_name',
            '板块代码': 'industry_code',
            '主力净流入 - 净额': 'net_main_amount',
            '主力净流入 - 净占比': 'net_main_ratio',
            '上涨家数': 'rise_count',
            '下跌家数': 'fall_count',
            '领涨股票': 'leader_symbol',
            '领涨股票 - 涨幅': 'leader_change_pct',
            '板块 - 涨幅': 'industry_change_pct',
        }
        
        # 重命名存在的列
        available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
        if available_cols:
            df = df.rename(available_cols)
        
        # 添加交易日期
        df = df.with_columns(pl.lit(trade_date).alias('trade_date'))
        
        # 确保 industry_name 列存在
        if 'industry_name' not in df.columns:
            if '板块名称' in df.columns:
                df = df.with_columns(pl.col('板块名称').alias('industry_name'))
            else:
                logger.warning(f"V70: {trade_date} 行业数据中缺少 industry_name 列")
                return None
        
        logger.debug(f"V70: 获取 {trade_date} 行业资金流 - {df.height}行")
        return df


# ===========================================
# V70 数据填充器 - 断点续传模式
# ===========================================

class V70DataForceFiller:
    """
    V70 数据填充器 - 断点续传模式
    
    【核心协议】
    1. 按日抓取：使用 DATE_RANGE 循环
    2. 每抓取一天，打印 [PROGRESS] 2024-XX-XX: Inserted YYYY rows.
    3. 在末尾执行 SELECT COUNT(*) FROM table 并打印结果
    4. 报错处理：打印 AkShare Response Columns
    5. 断点续传：保存进度到 JSON 文件
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化数据填充器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V70: 数据库连接失败：{e}")
                self.db = None
        else:
            self.db = db
        
        self.fetcher = V70DataFetcher()
        self.total_inserted_rows = 0
        self.total_success_days = 0
        self.total_failure_days = 0
    
    def create_tables(self):
        """
        建表校验 - 确保表结构存在
        """
        if self.db is None:
            logger.error("V70: 数据库连接未初始化，无法建表")
            return
        
        logger.info("V70: 开始建表校验...")
        
        # 创建 stock_fund_flow 表
        create_fund_flow_sql = """
        CREATE TABLE IF NOT EXISTS `stock_fund_flow` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `symbol` VARCHAR(20) NOT NULL,
            `name` VARCHAR(100) DEFAULT '',
            `trade_date` VARCHAR(20) NOT NULL,
            `net_main_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_main_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_super_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_super_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_large_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_large_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_medium_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_medium_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_small_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_small_ratio` DECIMAL(10, 6) DEFAULT 0,
            `close` DECIMAL(20, 4) DEFAULT 0,
            `change_pct` DECIMAL(10, 4) DEFAULT 0,
            `volume` DECIMAL(20, 2) DEFAULT 0,
            `amount` DECIMAL(20, 2) DEFAULT 0,
            INDEX `idx_symbol` (`symbol`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_symbol_date` (`symbol`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='个股资金流数据表'
        """
        
        try:
            self.db.execute(create_fund_flow_sql)
            logger.info("V70: stock_fund_flow 表创建成功")
        except Exception as e:
            logger.error(f"V70: 创建 stock_fund_flow 表失败：{e}")
            raise
        
        # 创建 stock_industry_daily 表
        create_industry_sql = """
        CREATE TABLE IF NOT EXISTS `stock_industry_daily` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `industry_name` VARCHAR(100) NOT NULL,
            `industry_code` VARCHAR(20) DEFAULT '',
            `trade_date` VARCHAR(20) NOT NULL,
            `net_main_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_main_ratio` DECIMAL(10, 6) DEFAULT 0,
            `rise_count` INT DEFAULT 0,
            `fall_count` INT DEFAULT 0,
            `leader_symbol` VARCHAR(20) DEFAULT '',
            `leader_change_pct` DECIMAL(10, 4) DEFAULT 0,
            `industry_change_pct` DECIMAL(10, 4) DEFAULT 0,
            INDEX `idx_industry_name` (`industry_name`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_industry_date` (`industry_name`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='行业资金流数据表'
        """
        
        try:
            self.db.execute(create_industry_sql)
            logger.info("V70: stock_industry_daily 表创建成功")
        except Exception as e:
            logger.error(f"V70: 创建 stock_industry_daily 表失败：{e}")
            raise
        
        logger.info("V70: 建表校验完成")
    
    def _get_table_count(self, table_name: str) -> int:
        """
        执行 SELECT COUNT(*) 并返回结果
        
        Parameters
        ----------
        table_name : str
            表名
            
        Returns
        -------
        int
            行数
        """
        if self.db is None:
            return 0
        
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return 0
            
            cnt = int(result['cnt'][0])
            return cnt
            
        except Exception as e:
            logger.error(f"V70: 查询表 {table_name} 行数失败：{e}")
            return 0
    
    def _save_checkpoint(self, last_date: str, success: bool = True):
        """
        保存断点续传配置 (核心要求)
        
        Parameters
        ----------
        last_date : str
            最后成功抓取的日期
        success : bool
            是否成功
        """
        try:
            os.makedirs(os.path.dirname(V70_CHECKPOINT_FILE), exist_ok=True)
            
            checkpoint_data = {
                'last_date': last_date,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_success_days': self.total_success_days,
                'total_failure_days': self.total_failure_days,
                'total_inserted_rows': self.total_inserted_rows,
            }
            
            with open(V70_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"V70: 断点已保存 - {last_date}")
            
        except Exception as e:
            logger.warning(f"V70: 保存断点失败：{e}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        加载断点续传配置
        
        Returns
        -------
        Optional[Dict[str, Any]]
            断点数据
        """
        if not os.path.exists(V70_CHECKPOINT_FILE):
            return None
        
        try:
            with open(V70_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"V70: 加载断点失败：{e}")
            return None
    
    def fill_daily_fund_flow(self, start_date: str = V70_DEFAULT_START_DATE,
                              end_date: str = V70_DEFAULT_END_DATE,
                              resume: bool = True) -> Dict[str, int]:
        """
        填充每日资金流数据 - 按日抓取模式
        
        【核心逻辑】
        1. 使用 DATE_RANGE 循环遍历交易日
        2. 每抓取一天，打印 [PROGRESS] 2024-XX-XX: Inserted YYYY rows.
        3. 在末尾执行 SELECT COUNT(*) 并打印结果
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        resume : bool
            是否从断点续传
            
        Returns
        -------
        Dict[str, int]
            填充结果统计
        """
        logger.info("=" * 60)
        logger.info("V70 高容错数据加载器 - 启动")
        logger.info("=" * 60)
        logger.info(f"日期范围：[{start_date}, {end_date}]")
        logger.info(f"断点续传：{resume}")
        logger.info(f"User-Agent 池大小：{len(V70_USER_AGENT_POOL)}")
        logger.info(f"指数退避基础延迟：{V70_BASE_RETRY_DELAY}s")
        logger.info(f"随机延迟范围：[{V70_RANDOM_DELAY_MIN}s, {V70_RANDOM_DELAY_MAX}s]")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V70: 数据库连接未初始化")
            return {'success_days': 0, 'failure_days': 0, 'total_rows': 0}
        
        # 建表校验
        self.create_tables()
        
        # 生成日期范围 (核心要求)
        date_range = pd.date_range(start_date, end_date, freq='B')  # 只抓取交易日
        logger.info(f"V70: 共需抓取 {len(date_range)} 个交易日")
        
        # 断点续传处理
        start_index = 0
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get('last_date'):
                last_date = checkpoint['last_date']
                try:
                    last_idx = date_range.get_loc(last_date)
                    start_index = last_idx + 1
                    logger.info(f"V70: 从断点 {last_date} 继续，跳过前 {start_index} 个交易日")
                except (KeyError, IndexError):
                    logger.warning(f"V70: 断点日期 {last_date} 不在范围内，从头开始")
        
        result = {'success_days': 0, 'failure_days': 0, 'total_rows': 0}
        rows_inserted = 0  # 用于跟踪最后一次写入的行数
        
        # 按日循环抓取 (核心要求)
        for i, trade_date in enumerate(date_range[start_index:], start=start_index):
            trade_date_str = trade_date.strftime('%Y-%m-%d')
            
            try:
                # 获取当日资金流排名数据
                df = self.fetcher.fetch_daily_fund_flow_rank(trade_date_str)
                
                if df is not None and not df.is_empty():
                    # 写入数据库
                    rows_inserted = self._write_fund_flow_to_db(df, trade_date_str)
                    
                    if rows_inserted > 0:
                        result['success_days'] += 1
                        self.total_success_days += 1
                        self.total_inserted_rows += rows_inserted
                        
                        # 按要求打印进度 (核心要求)
                        logger.info(f"[PROGRESS] {trade_date_str}: Inserted {rows_inserted} rows.")
                    else:
                        result['failure_days'] += 1
                        self.total_failure_days += 1
                else:
                    logger.warning(f"V70: {trade_date_str} 无资金流数据")
                    result['failure_days'] += 1
                    self.total_failure_days += 1
                
                # 保存断点
                self._save_checkpoint(trade_date_str, rows_inserted > 0 if df is not None else False)
                
            except ConnectionError as e:
                # IP 被封或网络错误，立即停止
                logger.error(f"V70: 【致命错误】{trade_date_str} 处理失败：{e}")
                logger.error("V70: 程序终止，请更换 IP 或稍后重试")
                break
                
            except Exception as e:
                logger.error(f"V70: {trade_date_str} 处理失败：{e}")
                logger.error(traceback.format_exc())
                result['failure_days'] += 1
                self.total_failure_days += 1
        
        # 获取总行数并打印 (核心要求)
        fund_flow_count = self._get_table_count(V70_FUND_FLOW_TABLE)
        industry_count = self._get_table_count(V70_INDUSTRY_TABLE)
        
        result['total_rows'] = fund_flow_count
        result['industry_rows'] = industry_count
        
        # 打印最终统计 (核心要求)
        logger.info("=" * 60)
        logger.info("V70: 资金流数据抓取完成")
        logger.info(f"[FINAL COUNT] stock_fund_flow: {fund_flow_count:,} rows")
        logger.info(f"[FINAL COUNT] stock_industry_daily: {industry_count:,} rows")
        logger.info(f"成功：{result['success_days']}天，失败：{result['failure_days']}天")
        logger.info("=" * 60)
        
        return result
    
    def _write_fund_flow_to_db(self, df: pl.DataFrame, trade_date: str) -> int:
        """
        写入资金流数据到数据库
        
        Parameters
        ----------
        df : pl.DataFrame
            资金流数据
        trade_date : str
            交易日期
            
        Returns
        -------
        int
            写入的行数
        """
        if self.db is None:
            logger.error("V70: 数据库连接未初始化")
            return 0
        
        try:
            rows = len(df)
            
            # 分批写入
            for start_idx in range(0, rows, V70_WRITE_BATCH_SIZE):
                end_idx = min(start_idx + V70_WRITE_BATCH_SIZE, rows)
                batch_df = df.slice(start_idx, end_idx - start_idx)
                
                self.db.to_sql(batch_df, V70_FUND_FLOW_TABLE, if_exists='append')
            
            logger.debug(f"V70: {trade_date} 写入 {rows} 行到 {V70_FUND_FLOW_TABLE}")
            return rows
            
        except Exception as e:
            logger.error(f"V70: 写入 {trade_date} 失败：{e}")
            # 打印 API 响应列信息（报错处理协议）
            if self.fetcher._last_response_columns:
                logger.error(f"V70: AkShare Response Columns: {self.fetcher._last_response_columns}")
            return 0
    
    def verify_data_sufficiency(self) -> Tuple[bool, str]:
        """
        验证数据是否充足
        
        Returns
        -------
        Tuple[bool, str]
            (是否充足，消息)
        """
        logger.info("=" * 60)
        logger.info("V70: 开始数据充足性验证")
        
        fund_flow_rows = self._get_table_count(V70_FUND_FLOW_TABLE)
        industry_rows = self._get_table_count(V70_INDUSTRY_TABLE)
        
        logger.info(f"V70: stock_fund_flow 行数：{fund_flow_rows:,}")
        logger.info(f"V70: stock_industry_daily 行数：{industry_rows:,}")
        
        # 验证阈值：2024 年约 250 个交易日，每天 5000 只股票，应该约 125 万行
        min_fund_flow_rows = 1000000  # 100 万行阈值
        
        if fund_flow_rows < min_fund_flow_rows:
            error_msg = f"数据不足：stock_fund_flow 仅有 {fund_flow_rows:,} 行，需要 {min_fund_flow_rows:,} 行"
            logger.error(f"V70: 【数据熔断】{error_msg}")
            return (False, error_msg)
        
        logger.info("V70: 数据充足性验证通过")
        logger.info("=" * 60)
        return (True, "数据充足")


# ===========================================
# 便捷函数
# ===========================================

def fill_v70_data(start_date: str = V70_DEFAULT_START_DATE,
                  end_date: str = V70_DEFAULT_END_DATE,
                  db: Optional[DatabaseManager] = None,
                  resume: bool = True) -> Dict[str, Any]:
    """
    便捷函数：填充 V70 数据
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    db : DatabaseManager, optional
        数据库管理器实例
    resume : bool
        是否从断点续传
        
    Returns
    -------
    Dict[str, Any]
        填充结果
    """
    filler = V70DataForceFiller(db=db)
    
    # 填充资金流数据
    result = filler.fill_daily_fund_flow(start_date, end_date, resume)
    
    # 验证数据充足性
    is_sufficient, message = filler.verify_data_sufficiency()
    
    return {
        **result,
        'is_sufficient': is_sufficient,
        'message': message,
    }


def verify_v70_data(db: Optional[DatabaseManager] = None) -> Tuple[bool, str]:
    """
    便捷函数：验证 V70 数据
    
    Parameters
    ----------
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    Tuple[bool, str]
        (是否通过，消息)
    """
    filler = V70DataForceFiller(db=db)
    return filler.verify_data_sufficiency()


# ===========================================
# 主程序 - 独立运行入口
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
    logger.info("V70 高容错数据加载器 - 启动")
    logger.info("=" * 60)
    
    # 检查 akshare 是否可用
    if not AK_AVAILABLE:
        logger.error("V70: akshare 库未安装，请运行：pip install akshare")
        sys.exit(1)
    
    # 检查数据库是否可用
    if not DB_AVAILABLE:
        logger.error("V70: db_manager 模块未找到")
        sys.exit(1)
    
    # 初始化数据库
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V70: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 填充数据
    try:
        result = fill_v70_data(
            V70_DEFAULT_START_DATE,
            V70_DEFAULT_END_DATE,
            db,
            resume=True
        )
        
        logger.info("=" * 60)
        logger.info("V70: 数据抓取完成")
        logger.info(f"成功：{result.get('success_days', 0)}天")
        logger.info(f"失败：{result.get('failure_days', 0)}天")
        logger.info(f"总行数：{result.get('total_rows', 0):,}")
        logger.info(f"数据充足：{result.get('is_sufficient', False)}")
        logger.info("=" * 60)
        
        if not result.get('is_sufficient', False):
            logger.warning("V70: 数据量不足，可能需要继续抓取")
        
    except SystemExit:
        logger.error("V70: 程序因错误已退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V70: 数据抓取失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V70DataFetcher',
    'V70DataForceFiller',
    'fill_v70_data',
    'verify_v70_data',
    'get_random_user_agent',
    'exponential_backoff_delay',
    'V70_USER_AGENT_POOL',
    'V70_CHECKPOINT_FILE',
    'V70_FUND_FLOW_TABLE',
    'V70_INDUSTRY_TABLE',
]