"""
V-Loader V2 - Data Self-Healing Loader
======================================

【核心功能】
1. 自动检查 MySQL 中四张表的数据完整性
2. 对比交易日历，发现缺失年份立即启动 Tushare 拉取
3. Self-Healing: 针对 Tushare 限频实现动态 sleep
4. 断点续传：支持失败后从中断点继续
5. 数据质量校验：自动打印各表各年份的总行数

【数据表】
- stock_daily: 股票日线行情
- index_daily: 指数日线行情
- stock_industry_daily: 股票行业分类
- stock_fund_flow: 主力资金流向

【验收红线】
- stock_daily 单年行数必须 > 800,000
- close 字段无 NULL
- 支持 Tushare 每分钟 200/500 次限频
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Tushare 配置
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
TUSHARE_API_URL = "http://api.tushare.pro"

# MySQL 配置
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:123456@localhost:3306/quantitative_trading")

# 目标年份
TARGET_YEARS = [2018, 2020, 2022]

# 限频配置
TUSHARE_RATE_LIMIT = 200  # 每分钟请求数
RATE_LIMIT_WINDOW = 60    # 秒

# 断点续传配置
CHECKPOINT_DIR = Path(".v_loader_checkpoint")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"


@dataclass
class DataStatus:
    """数据状态"""
    table_name: str
    year: int
    expected_days: int = 0
    actual_rows: int = 0
    missing: bool = True
    null_close_count: int = 0


@dataclass
class Checkpoint:
    """断点续传检查点"""
    current_table: str = ""
    current_year: int = 0
    current_symbol_index: int = 0
    completed_tables: Set[str] = field(default_factory=set)
    completed_years: Dict[str, Set[int]] = field(default_factory=dict)
    failed_symbols: List[str] = field(default_factory=list)
    last_update: str = ""
    
    def save(self):
        """保存检查点"""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            'current_table': self.current_table,
            'current_year': self.current_year,
            'current_symbol_index': self.current_symbol_index,
            'completed_tables': list(self.completed_tables),
            'completed_years': {k: list(v) for k, v in self.completed_years.items()},
            'failed_symbols': self.failed_symbols,
            'last_update': datetime.now().isoformat(),
        }
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"[Checkpoint] Saved: {self.current_table}/{self.current_year}")
    
    @classmethod
    def load(cls) -> 'Checkpoint':
        """加载检查点"""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls(
                    current_table=data.get('current_table', ''),
                    current_year=data.get('current_year', 0),
                    current_symbol_index=data.get('current_symbol_index', 0),
                    completed_tables=set(data.get('completed_tables', [])),
                    completed_years={k: set(v) for k, v in data.get('completed_years', {}).items()},
                    failed_symbols=data.get('failed_symbols', []),
                    last_update=data.get('last_update', ''),
                )
            except Exception as e:
                logger.warning(f"[Checkpoint] Load failed: {e}, starting fresh")
        return cls()
    
    def clear(self):
        """清除检查点"""
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        logger.info("[Checkpoint] Cleared")


class TushareRateLimiter:
    """Tushare 限流器"""
    
    def __init__(self, rate_limit: int = TUSHARE_RATE_LIMIT, window: int = RATE_LIMIT_WINDOW):
        self.rate_limit = rate_limit
        self.window = window
        self.requests: List[float] = []
        self.total_requests = 0
        self.wait_count = 0
    
    def wait_if_needed(self):
        """如果需要，等待以遵守限频"""
        now = time.time()
        
        # 移除窗口外的请求
        self.requests = [t for t in self.requests if now - t < self.window]
        
        # 如果达到限频，等待
        if len(self.requests) >= self.rate_limit:
            wait_time = self.window - (now - self.requests[0]) + 0.5
            if wait_time > 0:
                self.wait_count += 1
                logger.warning(f"[RateLimiter] Hit rate limit, sleeping {wait_time:.1f}s (wait #{self.wait_count})")
                time.sleep(wait_time)
                now = time.time()
                self.requests = [t for t in self.requests if now - t < self.window]
        
        self.requests.append(time.time())
        self.total_requests += 1


class TushareAPI:
    """Tushare API 客户端"""
    
    def __init__(self, token: str = TUSHARE_TOKEN):
        self.token = token
        self.rate_limiter = TushareRateLimiter()
        self.retry_count = 3
        self.retry_delay = 2
        
        if not self.token:
            logger.error("[TushareAPI] No token provided")
            raise ValueError("Tushare token is required")
        
        logger.info(f"[TushareAPI] Initialized with rate_limit={TUSHARE_RATE_LIMIT}/min")
    
    def _request(self, api_name: str, params: dict) -> Optional[pd.DataFrame]:
        """发送 API 请求"""
        import requests
        
        for attempt in range(self.retry_count):
            try:
                # 遵守限频
                self.rate_limiter.wait_if_needed()
                
                # 发送请求
                response = requests.post(
                    TUSHARE_API_URL,
                    json={"api_name": api_name, "token": self.token, "params": params},
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.warning(f"[TushareAPI] HTTP {response.status_code}, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                result = response.json()
                
                if result.get("code") != 0:
                    error_msg = result.get("msg", "Unknown error")
                    logger.warning(f"[TushareAPI] API error: {error_msg}")
                    
                    # 限频错误，延长等待
                    if "权限" in error_msg or "积分" in error_msg:
                        logger.error(f"[TushareAPI] Permission/points error: {error_msg}")
                        time.sleep(10)
                    elif "频繁" in error_msg or "limit" in error_msg.lower():
                        logger.warning(f"[TushareAPI] Rate limited, waiting 30s...")
                        time.sleep(30)
                        continue
                    
                    return None
                
                # 解析数据
                data = result.get("data", {})
                if not data or "items" not in data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data["items"], columns=data.get("fields", []))
                return df
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"[TushareAPI] Request failed (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"[TushareAPI] Unexpected error: {e}")
                return None
        
        logger.error(f"[TushareAPI] Failed after {self.retry_count} attempts")
        return None
    
    def get_trade_cal(self, exchange: str = "SSE", start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """获取交易日历"""
        params = {
            "exchange": exchange,
            "start_date": start_date,
            "end_date": end_date,
            "is_open": "1"
        }
        return self._request("trade_cal", params)
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self._request("stock_basic", {
            "fields": "ts_code,symbol,name,area,industry,list_date,status"
        })
    
    def get_daily(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """获取日线行情"""
        return self._request("daily", {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date
        })
    
    def get_index_daily(self, ts_code: str = "000905.SH", start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """获取指数日线"""
        return self._request("index_daily", {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date
        })
    
    def get_stock_industry(self, ts_code: str = "") -> pd.DataFrame:
        """获取股票行业分类"""
        return self._request("stock_industry", {
            "ts_code": ts_code,
            "src": "SW2021"
        })
    
    def get_moneyflow(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """获取资金流向"""
        return self._request("moneyflow", {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date
        })


class MySQLClient:
    """MySQL 客户端"""
    
    def __init__(self, db_url: str = DATABASE_URL):
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import QueuePool
        
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        logger.info("[MySQLClient] Connected")
    
    def check_table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        with self.engine.connect() as conn:
            result = conn.execute(text(
                f"SHOW TABLES LIKE '{table_name}'"
            ))
            return result.fetchone() is not None
    
    def get_existing_data(self, table_name: str, year: int) -> pd.DataFrame:
        """获取某年的现有数据"""
        query = text(f"SELECT * FROM {table_name} WHERE YEAR(trade_date) = :year")
        try:
            df = pd.read_sql(query, self.engine, params={"year": year})
            return df
        except Exception as e:
            logger.warning(f"[MySQL] Failed to get data for {table_name}/{year}: {e}")
            return pd.DataFrame()
    
    def get_row_count(self, table_name: str, year: int) -> int:
        """获取某年的行数"""
        from sqlalchemy import text
        query = text(f"SELECT COUNT(*) as cnt FROM {table_name} WHERE YEAR(trade_date) = :year")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"year": year})
                row = result.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.warning(f"[MySQL] Failed to count rows: {e}")
            return 0
    
    def get_null_close_count(self, table_name: str, year: int) -> int:
        """获取 close 字段为 NULL 的行数"""
        from sqlalchemy import text
        
        # 只有 stock_daily 和 index_daily 有 close 字段
        tables_with_close = {'stock_daily', 'index_daily'}
        if table_name not in tables_with_close:
            return 0
        
        query = text(f"SELECT COUNT(*) as cnt FROM {table_name} WHERE YEAR(trade_date) = :year AND (close IS NULL OR close = '')")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"year": year})
                row = result.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.warning(f"[MySQL] Failed to count null close: {e}")
            return 0
    
    def insert_data(self, table_name: str, df: pd.DataFrame):
        """插入数据"""
        if df.empty:
            return
        
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000
            )
            logger.debug(f"[MySQL] Inserted {len(df)} rows to {table_name}")
        except Exception as e:
            logger.error(f"[MySQL] Insert failed: {e}")
            raise
    
    def get_distinct_years(self, table_name: str) -> List[int]:
        """获取表中已有的年份"""
        query = text(f"SELECT DISTINCT YEAR(trade_date) as year FROM {table_name} ORDER BY year")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"[MySQL] Failed to get years: {e}")
            return []


class VLoaderV2:
    """V-Loader V2 - 数据自愈拉取器"""
    
    # 表配置
    TABLE_CONFIG = {
        'stock_daily': {
            'ts_code_col': 'ts_code',
            'date_col': 'trade_date',
            'required_cols': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
            'min_rows_per_year': 800000,
        },
        'index_daily': {
            'ts_code_col': 'ts_code',
            'date_col': 'trade_date',
            'required_cols': ['ts_code', 'trade_date', 'close', 'open', 'high', 'low'],
            'min_rows_per_year': 200,  # 指数行数较少
        },
        'stock_industry_daily': {
            'ts_code_col': 'ts_code',
            'date_col': 'trade_date',
            'required_cols': ['ts_code', 'trade_date', 'industry_name', 'industry_code'],
            'min_rows_per_year': 500000,
        },
        'stock_fund_flow': {
            'ts_code_col': 'ts_code',
            'date_col': 'trade_date',
            'required_cols': ['ts_code', 'trade_date', 'net_main_amount', 'net_main_rate'],
            'min_rows_per_year': 500000,
        },
    }
    
    def __init__(self, target_years: List[int] = None):
        self.target_years = target_years or TARGET_YEARS
        self.api = TushareAPI()
        self.db = MySQLClient()
        self.checkpoint = Checkpoint.load()
        self.stock_list: Optional[pd.DataFrame] = None
        
        logger.info(f"[VLoaderV2] Initialized for years {self.target_years}")
    
    def load_stock_list(self) -> pd.DataFrame:
        """加载股票列表"""
        if self.stock_list is not None:
            return self.stock_list
        
        logger.info("[StockList] Loading stock list...")
        df = self.api.get_stock_list()
        
        if df is not None and not df.empty:
            # 只保留正常交易的股票
            df = df[df.get('status', 'L') == 'L']  # 正常上市
            self.stock_list = df
            logger.info(f"[StockList] Loaded {len(df)} stocks")
        else:
            # 使用备用列表
            logger.warning("[StockList] Using fallback stock list")
            self.stock_list = pd.DataFrame({
                'ts_code': [f"00000{i}.SZ" for i in range(1, 100)] + 
                          [f"60000{i}.SH" for i in range(1, 100)]
            })
        
        return self.stock_list
    
    def get_trade_days(self, year: int) -> List[str]:
        """获取某年的交易日"""
        logger.info(f"[TradeCal] Getting trading days for {year}...")
        
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        df = self.api.get_trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)
        
        if df is not None and not df.empty:
            trade_days = df['cal_date'].tolist()
            logger.info(f"[TradeCal] Found {len(trade_days)} trading days in {year}")
            return trade_days
        
        # 备用方案：使用 pandas 生成工作日
        logger.warning("[TradeCal] Using fallback trading days")
        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='B')
        return [d.strftime('%Y%m%d') for d in dates]
    
    def check_data_status(self, table_name: str, year: int) -> DataStatus:
        """检查数据状态"""
        status = DataStatus(table_name=table_name, year=year)
        
        # 计算期望天数
        trade_days = self.get_trade_days(year)
        status.expected_days = len(trade_days)
        
        # 获取实际行数
        status.actual_rows = self.db.get_row_count(table_name, year)
        
        # 检查 NULL
        status.null_close_count = self.db.get_null_close_count(table_name, year)
        
        # 判断是否缺失
        config = self.TABLE_CONFIG.get(table_name, {})
        min_rows = config.get('min_rows_per_year', 100000)
        status.missing = status.actual_rows < min_rows
        
        return status
    
    def fetch_stock_daily(self, year: int, symbols: List[str] = None) -> pd.DataFrame:
        """拉取股票日线数据"""
        logger.info(f"[Fetch] stock_daily for {year}...")
        
        if symbols is None:
            stock_list = self.load_stock_list()
            symbols = stock_list['ts_code'].tolist()
        
        all_data = []
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        for idx, ts_code in enumerate(symbols):
            try:
                df = self.api.get_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                
                if (idx + 1) % 100 == 0:
                    logger.debug(f"[Fetch] Progress: {idx + 1}/{len(symbols)}")
                    
            except Exception as e:
                logger.warning(f"[Fetch] Failed for {ts_code}: {e}")
                self.checkpoint.failed_symbols.append(ts_code)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"[Fetch] stock_daily: {len(result)} rows")
            return result
        
        return pd.DataFrame()
    
    def fetch_index_daily(self, year: int) -> pd.DataFrame:
        """拉取指数日线数据"""
        logger.info(f"[Fetch] index_daily for {year}...")
        
        indices = ['000905.SH', '000001.SH', '399006.SZ']  # 中证 500, 上证指数，创业板指
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        all_data = []
        for ts_code in indices:
            df = self.api.get_index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"[Fetch] index_daily: {len(result)} rows")
            return result
        
        return pd.DataFrame()
    
    def fetch_stock_industry(self, year: int, symbols: List[str] = None) -> pd.DataFrame:
        """拉取股票行业分类"""
        logger.info(f"[Fetch] stock_industry_daily for {year}...")
        
        if symbols is None:
            stock_list = self.load_stock_list()
            symbols = stock_list['ts_code'].tolist()[:500]  # 限流，先拉 500 只
        
        all_data = []
        trade_date = f"{year}1231"  # 使用年末数据
        
        for idx, ts_code in enumerate(symbols[:500]):
            try:
                df = self.api.get_stock_industry(ts_code=ts_code)
                
                if df is not None and not df.empty:
                    df['trade_date'] = trade_date
                    all_data.append(df)
                
                if (idx + 1) % 50 == 0:
                    logger.debug(f"[Fetch] Industry progress: {idx + 1}/{min(len(symbols), 500)}")
                    
            except Exception as e:
                logger.warning(f"[Fetch] Industry failed for {ts_code}: {e}")
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"[Fetch] stock_industry_daily: {len(result)} rows")
            return result
        
        return pd.DataFrame()
    
    def fetch_stock_fund_flow(self, year: int, symbols: List[str] = None) -> pd.DataFrame:
        """拉取资金流向数据"""
        logger.info(f"[Fetch] stock_fund_flow for {year}...")
        
        if symbols is None:
            stock_list = self.load_stock_list()
            symbols = stock_list['ts_code'].tolist()[:500]  # 限流
        
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        all_data = []
        
        for idx, ts_code in enumerate(symbols[:500]):
            try:
                df = self.api.get_moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                
                if (idx + 1) % 50 == 0:
                    logger.debug(f"[Fetch] Flow progress: {idx + 1}/{min(len(symbols), 500)}")
                    
            except Exception as e:
                logger.warning(f"[Fetch] Flow failed for {ts_code}: {e}")
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"[Fetch] stock_fund_flow: {len(result)} rows")
            return result
        
        return pd.DataFrame()
    
    def transform_stock_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换 stock_daily 格式"""
        if df.empty:
            return df
        
        result = df.copy()
        
        # 重命名列
        rename_map = {
            'trade_date': 'trade_date',
            'ts_code': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'pct_chg': 'pct_chg',
        }
        
        result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})
        
        # 确保必要列存在
        required = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        for col in required:
            if col not in result.columns:
                result[col] = 0
        
        # 日期格式转换
        result['trade_date'] = pd.to_datetime(result['trade_date']).dt.strftime('%Y%m%d').astype(int)
        
        # 符号格式转换
        result['symbol'] = result['symbol'].astype(str)
        
        return result[required]
    
    def transform_index_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换 index_daily 格式"""
        if df.empty:
            return df
        
        result = df.copy()
        
        required = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pct_chg']
        
        # 日期格式转换
        if 'trade_date' in result.columns:
            result['trade_date'] = pd.to_datetime(result['trade_date']).dt.strftime('%Y%m%d').astype(int)
        
        return result
    
    def transform_stock_industry(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换 stock_industry 格式"""
        if df.empty:
            return df
        
        result = df.copy()
        
        rename_map = {
            'ts_code': 'symbol',
            'industry_name': 'industry_name',
            'industry_code': 'industry_code',
        }
        
        result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})
        
        # 日期格式转换
        if 'trade_date' in result.columns:
            result['trade_date'] = pd.to_datetime(result['trade_date']).dt.strftime('%Y%m%d').astype(int)
        
        required = ['symbol', 'trade_date', 'industry_name', 'industry_code']
        for col in required:
            if col not in result.columns:
                result[col] = ''
        
        return result[required]
    
    def transform_stock_fund_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换 stock_fund_flow 格式"""
        if df.empty:
            return df
        
        result = df.copy()
        
        rename_map = {
            'ts_code': 'symbol',
            'buy_sm_amount': 'net_main_amount',
            'buy_sm_rate': 'net_main_rate',
        }
        
        result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})
        
        # 日期格式转换
        if 'trade_date' in result.columns:
            result['trade_date'] = pd.to_datetime(result['trade_date']).dt.strftime('%Y%m%d').astype(int)
        
        # 符号格式
        result['symbol'] = result['symbol'].astype(str)
        
        required = ['symbol', 'trade_date', 'net_main_amount', 'net_main_rate']
        for col in required:
            if col not in result.columns:
                result[col] = 0
        
        return result[required]
    
    def fetch_table_data(self, table_name: str, year: int) -> Optional[pd.DataFrame]:
        """拉取某表某年的数据"""
        logger.info(f"[Fetch] Starting {table_name} for {year}...")
        
        try:
            if table_name == 'stock_daily':
                raw = self.fetch_stock_daily(year)
                result = self.transform_stock_daily(raw)
                
            elif table_name == 'index_daily':
                raw = self.fetch_index_daily(year)
                result = self.transform_index_daily(raw)
                
            elif table_name == 'stock_industry_daily':
                raw = self.fetch_stock_industry(year)
                result = self.transform_stock_industry(raw)
                
            elif table_name == 'stock_fund_flow':
                raw = self.fetch_stock_fund_flow(year)
                result = self.transform_stock_fund_flow(raw)
                
            else:
                logger.error(f"[Fetch] Unknown table: {table_name}")
                return None
            
            if result is not None and not result.empty:
                logger.info(f"[Fetch] {table_name}/{year}: {len(result)} rows fetched")
                return result
            
            logger.warning(f"[Fetch] {table_name}/{year}: No data returned")
            return None
            
        except Exception as e:
            logger.error(f"[Fetch] {table_name}/{year} failed: {e}")
            return None
    
    def save_data(self, table_name: str, df: pd.DataFrame):
        """保存数据到 MySQL"""
        if df is None or df.empty:
            logger.warning(f"[Save] {table_name}: No data to save")
            return
        
        try:
            self.db.insert_data(table_name, df)
            logger.info(f"[Save] {table_name}: Saved {len(df)} rows")
        except Exception as e:
            logger.error(f"[Save] {table_name} failed: {e}")
            raise
    
    def run_full_check(self):
        """运行完整检查"""
        logger.info("=" * 80)
        logger.info("V-Loader V2 - Data Self-Healing")
        logger.info("=" * 80)
        logger.info(f"Target Years: {self.target_years}")
        logger.info(f"Tables: {list(self.TABLE_CONFIG.keys())}")
        logger.info("=" * 80)
        
        # 检查已有数据
        logger.info("\n[Check] Scanning existing data...")
        
        tables_to_fetch = []
        
        for table_name in self.TABLE_CONFIG.keys():
            logger.info(f"\n[Table] {table_name}")
            
            for year in self.target_years:
                status = self.check_data_status(table_name, year)
                
                config = self.TABLE_CONFIG[table_name]
                min_rows = config.get('min_rows_per_year', 100000)
                
                status_str = f"  {year}: {status.actual_rows:,} rows"
                status_str += f" (need {min_rows:,})"
                
                if status.missing:
                    status_str += " [MISSING]"
                    tables_to_fetch.append((table_name, year))
                else:
                    status_str += " [OK]"
                
                if status.null_close_count > 0:
                    status_str += f" [NULL_CLOSE: {status.null_close_count}]"
                
                logger.info(status_str)
        
        if not tables_to_fetch:
            logger.info("\n[Check] All data is complete!")
            self.print_summary()
            return
        
        logger.info(f"\n[Plan] Need to fetch {len(tables_to_fetch)} tasks:")
        for table_name, year in tables_to_fetch:
            logger.info(f"  - {table_name} ({year})")
        
        # 执行拉取
        logger.info("\n[Fetch] Starting data fetch...")
        
        for table_name, year in tables_to_fetch:
            self.checkpoint.current_table = table_name
            self.checkpoint.current_year = year
            self.checkpoint.save()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[Task] {table_name} - {year}")
            logger.info(f"{'='*60}")
            
            # 检查是否已完成
            if table_name not in self.checkpoint.completed_years:
                self.checkpoint.completed_years[table_name] = set()
            
            if year in self.checkpoint.completed_years[table_name]:
                logger.info(f"[Skip] {table_name}/{year} already completed")
                continue
            
            # 拉取数据
            df = self.fetch_table_data(table_name, year)
            
            if df is not None and not df.empty:
                # 保存数据
                self.save_data(table_name, df)
                
                # 标记完成
                self.checkpoint.completed_years[table_name].add(year)
                self.checkpoint.save()
                
                # 验证
                status = self.check_data_status(table_name, year)
                if not status.missing:
                    logger.info(f"[OK] {table_name}/{year} completed successfully")
                else:
                    logger.warning(f"[WARN] {table_name}/{year} may still be incomplete")
            else:
                logger.error(f"[FAIL] {table_name}/{year} fetch failed")
        
        # 清除检查点
        self.checkpoint.clear()
        
        # 打印摘要
        self.print_summary()
    
    def print_summary(self):
        """打印数据摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("V-Loader V2 - Data Summary")
        logger.info("=" * 80)
        
        for table_name in self.TABLE_CONFIG.keys():
            logger.info(f"\n[Table] {table_name}")
            config = self.TABLE_CONFIG[table_name]
            min_rows = config.get('min_rows_per_year', 100000)
            
            for year in self.target_years:
                row_count = self.db.get_row_count(table_name, year)
                null_count = self.db.get_null_close_count(table_name, year)
                
                status = "✓" if row_count >= min_rows else "✗"
                null_status = "" if null_count == 0 else f" [NULL: {null_count}]"
                
                logger.info(f"  {year}: {row_count:,} rows {status}{null_status}")
        
        logger.info("\n" + "=" * 80)
        logger.info("V-Loader V2 Complete")
        logger.info("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V-Loader V2 - Data Self-Healing")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Target years (default: 2018, 2020, 2022)"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear checkpoint and start fresh"
    )
    
    args = parser.parse_args()
    
    # 清除检查点
    if args.clear_checkpoint:
        Checkpoint().clear()
    
    # 运行
    years = args.years if args.years else TARGET_YEARS
    loader = VLoaderV2(target_years=years)
    loader.run_full_check()


if __name__ == "__main__":
    main()