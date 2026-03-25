"""
V64 Data Fetcher Module - 自动化数据工程

【V64 数据工程核心功能】
1. 拉取申万一级行业分类并存入 stock_industry
2. 拉取过去一年的个股资金流向数据并存入 stock_fund_flow
3. 断线重连和频率控制（每秒不超过 2 次请求）

作者：量化系统
版本：V64.0
日期：2026-03-24
"""

import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import polars as pl
from loguru import logger

try:
    import akshare as ak
except ImportError:
    logger.error("请安装 akshare: pip install akshare")
    raise


# ===========================================
# V64 配置常量
# ===========================================

V64_MAX_RETRIES = 3  # 最大重试次数
V64_RETRY_DELAY = 2.0  # 重试延迟（秒）
V64_RATE_LIMIT_DELAY = 0.6  # 请求间隔（秒），确保每秒不超过 2 次
V64_DEFAULT_CHUNK_SIZE = 100  # 批量处理 chunk 大小


# ===========================================
# V64 工具函数
# ===========================================

def rate_limit_sleep(last_request_time: float) -> float:
    """
    计算需要等待的时间以满足频率限制
    
    Parameters
    ----------
    last_request_time : float
        上次请求的时间戳
    
    Returns
    -------
    float
        需要等待的秒数
    """
    current_time = time.time()
    elapsed = current_time - last_request_time
    
    if elapsed < V64_RATE_LIMIT_DELAY:
        return V64_RATE_LIMIT_DELAY - elapsed
    return 0.0


def retry_with_backoff(func, *args, max_retries: int = V64_MAX_RETRIES, **kwargs):
    """
    带退避的重试装饰器
    
    Parameters
    ----------
    func : callable
        要执行的函数
    max_retries : int
        最大重试次数
    args : tuple
        位置参数
    kwargs : dict
        关键字参数
    
    Returns
    -------
    Any
        函数执行结果
    
    Raises
    ------
    Exception
        如果所有重试都失败
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = V64_RETRY_DELAY * (2 ** attempt)  # 指数退避
                logger.warning(f"请求失败，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
            else:
                logger.error(f"请求最终失败 (已重试 {max_retries}次): {e}")
    
    raise last_exception


# ===========================================
# V64 行业数据拉取器
# ===========================================

class V64IndustryFetcher:
    """
    V64 行业数据拉取器 - 申万一级行业分类
    
    【核心功能】
    1. 拉取申万一级行业成分股
    2. 拉取行业指数行情
    3. 数据格式化与存储
    """
    
    def __init__(self, db=None):
        """
        初始化行业数据拉取器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        self.db = db
        self.last_request_time = 0.0
    
    def _wait_rate_limit(self):
        """等待以满足频率限制"""
        wait_time = rate_limit_sleep(self.last_request_time)
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def fetch_sw_industry_constituents(self) -> pl.DataFrame:
        """
        拉取申万一级行业成分股
        
        Returns
        -------
        pl.DataFrame
            行业成分股数据，包含列：
            - symbol: 股票代码
            - industry_name: 行业名称
            - industry_code: 行业代码
        """
        logger.info("V64: 开始拉取申万一级行业成分股...")
        
        def _fetch():
            self._wait_rate_limit()
            df = ak.stock_board_industry_cons_sw()
            return df
        
        try:
            df = retry_with_backoff(_fetch)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                logger.warning("V64: 申万行业成分股数据为空")
                return self._empty_industry_df()
            
            result = self._format_constituents(df)
            logger.info(f"V64: 成功拉取 {len(result)} 条行业成分股数据")
            return result
            
        except Exception as e:
            logger.error(f"V64: 拉取行业成分股失败：{e}")
            logger.error(traceback.format_exc())
            return self._empty_industry_df()
    
    def _empty_industry_df(self) -> pl.DataFrame:
        """返回空 schema 的 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'industry_name': pl.Utf8,
            'industry_code': pl.Utf8
        })
    
    def _format_constituents(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        格式化行业成分股数据
        
        Parameters
        ----------
        df : pl.DataFrame
            原始数据
        
        Returns
        -------
        pl.DataFrame
            格式化后的数据
        """
        result = df.clone()
        
        column_mapping = {
            '股票代码': 'symbol',
            '股票名称': 'stock_name',
            '所属行业': 'industry_name',
            '行业代码': 'industry_code',
            '申万一级': 'industry_name',
            '申万一级代码': 'industry_code',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result = result.rename({old_col: new_col})
        
        if 'symbol' not in result.columns:
            for col in ['股票代码', 'code', 'symbol_code']:
                if col in result.columns:
                    result = result.rename({col: 'symbol'})
                    break
        
        if 'industry_name' not in result.columns:
            result = result.with_columns(pl.lit('Unknown').alias('industry_name'))
        
        if 'industry_code' not in result.columns:
            result = result.with_columns(
                pl.col('industry_name').hash().cast(pl.Utf8).alias('industry_code')
            )
        
        select_cols = ['symbol', 'industry_name', 'industry_code']
        available_cols = [c for c in select_cols if c in result.columns]
        
        result = result.select(available_cols)
        
        if 'symbol' in result.columns:
            result = result.with_columns([
                pl.col('symbol').cast(pl.Utf8).str.strip_chars().alias('symbol')
            ])
        
        return result
    
    def save_to_db(self, df: pl.DataFrame, table_name: str = "stock_industry"):
        """
        保存数据到数据库
        
        Parameters
        ----------
        df : pl.DataFrame
            要保存的数据
        table_name : str
            表名
        """
        if self.db is None:
            logger.error("V64: 数据库连接未初始化")
            return
        
        if df.is_empty():
            logger.warning("V64: 数据为空，跳过保存")
            return
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            df = df.with_columns(pl.lit(today).alias('trade_date'))
            self.db.to_sql(df, table_name, if_exists='append')
            logger.info(f"V64: 成功保存 {len(df)} 条数据到 {table_name}")
            
        except Exception as e:
            logger.error(f"V64: 保存数据失败：{e}")


# ===========================================
# V64 资金流数据拉取器
# ===========================================

class V64FundFlowFetcher:
    """
    V64 资金流数据拉取器 - 个股资金流向
    
    【核心功能】
    1. 拉取个股资金流向数据（主力净流入、超大单、大单等）
    2. 支持批量拉取和历史数据
    3. 断线重连和频率控制
    """
    
    def __init__(self, db=None):
        """
        初始化资金流数据拉取器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        self.db = db
        self.last_request_time = 0.0
    
    def _wait_rate_limit(self):
        """等待以满足频率限制"""
        wait_time = rate_limit_sleep(self.last_request_time)
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def fetch_stock_fund_flow(self, symbol: str, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pl.DataFrame:
        """
        拉取单只股票的资金流向数据
        
        Parameters
        ----------
        symbol : str
            股票代码，格式如 000001.SZ
        start_date : str, optional
            开始日期，默认拉取过去一年
        end_date : str, optional
            结束日期，默认到今天
        
        Returns
        -------
        pl.DataFrame
            资金流向数据
        """
        logger.info(f"V64: 开始拉取 {symbol} 资金流向数据...")
        
        if start_date is None:
            end_date_obj = datetime.now()
            start_date = (end_date_obj - timedelta(days=365)).strftime("%Y-%m-%d")
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        def _fetch():
            self._wait_rate_limit()
            df = ak.stock_individual_fund_flow(symbol=symbol, 
                                               start_date=start_date,
                                               end_date=end_date)
            return df
        
        try:
            df = retry_with_backoff(_fetch)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                logger.warning(f"V64: {symbol} 资金流向数据为空")
                return self._empty_fund_flow_df()
            
            result = self._format_fund_flow(df, symbol)
            logger.info(f"V64: 成功拉取 {symbol} {len(result)} 条资金流向数据")
            return result
            
        except Exception as e:
            logger.error(f"V64: 拉取 {symbol} 资金流向失败：{e}")
            logger.error(traceback.format_exc())
            return self._empty_fund_flow_df()
    
    def _empty_fund_flow_df(self) -> pl.DataFrame:
        """返回空 schema 的 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'net_main_amount': pl.Float64,
            'net_super_amount': pl.Float64,
            'net_large_amount': pl.Float64,
            'net_medium_amount': pl.Float64,
            'net_small_amount': pl.Float64,
            'close': pl.Float64,
            'change_percent': pl.Float64
        })
    
    def _format_fund_flow(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        格式化资金流向数据
        """
        result = df.clone()
        
        column_mapping = {
            '日期': 'trade_date',
            'Date': 'trade_date',
            '收盘价': 'close',
            '涨跌幅': 'change_percent',
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
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result = result.rename({old_col: new_col})
        
        if 'symbol' not in result.columns:
            result = result.with_columns(pl.lit(symbol).alias('symbol'))
        
        required_cols = ['symbol', 'trade_date']
        for col in required_cols:
            if col not in result.columns:
                if col == 'symbol':
                    result = result.with_columns(pl.lit(symbol).alias(col))
                elif col == 'trade_date':
                    result = result.with_columns(pl.lit('').alias(col))
        
        numeric_cols = ['net_main_amount', 'net_super_amount', 'net_large_amount',
                        'net_medium_amount', 'net_small_amount', 'close', 'change_percent']
        
        for col in numeric_cols:
            if col in result.columns:
                result = result.with_columns([
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                ])
            else:
                result = result.with_columns(pl.lit(0.0).alias(col))
        
        if 'trade_date' in result.columns:
            result = result.with_columns([
                pl.col('trade_date').cast(pl.Utf8).str.strip_chars().alias('trade_date')
            ])
        
        select_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_super_amount',
                       'net_large_amount', 'net_medium_amount', 'net_small_amount',
                       'close', 'change_percent']
        
        available_cols = [c for c in select_cols if c in result.columns]
        result = result.select(available_cols)
        
        return result
    
    def fetch_batch_fund_flow(self, symbols: List[str],
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               chunk_size: int = V64_DEFAULT_CHUNK_SIZE) -> pl.DataFrame:
        """
        批量拉取多只股票的资金流向数据
        """
        logger.info(f"V64: 开始批量拉取 {len(symbols)} 只股票的资金流向数据...")
        
        all_data = []
        processed = 0
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            
            for symbol in chunk:
                try:
                    df = self.fetch_stock_fund_flow(symbol, start_date, end_date)
                    
                    if not df.is_empty():
                        all_data.append(df)
                    
                    processed += 1
                    
                    if processed % 50 == 0:
                        logger.info(f"V64: 进度 {processed}/{len(symbols)}")
                    
                except Exception as e:
                    logger.error(f"V64: 拉取 {symbol} 失败：{e}")
                    continue
        
        if all_data:
            result = pl.concat(all_data, how='vertical_relaxed')
            logger.info(f"V64: 批量拉取完成，共 {len(result)} 条数据")
            return result
        
        logger.warning("V64: 批量拉取未获得任何数据")
        return self._empty_fund_flow_df()
    
    def save_to_db(self, df: pl.DataFrame, table_name: str = "stock_fund_flow"):
        """
        保存资金流向数据到数据库
        """
        if self.db is None:
            logger.error("V64: 数据库连接未初始化")
            return
        
        if df.is_empty():
            logger.warning("V64: 数据为空，跳过保存")
            return
        
        try:
            self.db.to_sql(df, table_name, if_exists='append')
            logger.info(f"V64: 成功保存 {len(df)} 条资金流向数据到 {table_name}")
            
        except Exception as e:
            logger.error(f"V64: 保存数据失败：{e}")


# ===========================================
# V64 综合数据拉取器
# ===========================================

class V64DataFetcher:
    """
    V64 综合数据拉取器
    
    【核心功能】
    1. 统一入口拉取所有 V64 所需数据
    2. 行业分类数据
    3. 资金流向数据
    """
    
    def __init__(self, db=None):
        """
        初始化综合数据拉取器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        self.db = db
        self.industry_fetcher = V64IndustryFetcher(db=db)
        self.fund_flow_fetcher = V64FundFlowFetcher(db=db)
    
    def fetch_all_data(self, symbols: Optional[List[str]] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        """
        拉取所有 V64 所需数据
        
        Parameters
        ----------
        symbols : List[str], optional
            股票代码列表，None 表示全部
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期
        
        Returns
        -------
        Dict[str, pl.DataFrame]
            数据字典，包含：
            - industry: 行业分类数据
            - fund_flow: 资金流向数据
        """
        logger.info("=" * 60)
        logger.info("V64: 开始拉取所有数据")
        logger.info("=" * 60)
        
        results = {}
        
        logger.info("V64: 拉取行业分类数据...")
        industry_data = self.industry_fetcher.fetch_sw_industry_constituents()
        results['industry'] = industry_data
        
        if symbols is None and not industry_data.is_empty():
            symbols = industry_data['symbol'].unique().to_list()
            logger.info(f"V64: 从行业数据中获取 {len(symbols)} 只股票")
        
        if symbols:
            logger.info(f"V64: 拉取 {len(symbols)} 只股票的资金流向数据...")
            fund_flow_data = self.fund_flow_fetcher.fetch_batch_fund_flow(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            results['fund_flow'] = fund_flow_data
        else:
            results['fund_flow'] = self.fund_flow_fetcher._empty_fund_flow_df()
        
        logger.info("=" * 60)
        logger.info("V64: 数据拉取完成")
        logger.info(f"  - 行业分类：{len(results['industry'])} 条")
        logger.info(f"  - 资金流向：{len(results['fund_flow'])} 条")
        logger.info("=" * 60)
        
        return results
    
    def save_all_to_db(self, data: Dict[str, pl.DataFrame]):
        """
        保存所有数据到数据库
        
        Parameters
        ----------
        data : Dict[str, pl.DataFrame]
            数据字典
        """
        logger.info("V64: 开始保存数据到数据库...")
        
        if 'industry' in data and not data['industry'].is_empty():
            self.industry_fetcher.save_to_db(data['industry'], table_name='stock_industry')
        
        if 'fund_flow' in data and not data['fund_flow'].is_empty():
            self.fund_flow_fetcher.save_to_db(data['fund_flow'], table_name='stock_fund_flow')
        
        logger.info("V64: 数据保存完成")


def run_data_fetcher(db=None, symbols: Optional[List[str]] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, pl.DataFrame]:
    """
    便捷函数：运行 V64 数据拉取
    
    Parameters
    ----------
    db : DatabaseManager, optional
        数据库管理器实例
    symbols : List[str], optional
        股票代码列表
    start_date : str, optional
        开始日期
    end_date : str, optional
        结束日期
    
    Returns
    -------
    Dict[str, pl.DataFrame]
        数据字典
    """
    fetcher = V64DataFetcher(db=db)
    data = fetcher.fetch_all_data(symbols=symbols, start_date=start_date, end_date=end_date)
    fetcher.save_all_to_db(data)
    return data


__all__ = [
    'V64IndustryFetcher',
    'V64FundFlowFetcher',
    'V64DataFetcher',
    'run_data_fetcher',
    'V64_MAX_RETRIES',
    'V64_RETRY_DELAY',
    'V64_RATE_LIMIT_DELAY',
    'V64_DEFAULT_CHUNK_SIZE',
]