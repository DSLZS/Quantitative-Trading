"""
Data Loader Module - V110 Enhanced with Auto-SQL-Healer.

负责数据拉取、补全和校验。
核心功能:
    - 从 Tushare API 获取 A 股股票和基金数据
    - 资产类型自动识别 (股票/基金)
    - 获取日线数据和复权因子
    - 频率限制控制
    - 数据完整性校验
    - 2024 年 total_mv 数据防御检查
    - V110 新增：自动 SQL 补全逻辑 (从原始库补全缺失字段)

【V110 数据自愈增强】
- 检测字段缺失时，自动从 SQL 原始库补全
- 严禁直接用均值填充或跳过运行
- 必须记录 [V110][DataAudit] 日志
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import tushare as ts
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# Load environment variables
load_dotenv()


class DataLoaderError(Exception):
    """Data loader 自定义异常"""
    pass


class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告"""
    pass


class SQLAutoHealer:
    """
    【V110 核心】SQL 自动补全引擎。
    
    【职责】
    1. 检测 DataFrame 中缺失的字段
    2. 自动从 MySQL 原始库查询补全
    3. 记录 [V110][DataAudit] 日志
    
    【补全策略】
    - total_mv 缺失：从 stock_daily_basic 表补全
    - turnover_rate 缺失：从 stock_daily_basic 表补全
    - industry_code 缺失：从 stock_info 表补全
    - vwap 缺失：从 stock_daily 表计算 (high+low+close)/3
    - amount 缺失：从 stock_daily 表补全
    """
    
    def __init__(self, db_url: Optional[str] = None) -> None:
        """
        初始化 SQL 自动补全引擎。
        
        Args:
            db_url: 数据库连接 URL
        """
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.engine = None
        self.audit_log = []
        
        if self.db_url:
            try:
                self.engine = create_engine(
                    self.db_url,
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
                logger.info("[V110][SQLAutoHealer] Database connection pool initialized")
            except Exception as e:
                logger.warning(f"[V110][SQLAutoHealer] Failed to create connection pool: {e}")
                self.engine = None
    
    def _log_audit(self, action: str, column: str, status: str, details: str = "") -> None:
        """记录审计日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details,
        }
        self.audit_log.append(log_entry)
        logger.info(f"[V110][DataAudit] {action} - Column: {column}, Status: {status}, {details}")
    
    def check_missing_columns(self, df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        """
        检查缺失的列。
        
        Args:
            df: 输入 DataFrame
            required_columns: 必需的列名列表
            
        Returns:
            缺失的列名列表
        """
        actual_columns = set(df.columns) if not df.empty else set()
        missing = [col for col in required_columns if col not in actual_columns]
        
        if missing:
            self._log_audit(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns from required {len(required_columns)}"
            )
        
        return missing
    
    def heal_from_sql(self, df: pd.DataFrame, symbols: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        【核心方法】从 SQL 补全缺失字段。
        
        Args:
            df: 原始 DataFrame
            symbols: 股票代码列表
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            补全后的 DataFrame
        """
        if not self.engine:
            logger.warning("[V110][SQLAutoHealer] No database connection, cannot heal from SQL")
            self._log_audit(
                action="SQLHealAttempt",
                column="ALL",
                status="FAILED",
                details="No database connection"
            )
            return df
        
        result = df.copy()
        missing_columns = self.check_missing_columns(
            result,
            ['total_mv', 'turnover_rate', 'industry_code', 'vwap', 'amount', 'pe_ttm', 'pb']
        )
        
        if not missing_columns:
            self._log_audit(
                action="SQLHealCheck",
                column="ALL",
                status="OK",
                details="No missing columns detected"
            )
            return result
        
        logger.info(f"[V110][SQLAutoHealer] Attempting to heal {len(missing_columns)} columns from SQL...")
        
        # 按缺失列类型分组处理
        basic_columns = ['total_mv', 'turnover_rate', 'pe_ttm', 'pb']
        info_columns = ['industry_code']
        computed_columns = ['vwap']
        amount_columns = ['amount']
        
        # 1. 从 stock_daily_basic 补全
        need_basic = [col for col in missing_columns if col in basic_columns]
        if need_basic and symbols:
            healed_basic = self._heal_from_daily_basic(symbols, start_date, end_date, need_basic)
            if healed_basic is not None and not healed_basic.empty:
                for col in need_basic:
                    if col in healed_basic.columns:
                        result = self._merge_column(result, healed_basic, col, ['symbol', 'trade_date'])
                        self._log_audit(
                            action="HealedFromSQL",
                            column=col,
                            status="SUCCESS",
                            details=f"From stock_daily_basic, {len(healed_basic)} rows"
                        )
        
        # 2. 从 stock_info 补全 industry_code
        need_info = [col for col in missing_columns if col in info_columns]
        if need_info and symbols:
            healed_info = self._heal_stock_info(symbols, need_info)
            if healed_info is not None and not healed_info.empty:
                for col in need_info:
                    if col in healed_info.columns:
                        result = self._merge_column(result, healed_info, col, ['symbol'])
                        self._log_audit(
                            action="HealedFromSQL",
                            column=col,
                            status="SUCCESS",
                            details=f"From stock_info, {len(healed_info)} stocks"
                        )
        
        # 3. 计算 vwap
        need_computed = [col for col in missing_columns if col in computed_columns]
        if need_computed:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
                self._log_audit(
                    action="HealedComputed",
                    column='vwap',
                    status="SUCCESS",
                    details="Computed as (high + low + close) / 3"
                )
        
        # 4. 从 stock_daily 补全 amount
        need_amount = [col for col in missing_columns if col in amount_columns]
        if need_amount and symbols:
            healed_amount = self._heal_from_daily(symbols, start_date, end_date, need_amount)
            if healed_amount is not None and not healed_amount.empty:
                for col in need_amount:
                    if col in healed_amount.columns:
                        result = self._merge_column(result, healed_amount, col, ['symbol', 'trade_date'])
                        self._log_audit(
                            action="HealedFromSQL",
                            column=col,
                            status="SUCCESS",
                            details=f"From stock_daily, {len(healed_amount)} rows"
                        )
        
        # 统计最终结果
        still_missing = self.check_missing_columns(result, missing_columns)
        if still_missing:
            self._log_audit(
                action="SQLHealPartial",
                column=", ".join(still_missing),
                status="PARTIAL",
                details=f"Still missing {len(still_missing)} columns after SQL heal"
            )
        else:
            self._log_audit(
                action="SQLHealComplete",
                column="ALL",
                status="SUCCESS",
                details=f"All {len(missing_columns)} columns healed"
            )
        
        return result
    
    def _merge_column(self, df: pd.DataFrame, source_df: pd.DataFrame,
                      column: str, keys: List[str]) -> pd.DataFrame:
        """
        将 source_df 的列合并到 df 中。
        
        Args:
            df: 目标 DataFrame
            source_df: 源 DataFrame
            column: 要合并的列名
            keys: 合并键
            
        Returns:
            合并后的 DataFrame
        """
        if column not in source_df.columns:
            return df
        
        result = df.copy()
        
        # 如果列不存在，直接添加
        if column not in result.columns:
            # 按 keys 合并
            select_cols = keys + [column]
            available_cols = [c for c in select_cols if c in source_df.columns]
            if len(available_cols) == len(select_cols):
                merge_source = source_df[available_cols].drop_duplicates(subset=keys, keep='first')
                
                if result.empty:
                    return merge_source
                
                result = result.merge(merge_source, on=keys, how='left')
        
        return result
    
    def _heal_from_daily_basic(self, symbols: List[str], start_date: str,
                                end_date: str, columns: List[str]) -> Optional[pd.DataFrame]:
        """从 stock_daily_basic 表补全数据"""
        if not symbols:
            return None
        
        # 构建查询
        column_map = {
            'total_mv': 'total_mv',
            'turnover_rate': 'turnover_rate',
            'pe_ttm': 'pe_ttm',
            'pb': 'pb',
        }
        
        select_columns = ['symbol', 'trade_date']
        for col in columns:
            if col in column_map and column_map[col] not in select_columns:
                select_columns.append(column_map[col])
        
        select_str = ', '.join(select_columns)
        
        # 构建 IN 子句
        symbols_str = ', '.join([f"'{s}'" for s in symbols[:100]])  # 限制 100 只股票避免 SQL 过长
        
        query = text(f"""
            SELECT {select_str}
            FROM stock_daily_basic
            WHERE symbol IN ({symbols_str})
            AND trade_date BETWEEN :start_date AND :end_date
            ORDER BY symbol, trade_date
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn, params={
                    'start_date': start_date,
                    'end_date': end_date,
                })
            
            if len(df) > 0:
                logger.info(f"[V110][SQLAutoHealer] Healed {len(df)} rows from stock_daily_basic")
                return df
            else:
                logger.warning("[V110][SQLAutoHealer] No data from stock_daily_basic")
                return None
                
        except Exception as e:
            logger.error(f"[V110][SQLAutoHealer] Failed to query stock_daily_basic: {e}")
            return None
    
    def _heal_from_daily(self, symbols: List[str], start_date: str,
                         end_date: str, columns: List[str]) -> Optional[pd.DataFrame]:
        """从 stock_daily 表补全数据"""
        if not symbols:
            return None
        
        select_columns = ['symbol', 'trade_date']
        for col in columns:
            if col not in select_columns:
                select_columns.append(col)
        
        select_str = ', '.join(select_columns)
        symbols_str = ', '.join([f"'{s}'" for s in symbols[:100]])
        
        query = text(f"""
            SELECT {select_str}
            FROM stock_daily
            WHERE symbol IN ({symbols_str})
            AND trade_date BETWEEN :start_date AND :end_date
            ORDER BY symbol, trade_date
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn, params={
                    'start_date': start_date,
                    'end_date': end_date,
                })
            
            if len(df) > 0:
                logger.info(f"[V110][SQLAutoHealer] Healed {len(df)} rows from stock_daily")
                return df
            else:
                logger.warning("[V110][SQLAutoHealer] No data from stock_daily")
                return None
                
        except Exception as e:
            logger.error(f"[V110][SQLAutoHealer] Failed to query stock_daily: {e}")
            return None
    
    def _heal_stock_info(self, symbols: List[str], columns: List[str]) -> Optional[pd.DataFrame]:
        """从 stock_info 表补全股票信息"""
        if not symbols:
            return None
        
        # industry_code 映射
        column_map = {
            'industry_code': 'industry_code',
        }
        
        select_columns = ['symbol']
        for col in columns:
            if col in column_map and column_map[col] not in select_columns:
                select_columns.append(column_map[col])
        
        select_str = ', '.join(select_columns)
        symbols_str = ', '.join([f"'{s}'" for s in symbols[:100]])
        
        query = text(f"""
            SELECT {select_str}
            FROM stock_info
            WHERE symbol IN ({symbols_str})
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)
            
            if len(df) > 0:
                logger.info(f"[V110][SQLAutoHealer] Healed {len(df)} stocks from stock_info")
                return df
            else:
                logger.warning("[V110][SQLAutoHealer] No data from stock_info")
                return None
                
        except Exception as e:
            logger.error(f"[V110][SQLAutoHealer] Failed to query stock_info: {e}")
            return None
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log
    
    def get_missing_count(self) -> int:
        """获取缺失字段处理总数"""
        return len([log for log in self.audit_log if 'Missing' in log.get('action', '')])
    
    def get_healed_count(self) -> int:
        """获取成功补全的字段数"""
        return len([log for log in self.audit_log if log.get('status') == 'SUCCESS'])


class DataLoader:
    """
    V110 统一数据加载器 - 增强版。
    
    功能特性:
        - 获取日线价格数据和复权因子
        - 频率限制控制
        - 数据完整性校验
        - 2024 年 total_mv 数据防御检查
        - V110 新增：SQL 自动补全引擎
    
    使用示例:
        >>> loader = DataLoader()
        >>> df = loader.load_data("000001.SZ", "20240101", "20241231")
    """
    
    REQUESTS_PER_MINUTE = 60
    SLEEP_BETWEEN_REQUESTS = 0.5
    
    # V110 必需字段列表
    REQUIRED_COLUMNS = [
        'symbol', 'trade_date', 'open', 'high', 'low', 'close',
        'volume', 'amount', 'turnover_rate', 'total_mv',
        'pre_close', 'change', 'pct_chg',
    ]
    
    def __init__(self, token: Optional[str] = None, db_url: Optional[str] = None) -> None:
        """
        初始化数据加载器。
        
        Args:
            token: Tushare API token，从环境变量读取
            db_url: 数据库连接 URL，从环境变量读取
            
        Raises:
            DataLoaderError: 当必要配置缺失时抛出
        """
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token or self.token == "your_tushare_token_here":
            logger.warning("TUSHARE_TOKEN not configured. Please set it in .env file.")
            # 不抛出错误，允许从 SQL 加载
        
        if self.token:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
        else:
            self.pro = None
        
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.request_count = 0
        self.last_request_time = time.time()
        
        # V110 SQL 自动补全引擎
        self.sql_healer = SQLAutoHealer(self.db_url)
        
        logger.info("DataLoader V110 initialized")
        if self.sql_healer.engine:
            logger.info("  SQL Auto-Healer: Enabled")
        else:
            logger.info("  SQL Auto-Healer: Disabled (no database connection)")
    
    def _rate_limit(self) -> None:
        """执行 API 请求的频率限制。"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.SLEEP_BETWEEN_REQUESTS:
            sleep_time = self.SLEEP_BETWEEN_REQUESTS - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count % 10 == 0:
            logger.debug(f"Tushare API requests: {self.request_count}")
    
    def _get_asset_type(self, ts_code: str) -> str:
        """
        根据股票代码前缀自动识别资产类型。
        
        Args:
            ts_code: Tushare 代码
            
        Returns:
            "STOCK" 或 "FUND"
        """
        code = ts_code.split(".")[0]
        fund_prefixes = ("51", "58", "15", "16")
        return "FUND" if code.startswith(fund_prefixes) else "STOCK"
    
    def check_2024_total_mv(self, ts_code: str, start_date: str, end_date: str) -> bool:
        """
        【数据防御】检查 2024 年 total_mv 数据是否存在。
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 是否存在 total_mv 数据
        """
        logger.info(f"[数据防御] 检查 {ts_code} 的 2024 年 total_mv 数据...")
        
        try:
            if self.db_url and self.sql_healer.engine:
                query = text("""
                    SELECT COUNT(*) as cnt 
                    FROM stock_daily 
                    WHERE symbol = :symbol 
                    AND trade_date >= :start_date 
                    AND total_mv IS NOT NULL
                """)
                
                with self.sql_healer.engine.connect() as conn:
                    result = conn.execute(query, {
                        "symbol": ts_code,
                        "start_date": start_date,
                    })
                    row = result.fetchone()
                    cnt = row[0] if row else 0
                
                if cnt == 0:
                    logger.warning(f"[数据防御] {ts_code} 的 2024 年 total_mv 数据缺失，需要补取")
                    return False
                else:
                    logger.info(f"[数据防御] {ts_code} 的 2024 年 total_mv 数据完整 ({cnt} 条)")
                    return True
                    
        except Exception as e:
            logger.warning(f"[数据防御] 检查 total_mv 失败：{e}")
            return False
        
        return False
    
    def fetch_daily_basic(self, ts_code: str, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """获取 daily_basic 数据（包括 total_mv）。"""
        try:
            self._rate_limit()
            
            if self.pro is None:
                return None
            
            if self._get_asset_type(ts_code) != "STOCK":
                return None
            
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            
            if df is None or df.empty:
                logger.debug(f"No daily_basic data for {ts_code}")
                return None
            
            pl_df = pl.from_pandas(df)
            logger.debug(f"Fetched {len(pl_df)} rows of daily_basic data for {ts_code}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch daily_basic data for {ts_code}: {e}")
            return None
    
    def fetch_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """从 Tushare 获取日线价格数据。"""
        if self.pro is None:
            return None
        
        try:
            self._rate_limit()
            asset_type = self._get_asset_type(ts_code)
            
            if asset_type == "STOCK":
                df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = self.pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                logger.warning(f"No daily data for {ts_code} ({asset_type})")
                return None
            
            pl_df = pl.from_pandas(df)
            
            if asset_type == "FUND" and "vol" in pl_df.columns:
                pl_df = pl_df.with_columns(pl.col("vol").alias("volume"))
            
            logger.debug(f"Fetched {len(pl_df)} rows of daily data for {ts_code} ({asset_type})")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch daily data for {ts_code}: {e}")
            return None
    
    def fetch_adj_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """获取复权因子数据。"""
        if self.pro is None:
            return None
        
        try:
            self._rate_limit()
            asset_type = self._get_asset_type(ts_code)
            
            if asset_type == "STOCK":
                df = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = self.pro.fund_adj(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                logger.warning(f"No adj_factor data for {ts_code} ({asset_type})")
                return None
            
            pl_df = pl.from_pandas(df)
            logger.debug(f"Fetched {len(pl_df)} rows of adj_factor data for {ts_code}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch adj_factor data for {ts_code}: {e}")
            return None
    
    def transform_data(
        self,
        daily_df: pl.DataFrame,
        adj_factor_df: pl.DataFrame,
        daily_basic_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """转换并合并日线数据、复权因子和 daily_basic 数据。"""
        # 重命名 ts_code 为 symbol
        daily_df = daily_df.with_columns(pl.col("ts_code").alias("symbol"))
        adj_factor_df = adj_factor_df.with_columns(pl.col("ts_code").alias("symbol"))
        
        # 日期转换
        daily_df = daily_df.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date")
        )
        adj_factor_df = adj_factor_df.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date")
        )
        
        # 合并日线数据和复权因子
        merged_df = daily_df.join(
            adj_factor_df.select(["symbol", "trade_date", "adj_factor"]),
            on=["symbol", "trade_date"],
            how="left",
        )
        
        # 计算复权价格
        if "adj_factor" in merged_df.columns:
            merged_df = merged_df.with_columns([
                (pl.col("close") * pl.col("adj_factor") / 1000).alias("adj_close"),
                (pl.col("open") * pl.col("adj_factor") / 1000).alias("adj_open"),
                (pl.col("high") * pl.col("adj_factor") / 1000).alias("adj_high"),
                (pl.col("low") * pl.col("adj_factor") / 1000).alias("adj_low"),
            ])
        else:
            merged_df = merged_df.with_columns([
                pl.col("close").alias("adj_close"),
                pl.col("open").alias("adj_open"),
                pl.col("high").alias("adj_high"),
                pl.col("low").alias("adj_low"),
            ])
        
        # 合并 daily_basic 数据
        if daily_basic_df is not None and not daily_basic_df.is_empty():
            daily_basic_df = daily_basic_df.with_columns([
                pl.col("ts_code").alias("symbol"),
                pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date"),
            ])
            
            merged_df = merged_df.join(
                daily_basic_df.select([
                    "symbol", "trade_date", "turnover_rate", "volume_ratio", "total_mv", "pe_ttm", "pb"
                ]),
                on=["symbol", "trade_date"],
                how="left",
            )
            
            # volume_ratio -> vol_ratio
            if "volume_ratio" in merged_df.columns:
                merged_df = merged_df.with_columns(pl.col("volume_ratio").alias("vol_ratio"))
        
        # 选择最终列
        result_df = merged_df.select([
            pl.col("symbol"),
            pl.col("trade_date"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("pre_close"),
            pl.col("change"),
            pl.col("pct_chg"),
            pl.col("vol").alias("volume"),
            pl.col("amount"),
            pl.col("adj_factor").fill_null(1000),
            pl.col("turnover_rate").fill_null(None).cast(pl.Float64),
            pl.col("vol_ratio").fill_null(None).cast(pl.Float64),
            pl.col("total_mv").fill_null(None).cast(pl.Float64),
            pl.col("pe_ttm").fill_null(None).cast(pl.Float64),
            pl.col("pb").fill_null(None).cast(pl.Float64),
            pl.col("adj_open"),
            pl.col("adj_high"),
            pl.col("adj_low"),
            pl.col("adj_close"),
        ])
        
        result_df = result_df.sort(["symbol", "trade_date"])
        return result_df
    
    def load_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        check_mv: bool = True,
    ) -> pl.DataFrame:
        """加载单只股票的完整数据。"""
        logger.info(f"Loading data for {ts_code} from {start_date} to {end_date}")
        
        # 【数据防御】检查 2024 年 total_mv 数据
        if check_mv and "2024" in end_date:
            has_mv = self.check_2024_total_mv(ts_code, start_date, end_date)
            if not has_mv:
                logger.warning(f"[数据防御] {ts_code} 的 total_mv 数据缺失，尝试补取...")
        
        # 获取日线数据
        if self.pro:
            daily_df = self.fetch_daily_data(ts_code, start_date, end_date)
        else:
            daily_df = None
        
        if daily_df is None or daily_df.is_empty():
            # 尝试从 SQL 加载
            logger.info(f"[V110] Tushare unavailable, attempting SQL load for {ts_code}...")
            daily_df = self._load_from_sql(ts_code, start_date, end_date)
            if daily_df is None:
                raise DataLoaderError(f"No daily data for {ts_code}")
        
        # 获取复权因子
        if self.pro:
            adj_factor_df = self.fetch_adj_factor(ts_code, start_date, end_date)
        else:
            adj_factor_df = None
        
        if adj_factor_df is None or adj_factor_df.is_empty():
            logger.warning(f"No adj_factor data for {ts_code}, using default")
            adj_factor_df = daily_df.select(["ts_code", "trade_date"]).with_columns(
                pl.lit(1000).alias("adj_factor")
            )
        
        # 获取 daily_basic 数据
        daily_basic_df = None
        if self.pro and self._get_asset_type(ts_code) == "STOCK":
            daily_basic_df = self.fetch_daily_basic(ts_code, start_date, end_date)
        
        # 转换和合并数据
        result_df = self.transform_data(daily_df, adj_factor_df, daily_basic_df)
        
        logger.info(f"Loaded {len(result_df)} rows for {ts_code}")
        return result_df
    
    def _load_from_sql(self, ts_code: str, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """从 SQL 加载单只股票数据"""
        if not self.db_url or not self.sql_healer.engine:
            return None
        
        query = text("""
            SELECT symbol, trade_date, open, high, low, close, pre_close,
                   change, pct_chg, volume, amount, turnover_rate, total_mv
            FROM stock_daily
            WHERE symbol = :symbol
            AND trade_date BETWEEN :start_date AND :end_date
            ORDER BY symbol, trade_date
        """)
        
        try:
            with self.sql_healer.engine.connect() as conn:
                df = pd.read_sql_query(query, conn, params={
                    'symbol': ts_code,
                    'start_date': start_date,
                    'end_date': end_date,
                })
            
            if len(df) > 0:
                logger.info(f"[V110] Loaded {len(df)} rows from SQL for {ts_code}")
                return pl.from_pandas(df)
            else:
                return None
                
        except Exception as e:
            logger.error(f"[V110] Failed to load from SQL: {e}")
            return None
    
    def load_index_constituents(
        self,
        index_code: str = "000300.SH",
        start_date: str = "20240101",
        end_date: str = None,
    ) -> dict[str, pl.DataFrame]:
        """加载指数所有成分股数据。"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        # 获取成分股列表
        try:
            if self.pro:
                self._rate_limit()
                df = self.pro.index_member(ts_code=index_code)
                if df is None or df.empty:
                    logger.error(f"No constituents for {index_code}")
                    return {}
                
                stock_codes = pl.from_pandas(df)["con_code"].to_list()
                logger.info(f"Found {len(stock_codes)} constituents in {index_code}")
            else:
                # 从 SQL 获取成分股
                stock_codes = self._load_index_constituents_from_sql(index_code)
                if not stock_codes:
                    return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch index constituents: {e}")
            return {}
        
        # 加载每只股票数据
        result = {}
        for i, ts_code in enumerate(stock_codes):
            try:
                df = self.load_data(ts_code, start_date, end_date)
                result[ts_code] = df
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(stock_codes)} stocks loaded")
                    
            except Exception as e:
                logger.error(f"Failed to load {ts_code}: {e}")
        
        return result
    
    def _load_index_constituents_from_sql(self, index_code: str) -> List[str]:
        """从 SQL 加载指数成分股"""
        if not self.sql_healer.engine:
            return []
        
        query = text("""
            SELECT con_code FROM stock_index_constituent
            WHERE index_code = :index_code
            AND is_in = 1
        """)
        
        try:
            with self.sql_healer.engine.connect() as conn:
                result = conn.execute(query, {'index_code': index_code})
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to load index constituents from SQL: {e}")
            return []
    
    def save_to_parquet(
        self,
        data_dict: dict[str, pl.DataFrame],
        output_path: str,
    ) -> None:
        """保存数据到 Parquet 文件。"""
        if not data_dict:
            logger.warning("No data to save")
            return
        
        # 合并所有股票数据
        all_dfs = []
        for ts_code, df in data_dict.items():
            df_with_code = df.with_columns(pl.lit(ts_code).alias("ts_code"))
            all_dfs.append(df_with_code)
        
        if all_dfs:
            combined_df = pl.concat(all_dfs)
            
            # 创建输出目录
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存 Parquet
            combined_df.write_parquet(output_path, compression="snappy")
            logger.info(f"Saved {len(combined_df)} rows to {output_path}")
    
    def load_from_parquet(self, parquet_path: str) -> pl.DataFrame:
        """从 Parquet 文件加载数据。"""
        logger.info(f"Loading data from {parquet_path}")
        df = pl.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def heal_dataframe(self, df: pd.DataFrame, symbols: List[str],
                       start_date: str, end_date: str) -> pd.DataFrame:
        """
        【V110 新增】对已有 DataFrame 进行字段补全。
        
        Args:
            df: 原始 DataFrame
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            补全后的 DataFrame
        """
        return self.sql_healer.heal_from_sql(df, symbols, start_date, end_date)
    
    def get_sql_audit_log(self) -> List[Dict]:
        """获取 SQL 补全审计日志"""
        return self.sql_healer.get_audit_log()


def get_loader(token: Optional[str] = None) -> DataLoader:
    """获取 DataLoader 实例。"""
    return DataLoader(token)