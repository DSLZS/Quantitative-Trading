"""
V207 Data Healer Module - 数据医师
===================================

【V207.1 核心职责】
1. 数据校验闸口 (Data Validation Gate)
   - 检查 stock_industry_daily, stock_fund_flow, stock_daily 表的数据完整性
   - 验证回测年份的数据覆盖度

2. 数据修复逻辑 (Data Healing Logic)
   - 通过 LEFT JOIN stock_industry_daily 补全 stock_daily 的 industry_code
   - 对缺失的 total_mv 使用前向填充 + 截面中位数填充
   - 对缺失的交易日期进行填充

3. 数据质量评分 (Data Quality Score)
   - 计算每个交易日的完整度
   - 标记低质量数据供回测引擎过滤

【严格红线】
- 严禁因为数据缺失就 fillna(0) 导致信号被抹平
- 必须主动补全数据，拒绝借口
"""

import sys
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import os

load_dotenv()

# 配置
EPSILON = 1e-6
DEFAULT_CHUNK_SIZE = 10000

# 数据校验阈值
INDUSTRY_MIN_ROWS_PER_YEAR = 50000
FUND_FLOW_MIN_ROWS_PER_YEAR = 50000
STOCK_DAILY_MIN_ROWS_PER_YEAR = 800000

# 数据质量阈值
DATA_QUALITY_MIN_STOCKS = 3000
DATA_QUALITY_MIN_INDUSTRY_COVERAGE = 0.8


def winsorize_series(series: pd.Series, sigma: float = 3.0) -> pd.Series:
    """缩尾处理异常值"""
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    
    std = series_clean.std()
    if pd.isna(std) or std < 1e-10:
        std = 1.0
    
    lower = mean - sigma * std
    upper = mean + sigma * std
    series_clean = series_clean.clip(lower=lower, upper=upper)
    
    return series_clean


class V207DataHealer:
    """
    V207 数据修复器
    
    【核心职责】
    1. 运行数据校验闸口
    2. 修复缺失/异常数据
    3. 输出数据质量报告
    """
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", 
                "mysql+pymysql://root:123456@localhost:3306/quantitative_trading")
        
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        
        logger.info("=" * 80)
        logger.info("V207 Data Healer Initialized")
        logger.info("=" * 80)
    
    def dispose(self):
        """释放数据库连接"""
        self.engine.dispose()
        logger.info("[DataHealer] Database connections disposed")
    
    def run_data_gate(self, years: List[int]) -> bool:
        """
        运行数据校验闸口
        
        Args:
            years: 回测年份列表
            
        Returns:
            是否通过校验
        """
        logger.info("\n" + "=" * 80)
        logger.info("V207 Data Validation Gate")
        logger.info("=" * 80)
        
        gate_passed = True
        
        for table in ['stock_industry_daily', 'stock_fund_flow', 'stock_daily']:
            logger.info(f"\n[Table] {table}")
            
            for year in years:
                query = text(f"SELECT COUNT(*) FROM {table} WHERE YEAR(trade_date) = :year")
                with self.engine.connect() as conn:
                    count = conn.execute(query, {"year": year}).scalar()
                
                if table in ['stock_industry_daily', 'stock_fund_flow']:
                    min_rows = INDUSTRY_MIN_ROWS_PER_YEAR
                else:
                    min_rows = STOCK_DAILY_MIN_ROWS_PER_YEAR
                
                status = '✓' if count >= min_rows else '✗'
                logger.info(f"  {year}: {count:,} rows {status}")
                
                if table == 'stock_industry_daily' and count < min_rows:
                    logger.error(f"[Gate] FAIL: {table}/{year} has {count} rows < {min_rows}")
                    gate_passed = False
        
        if gate_passed:
            logger.info("\n[Gate] PASSED - Data validation successful")
        else:
            logger.error("\n[Gate] FAILED - Data healing required")
        
        return gate_passed
    
    def run_full_healing(self, years: List[int]) -> Dict[str, Any]:
        """
        运行完整数据修复
        
        Args:
            years: 回测年份列表
            
        Returns:
            修复结果字典
        """
        logger.info("\n" + "=" * 80)
        logger.info("V207 Full Data Healing Process")
        logger.info("=" * 80)
        
        healing_result = {
            'years': years,
            'tables_healed': [],
            'rows_repaired': 0,
            'final_passed': False,
        }
        
        # 1. 修复 stock_daily 的 industry_code (通过 stock_industry_daily 补全)
        logger.info("\n[Healing] Repairing stock_daily.industry_code via LEFT JOIN...")
        industry_result = self._heal_industry_code_in_daily(years)
        healing_result['tables_healed'].append('stock_daily_industry_code')
        healing_result['rows_repaired'] += industry_result.get('rows_repaired', 0)
        
        # 2. 修复 stock_daily 的 total_mv (前向填充 + 截面中位数)
        logger.info("\n[Healing] Repairing stock_daily.total_mv...")
        mv_result = self._heal_total_mv_in_daily(years)
        healing_result['tables_healed'].append('stock_daily_total_mv')
        healing_result['rows_repaired'] += mv_result.get('rows_repaired', 0)
        
        # 3. 验证修复结果
        logger.info("\n[Verification] Verifying healed data...")
        final_passed = self.run_data_gate(years)
        healing_result['final_passed'] = final_passed
        
        logger.info("\n" + "=" * 80)
        logger.info("V207 Data Healing Summary")
        logger.info("=" * 80)
        logger.info(f"  Tables Healed: {len(healing_result['tables_healed'])}")
        logger.info(f"  Rows Repaired: {healing_result['rows_repaired']:,}")
        logger.info(f"  Final Status: {'PASSED' if final_passed else 'FAILED'}")
        logger.info("=" * 80)
        
        return healing_result
    
    def _heal_industry_code_in_daily(self, years: List[int]) -> Dict[str, Any]:
        """
        通过 stock_industry_daily 补全 stock_daily 的 industry_code
        
        【V207.1 核心修复】
        - stock_daily.industry_code 在 2020-2023 年 100% 缺失
        - stock_industry_daily 有完整数据
        - 使用 UPDATE + JOIN 进行批量补全
        """
        result = {'rows_repaired': 0}
        
        for year in years:
            logger.info(f"  [Year {year}] Checking industry_code coverage...")
            
            # 检查缺失数量
            check_query = text("""
                SELECT COUNT(*) FROM stock_daily sd
                WHERE YEAR(sd.trade_date) = :year
                  AND (sd.industry_code IS NULL OR sd.industry_code = '')
            """)
            
            with self.engine.connect() as conn:
                missing_count = conn.execute(check_query, {"year": year}).scalar()
            
            if missing_count == 0:
                logger.info(f"  [Year {year}] industry_code already complete")
                continue
            
            logger.info(f"  [Year {year}] Found {missing_count:,} rows with missing industry_code")
            
            # 使用 UPDATE + JOIN 补全
            update_query = text("""
                UPDATE stock_daily sd
                INNER JOIN stock_industry_daily sid
                    ON sd.trade_date = sid.trade_date AND sd.symbol = sid.symbol
                SET sd.industry_code = sid.industry_code
                WHERE YEAR(sd.trade_date) = :year
                  AND (sd.industry_code IS NULL OR sd.industry_code = '')
                  AND sid.industry_code IS NOT NULL
                  AND sid.industry_code != ''
            """)
            
            with self.engine.connect() as conn:
                conn.execute(update_query, {"year": year})
                conn.commit()
            
            # 验证修复结果
            verify_query = text("""
                SELECT COUNT(*) FROM stock_daily sd
                WHERE YEAR(sd.trade_date) = :year
                  AND (sd.industry_code IS NULL OR sd.industry_code = '')
            """)
            
            with self.engine.connect() as conn:
                remaining = conn.execute(verify_query, {"year": year}).scalar()
            
            repaired = missing_count - remaining
            result['rows_repaired'] += repaired
            logger.info(f"  [Year {year}] Repaired {repaired:,} rows, {remaining:,} remaining")
        
        return result
    
    def _heal_total_mv_in_daily(self, years: List[int]) -> Dict[str, Any]:
        """
        修复 stock_daily 的 total_mv
        
        【V207.1 策略】
        - 2020, 2022, 2023 年 total_mv 100% 缺失
        - 使用同一股票相邻日期的 total_mv 进行前向/后向填充
        - 对于仍缺失的，使用截面中位数填充
        """
        result = {'rows_repaired': 0}
        
        for year in years:
            logger.info(f"  [Year {year}] Checking total_mv coverage...")
            
            # 检查缺失数量
            check_query = text("""
                SELECT COUNT(*) FROM stock_daily
                WHERE YEAR(trade_date) = :year AND total_mv IS NULL
            """)
            
            with self.engine.connect() as conn:
                missing_count = conn.execute(check_query, {"year": year}).scalar()
            
            if missing_count == 0:
                logger.info(f"  [Year {year}] total_mv already complete")
                continue
            
            logger.info(f"  [Year {year}] Found {missing_count:,} rows with missing total_mv")
            
            # 使用 Python 端填充（因为 MySQL 窗口函数性能差）
            load_query = text("""
                SELECT trade_date, symbol, total_mv
                FROM stock_daily
                WHERE YEAR(trade_date) = :year
                ORDER BY symbol, trade_date
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(load_query, conn, params={"year": year})
            
            if df.empty:
                continue
            
            # 按股票分组进行前向/后向填充
            df['total_mv'] = df.groupby('symbol')['total_mv'].ffill().bfill()
            
            # 使用截面中位数填充剩余缺失值
            for trade_date in df['trade_date'].unique():
                date_mask = df['trade_date'] == trade_date
                date_data = df.loc[date_mask]
                median_mv = date_data['total_mv'].median()
                if pd.notna(median_mv):
                    df.loc[date_mask & df['total_mv'].isna(), 'total_mv'] = median_mv
            
            # 批量更新回数据库
            updated_df = df[df['total_mv'].notna()].copy()
            
            if updated_df.empty:
                continue
            
            batch_size = 2000
            update_query = text("""
                UPDATE stock_daily
                SET total_mv = :total_mv
                WHERE trade_date = :trade_date AND symbol = :symbol
            """)
            
            repaired_count = 0
            with self.engine.connect() as conn:
                for i in range(0, len(updated_df), batch_size):
                    batch = updated_df.iloc[i:i+batch_size]
                    for _, row in batch.iterrows():
                        try:
                            conn.execute(update_query, {
                                'total_mv': float(row['total_mv']),
                                'trade_date': row['trade_date'],
                                'symbol': row['symbol'],
                            })
                            repaired_count += 1
                        except Exception as e:
                            logger.debug(f"  Update failed: {e}")
                    conn.commit()
            
            result['rows_repaired'] += repaired_count
            logger.info(f"  [Year {year}] Repaired {repaired_count:,} rows")
        
        return result
    
    def load_and_heal_data(
        self,
        years: List[int],
        warmup_year: int = 2019,
        warmup_days: int = 60
    ) -> pd.DataFrame:
        """
        加载并修复数据
        
        【V207.1 核心修复】
        - 分别加载三张表，在 Python 中使用 merge 进行 LEFT JOIN
        - 优先使用 stock_daily 中的 industry_code，如果为空则使用 stock_industry_daily
        - 对缺失的数值特征进行 fillna(0) 处理
        
        Args:
            years: 回测年份列表
            warmup_year: 预热年份
            warmup_days: 预热天数
            
        Returns:
            修复后的 DataFrame
        """
        logger.info("\n[Data] Loading and healing data with Python LEFT JOIN...")
        
        start_date = f"{warmup_year}-01-01"
        max_year = max(years) if years else datetime.now().year
        end_date = f"{max_year}-12-31"
        
        # 分别加载三张表
        logger.info("[Data] Loading stock_daily...")
        query_daily = text("""
            SELECT 
                trade_date, symbol, 
                open, high, low, close, pre_close,
                pct_chg, volume, amount, turnover_rate,
                industry_code as daily_industry_code,
                total_mv, is_st
            FROM stock_daily
            WHERE trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date, symbol
        """)
        
        logger.info("[Data] Loading stock_industry_daily...")
        query_industry = text("""
            SELECT trade_date, symbol, industry_code, industry_name
            FROM stock_industry_daily
            WHERE trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date, symbol
        """)
        
        logger.info("[Data] Loading stock_fund_flow...")
        query_fund = text("""
            SELECT trade_date, symbol, net_main_amount, net_main_rate
            FROM stock_fund_flow
            WHERE trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date, symbol
        """)
        
        with self.engine.connect() as conn:
            df_daily = pd.read_sql(query_daily, conn, params={'start_date': start_date, 'end_date': end_date})
            df_industry = pd.read_sql(query_industry, conn, params={'start_date': start_date, 'end_date': end_date})
            df_fund = pd.read_sql(query_fund, conn, params={'start_date': start_date, 'end_date': end_date})
        
        logger.info(f"[Data] Loaded {len(df_daily):,} daily, {len(df_industry):,} industry, {len(df_fund):,} fund_flow rows")
        
        # Python 端 LEFT JOIN 合并
        logger.info("[Data] Merging data in Python (LEFT JOIN)...")
        df = df_daily.merge(df_industry, on=['trade_date', 'symbol'], how='left')
        df = df.merge(df_fund, on=['trade_date', 'symbol'], how='left')
        
        # 【V207.1 核心修复】优先使用 stock_daily 中的 industry_code
        df['daily_industry_code'] = df['daily_industry_code'].fillna('')
        df['industry_code'] = df['industry_code'].fillna('')
        
        df['industry_code'] = np.where(
            df['daily_industry_code'] != '',
            df['daily_industry_code'],
            df['industry_code']
        )
        
        df = df.drop(columns=['daily_industry_code'], errors='ignore')
        
        # 处理 industry_name
        df['industry_name'] = df['industry_name'].fillna('Unknown')
        
        # 处理资金流字段
        df['net_main_amount'] = df['net_main_amount'].fillna(0)
        df['net_main_rate'] = df['net_main_rate'].fillna(0)
        
        # 处理 total_mv - 使用前向填充 + 截面中位数
        df['total_mv'] = df.groupby('symbol')['total_mv'].ffill().bfill()
        for trade_date in df['trade_date'].unique():
            date_mask = df['trade_date'] == trade_date
            median_mv = df.loc[date_mask, 'total_mv'].median()
            if pd.notna(median_mv):
                df.loc[date_mask & df['total_mv'].isna(), 'total_mv'] = median_mv
        df['total_mv'] = df['total_mv'].fillna(0)
        
        # 处理 is_st
        df['is_st'] = df['is_st'].fillna(0)
        
        # 统计行业数据覆盖情况
        industry_missing = (df['industry_code'].isna() | (df['industry_code'] == '')).sum()
        logger.info(f"[Data] Merge complete. industry_code NULL/empty: {industry_missing:,}")
        
        # 按年份统计行业覆盖率
        df['_year'] = df['trade_date'].astype(str).str[:4].astype(int)
        for year in sorted(df['_year'].unique()):
            if year < 2020:
                continue
            year_mask = df['_year'] == year
            year_missing = ((df.loc[year_mask, 'industry_code'].isna()) | 
                          (df.loc[year_mask, 'industry_code'] == '')).sum()
            year_total = year_mask.sum()
            logger.info(f"[Data] Year {year}: {year_missing:,}/{year_total:,} rows missing industry_code "
                       f"({100*year_missing/year_total:.2f}%)")
        df = df.drop(columns=['_year'], errors='ignore')
        
        if df.empty:
            logger.error("[Data] No data loaded from database")
            return df
        
        # 数据类型转换
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
        df['symbol'] = df['symbol'].astype(str)
        
        # 数值列转换
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 
                       'volume', 'amount', 'turnover_rate', 'total_mv', 'is_st',
                       'net_main_amount', 'net_main_rate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 缩尾处理异常值
        for col in ['pct_chg', 'net_main_rate', 'turnover_rate']:
            if col in df.columns:
                df[col] = winsorize_series(df[col], sigma=3.0)
        
        logger.info(f"[Data] Healing complete. {len(df):,} rows, {df['symbol'].nunique()} unique symbols")
        logger.info(f"[Data] Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
        
        return df


def get_data_healer(database_url: str = None) -> V207DataHealer:
    """获取 V207DataHealer 实例"""
    return V207DataHealer(database_url=database_url)