"""
V195 Player (选手脚本) - 因子动物园信号计算 (第五轮迭代 - 回滚 V191 逻辑)

【核心职责】
1. 从数据库读取行情数据
2. 计算 V191 经典逻辑：Rank(Mom) * Rank(1/Vol)
3. 输出 signals.csv 供裁判使用

【第五轮优化策略 - 回滚 V191】
分析第四轮失败原因：
- 因子方向完全反了，IC 变成负数
- 趋势动量 (正向) 在 A 股不适用
- 聪明钱因子交互项可能过度复杂

优化方向:
回到 V191 经典逻辑，并做微调:
1. 动量因子：使用短期反转 (负向)
2. 波动因子：低波 (正向)
3. 使用乘法而非加法，增强因子间协同

【因子逻辑】(第五轮 - V191 回滚)
F1 (Reversal):    rank(delay(close, 5) / close)     - 短期反转
F2 (Volatility):  rank(1 / stddev(pct_chg, 5))     - 低波动

Score = F1 * F2  (乘法协同)

【红线】
1. 仅输出 signals.csv，不执行回测
2. 严禁调用裁判脚本的任何内部函数
3. 严禁使用未来函数 - 必须严格 shift(1)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

# 配置
VERSION = "V195_R5"

# 数据库配置
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')

# 配置 logger - 静默模式
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class V195Player:
    """
    V195 Player - 因子计算与信号生成 (第五轮迭代 - V191 回滚)
    
    经典逻辑：Rank(Mom) * Rank(1/Vol)
    """
    
    def __init__(self):
        self.factor_log = []
    
    def compute_reversal_factor(self, df: pl.DataFrame) -> pl.Series:
        """
        F1 Reversal: rank(delay(close, 5) / close)
        
        短期反转效应：前期跌幅越大，反弹概率越高
        """
        momentum = df.with_columns(
            (pl.col("close").shift(5).over("symbol") / pl.col("close") - 1.0)
            .alias("reversal_5")
        ).get_column("reversal_5")
        
        momentum_rank = self._cross_sectional_rank(momentum, df.get_column("trade_date"))
        
        self.factor_log.append({
            'factor': 'F1_reversal_5',
            'formula': 'rank(delay(close, 5) / close)',
        })
        
        return momentum_rank
    
    def compute_volatility_factor(self, df: pl.DataFrame) -> pl.Series:
        """
        F2 Volatility: rank(1 / stddev(pct_chg, 5))
        
        低波因子：波动率越低，得分越高
        """
        volatility = df.with_columns(
            pl.col("pct_chg")
            .shift(1)
            .over("symbol")
            .rolling_std(window_size=5, min_samples=3)
            .alias("volatility_5")
        ).get_column("volatility_5")
        
        inv_volatility = 1.0 / (volatility + 1e-6)
        inv_vol_rank = self._cross_sectional_rank(inv_volatility, df.get_column("trade_date"))
        
        self.factor_log.append({
            'factor': 'F2_volatility_5',
            'formula': 'rank(1 / stddev(pct_chg, 5))',
        })
        
        return inv_vol_rank
    
    def _cross_sectional_rank(self, values: pl.Series, dates: pl.Series) -> pl.Series:
        """横截面排名 (0-1)"""
        temp_df = pl.DataFrame({
            "trade_date": dates,
            "value": values
        })
        
        ranked = temp_df.with_columns(
            pl.col("value")
            .rank(method="average")
            .over("trade_date")
            .alias("rank")
        )
        
        ranked = ranked.with_columns(
            (pl.col("rank") / (pl.col("rank").max().over("trade_date") + 1e-6))
            .alias("normalized_rank")
        )
        
        return ranked.get_column("normalized_rank").fill_null(0.5)
    
    def compute_composite_score(self, df: pl.DataFrame) -> pl.Series:
        """
        计算综合得分：Score = F1 * F2 (乘法协同)
        
        V191 经典逻辑：Rank(Mom) * Rank(1/Vol)
        - 反转因子 * 低波因子
        - 两者都高时得分最高
        """
        f1 = self.compute_reversal_factor(df)
        f2 = self.compute_volatility_factor(df)
        
        # 乘法协同
        composite = f1 * f2
        
        composite = composite.fill_nan(0.25).fill_null(0.25)
        
        return composite
    
    def generate_signals(self, df: pl.DataFrame, output_path: str = "signals.csv") -> pl.DataFrame:
        """生成交易信号并输出到 CSV"""
        score = self.compute_composite_score(df)
        
        output = df.select(["symbol", "trade_date"]).with_columns(
            score.alias("score"),
            pl.lit(VERSION).alias("version")
        )
        
        output = output.sort(["trade_date", "score"], descending=[False, True])
        output.write_csv(output_path)
        
        return output


class V195DataLoader:
    """V195 数据加载器 - 流式处理，避免 OOM"""
    
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.engine = create_engine(db_url, pool_size=5, max_overflow=10)
    
    def load_year_data(self, year: int, chunk_size: int = 50000) -> pl.DataFrame:
        """加载指定年份的数据"""
        query = text("""
            SELECT symbol, trade_date, close, pct_chg, volume, turnover_rate, high, low, pre_close
            FROM stock_daily
            WHERE YEAR(trade_date) = :year
            ORDER BY trade_date, symbol
        """)
        
        chunks = []
        for chunk in pd.read_sql_query(query, self.engine, params={'year': year}, chunksize=chunk_size):
            pl_chunk = pl.from_pandas(chunk)
            chunks.append(pl_chunk)
        
        if not chunks:
            return pl.DataFrame()
        
        df = pl.concat(chunks)
        
        stock_count = df['symbol'].n_unique()
        if stock_count < 4000:
            print(f"WARNING: Year {year} stock count {stock_count} < 4000")
        
        return df
    
    def load_date_range_data(
        self,
        start_date: str,
        end_date: str,
        chunk_size: int = 50000
    ) -> pl.DataFrame:
        """加载指定日期范围的数据"""
        query = text("""
            SELECT symbol, trade_date, close, pct_chg, volume, turnover_rate, high, low, pre_close
            FROM stock_daily
            WHERE trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date, symbol
        """)
        
        chunks = []
        for chunk in pd.read_sql_query(
            query, self.engine,
            params={'start_date': start_date, 'end_date': end_date},
            chunksize=chunk_size
        ):
            pl_chunk = pl.from_pandas(chunk)
            chunks.append(pl_chunk)
        
        if not chunks:
            return pl.DataFrame()
        
        df = pl.concat(chunks)
        return df


def run_player(
    years: List[int] = None,
    output_path: str = "signals.csv"
) -> bool:
    """运行 Player 生成信号"""
    if years is None:
        years = [2023, 2024, 2025]
    
    print(f"V195 Player R5 - V191 经典逻辑回滚 (第五轮迭代)")
    print(f"Years: {years}")
    print(f"Logic: Rank(Mom) * Rank(1/Vol)")
    
    loader = V195DataLoader()
    all_data = []
    
    for year in years:
        df = loader.load_year_data(year)
        if not df.is_empty():
            df = df.with_columns([
                pl.col("close").cast(pl.Float64),
                pl.col("pct_chg").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("turnover_rate").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("pre_close").cast(pl.Float64),
            ])
            all_data.append(df)
            print(f"  Loaded year {year}: {len(df):,} rows, {df['symbol'].n_unique()} stocks")
    
    if not all_data:
        print("ERROR: No data loaded")
        return False
    
    full_df = pl.concat(all_data, how="vertical_relaxed")
    print(f"Total data: {len(full_df):,} rows")
    
    player = V195Player()
    player.generate_signals(full_df, output_path)
    
    print(f"Signals saved to {output_path}")
    print(f"Factor log:")
    for log in player.factor_log:
        print(f"  {log['factor']}: {log['formula']}")
    
    print(f"\nScore = F1 * F2 (乘法协同)")
    
    return True


def main():
    """主函数"""
    print("=" * 70)
    print("V195 Player R5 - V191 经典逻辑回滚 (第五轮迭代)")
    print("=" * 70)
    
    success = run_player()
    
    if success:
        print("\n" + "=" * 70)
        print("V195 Player R5 completed. signals.csv generated.")
        print("=" * 70)
    else:
        print("\nERROR: Player execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()