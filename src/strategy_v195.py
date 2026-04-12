"""
V195 Strategy (选手脚本) - 因子动物园线性组合

【核心逻辑】
F1 (Momentum): rank(close / delay(close, 5))
F2 (Volatility): rank(1 / stddev(pct_chg, 5))
F3 (Liquidity): rank(turnover_rate)
F4 (Volume_Pump): rank(volume / mean(volume, 20))

组合逻辑：Score = F1*0.3 + F2*0.3 + F3*0.2 + F4*0.2

【红线】
1. 仅负责读取数据、计算因子并输出 signals.csv
2. 严禁调用裁判脚本的任何内部函数
3. 严禁输出大数据 - 只输出 signals.csv
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

# 配置
VERSION = "V195"
DEFAULT_WEIGHTS = {
    'F1_momentum': 0.3,
    'F2_volatility': 0.3,
    'F3_liquidity': 0.2,
    'F4_volume_pump': 0.2
}

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class V195FactorCalculator:
    """
    V195 因子计算器 - 使用 Polars 进行向量化计算
    
    因子动物园：
    F1 (Momentum): rank(close / delay(close, 5))
    F2 (Volatility): rank(1 / stddev(pct_chg, 5))
    F3 (Liquidity): rank(turnover_rate)
    F4 (Volume_Pump): rank(volume / mean(volume, 20))
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        初始化因子计算器
        
        Args:
            weights: 因子权重字典，默认使用 DEFAULT_WEIGHTS
        """
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self.factor_log = []
        
        # 验证权重和为 1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
    
    def compute_momentum_factor(self, df: pl.DataFrame) -> pl.Series:
        """
        F1 Reversal: rank(delay(close, 5) / close)
        
        A 股短期反转效应：前期跌幅越大，反弹概率越高
        注意：这里是反转因子，不是动量因子
        """
        # 计算 5 日反转：close.shift(5) / close (前期价格/当前价格)
        # 前期价格相对当前价格越高，说明近期下跌，反弹概率越大
        momentum = df.with_columns(
            (pl.col("close").shift(5).over("symbol") / pl.col("close") - 1.0)
            .alias("reversal_5")
        ).get_column("reversal_5")
        
        # 横截面排名 (0-1)
        momentum_rank = self._cross_sectional_rank(momentum, df.get_column("trade_date"))
        
        self.factor_log.append({
            'factor': 'F1_reversal',
            'formula': 'rank(delay(close, 5) / close)',
            'weight': self.weights['F1_momentum']
        })
        
        return momentum_rank
    
    def compute_volatility_factor(self, df: pl.DataFrame) -> pl.Series:
        """
        F2 Volatility: rank(1 / stddev(pct_chg, 5))
        
        计算 5 日波动率倒数排名因子
        """
        # 计算 5 日波动率
        volatility = df.with_columns(
            pl.col("pct_chg")
            .shift(1)
            .over("symbol")
            .rolling_std(window_size=5, min_periods=3)
            .alias("volatility_5")
        ).get_column("volatility_5")
        
        # 计算 1 / volatility，避免除零
        inv_volatility = 1.0 / (volatility + 1e-6)
        
        # 横截面排名 (0-1)
        inv_vol_rank = self._cross_sectional_rank(inv_volatility, df.get_column("trade_date"))
        
        self.factor_log.append({
            'factor': 'F2_volatility',
            'formula': 'rank(1 / stddev(pct_chg, 5))',
            'weight': self.weights['F2_volatility']
        })
        
        return inv_vol_rank
    
    def compute_liquidity_factor(self, df: pl.DataFrame) -> pl.Series:
        """
        F3 Liquidity (Reversal): 1 - rank(turnover_rate)
        
        A 股高换手率往往是见顶信号，取反向
        低换手率=低关注度=后续上涨概率高
        """
        turnover = df.get_column("turnover_rate").fill_null(0)
        
        # 横截面排名后取反 (1 - rank)
        liquidity_rank = self._cross_sectional_rank(turnover, df.get_column("trade_date"))
        liquidity_rank = 1.0 - liquidity_rank
        
        self.factor_log.append({
            'factor': 'F3_liquidity_reversal',
            'formula': '1 - rank(turnover_rate)',
            'weight': self.weights['F3_liquidity']
        })
        
        return liquidity_rank
    
    def compute_volume_pump_factor(self, df: pl.DataFrame) -> pl.Series:
        """
        F4 Volume_Pump (Reversal): rank(mean(volume, 20) / volume)
        
        A 股放量往往是阶段性顶部，取反向
        缩量=低关注度=后续上涨概率高
        """
        # 计算 20 日平均成交量
        volume = df.get_column("volume")
        mean_volume = df.with_columns(
            pl.col("volume")
            .shift(1)
            .over("symbol")
            .rolling_mean(window_size=20, min_periods=10)
            .alias("mean_volume_20")
        ).get_column("mean_volume_20")
        
        # 计算 mean_volume / volume (取倒数，缩量高分)
        volume_ratio = (mean_volume + 1e-6) / volume
        
        # 横截面排名 (0-1)
        volume_rank = self._cross_sectional_rank(volume_ratio, df.get_column("trade_date"))
        
        self.factor_log.append({
            'factor': 'F4_volume_reversal',
            'formula': 'rank(mean(volume, 20) / volume)',
            'weight': self.weights['F4_volume_pump']
        })
        
        return volume_rank
    
    def _cross_sectional_rank(self, values: pl.Series, dates: pl.Series) -> pl.Series:
        """
        横截面排名 (0-1)
        
        对每个交易日的数据进行排名，返回 0-1 之间的值
        """
        # 创建临时 DataFrame 进行排名
        temp_df = pl.DataFrame({
            "trade_date": dates,
            "value": values
        })
        
        # 按日期分组排名
        ranked = temp_df.with_columns(
            pl.col("value")
            .rank(method="average")
            .over("trade_date")
            .alias("rank")
        )
        
        # 归一化到 0-1
        max_rank = temp_df.select(
            pl.col("value").count().over("trade_date")
        ).get_column("value")
        
        ranked = ranked.with_columns(
            (pl.col("rank") / (pl.col("rank").max().over("trade_date") + 1e-6)).alias("normalized_rank")
        )
        
        return ranked.get_column("normalized_rank").fill_null(0.5)
    
    def compute_composite_score(self, df: pl.DataFrame) -> pl.Series:
        """
        计算综合得分：Score = F1*0.3 + F2*0.3 + F3*0.2 + F4*0.2
        
        Args:
            df: Polars DataFrame，包含必要的列
            
        Returns:
            综合得分 Series
        """
        logger.info("Computing V195 composite score...")
        
        # 计算各因子
        f1 = self.compute_momentum_factor(df)
        f2 = self.compute_volatility_factor(df)
        f3 = self.compute_liquidity_factor(df)
        f4 = self.compute_volume_pump_factor(df)
        
        # 线性组合
        composite = (
            f1 * self.weights['F1_momentum'] +
            f2 * self.weights['F2_volatility'] +
            f3 * self.weights['F3_liquidity'] +
            f4 * self.weights['F4_volume_pump']
        )
        
        # 处理 NaN 和 Inf
        composite = composite.fill_nan(0.5).fill_null(0.5)
        
        logger.info(f"Composite score computed: mean={composite.mean():.4f}, std={composite.std():.4f}")
        
        return composite


class V195DataLoader:
    """
    V195 数据加载器 - 流式处理，避免 OOM
    """
    
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.engine = create_engine(db_url, pool_size=5, max_overflow=10)
    
    def load_year_data(self, year: int, chunk_size: int = 50000) -> pl.DataFrame:
        """
        加载指定年份的数据
        
        Args:
            year: 年份
            chunk_size: 每次读取的行数
            
        Returns:
            Polars DataFrame
        """
        logger.info(f"Loading data for year {year}...")
        
        query = text("""
            SELECT symbol, trade_date, close, pct_chg, volume, turnover_rate, high, low
            FROM stock_daily
            WHERE YEAR(trade_date) = :year
            ORDER BY trade_date, symbol
        """)
        
        chunks = []
        for chunk in pd.read_sql_query(query, self.engine, params={'year': year}, chunksize=chunk_size):
            # 转换为 Polars DataFrame
            pl_chunk = pl.from_pandas(chunk)
            chunks.append(pl_chunk)
            logger.info(f"  Loaded chunk: {len(pl_chunk):,} rows")
        
        if not chunks:
            logger.warning(f"No data for year {year}")
            return pl.DataFrame()
        
        df = pl.concat(chunks)
        logger.info(f"  Total loaded: {len(df):,} rows")
        
        # 数据校验
        stock_count = df['symbol'].n_unique()
        if stock_count < 4000:
            logger.warning(f"Stock count {stock_count} < 4000, may need data healing")
        
        return df
    
    def load_date_range_data(
        self,
        start_date: str,
        end_date: str,
        chunk_size: int = 50000
    ) -> pl.DataFrame:
        """
        加载指定日期范围的数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            chunk_size: 每次读取的行数
            
        Returns:
            Polars DataFrame
        """
        logger.info(f"Loading data from {start_date} to {end_date}...")
        
        query = text("""
            SELECT symbol, trade_date, close, pct_chg, volume, turnover_rate, high, low
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
            logger.warning("No data found")
            return pl.DataFrame()
        
        df = pl.concat(chunks)
        logger.info(f"  Total loaded: {len(df):,} rows")
        
        return df


def generate_signals(
    df: pl.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    output_path: str = "signals.csv"
) -> pl.DataFrame:
    """
    生成交易信号并输出到 CSV
    
    Args:
        df: 包含行情数据的 DataFrame
        weights: 因子权重
        output_path: 输出文件路径
        
    Returns:
        包含信号的 DataFrame
    """
    calculator = V195FactorCalculator(weights)
    
    # 计算综合得分
    score = calculator.compute_composite_score(df)
    
    # 创建输出 DataFrame
    output = df.select(["symbol", "trade_date"]).with_columns(
        score.alias("score"),
        pl.lit(VERSION).alias("version")
    )
    
    # 按日期和得分排序
    output = output.sort(["trade_date", "score"], descending=[False, True])
    
    # 输出到 CSV
    output.write_csv(output_path)
    logger.info(f"Signals saved to {output_path}")
    
    # 输出因子日志
    logger.info("Factor log:")
    for log in calculator.factor_log:
        logger.info(f"  {log['factor']}: {log['formula']} (weight={log['weight']})")
    
    return output


def main():
    """主函数 - 示例用法"""
    import pandas as pd  # 用于 SQL 读取
    
    print("=" * 70)
    print("V195 Strategy - 因子动物园线性组合")
    print("=" * 70)
    
    # 默认权重
    weights = DEFAULT_WEIGHTS.copy()
    print(f"\n因子权重配置:")
    for factor, weight in weights.items():
        print(f"  {factor}: {weight}")
    
    # 加载数据
    loader = V195DataLoader()
    
    # 示例：加载 2023-2025 年数据
    years = [2023, 2024, 2025]
    all_data = []
    
    for year in years:
        df = loader.load_year_data(year)
        if not df.is_empty():
            all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded")
        return
    
    full_df = pl.concat(all_data)
    logger.info(f"Total data: {len(full_df):,} rows")
    
    # 生成信号
    output = generate_signals(full_df, weights, output_path="signals.csv")
    
    print("\n" + "=" * 70)
    print("V195 Strategy completed. signals.csv generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()