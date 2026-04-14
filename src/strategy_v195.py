"""
V195 Strategy (选手脚本) - R9: 激进因子组合

【核心逻辑 - A 股特性增强版】
基于 A 股市场微观结构，引入强预测力因子：
1. 1 日反转 (Reversal_1): A 股超短期反转效应极强
2. 小市值 (Small Cap): 小市值股票长期超额收益
3. 特异度 (Idiosyncratic): 低特异波动率股票表现更好
4. 量价背离 (Price_Volume_Divergence): 价升量缩或价跌量增代表反转信号

【核心公式】
Score = w1 * Rank(Reversal_1) + w2 * Rank(Small_Cap) + w3 * Rank(1/IdioVol) + w4 * Rank(PV_Divergence)

权重配置：
w1=0.30 (1 日反转), w2=0.25 (小市值), w3=0.25 (特异度), w4=0.20 (量价背离)

【物理架构解耦】
1. 本脚本仅负责：读取数据、计算因子、输出 signals.csv
2. 严禁调用裁判脚本的任何内部函数
3. 严禁执行回测逻辑
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

# 配置
VERSION = "V195_R9"  # Round 9: 激进因子组合

# 权重配置
WEIGHT_REVERSAL_1 = 0.30
WEIGHT_SMALL_CAP = 0.25
WEIGHT_IDIOVOL = 0.25
WEIGHT_PV_DIVERGENCE = 0.20

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class V195FactorCalculator:
    """
    V195 因子计算器 - R9: 激进因子组合
    """
    
    def __init__(self):
        self.factor_log = []
        self.market_cap_cache = {}
    
    def compute_reversal_1_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Reversal_1: -1 * pct_chg
        
        A 股超短期反转：昨日跌幅越大，今日反弹概率越高
        取负号使得昨日跌幅大的股票得分高
        """
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 使用昨日涨跌幅作为反转因子 (shift(1) 确保使用昨日数据)
        df['ret_1'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x.shift(1))
        
        # 反转因子：收益率取负
        df['reversal_1'] = -df['ret_1']
        
        # 横截面排名 (0-1)
        df['reversal_1_rank'] = df.groupby('trade_date')['reversal_1'].transform(
            lambda x: x.rank(pct=True, ascending=False)
        )
        
        self.factor_log.append({
            'factor': 'reversal_1',
            'formula': '-1 * pct_chg (yesterday)',
            'weight': WEIGHT_REVERSAL_1,
            'type': 'rank',
            'effect': 'ultra short-term reversal (超短期反转)'
        })
        
        return df['reversal_1_rank']
    
    def compute_small_cap_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Small Cap: -1 * log(market_cap)
        
        小市值效应：市值越小，得分越高
        使用 close * volume 作为市值代理变量
        """
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 使用成交额作为市值代理 (close * volume)
        df['market_value_proxy'] = df['close'] * df['volume']
        
        # 取对数
        df['log_mv'] = np.log1p(df['market_value_proxy'])
        
        # 横截面排名 (ascending=True 使得小市值排名高)
        df['small_cap_rank'] = df.groupby('trade_date')['log_mv'].transform(
            lambda x: x.rank(pct=True, ascending=True)
        )
        
        self.factor_log.append({
            'factor': 'small_cap',
            'formula': '-1 * log(close * volume)',
            'weight': WEIGHT_SMALL_CAP,
            'type': 'rank',
            'effect': 'small cap premium (小市值溢价)'
        })
        
        return df['small_cap_rank']
    
    def compute_idiovol_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Idiosyncratic Volatility: 1 / std(ret, 20)
        
        特异度因子：特异波动率越低，得分越高
        """
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算 20 日收益率标准差 (使用 shift(1) 避免未来函数)
        df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x.shift(1))
        df['idiovol_20'] = df.groupby('symbol')['ret'].transform(
            lambda x: x.rolling(window=20, min_periods=10).std()
        )
        
        # 计算 1 / idiovol
        df['inv_idiovol'] = 1.0 / (df['idiovol_20'] + 1e-6)
        
        # 横截面排名 (0-1)
        df['idiovol_rank'] = df.groupby('trade_date')['inv_idiovol'].transform(
            lambda x: x.rank(pct=True, ascending=False)
        )
        
        self.factor_log.append({
            'factor': 'idiovol_20',
            'formula': '1 / std(ret, 20)',
            'weight': WEIGHT_IDIOVOL,
            'type': 'rank',
            'effect': 'low idiosyncratic volatility (低特异度)'
        })
        
        return df['idiovol_rank']
    
    def compute_pv_divergence_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Price-Volume Divergence: -1 * corr(pct_chg, turnover_rate, 20)
        
        量价背离：价格与成交量相关性越低，越可能出现反转
        负相关代表价跌量增或价升量缩，是买入信号
        """
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算 20 日量价相关性
        def rolling_corr(x):
            if len(x.dropna()) < 10:
                return np.nan
            return x['pct_chg'].rolling(window=20, min_periods=10).corr(x['turnover_rate'])
        
        # 使用 shift(1) 避免未来函数
        df['pv_corr'] = df.groupby('symbol').apply(
            lambda g: g.shift(1)[['pct_chg', 'turnover_rate']].rolling(
                window=20, min_periods=10
            ).corr().iloc[::2, 0].values if len(g) > 20 else np.nan
        ).reset_index(level=0, drop=True)
        
        # 简化版本：直接使用量价变化方向
        df['price_change'] = df.groupby('symbol')['close'].pct_change().shift(1)
        df['volume_change'] = df.groupby('symbol')['volume'].pct_change().shift(1)
        
        # 量价背离：价格与成交量变化方向相反
        df['pv_divergence'] = -df['price_change'] * df['volume_change']
        
        # 横截面排名 (0-1)
        df['pv_div_rank'] = df.groupby('trade_date')['pv_divergence'].transform(
            lambda x: x.rank(pct=True, ascending=False)
        )
        
        self.factor_log.append({
            'factor': 'pv_divergence',
            'formula': '-1 * price_change * volume_change',
            'weight': WEIGHT_PV_DIVERGENCE,
            'type': 'rank',
            'effect': 'price-volume divergence (量价背离)'
        })
        
        return df['pv_div_rank']
    
    def compute_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算综合得分：加权线性组合
        """
        logger.info("Computing composite score: aggressive factor combination...")
        
        # 计算各因子
        reversal_1_rank = self.compute_reversal_1_factor(df).values
        small_cap_rank = self.compute_small_cap_factor(df).values
        idiovol_rank = self.compute_idiovol_factor(df).values
        pv_div_rank = self.compute_pv_divergence_factor(df).values
        
        # 加权组合
        composite = (
            WEIGHT_REVERSAL_1 * reversal_1_rank +
            WEIGHT_SMALL_CAP * small_cap_rank +
            WEIGHT_IDIOVOL * idiovol_rank +
            WEIGHT_PV_DIVERGENCE * pv_div_rank
        )
        
        # 处理 NaN
        median_score = np.nanmedian(composite)
        if np.isnan(median_score):
            median_score = 0.25
        composite = np.nan_to_num(composite, nan=median_score)
        
        logger.info(f"Composite score computed: mean={np.mean(composite):.4f}, std={np.std(composite):.4f}")
        
        self.factor_log.append({
            'factor': 'composite_score',
            'formula': f'{WEIGHT_REVERSAL_1}*Rev1 + {WEIGHT_SMALL_CAP}*Cap + {WEIGHT_IDIOVOL}*Idio + {WEIGHT_PV_DIVERGENCE}*PVDiv',
            'type': 'linear_combination',
            'effect': 'aggressive factor fusion'
        })
        
        return composite


class V195DataLoader:
    """
    V195 数据加载器 - 流式处理，避免 OOM
    """
    
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.engine = create_engine(db_url, pool_size=5, max_overflow=10)
    
    def load_year_data(self, year: int, chunk_size: int = 50000) -> pd.DataFrame:
        """
        加载指定年份的数据
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
            chunks.append(chunk)
            logger.info(f"  Loaded chunk: {len(chunk):,} rows")
        
        if not chunks:
            logger.warning(f"No data for year {year}")
            return pd.DataFrame()
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Total loaded: {len(df):,} rows")
        
        # 数据校验
        stock_count = df['symbol'].nunique()
        if stock_count < 4000:
            logger.warning(f"Stock count {stock_count} < 4000, may need data healing")
        
        return df
    
    def load_date_range_data(
        self,
        start_date: str,
        end_date: str,
        chunk_size: int = 50000
    ) -> pd.DataFrame:
        """
        加载指定日期范围的数据
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
            chunks.append(chunk)
        
        if not chunks:
            logger.warning("No data found")
            return pd.DataFrame()
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Total loaded: {len(df):,} rows")
        
        return df


def generate_signals(
    df: pd.DataFrame,
    output_path: str = "signals.csv"
) -> pd.DataFrame:
    """
    生成交易信号并输出到 CSV
    """
    calculator = V195FactorCalculator()
    
    # 确保数据类型正确
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['pct_chg'] = df['pct_chg'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['turnover_rate'] = df['turnover_rate'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # 计算综合得分
    score = calculator.compute_composite_score(df)
    
    # 创建输出 DataFrame
    output = pd.DataFrame({
        'symbol': df['symbol'].values,
        'trade_date': df['trade_date'].values,
        'score': score,
        'version': VERSION
    })
    
    # 按日期和得分排序
    output = output.sort_values(['trade_date', 'score'], ascending=[True, False])
    
    # 输出到 CSV
    output.to_csv(output_path, index=False)
    logger.info(f"Signals saved to {output_path}")
    
    # 输出因子日志
    logger.info("Factor log:")
    for log in calculator.factor_log:
        weight_str = f" (weight={log['weight']})" if 'weight' in log else ""
        logger.info(f"  {log['factor']}: {log['formula']}{weight_str} - {log.get('effect', '')}")
    
    # 输出分数分布
    logger.info(f"Score distribution: mean={output['score'].mean():.4f}, std={output['score'].std():.4f}")
    logger.info(f"Score range: [{output['score'].min():.4f}, {output['score'].max():.4f}]")
    
    return output


def main():
    """主函数 - 示例用法"""
    print("=" * 70)
    print("V195 Strategy - R9: 激进因子组合")
    print("=" * 70)
    
    print(f"\n核心公式：Score = {WEIGHT_REVERSAL_1}*Rev1 + {WEIGHT_SMALL_CAP}*Cap + {WEIGHT_IDIOVOL}*Idio + {WEIGHT_PV_DIVERGENCE}*PVDiv")
    print("A 股逻辑：1 日反转 + 小市值 + 低特异度 + 量价背离 四因子融合")
    
    # 加载数据
    loader = V195DataLoader()
    
    # 示例：加载 2023-2025 年数据
    years = [2023, 2024, 2025]
    all_data = []
    
    for year in years:
        df = loader.load_year_data(year)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded")
        return
    
    full_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total data: {len(full_df):,} rows")
    
    # 生成信号
    output = generate_signals(full_df, output_path="signals.csv")
    
    print("\n" + "=" * 70)
    print("V195 Strategy completed. signals.csv generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()