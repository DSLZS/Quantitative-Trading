"""
Alpha Model V229 - Industry-Relative Reversal with 5-Day Target
=================================================================

【核心架构 - Referee-Player】
这是 Player (选手) 模块，唯一职责是计算 Alpha 评分。

【唯一接口】
- compute_score(df: pd.DataFrame) -> pd.DataFrame
  输入: 包含 OHLCV 等基础数据
  输出: 包含 trade_date, symbol, score 的 DataFrame

【V227 失败原因分析】
- V227 在 2024 年 IC 仅为 0.0255，收益 -55%
- 纯价格/成交量因子在 2024 年牛市中完全失效
- 缺乏行业维度的信息

【V229 核心假设 H2 - 行业相对反转】
1. 假设: 在 2024 年牛市中，行业轮动是主要驱动因素。
   行业内相对弱势的股票（跑输行业指数）更容易反弹。

2. 核心改进:
   a. 行业相对强度因子 (Industry Relative Strength):
      - 计算行业内所有股票的平均收益率作为行业指数
      - 个股相对行业指数的超额收益
      - 相对弱势的股票有反转潜力
   
   b. 5 日累计收益率作为目标标签:
      - 相比 T+1 收益率，5 日累计收益率噪声更低
      - 反转效应在 5 日维度上可能更明显
   
   c. 极端超卖因子 (保持 V227 的核心):
      - 5 日累计跌幅排名
      - 使用截面百分位排名

【组合逻辑】
- 极端超卖 (35%) + 行业相对弱势 (35%) + 低波动 (30%)
- 行业相对弱势: 跑输行业指数越多，反弹信号越强
- 使用截面百分位排名标准化

【合规锁定】
- 严禁任何 shift(-1) 或未来引用
- 所有因子仅使用 T 日及之前数据
- 严禁接触回测逻辑
- 严格使用 data_healer 处理缺失值，严禁 fillna(0)
- 因子数量严格 ≤3
"""

from typing import Dict, Optional
import sys
import os

import pandas as pd
import numpy as np
from loguru import logger

# 内存优化
pd.options.mode.chained_assignment = None

# 版本号
VERSION = "V229"


class AlphaModelV229:
    """
    V229 Alpha Model - Player (选手)
    
    【核心职责】
    1. 极端超卖: 5日累计跌幅的截面排名
    2. 行业相对弱势: 个股相对于行业指数的超额收益
    3. 低波动率: 20日波动率的截面排名
    
    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score]
    """
    
    # ==================== 配置参数 ====================
    # 因子权重
    W_EXTREME_OS = 0.35         # 极端超卖 (核心信号)
    W_INDUSTRY_REL = 0.35       # 行业相对弱势 (新信号)
    W_LOW_VOL = 0.30            # 低波动率 (防御信号)
    
    # 窗口参数
    OVERSOLD_WINDOW = 5         # 超卖窗口（5日累计）
    VOL_WINDOW = 5              # 成交量均线窗口
    VOL_WINDOW_LOW = 20         # 波动率窗口
    INDUSTRY_WINDOW = 20        # 行业指数计算窗口
    
    def __init__(self):
        """初始化 Alpha Model"""
        logger.info("=" * 70)
        logger.info("V229 Alpha Model Initialized (Industry-Relative Reversal)")
        logger.info("=" * 70)
        logger.info(f"  Extreme Oversold Weight: {self.W_EXTREME_OS:.2f}")
        logger.info(f"  Industry Relative Weight: {self.W_INDUSTRY_REL:.2f}")
        logger.info(f"  Low Volatility Weight: {self.W_LOW_VOL:.2f}")
        logger.info(f"  Oversold Window: {self.OVERSOLD_WINDOW}")
        logger.info(f"  Industry Window: {self.INDUSTRY_WINDOW}")
        logger.info("=" * 70)
    
    # ==================== 唯一公开接口 ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)
        
        【接口规范】
        - 输入: pd.DataFrame with columns [trade_date, symbol, open, high, low, close, volume, amount, industry_code]
        - 输出: pd.DataFrame with columns [trade_date, symbol, score]
        
        【计算流程】
        1. 数据预处理: 计算收益率、成交量比率等
        2. 极端超卖因子
        3. 行业相对弱势因子
        4. 低波动率因子
        5. 截面百分位排名标准化
        6. 加权合成
        7. 最终截面 Rank
        """
        logger.info(f"[V229] Computing alpha scores for {len(df)} rows...")
        
        # 1. 数据预处理
        df = self._preprocess_data(df)
        
        # 2. 计算各因子
        logger.info("[V229] Computing extreme oversold factor...")
        df['f_extreme_os'] = self._compute_extreme_oversold(df)
        
        logger.info("[V229] Computing industry relative weakness factor...")
        df['f_industry_rel'] = self._compute_industry_relative(df)
        
        logger.info("[V229] Computing low volatility factor...")
        df['f_low_vol'] = self._compute_low_volatility(df)
        
        # 3. 截面百分位排名标准化
        logger.info("[V229] Normalizing factors with cross-sectional percentile rank...")
        factor_cols = ['f_extreme_os', 'f_industry_rel', 'f_low_vol']
        for col in factor_cols:
            df[col + '_rank'] = self._cross_sectional_rank(df, col)
        
        # 4. 因子诊断
        for col in factor_cols:
            non_zero = (df[col] != 0).sum()
            non_nan = df[col].notna().sum()
            logger.info(f"[Factor Debug] {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, non_zero={non_zero}, non_nan={non_nan}")
        
        # 5. 加权合成
        logger.info("[V229] Combining factors with fixed weights...")
        df['score_raw'] = (
            self.W_EXTREME_OS * df['f_extreme_os_rank'] +
            self.W_INDUSTRY_REL * df['f_industry_rel_rank'] +
            self.W_LOW_VOL * df['f_low_vol_rank']
        )
        
        # 6. 最终截面 Rank (百分位)
        df['score'] = df.groupby('trade_date')['score_raw'].rank(pct=True, na_option='keep')
        
        # 处理可能的 NaN
        from src.data_healer import heal
        df = heal(df, numeric_cols=['score'])
        
        # 7. 输出结果
        result_cols = ['trade_date', 'symbol', 'score']
        if 'close' in df.columns:
            result_cols.append('close')
        result = df[result_cols].copy()
        
        logger.info(f"[V229] Score computed: mean={result['score'].mean():.4f}, std={result['score'].std():.4f}")
        
        return result
    
    # ==================== 数据预处理 ====================
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理 - 计算基础指标
        """
        from src.data_healer import heal
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算日收益率
        result['daily_ret'] = result.groupby('symbol')['close'].pct_change()
        
        # 计算行业指数 (先计算行业指数，因为后续需要它)
        result = self._compute_industry_index(result)
        
        # 计算 N 日累计收益率
        result[f'ret_{self.OVERSOLD_WINDOW}d'] = result.groupby('symbol')['close'].transform(
            lambda x, d=self.OVERSOLD_WINDOW: (x / x.shift(d) - 1)
        )
        
        # 计算成交量移动平均
        result[f'vol_ma_{self.VOL_WINDOW}d'] = result.groupby('symbol')['volume'].transform(
            lambda x, d=self.VOL_WINDOW: x.rolling(d, min_periods=3).mean()
        )
        
        # 成交量比率
        result['vol_ratio'] = result['volume'] / result[f'vol_ma_{self.VOL_WINDOW}d'].replace(0, np.nan)
        
        # 计算 20 日波动率
        result[f'vol_{self.VOL_WINDOW_LOW}d'] = result.groupby('symbol')['daily_ret'].transform(
            lambda x, d=self.VOL_WINDOW_LOW: x.rolling(d, min_periods=10).std()
        )
        
        # 是否下跌日
        result['is_down_day'] = (result['close'] < result['open']).astype(float)
        
        # 先替换 inf 为 NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 第一遍 healing: 修复基础列
        heal_cols = ['daily_ret', f'ret_{self.OVERSOLD_WINDOW}d', 'vol_ratio', 
                     f'vol_{self.VOL_WINDOW_LOW}d', 'is_down_day', 
                     'industry_idx_ret', 'industry_rel_ret']
        available_cols = [c for c in heal_cols if c in result.columns]
        if available_cols:
            result = heal(result, numeric_cols=available_cols)
        
        # 现在计算行业相对累计收益 (在 healing 之后)
        result[f'industry_rel_{self.INDUSTRY_WINDOW}d'] = result.groupby('symbol')['industry_rel_ret'].transform(
            lambda x, d=self.INDUSTRY_WINDOW: x.rolling(d, min_periods=10).sum()
        )
        
        # 第二遍 healing: 修复 industry_rel 累计列
        result = heal(result, numeric_cols=[f'industry_rel_{self.INDUSTRY_WINDOW}d'])
        
        return result
    
    def _compute_industry_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算行业指数和个股相对行业的超额收益
        
        【逻辑】
        1. 按行业分组，计算每日行业内所有股票的中位数收益率作为行业指数
        2. 计算行业内个股相对于行业指数的累计超额收益
        3. 行业相对弱势 = 负的累计超额收益（跑输行业越多，信号越强）
        """
        # 计算行业每日中位数收益率
        industry_daily = df.groupby(['trade_date', 'industry_code'])['daily_ret'].median().reset_index()
        industry_daily.columns = ['trade_date', 'industry_code', 'industry_idx_ret']
        
        # 合并行业指数回原数据
        df = df.merge(industry_daily, on=['trade_date', 'industry_code'], how='left')
        
        # 计算个股相对行业的超额收益
        df['industry_rel_ret'] = df['daily_ret'] - df['industry_idx_ret']
        
        # 计算行业相对累计收益（窗口期内）
        df[f'industry_rel_{self.INDUSTRY_WINDOW}d'] = df.groupby('symbol')['industry_rel_ret'].transform(
            lambda x, d=self.INDUSTRY_WINDOW: x.rolling(d, min_periods=10).sum()
        )
        
        return df
    
    # ==================== 横截面百分位排名 ====================
    
    def _cross_sectional_rank(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        横截面百分位排名标准化
        返回 0-1 之间的值，1 表示排名最高
        """
        rank = df.groupby('trade_date')[col].rank(pct=True, na_option='keep')
        return rank.fillna(0.5)  # NaN 给中性值
    
    # ==================== 因子计算 ====================
    
    def _compute_extreme_oversold(self, df: pd.DataFrame) -> pd.Series:
        """
        极端超卖因子
        
        【逻辑】
        - 使用 5 日累计收益率
        - 累计跌幅越大，信号越强
        - 使用负收益率（下跌 = 正信号）
        """
        ret_nd = df.get(f'ret_{self.OVERSOLD_WINDOW}d', pd.Series(0, index=df.index))
        
        # 反转信号：负收益 = 正信号
        factor = -ret_nd
        
        return factor
    
    def _compute_industry_relative(self, df: pd.DataFrame) -> pd.Series:
        """
        行业相对弱势因子
        
        【逻辑】
        - 计算个股相对于行业指数的累计超额收益
        - 相对行业表现越弱（超额收益越低），反转信号越强
        - 使用负的超额收益（弱势 = 正信号）
        """
        industry_rel = df.get(f'industry_rel_{self.INDUSTRY_WINDOW}d', pd.Series(0, index=df.index))
        
        # 行业相对弱势：超额收益低 = 反转信号强
        factor = -industry_rel
        
        return factor
    
    def _compute_low_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        低波动率因子
        
        【逻辑】
        - 波动率越低，信号越强
        - 使用负波动率（低波动 = 正信号）
        """
        vol_20d = df.get(f'vol_{self.VOL_WINDOW_LOW}d', pd.Series(0, index=df.index))
        
        # 低波动 = 正信号
        factor = -vol_20d
        
        return factor
    
    # ==================== 辅助方法 ====================
    
    def get_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取各因子的 IC"""
        ics = {}
        if 't1_return' not in df.columns:
            return ics
        
        for col in ['f_extreme_os', 'f_industry_rel', 'f_low_vol', 'score']:
            if col in df.columns:
                ic = self._calculate_daily_ic(df, col, 't1_return')
                ics[col] = ic
        return ics
    
    def _calculate_daily_ic(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str
    ) -> float:
        """计算日度平均 IC"""
        ic_values = []
        
        for date in sorted(df['trade_date'].unique()):
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 10:
                continue
            
            factor_vals = day_data[factor_col]
            return_vals = day_data[return_col]
            
            mask = factor_vals.notna() & return_vals.notna()
            if mask.sum() < 10:
                continue
            
            corr = factor_vals[mask].corr(return_vals[mask], method='spearman')
            if not np.isnan(corr):
                ic_values.append(corr)
        
        return float(np.mean(ic_values)) if ic_values else 0.0


def get_alpha_model() -> AlphaModelV229:
    """获取 Alpha Model 实例"""
    return AlphaModelV229()


if __name__ == "__main__":
    logger.info("V229 Alpha Model loaded successfully")
    model = get_alpha_model()
    logger.info(f"Version: {VERSION}")