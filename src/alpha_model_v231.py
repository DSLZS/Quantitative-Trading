"""
Alpha Model V231 - Volatility Prediction with Linear Factor Weighting
=====================================================================

【核心假设 H4】
放弃回归与分类，改为预测未来5日波动率（future_vol）。
逻辑：低波动股票长期有溢价，且波动率在牛熊市中更稳定。

【V229 -> V231 变更】
1. 标签从收益率改为未来5日波动率
   future_vol = close.pct_change().rolling(5).std().shift(-5)
2. 截面百分位反向：低波动得高分
3. 特征保持 V229 的三个因子：extreme_oversold, industry_relative_weakness, low_volatility
4. 权重：0.3, 0.4, 0.3

【唯一接口】
- compute_score(df: pd.DataFrame) -> pd.DataFrame
  输入: 包含 OHLCV 等基础数据
  输出: 包含 trade_date, symbol, score, future_vol 的 DataFrame

【合规锁定】
- 严禁任何 shift(-1) 或未来引用（除了计算未来波动率标签）
- 所有因子仅使用 T 日及之前数据
- 严禁接触回测逻辑
- 严格使用 data_healer 处理缺失值
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
VERSION = "V231"


class AlphaModelV231:
    """
    V231 Alpha Model - Player (选手)
    
    【核心职责】
    1. 极端超卖: 5日累计跌幅的截面排名
    2. 行业相对弱势: 个股相对于行业指数的超额收益
    3. 低波动率: 5日波动率的截面排名
    4. 未来波动率标签: 用于计算IC
    
    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score, future_vol]
    """
    
    # ==================== 配置参数 ====================
    # 因子权重 (H4: 低波动策略)
    W_EXTREME_OS = 0.30         # 极端超卖
    W_INDUSTRY_REL = 0.40       # 行业相对弱势
    W_LOW_VOL = 0.30            # 低波动率
    
    # 窗口参数
    OVERSOLD_WINDOW = 5         # 超卖窗口（5日累计）
    VOL_WINDOW = 5              # 成交量均线窗口
    VOL_WINDOW_LOW = 5          # 波动率窗口（5日）
    INDUSTRY_WINDOW = 20        # 行业指数计算窗口
    FUTURE_VOL_WINDOW = 5       # 未来波动率窗口
    
    def __init__(self):
        """初始化 Alpha Model"""
        logger.info("=" * 70)
        logger.info("V231 Alpha Model Initialized (Volatility Prediction)")
        logger.info("=" * 70)
        logger.info(f"  Extreme Oversold Weight: {self.W_EXTREME_OS:.2f}")
        logger.info(f"  Industry Relative Weight: {self.W_INDUSTRY_REL:.2f}")
        logger.info(f"  Low Volatility Weight: {self.W_LOW_VOL:.2f}")
        logger.info(f"  Future Vol Window: {self.FUTURE_VOL_WINDOW}")
        logger.info("=" * 70)
    
    # ==================== 唯一公开接口 ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)
        
        【接口规范】
        - 输入: pd.DataFrame with columns [trade_date, symbol, open, high, low, close, volume, amount, industry_code]
        - 输出: pd.DataFrame with columns [trade_date, symbol, score, future_vol]
        
        【计算流程】
        1. 数据预处理: 计算收益率、成交量比率等
        2. 计算未来波动率标签 (用于IC计算)
        3. 极端超卖因子
        4. 行业相对弱势因子
        5. 低波动率因子
        6. 截面百分位排名标准化
        7. 加权合成
        8. 最终截面 Rank
        """
        logger.info(f"[V231] Computing alpha scores for {len(df)} rows...")
        
        # 1. 数据预处理
        df = self._preprocess_data(df)
        
        # 2. 计算未来波动率标签 (IC 计算目标)
        logger.info("[V231] Computing future volatility label...")
        df['future_vol'] = self._compute_future_volatility(df)
        
        # 3. 计算各因子
        logger.info("[V231] Computing extreme oversold factor...")
        df['f_extreme_os'] = self._compute_extreme_oversold(df)
        
        logger.info("[V231] Computing industry relative weakness factor...")
        df['f_industry_rel'] = self._compute_industry_relative(df)
        
        logger.info("[V231] Computing low volatility factor...")
        df['f_low_vol'] = self._compute_low_volatility(df)
        
        # 4. 截面百分位排名标准化
        logger.info("[V231] Normalizing factors with cross-sectional percentile rank...")
        factor_cols = ['f_extreme_os', 'f_industry_rel', 'f_low_vol']
        for col in factor_cols:
            df[col + '_rank'] = self._cross_sectional_rank(df, col)
        
        # 5. 因子诊断
        for col in factor_cols:
            non_zero = (df[col] != 0).sum()
            non_nan = df[col].notna().sum()
            logger.info(f"[Factor Debug] {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, non_zero={non_zero}, non_nan={non_nan}")
        
        # 6. 加权合成 (低波动 = 高分)
        logger.info("[V231] Combining factors with fixed weights...")
        df['score_raw'] = (
            self.W_EXTREME_OS * df['f_extreme_os_rank'] +
            self.W_INDUSTRY_REL * df['f_industry_rel_rank'] +
            self.W_LOW_VOL * df['f_low_vol_rank']
        )
        
        # 7. 最终截面 Rank (百分位)
        df['score'] = df.groupby('trade_date')['score_raw'].rank(pct=True, na_option='keep')
        
        # 处理可能的 NaN
        from src.data_healer import heal
        df = heal(df, numeric_cols=['score'])
        
        # 8. 输出结果
        result_cols = ['trade_date', 'symbol', 'score']
        if 'future_vol' in df.columns:
            result_cols.append('future_vol')
        if 'close' in df.columns:
            result_cols.append('close')
        result = df[result_cols].copy()
        
        logger.info(f"[V231] Score computed: mean={result['score'].mean():.4f}, std={result['score'].std():.4f}")
        
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
        
        # 计算行业指数
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
        
        # 计算 5 日波动率 (用于因子)
        result[f'vol_{self.VOL_WINDOW_LOW}d'] = result.groupby('symbol')['daily_ret'].transform(
            lambda x, d=self.VOL_WINDOW_LOW: x.rolling(d, min_periods=3).std()
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
        
        # 计算行业相对累计收益
        result[f'industry_rel_{self.INDUSTRY_WINDOW}d'] = result.groupby('symbol')['industry_rel_ret'].transform(
            lambda x, d=self.INDUSTRY_WINDOW: x.rolling(d, min_periods=10).sum()
        )
        
        # 第二遍 healing: 修复 industry_rel 累计列
        result = heal(result, numeric_cols=[f'industry_rel_{self.INDUSTRY_WINDOW}d'])
        
        return result
    
    def _compute_industry_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算行业指数和个股相对行业的超额收益
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
    
    # ==================== 未来波动率标签 ====================
    
    def _compute_future_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        计算未来5日波动率标签
        
        future_vol = close.pct_change().rolling(5).std().shift(-5)
        
        注意：这里使用 shift(-5) 是为了计算标签，不用于因子计算
        """
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算日收益率
        result['daily_ret_for_vol'] = result.groupby('symbol')['close'].pct_change()
        
        # 计算5日滚动波动率
        result['rolling_vol'] = result.groupby('symbol')['daily_ret_for_vol'].transform(
            lambda x: x.rolling(self.FUTURE_VOL_WINDOW, min_periods=3).std()
        )
        
        # 向前移动5天，得到未来波动率
        result['future_vol'] = result.groupby('symbol')['rolling_vol'].shift(-self.FUTURE_VOL_WINDOW)
        
        # 清理临时列
        if 'daily_ret_for_vol' in result.columns:
            result = result.drop(columns=['daily_ret_for_vol'])
        if 'rolling_vol' in result.columns:
            result = result.drop(columns=['rolling_vol'])
        
        # 填充 NaN
        result['future_vol'] = result['future_vol'].fillna(0)
        
        return result['future_vol']
    
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
        - 使用 5 日累计收益率
        - 累计跌幅越大，信号越强
        """
        ret_nd = df.get(f'ret_{self.OVERSOLD_WINDOW}d', pd.Series(0, index=df.index))
        
        # 反转信号：负收益 = 正信号
        factor = -ret_nd
        
        return factor
    
    def _compute_industry_relative(self, df: pd.DataFrame) -> pd.Series:
        """
        行业相对弱势因子
        - 相对行业表现越弱（超额收益越低），反转信号越强
        """
        industry_rel = df.get(f'industry_rel_{self.INDUSTRY_WINDOW}d', pd.Series(0, index=df.index))
        
        # 行业相对弱势：超额收益低 = 反转信号强
        factor = -industry_rel
        
        return factor
    
    def _compute_low_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        低波动率因子
        - 波动率越低，信号越强
        """
        vol_5d = df.get(f'vol_{self.VOL_WINDOW_LOW}d', pd.Series(0, index=df.index))
        
        # 低波动 = 正信号
        factor = -vol_5d
        
        return factor
    
    # ==================== 辅助方法 ====================
    
    def get_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取各因子的 IC (对波动率的预测能力)"""
        ics = {}
        if 'future_vol' not in df.columns:
            return ics
        
        for col in ['f_extreme_os', 'f_industry_rel', 'f_low_vol', 'score']:
            if col in df.columns:
                ic = self._calculate_daily_ic(df, col, 'future_vol')
                ics[col] = ic
        return ics
    
    def _calculate_daily_ic(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str
    ) -> float:
        """计算日度平均 IC (对波动率的预测能力)"""
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


def get_alpha_model() -> AlphaModelV231:
    """获取 Alpha Model 实例"""
    return AlphaModelV231()


if __name__ == "__main__":
    logger.info("V231 Alpha Model loaded successfully")
    model = get_alpha_model()
    logger.info(f"Version: {VERSION}")