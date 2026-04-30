"""
Alpha Model V228 - Nonlinear Interaction: Momentum-Confirmed Reversal
=====================================================================

【核心架构 - Referee-Player】
这是 Player (选手) 模块，唯一职责是计算 Alpha 评分。

【唯一接口】
- compute_score(df: pd.DataFrame) -> pd.DataFrame
  输入: 包含 OHLCV 等基础数据
  输出: 包含 trade_date, symbol, score 的 DataFrame

【V227 失败原因分析】
- 2024年三个因子方向错误：
  1. 超卖股不反弹（牛市中超卖股票可能基本面差，无人抄底）
  2. 放量下跌是洗盘（主力资金在洗盘，但V227无法区分）
  3. 低波动跑输高波动（牛市中高波动股票弹性更大）
- 线性组合无法在牛市和熊市同时有效

【V228 核心改进 - 非线性交互项】
假设 H1: 动量确认反转 (Momentum-Confirmed Reversal)
- 经济逻辑：在牛市中，真正反弹的股票是那些前期超卖但近期已显现正动量的股票
- 单纯超卖 = 可能基本面差；超卖 + 正动量 = 资金已开始抄底
- 交互项：f_momentum_confirmed = f_extreme_os * sign(momentum_20d) * |momentum_20d|
- 这样在牛市中，超卖且有正动量的股票得分最高

1. 极端超卖因子 (Extreme Oversold):
   - 5 日累计跌幅（负收益 = 潜在反弹空间）
   - 使用截面百分位排名

2. 动量确认因子 (Momentum Confirmation):
   - 20日收益率（正动量 = 资金已开始流入）
   - 与超卖因子形成交互：只有超卖+正动量才给高分

3. 动量确认反转交互项 (Momentum-Confirmed Reversal):
   - f_extreme_os * clip(momentum_20d, 0, inf)
   - 即：超卖且正动量的股票得分最高
   - 超卖但负动量 = 基本面差，不选
   - 超卖且正动量 = 资金抄底，强选

【组合逻辑】
- 极端超卖 (40%) + 放量下跌 (30%) + 动量确认反转 (30%)
- 使用截面百分位排名
- 移除了低波动因子（牛市中失效）

【合规锁定】
- 严禁任何 shift(-1) 或 next_ret 引用
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
VERSION = "V228"


class AlphaModelV228:
    """
    V228 Alpha Model - Player (选手)
    
    【核心职责】
    1. 极端超卖: 5日累计跌幅
    2. 放量下跌: 下跌日的成交量放大
    3. 动量确认反转: 超卖 * 正动量的交互项
    
    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score]
    """
    
    # ==================== 配置参数 ====================
    # 因子权重 (3 因子线性组合，包含交互项)
    W_EXTREME_OS = 0.40        # 极端超卖 (基础信号)
    W_VOL_SURGE = 0.30         # 放量下跌确认
    W_MOM_CONFIRMED = 0.30     # 动量确认反转交互项
    
    # 窗口参数
    OVERSOLD_WINDOW = 5        # 超卖窗口（5日累计）
    VOL_WINDOW = 5             # 成交量均线窗口
    MOMENTUM_WINDOW = 20       # 动量窗口
    
    def __init__(self):
        """初始化 Alpha Model"""
        logger.info("=" * 70)
        logger.info("V228 Alpha Model Initialized (Momentum-Confirmed Reversal)")
        logger.info("=" * 70)
        logger.info(f"  Extreme Oversold Weight: {self.W_EXTREME_OS:.2f}")
        logger.info(f"  Volume Surge Weight: {self.W_VOL_SURGE:.2f}")
        logger.info(f"  Momentum-Confirmed Weight: {self.W_MOM_CONFIRMED:.2f}")
        logger.info(f"  Oversold Window: {self.OVERSOLD_WINDOW}")
        logger.info(f"  Volume Window: {self.VOL_WINDOW}")
        logger.info(f"  Momentum Window: {self.MOMENTUM_WINDOW}")
        logger.info("=" * 70)
    
    # ==================== 唯一公开接口 ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)
        
        【接口规范】
        - 输入: pd.DataFrame with columns [trade_date, symbol, open, high, low, close, volume, amount]
        - 输出: pd.DataFrame with columns [trade_date, symbol, score]
        
        【计算流程】
        1. 数据预处理: 计算收益率、成交量比率等
        2. 极端超卖因子
        3. 放量下跌因子
        4. 动量确认反转交互项
        5. 截面百分位排名标准化
        6. 加权合成
        7. 最终截面 Rank
        """
        logger.info(f"[V228] Computing alpha scores for {len(df)} rows...")
        sys.stdout.flush()
        
        # 1. 数据预处理
        df = self._preprocess_data(df)
        sys.stdout.flush()
        
        # 2. 计算各因子
        logger.info("[V228] Computing extreme oversold factor...")
        df['f_extreme_os'] = self._compute_extreme_oversold(df)
        sys.stdout.flush()
        
        logger.info("[V228] Computing volume surge on decline factor...")
        df['f_vol_surge'] = self._compute_volume_surge(df)
        sys.stdout.flush()
        
        logger.info("[V228] Computing momentum-confirmed reversal interaction...")
        df['f_mom_confirmed'] = self._compute_momentum_confirmed(df)
        sys.stdout.flush()
        
        # 3. 截面百分位排名标准化
        logger.info("[V228] Normalizing factors with cross-sectional percentile rank...")
        factor_cols = ['f_extreme_os', 'f_vol_surge', 'f_mom_confirmed']
        for col in factor_cols:
            df[col + '_rank'] = self._cross_sectional_rank(df, col)
        sys.stdout.flush()
        
        # 4. 因子诊断
        for col in factor_cols:
            non_zero = (df[col] != 0).sum()
            non_nan = df[col].notna().sum()
            logger.info(f"[Factor Debug] {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, non_zero={non_zero}, non_nan={non_nan}")
        sys.stdout.flush()
        
        # 5. 加权合成
        logger.info("[V228] Combining factors with fixed weights...")
        df['score_raw'] = (
            self.W_EXTREME_OS * df['f_extreme_os_rank'] +
            self.W_VOL_SURGE * df['f_vol_surge_rank'] +
            self.W_MOM_CONFIRMED * df['f_mom_confirmed_rank']
        )
        sys.stdout.flush()
        
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
        
        logger.info(f"[V228] Score computed: mean={result['score'].mean():.4f}, std={result['score'].std():.4f}")
        sys.stdout.flush()
        
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
        
        # 计算 N 日累计收益率
        result[f'ret_{self.OVERSOLD_WINDOW}d'] = result.groupby('symbol')['close'].transform(
            lambda x, d=self.OVERSOLD_WINDOW: (x / x.shift(d) - 1)
        )
        
        # 计算动量窗口收益率
        result[f'ret_{self.MOMENTUM_WINDOW}d'] = result.groupby('symbol')['close'].transform(
            lambda x, d=self.MOMENTUM_WINDOW: (x / x.shift(d) - 1)
        )
        
        # 计算成交量移动平均
        result[f'vol_ma_{self.VOL_WINDOW}d'] = result.groupby('symbol')['volume'].transform(
            lambda x, d=self.VOL_WINDOW: x.rolling(d, min_periods=3).mean()
        )
        
        # 成交量比率
        result['vol_ratio'] = result['volume'] / result[f'vol_ma_{self.VOL_WINDOW}d'].replace(0, np.nan)
        
        # 是否下跌日
        result['is_down_day'] = (result['close'] < result['open']).astype(float)
        
        # 填充 NaN (使用 data_healer)
        heal_cols = ['daily_ret', f'ret_{self.OVERSOLD_WINDOW}d', f'ret_{self.MOMENTUM_WINDOW}d', 
                     'vol_ratio', 'is_down_day']
        
        result = result.replace([np.inf, -np.inf], np.nan)
        available_cols = [c for c in heal_cols if c in result.columns]
        if available_cols:
            result = heal(result, numeric_cols=available_cols)
        
        return result
    
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
        - 累计跌幅越大，信号越强（负收益 = 正信号）
        """
        ret_nd = df.get(f'ret_{self.OVERSOLD_WINDOW}d', pd.Series(0, index=df.index))
        
        # 反转信号：负收益 = 正信号
        factor = -ret_nd
        
        return factor
    
    def _compute_volume_surge(self, df: pd.DataFrame) -> pd.Series:
        """
        放量下跌确认因子
        
        【逻辑】
        - 仅在下跌日（close < open）时有效
        - 成交量放大倍数越高，信号越强
        - 上涨日给中性值
        """
        vol_ratio = df['vol_ratio'].fillna(1.0)
        is_down = df['is_down_day'].fillna(0.5)
        
        # 放量下跌信号：下跌日 * 成交量比率
        factor = vol_ratio * is_down
        
        return factor
    
    def _compute_momentum_confirmed(self, df: pd.DataFrame) -> pd.Series:
        """
        动量确认反转交互项
        
        【逻辑】
        - 交互项：f_extreme_os * clip(momentum_20d, 0, inf)
        - 超卖且正动量 = 资金已开始抄底，强信号
        - 超卖但负动量 = 基本面差，弱信号
        - 不超卖 = 无意义
        
        【经济解释】
        - 在牛市中，单纯超卖可能因为基本面差
        - 但如果超卖股票已经出现正动量（资金开始流入）
        - 说明是错杀，反弹概率更高
        """
        ret_nd = df.get(f'ret_{self.OVERSOLD_WINDOW}d', pd.Series(0, index=df.index))
        ret_20d = df.get(f'ret_{self.MOMENTUM_WINDOW}d', pd.Series(0, index=df.index))
        
        # 超卖信号（负收益 = 超卖）
        oversold_signal = -ret_nd
        
        # 动量确认（只取正动量，负动量设为0）
        # 这样交互项只在正动量时有效
        momentum_confirm = ret_20d.clip(lower=0)
        
        # 交互项：超卖 * 正动量确认
        interaction = oversold_signal * momentum_confirm
        
        return interaction
    
    # ==================== 辅助方法 ====================
    
    def get_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取各因子的 IC"""
        ics = {}
        if 't1_return' not in df.columns:
            return ics
        
        for col in ['f_extreme_os', 'f_vol_surge', 'f_mom_confirmed', 'score']:
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


def get_alpha_model() -> AlphaModelV228:
    """获取 Alpha Model 实例"""
    return AlphaModelV228()


if __name__ == "__main__":
    logger.info("V228 Alpha Model loaded successfully")
    model = get_alpha_model()
    logger.info(f"Version: {VERSION}")