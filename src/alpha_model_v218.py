"""
Alpha Model V218 - Market State Adapter + Feature Decoupling
=============================================================

【核心架构 - Referee-Player】
这是 Player (选手) 模块，唯一职责是计算 Alpha 评分。

【唯一接口】
- compute_score(df: pd.DataFrame) -> pd.DataFrame
  输入: 包含 OHLCV 等基础数据的 DataFrame
  输出: 包含 trade_date, symbol, score 的 DataFrame

【内部结构 - 特征解耦】
1. _sub_model_reversal(): 短期反转子模型 (适合极值恐慌市场)
2. _sub_model_momentum(): 中期动量子模型 (适合趋势爆发市场)
3. _market_state_adapter(): 市场状态适配器 (动态权重门控)

【核心演进 - V218】
- 针对 2024 年 IC 失效问题，引入市场状态识别
- 使用全市场平均波动率和趋势强度判定市场状态
- 动态权重: W_t = sigmoid(alpha * vol_norm + beta * trend_strength)
- Final_Score = W_t * Score_Rev + (1 - W_t) * Score_Mom

【合规锁定】
- 严禁任何 shift(-1) 或 next_ret 引用
- 所有因子仅使用 T-1 日及之前数据
- 严禁接触回测逻辑
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
VERSION = "V218"


class AlphaModelV218:
    """
    V218 Alpha Model - Player (选手)
    
    【核心职责】
    1. 计算短期反转得分 (Score_Rev)
    2. 计算中期动量得分 (Score_Mom)
    3. 识别市场状态并动态调整权重
    4. 输出最终评分 (Final_Score)
    
    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score]
    """
    
    # ==================== 配置参数 ====================
    # 反转子模型参数
    REV_WINDOWS = [5, 10, 20]
    
    # 动光子模型参数
    MOM_WINDOWS = [20, 60]
    
    # 市场状态识别参数
    VOL_WINDOW = 20
    TREND_WINDOW = 20
    
    # 动态权重参数
    ALPHA_VOL = 2.0  # 波动率权重系数
    BETA_TREND = 1.5  # 趋势权重系数
    
    # 基础权重
    BASE_W_REV = 0.6
    BASE_W_MOM = 0.4
    
    def __init__(self):
        """初始化 Alpha Model"""
        logger.info("=" * 70)
        logger.info("V218 Alpha Model Initialized (Player)")
        logger.info("=" * 70)
        logger.info(f"  Rev Windows: {self.REV_WINDOWS}")
        logger.info(f"  Mom Windows: {self.MOM_WINDOWS}")
        logger.info(f"  Alpha Vol: {self.ALPHA_VOL}, Beta Trend: {self.BETA_TREND}")
        logger.info("=" * 70)
    
    # ==================== 唯一公开接口 ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)
        
        【接口规范】
        - 输入: pd.DataFrame with columns [trade_date, symbol, open, high, low, close, volume, ...]
        - 输出: pd.DataFrame with columns [trade_date, symbol, score]
        
        【计算流程】
        1. 数据预处理 (计算衍生特征)
        2. 计算反转得分 (Score_Rev) - 使用横截面 z-score
        3. 计算动量得分 (Score_Mom) - 使用横截面 z-score
        4. 识别市场状态，计算动态权重 W_t
        5. 融合得分: Final_Score = W_t * Score_Rev + (1 - W_t) * Score_Mom
        6. 最终横截面 Rank 处理
        """
        logger.info(f"[V218] Computing alpha scores for {len(df)} rows...")
        
        # 1. 数据预处理
        df = self._preprocess_data(df)
        
        # 2. 计算反转得分 (横截面 z-score)
        logger.info("[V218] Computing reversal scores...")
        df['score_rev'] = self._sub_model_reversal(df)
        
        # 3. 计算动量得分 (横截面 z-score)
        logger.info("[V218] Computing momentum scores...")
        df['score_mom'] = self._sub_model_momentum(df)
        
        # 4. 计算市场状态权重
        logger.info("[V218] Computing market state weights...")
        df['w_rev'] = self._market_state_adapter(df)
        df['w_mom'] = 1.0 - df['w_rev']
        
        # 5. 融合得分
        df['score_raw'] = df['w_rev'] * df['score_rev'] + df['w_mom'] * df['score_mom']
        
        # 6. 最终横截面 Rank 处理
        df['score'] = df.groupby('trade_date')['score_raw'].rank(pct=True, na_option='keep')
        df['score'] = df['score'].fillna(0.5)
        
        # 7. 输出结果
        result_cols = ['trade_date', 'symbol', 'score']
        if 'close' in df.columns:
            result_cols.append('close')
        result = df[result_cols].copy()
        
        logger.info(f"[V218] Score computed: mean={result['score'].mean():.4f}, std={result['score'].std():.4f}")
        
        return result
    
    # ==================== 数据预处理 ====================
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        【处理步骤】
        1. 确保排序正确
        2. 计算基础衍生特征
        3. 处理异常值
        """
        result = df.copy()
        
        # 确保排序
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算日收益率
        result['ret_1d'] = result.groupby('symbol')['close'].pct_change(1)
        
        # 计算各窗口收益率
        for w in [5, 10, 20, 60]:
            result[f'ret_{w}d'] = result.groupby('symbol')['close'].pct_change(w)
        
        # 计算波动率 (滚动 std)
        result['vol_20d'] = result.groupby('symbol')['ret_1d'].transform(
            lambda x: x.rolling(self.VOL_WINDOW, min_periods=10).std()
        )
        
        # 计算 MA20
        result['ma_20'] = result.groupby('symbol')['close'].transform(
            lambda x: x.rolling(self.TREND_WINDOW, min_periods=10).mean()
        )
        result['trend_strength'] = (result['close'] - result['ma_20']) / result['ma_20']
        
        # 换手率变化
        if 'turnover_rate' in result.columns:
            result['turnover_chg'] = result.groupby('symbol')['turnover_rate'].pct_change(5)
        else:
            result['turnover_chg'] = 0.0
        
        # 成交量比率
        if 'volume' in result.columns:
            result['vol_ratio'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / x.rolling(20, min_periods=10).mean()
            )
        else:
            result['vol_ratio'] = 1.0
        
        # 处理无穷大和 NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 填充 NaN
        fill_cols = ['ret_1d', 'vol_20d', 'trend_strength', 'turnover_chg', 'vol_ratio']
        for col in fill_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
        
        # 填充收益率 NaN
        for w in [5, 10, 20, 60]:
            col = f'ret_{w}d'
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
        
        logger.debug(f"[Preprocess] Data shape: {result.shape}")
        
        return result
    
    # ==================== 横截面标准化 ====================
    
    def _cross_sectional_zscore(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        横截面 z-score 标准化
        
        【方法】
        对每个 trade_date，计算该日所有股票的 z-score
        z = (x - mean) / std
        
        Args:
            df: 数据
            col: 列名
            
        Returns:
            z-score 标准化后的 Series
        """
        # 计算每日均值和标准差
        daily_mean = df.groupby('trade_date')[col].transform('mean')
        daily_std = df.groupby('trade_date')[col].transform('std')
        
        # 计算 z-score
        zscore = (df[col] - daily_mean) / daily_std.replace(0, np.nan)
        
        # 裁剪极端值 (-3, 3)
        zscore = zscore.clip(-3, 3)
        
        return zscore.fillna(0.0)
    
    def _cross_sectional_rank(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        横截面 Rank 处理
        
        【方法】
        对每个 trade_date，计算该日所有股票的百分位 Rank
        
        Args:
            df: 数据
            col: 列名
            
        Returns:
            Rank 百分位 Series (0~1)
        """
        return df.groupby('trade_date')[col].rank(pct=True, na_option='keep').fillna(0.5)
    
    # ==================== 反转子模型 ====================
    
    def _sub_model_reversal(self, df: pd.DataFrame) -> pd.Series:
        """
        短期反转子模型 (适合极值恐慌市场)
        
        【核心逻辑】
        - 短期下跌的股票倾向于反弹 (均值回归)
        - 使用多窗口反转信号: 5d, 10d, 20d
        - 成交量放大 + 价格下跌 = 恐慌信号 (更强反转)
        
        【因子】
        1. ret_5d: 5日收益率 (负值表示下跌，预期反弹)
        2. ret_10d: 10日收益率
        3. ret_20d: 20日收益率
        4. vol_ratio: 成交量比率 (放量下跌更强)
        5. turnover_chg: 换手率变化
        """
        # 1. 价格反转因子 (负收益 -> 正得分)
        rev_5d = -df['ret_5d']
        rev_10d = -df['ret_10d']
        rev_20d = -df['ret_20d']
        
        # 2. 多窗口加权融合
        score_raw = 0.4 * rev_5d + 0.35 * rev_10d + 0.25 * rev_20d
        
        # 3. 横截面 z-score 标准化 (使用 score_raw 值直接计算)
        daily_mean = score_raw.groupby(df['trade_date']).transform('mean')
        daily_std = score_raw.groupby(df['trade_date']).transform('std')
        score_z = (score_raw - daily_mean) / daily_std.replace(0, np.nan)
        score_z = score_z.clip(-3, 3).fillna(0.0)
        
        # 4. 成交量确认调整
        vol_confirm = df['vol_ratio'] * (rev_5d > 0).astype(float)
        vc_mean = vol_confirm.groupby(df['trade_date']).transform('mean')
        vc_std = vol_confirm.groupby(df['trade_date']).transform('std')
        vol_confirm_z = ((vol_confirm - vc_mean) / vc_std.replace(0, np.nan)).clip(-3, 3).fillna(0.0)
        
        # 5. 换手率调整
        to_mean = df['turnover_chg'].groupby(df['trade_date']).transform('mean')
        to_std = df['turnover_chg'].groupby(df['trade_date']).transform('std')
        turnover_z = ((df['turnover_chg'] - to_mean) / to_std.replace(0, np.nan)).clip(-3, 3).fillna(0.0)
        
        # 6. 最终得分
        final_score = score_z * (0.7 + 0.2 * vol_confirm_z.clip(-2, 2)) + 0.1 * turnover_z
        
        return final_score
    
    # ==================== 动光子模型 ====================
    
    def _sub_model_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        中期动光子模型 (适合趋势爆发市场)
        
        【核心逻辑】
        - 中期上涨的股票倾向于继续上涨 (趋势延续)
        - 使用多窗口动量信号: 20d, 60d
        
        【因子】
        1. ret_20d: 20日收益率 (中期动量)
        2. ret_60d: 60日收益率 (长期动量)
        3. trend_strength: 趋势强度 (价格相对 MA20 的位置)
        4. vol_stability: 波动率稳定性 (低波动上涨更可靠)
        """
        # 1. 价格动量因子 (正收益 -> 正得分)
        mom_20d = df['ret_20d']
        mom_60d = df['ret_60d']
        
        # 2. 趋势强度因子
        trend = df['trend_strength']
        
        # 3. 波动率稳定性 (低波动 -> 高分)
        vol_stability = -df['vol_20d']
        
        # 4. 多窗口加权融合
        score_raw = 0.5 * mom_20d + 0.3 * mom_60d + 0.2 * trend
        
        # 5. 横截面 z-score 标准化 (直接计算)
        sr_mean = score_raw.groupby(df['trade_date']).transform('mean')
        sr_std = score_raw.groupby(df['trade_date']).transform('std')
        score_z = ((score_raw - sr_mean) / sr_std.replace(0, np.nan)).clip(-3, 3).fillna(0.0)
        
        # 6. 波动率稳定性调整
        vs_mean = vol_stability.groupby(df['trade_date']).transform('mean')
        vs_std = vol_stability.groupby(df['trade_date']).transform('std')
        vol_stab_z = ((vol_stability - vs_mean) / vs_std.replace(0, np.nan)).clip(-3, 3).fillna(0.0)
        
        # 7. 最终得分
        final_score = score_z + 0.1 * vol_stab_z
        
        return final_score
    
    # ==================== 市场状态适配器 ====================
    
    def _market_state_adapter(self, df: pd.DataFrame) -> pd.Series:
        """
        市场状态适配器 (动态权重门控)
        
        【核心逻辑】
        1. 识别市场状态:
           - CRISIS (极值恐慌): 高波动 + 下跌趋势 -> 适合反转 (W_rev 高)
           - TREND (趋势爆发): 低波动 + 上涨趋势 -> 适合动量 (W_rev 低)
           - NORMAL (正常区间): 介于两者之间 -> 基础权重
        
        2. 动态权重计算:
           W_t = sigmoid(alpha * vol_norm + beta * trend_strength)
        """
        # 1. 计算市场整体波动率 (全市场平均)
        market_vol = df.groupby('trade_date')['vol_20d'].mean()
        
        # 2. 计算市场整体趋势 (全市场平均)
        market_trend = df.groupby('trade_date')['trend_strength'].mean()
        
        # 3. 标准化波动率 (使用 60 日滚动窗口)
        vol_mean = market_vol.rolling(60, min_periods=20).mean()
        vol_std = market_vol.rolling(60, min_periods=20).std()
        vol_norm = (market_vol - vol_mean) / vol_std.replace(0, np.nan)
        vol_norm = vol_norm.fillna(0.0)
        
        # 4. 标准化趋势
        trend_mean = market_trend.rolling(60, min_periods=20).mean()
        trend_std = market_trend.rolling(60, min_periods=20).std()
        trend_norm = (market_trend - trend_mean) / trend_std.replace(0, np.nan)
        trend_norm = trend_norm.fillna(0.0)
        
        # 5. 计算动态权重
        # 高波动 -> 高 W_rev, 下跌趋势 -> 高 W_rev
        raw_weight = self.ALPHA_VOL * vol_norm - self.BETA_TREND * trend_norm
        
        # Sigmoid 函数映射到 (0, 1)
        sigmoid_weight = 1.0 / (1.0 + np.exp(-np.clip(raw_weight, -10, 10)))
        
        # 映射到 [0.2, 0.8] 范围
        w_rev = 0.2 + 0.6 * sigmoid_weight
        
        # 6. 将日期级别的权重映射回股票级别
        w_rev_df = pd.DataFrame({
            'trade_date': w_rev.index,
            'w_rev': w_rev.values
        })
        
        result = df[['trade_date']].merge(w_rev_df, on='trade_date', how='left')
        w_rev_final = result['w_rev'].fillna(self.BASE_W_REV)
        
        logger.debug(f"[Market State] W_rev range: [{w_rev_final.min():.3f}, {w_rev_final.max():.3f}]")
        
        return w_rev_final
    
    # ==================== 辅助方法 ====================
    
    def get_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取各因子的 IC (用于调试和分析)"""
        ics = {}
        
        if 't1_return' not in df.columns:
            return ics
        
        for col in ['score_rev', 'score_mom', 'w_rev']:
            if col in df.columns:
                ic = self._calculate_daily_ic(df, col, 't1_return')
                ics[col] = ic
        
        if 'score' in df.columns:
            ics['final_score'] = self._calculate_daily_ic(df, 'score', 't1_return')
        
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
            
            factor_clean = factor_vals[mask]
            return_clean = return_vals[mask]
            
            corr = factor_clean.corr(return_clean, method='spearman')
            
            if not np.isnan(corr):
                ic_values.append(corr)
        
        return float(np.mean(ic_values)) if ic_values else 0.0


def get_alpha_model() -> AlphaModelV218:
    """获取 Alpha Model 实例"""
    return AlphaModelV218()


if __name__ == "__main__":
    logger.info("V218 Alpha Model loaded successfully")
    model = get_alpha_model()
    logger.info(f"Version: {VERSION}")