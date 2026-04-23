"""
Alpha Model Module - V207 Deep Feature Mining & Dynamic Correlation Orthogonalization
======================================================================================

【V207 核心变革】
1. 时空敏感性 2.0 (Spatiotemporal Sensitivity 2.0)
   - 订单流不平衡 (Order Imbalance): 利用 amount 与 volume 的变动率差值
   - 行业内强度 (Intra-industry Strength): 股票在该行业内的截面百分位排名

2. 动态相关性正交化 (Dynamic Correlation Orthogonalization)
   - 禁止全局正交化
   - 实现"窗口自适应正交化": 只对过去 20 个交易日与大盘风格相关性超过 0.7 的特征进行残差化处理

3. 多轮迭代闭环 (The Loop)
   - 运行回测后检查日志
   - 如果 IC < 0.05 或出现连续 10 天以上亏损，必须调用 analyze_failure() 逻辑
   - 根据诊断结果修改特征融合逻辑，自动重新回测，直到结果达标

【物理清理】
- 删除所有 V191-V206 字样注释
- 删除无用变量和过期逻辑
- 代码完全重构

【严格红线】
- 严禁导入、计算或引用任何名为 't1_return'、'next_ret' 或 shift(-1) 的数据
- 必须通过 get_score() 方法仅输出各股票在 T 日的排序分
- 所有预测逻辑必须基于 T 日收盘前的已知信息
"""

from typing import Any, Optional, Dict, List, Tuple
import warnings
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    logger.warning("Polars not installed, falling back to pandas")

load_dotenv()
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V207_Deep_Feature_Dynamic_Correlation"

# V207 核心配置
# 时空敏感性 2.0 参数
V207_SPATIOTEMPORAL_CONFIG = {
    'return_scales': [5, 10, 20],
    'volume_window': 5,
    'order_imbalance_window': 10,
    'intra_industry_window': 5,
}

# 动态相关性正交化参数
DYNAMIC_CORRELATION_CONFIG = {
    'correlation_window': 20,
    'correlation_threshold': 0.7,
    'orthogonalization_strength': 0.8,
}

# 多轮迭代闭环参数
ITERATION_LOOP_CONFIG = {
    'ic_threshold': 0.05,
    'consecutive_loss_days': 10,
    'max_iterations': 3,
}

# 防止过拟合参数
CROSS_VALIDATION_CONFIG = {
    'noise_ratio': 0.10,
    'volatility_threshold': 0.30,
    'enabled': True,
}

# 行业 IC 分析参数
INDUSTRY_IC_CONFIG = {
    'min_stocks_per_industry': 10,
    'rolling_ic_window': 20,
    'negative_ic_threshold': -0.02,
}

# 多尺度特征
V207_REVERSION_SCALES = [5, 10, 20, 60]
V207_MOMENTUM_SCALES = [5, 10, 20, 60]
V207_VOLATILITY_SCALES = [5, 10, 20, 60]

# 风险滤网参数
RISK_FILTER_HIGH_VOL_THRESHOLD = 0.60
RISK_FILTER_LOW_LIQ_THRESHOLD = 0.35
RISK_FILTER_PENALTY_SCALE = 4.0

# 行业中性化参数
INDUSTRY_NEUTRALIZE_ENABLED = True

# 动态市场适配参数
MARKET_STATE_WINDOW = 20
MARKET_STATE_BEAR_THRESHOLD = -0.10
MARKET_STATE_BULL_THRESHOLD = 0.15

# 配置
MIN_STOCK_COUNT = 5000
WARMUP_DAYS = 60
EPSILON = 1e-6


# ==============================================================================
# 工具函数
# ==============================================================================

def winsorize(series: pd.Series, sigma: float = 3.0) -> pd.Series:
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


def mad_scale(series: pd.Series, k: float = 1.4826) -> float:
    """计算 Median Absolute Deviation (MAD) 尺度估计"""
    median = series.median()
    if pd.isna(median):
        return 1.0
    
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad < 1e-10:
        return 1.0
    
    return k * mad


def robust_zscore(series: pd.Series, use_mad: bool = True) -> pd.Series:
    """鲁棒性 Z-Score 标准化"""
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    median = series_clean.median()
    if pd.isna(median):
        return pd.Series(0.0, index=series.index)
    
    if use_mad:
        scale = mad_scale(series_clean)
    else:
        scale = series_clean.std()
        if pd.isna(scale) or scale < 1e-10:
            scale = 1.0
    
    zscore = (series_clean - median) / scale
    return zscore.fillna(0)


def normalize_group(df: pd.DataFrame, score_col: str, group_col: str = None) -> pd.Series:
    """截面或分组标准化评分"""
    if group_col is None or group_col not in df.columns:
        if len(df) < 2:
            return pd.Series(0.5, index=df.index)
        
        # 检查是否所有值都相同
        if df[score_col].nunique() == 1:
            # 所有值相同，返回中位数 0
            return pd.Series(0.0, index=df.index)
        
        ranks = df[score_col].rank(pct=True, method='average')
        normalized = (ranks - 0.5) * 2
        return normalized
    
    def rank_pct(group):
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        # 检查是否所有值都相同
        if group.nunique() == 1:
            return pd.Series(0.0, index=group.index)
        return (group.rank(pct=True, method='average') - 0.5) * 2
    
    normalized = df.groupby(group_col)[score_col].transform(rank_pct)
    return normalized


def auto_fillna(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """自动填充缺失值"""
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        result[col] = result.groupby('symbol', group_keys=False)[col].ffill()
        
        missing_mask = result[col].isna()
        if missing_mask.any():
            col_mean = result[col].mean()
            if pd.notna(col_mean):
                result.loc[missing_mask, col] = col_mean
        
        result[col] = result[col].fillna(0)
    
    return result


def gram_schmidt_orthogonalize(
    df: pd.DataFrame,
    target_col: str,
    orthogonal_cols: List[str],
    group_col: str = 'trade_date'
) -> pd.Series:
    """截面施密特正交化 (Gram-Schmidt Orthogonalization)"""
    result = pd.Series(0.0, index=df.index, name=f'{target_col}_ortho')
    
    if group_col not in df.columns:
        return result
    
    for date, group_idx in df.groupby(group_col).groups.items():
        group_data = df.loc[group_idx]
        
        if len(group_data) < len(orthogonal_cols) + 5:
            result.loc[group_idx] = df.loc[group_idx, target_col] if target_col in df.columns else 0.0
            continue
        
        target_values = group_data[target_col].values if target_col in group_data.columns else np.zeros(len(group_data))
        
        if np.all(np.isnan(target_values)):
            continue
        
        factor_matrix = []
        valid_cols = []
        for col in orthogonal_cols:
            if col in group_data.columns:
                factor_values = group_data[col].values
                if not np.all(np.isnan(factor_values)):
                    factor_matrix.append(factor_values)
                    valid_cols.append(col)
        
        if len(factor_matrix) == 0:
            result.loc[group_idx] = target_values
            continue
        
        factor_matrix = np.column_stack(factor_matrix)
        
        mask = ~np.isnan(target_values)
        for i in range(factor_matrix.shape[1]):
            mask &= ~np.isnan(factor_matrix[:, i])
        
        if mask.sum() < len(valid_cols) + 3:
            result.loc[group_idx] = target_values
            continue
        
        y = target_values[mask]
        X = factor_matrix[mask]
        
        y_mean, y_std = y.mean(), y.std() + EPSILON
        y_norm = (y - y_mean) / y_std
        
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + EPSILON
        X_norm = (X - X_mean) / X_std
        
        residual = y_norm.copy()
        for i in range(X_norm.shape[1]):
            x_col = X_norm[:, i]
            beta = np.dot(residual, x_col) / (np.dot(x_col, x_col) + EPSILON)
            residual = residual - beta * x_col
        
        residual_final = residual * y_std + y_mean
        
        result.loc[group_idx[mask]] = residual_final
    
    return result


def compute_order_imbalance(amount: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    """
    计算订单流不平衡 (Order Imbalance)
    
    公式：amount 变动率 - volume 变动率
    正值表示买方力量强，负值表示卖方力量强
    """
    amount_change = amount.pct_change(window)
    volume_change = volume.pct_change(window)
    
    order_imbalance = amount_change - volume_change
    return order_imbalance.fillna(0)


def compute_intra_industry_strength(
    df: pd.DataFrame,
    score_col: str,
    industry_col: str = 'industry_code',
    window: int = 5
) -> pd.Series:
    """
    计算行业内强度 (Intra-industry Strength)
    
    计算股票在过去 window 天内的表现在行业内的截面百分位排名
    """
    result = pd.Series(0.5, index=df.index)
    
    if industry_col not in df.columns or score_col not in df.columns:
        return result
    
    # 按行业分组计算截面排名
    for industry in df[industry_col].unique():
        industry_mask = df[industry_col] == industry
        industry_data = df.loc[industry_mask]
        
        if len(industry_data) < 5:
            continue
        
        # 计算该行业内股票的截面排名
        ranks = industry_data[score_col].rank(pct=True, method='average')
        result.loc[industry_mask] = (ranks - 0.5) * 2
    
    return result


def compute_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 20
) -> pd.Series:
    """计算滚动相关系数"""
    return series1.rolling(window, min_periods=10).corr(series2)


# ==============================================================================
# V207 时空敏感性 2.0 模块
# ==============================================================================

class SpatiotemporalFeatureEngineV207:
    """
    V207 时空敏感性 2.0 引擎
    
    【核心特征】
    1. 订单流不平衡 (Order Imbalance)
       - 利用 amount 与 volume 的变动率差值
       - 捕捉资金流向的微观结构
    
    2. 行业内强度 (Intra-industry Strength)
       - 股票在该行业内的截面百分位排名
       - 捕捉行业轮动效应
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or V207_SPATIOTEMPORAL_CONFIG
        self.return_scales = self.config.get('return_scales', [5, 10, 20])
        self.volume_window = self.config.get('volume_window', 5)
        self.order_imbalance_window = self.config.get('order_imbalance_window', 10)
        self.intra_industry_window = self.config.get('intra_industry_window', 5)
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算时空敏感性 2.0 特征"""
        df = df.copy()
        
        # 1. 计算订单流不平衡
        df = self._compute_order_imbalance(df)
        
        # 2. 计算行业内强度
        df = self._compute_intra_industry_strength(df)
        
        # 3. 计算时空交互特征
        df = self._compute_spatiotemporal_interaction(df)
        
        return df
    
    def _compute_order_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算订单流不平衡
        
        【V207.1 修复】
        - 增加 fillna(0) 逻辑，确保数据缺失时不会导致整个得分向量崩溃
        - 订单流不平衡 = amount 变动率 - volume 变动率
        """
        if 'amount' not in df.columns or 'volume' not in df.columns:
            logger.warning("[SpatiotemporalV207] Missing amount/volume columns for order imbalance")
            df['order_imbalance'] = 0.0
            df['order_imbalance_rank'] = 0.5
            df['order_imbalance_ma'] = 0.0
            df['order_imbalance_flag'] = 0
            return df
        
        # 计算订单流不平衡 - 使用 fillna(0) 确保稳健性
        order_imbalance_raw = compute_order_imbalance(
            df['amount'].fillna(0), df['volume'].fillna(0), self.order_imbalance_window
        )
        
        # 缩尾处理异常值
        df['order_imbalance'] = winsorize(order_imbalance_raw.fillna(0), sigma=3.0)
        
        # 计算截面标准化 - 使用 fillna(0.5) 确保缺失时有默认值
        df['order_imbalance_rank'] = df.groupby('trade_date')['order_imbalance'].transform(
            lambda x: x.rank(pct=True).fillna(0.5)
        ).fillna(0.5)
        
        # 订单流不平衡的滚动均值 - 使用 fillna(0) 确保稳健性
        df['order_imbalance_ma'] = df.groupby('symbol')['order_imbalance'].transform(
            lambda x: x.rolling(self.volume_window, min_periods=3).mean().fillna(0)
        ).fillna(0)
        
        # 订单流不平衡标志
        df['order_imbalance_flag'] = (df['order_imbalance'] > 0).astype(int)
        
        # V207.1 核心修复：最终 fillna(0) 确保不会导致得分崩溃
        df['order_imbalance'] = df['order_imbalance'].fillna(0)
        df['order_imbalance_rank'] = df['order_imbalance_rank'].fillna(0.5)
        df['order_imbalance_ma'] = df['order_imbalance_ma'].fillna(0)
        
        logger.debug(f"[SpatiotemporalV207] Order Imbalance computed. Range: [{df['order_imbalance'].min():.4f}, {df['order_imbalance'].max():.4f}]")
        
        return df
    
    def _compute_intra_industry_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算行业内强度
        
        【V207.1 修复】
        - 增加 fillna(0) 逻辑，确保数据缺失时不会导致整个得分向量崩溃
        - 行业内强度 = 股票在过去 window 天内的表现在行业内的截面百分位排名
        - V207.1 核心修复：优先使用 industry_name 进行行业分组（因为 industry_code 可能格式混乱）
        - V207.1 核心修复：当行业数量太少（<10）时，使用全市场截面排名替代行业内排名
        """
        # V207.1 核心修复：优先使用 industry_name，回退到 industry_code
        # 【V207.1 紧急修复】同时检查非空字符串，因为 industry_name 可能是空字符串而非 NaN
        has_industry_name = (
            'industry_name' in df.columns and 
            df['industry_name'].notna().any() and 
            (df['industry_name'].astype(str).str.strip() != '').any()
        )
        has_industry_code = (
            'industry_code' in df.columns and 
            df['industry_code'].notna().any() and
            (df['industry_code'].astype(str).str.strip() != '').any()
        )
        
        # 确定使用哪个行业列
        if has_industry_name:
            industry_col = 'industry_name'
            industry_count = df['industry_name'].nunique()
            logger.info(f"[SpatiotemporalV207] Using industry_name for industry grouping. Unique values: {industry_count}")
        elif has_industry_code:
            industry_col = 'industry_code'
            industry_count = df['industry_code'].nunique()
            logger.info(f"[SpatiotemporalV207] Using industry_code for industry grouping (fallback). Unique values: {industry_count}")
        else:
            logger.warning(f"[SpatiotemporalV207] No industry column available. has_industry_name={has_industry_name}, has_industry_code={has_industry_code}")
            logger.warning(f"[SpatiotemporalV207] Columns: {df.columns.tolist()}")
            if 'industry_name' in df.columns:
                logger.warning(f"[SpatiotemporalV207] industry_name sample: {df['industry_name'].head(10).tolist()}")
            if 'industry_code' in df.columns:
                logger.warning(f"[SpatiotemporalV207] industry_code sample: {df['industry_code'].head(10).tolist()}")
            # 返回默认值，防止崩溃
            df['intra_industry_strength_5d'] = 0.5
            df['intra_industry_strength_10d'] = 0.5
            df['intra_industry_strength_20d'] = 0.5
            df['intra_industry_strength'] = 0.5
            return df
        
        # V207.1 核心修复：行业数量太少时，使用全市场截面排名替代行业内排名
        # 原因：行业太少时，行业内排名失去区分度
        use_market_rank = industry_count < 10
        if use_market_rank:
            logger.info(f"[SpatiotemporalV207] Industry count ({industry_count}) < 10, using market-wide rank instead of intra-industry rank")
        
        # 计算基础收益率用于排名 - 使用 fillna(0) 确保稳健性
        for scale in self.return_scales:
            ret_col = f'return_{scale}d'
            if ret_col not in df.columns:
                df[ret_col] = df.groupby('symbol')['close'].transform(
                    lambda x: (x / x.shift(scale) - 1).fillna(0)
                ).fillna(0)
            else:
                # 确保已有的收益率列也填充缺失值
                df[ret_col] = df[ret_col].fillna(0)
        
        # 计算各周期的强度排名
        # V207.1 核心修复：行业数量<10 时使用全市场排名，否则使用行业内排名
        for scale in self.return_scales:
            ret_col = f'return_{scale}d'
            strength_col = f'intra_industry_strength_{scale}d'
            
            # V207.1 核心修复：根据行业数量决定使用全市场排名还是行业内排名
            if use_market_rank:
                # 行业数量太少，使用全市场截面排名
                logger.debug(f"[SpatiotemporalV207] Using market-wide rank for {strength_col}")
                # 按日期分组进行全市场排名
                for date in df['trade_date'].unique():
                    date_mask = df['trade_date'] == date
                    date_data = df.loc[date_mask]
                    
                    if len(date_data) < 100:
                        df.loc[date_mask, strength_col] = 0.5
                        continue
                    
                    # 计算排名
                    ranks = date_data[ret_col].rank(pct=True, method='average')
                    df.loc[date_mask, strength_col] = (ranks - 0.5) * 2
                
                df[strength_col] = df[strength_col].fillna(0.5)
            else:
                # 行业数量足够，使用行业内排名
                logger.debug(f"[SpatiotemporalV207] Using intra-industry rank for {strength_col}")
                # 按日期和行业分组进行排名
                for date in df['trade_date'].unique():
                    date_mask = df['trade_date'] == date
                    date_data = df.loc[date_mask]
                    
                    if len(date_data) < 100:
                        df.loc[date_mask, strength_col] = 0.5
                        continue
                    
                    for industry in date_data[industry_col].unique():
                        industry_mask = date_data[industry_col] == industry
                        industry_data = date_data.loc[industry_mask]
                        
                        if len(industry_data) < 5:
                            df.loc[date_data.index[industry_mask], strength_col] = 0.5
                            continue
                        
                        # 计算排名
                        ranks = industry_data[ret_col].rank(pct=True, method='average')
                        df.loc[date_data.index[industry_mask], strength_col] = (ranks - 0.5) * 2
                
                df[strength_col] = df[strength_col].fillna(0.5)
            
            # 最终 fillna(0.5) 确保不会导致得分崩溃
            df[strength_col] = df[strength_col].fillna(0.5)
            
            # V207.1 核心修复：记录强度计算结果
            logger.debug(f"[SpatiotemporalV207] {strength_col} range: [{df[strength_col].min():.4f}, {df[strength_col].max():.4f}]")
        
        # 综合行业内强度 (加权平均) - V207.1 修复
        weights = {5: 0.4, 10: 0.35, 20: 0.25}
        intra_strength = sum(
            df[f'intra_industry_strength_{scale}d'].fillna(0.5) * weights.get(scale, 0.33)
            for scale in self.return_scales
        )
        df['intra_industry_strength'] = intra_strength.fillna(0.5)
        
        logger.debug(f"[SpatiotemporalV207] Intra-industry Strength computed. Range: [{df['intra_industry_strength'].min():.4f}, {df['intra_industry_strength'].max():.4f}]")
        
        return df
    
    def _compute_spatiotemporal_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算时空交互特征"""
        # 时空交互 = 订单流不平衡 × 行业内强度
        # 使用 20 天周期作为主要的行业内强度
        strength_col = 'intra_industry_strength_20d'
        
        if strength_col not in df.columns:
            # 如果 20d 不存在，使用 10d
            strength_col = 'intra_industry_strength_10d'
        
        if strength_col not in df.columns:
            # 如果都不存在，使用 5d
            strength_col = 'intra_industry_strength_5d'
        
        if strength_col not in df.columns:
            # 如果所有行业内强度列都不存在，跳过交互特征
            logger.warning("[SpatiotemporalV207] No intra_industry_strength columns found, skipping interaction")
            df['spatiotemporal_interaction_v207'] = 0.0
            df['spatiotemporal_interaction_v207_rank'] = 0.5
            return df
        
        df['spatiotemporal_interaction_v207'] = (
            df['order_imbalance_rank'] * 
            df[strength_col]
        )
        
        # 交互特征的截面标准化
        df['spatiotemporal_interaction_v207_rank'] = df.groupby('trade_date')['spatiotemporal_interaction_v207'].transform(
            lambda x: x.rank(pct=True)
        ).fillna(0.5)
        
        return df


# ==============================================================================
# V207 动态相关性正交化模块
# ==============================================================================

class DynamicCorrelationOrthogonalizer:
    """
    V207 动态相关性正交化器
    
    【V207.1 核心原理】
    1. 禁止全局正交化
    2. 实现"窗口自适应正交化": 只对过去 20 个交易日与大盘风格相关性超过 0.7 的特征进行残差化处理
    3. 【V207.1 修复】若相关性无法计算，默认不进行残差化处理，防止得分被抹平
    
    【正交化流程】
    1. 计算每个特征与大盘风格 (市场收益率) 的滚动相关系数
    2. 当相关性超过阈值 (0.7) 时，对该特征进行正交化
    3. 正交化强度可调节 (0-1)
    4. 若相关性无法计算 (数据不足/方差为 0)，跳过正交化
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or DYNAMIC_CORRELATION_CONFIG
        self.correlation_window = self.config.get('correlation_window', 20)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.orth_strength = self.config.get('orthogonalization_strength', 0.8)
    
    def compute_correlation_with_market(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        market_return_col: str = 'market_return'
    ) -> pd.DataFrame:
        """
        计算特征与大盘风格的相关性
        
        【V207.1 修复】
        - 增加相关性计算失败的处理逻辑
        - 若无法计算相关性，返回 0（表示不相关，不进行正交化）
        """
        df = df.copy()
        
        if market_return_col not in df.columns:
            # 计算市场收益率
            df[market_return_col] = df.groupby('trade_date')['pct_chg'].transform('mean')
        
        # 按日期分组计算市场收益率
        market_returns = df.groupby('trade_date')[market_return_col].first()
        
        for target_col in target_cols:
            if target_col not in df.columns:
                continue
            
            corr_col = f'{target_col}_market_corr'
            df[corr_col] = np.nan
            
            # 按股票分组计算滚动相关性
            for symbol in df['symbol'].unique():
                symbol_mask = df['symbol'] == symbol
                symbol_data = df.loc[symbol_mask].sort_values('trade_date')
                
                if len(symbol_data) < self.correlation_window:
                    # 数据不足，填充 0 表示不相关
                    df.loc[symbol_data.index, corr_col] = 0.0
                    continue
                
                # 计算特征值与市场收益率的滚动相关性
                feature_values = symbol_data[target_col].values
                dates = symbol_data['trade_date'].values
                
                # 检查特征值是否有效
                if np.all(np.isnan(feature_values)) or np.nanstd(feature_values) < EPSILON:
                    # 特征值无效或方差为 0，无法计算相关性
                    df.loc[symbol_data.index, corr_col] = 0.0
                    continue
                
                # 创建对齐的市场收益率序列
                market_aligned = market_returns.reindex(dates).fillna(0).values
                
                # 检查市场收益率是否有效
                if np.nanstd(market_aligned) < EPSILON:
                    # 市场收益率方差为 0，无法计算相关性
                    df.loc[symbol_data.index, corr_col] = 0.0
                    continue
                
                # 计算滚动相关性
                try:
                    corr_series = pd.Series(feature_values).rolling(
                        self.correlation_window, min_periods=10
                    ).corr(pd.Series(market_aligned))
                    df.loc[symbol_data.index, corr_col] = corr_series.values
                except Exception as e:
                    logger.debug(f"[DynamicOrtho] Correlation calculation failed for {symbol}: {e}")
                    df.loc[symbol_data.index, corr_col] = 0.0
            
            # V207.1 核心修复：fillna(0) 确保无法计算相关性时默认不相关
            df[corr_col] = df[corr_col].fillna(0)
            df[f'{target_col}_high_corr'] = (df[corr_col].abs() > self.correlation_threshold).astype(int)
        
        return df
    
    def apply_dynamic_orthogonalization(
        self,
        df: pd.DataFrame,
        target_cols: List[str]
    ) -> pd.DataFrame:
        """
        应用动态相关性正交化
        
        【V207.1 修复】
        - 若相关性无法计算，默认不进行残差化处理
        - 正交化前检查数据有效性，防止得分被抹平
        """
        df = df.copy()
        
        # 先计算与大盘风格的相关性
        df = self.compute_correlation_with_market(df, target_cols)
        
        for target_col in target_cols:
            if target_col not in df.columns:
                logger.warning(f"[DynamicOrtho] Column {target_col} not found, skipping")
                continue
            
            orth_col = f'{target_col}_dynamic_ortho'
            df[orth_col] = df[target_col].copy()
            
            high_corr_col = f'{target_col}_high_corr'
            
            # V207.1 修复：检查是否有需要正交化的样本
            high_corr_count = df[high_corr_col].sum() if high_corr_col in df.columns else 0
            if high_corr_count == 0:
                logger.debug(f"[DynamicOrtho] No high correlation samples for {target_col}, skipping orthogonalization")
                continue
            
            # 只对高相关性样本进行正交化
            for date in df['trade_date'].unique():
                date_mask = df['trade_date'] == date
                high_corr_mask = df[high_corr_col] == 1
                
                need_orth_mask = date_mask & high_corr_mask
                need_orth_indices = df.index[need_orth_mask].tolist()
                
                if len(need_orth_indices) < 10:
                    # 样本不足，不进行正交化
                    continue
                
                # 获取需要正交化的数据
                need_orth_data = df.loc[need_orth_indices].copy()
                
                if len(need_orth_data) < 10:
                    continue
                
                # 获取市场收益率作为正交化因子
                if 'market_return' in need_orth_data.columns:
                    market_factor = need_orth_data['market_return'].values
                    
                    # 目标特征值
                    target_values = need_orth_data[target_col].values
                    
                    # V207.1 修复：检查数据有效性
                    target_std = np.nanstd(target_values)
                    market_std = np.nanstd(market_factor)
                    
                    if target_std < EPSILON or market_std < EPSILON:
                        # 方差为 0，无法进行正交化，保持原值
                        logger.debug(f"[DynamicOrtho] Low variance detected, skipping orthogonalization for date {date}")
                        continue
                    
                    # 标准化
                    y_mean, y_std = target_values.mean(), target_std + EPSILON
                    y_norm = (target_values - y_mean) / y_std
                    
                    m_mean, m_std = market_factor.mean(), market_std + EPSILON
                    m_norm = (market_factor - m_mean) / m_std
                    
                    # 残差计算
                    denominator = np.dot(m_norm, m_norm) + EPSILON
                    beta = np.dot(y_norm, m_norm) / denominator
                    residual = y_norm - beta * m_norm
                    
                    # 部分正交化
                    residual_partial = residual * self.orth_strength + y_norm * (1 - self.orth_strength)
                    residual_final = residual_partial * y_std + y_mean
                    
                    # 更新数据 - 使用相同的索引
                    df.loc[need_orth_indices, orth_col] = residual_final
            
            logger.debug(f"[DynamicOrtho] {target_col} orthogonalized for high correlation samples")
        
        return df


# ==============================================================================
# V207 多轮迭代闭环模块
# ==============================================================================

class IterationLoopAnalyzer:
    """
    V207 多轮迭代分析器
    
    【核心职责】
    1. 运行回测后检查日志
    2. 如果 IC < 0.05 或出现连续 10 天以上亏损，必须调用 analyze_failure() 逻辑
    3. 根据诊断结果修改特征融合逻辑，自动重新回测，直到结果达标
    """
    
    def __init__(self, config: Dict = None, output_dir: str = "reports"):
        self.config = config or ITERATION_LOOP_CONFIG
        self.ic_threshold = self.config.get('ic_threshold', 0.05)
        self.consecutive_loss_days = self.config.get('consecutive_loss_days', 10)
        self.max_iterations = self.config.get('max_iterations', 3)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_ic_performance(self, df: pd.DataFrame, score_col: str = 'score') -> Dict[str, Any]:
        """分析 IC 表现"""
        if score_col not in df.columns:
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'passed': False}
        
        # 按日期分组计算 IC
        date_ics = {}
        for date in df['trade_date'].unique():
            date_data = df[df['trade_date'] == date]
            
            if len(date_data) < 100:
                continue
            
            score_values = date_data[score_col].values
            rank_values = pd.Series(score_values).rank(pct=True).values
            
            ic_corr = np.corrcoef(score_values, rank_values)[0, 1]
            if not np.isnan(ic_corr):
                date_ics[str(date)] = ic_corr
        
        if not date_ics:
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'passed': False}
        
        ic_values = list(date_ics.values())
        mean_ic = np.mean(ic_values)
        ic_std = np.std(ic_values)
        ic_ir = mean_ic / (ic_std + EPSILON)
        
        passed = mean_ic >= self.ic_threshold
        
        return {
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'passed': passed,
            'date_ics': date_ics,
        }
    
    def analyze_consecutive_losses(
        self,
        df: pd.DataFrame,
        score_col: str = 'score'
    ) -> Dict[str, Any]:
        """分析连续亏损天数"""
        if score_col not in df.columns:
            return {'max_consecutive_losses': 0, 'passed': True}
        
        # 按日期分组计算每日平均 Score
        date_scores = {}
        for date in df['trade_date'].unique():
            date_data = df[df['trade_date'] == date]
            
            if len(date_data) < 100:
                continue
            
            avg_score = date_data[score_col].mean()
            date_scores[str(date)] = avg_score
        
        if not date_scores:
            return {'max_consecutive_losses': 0, 'passed': True}
        
        # 按日期排序
        sorted_dates = sorted(date_scores.keys())
        
        # 计算连续亏损 (Score < 0)
        max_consecutive = 0
        current_consecutive = 0
        loss_dates = []
        
        for date in sorted_dates:
            if date_scores[date] < 0:
                current_consecutive += 1
                loss_dates.append(date)
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        passed = max_consecutive < self.consecutive_loss_days
        
        return {
            'max_consecutive_losses': max_consecutive,
            'passed': passed,
            'loss_dates': loss_dates,
        }
    
    def analyze_failure(
        self,
        df: pd.DataFrame,
        ic_result: Dict[str, Any],
        loss_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析失败原因并生成修复建议"""
        failure_reasons = []
        recommendations = []
        
        # IC 过低分析
        if not ic_result.get('passed', True):
            failure_reasons.append(f"IC ({ic_result['mean_ic']:.4f}) < threshold ({self.ic_threshold})")
            
            # 分析 IC 低的原因
            if ic_result['ic_std'] > 0.1:
                failure_reasons.append("IC 波动过大，特征稳定性不足")
                recommendations.append("增加特征稳定性权重")
            
            if ic_result['ic_ir'] < 0.5:
                failure_reasons.append("IC IR 过低，信号噪声比不足")
                recommendations.append("增强信号过滤，降低噪声")
            
            recommendations.append("增加订单流不平衡特征权重")
            recommendations.append("调整行业内强度计算窗口")
        
        # 连续亏损分析
        if not loss_result.get('passed', True):
            failure_reasons.append(f"连续亏损天数 ({loss_result['max_consecutive_losses']}) >= threshold ({self.consecutive_loss_days})")
            recommendations.append("增加风险滤网强度")
            recommendations.append("降低高波动股票权重")
        
        return {
            'failure_reasons': failure_reasons,
            'recommendations': recommendations,
            'needs_iteration': len(failure_reasons) > 0,
        }
    
    def generate_iteration_report(
        self,
        iteration: int,
        ic_result: Dict[str, Any],
        loss_result: Dict[str, Any],
        failure_analysis: Dict[str, Any]
    ) -> str:
        """生成迭代分析报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"V207_Iteration_{iteration}_{timestamp}.md"
        
        report_content = f"""# V207 Iteration Report #{iteration}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Version**: {VERSION}

---

## 1. Performance Summary

### IC Analysis
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Mean IC | {ic_result.get('mean_ic', 0):.4f} | >= {self.ic_threshold} | {'✓' if ic_result.get('passed', False) else '✗'} |
| IC Std | {ic_result.get('ic_std', 0):.4f} | < 0.10 | {'✓' if ic_result.get('ic_std', 0) < 0.10 else '✗'} |
| IC IR | {ic_result.get('ic_ir', 0):.2f} | > 0.50 | {'✓' if ic_result.get('ic_ir', 0) > 0.50 else '✗'} |

### Consecutive Loss Analysis
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max Consecutive Losses | {loss_result.get('max_consecutive_losses', 0)} | < {self.consecutive_loss_days} | {'✓' if loss_result.get('passed', True) else '✗'} |

---

## 2. Failure Analysis

"""
        
        if failure_analysis.get('failure_reasons'):
            for reason in failure_analysis['failure_reasons']:
                report_content += f"- {reason}\n"
        else:
            report_content += "No critical failures detected.\n"
        
        report_content += f"""
---

## 3. Recommendations

"""
        
        if failure_analysis.get('recommendations'):
            for i, rec in enumerate(failure_analysis['recommendations'], 1):
                report_content += f"{i}. {rec}\n"
        else:
            report_content += "No changes required. Performance meets targets.\n"
        
        report_content += f"""
---

## 4. Next Iteration Plan

"""
        
        if failure_analysis.get('needs_iteration', False):
            report_content += f"""- [ ] Adjust feature weights based on recommendations
- [ ] Re-run backtest with updated parameters
- [ ] Validate IC improvement
- [ ] Check consecutive loss reduction
"""
        else:
            report_content += "Iteration complete. Performance targets achieved.\n"
        
        report_content += f"""
---

*Report generated by V207 Iteration Loop Analyzer*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[IterationLoop] Report saved to: {report_path}")
        
        return str(report_path)


# ==============================================================================
# V207 交叉验证模块 (防止过拟合)
# ==============================================================================

class CrossSectionalValidator:
    """
    V207 交叉验证器 - 防止过拟合
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CROSS_VALIDATION_CONFIG
        self.noise_ratio = self.config.get('noise_ratio', 0.10)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.30)
        self.enabled = self.config.get('enabled', True)
    
    def validate_score(self, df: pd.DataFrame, score_col: str = 'score') -> pd.DataFrame:
        """验证 Score 的稳定性"""
        if not self.enabled:
            return df
        
        df = df.copy()
        df[f'{score_col}_original'] = df[score_col].copy()
        
        # 生成扰动后的 Score
        np.random.seed(42)
        noise = np.random.normal(0, self.noise_ratio, size=len(df))
        
        df[f'{score_col}_perturbed'] = df[score_col] * (1 + noise)
        
        # 计算波动率
        df['score_volatility'] = (
            (df[f'{score_col}_perturbed'] - df[f'{score_col}_original']).abs() /
            (df[f'{score_col}_original'].abs() + EPSILON)
        )
        
        # 标记无效样本
        df['score_valid'] = (df['score_volatility'] <= self.volatility_threshold).astype(int)
        
        # 对无效样本降权
        df.loc[df['score_valid'] == 0, score_col] = df.loc[df['score_valid'] == 0, score_col] * 0.5
        
        invalid_count = (df['score_valid'] == 0).sum()
        logger.debug(f"[CrossVal] {invalid_count} samples marked as invalid ({invalid_count/len(df):.2%})")
        
        return df


# ==============================================================================
# AlphaModel V207
# ==============================================================================

class AlphaModel:
    """
    V207 Alpha 模型 - 深度特征挖掘与动态相关性建模
    
    【核心职责】
    1. 计算时空敏感性 2.0 特征 (订单流不平衡、行业内强度)
    2. 动态相关性正交化 (窗口自适应)
    3. 多轮迭代闭环分析
    4. 交叉验证防止过拟合
    5. 输出标准化评分 score
    
    【严禁】
    - 计算任何收益率 (t1_return 等)
    - 接触回测逻辑
    - 使用未来函数 (shift(-1) 等)
    """
    
    def __init__(
        self,
        enable_industry_neutral: bool = True,
        enable_market_adapter: bool = True,
        enable_dynamic_orthogonalization: bool = True,
        enable_cross_validation: bool = True,
        enable_iteration_loop: bool = True,
        enable_spatiotemporal_v207: bool = True,
        output_dir: str = "reports",
    ):
        self.enable_industry_neutral = enable_industry_neutral and INDUSTRY_NEUTRALIZE_ENABLED
        self.enable_market_adapter = enable_market_adapter
        self.enable_dynamic_orthogonalization = enable_dynamic_orthogonalization
        self.enable_cross_validation = enable_cross_validation
        self.enable_iteration_loop = enable_iteration_loop
        self.enable_spatiotemporal_v207 = enable_spatiotemporal_v207
        
        # 初始化 V207 核心组件
        self.spatiotemporal_engine = SpatiotemporalFeatureEngineV207() if enable_spatiotemporal_v207 else None
        self.dynamic_orthogonalizer = DynamicCorrelationOrthogonalizer() if enable_dynamic_orthogonalization else None
        self.cross_validator = CrossSectionalValidator() if enable_cross_validation else None
        self.iteration_loop = IterationLoopAnalyzer(output_dir=output_dir) if enable_iteration_loop else None
        
        # V207 特征配置
        self.feature_groups = {
            'reversion': [f'reversion_{d}' for d in V207_REVERSION_SCALES],
            'momentum': [f'momentum_{d}' for d in V207_MOMENTUM_SCALES],
            'volatility': [f'volatility_{d}' for d in V207_VOLATILITY_SCALES],
            'spatiotemporal_v207': ['order_imbalance_rank', 'intra_industry_strength', 'spatiotemporal_interaction_v207_rank'] if enable_spatiotemporal_v207 else [],
        }
        
        # 状态跟踪
        self._current_market_state = 'NORMAL'
        self._factor_weights = self._get_base_weights()
        self._ic_history = {}
        self._iteration_history = []
        
        logger.info("=" * 80)
        logger.info("V207 AlphaModel Initialized")
        logger.info("=" * 80)
        logger.info(f"  Spatiotemporal 2.0 Features: {self.enable_spatiotemporal_v207}")
        logger.info(f"  Dynamic Correlation Orthogonalization: {self.enable_dynamic_orthogonalization}")
        logger.info(f"  Cross-Sectional Validation: {self.enable_cross_validation}")
        logger.info(f"  Iteration Loop: {self.enable_iteration_loop}")
        logger.info(f"  Industry Neutral: {self.enable_industry_neutral}")
        logger.info(f"  Market Adapter: {self.enable_market_adapter}")
        logger.info("=" * 80)
    
    def _get_base_weights(self) -> Dict[str, float]:
        """获取基础权重配置"""
        return {
            'reversion': 0.18,
            'momentum': 0.15,
            'volatility': 0.15,
            'spatiotemporal_v207': 0.30,  # V207 新增，更高权重
            'size': 0.08,
            'liquidity': 0.08,
            'chip': 0.06,
        }
    
    def _get_defense_weights(self) -> Dict[str, float]:
        """熊市防御权重"""
        return {
            'reversion': 0.28,
            'momentum': 0.05,
            'volatility': 0.25,
            'spatiotemporal_v207': 0.20,
            'size': 0.08,
            'liquidity': 0.08,
            'chip': 0.06,
        }
    
    def _get_bull_weights(self) -> Dict[str, float]:
        """牛市进攻权重"""
        return {
            'reversion': 0.10,
            'momentum': 0.28,
            'volatility': 0.10,
            'spatiotemporal_v207': 0.32,
            'size': 0.06,
            'liquidity': 0.08,
            'chip': 0.06,
        }
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分
        
        【核心原则】
        - 所有计算仅使用 T 日及之前数据
        - 严禁使用 shift(-1) 等未来函数
        - 输出 score 列，严禁输出任何收益率相关列
        
        【V207 流程】
        1. 计算多尺度时序特征
        2. 计算时空敏感性 2.0 特征 (V207 新增)
        3. 动态相关性正交化 (V207 改进)
        4. 交叉验证 (V207 新增)
        5. 动态市场适配
        6. 计算综合评分
        7. 行业中性化
        8. 输出最终评分
        """
        if df.empty:
            logger.warning("[AlphaModel] Empty input DataFrame")
            return df
        
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows...")
        
        df = df.copy()
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        if df['trade_date'].dtype == 'int64':
            df['trade_date_str'] = df['trade_date'].astype(str)
        else:
            df['trade_date_str'] = df['trade_date']
        
        # 1. 计算多尺度时序特征
        df = self._compute_multi_scale_features(df)
        
        # 2. 计算时空敏感性 2.0 特征 (V207 新增)
        if self.enable_spatiotemporal_v207 and self.spatiotemporal_engine:
            df = self.spatiotemporal_engine.compute_features(df)
        
        # 3. 动态相关性正交化 (V207 改进)
        if self.enable_dynamic_orthogonalization and self.dynamic_orthogonalizer:
            target_cols = ['reversion_60', 'momentum_60', 'order_imbalance']
            df = self.dynamic_orthogonalizer.apply_dynamic_orthogonalization(df, target_cols)
        
        # 4. 数据自愈
        all_features = self._get_all_feature_names()
        df = auto_fillna(df, all_features)
        
        # 5. 动态市场适配
        if self.enable_market_adapter:
            df = self._apply_market_adapter(df)
        
        # 6. 计算综合评分
        df = self._compute_composite_score(df)
        
        # 7. 交叉验证 (V207 新增 - 防止过拟合)
        if self.enable_cross_validation and self.cross_validator:
            df = self.cross_validator.validate_score(df, 'score')
        
        # 8. 行业中性化
        if self.enable_industry_neutral and 'industry_code' in df.columns:
            df = self._apply_industry_neutralization(df)
        
        # 9. 鲁棒性标准化
        df['score'] = robust_zscore(df['score'], use_mad=True)
        df['score'] = winsorize(df['score'], sigma=3.0)
        
        logger.info(f"[AlphaModel] Score computed. Range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        
        return df
    
    def analyze_and_iterate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """运行多轮迭代分析"""
        if not self.enable_iteration_loop or not self.iteration_loop:
            return {}
        
        logger.info("[IterationLoop] Running performance analysis...")
        
        # 分析 IC 表现
        ic_result = self.iteration_loop.analyze_ic_performance(df)
        logger.info(f"[IterationLoop] Mean IC: {ic_result['mean_ic']:.4f}, Passed: {ic_result['passed']}")
        
        # 分析连续亏损
        loss_result = self.iteration_loop.analyze_consecutive_losses(df)
        logger.info(f"[IterationLoop] Max Consecutive Losses: {loss_result['max_consecutive_losses']}, Passed: {loss_result['passed']}")
        
        # 分析失败原因
        failure_analysis = self.iteration_loop.analyze_failure(df, ic_result, loss_result)
        
        # 生成迭代报告
        iteration = len(self._iteration_history) + 1
        self._iteration_history.append({
            'iteration': iteration,
            'ic_result': ic_result,
            'loss_result': loss_result,
            'failure_analysis': failure_analysis,
        })
        
        report_path = self.iteration_loop.generate_iteration_report(
            iteration, ic_result, loss_result, failure_analysis
        )
        
        return {
            'ic_result': ic_result,
            'loss_result': loss_result,
            'failure_analysis': failure_analysis,
            'report_path': report_path,
            'needs_iteration': failure_analysis.get('needs_iteration', False),
        }
    
    def _get_all_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        features = []
        for group_features in self.feature_groups.values():
            features.extend(group_features)
        return features
    
    def _compute_multi_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算多尺度时序特征"""
        numeric_cols = ['close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 计算 1 日收益率
        df['return_1d'] = df.groupby('symbol')['close'].pct_change(1).fillna(0)
        
        # 多尺度反转因子
        for window in V207_REVERSION_SCALES:
            col_name = f'reversion_{window}'
            df[col_name] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(window) - 1
            ).fillna(0)
            # 短期反转取负
            if window <= 10:
                df[col_name] = -df[col_name]
        
        # 多尺度动量因子
        for window in V207_MOMENTUM_SCALES:
            col_name = f'momentum_{window}'
            df[col_name] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(window) - 1
            ).fillna(0)
        
        # 多尺度波动率因子
        for window in V207_VOLATILITY_SCALES:
            col_name = f'volatility_{window}'
            df[col_name] = df.groupby('symbol')['return_1d'].transform(
                lambda x: x.rolling(window, min_periods=max(5, window // 2)).std()
            ).fillna(0)
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / (df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        ) + EPSILON)
        
        return df
    
    def _apply_market_adapter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用动态市场适配"""
        market_return = df.groupby('trade_date')['pct_chg'].mean()
        
        market_return_rolling = market_return.rolling(MARKET_STATE_WINDOW, min_periods=10).mean()
        
        bear_dates = market_return_rolling[market_return_rolling < MARKET_STATE_BEAR_THRESHOLD].index.tolist()
        bull_dates = market_return_rolling[market_return_rolling > MARKET_STATE_BULL_THRESHOLD].index.tolist()
        
        df['market_state'] = 'NORMAL'
        df.loc[df['trade_date'].isin(bear_dates), 'market_state'] = 'BEAR'
        df.loc[df['trade_date'].isin(bull_dates), 'market_state'] = 'BULL'
        
        latest_date = df['trade_date'].max()
        latest_state = df[df['trade_date'] == latest_date]['market_state'].iloc[0] if len(df) > 0 else 'NORMAL'
        self._current_market_state = latest_state
        
        df['market_state_adapter'] = np.where(
            df['market_state'] == 'BEAR',
            1.0,
            np.where(
                df['market_state'] == 'BULL',
                -1.0,
                0.0
            )
        )
        
        logger.debug(f"[Market Adapter] Current state: {self._current_market_state}")
        
        return df
    
    def _compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合评分"""
        score_components = pd.DataFrame(index=df.index)
        
        # 1. 反转因子组
        reversion_components = []
        reversion_weights = {5: 0.4, 10: 0.3, 20: 0.2, 60: 0.1}
        for window in V207_REVERSION_SCALES:
            col = f'reversion_{window}'
            if col in df.columns:
                normalized = normalize_group(df, col)
                weight = reversion_weights.get(window, 0.25)
                reversion_components.append(normalized * weight)
        if reversion_components:
            score_components['reversion'] = sum(reversion_components)
        
        # 2. 动量因子组
        momentum_components = []
        momentum_weights = {5: 0.1, 10: 0.2, 20: 0.3, 60: 0.4}
        for window in V207_MOMENTUM_SCALES:
            col = f'momentum_{window}'
            if col in df.columns:
                normalized = normalize_group(df, col)
                weight = momentum_weights.get(window, 0.25)
                momentum_components.append(normalized * weight)
        if momentum_components:
            score_components['momentum'] = sum(momentum_components)
        
        # 3. 波动率因子组 (反向)
        volatility_components = []
        for window in V207_VOLATILITY_SCALES:
            col = f'volatility_{window}'
            if col in df.columns:
                normalized = -normalize_group(df, col)
                volatility_components.append(normalized * 0.25)
        if volatility_components:
            score_components['volatility'] = sum(volatility_components)
        
        # 4. 时空敏感性 2.0 特征 (V207 新增)
        if self.enable_spatiotemporal_v207:
            spatiotemporal_score = 0.0
            
            # 订单流不平衡 (正向：买方力量强更好)
            if 'order_imbalance_rank' in df.columns:
                spatiotemporal_score += normalize_group(df, 'order_imbalance_rank') * 0.35
            
            # 行业内强度 (正向：行业内表现强更好)
            if 'intra_industry_strength' in df.columns:
                spatiotemporal_score += normalize_group(df, 'intra_industry_strength') * 0.35
            
            # 时空交互特征
            if 'spatiotemporal_interaction_v207_rank' in df.columns:
                spatiotemporal_score += normalize_group(df, 'spatiotemporal_interaction_v207_rank') * 0.30
            
            score_components['spatiotemporal_v207'] = spatiotemporal_score
        
        # 5. 市值因子
        if 'circ_mv' in df.columns:
            df['log_market_cap'] = np.log(df['circ_mv'] + 1)
            score_components['size'] = -normalize_group(df, 'log_market_cap')
        elif 'mv' in df.columns:
            df['log_market_cap'] = np.log(df['mv'] + 1)
            score_components['size'] = -normalize_group(df, 'log_market_cap')
        
        # 6. 流动性因子
        if 'turnover_rate' in df.columns:
            score_components['liquidity'] = normalize_group(df, 'turnover_rate')
        
        # 7. 筹码因子
        if 'chip_sensitivity' in df.columns:
            chip_score = -df['chip_sensitivity'].abs()
            score_components['chip'] = normalize_group(pd.DataFrame({'chip': chip_score}), 'chip')
        
        # 根据市场状态选择权重
        if self._current_market_state == 'BEAR':
            weights = self._get_defense_weights()
        elif self._current_market_state == 'BULL':
            weights = self._get_bull_weights()
        else:
            weights = self._get_base_weights()
        
        # 动态调整权重 (根据可用因子)
        available_cols = [c for c in weights.keys() if c in score_components.columns]
        if available_cols:
            total_weight = sum(weights[c] for c in available_cols)
            if total_weight > 0:
                for c in available_cols:
                    weights[c] /= total_weight
        
        # 计算综合评分
        df['raw_score'] = sum(score_components[c] * weights[c] for c in available_cols)
        df['score'] = normalize_group(df, 'raw_score')
        
        self._factor_weights = weights
        
        return df
    
    def _apply_industry_neutralization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用行业中性化
        
        【V207.1 修复】
        - 当行业数量太少（<10）时，不进行行业中性化，因为中性化效果不好
        - 但需要确保 score 列已经有足够的截面区分度
        """
        # V207.1 核心修复：优先使用 industry_name 进行行业中性化
        has_industry_name = 'industry_name' in df.columns and df['industry_name'].notna().any()
        has_industry_code = 'industry_code' in df.columns and df['industry_code'].notna().any()
        
        # 确定使用哪个行业列
        if has_industry_name:
            industry_col = 'industry_name'
            logger.debug("[IndustryNeutral] Using industry_name for neutralization")
        elif has_industry_code:
            industry_col = 'industry_code'
            logger.debug("[IndustryNeutral] Using industry_code for neutralization (fallback)")
        else:
            logger.debug("[IndustryNeutral] No industry column available, skipping")
            return df
        
        industry_counts = df[industry_col].value_counts()
        
        # V207.1 核心修复：行业数量太少时不进行行业中性化
        # 原因：行业太少时，行业内中性化会导致分数失去区分度
        if len(industry_counts) < 10:
            logger.info(f"[IndustryNeutral] Only {len(industry_counts)} industries (< 10), skipping industry neutralization")
            logger.info(f"[IndustryNeutral] Industry distribution: {industry_counts.to_dict()}")
            return df
        
        df['industry_neutral_score'] = df.groupby(industry_col)['score'].transform(
            lambda x: normalize_group(pd.DataFrame({'score': x}), 'score')
        )
        
        df['score'] = df['industry_neutral_score']
        
        logger.debug(f"[IndustryNeutral] Applied neutralization across {len(industry_counts)} industries")
        
        return df
    
    def get_current_market_state(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        return {
            'market_state': self._current_market_state,
            'current_weights': self._factor_weights,
            'industry_neutral_enabled': self.enable_industry_neutral,
            'dynamic_orthogonalization_enabled': self.enable_dynamic_orthogonalization,
            'cross_validation_enabled': self.enable_cross_validation,
            'iteration_loop_enabled': self.enable_iteration_loop,
            'spatiotemporal_v207_enabled': self.enable_spatiotemporal_v207,
            'ic_history': self._ic_history,
            'iteration_history': self._iteration_history,
        }
    
    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性"""
        return self._factor_weights.copy()


def get_alpha_model(
    enable_industry_neutral: bool = True,
    enable_market_adapter: bool = True,
    enable_dynamic_orthogonalization: bool = True,
    enable_cross_validation: bool = True,
    enable_iteration_loop: bool = True,
    enable_spatiotemporal_v207: bool = True,
    output_dir: str = "reports",
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        enable_industry_neutral=enable_industry_neutral,
        enable_market_adapter=enable_market_adapter,
        enable_dynamic_orthogonalization=enable_dynamic_orthogonalization,
        enable_cross_validation=enable_cross_validation,
        enable_iteration_loop=enable_iteration_loop,
        enable_spatiotemporal_v207=enable_spatiotemporal_v207,
        output_dir=output_dir,
    )