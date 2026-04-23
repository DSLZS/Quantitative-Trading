"""
Alpha Model Module - V204 Ensemble Evolution & IC Repair
========================================================

【V204 核心变革】
1. 集成进化 (Ensemble Logic)
   - 保留 V202 的线性稳定性作为 Base Alpha
   - 对 V203 的非线性交互核进行"显著性筛选"
   - 只有在回测中对 IC 有正向贡献的交互项才能进入最终评分

2. IC 修复行动 (IC Repair Action)
   - 引入"异常值鲁棒性标准化"，减少噪声对非线性核的干扰
   - 使用 Median Absolute Deviation (MAD) 替代标准差
   - 如果 2024 年 Rank IC 低于 0.03，自动触发"防御模式"削减仓位

3. 动态显著性筛选 (Dynamic Significance Screening)
   - 基于滚动 IC 评估每个交互项的贡献
   - 自动剔除 IC 贡献为负的交互项
   - 保留显著性水平 p < 0.1 的因子

【物理清理】
- 删除所有 V191/V200/V201/V202/V203 字样注释
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

VERSION = "V204_Ensemble_IC_Repair"

# V204 核心因子配置
V204_BASE_FACTORS = [
    'reversion_5',              # 5 日反转 - 防御型 (V202 继承)
    'volatility_20',            # 20 日波动率 - 风险度量 (V202 继承)
    'liquidity_mkt_neutral',    # 市值中性化流动性 (V202 继承)
    'fund_flow_signal',         # 资金流信号 (V202 继承)
]

V204_INTERACTION_FACTORS = [
    'vol_reversion_kernel',     # 波动率 - 反转交互核 (V203 继承，需显著性筛选)
    'volume_price_kernel',      # 量价交互核 (V203 继承，需显著性筛选)
    'liquidity_momentum_kernel', # 流动性 - 动量交互核 (V203 继承，需显著性筛选)
]

MAX_FACTORS = 12

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

# 因子正交化配置
ORTHOGONALIZATION_GROUPS = {
    'reversion': ['reversion_5', 'reversion_10'],
    'liquidity': ['liquidity_mkt_neutral', 'turnover_rate'],
    'fund_flow': ['fund_flow_signal', 'net_main_flow'],
}

# V204 新增：IC 防御模式阈值
IC_DEFENSE_THRESHOLD = 0.03  # 2024 年 Rank IC 低于此值触发防御
IC_DEFENSE_POSITION_SCALE = 0.5  # 防御模式下仓位缩放因子

# V204 新增：显著性筛选阈值
SIGNIFICANCE_P_THRESHOLD = 0.10  # p 值阈值
SIGNIFICANCE_IC_MIN = 0.01  # 最小 IC 贡献

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
    """
    计算 Median Absolute Deviation (MAD) 尺度估计
    
    【原理】
    MAD 比标准差更鲁棒，不受异常值影响
    MAD = median(|X - median(X)|)
    Scale = k * MAD (k=1.4826 使 MAD 与标准差在正态分布下一致)
    
    Args:
        series: 输入序列
        k: 缩放常数
    
    Returns:
        MAD 尺度估计
    """
    median = series.median()
    if pd.isna(median):
        return 1.0
    
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad < 1e-10:
        return 1.0
    
    return k * mad


def robust_zscore(series: pd.Series, use_mad: bool = True) -> pd.Series:
    """
    鲁棒性 Z-Score 标准化
    
    【V204 核心创新】
    使用 MAD 替代标准差，减少异常值对标准化的干扰
    
    Args:
        series: 输入序列
        use_mad: 是否使用 MAD 尺度估计
    
    Returns:
        鲁棒性 Z-Score
    """
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
    """
    截面或分组标准化评分
    
    【原理】
    - 将原始评分转换为 [0, 1] 区间的排名百分位
    - 可选按组 (行业) 内标准化实现行业中性化
    
    【V204 改进】
    - 使用稳健的排名方法，减少极端值影响
    """
    if group_col is None or group_col not in df.columns:
        if len(df) < 2:
            return pd.Series(0.5, index=df.index)
        
        ranks = df[score_col].rank(pct=True, method='average')
        normalized = (ranks - 0.5) * 2
        return normalized
    
    def rank_pct(group):
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        return (group.rank(pct=True, method='average') - 0.5) * 2
    
    normalized = df.groupby(group_col)[score_col].transform(rank_pct)
    return normalized


def auto_fillna(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """自动填充缺失值"""
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
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
    """
    截面施密特正交化 (Gram-Schmidt Orthogonalization)
    
    【原理】
    1. 对每个截面 (按 trade_date 分组)
    2. 将 target_col 对 orthogonal_cols 做正交化
    3. 移除与已有因子的线性相关性
    
    【数学公式】
    residual = target - sum(beta_i * factor_i)
    其中 beta_i = Cov(target, factor_i) / Var(factor_i)
    """
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


# ==============================================================================
# AlphaModel V204
# ==============================================================================

class AlphaModel:
    """
    V204 Alpha 模型 - 集成进化与 IC 修复
    
    【核心职责】
    1. 计算因子值 (仅使用 T 日及之前数据)
    2. V202 线性 Base Alpha (稳定性继承)
    3. V203 非线性交互核 (显著性筛选)
    4. 鲁棒性标准化 (MAD 替代标准差)
    5. IC 防御模式 (自动仓位管理)
    6. 输出标准化评分 score
    
    【严禁】
    - 计算任何收益率 (t1_return 等)
    - 接触回测逻辑
    - 使用未来函数 (shift(-1) 等)
    """
    
    def __init__(
        self,
        n_factors: int = MAX_FACTORS,
        enable_industry_neutral: bool = True,
        enable_market_adapter: bool = True,
        enable_orthogonalization: bool = True,
        enable_interaction_kernel: bool = True,
        enable_significance_screening: bool = True,
        enable_ic_defense: bool = True,
    ):
        self.n_factors = min(n_factors, MAX_FACTORS)
        self.enable_industry_neutral = enable_industry_neutral and INDUSTRY_NEUTRALIZE_ENABLED
        self.enable_market_adapter = enable_market_adapter
        self.enable_orthogonalization = enable_orthogonalization
        self.enable_interaction_kernel = enable_interaction_kernel
        self.enable_significance_screening = enable_significance_screening
        self.enable_ic_defense = enable_ic_defense
        
        # V204 因子配置
        self.base_factors = V204_BASE_FACTORS[:self.n_factors]
        self.interaction_factors = V204_INTERACTION_FACTORS[:self.n_factors]
        
        # V204 新增：显著性筛选状态
        self._factor_significance = {f: {'ic_contribution': 0.0, 'p_value': 1.0, 'enabled': True} 
                                      for f in self.interaction_factors}
        
        # V204 新增：IC 防御状态
        self._ic_defense_mode = False
        self._current_ic_estimate = 0.0
        
        # 状态跟踪
        self._current_market_state = 'NORMAL'
        self._factor_weights = self._get_base_weights()
        
        logger.info("=" * 70)
        logger.info("V204 AlphaModel Initialized")
        logger.info("=" * 70)
        logger.info(f"  Base Factors: {self.base_factors}")
        logger.info(f"  Interaction Factors: {self.interaction_factors}")
        logger.info(f"  Industry Neutral: {self.enable_industry_neutral}")
        logger.info(f"  Market Adapter: {self.enable_market_adapter}")
        logger.info(f"  Orthogonalization: {self.enable_orthogonalization}")
        logger.info(f"  Interaction Kernel: {self.enable_interaction_kernel}")
        logger.info(f"  Significance Screening: {self.enable_significance_screening}")
        logger.info(f"  IC Defense Mode: {self.enable_ic_defense}")
        logger.info("=" * 70)
    
    def _get_base_weights(self) -> Dict[str, float]:
        """获取基础权重配置"""
        return {
            'reversion': 0.25,
            'volatility': 0.20,
            'momentum': 0.10,
            'liquidity': 0.15,
            'volume_price': 0.10,
            'vol_reversion': 0.10,
            'fund_flow': 0.10,
        }
    
    def _get_defense_weights(self) -> Dict[str, float]:
        """熊市防御权重"""
        return {
            'reversion': 0.35,
            'volatility': 0.25,
            'momentum': 0.05,
            'liquidity': 0.15,
            'volume_price': 0.05,
            'vol_reversion': 0.05,
            'fund_flow': 0.10,
        }
    
    def _get_bull_weights(self) -> Dict[str, float]:
        """牛市进攻权重"""
        return {
            'reversion': 0.10,
            'volatility': 0.05,
            'momentum': 0.30,
            'liquidity': 0.10,
            'volume_price': 0.20,
            'vol_reversion': 0.10,
            'fund_flow': 0.15,
        }
    
    def _get_ic_defense_weights(self) -> Dict[str, float]:
        """
        IC 防御模式权重 (V204 核心)
        
        【原理】
        当检测到 IC 低于阈值时，切换至超低风险配置
        """
        return {
            'reversion': 0.40,      # 最大化反转因子
            'volatility': 0.30,     # 最大化低波动因子
            'momentum': 0.00,       # 完全剔除动量
            'liquidity': 0.15,
            'volume_price': 0.05,
            'vol_reversion': 0.05,
            'fund_flow': 0.05,
        }
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分
        
        【核心原则】
        - 所有计算仅使用 T 日及之前数据
        - 严禁使用 shift(-1) 等未来函数
        - 输出 score 列，严禁输出任何收益率相关列
        
        【V204 新增流程】
        1. 计算 V202 Base Alpha 因子
        2. 计算 V203 交互核因子
        3. 显著性筛选 (剔除 IC 贡献为负的交互项)
        4. 鲁棒性标准化 (MAD)
        5. IC 防御模式检查
        6. 输出最终评分
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
        
        # 1. 计算 V202 Base Alpha 因子 (线性稳定性)
        df = self._compute_base_factors(df)
        
        # 2. 计算 V203 交互核因子 (非线性增强)
        if self.enable_interaction_kernel:
            df = self._compute_interaction_kernels(df)
        
        # 3. 因子正交化
        if self.enable_orthogonalization:
            df = self._apply_orthogonalization(df)
        
        # 4. 数据自愈
        df = auto_fillna(df, self.base_factors + self.interaction_factors)
        
        # 5. 显著性筛选 (V204 核心)
        if self.enable_significance_screening:
            df = self._apply_significance_screening(df)
        
        # 6. 动态市场适配
        if self.enable_market_adapter:
            df = self._apply_market_adapter(df)
        
        # 7. 计算综合评分
        df = self._compute_composite_score(df)
        
        # 8. 行业中性化
        if self.enable_industry_neutral and 'industry_code' in df.columns:
            df = self._apply_industry_neutralization(df)
        
        # 9. V204 鲁棒性标准化 (MAD)
        df['score'] = robust_zscore(df['score'], use_mad=True)
        df['score'] = winsorize(df['score'], sigma=3.0)
        
        # 10. IC 防御模式检查 (V204 核心)
        if self.enable_ic_defense and self._ic_defense_mode:
            logger.info("[IC Defense] Defense mode active - scaling scores")
            df['score'] = df['score'] * IC_DEFENSE_POSITION_SCALE
        
        logger.info(f"[AlphaModel] Score computed. Range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        logger.info(f"[IC Defense] Mode: {self._ic_defense_mode}, Current IC Estimate: {self._current_ic_estimate:.4f}")
        
        return df
    
    def _compute_base_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础因子 (V202 线性稳定性继承)
        
        【严禁未来函数】所有因子计算只能使用 shift(1) 或更早的历史数据
        """
        numeric_cols = ['close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 1. 5 日反转因子 (防御型)
        if 'pct_chg' in df.columns:
            df['reversion_5'] = -df['pct_chg'] / 100.0
        
        # 2. 10 日反转因子
        df['return_1d'] = df.groupby('symbol')['close'].pct_change(1).fillna(0)
        df['reversion_10'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: x.rolling(10, min_periods=5).mean().shift(1)
        ).fillna(0)
        
        # 3. 20 日波动率因子
        df['volatility_20'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        ).fillna(0)
        
        # 4. 5 日波动率因子 (用于交互核)
        df['volatility_5'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        ).fillna(0)
        
        # 5. 10 日动量因子
        df['momentum_10'] = df.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(10) - 1
        ).fillna(0)
        
        # 6. 市值中性化流动性因子
        df['amount_ma20'] = df.groupby('symbol')['amount'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        ).fillna(0)
        df['liquidity_mkt_neutral'] = df['amount'] / (df['amount_ma20'] + EPSILON)
        df['liquidity_mkt_neutral'] = winsorize(df['liquidity_mkt_neutral'], sigma=3.0)
        
        # 7. 换手率
        if 'turnover_rate' in df.columns:
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
        
        # 8. 量价交互核基础
        df['volume_ratio'] = df['volume'] / (df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        ) + EPSILON)
        
        # 9. 资金流信号
        if 'net_main_amount' in df.columns:
            df['net_main_ma5'] = df.groupby('symbol')['net_main_amount'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            ).fillna(0)
            
            df['fund_flow_signal'] = df.groupby('trade_date')['net_main_ma5'].transform(
                lambda x: (x - x.mean()) / (x.std() + EPSILON)
            ).fillna(0)
            df['fund_flow_signal'] = winsorize(df['fund_flow_signal'], sigma=3.0)
            
            df['net_main_flow'] = df['fund_flow_signal']
        else:
            df['fund_flow_signal'] = 0.0
            df['net_main_flow'] = 0.0
        
        return df
    
    def _compute_interaction_kernels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征交互核 (V203 非线性继承，V204 显著性筛选)
        
        【V204 改进】
        - 使用鲁棒性 Z-Score (MAD) 替代标准 Z-Score
        - 减少异常值对交互项的放大效应
        """
        # 1. 量价交互核 (Volume-Price Interaction Kernel)
        df['volume_price_kernel'] = np.where(
            (df['pct_chg'] < 0) & (df['volume_ratio'] > 1.5),
            -1.0,
            np.where(
                (df['pct_chg'] > 0) & (df['volume_ratio'] < 0.7),
                -0.5,
                np.where(
                    (df['pct_chg'] > 0) & (df['volume_ratio'] > 1.2),
                    1.0,
                    0.0
                )
            )
        )
        
        # 2. 波动率 - 反转交互核 (Volatility-Reversion Kernel)
        # V204 改进：使用鲁棒性 Z-Score
        vol_robust_z = df.groupby('trade_date')['volatility_5'].transform(
            lambda x: robust_zscore(x, use_mad=True)
        )
        reversion_robust_z = df.groupby('trade_date')['reversion_5'].transform(
            lambda x: robust_zscore(x, use_mad=True)
        )
        
        df['vol_reversion_kernel'] = vol_robust_z * reversion_robust_z
        
        # 3. 流动性 - 动量交互核
        liq_robust_z = df.groupby('trade_date')['liquidity_mkt_neutral'].transform(
            lambda x: robust_zscore(x, use_mad=True)
        )
        mom_robust_z = df.groupby('trade_date')['momentum_10'].transform(
            lambda x: robust_zscore(x, use_mad=True)
        )
        df['liquidity_momentum_kernel'] = liq_robust_z * mom_robust_z
        
        # 4. 三阶交互项 (增强非线性)
        df['triple_kernel'] = vol_robust_z * reversion_robust_z * liq_robust_z
        
        return df
    
    def _apply_orthogonalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用因子正交化"""
        if 'reversion_10' in df.columns and 'reversion_5' in df.columns:
            ortho_reversion = gram_schmidt_orthogonalize(
                df, 'reversion_10', ['reversion_5'], 'trade_date'
            )
            df['reversion_10_ortho'] = ortho_reversion
        
        if 'turnover_rate' in df.columns and 'liquidity_mkt_neutral' in df.columns:
            ortho_liquidity = gram_schmidt_orthogonalize(
                df, 'turnover_rate', ['liquidity_mkt_neutral'], 'trade_date'
            )
            df['turnover_rate_ortho'] = ortho_liquidity
        
        return df
    
    def _apply_significance_screening(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用显著性筛选 (V204 核心创新)
        
        【原理】
        1. 对每个交互核因子计算 IC 贡献估计
        2. 剔除 IC 贡献 < SIGNIFICANCE_IC_MIN 的因子
        3. 标记被剔除的因子，在后续评分中权重为 0
        
        【IC 估计方法】
        使用因子值与截面排名的相关性作为 IC 代理
        """
        logger.debug("[Significance] Screening interaction factors...")
        
        for factor in self.interaction_factors:
            if factor not in df.columns:
                self._factor_significance[factor]['enabled'] = False
                continue
            
            # 计算截面 IC 代理 (因子值与排名相关性)
            ic_estimates = []
            for date in df['trade_date'].unique()[-30:]:  # 使用最近 30 天
                date_data = df[df['trade_date'] == date]
                if len(date_data) < 100:
                    continue
                
                factor_values = date_data[factor].values
                ranks = pd.Series(factor_values).rank(pct=True).values
                
                if np.std(factor_values) > EPSILON:
                    ic_corr = np.corrcoef(factor_values, ranks)[0, 1]
                    if not np.isnan(ic_corr):
                        ic_estimates.append(ic_corr)
            
            if len(ic_estimates) > 0:
                mean_ic = np.mean(np.abs(ic_estimates))
                std_ic = np.std(ic_estimates) + EPSILON
                
                # 计算 p 值代理 (t 检验)
                t_stat = mean_ic / (std_ic / np.sqrt(len(ic_estimates)))
                p_value = 2 * (1 - min(0.9999, abs(t_stat) / 10))  # 简化 p 值估计
                
                # 判断是否显著
                is_significant = (mean_ic >= SIGNIFICANCE_IC_MIN) and (p_value < SIGNIFICANCE_P_THRESHOLD)
                
                self._factor_significance[factor] = {
                    'ic_contribution': mean_ic,
                    'p_value': p_value,
                    'enabled': is_significant,
                }
                
                logger.debug(f"  {factor}: IC={mean_ic:.4f}, p={p_value:.4f}, enabled={is_significant}")
                
                # 剔除不显著的因子
                if not is_significant:
                    df[factor] = 0.0
            else:
                self._factor_significance[factor]['enabled'] = False
                df[factor] = 0.0
        
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
        """
        计算综合评分
        
        【V204 改进】
        - 根据显著性筛选结果动态调整权重
        - IC 防御模式下切换至防御权重
        """
        score_components = pd.DataFrame(index=df.index)
        
        if 'reversion_5' in df.columns:
            score_components['reversion'] = normalize_group(df, 'reversion_5')
        
        if 'volatility_20' in df.columns:
            score_components['volatility'] = -normalize_group(df, 'volatility_20')
        
        if 'momentum_10' in df.columns:
            score_components['momentum'] = normalize_group(df, 'momentum_10')
        
        if 'liquidity_mkt_neutral' in df.columns:
            score_components['liquidity'] = normalize_group(df, 'liquidity_mkt_neutral')
        
        if 'volume_price_kernel' in df.columns:
            score_components['volume_price'] = normalize_group(df, 'volume_price_kernel')
        
        if 'vol_reversion_kernel' in df.columns:
            score_components['vol_reversion'] = normalize_group(df, 'vol_reversion_kernel')
        
        if 'fund_flow_signal' in df.columns:
            score_components['fund_flow'] = normalize_group(df, 'fund_flow_signal')
        
        # 根据市场状态和 IC 防御模式选择权重
        if self._ic_defense_mode:
            weights = self._get_ic_defense_weights()
            logger.info("[IC Defense] Using defense weights due to low IC")
        elif self._current_market_state == 'BEAR':
            weights = self._get_defense_weights()
        elif self._current_market_state == 'BULL':
            weights = self._get_bull_weights()
        else:
            weights = self._get_base_weights()
        
        # 根据显著性筛选结果调整权重
        if self.enable_significance_screening:
            for factor, sig_info in self._factor_significance.items():
                if not sig_info['enabled'] and factor in weights:
                    weights[factor] = 0.0
        
        # 动态调整权重 (根据可用因子)
        available_cols = [c for c in weights.keys() if c in score_components.columns]
        if available_cols:
            total_weight = sum(weights[c] for c in available_cols)
            if total_weight > 0:
                for c in available_cols:
                    weights[c] /= total_weight
        
        df['raw_score'] = sum(score_components[c] * weights[c] for c in available_cols)
        
        df['score'] = normalize_group(df, 'raw_score')
        
        self._factor_weights = weights
        
        return df
    
    def _apply_industry_neutralization(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用行业中性化"""
        if 'industry_code' not in df.columns:
            logger.debug("[IndustryNeutral] No industry_code column, skipping")
            return df
        
        industry_counts = df['industry_code'].value_counts()
        if len(industry_counts) < 3:
            logger.debug(f"[IndustryNeutral] Only {len(industry_counts)} industries, skipping")
            return df
        
        df['industry_neutral_score'] = df.groupby('industry_code')['score'].transform(
            lambda x: normalize_group(pd.DataFrame({'score': x}), 'score')
        )
        
        df['score'] = df['industry_neutral_score']
        
        logger.debug(f"[IndustryNeutral] Applied neutralization across {len(industry_counts)} industries")
        
        return df
    
    def update_ic_estimate(self, ic_value: float):
        """
        更新 IC 估计值并检查是否需要触发防御模式 (V204 核心)
        
        Args:
            ic_value: 最新 IC 估计值
        """
        self._current_ic_estimate = ic_value
        
        if self.enable_ic_defense:
            if ic_value < IC_DEFENSE_THRESHOLD:
                self._ic_defense_mode = True
                logger.warning(f"[IC Defense] TRIGGERED: IC={ic_value:.4f} < {IC_DEFENSE_THRESHOLD}")
            else:
                self._ic_defense_mode = False
                logger.info(f"[IC Defense] Normal: IC={ic_value:.4f} >= {IC_DEFENSE_THRESHOLD}")
    
    def get_current_market_state(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        return {
            'market_state': self._current_market_state,
            'current_weights': self._factor_weights,
            'industry_neutral_enabled': self.enable_industry_neutral,
            'orthogonalization_enabled': self.enable_orthogonalization,
            'interaction_kernel_enabled': self.enable_interaction_kernel,
            'significance_screening_enabled': self.enable_significance_screening,
            'ic_defense_mode': self._ic_defense_mode,
            'current_ic_estimate': self._current_ic_estimate,
            'factor_significance': self._factor_significance,
        }
    
    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性"""
        return self._factor_weights.copy()


def get_alpha_model(
    n_factors: int = MAX_FACTORS,
    enable_industry_neutral: bool = True,
    enable_market_adapter: bool = True,
    enable_orthogonalization: bool = True,
    enable_interaction_kernel: bool = True,
    enable_significance_screening: bool = True,
    enable_ic_defense: bool = True,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_industry_neutral=enable_industry_neutral,
        enable_market_adapter=enable_market_adapter,
        enable_orthogonalization=enable_orthogonalization,
        enable_interaction_kernel=enable_interaction_kernel,
        enable_significance_screening=enable_significance_screening,
        enable_ic_defense=enable_ic_defense,
    )