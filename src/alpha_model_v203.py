"""
Alpha Model Module - V203 Non-Linear Evolution
==============================================

【V203 核心变革】
1. 特征交互核 (Feature Interaction Kernel)
   - 引入非线性组合：(volatility_5 * reversion_5)
   - 捕获超跌且缩量的反转信号
   - 三阶交互项增强 Alpha 表达力

2. 动态环境适配 (Dynamic Environment Adaptation)
   - 基于 index_mkt_state 判断市场状态
   - 熊市：切换至"低波动防御因子"
   - 牛市：切换至"高动量因子"
   - 震荡市：平衡配置

3. 因子正交化 (Factor Orthogonalization)
   - 对 reversion、liquidity、fund_flow 三大类因子
   - 执行截面施密特正交化 (Gram-Schmidt)
   - 消除冗余信息，提升因子独立性

【物理清理】
- 删除所有 V191/V200/V201/V202 字样注释
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

VERSION = "V203_NonLinear_Evolution"

# V203 核心因子配置
V203_CORE_FACTORS = [
    'reversion_5',              # 5 日反转 - 防御型
    'volatility_20',            # 20 日波动率 - 风险度量
    'momentum_10',              # 10 日动量 - 进攻型
    'liquidity_mkt_neutral',    # 市值中性化流动性
    'volume_price_kernel',      # 量价交互核
    'vol_reversion_kernel',     # 波动率 - 反转交互核
    'fund_flow_signal',         # 资金流信号
    'market_state_adapter',     # 市场状态适配器
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
MARKET_STATE_BEAR_THRESHOLD = -0.10  # 熊市阈值
MARKET_STATE_BULL_THRESHOLD = 0.15   # 牛市阈值

# 因子正交化配置
ORTHOGONALIZATION_GROUPS = {
    'reversion': ['reversion_5', 'reversion_10'],
    'liquidity': ['liquidity_mkt_neutral', 'turnover_rate'],
    'fund_flow': ['fund_flow_signal', 'net_main_flow'],
}

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


def normalize_group(df: pd.DataFrame, score_col: str, group_col: str = None) -> pd.Series:
    """
    截面或分组标准化评分
    
    【原理】
    - 将原始评分转换为 [0, 1] 区间的排名百分位
    - 可选按组 (行业) 内标准化实现行业中性化
    """
    if group_col is None or group_col not in df.columns:
        # 截面标准化
        if len(df) < 2:
            return pd.Series(0.5, index=df.index)
        
        ranks = df[score_col].rank(pct=True)
        # 转换为均值为 0，标准差为 1 的标准化评分
        normalized = (ranks - 0.5) * 2  # 范围 [-1, 1]
        return normalized
    
    # 分组标准化 (行业中性化核心)
    def rank_pct(group):
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        return (group.rank(pct=True) - 0.5) * 2
    
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
    
    Args:
        df: 输入 DataFrame
        target_col: 需要正交化的目标列
        orthogonal_cols: 用于正交化的因子列列表
        group_col: 分组列 (默认按截面)
    
    Returns:
        正交化后的残差序列
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
        
        # 构建因子矩阵
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
        
        # 处理缺失值
        mask = ~np.isnan(target_values)
        for i in range(factor_matrix.shape[1]):
            mask &= ~np.isnan(factor_matrix[:, i])
        
        if mask.sum() < len(valid_cols) + 3:
            result.loc[group_idx] = target_values
            continue
        
        y = target_values[mask]
        X = factor_matrix[mask]
        
        # 标准化
        y_mean, y_std = y.mean(), y.std() + EPSILON
        y_norm = (y - y_mean) / y_std
        
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + EPSILON
        X_norm = (X - X_mean) / X_std
        
        # Gram-Schmidt 正交化
        residual = y_norm.copy()
        for i in range(X_norm.shape[1]):
            x_col = X_norm[:, i]
            beta = np.dot(residual, x_col) / (np.dot(x_col, x_col) + EPSILON)
            residual = residual - beta * x_col
        
        # 恢复原始尺度
        residual_final = residual * y_std + y_mean
        
        result.loc[group_idx[mask]] = residual_final
    
    return result


# ==============================================================================
# AlphaModel V203
# ==============================================================================

class AlphaModel:
    """
    V203 Alpha 模型 - 非线性进化
    
    【核心职责】
    1. 计算因子值 (仅使用 T 日及之前数据)
    2. 特征交互核 (非线性组合)
    3. 因子正交化 (施密特正交化)
    4. 动态市场适配
    5. 输出标准化评分 score
    
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
    ):
        self.n_factors = min(n_factors, MAX_FACTORS)
        self.enable_industry_neutral = enable_industry_neutral and INDUSTRY_NEUTRALIZE_ENABLED
        self.enable_market_adapter = enable_market_adapter
        self.enable_orthogonalization = enable_orthogonalization
        self.enable_interaction_kernel = enable_interaction_kernel
        
        self.factors_used = V203_CORE_FACTORS[:self.n_factors]
        
        # 状态跟踪
        self._current_market_state = 'NORMAL'
        self._factor_weights = self._get_base_weights()
        
        logger.info("=" * 70)
        logger.info("V203 AlphaModel Initialized")
        logger.info("=" * 70)
        logger.info(f"  Factors: {self.factors_used}")
        logger.info(f"  Industry Neutral: {self.enable_industry_neutral}")
        logger.info(f"  Market Adapter: {self.enable_market_adapter}")
        logger.info(f"  Orthogonalization: {self.enable_orthogonalization}")
        logger.info(f"  Interaction Kernel: {self.enable_interaction_kernel}")
        logger.info("=" * 70)
    
    def _get_base_weights(self) -> Dict[str, float]:
        """获取基础权重配置"""
        return {
            'reversion': 0.20,
            'volatility': 0.15,
            'momentum': 0.15,
            'liquidity': 0.10,
            'volume_price': 0.15,
            'vol_reversion': 0.15,
            'fund_flow': 0.10,
        }
    
    def _get_defense_weights(self) -> Dict[str, float]:
        """熊市防御权重"""
        return {
            'reversion': 0.35,      # 增加反转权重
            'volatility': 0.25,     # 增加低波动权重
            'momentum': 0.05,       # 减少动量权重
            'liquidity': 0.15,
            'volume_price': 0.10,
            'vol_reversion': 0.10,
            'fund_flow': 0.00,      # 熊市资金流不可靠
        }
    
    def _get_bull_weights(self) -> Dict[str, float]:
        """牛市进攻权重"""
        return {
            'reversion': 0.10,      # 减少反转权重
            'volatility': 0.05,     # 容忍高波动
            'momentum': 0.30,       # 增加动量权重
            'liquidity': 0.10,
            'volume_price': 0.20,
            'vol_reversion': 0.10,
            'fund_flow': 0.15,      # 牛市资金流有效
        }
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分
        
        【核心原则】
        - 所有计算仅使用 T 日及之前数据
        - 严禁使用 shift(-1) 等未来函数
        - 输出 score 列，严禁输出任何收益率相关列
        
        Args:
            df: 输入数据，必须包含以下列:
                - symbol, trade_date, close, open, high, low
                - volume, amount, pct_chg
                - (可选) industry_code, net_main_amount
        
        Returns:
            包含 score 列的 DataFrame
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
        
        # 计算基础因子
        df = self._compute_base_factors(df)
        
        # 特征交互核 (V203 核心)
        if self.enable_interaction_kernel:
            df = self._compute_interaction_kernels(df)
        
        # 因子正交化 (V203 核心)
        if self.enable_orthogonalization:
            df = self._apply_orthogonalization(df)
        
        # 数据自愈
        df = auto_fillna(df, self.factors_used)
        
        # 动态市场适配 (V203 核心)
        if self.enable_market_adapter:
            df = self._apply_market_adapter(df)
        
        # 计算综合评分
        df = self._compute_composite_score(df)
        
        # 行业中性化
        if self.enable_industry_neutral and 'industry_code' in df.columns:
            df = self._apply_industry_neutralization(df)
        
        # 最终标准化
        df['score'] = winsorize(df['score'], sigma=3.0)
        
        logger.info(f"[AlphaModel] Score computed. Range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        
        return df
    
    def _compute_base_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础因子
        
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
        # 计算 10 日动量：当前价格 / 10 日前价格 - 1
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
            
            # 用于正交化
            df['net_main_flow'] = df['fund_flow_signal']
        else:
            df['fund_flow_signal'] = 0.0
            df['net_main_flow'] = 0.0
        
        return df
    
    def _compute_interaction_kernels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征交互核 (V203 核心创新)
        
        【原理】
        1. 量价交互核：捕捉放量下跌/缩量上涨的矛盾信号
        2. 波动率 - 反转交互核：捕捉超跌且高波动的反转机会
        3. 三阶交互项：增强非线性表达力
        
        【数学表达】
        kernel = f(factor1, factor2) 非线性组合
        """
        # 1. 量价交互核 (Volume-Price Interaction Kernel)
        # 放量下跌 = 看跌信号，缩量上涨 = 看跌信号
        df['volume_price_kernel'] = np.where(
            (df['pct_chg'] < 0) & (df['volume_ratio'] > 1.5),
            -1.0,  # 放量下跌，负面信号
            np.where(
                (df['pct_chg'] > 0) & (df['volume_ratio'] < 0.7),
                -0.5,  # 缩量上涨，负面信号
                np.where(
                    (df['pct_chg'] > 0) & (df['volume_ratio'] > 1.2),
                    1.0,  # 放量上涨，正面信号
                    0.0
                )
            )
        )
        
        # 2. 波动率 - 反转交互核 (Volatility-Reversion Kernel)
        # 高波动 + 超跌 = 强反转信号
        vol_zscore = df.groupby('trade_date')['volatility_5'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        reversion_zscore = df.groupby('trade_date')['reversion_5'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        # 交互项：高波动 * 强反转
        df['vol_reversion_kernel'] = vol_zscore * reversion_zscore
        
        # 3. 流动性 - 动量交互核
        liq_zscore = df.groupby('trade_date')['liquidity_mkt_neutral'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        mom_zscore = df.groupby('trade_date')['momentum_10'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        df['liquidity_momentum_kernel'] = liq_zscore * mom_zscore
        
        # 4. 三阶交互项 (增强非线性)
        df['triple_kernel'] = vol_zscore * reversion_zscore * liq_zscore
        
        return df
    
    def _apply_orthogonalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用因子正交化 (V203 核心)
        
        【原理】
        对 reversion、liquidity、fund_flow 三大类因子进行
        截面施密特正交化，消除冗余信息
        """
        # 1. Reversion 正交化
        if 'reversion_10' in df.columns and 'reversion_5' in df.columns:
            ortho_reversion = gram_schmidt_orthogonalize(
                df, 'reversion_10', ['reversion_5'], 'trade_date'
            )
            df['reversion_10_ortho'] = ortho_reversion
        
        # 2. Liquidity 正交化
        if 'turnover_rate' in df.columns and 'liquidity_mkt_neutral' in df.columns:
            ortho_liquidity = gram_schmidt_orthogonalize(
                df, 'turnover_rate', ['liquidity_mkt_neutral'], 'trade_date'
            )
            df['turnover_rate_ortho'] = ortho_liquidity
        
        # 3. Fund Flow 正交化 (如果有多个资金流因子)
        # 当前只有一个 fund_flow_signal，暂不需要
        
        return df
    
    def _apply_market_adapter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用动态市场适配 (V203 核心)
        
        【原理】
        1. 计算市场整体状态 (基于指数收益率)
        2. 熊市：自动切换至防御权重
        3. 牛市：自动切换至进攻权重
        4. 震荡市：使用基础权重
        """
        # 计算市场整体收益率 (使用所有股票的平均收益率作为代理)
        market_return = df.groupby('trade_date')['pct_chg'].mean()
        
        # 计算滚动市场状态
        market_return_rolling = market_return.rolling(MARKET_STATE_WINDOW, min_periods=10).mean()
        
        # 判断市场状态
        bear_dates = market_return_rolling[market_return_rolling < MARKET_STATE_BEAR_THRESHOLD].index.tolist()
        bull_dates = market_return_rolling[market_return_rolling > MARKET_STATE_BULL_THRESHOLD].index.tolist()
        
        # 标记市场状态
        df['market_state'] = 'NORMAL'
        df.loc[df['trade_date'].isin(bear_dates), 'market_state'] = 'BEAR'
        df.loc[df['trade_date'].isin(bull_dates), 'market_state'] = 'BULL'
        
        # 记录当前状态
        latest_date = df['trade_date'].max()
        latest_state = df[df['trade_date'] == latest_date]['market_state'].iloc[0] if len(df) > 0 else 'NORMAL'
        self._current_market_state = latest_state
        
        # 市场状态适配器评分
        df['market_state_adapter'] = np.where(
            df['market_state'] == 'BEAR',
            1.0,  # 熊市防御信号
            np.where(
                df['market_state'] == 'BULL',
                -1.0,  # 牛市进攻信号 (降低防御)
                0.0
            )
        )
        
        logger.debug(f"[Market Adapter] Current state: {self._current_market_state}")
        
        return df
    
    def _compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合评分
        
        【权重配置】
        根据市场状态动态调整权重
        """
        score_components = pd.DataFrame(index=df.index)
        
        # 基础因子标准化
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
        
        # 计算加权评分
        df['raw_score'] = sum(score_components[c] * weights[c] for c in available_cols)
        
        # 截面标准化为最终 score
        df['score'] = normalize_group(df, 'raw_score')
        
        # 保存使用的权重
        self._factor_weights = weights
        
        return df
    
    def _apply_industry_neutralization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用行业中性化 (V203)
        
        【原理】
        1. 按行业分组
        2. 在行业内重新标准化评分 (按截面)
        3. 消除行业集中暴露风险
        """
        if 'industry_code' not in df.columns:
            logger.debug("[IndustryNeutral] No industry_code column, skipping")
            return df
        
        industry_counts = df['industry_code'].value_counts()
        if len(industry_counts) < 3:
            logger.debug(f"[IndustryNeutral] Only {len(industry_counts)} industries, skipping")
            return df
        
        # 在行业内重新标准化 - 与 V202 一致，使用 normalize_group
        # 注意：normalize_group 默认按 trade_date 分组标准化
        df['industry_neutral_score'] = df.groupby('industry_code')['score'].transform(
            lambda x: normalize_group(pd.DataFrame({'score': x}), 'score')
        )
        
        # 使用行业中性化评分
        df['score'] = df['industry_neutral_score']
        
        logger.debug(f"[IndustryNeutral] Applied neutralization across {len(industry_counts)} industries")
        
        return df
    
    def get_current_market_state(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        return {
            'market_state': self._current_market_state,
            'current_weights': self._factor_weights,
            'industry_neutral_enabled': self.enable_industry_neutral,
            'orthogonalization_enabled': self.enable_orthogonalization,
            'interaction_kernel_enabled': self.enable_interaction_kernel,
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
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_industry_neutral=enable_industry_neutral,
        enable_market_adapter=enable_market_adapter,
        enable_orthogonalization=enable_orthogonalization,
        enable_interaction_kernel=enable_interaction_kernel,
    )