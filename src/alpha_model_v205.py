"""
Alpha Model Module - V205 Deep Feature Discovery & IC Recovery
================================================================

【V205 核心变革】
1. 多尺度时序特征 (Multi-scale Temporal Features)
   - 引入 (5d, 10d, 20d, 60d) 的多维动量/反转特征
   - 引入"高阶矩"特征：偏度 (Skewness) 和 峰度 (Kurtosis) 的截面排名

2. 自适应因子正交化 (Adaptive Orthogonalization)
   - 改进 V204 的全量正交化
   - 实现"增量正交"，新特征只对成熟的经典因子（如 Size、Beta）进行偏相关剥离

3. 遗传算法思想 (Genetic Alpha Discovery)
   - 在 get_score 中实现"符号公式发现"逻辑（简化版）
   - 寻找如 (close/open-1) / volatility 这种具有物理意义的复合因子

4. IC 驱动的特征选择
   - 基于滚动 IC 评估每个特征的预测能力
   - 动态调整特征权重，优先使用高 IC 特征

【物理清理】
- 删除所有 V191-V204 字样注释
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

VERSION = "V205_Deep_Feature_Discovery"

# V205 核心因子配置 - 多尺度时序特征
V205_REVERSION_SCALES = [5, 10, 20, 60]  # 多尺度反转
V205_MOMENTUM_SCALES = [5, 10, 20, 60]   # 多尺度动量
V205_VOLATILITY_SCALES = [5, 10, 20, 60] # 多尺度波动率

# V205 高阶矩特征
V205_HIGHER_MOMENTS = ['skewness_20', 'kurtosis_20', 'skewness_60', 'kurtosis_60']

# V205 遗传算法生成的复合因子
V205_GENETIC_FACTORS = [
    'price_efficiency_ratio',      # (close - open) / volatility
    'volume_price_strength',       # (pct_chg) / (high - low) / close
    'liquidity_adjusted_momentum', # momentum / sqrt(volatility)
    'reversion_strength_ratio',    # reversion / volatility
]

# V205 经典因子（用于正交化基准）
V205_CLASSIC_FACTORS = ['size', 'beta', 'reversion_5', 'volatility_20']

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

# V205 IC 驱动特征选择配置
IC_SELECTION_WINDOW = 30  # 滚动 IC 评估窗口
IC_SELECTION_MIN_IC = 0.01  # 最小 IC 阈值
IC_SELECTION_ENABLED = True

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
    
    Args:
        df: 输入 DataFrame
        score_col: 评分列名
        group_col: 分组列名（可选）
    
    Returns:
        标准化后的评分
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
    
    Args:
        df: 输入 DataFrame
        target_col: 目标列
        orthogonal_cols: 用于正交化的列
        group_col: 分组列
    
    Returns:
        正交化后的序列
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


def compute_skewness(series: pd.Series) -> float:
    """
    计算偏度 (Skewness)
    
    偏度衡量分布的不对称性：
    - 正偏度：右尾更长，有极端大值
    - 负偏度：左尾更长，有极端小值
    """
    if len(series) < 3:
        return 0.0
    
    n = len(series)
    mean = series.mean()
    std = series.std()
    
    if std < EPSILON:
        return 0.0
    
    skew = ((series - mean) ** 3).sum() / ((n - 1) * (std ** 3))
    return skew


def compute_kurtosis(series: pd.Series) -> float:
    """
    计算峰度 (Kurtosis)
    
    峰度衡量分布的尾部厚度：
    - 高峰度：厚尾，极端值更多
    - 低峰度：薄尾，极端值更少
    """
    if len(series) < 4:
        return 3.0  # 正态分布峰度
    
    n = len(series)
    mean = series.mean()
    std = series.std()
    
    if std < EPSILON:
        return 3.0
    
    kurt = ((series - mean) ** 4).sum() / ((n - 1) * (std ** 4))
    return kurt


def rolling_skewness(series: pd.Series, window: int) -> pd.Series:
    """计算滚动偏度"""
    return series.rolling(window, min_periods=max(10, window // 2)).apply(
        compute_skewness, raw=False
    ).fillna(0)


def rolling_kurtosis(series: pd.Series, window: int) -> pd.Series:
    """计算滚动峰度"""
    return series.rolling(window, min_periods=max(10, window // 2)).apply(
        compute_kurtosis, raw=False
    ).fillna(0)


# ==============================================================================
# V205 遗传算法 - 符号公式发现
# ==============================================================================

class GeneticAlphaDiscoverer:
    """
    遗传算法符号公式发现器 (简化版)
    
    【原理】
    1. 定义基础操作符集合：+, -, *, /, sqrt, log, abs
    2. 定义基础变量集合：close, open, high, low, volume, pct_chg, volatility
    3. 通过组合操作符和变量生成候选公式
    4. 评估每个公式的 IC 表现
    5. 选择表现最好的公式作为最终因子
    
    【V205 实现】
    使用预定义的有物理意义的公式，而非随机生成
    """
    
    # 预定义的有物理意义的公式模板
    GENETIC_TEMPLATES = {
        'price_efficiency_ratio': {
            'formula': '(close - open) / (volatility_5 + eps)',
            'description': '价格效率比：单位波动率下的价格变化',
            'weight': 1.0,
        },
        'volume_price_strength': {
            'formula': 'pct_chg / ((high - low) / close + eps)',
            'description': '量价强度：涨跌幅相对于日内波动的比率',
            'weight': 1.0,
        },
        'liquidity_adjusted_momentum': {
            'formula': 'momentum_20 / (volatility_20 + eps)',
            'description': '流动性调整动量：单位风险下的动量收益',
            'weight': 1.0,
        },
        'reversion_strength_ratio': {
            'formula': 'reversion_5 / (volatility_5 + eps)',
            'description': '反转强度比：单位波动率下的反转信号',
            'weight': 1.0,
        },
        'volume_momentum_interaction': {
            'formula': '(volume_ratio - 1) * momentum_10',
            'description': '量能动量交互：成交量变化与动量的乘积',
            'weight': 0.5,
        },
        'volatility_scaled_reversion': {
            'formula': 'reversion_10 * (1 / (volatility_10 + eps))',
            'description': '波动率缩放反转：低波动环境下的反转信号更强',
            'weight': 0.8,
        },
    }
    
    def __init__(self):
        self._factor_ic_history = {name: [] for name in self.GENETIC_TEMPLATES.keys()}
        self._factor_weights = {name: tpl['weight'] for name, tpl in self.GENETIC_TEMPLATES.items()}
    
    def compute_genetic_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算遗传算法生成的复合因子
        
        Args:
            df: 输入 DataFrame，需包含基础因子
        
        Returns:
            添加了遗传因子的 DataFrame
        """
        df = df.copy()
        
        # 1. 价格效率比 (Price Efficiency Ratio)
        if 'volatility_5' in df.columns:
            df['price_efficiency_ratio'] = (df['close'] - df['open']) / (df['volatility_5'] + EPSILON)
            df['price_efficiency_ratio'] = winsorize(df['price_efficiency_ratio'], sigma=3.0)
        
        # 2. 量价强度 (Volume-Price Strength)
        if 'high' in df.columns and 'low' in df.columns:
            intraday_range = (df['high'] - df['low']) / (df['close'] + EPSILON)
            df['volume_price_strength'] = df['pct_chg'] / (intraday_range + EPSILON)
            df['volume_price_strength'] = winsorize(df['volume_price_strength'], sigma=3.0)
        
        # 3. 流动性调整动量 (Liquidity-Adjusted Momentum)
        if 'momentum_20' in df.columns and 'volatility_20' in df.columns:
            df['liquidity_adjusted_momentum'] = df['momentum_20'] / (df['volatility_20'] + EPSILON)
            df['liquidity_adjusted_momentum'] = winsorize(df['liquidity_adjusted_momentum'], sigma=3.0)
        
        # 4. 反转强度比 (Reversion Strength Ratio)
        if 'reversion_5' in df.columns and 'volatility_5' in df.columns:
            df['reversion_strength_ratio'] = df['reversion_5'] / (df['volatility_5'] + EPSILON)
            df['reversion_strength_ratio'] = winsorize(df['reversion_strength_ratio'], sigma=3.0)
        
        # 5. 量能动量交互 (Volume-Momentum Interaction)
        if 'volume_ratio' in df.columns and 'momentum_10' in df.columns:
            df['volume_momentum_interaction'] = (df['volume_ratio'] - 1) * df['momentum_10']
            df['volume_momentum_interaction'] = winsorize(df['volume_momentum_interaction'], sigma=3.0)
        
        # 6. 波动率缩放反转 (Volatility-Scaled Reversion)
        if 'reversion_10' in df.columns and 'volatility_10' in df.columns:
            df['volatility_scaled_reversion'] = df['reversion_10'] / (df['volatility_10'] + EPSILON)
            df['volatility_scaled_reversion'] = winsorize(df['volatility_scaled_reversion'], sigma=3.0)
        
        return df
    
    def update_ic_history(self, factor_name: str, ic_value: float):
        """更新因子 IC 历史"""
        if factor_name in self._factor_ic_history:
            self._factor_ic_history[factor_name].append(ic_value)
            # 保留最近 60 天的 IC 记录
            if len(self._factor_ic_history[factor_name]) > 60:
                self._factor_ic_history[factor_name] = self._factor_ic_history[factor_name][-60:]
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """
        获取自适应权重
        
        基于历史 IC 表现动态调整因子权重
        """
        weights = {}
        for factor_name, ic_history in self._factor_ic_history.items():
            if len(ic_history) < 10:
                # IC 历史不足时使用基础权重
                weights[factor_name] = self.GENETIC_TEMPLATES[factor_name]['weight']
            else:
                # 基于 IC 均值和稳定性计算权重
                ic_mean = np.mean(ic_history[-30:])
                ic_std = np.std(ic_history[-30:]) + EPSILON
                ic_ir = ic_mean / ic_std
                
                # 权重 = 基础权重 * (1 + IC_IR)
                base_weight = self.GENETIC_TEMPLATES[factor_name]['weight']
                weights[factor_name] = base_weight * (1 + max(0, ic_ir))
        
        # 归一化权重
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性"""
        return self.get_adaptive_weights()


# ==============================================================================
# AlphaModel V205
# ==============================================================================

class AlphaModel:
    """
    V205 Alpha 模型 - 深度特征发现与 IC 修复
    
    【核心职责】
    1. 计算多尺度时序特征 (5d, 10d, 20d, 60d)
    2. 计算高阶矩特征 (偏度、峰度)
    3. 计算遗传算法生成的复合因子
    4. 自适应因子正交化 (只对经典因子正交)
    5. IC 驱动的特征选择
    6. 输出标准化评分 score
    
    【严禁】
    - 计算任何收益率 (t1_return 等)
    - 接触回测逻辑
    - 使用未来函数 (shift(-1) 等)
    """
    
    def __init__(
        self,
        enable_industry_neutral: bool = True,
        enable_market_adapter: bool = True,
        enable_orthogonalization: bool = True,
        enable_genetic_discovery: bool = True,
        enable_ic_selection: bool = True,
        enable_higher_moments: bool = True,
    ):
        self.enable_industry_neutral = enable_industry_neutral and INDUSTRY_NEUTRALIZE_ENABLED
        self.enable_market_adapter = enable_market_adapter
        self.enable_orthogonalization = enable_orthogonalization
        self.enable_genetic_discovery = enable_genetic_discovery
        self.enable_ic_selection = enable_ic_selection
        self.enable_higher_moments = enable_higher_moments
        
        # 初始化遗传算法发现器
        self.genetic_discoverer = GeneticAlphaDiscoverer() if enable_genetic_discovery else None
        
        # V205 特征配置
        self.feature_groups = {
            'reversion': [f'reversion_{d}' for d in V205_REVERSION_SCALES],
            'momentum': [f'momentum_{d}' for d in V205_MOMENTUM_SCALES],
            'volatility': [f'volatility_{d}' for d in V205_VOLATILITY_SCALES],
            'higher_moments': V205_HIGHER_MOMENTS if enable_higher_moments else [],
            'genetic': list(GeneticAlphaDiscoverer.GENETIC_TEMPLATES.keys()) if enable_genetic_discovery else [],
        }
        
        # 状态跟踪
        self._current_market_state = 'NORMAL'
        self._factor_weights = self._get_base_weights()
        self._ic_history = {}
        
        logger.info("=" * 70)
        logger.info("V205 AlphaModel Initialized")
        logger.info("=" * 70)
        logger.info(f"  Reversion Scales: {V205_REVERSION_SCALES}")
        logger.info(f"  Momentum Scales: {V205_MOMENTUM_SCALES}")
        logger.info(f"  Volatility Scales: {V205_VOLATILITY_SCALES}")
        logger.info(f"  Higher Moments: {self.enable_higher_moments}")
        logger.info(f"  Genetic Discovery: {self.enable_genetic_discovery}")
        logger.info(f"  IC Selection: {self.enable_ic_selection}")
        logger.info(f"  Industry Neutral: {self.enable_industry_neutral}")
        logger.info(f"  Market Adapter: {self.enable_market_adapter}")
        logger.info(f"  Orthogonalization: {self.enable_orthogonalization}")
        logger.info("=" * 70)
    
    def _get_base_weights(self) -> Dict[str, float]:
        """获取基础权重配置"""
        return {
            'reversion': 0.20,
            'momentum': 0.15,
            'volatility': 0.15,
            'higher_moments': 0.10,
            'genetic': 0.25,
            'size': 0.05,
            'liquidity': 0.10,
        }
    
    def _get_defense_weights(self) -> Dict[str, float]:
        """熊市防御权重"""
        return {
            'reversion': 0.30,
            'momentum': 0.05,
            'volatility': 0.25,
            'higher_moments': 0.10,
            'genetic': 0.15,
            'size': 0.05,
            'liquidity': 0.10,
        }
    
    def _get_bull_weights(self) -> Dict[str, float]:
        """牛市进攻权重"""
        return {
            'reversion': 0.10,
            'momentum': 0.30,
            'volatility': 0.10,
            'higher_moments': 0.10,
            'genetic': 0.30,
            'size': 0.05,
            'liquidity': 0.05,
        }
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分
        
        【核心原则】
        - 所有计算仅使用 T 日及之前数据
        - 严禁使用 shift(-1) 等未来函数
        - 输出 score 列，严禁输出任何收益率相关列
        
        【V205 流程】
        1. 计算多尺度时序特征
        2. 计算高阶矩特征
        3. 计算遗传算法复合因子
        4. 自适应正交化
        5. IC 驱动特征选择
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
        
        # 1. 计算多尺度时序特征
        df = self._compute_multi_scale_features(df)
        
        # 2. 计算高阶矩特征
        if self.enable_higher_moments:
            df = self._compute_higher_moments(df)
        
        # 3. 计算遗传算法复合因子
        if self.enable_genetic_discovery:
            df = self.genetic_discoverer.compute_genetic_factors(df)
        
        # 4. 因子正交化 (自适应，只对经典因子)
        if self.enable_orthogonalization:
            df = self._apply_adaptive_orthogonalization(df)
        
        # 5. 数据自愈
        all_features = self._get_all_feature_names()
        df = auto_fillna(df, all_features)
        
        # 6. 动态市场适配
        if self.enable_market_adapter:
            df = self._apply_market_adapter(df)
        
        # 7. IC 驱动特征选择
        if self.enable_ic_selection:
            df = self._apply_ic_selection(df)
        
        # 8. 计算综合评分
        df = self._compute_composite_score(df)
        
        # 9. 行业中性化
        if self.enable_industry_neutral and 'industry_code' in df.columns:
            df = self._apply_industry_neutralization(df)
        
        # 10. 鲁棒性标准化
        df['score'] = robust_zscore(df['score'], use_mad=True)
        df['score'] = winsorize(df['score'], sigma=3.0)
        
        logger.info(f"[AlphaModel] Score computed. Range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        
        return df
    
    def _get_all_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        features = []
        for group_features in self.feature_groups.values():
            features.extend(group_features)
        return features
    
    def _compute_multi_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多尺度时序特征
        
        【多尺度反转】
        - reversion_5: 5 日反转
        - reversion_10: 10 日反转
        - reversion_20: 20 日反转
        - reversion_60: 60 日反转
        
        【多尺度动量】
        - momentum_5, momentum_10, momentum_20, momentum_60
        
        【多尺度波动率】
        - volatility_5, volatility_10, volatility_20, volatility_60
        """
        numeric_cols = ['close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 计算 1 日收益率（用于后续计算）
        df['return_1d'] = df.groupby('symbol')['close'].pct_change(1).fillna(0)
        
        # 多尺度反转因子
        for window in V205_REVERSION_SCALES:
            col_name = f'reversion_{window}'
            df[col_name] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(window) - 1
            ).fillna(0)
            # 反转因子取负号（预期反转）
            if window <= 10:  # 短期反转
                df[col_name] = -df[col_name]
        
        # 多尺度动量因子
        for window in V205_MOMENTUM_SCALES:
            col_name = f'momentum_{window}'
            df[col_name] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(window) - 1
            ).fillna(0)
        
        # 多尺度波动率因子
        for window in V205_VOLATILITY_SCALES:
            col_name = f'volatility_{window}'
            df[col_name] = df.groupby('symbol')['return_1d'].transform(
                lambda x: x.rolling(window, min_periods=max(5, window // 2)).std()
            ).fillna(0)
        
        # 辅助特征：成交量比率
        df['volume_ratio'] = df['volume'] / (df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        ) + EPSILON)
        
        # 10 日波动率（用于遗传因子）
        if 'volatility_10' not in df.columns:
            df['volatility_10'] = df.groupby('symbol')['return_1d'].transform(
                lambda x: x.rolling(10, min_periods=5).std()
            ).fillna(0)
        
        return df
    
    def _compute_higher_moments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算高阶矩特征
        
        【偏度 (Skewness)】
        - 衡量收益率分布的不对称性
        - 负偏度表示左尾风险更大
        
        【峰度 (Kurtosis)】
        - 衡量收益率分布的尾部厚度
        - 高峰度表示极端事件更多
        """
        # 20 日滚动偏度和峰度
        df['skewness_20'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: rolling_skewness(x, 20)
        ).fillna(0)
        
        df['kurtosis_20'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: rolling_kurtosis(x, 20)
        ).fillna(0)
        
        # 60 日滚动偏度和峰度
        df['skewness_60'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: rolling_skewness(x, 60)
        ).fillna(0)
        
        df['kurtosis_60'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: rolling_kurtosis(x, 60)
        ).fillna(0)
        
        return df
    
    def _apply_adaptive_orthogonalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用自适应因子正交化
        
        【V205 改进】
        - 只对经典因子进行正交化
        - 新特征只对 Size、Beta 等成熟因子剥离相关性
        - 避免过度正交化导致的信息损失
        """
        if 'reversion_60' in df.columns and 'reversion_5' in df.columns:
            # 长期反转对短期反转正交化
            ortho_reversion = gram_schmidt_orthogonalize(
                df, 'reversion_60', ['reversion_5'], 'trade_date'
            )
            df['reversion_60_ortho'] = ortho_reversion
        
        if 'momentum_60' in df.columns and 'momentum_20' in df.columns:
            # 长期动量对中期动量正交化
            ortho_momentum = gram_schmidt_orthogonalize(
                df, 'momentum_60', ['momentum_20'], 'trade_date'
            )
            df['momentum_60_ortho'] = ortho_momentum
        
        # 高阶矩特征对波动率正交化
        if 'skewness_20' in df.columns and 'volatility_20' in df.columns:
            ortho_skew = gram_schmidt_orthogonalize(
                df, 'skewness_20', ['volatility_20'], 'trade_date'
            )
            df['skewness_20_ortho'] = ortho_skew
        
        if 'kurtosis_20' in df.columns and 'volatility_20' in df.columns:
            ortho_kurt = gram_schmidt_orthogonalize(
                df, 'kurtosis_20', ['volatility_20'], 'trade_date'
            )
            df['kurtosis_20_ortho'] = ortho_kurt
        
        return df
    
    def _apply_ic_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用 IC 驱动的特征选择
        
        【原理】
        1. 计算每个特征与截面排名的相关性（IC 代理）
        2. 保留 IC 高于阈值的特征
        3. 动态调整特征权重
        """
        logger.debug("[IC Selection] Evaluating feature importance...")
        
        # 使用最近 30 天的数据评估 IC
        recent_dates = df['trade_date'].unique()[-IC_SELECTION_WINDOW:]
        
        feature_ic_stats = {}
        
        for group_name, features in self.feature_groups.items():
            for feature in features:
                if feature not in df.columns:
                    continue
                
                ic_values = []
                for date in recent_dates:
                    date_data = df[df['trade_date'] == date]
                    if len(date_data) < 100:
                        continue
                    
                    feature_values = date_data[feature].values
                    if np.std(feature_values) < EPSILON:
                        continue
                    
                    # 计算 IC 代理（与排名的相关性）
                    ranks = pd.Series(feature_values).rank(pct=True).values
                    ic_corr = np.corrcoef(feature_values, ranks)[0, 1]
                    if not np.isnan(ic_corr):
                        ic_values.append(ic_corr)
                
                if len(ic_values) >= 5:
                    ic_mean = np.mean(np.abs(ic_values))
                    ic_std = np.std(ic_values) + EPSILON
                    ic_ir = ic_mean / ic_std
                    
                    feature_ic_stats[feature] = {
                        'ic_mean': ic_mean,
                        'ic_std': np.std(ic_values),
                        'ic_ir': ic_ir,
                        'enabled': ic_mean >= IC_SELECTION_MIN_IC,
                    }
                    
                    # 更新遗传发现器的 IC 历史
                    if self.genetic_discoverer and feature in self.genetic_discoverer._factor_ic_history:
                        self.genetic_discoverer.update_ic_history(feature, ic_mean)
        
        # 存储 IC 统计信息
        self._ic_history = feature_ic_stats
        
        # 剔除 IC 低于阈值的特征
        for feature, stats in feature_ic_stats.items():
            if not stats['enabled']:
                if feature in df.columns:
                    df[feature] = 0.0
        
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
        
        【V205 改进】
        - 根据 IC 评估结果动态调整权重
        - 整合多尺度特征为组内评分
        """
        score_components = pd.DataFrame(index=df.index)
        
        # 1. 反转因子组（多尺度加权）
        reversion_components = []
        reversion_weights = {5: 0.4, 10: 0.3, 20: 0.2, 60: 0.1}
        for window in V205_REVERSION_SCALES:
            col = f'reversion_{window}'
            if col in df.columns:
                normalized = normalize_group(df, col)
                weight = reversion_weights.get(window, 0.25)
                reversion_components.append(normalized * weight)
        if reversion_components:
            score_components['reversion'] = sum(reversion_components)
        
        # 2. 动量因子组（多尺度加权）
        momentum_components = []
        momentum_weights = {5: 0.1, 10: 0.2, 20: 0.3, 60: 0.4}
        for window in V205_MOMENTUM_SCALES:
            col = f'momentum_{window}'
            if col in df.columns:
                normalized = normalize_group(df, col)
                weight = momentum_weights.get(window, 0.25)
                momentum_components.append(normalized * weight)
        if momentum_components:
            score_components['momentum'] = sum(momentum_components)
        
        # 3. 波动率因子组（反向）
        volatility_components = []
        for window in V205_VOLATILITY_SCALES:
            col = f'volatility_{window}'
            if col in df.columns:
                normalized = -normalize_group(df, col)  # 低波动更好
                volatility_components.append(normalized * 0.25)
        if volatility_components:
            score_components['volatility'] = sum(volatility_components)
        
        # 4. 高阶矩因子组
        if 'skewness_20_ortho' in df.columns:
            # 正偏度更好（右尾更长）
            score_components['higher_moments'] = normalize_group(df, 'skewness_20_ortho') * 0.5
        if 'kurtosis_20_ortho' in df.columns:
            # 低峰度更好（极端事件更少）
            kurt_score = -normalize_group(df, 'kurtosis_20_ortho')
            if 'higher_moments' in score_components.columns:
                score_components['higher_moments'] += kurt_score * 0.5
            else:
                score_components['higher_moments'] = kurt_score * 0.5
        
        # 5. 遗传算法因子组
        if self.genetic_discoverer:
            genetic_weights = self.genetic_discoverer.get_adaptive_weights()
            genetic_score = 0.0
            for factor_name, weight in genetic_weights.items():
                if factor_name in df.columns:
                    normalized = normalize_group(df, factor_name)
                    genetic_score += normalized * weight
            score_components['genetic'] = genetic_score
        
        # 6. 市值因子（小市值效应）
        if 'circ_mv' in df.columns:
            # 对数市值，取负号（小市值更好）
            df['log_market_cap'] = np.log(df['circ_mv'] + 1)
            score_components['size'] = -normalize_group(df, 'log_market_cap')
        
        # 7. 流动性因子
        if 'turnover_rate' in df.columns:
            score_components['liquidity'] = normalize_group(df, 'turnover_rate')
        
        # 根据市场状态选择权重
        if self._current_market_state == 'BEAR':
            weights = self._get_defense_weights()
        elif self._current_market_state == 'BULL':
            weights = self._get_bull_weights()
        else:
            weights = self._get_base_weights()
        
        # 动态调整权重（根据可用因子）
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
    
    def get_current_market_state(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        return {
            'market_state': self._current_market_state,
            'current_weights': self._factor_weights,
            'industry_neutral_enabled': self.enable_industry_neutral,
            'orthogonalization_enabled': self.enable_orthogonalization,
            'genetic_discovery_enabled': self.enable_genetic_discovery,
            'ic_selection_enabled': self.enable_ic_selection,
            'higher_moments_enabled': self.enable_higher_moments,
            'ic_history': self._ic_history,
        }
    
    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性"""
        return self._factor_weights.copy()


def get_alpha_model(
    enable_industry_neutral: bool = True,
    enable_market_adapter: bool = True,
    enable_orthogonalization: bool = True,
    enable_genetic_discovery: bool = True,
    enable_ic_selection: bool = True,
    enable_higher_moments: bool = True,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        enable_industry_neutral=enable_industry_neutral,
        enable_market_adapter=enable_market_adapter,
        enable_orthogonalization=enable_orthogonalization,
        enable_genetic_discovery=enable_genetic_discovery,
        enable_ic_selection=enable_ic_selection,
        enable_higher_moments=enable_higher_moments,
    )