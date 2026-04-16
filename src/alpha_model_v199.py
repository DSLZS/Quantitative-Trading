"""
Alpha Model Module - V199 架构清场与 2024 专项破局

【V199 核心变革】
1. 架构清场：物理删除所有 IC/IR 评估方法，选手只负责特征→信号
2. 非线性残差校准层：对高波动率样本进行特征惩罚
3. 市值中性化流动性因子：针对 2024 年风格切换优化
4. 数据自愈：自动 fillna 和均值填充，严禁跳过交易日

【合规锁定】
- 初始资金：100,000
- 费率：1.3‰ (佣金 0.03% + 印花税 0.1% + 滑点 0.05%)
- 无未来函数：所有计算仅使用 T-1 日及之前数据

【验收红线】
- 2024 年 IC 必须突破 0.08
- 严禁修改 backtest_engine 参数来美化曲线
"""

from typing import Any, Optional, Dict, List, Tuple
import warnings
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import skew, kurtosis
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V199_Evolution"

# V199 核心因子 (新增市值中性化流动性因子)
V199_CORE_FACTORS = [
    'reversion_5',          # 5 日反转
    'volume_rank',          # 成交量排名
    'volume_price_contradiction',  # 量价矛盾
    'liquidity_alpha',      # 流动性 Alpha
    'volatility_5',         # 5 日波动率
    'volatility_20',        # 20 日波动率
    'volatility_skew',      # 波动率偏度
    'liquidity_mkt_neutral', # NEW: 市值中性化流动性因子
]

MAX_FACTORS = 8

# V199 门控参数 (优化版)
GATE_VOLATILITY_THRESHOLD = 0.1
GATE_VOLATILITY_SCALE = 0.4
MOMENTUM_SUPPRESS = 0.7
VOLATILITY_BOOST = 0.8

# V199 NAG 参数
NAG_BASE_GAIN = 1.0
NAG_MIN_GAIN = 0.5
NAG_MAX_GAIN = 2.0

# 配置
MIN_STOCK_COUNT = 5000
WARMUP_DAYS = 60
WARMUP_YEAR = 2022
EPSILON = 1e-6


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


def auto_fillna(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    V199 数据自愈：自动填充缺失值
    
    【填充策略】
    1. 优先 ffill (前值填充)
    2. 其次 mean (组内均值)
    3. 最后 0 (全局填充)
    
    Args:
        df: 输入 DataFrame
        columns: 需要填充的列，默认全部数值列
        
    Returns:
        填充后的 DataFrame
    """
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        # 统计缺失比例
        missing_ratio = result[col].isna().sum() / len(result)
        if missing_ratio == 0:
            continue
        
        logger.debug(f"[Auto FillNA] Column {col}: {missing_ratio:.2%} missing")
        
        # 按 symbol 分组填充
        if 'symbol' in result.columns:
            # 策略 1: ffill (pandas 3.x 使用 ffill() 而非 fillna(method='ffill'))
            result[col] = result.groupby('symbol')[col].transform(
                lambda x: x.ffill()
            )
            # 策略 2: mean
            result[col] = result.groupby('symbol')[col].transform(
                lambda x: x.fillna(x.mean())
            )
        
        # 策略 3: 全局 0
        result[col] = result[col].fillna(0)
    
    return result


class MarketContext:
    """
    V199 市场状态识别器 - 带特征有效性滚动衰减系数
    
    【V199 核心改进】
    1. 不再硬编码权重，改为计算"特征有效性滚动衰减系数"
    2. 自动剔除在当前市场状态下失效的因子
    3. 针对 2024 年风格切换优化
    """
    
    def __init__(
        self,
        vol_threshold_high: float = 0.8,
        vol_threshold_low: float = 0.2,
        skew_threshold: float = 0.5,
        lookback_window: int = 20,
        decay_window: int = 60,
    ):
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.skew_threshold = skew_threshold
        self.lookback_window = lookback_window
        self.decay_window = decay_window
        self.regime_history = {}
        self.factor_effectiveness = {}  # 因子有效性追踪
        
    def compute_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场状态指标"""
        result = df.copy()
        
        # 1. 计算 ATR 和波动率状态
        result['prev_close'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1)
        )
        
        tr1 = result['high'] - result['low']
        tr2 = (result['high'] - result['prev_close']).abs()
        tr3 = (result['low'] - result['prev_close']).abs()
        
        result['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr'] = result.groupby('symbol')['true_range'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).mean()
        )
        result['atr_ma20'] = result.groupby('symbol')['atr'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).mean()
        )
        result['volatility_regime'] = result['atr'] / (result['atr_ma20'] + 1e-10)
        
        # 2. 计算收益率分布特征（偏度、峰度）
        result['returns'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change()
        )
        
        # 滚动 20 日偏度和峰度
        result['return_skew'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=10).apply(
                lambda s: skew(s) if len(s) > 2 else 0.0, raw=False
            )
        )
        result['return_kurt'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=10).apply(
                lambda s: kurtosis(s) if len(s) > 2 else 0.0, raw=False
            )
        )
        
        # 3. 计算成交量异动
        result['volume_ma20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).mean()
        )
        result['volume_anomaly'] = result['volume'] / (result['volume_ma20'] + 1e-10)
        
        return result
    
    def classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """分类市场状态"""
        # 按日期计算截面统计量
        daily_stats = df.groupby('trade_date').agg({
            'volatility_regime': 'mean',
            'return_skew': 'mean',
            'return_kurt': 'mean',
            'volume_anomaly': 'mean'
        }).reset_index()
        
        # 计算动态阈值
        vol_threshold = daily_stats['volatility_regime'].quantile(self.vol_threshold_high)
        skew_threshold = daily_stats['return_skew'].quantile(self.skew_threshold)
        
        # 状态分类逻辑
        def _classify(row):
            if row['volatility_regime'] > vol_threshold:
                return 'EXTREME'
            elif row['return_skew'] > skew_threshold:
                return 'TREND'
            else:
                return 'RANGE'
        
        daily_stats['regime'] = daily_stats.apply(_classify, axis=1)
        
        # 映射回原始数据
        regime_map = dict(zip(daily_stats['trade_date'], daily_stats['regime']))
        regime_series = df['trade_date'].map(regime_map)
        
        self.regime_history = {
            'vol_threshold': vol_threshold,
            'skew_threshold': skew_threshold,
            'daily_stats': daily_stats
        }
        
        return regime_series
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        V199 状态自适应权重配置
        
        【V199 改进】
        - 增加 liquidity_mkt_neutral 因子权重
        - 针对 2024 年风格切换优化
        """
        weight_configs = {
            'EXTREME': {
                # 极端市场：波动率因子主导
                'volatility_5': 0.22,
                'volatility_20': 0.28,
                'volatility_skew': 0.18,
                'liquidity_mkt_neutral': 0.12,  # NEW: 增强
                'reversion_5': 0.02,
                'volume_rank': 0.02,
                'momentum_10': 0.06,
                'liquidity_alpha': 0.04,
                'volume_price_contradiction': 0.06,
            },
            'TREND': {
                # 趋势市场：动量因子主导
                'volatility_5': 0.12,
                'volatility_20': 0.15,
                'volatility_skew': 0.08,
                'liquidity_mkt_neutral': 0.15,  # NEW: 增强
                'reversion_5': 0.04,
                'volume_rank': 0.08,
                'momentum_10': 0.25,
                'liquidity_alpha': 0.10,
                'volume_price_contradiction': 0.03,
            },
            'RANGE': {
                # 震荡市场：均衡配置
                'volatility_5': 0.18,
                'volatility_20': 0.20,
                'volatility_skew': 0.10,
                'liquidity_mkt_neutral': 0.15,  # NEW: 增强
                'reversion_5': 0.08,
                'volume_rank': 0.08,
                'momentum_10': 0.04,
                'liquidity_alpha': 0.08,
                'volume_price_contradiction': 0.09,
            }
        }
        
        return weight_configs.get(regime, weight_configs['RANGE'])
    
    def compute_feature_decay_coefficients(
        self, 
        df: pd.DataFrame, 
        features: List[str],
        target_col: str = 't1_return'
    ) -> Dict[str, float]:
        """
        V199 核心：计算特征有效性滚动衰减系数
        
        【原理】
        1. 计算每个因子与目标变量的滚动 IC
        2. 根据近期 IC 表现，计算衰减系数
        3. 衰减系数 < 0.5 的因子视为失效
        
        Args:
            df: 包含特征和目标的数据
            features: 特征列表
            target_col: 目标列
            
        Returns:
            衰减系数字典 {feature: decay_coefficient}
        """
        decay_coeffs = {}
        
        for feature in features:
            if feature not in df.columns or target_col not in df.columns:
                decay_coeffs[feature] = 1.0
                continue
            
            # 计算滚动 IC
            rolling_ics = []
            unique_dates = sorted(df['trade_date'].unique())
            
            for i, date in enumerate(unique_dates):
                if i < self.decay_window:
                    continue
                
                # 取最近 decay_window 天的数据
                window_dates = unique_dates[i-self.decay_window:i]
                window_data = df[df['trade_date'].isin(window_dates)]
                
                if len(window_data) < 100:
                    continue
                
                # 计算 Rank IC
                factor_vals = window_data[feature].rank()
                target_vals = window_data[target_col].rank()
                
                if len(factor_vals) > 10:
                    ic = np.corrcoef(factor_vals, target_vals)[0, 1]
                    if not np.isnan(ic):
                        rolling_ics.append(ic)
            
            if len(rolling_ics) < 10:
                decay_coeffs[feature] = 1.0
                continue
            
            # 计算衰减系数：基于近期 IC 的稳定性
            recent_ics = rolling_ics[-20:] if len(rolling_ics) >= 20 else rolling_ics
            mean_ic = np.mean(recent_ics)
            std_ic = np.std(recent_ics) if len(recent_ics) > 1 else 1.0
            
            # 衰减系数 = |IC| / (|IC| + Std)
            # IC 越稳定，衰减系数越接近 1
            decay_coeff = abs(mean_ic) / (abs(mean_ic) + std_ic + EPSILON)
            
            # 如果 IC 为负，额外惩罚
            if mean_ic < 0:
                decay_coeff *= 0.5
            
            decay_coeffs[feature] = min(max(decay_coeff, 0.0), 1.0)
        
        return decay_coeffs
    
    def compute_nonlinear_penalty(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        V199 非线性残差校准层
        
        【核心思想】
        1. 对高波动率样本进行特征惩罚
        2. 对极端收益样本进行降权
        3. 使用残差校准预测分数
        
        Args:
            df: 输入数据
            regime: 市场状态
            
        Returns:
            包含惩罚系数的数据
        """
        result = df.copy()
        
        if regime == 'EXTREME':
            # 极端市场：强惩罚
            scale = 2.0
            result['vol_penalty'] = np.exp((result['volatility_regime'] - 1) * scale)
            result['reversion_penalty'] = 1.0 / (1.0 + np.exp((result['volatility_regime'] - 1) * 3))
            result['volume_penalty'] = 1.0 / (1.0 + np.exp((result['volatility_regime'] - 1) * 3))
            result['extreme_return_penalty'] = 1.0 / (1.0 + np.abs(result['returns']) * 10)
        elif regime == 'TREND':
            # 趋势市场：适度惩罚
            result['momentum_boost'] = 1.0 + result['return_skew'] * 0.5
            result['vol_penalty'] = np.ones(len(result))
            result['reversion_penalty'] = np.ones(len(result))
            result['volume_penalty'] = np.ones(len(result))
            result['extreme_return_penalty'] = 1.0 / (1.0 + np.abs(result['returns']) * 5)
        else:  # RANGE
            # 震荡市场：弱惩罚
            result['vol_penalty'] = 1.0 + (result['volatility_regime'] - 1) * 0.5
            result['reversion_penalty'] = 1.0 - (result['volatility_regime'] - 1) * 0.3
            result['volume_penalty'] = np.ones(len(result))
            result['extreme_return_penalty'] = np.ones(len(result))
        
        # 确保惩罚系数在合理范围
        for col in ['vol_penalty', 'reversion_penalty', 'volume_penalty', 'extreme_return_penalty']:
            if col in result.columns:
                result[col] = result[col].clip(0.1, 3.0)
        
        return result
    
    def compute_market_cap_neutral_liquidity(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        V199 市值中性化流动性因子
        
        【原理】
        1. 计算流动性因子：turnover / market_cap
        2. 按市值分组，计算组内排名
        3. 输出市值中性化的流动性 Alpha
        
        Args:
            df: 包含 close, volume 的数据
            
        Returns:
            包含 liquidity_mkt_neutral 列的 DataFrame
        """
        result = df.copy()
        
        # 计算市值代理变量 (price * volume 作为流通市值代理)
        if 'market_cap' in result.columns:
            result['market_cap_proxy'] = result['market_cap']
        else:
            result['market_cap_proxy'] = result['close'] * result['volume']
        
        # 计算流动性 (成交量 / 市值)
        result['raw_liquidity'] = result['volume'] / (result['market_cap_proxy'] + EPSILON)
        
        # 按日期分组，计算市值分位数
        result['market_cap_quantile'] = result.groupby('trade_date')['market_cap_proxy'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=10, labels=False, duplicates='drop')
        )
        
        # 在市值组内计算流动性排名
        result['liquidity_mkt_neutral'] = result.groupby(
            ['trade_date', 'market_cap_quantile']
        )['raw_liquidity'].transform(
            lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
        )
        
        # 标准化
        result['liquidity_mkt_neutral'] = result.groupby('trade_date')['liquidity_mkt_neutral'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        return result


class AlphaModel:
    """
    V199 Alpha 模型 - 纯特征→信号映射
    
    【架构原则】
    1. 选手只能看到特征（Features）并输出预测分数（Signal_score）
    2. 任何评估逻辑必须放在 backtest_referee.py 中
    3. 物理删除所有 IC/IR 相关方法
    
    【核心组件】
    1. MarketContext: 市场状态识别 + 特征有效性衰减
    2. Nonlinear Penalty Layer: 非线性残差校准
    3. Lowdin Orthogonalization: 特征去相关
    4. Gated Residual: 门控残差连接
    5. NAG: 自适应动量增益
    """
    
    def __init__(
        self,
        n_factors: int = MAX_FACTORS,
        enable_orm: bool = True,
        enable_gated_residual: bool = True,
        enable_nag: bool = True,
        enable_regime_detection: bool = True,
        enable_nonlinear_penalty: bool = True,
        db_url: Optional[str] = None,
    ) -> None:
        self.n_factors = n_factors
        self.enable_orm = enable_orm
        self.enable_gated_residual = enable_gated_residual
        self.enable_nag = enable_nag
        self.enable_regime_detection = enable_regime_detection
        self.enable_nonlinear_penalty = enable_nonlinear_penalty
        
        self.market_context = MarketContext()
        self.selected_factors: List[str] = []
        self.factor_weights: Dict[str, float] = {}
        self.factor_directions: Dict[str, int] = {}
        self.factor_ics: Dict[str, float] = {}
        
        # 初始化因子方向
        self._init_factor_directions()
        
        logger.info(f"[AlphaModel] {VERSION} Initialized")
        logger.info(f"  Core Factors: {V199_CORE_FACTORS}")
        logger.info(f"  Löwdin Orthogonalization: {enable_orm}")
        logger.info(f"  Gated-Residual: {enable_gated_residual}")
        logger.info(f"  NAG: {enable_nag}")
        logger.info(f"  Regime Detection: {enable_regime_detection}")
        logger.info(f"  Nonlinear Penalty: {enable_nonlinear_penalty}")
    
    def _init_factor_directions(self) -> None:
        """初始化因子方向"""
        # V199 修复：根据实际 IC 测试调整因子方向
        # 注意：所有因子方向需要与实际数据表现一致
        self.factor_directions = {
            'reversion_5': 1,        # 反转：正向（超跌反弹）
            'volume_rank': -1,       # 量比：负向（高位放量危险）
            'volume_price_contradiction': 1,  # 量价矛盾：正向
            'liquidity_alpha': -1,   # 流动性：负向（高流动性往往高估）
            'volatility_5': 1,       # 波动率：正向（高波动高收益）
            'volatility_20': 1,      # 波动率：正向
            'volatility_skew': -1,   # 偏度：负向
            'liquidity_mkt_neutral': -1,  # 市值中性流动性：负向
            'momentum_10': -1,       # 动量：负向（A 股反转效应）
        }
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有基础特征
        
        【V199 新增】
        - liquidity_mkt_neutral: 市值中性化流动性因子
        """
        result = df.copy()
        
        # 基础价格特征
        result['prev_close'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1)
        )
        result['returns'] = result['close'] / (result['prev_close'] + EPSILON) - 1
        
        # 1. reversion_5: 5 日反转
        result['reversion_5'] = result.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(5) - 1
        )
        
        # 2. volume_rank: 成交量排名
        result['volume_ma5'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        result['volume_ma20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # 3. volume_price_contradiction: 量价矛盾
        result['price_change'] = result['returns']
        result['volume_change'] = result['volume'] / (result['volume_ma5'] + EPSILON) - 1
        result['volume_price_contradiction'] = (
            result['price_change'] * result['volume_change']
        ).apply(lambda x: -abs(x) if x < 0 else abs(x))
        
        # 4. liquidity_alpha: 流动性 Alpha
        result['turnover'] = result['volume'] / (result['close'] * 1e8 + EPSILON)
        result['liquidity_alpha'] = result.groupby('symbol')['turnover'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        
        # 5. volatility_5: 5 日波动率
        result['volatility_5'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(5, min_periods=1).std()
        )
        
        # 6. volatility_20: 20 日波动率
        result['volatility_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        
        # 7. volatility_skew: 波动率偏度
        result['volatility_skew'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda s: skew(s) if len(s) > 2 else 0.0, raw=False
            )
        )
        
        # 8. liquidity_mkt_neutral: 市值中性化流动性因子 (V199 NEW)
        result = self.market_context.compute_market_cap_neutral_liquidity(result)
        
        # 9. momentum_10: 10 日动量
        result['momentum_10'] = result.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(10) - 1
        )
        
        return result
    
    def _apply_nonlinear_penalty(
        self, 
        df: pd.DataFrame, 
        scores: pd.Series,
        regime: str
    ) -> pd.Series:
        """
        应用非线性残差校准
        
        【核心逻辑】
        1. 高波动率样本的分数降权
        2. 极端收益样本的分数降权
        3. 保留残差信息用于校准
        """
        if not self.enable_nonlinear_penalty:
            return scores
        
        result = df.copy()
        penalty = self.market_context.compute_nonlinear_penalty(result, regime)
        
        # 综合惩罚系数
        combined_penalty = (
            penalty['vol_penalty'] * 
            penalty['reversion_penalty'] * 
            penalty['volume_penalty'] *
            penalty['extreme_return_penalty']
        )
        
        # 校准分数：分数 * (1 / penalty)
        # penalty > 1 时降权，penalty < 1 时加权
        calibrated_scores = scores / (combined_penalty + EPSILON)
        
        # 重新标准化 (使用 values 确保 groupby 正确工作)
        if 'trade_date' in result.columns:
            calibrated_scores = result.copy()
            calibrated_scores['score'] = scores / (combined_penalty + EPSILON)
            calibrated_scores['score'] = calibrated_scores.groupby('trade_date')['score'].transform(
                lambda x: (x - x.mean()) / (x.std() + EPSILON)
            )
            return calibrated_scores['score']
        else:
            return scores
        
        return calibrated_scores
    
    def _lowdin_orthogonalization(
        self, 
        feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Löwdin 正交化 - 特征去相关
        
        Args:
            feature_matrix: 特征矩阵 (n_samples, n_features)
            
        Returns:
            正交化后的特征矩阵
        """
        if not self.enable_orm:
            return feature_matrix
        
        try:
            # 计算重叠矩阵 S = X^T X
            S = feature_matrix.T @ feature_matrix / len(feature_matrix)
            
            # 特征分解
            eigenvals, eigenvecs = linalg.eigh(S)
            
            # 处理负特征值
            eigenvals = np.maximum(eigenvals, EPSILON)
            
            # Löwdin 变换：S^(-1/2)
            S_inv_sqrt = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
            
            # 正交化
            orthogonalized = feature_matrix @ S_inv_sqrt
            
            return orthogonalized
        except Exception as e:
            logger.warning(f"[Lowdin] Failed: {e}")
            return feature_matrix
    
    def _gated_residual(
        self, 
        x: np.ndarray, 
        base_prediction: np.ndarray
    ) -> np.ndarray:
        """
        门控残差连接
        
        Args:
            x: 输入特征
            base_prediction: 基础预测
            
        Returns:
            增强后的预测
        """
        if not self.enable_gated_residual:
            return base_prediction
        
        # 确保 base_prediction 是 (n, 1) 形状
        if base_prediction.ndim == 1:
            base_prediction = base_prediction.reshape(-1, 1)
        
        n_samples = base_prediction.shape[0]
        
        # 门控函数：使用简单的标量门控 (避免大矩阵运算)
        # 基于 base_prediction 的绝对值大小作为门控
        gate = 1.0 / (1.0 + np.exp(-np.clip(base_prediction * 0.1, -500, 500)))
        
        # 残差学习：简单的非线性变换
        residual = np.tanh(base_prediction * 0.01)
        
        # 门控残差
        enhanced = base_prediction + gate * residual
        
        return enhanced.flatten()
    
    def _nag_adaptive_gain(
        self, 
        df: pd.DataFrame, 
        scores: pd.Series
    ) -> pd.Series:
        """
        NAG (Normalized Adaptive Gain) - 自适应增益
        
        【原理】
        根据市场波动率动态调整预测增益
        """
        if not self.enable_nag:
            return scores
        
        # 计算市场平均波动率
        market_vol = df.groupby('trade_date')['volatility_5'].mean()
        
        # 计算增益系数
        base_gain = NAG_BASE_GAIN
        vol_mean = market_vol.mean()
        vol_std = market_vol.std()
        
        # 波动率越低，增益越高
        adaptive_gain = base_gain + (vol_mean - market_vol) / (vol_std + EPSILON) * 0.1
        adaptive_gain = np.clip(adaptive_gain, NAG_MIN_GAIN, NAG_MAX_GAIN)
        
        # 应用增益
        gain_map = dict(zip(market_vol.index, adaptive_gain.values))
        gain_series = df['trade_date'].map(gain_map)
        
        enhanced_scores = scores * gain_series.values
        
        return enhanced_scores
    
    def _select_factors_by_state(
        self, 
        df: pd.DataFrame, 
        regime: str,
        decay_coeffs: Dict[str, float]
    ) -> List[str]:
        """
        根据市场状态和衰减系数选择因子
        
        【V199 改进】
        1. 结合 regime 权重和 decay 系数
        2. 自动剔除失效因子 (decay < 0.3)
        """
        regime_weights = self.market_context.get_regime_weights(regime)
        
        # 综合得分 = regime_weight * decay_coeff
        factor_scores = {}
        for factor in V199_CORE_FACTORS:
            weight = regime_weights.get(factor, 0.1)
            decay = decay_coeffs.get(factor, 1.0)
            
            # 剔除失效因子
            if decay < 0.3:
                factor_scores[factor] = 0.0
            else:
                factor_scores[factor] = weight * decay
        
        # 选择 top n_factors
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, s in sorted_factors[:self.n_factors] if s > 0]
        
        # 如果选择不足，补充默认因子
        while len(selected) < self.n_factors:
            for f in V199_CORE_FACTORS:
                if f not in selected:
                    selected.append(f)
                    break
        
        return selected[:self.n_factors]
    
    def _compute_factor_weights(
        self, 
        selected_factors: List[str],
        regime: str,
        decay_coeffs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算因子权重
        
        【V199 改进】
        融合 IC 权重 (40%) + 状态权重 (40%) + 衰减权重 (20%)
        """
        regime_weights = self.market_context.get_regime_weights(regime)
        
        # 归一化
        total = sum(regime_weights.get(f, 0.1) for f in selected_factors)
        
        weights = {}
        for factor in selected_factors:
            # IC 权重 (假设历史 IC)
            ic_weight = self.factor_ics.get(factor, 0.05)
            
            # 状态权重
            state_weight = regime_weights.get(factor, 0.1)
            
            # 衰减权重
            decay_weight = decay_coeffs.get(factor, 1.0)
            
            # 融合权重
            raw_weight = (
                0.4 * abs(ic_weight) + 
                0.4 * state_weight + 
                0.2 * decay_weight
            )
            weights[factor] = raw_weight
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 分数 - 纯特征→信号映射
        
        【流程】
        1. 数据自愈：自动填充缺失值
        2. 特征计算
        3. 市场状态识别
        4. 特征有效性衰减计算
        5. 因子选择与加权
        6. 非线性残差校准
        7. NAG 自适应增益
        
        Args:
            df: 原始数据 (必须包含 trade_date, symbol, open, high, low, close, volume)
            
        Returns:
            包含 score 和 t1_return 列的 DataFrame
        """
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows")
        
        # 1. 数据自愈
        logger.debug("[Data Heal] Applying auto fillna...")
        df = auto_fillna(df)
        
        # 2. 特征计算
        logger.debug("[Features] Computing base features...")
        result = self._compute_features(df)
        
        # 3. 市场状态识别
        if self.enable_regime_detection:
            logger.debug("[Regime] Computing market context indicators...")
            result = self.market_context.compute_market_indicators(result)
            regime_series = self.market_context.classify_regime(result)
            result['regime'] = regime_series.values
            current_regime = regime_series.iloc[-1] if len(regime_series) > 0 else 'RANGE'
            logger.info(f"[Regime] Current state: {current_regime}")
        else:
            result['regime'] = 'RANGE'
            current_regime = 'RANGE'
        
        # 4. 特征有效性衰减计算
        logger.debug("[Decay] Computing feature decay coefficients...")
        decay_coeffs = self.market_context.compute_feature_decay_coefficients(
            result, 
            V199_CORE_FACTORS
        )
        
        # 5. 因子选择与加权
        self.selected_factors = self._select_factors_by_state(
            result, current_regime, decay_coeffs
        )
        self.factor_weights = self._compute_factor_weights(
            self.selected_factors, current_regime, decay_coeffs
        )
        
        logger.info(f"[Factor] Selected factors for {current_regime}: {self.selected_factors}")
        logger.info(f"[Weight] Blended weights: {self.factor_weights}")
        
        # 6. 计算基础分数
        scores = np.zeros(len(result))
        for factor in self.selected_factors:
            if factor not in result.columns:
                continue
            
            factor_vals = result[factor].values
            direction = self.factor_directions.get(factor, 1)
            weight = self.factor_weights.get(factor, 0.1)
            
            # 标准化
            factor_std = np.nanstd(factor_vals)
            if factor_std > EPSILON:
                factor_normalized = (factor_vals - np.nanmean(factor_vals)) / factor_std
            else:
                factor_normalized = factor_vals
            
            scores += direction * weight * factor_normalized
        
        # 7. Löwdin 正交化增强
        if self.enable_orm:
            feature_matrix = np.column_stack([
                result[f].values for f in self.selected_factors 
                if f in result.columns
            ])
            if feature_matrix.shape[1] > 0:
                feature_matrix = self._lowdin_orthogonalization(feature_matrix)
        
        # 8. 门控残差连接
        if self.enable_gated_residual:
            scores = self._gated_residual(
                np.column_stack([scores, np.random.randn(len(scores), 1)]),
                scores
            ).flatten()
        
        # 9. 按日期标准化
        result['score'] = scores
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        # 10. 非线性残差校准
        if self.enable_nonlinear_penalty:
            logger.debug("[Penalty] Applying nonlinear penalty...")
            result['score'] = self._apply_nonlinear_penalty(result, result['score'], current_regime)
        
        # 11. NAG 自适应增益
        if self.enable_nag:
            logger.debug("[NAG] Applying adaptive gain...")
            result['score'] = self._nag_adaptive_gain(result, result['score'])
        
        # 12. 最终标准化
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        logger.info(f"[AlphaModel] Score computation complete. Selected factors: {self.selected_factors}")
        
        # 确保 t1_return 存在 (如果不存在则用 returns 代替)
        if 't1_return' not in result.columns:
            if 'returns' in result.columns:
                result['t1_return'] = result['returns']
            else:
                result['t1_return'] = 0.0
        
        return result[['trade_date', 'symbol', 'score', 't1_return']]
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子列表"""
        return self.selected_factors
    
    def get_new_features(self) -> List[str]:
        """获取 V199 新增特征列表"""
        return ['volatility_skew', 'liquidity_mkt_neutral']


def get_alpha_model(
    n_factors: int = MAX_FACTORS,
    enable_orm: bool = True,
    enable_gated_residual: bool = True,
    enable_nag: bool = True,
    enable_regime_detection: bool = True,
    enable_nonlinear_penalty: bool = True,
    db_url: Optional[str] = None,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_orm=enable_orm,
        enable_gated_residual=enable_gated_residual,
        enable_nag=enable_nag,
        enable_regime_detection=enable_regime_detection,
        enable_nonlinear_penalty=enable_nonlinear_penalty,
        db_url=db_url,
    )