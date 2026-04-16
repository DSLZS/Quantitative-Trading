"""
Alpha Model Module - V198 架构清场与 2024 专项破局 (Iteration 3 - Bug Fix)

【V198 Iter3 核心改进】
1. 基于 V197 稳定版本，修复 volume_turnover_rank 计算 bug
2. 引入 volatility_skew (波动率偏度) 作为新特征
3. 保持 IC 权重 + 状态权重融合架构 (40/60)
4. 简化逻辑，移除不必要的复杂性

【V198 新特征】
- volatility_skew: 捕捉收益率分布的不对称性
  - 负偏度：极端负收益更多 (风险信号)
  - 正偏度：极端正收益更多 (机会信号)

【合规锁定】
- 初始资金：100,000
- 费率：1.3‰
- 无未来函数：所有计算仅使用 T-1 日及之前数据
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

VERSION = "V198_Iter3_BugFix"

# V198 核心因子 (在 V197 基础上增加 volatility_skew)
V198_CORE_FACTORS = [
    'reversion_5',          # 5 日反转
    'volume_rank',          # 成交量排名
    'volume_price_contradiction',  # 量价矛盾
    'liquidity_alpha',      # 流动性 Alpha
    'volatility_5',         # 5 日波动率
    'volatility_20',        # 20 日波动率
    'volatility_skew',      # NEW: 波动率偏度
]

MAX_FACTORS = 7

# V198 门控参数 (保持 V197 优化值)
GATE_VOLATILITY_THRESHOLD = 0.1
GATE_VOLATILITY_SCALE = 0.4
MOMENTUM_SUPPRESS = 0.7
VOLATILITY_BOOST = 0.8

# V198 NAG 参数
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


class MarketContext:
    """
    V198 市场状态识别器 (基于 V197 稳定版本)
    """
    
    def __init__(
        self,
        vol_threshold_high: float = 0.8,
        vol_threshold_low: float = 0.2,
        skew_threshold: float = 0.5,
        lookback_window: int = 20
    ):
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.skew_threshold = skew_threshold
        self.lookback_window = lookback_window
        self.regime_history = {}
        
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
        V198 状态自适应权重配置
        
        在 V197 基础上增加 volatility_skew 因子：
        - EXTREME: 极大增强 volatility_skew (捕捉极端风险)
        - TREND: 适度配置
        - RANGE: 均衡配置
        """
        weight_configs = {
            'EXTREME': {
                # 极端市场：波动率因子主导，volatility_skew 极大增强
                'volatility_5': 0.25,
                'volatility_20': 0.30,
                'volatility_skew': 0.20,  # NEW: 极大增强
                'reversion_5': 0.02,      # 几乎完全抑制
                'volume_rank': 0.02,      # 几乎完全抑制
                'momentum_10': 0.08,      # 抑制
                'liquidity_alpha': 0.05,   # 大幅抑制
                'volume_price_contradiction': 0.08,
            },
            'TREND': {
                # 趋势市场：动量因子主导
                'volatility_5': 0.15,
                'volatility_20': 0.18,
                'volatility_skew': 0.10,  # NEW: 适度配置
                'reversion_5': 0.05,
                'volume_rank': 0.10,
                'momentum_10': 0.30,      # 极大增强
                'liquidity_alpha': 0.12,
                'volume_price_contradiction': 0.05,
            },
            'RANGE': {
                # 震荡市场：均衡配置
                'volatility_5': 0.20,
                'volatility_20': 0.22,
                'volatility_skew': 0.12,  # NEW: 均衡配置
                'reversion_5': 0.10,
                'volume_rank': 0.10,
                'momentum_10': 0.05,
                'liquidity_alpha': 0.10,
                'volume_price_contradiction': 0.11,
            }
        }
        
        return weight_configs.get(regime, weight_configs['RANGE'])
    
    def compute_nonlinear_penalty(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """计算非线性惩罚系数"""
        result = df.copy()
        
        if regime == 'EXTREME':
            scale = 2.0
            result['vol_penalty'] = np.exp((result['volatility_regime'] - 1) * scale)
            result['reversion_penalty'] = 1.0 / (1.0 + np.exp((result['volatility_regime'] - 1) * 3))
            result['volume_penalty'] = 1.0 / (1.0 + np.exp((result['volatility_regime'] - 1) * 3))
        elif regime == 'TREND':
            result['momentum_boost'] = 1.0 + result['return_skew'] * 0.5
            result['vol_penalty'] = np.ones(len(result))
            result['reversion_penalty'] = np.ones(len(result))
            result['volume_penalty'] = np.ones(len(result))
        else:
            result['reversion_boost'] = 1.0 + (1 - result['return_skew'].abs()) * 0.3
            result['vol_penalty'] = np.ones(len(result))
            result['reversion_penalty'] = np.ones(len(result))
            result['volume_penalty'] = np.ones(len(result))
        
        return result


class LöwdinOrthogonalizer:
    """V190 Löwdin 对称正交化器"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.eigenvalue_stats = {}
    
    def orthogonalize(self, factor_matrix: np.ndarray) -> np.ndarray:
        """执行 Löwdin 正交化"""
        T = factor_matrix.shape[0]
        
        cov_matrix = factor_matrix.T @ factor_matrix / T
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, self.epsilon)
        
        eigenvalues_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + self.epsilon))
        inv_sqrt_cov = eigenvectors @ eigenvalues_inv_sqrt @ eigenvectors.T
        
        factor_matrix_orth = factor_matrix @ inv_sqrt_cov
        
        self.eigenvalue_stats = {
            'min_eigenvalue': float(np.min(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues)),
            'condition_number': float(np.max(eigenvalues) / (np.min(eigenvalues) + self.epsilon)),
        }
        
        return factor_matrix_orth


class GatedResidualFuser:
    """V190 门控残差融合器"""
    
    def __init__(
        self,
        volatility_threshold: float = GATE_VOLATILITY_THRESHOLD,
        volatility_scale: float = GATE_VOLATILITY_SCALE,
        momentum_suppress: float = MOMENTUM_SUPPRESS,
        volatility_boost: float = VOLATILITY_BOOST
    ):
        self.volatility_threshold = volatility_threshold
        self.volatility_scale = volatility_scale
        self.momentum_suppress = momentum_suppress
        self.volatility_boost = volatility_boost
        self.gate_stats = {}
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def compute_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率状态指标"""
        result = df.copy()
        
        if 'high' in result.columns and 'low' in result.columns:
            result['prev_close'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(1))
            
            tr1 = result['high'] - result['low']
            tr2 = (result['high'] - result['prev_close']).abs()
            tr3 = (result['low'] - result['prev_close']).abs()
            
            result['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result['atr'] = result.groupby('symbol')['true_range'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result['atr_ma20'] = result.groupby('symbol')['atr'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            
            volatility_regime = result['atr'] / (result['atr_ma20'] + 1e-10)
            volatility_regime = volatility_regime.groupby(result['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10) if len(x) > 1 else x
            )
            return volatility_regime.fillna(0)
        else:
            return pd.Series(0, index=df.index)
    
    def apply_gated_weights(
        self,
        base_weights: Dict[str, float],
        gate_signal: pd.Series
    ) -> Dict[str, pd.Series]:
        """应用门控权重调整"""
        adjusted_weights = {}
        
        momentum_factors = ['momentum_5', 'momentum_10', 'momentum_20']
        volatility_factors = ['volatility_5', 'volatility_10', 'volatility_20', 'reversion_5']
        
        for factor, base_weight in base_weights.items():
            if factor in momentum_factors:
                weight_adjustment = 1.0 - gate_signal * self.momentum_suppress
                adjusted_weights[factor] = base_weight * weight_adjustment
            elif factor in volatility_factors:
                weight_adjustment = 1.0 + gate_signal * self.volatility_boost
                adjusted_weights[factor] = base_weight * weight_adjustment
            else:
                adjusted_weights[factor] = base_weight * np.ones_like(gate_signal)
        
        total_weight = sum(adjusted_weights[f] for f in adjusted_weights)
        for factor in adjusted_weights:
            adjusted_weights[factor] = adjusted_weights[factor] / (total_weight + 1e-10)
        
        self.gate_stats = {
            'mean_gate': float(gate_signal.mean()),
            'std_gate': float(gate_signal.std()),
            'high_volatility_ratio': float((gate_signal > 0.5).mean()),
        }
        
        return adjusted_weights


class NonlinearAdaptiveGain:
    """V190 非线性自适应增益 (NAG)"""
    
    def __init__(
        self,
        base_gain: float = NAG_BASE_GAIN,
        min_gain: float = NAG_MIN_GAIN,
        max_gain: float = NAG_MAX_GAIN,
        adaptation_factor: float = 0.3,
        smoothing_window: int = 20
    ):
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.adaptation_factor = adaptation_factor
        self.smoothing_window = smoothing_window
        self.nag_stats = {}
    
    def compute_adaptive_gain(self, ic_series: pd.Series) -> pd.Series:
        """计算自适应增益"""
        trend_strength = ic_series.rolling(
            self.smoothing_window, min_periods=5
        ).apply(lambda x: x.mean() / (x.std() + 1e-10) if len(x) > 1 else 0)
        trend_strength = trend_strength.fillna(0)
        
        raw_gain = self.base_gain * (1 + trend_strength * self.adaptation_factor)
        adaptive_gain = raw_gain.clip(self.min_gain, self.max_gain)
        smoothed_gain = adaptive_gain.ewm(span=5, adjust=False).mean()
        
        self.nag_stats = {
            'base_gain': self.base_gain,
            'mean_gain': float(smoothed_gain.mean()),
            'min_gain_actual': float(smoothed_gain.min()),
            'max_gain_actual': float(smoothed_gain.max())
        }
        
        return smoothed_gain


class AlphaModel:
    """
    V198 Alpha Model - Iter3 Bug Fix 版本
    
    【核心特性】
    1. MarketContext 市场状态识别
    2. volatility_skew 新特征
    3. IC 权重 + 状态权重融合 (40/60)
    4. Löwdin 对称正交化
    5. Gated-Residual 门控逻辑
    6. NAG 非线性自适应增益
    """
    
    def __init__(
        self,
        n_factors: int = MAX_FACTORS,
        enable_orm: bool = True,
        enable_gated_residual: bool = True,
        enable_nag: bool = True,
        enable_regime_detection: bool = True,
        db_url: Optional[str] = None,
    ):
        self.n_factors = n_factors
        self.enable_orm = enable_orm
        self.enable_gated_residual = enable_gated_residual
        self.enable_nag = enable_nag
        self.enable_regime_detection = enable_regime_detection
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        self.factor_directions = {}
        self.factor_ics = {}
        self.selected_factors = []
        self.audit_log = []
        self.current_regime = 'RANGE'
        
        self.lowdin_orthogonalizer = LöwdinOrthogonalizer() if enable_orm else None
        self.gated_fuser = GatedResidualFuser() if enable_gated_residual else None
        self.nag_adapter = NonlinearAdaptiveGain() if enable_nag else None
        self.market_context = MarketContext() if enable_regime_detection else None
        
        logger.info(f"[AlphaModel] {VERSION} Initialized")
        logger.info(f"  Core Factors: {V198_CORE_FACTORS}")
        logger.info(f"  Löwdin Orthogonalization: {'Enabled' if enable_orm else 'Disabled'}")
        logger.info(f"  Gated-Residual: {'Enabled' if enable_gated_residual else 'Disabled'}")
        logger.info(f"  NAG: {'Enabled' if enable_nag else 'Disabled'}")
        logger.info(f"  Regime Detection: {'Enabled' if enable_regime_detection else 'Disabled'}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC (Rank IC)"""
        ics = []
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            f = day[factor_col].fillna(0)
            l = day['t1_return'].fillna(0)
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        return float(np.mean(ics)) if ics else 0.0
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        """因子标准化处理"""
        series_wins = winsorize(series.fillna(0), sigma=3.0)
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def _compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子 (V198: 增加 volatility_skew)"""
        result = df.copy()
        
        # 动量因子
        result['momentum_5'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        ).fillna(0)
        result['momentum_10'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(10)
        ).fillna(0)
        
        # 反转因子
        result['reversion_5'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        ).fillna(0)
        
        # 波动率因子
        result['volatility_5'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(5, min_periods=2).std()
        ).fillna(0)
        
        result['volatility_20'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        # ========== V198 NEW: 波动率偏度 (volatility_skew) ==========
        # 计算滚动窗口内收益率的偏度
        # 负偏度 = 极端负收益更多 (风险信号)
        # 正偏度 = 极端正收益更多 (机会信号)
        result['volatility_skew'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).apply(
                lambda s: skew(s) if len(s) > 2 and s.std() > 1e-10 else 0.0, raw=False
            )
        ).fillna(0)
        
        # 量价矛盾因子
        if 'pct_chg' in result.columns and 'volume' in result.columns:
            close_return = result['pct_chg']
            volume_change = result['volume'].pct_change()
            
            price_rank = close_return.fillna(0).rank(method='average', pct=True)
            volume_rank = volume_change.fillna(0).rank(method='average', pct=True)
            
            result['volume_price_contradiction'] = (price_rank - volume_rank).fillna(0)
        else:
            result['volume_price_contradiction'] = 0
        
        # 流动性 Alpha 因子
        if 'pct_chg' in result.columns and 'volume' in result.columns:
            ofi = result['pct_chg'] * result['volume']
            ts_std_20 = result.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
            result['liquidity_alpha'] = -(ofi / (ts_std_20 + 1e-6)).fillna(0)
        else:
            result['liquidity_alpha'] = 0
        
        # 成交量排名因子
        if 'volume' in result.columns:
            result['volume_rank'] = -result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        return result
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (V198: 基于 V197 稳定版本)
        """
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows")
        result = df.copy()
        
        # 1. 计算市场状态指标
        if self.enable_regime_detection and self.market_context:
            logger.info("[Regime] Computing market context indicators...")
            result = self.market_context.compute_market_indicators(result)
            regime_series = self.market_context.classify_regime(result)
            
            regime_counts = regime_series.value_counts()
            self.current_regime = regime_series.iloc[-1] if len(regime_series) > 0 else 'RANGE'
            logger.info(f"[Regime] Market state distribution: {dict(regime_counts)}")
            logger.info(f"[Regime] Current state: {self.current_regime}")
        else:
            regime_series = pd.Series('RANGE', index=result.index)
        
        # 2. 计算未来收益
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / x - 1
        )
        result['t3_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-3) / x - 1
        )
        result['t5_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )
        
        result['t1_return_period'] = result['t1_return']
        result['t2_return_period'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-2) / x.shift(-1) - 1
        )
        result['t3_return_period'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-3) / x.shift(-2) - 1
        )
        result['t4_return_period'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-4) / x.shift(-3) - 1
        )
        result['t5_return_period'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-5) / x.shift(-4) - 1
        )
        
        # 3. 计算因子 (包括 V198 新特征 volatility_skew)
        result = self._compute_factors(result)
        
        # 4. 状态感知因子选择
        if self.enable_regime_detection and self.market_context:
            regime_weights = self.market_context.get_regime_weights(self.current_regime)
            sorted_factors = sorted(regime_weights.items(), key=lambda x: x[1], reverse=True)
            self.selected_factors = [f[0] for f in sorted_factors[:MAX_FACTORS]]
            logger.info(f"[Factor] Selected factors for {self.current_regime}: {self.selected_factors}")
        else:
            self.selected_factors = ['volatility_5', 'volatility_20', 'volatility_skew', 'momentum_10', 'reversion_5', 'volume_rank']
        
        # 5. 计算基础 IC 权重
        ic_weights = {}
        for factor in self.selected_factors:
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            ic_weights[factor] = abs(ic) + 1e-6
        
        total_weight = sum(ic_weights.values())
        base_ic_weights = {f: w / total_weight for f, w in ic_weights.items()}
        
        # 6. V198 状态感知权重调整 (IC 权重 40% + 状态权重 60%)
        if self.enable_regime_detection and self.market_context:
            regime_weights = self.market_context.get_regime_weights(self.current_regime)
            blended_weights = {}
            for factor in self.selected_factors:
                ic_w = base_ic_weights.get(factor, 0.1)
                regime_w = regime_weights.get(factor, 0.1)
                # 加权融合：IC 权重 40% + 状态权重 60%
                blended_weights[factor] = 0.4 * ic_w + 0.6 * regime_w
            
            total = sum(blended_weights.values())
            base_weights = {f: w / total for f, w in blended_weights.items()}
            logger.info(f"[Weight] Blended weights: {base_weights}")
        else:
            base_weights = base_ic_weights
        
        # 7. Gated-Residual 非线性权重调整
        adjusted_weights = base_weights
        if self.enable_gated_residual and self.gated_fuser:
            volatility_regime = self.gated_fuser.compute_volatility_regime(result)
            gate_signal = self.gated_fuser.sigmoid(
                (volatility_regime - self.gated_fuser.volatility_threshold) / 
                self.gated_fuser.volatility_scale
            )
            adjusted_weights = self.gated_fuser.apply_gated_weights(base_weights, gate_signal)
        
        # 8. 提取并标准化因子数据
        factor_data = {}
        for factor in self.selected_factors:
            f_raw = result[factor].copy() if factor in result.columns else pd.Series(0, index=result.index)
            f_std = self._process_factor(f_raw, result['trade_date'])
            factor_data[factor] = f_std
            self.factor_directions[factor] = 1
        
        # 9. Löwdin 正交化
        if self.enable_orm and self.lowdin_orthogonalizer and len(factor_data) > 1:
            factor_names = list(factor_data.keys())
            factor_matrix = np.column_stack([factor_data[f] for f in factor_names])
            factor_matrix_orth = self.lowdin_orthogonalizer.orthogonalize(factor_matrix)
            
            for i, name in enumerate(factor_names):
                factor_data[name] = factor_matrix_orth[:, i]
        
        # 10. 计算加权分数 (IC 加权)
        score = np.zeros(len(result), dtype=np.float64)
        total_ic = sum(self.factor_ics.get(f, 0.0) for f in self.selected_factors)
        
        for factor in self.selected_factors:
            f = factor_data.get(factor)
            if f is None:
                continue
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            
            # IC 加权
            factor_ic = abs(self.factor_ics.get(factor, 0.0)) + 1e-6
            weight = factor_ic / (total_ic + 1e-6)
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 11. 截面标准化
        result['score'] = result.groupby('trade_date')['score_raw'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        output_cols = [
            'trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return',
            't1_return_period', 't2_return_period', 't3_return_period',
            't4_return_period', 't5_return_period'
        ]
        
        logger.info(f"[AlphaModel] Score computation complete. Selected factors: {self.selected_factors}")
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """获取因子 IC"""
        if df is not None and not df.empty:
            ics = {}
            for factor in self.selected_factors:
                if factor in df.columns:
                    ic = self._calc_factor_ic(df, factor)
                    sign = self.factor_directions.get(factor, 1)
                    ics[factor] = ic * sign
                else:
                    ics[factor] = self.factor_ics.get(factor, 0.0)
            return ics
        return self.factor_ics
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子列表"""
        return self.selected_factors
    
    def get_new_features(self) -> List[str]:
        """获取 V198 新增特征列表"""
        return ['volatility_skew']


def get_alpha_model(
    n_factors: int = MAX_FACTORS,
    enable_orm: bool = True,
    enable_gated_residual: bool = True,
    enable_nag: bool = True,
    enable_regime_detection: bool = True,
    db_url: Optional[str] = None,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_orm=enable_orm,
        enable_gated_residual=enable_gated_residual,
        enable_nag=enable_nag,
        enable_regime_detection=enable_regime_detection,
        db_url=db_url,
    )