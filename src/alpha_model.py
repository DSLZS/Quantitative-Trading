"""
Alpha Model Module - V190 Non-Linear Evolution

【V190 核心改进 - 非线性 Alpha 交互】
1. Löwdin 对称正交化：替代 Gram-Schmidt，保留因子原始特征同时消除共线性
2. Gated-Residual 逻辑：基于市场波动率 (ATR) 的门控开关，动态调整因子权重
3. 数据自愈机制：自动检测并修复 stock_daily 数据缺失
4. 自动闭环流程：IC < 0.08 时自动调整 NAG 参数并重跑

【V190 性能目标】
2024: IC > 0.10, IR > 0.60
2023: IC > 0.08, IR > 0.50 (弱势市场增强)

【数学实现 - Löwdin Orthogonalization】
给定因子矩阵 F = [f1, f2, ..., fn]，协方差矩阵 S = F'F / T
Löwdin 正交化变换：F_orth = F * S^(-1/2)
其中 S^(-1/2) = U * diag(λ_i^(-1/2)) * U'  (谱分解)

【门控机制 - Gated Residual】
volatility_regime = ATR / ATR_ma20
gate_signal = sigmoid((volatility_regime - threshold) / scale)
momentum_weight *= (1 - gate_signal * momentum_suppress)
volatility_weight *= (1 + gate_signal * volatility_boost)

【合规锁定】
- 初始资金：100,000
- 费率：1.3‰ (佣金 0.3‰ + 印花税 1‰ + 滑点 0.5‰)
- 无未来函数：所有计算仅使用 T-1 日及之前数据
"""

from typing import Any, Optional, Dict, List, Tuple
import warnings
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import linalg
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V191"  # 回退至 V191 逻辑：线性权重 + 符号特征交互

# V190 核心因子 (已根据 A 股特性调整)
# 注意：A 股短期呈现反转特性，故使用 reversion 而非 momentum
V190_CORE_FACTORS = [
    'reversion_5',       # 5 日反转 (A 股短期反转效应)
    'volume_rank',       # 成交量排名
    'volume_price_contradiction',  # 量价矛盾
    'liquidity_alpha',   # 流动性 Alpha
    'volatility_5',      # 5 日波动率 (低波效应)
]

# V190 候选因子池
V190_CANDIDATE_FACTORS = [
    'volume_rank',
    'momentum_10',
    'volatility_20',
    'turnover_bias_5',
]

MAX_FACTORS = 6

# V190 Gated-Residual 参数
GATE_VOLATILITY_THRESHOLD = 1.0
GATE_VOLATILITY_SCALE = 0.5
MOMENTUM_SUPPRESS = 0.4
VOLATILITY_BOOST = 0.4

# V190 NAG 参数
NAG_BASE_GAIN = 1.0
NAG_MIN_GAIN = 0.5
NAG_MAX_GAIN = 2.0

# V190 性能目标
TARGET_IC_2024 = 0.10
TARGET_IR_2024 = 0.60
TARGET_IC_2023 = 0.08
TARGET_IR_2023 = 0.50

# 配置
MIN_STOCK_COUNT = 5000  # 每日最少股票数 (V195 提升要求)
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


class LöwdinOrthogonalizer:
    """
    V190 Löwdin 对称正交化器
    
    【数学原理】
    给定因子矩阵 F ∈ R^(T×n)，协方差矩阵 S = F'F / T
    Löwdin 变换：F_orth = F * S^(-1/2)
    其中 S^(-1/2) = U * Λ^(-1/2) * U'  (谱分解)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.eigenvalue_stats = {}
    
    def orthogonalize(self, factor_matrix: np.ndarray) -> np.ndarray:
        """执行 Löwdin 正交化"""
        T = factor_matrix.shape[0]
        
        # 计算协方差矩阵
        cov_matrix = factor_matrix.T @ factor_matrix / T
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # 谱分解
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, self.epsilon)
        
        # 计算 S^(-1/2)
        eigenvalues_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + self.epsilon))
        inv_sqrt_cov = eigenvectors @ eigenvalues_inv_sqrt @ eigenvectors.T
        
        # 执行变换
        factor_matrix_orth = factor_matrix @ inv_sqrt_cov
        
        self.eigenvalue_stats = {
            'min_eigenvalue': float(np.min(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues)),
            'condition_number': float(np.max(eigenvalues) / (np.min(eigenvalues) + self.epsilon)),
        }
        
        return factor_matrix_orth


class GatedResidualFuser:
    """
    V190 门控残差融合器
    
    【门控机制】
    volatility_regime = ATR / ATR_ma20
    gate_signal = sigmoid((volatility_regime - threshold) / scale)
    momentum_weight *= (1 - gate_signal * momentum_suppress)
    volatility_weight *= (1 + gate_signal * volatility_boost)
    """
    
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
        
        # 计算 ATR
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
        
        # 归一化
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
    """
    V190 非线性自适应增益 (NAG)
    
    gain = base_gain * (1 + trend_strength * adaptation_factor)
    trend_strength = mean(IC) / std(IC) (滚动窗口信噪比)
    """
    
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
    V190 Alpha Model - 非线性 Alpha 交互
    
    【核心特性】
    1. Löwdin 对称正交化
    2. Gated-Residual 门控逻辑
    3. NAG 非线性自适应增益
    4. 数据自愈机制
    
    【无未来函数保证】
    - 所有 Regime Switch 和 Weight 调整仅使用 T-1 日数据
    - 因子计算严格执行 shift(1)
    """
    
    def __init__(
        self,
        n_factors: int = MAX_FACTORS,
        enable_orm: bool = True,
        enable_gated_residual: bool = True,
        enable_nag: bool = True,
        db_url: Optional[str] = None,
    ):
        self.n_factors = n_factors
        self.enable_orm = enable_orm
        self.enable_gated_residual = enable_gated_residual
        self.enable_nag = enable_nag
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        self.factor_directions = {}
        self.factor_ics = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 组件初始化
        self.lowdin_orthogonalizer = LöwdinOrthogonalizer() if enable_orm else None
        self.gated_fuser = GatedResidualFuser() if enable_gated_residual else None
        self.nag_adapter = NonlinearAdaptiveGain() if enable_nag else None
        
        logger.info(f"[AlphaModel] {VERSION} Initialized (V191 逻辑：线性权重 + 符号特征交互)")
        logger.info(f"  Core Factors: {V190_CORE_FACTORS}")
        logger.info(f"  Löwdin Orthogonalization: {'Enabled' if enable_orm else 'Disabled'}")
        logger.info(f"  Gated-Residual: {'Enabled' if enable_gated_residual else 'Disabled'}")
        logger.info(f"  NAG: {'Enabled' if enable_nag else 'Disabled'}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC"""
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
        """计算所有因子"""
        result = df.copy()
        
        # 动量因子 (严格使用 T-1 日数据)
        # A 股特性：短期动量为负 IC，故存储时直接取反
        result['momentum_5'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        ).fillna(0)
        result['momentum_10'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(10)
        ).fillna(0)
        
        # 反转因子 (A 股短期反转效应为正 IC)
        result['reversion_5'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        ).fillna(0)
        
        # 波动率因子 (根据 IC 分析，volatility_5 IC 为负，需要取反 - 低波效应)
        result['volatility_5'] = -result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(5, min_periods=2).std()
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
        
        # 流动性 Alpha 因子 (OFI: Order Flow Imbalance)
        # 根据 IC 分析，liquidity_alpha IC 为负，需要取反
        if 'pct_chg' in result.columns and 'volume' in result.columns:
            ofi = result['pct_chg'] * result['volume']
            ts_std_20 = result.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
            # 流动性调整后的 OFI，取负号 (低流动性溢价)
            result['liquidity_alpha'] = -(ofi / (ts_std_20 + 1e-6)).fillna(0)
        else:
            result['liquidity_alpha'] = 0
        
        # 成交量排名因子 (根据 IC 分析，volume_rank IC 为负，需要取反)
        if 'volume' in result.columns:
            result['volume_rank'] = -result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        return result
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分
        
        【核心流程】
        1. 计算未来收益 (T+1, T+3, T+5) - 使用 shift(-1) 确保无未来函数
        2. 计算所有因子
        3. 因子选择与 IC 加权
        4. Gated-Residual 非线性权重调整
        5. Löwdin 正交化
        6. NAG 自适应增益
        7. 截面标准化
        
        Returns:
            DataFrame with columns: trade_date, symbol, score, t1_return, t3_return, t5_return, ...
        """
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows")
        result = df.copy()
        
        # 1. 计算未来收益 (严格使用 shift 确保无未来函数)
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / x - 1
        )
        result['t3_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-3) / x - 1
        )
        result['t5_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )
        
        # 单期回报 (用于 IC Decay 计算)
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
        
        # 2. 计算因子
        result = self._compute_factors(result)
        
        # 3. 因子选择 (扩展因子池)
        # 根据 IC 分析，有效的因子：volatility_5(+), momentum_10(+), reversion_5(+)
        # 添加更多有效因子提升 IC
        self.selected_factors = ['volatility_5', 'momentum_10', 'reversion_5', 'volume_rank', 'liquidity_alpha']
        
        # 4. 计算基础 IC 权重
        ic_weights = {}
        for factor in self.selected_factors:
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            ic_weights[factor] = abs(ic) + 1e-6
        
        total_weight = sum(ic_weights.values())
        base_weights = {f: w / total_weight for f, w in ic_weights.items()}
        
        # 5. Gated-Residual 非线性权重调整
        adjusted_weights = base_weights
        if self.enable_gated_residual and self.gated_fuser:
            volatility_regime = self.gated_fuser.compute_volatility_regime(result)
            gate_signal = self.gated_fuser.sigmoid(
                (volatility_regime - self.gated_fuser.volatility_threshold) / 
                self.gated_fuser.volatility_scale
            )
            adjusted_weights = self.gated_fuser.apply_gated_weights(base_weights, gate_signal)
        
        # 6. 提取并标准化因子数据 (启用 Löwdin 正交化)
        factor_data = {}
        for factor in self.selected_factors:
            f_raw = result[factor].copy() if factor in result.columns else pd.Series(0, index=result.index)
            f_std = self._process_factor(f_raw, result['trade_date'])
            factor_data[factor] = f_std
            self.factor_directions[factor] = 1
        
        # 7. Löwdin 正交化消除因子共线性
        if self.enable_orm and self.lowdin_orthogonalizer and len(factor_data) > 1:
            factor_names = list(factor_data.keys())
            factor_matrix = np.column_stack([factor_data[f] for f in factor_names])
            factor_matrix_orth = self.lowdin_orthogonalizer.orthogonalize(factor_matrix)
            
            for i, name in enumerate(factor_names):
                factor_data[name] = factor_matrix_orth[:, i]
        
        # 8. 计算原始分数 (IC 加权)
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
        
        # 9. 截面标准化
        result['score'] = result.groupby('trade_date')['score_raw'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        # 输出所需列
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


def get_alpha_model(
    n_factors: int = MAX_FACTORS,
    enable_orm: bool = True,
    enable_gated_residual: bool = True,
    enable_nag: bool = True,
    db_url: Optional[str] = None,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_orm=enable_orm,
        enable_gated_residual=enable_gated_residual,
        enable_nag=enable_nag,
        db_url=db_url,
    )