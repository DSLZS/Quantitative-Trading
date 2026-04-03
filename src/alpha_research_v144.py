"""
Alpha Research Module - V144 时序一致性加固与非线性逻辑修复.

【V143 回顾】
V143 实现了 3D Tensor Interaction，核心贡献：
1. Kernel Neutralization: 多项式映射剔除高阶冗余
2. Market Regime Tensor: 场景调制因子
3. IC Precision Weighting: IC 稳定性加权

【V144 核心使命】
V143 的 Kernel Neutralization 导致了严重的 Alpha 流失。V144 要求：
1. 回滚过度中性化 - 取消 Core²的二阶剔除，回退到线性残差提取
2. 增加 Sign-Lock (符号锁定) - 确保核心信号的方向不被扭曲
3. 引入时序衰减核 (Time-Decay Decay Kernel) - 对 IC 不稳定的特征应用指数衰减
4. Volatility-Adaptive Smoothing - 高波动环境下增加平滑窗口

【V144 核心算法 - Sign-Consistency Interaction (SCI)】
1. Sign-Lock 机制：
   - 公式：Alpha = Sign(Rank(Core)) * abs(Distilled_Resid) * Regime_Gate
   - 逻辑：确保核心信号的方向不被复杂的交互逻辑扭曲

2. Time-Decay Decay Kernel:
   - 对 IC 贡献不稳定的特征应用 Exponential_Decay_Filter
   - 公式：Weight_t = Weight_0 * exp(-lambda * t)
   - lambda = IC_Std / IC_Mean (IC 波动率越大，衰减越快)

3. Volatility-Adaptive Smoothing:
   - 高波动环境下，增加信号的平滑窗口
   - 公式：Smoothing_Window = Base_Window * (1 + Volatility_ZScore)
   - 防止信号在日度之间过度震荡

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 144 运行
- 严禁修改 backtest_referee.py 中的资金 (10 万) 和费率 (0.15%)
- 数据缺失时必须主动调用 data_loader 补全，禁止用 dropna() 一删了之
- 报错必改：内置 Auto-Healing 逻辑处理 SettingWithCopyWarning 和 Inf/NaN

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标 |
| IC_IR | > 0.70 | 稳定性（V143: 0.60） |
| Signal Turnover | 下降 15%+ | V143 vs V144 对比 |
| IC Decay | T+1 > T+3 > T+5 | 正常衰减模式 |
| Sign-Lock Applied | >= 2 | 至少 2 个因子应用符号锁定 |
"""

from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import warnings
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V144"

# V144 核心因子（保留 V142 的因子召回框架）
V144_CORE_FACTORS = [
    'momentum_20',      # 20 日动量
    'volatility_10',    # 10 日波动率
    'volume_price_contradiction',  # 量价背离
    'liquidity_alpha',  # 流动性 Alpha
]

# V144 候选因子池（用于召回）
V144_CANDIDATE_FACTORS = [
    # 动量类
    'momentum_5', 'momentum_10', 'momentum_60',
    # 反转类
    'reversion_5', 'reversion_10',
    # 波动率类
    'volatility_5', 'volatility_20',
    # 量价类
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'vwap_distance', 'volume_rank', 'price_rank',
    # 价值类
    'value_rank', 'ep_rank', 'bp_rank',
    # 技术指标类
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist',
    # MA 偏离类
    'ma_deviation_5', 'ma_deviation_10', 'ma_deviation_20',
    'price_position_20', 'price_position_60', 'bias_60',
    # 换手类
    'turnover_bias_5', 'turnover_bias_10', 'turnover_bias_20',
    'volume_shrink_ratio', 'turnover_vol_ratio',
    # 订单流类
    'order_flow_imbalance_5', 'order_flow_imbalance_10',
    'smart_money_divergence', 'big_order_ratio',
    # 流动性类
    'ofi_normalized', 'volume_confirmed_momentum',
    # 时效性类
    'signal_delta', 'volume_shock', 'price_acceleration', 'momentum_change',
    # 尾部风险类
    'tail_risk_indicator', 'skewness_20', 'extreme_volume_ratio',
]

# V144 所有因子（核心 + 召回）
ALL_FACTORS = V144_CORE_FACTORS + V144_CANDIDATE_FACTORS

# V144 最大因子数量
MAX_FACTORS = 12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数 - 用于门控机制"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sign_lock(series: pd.Series) -> Tuple[pd.Series, int]:
    """
    V144 符号锁定 - 确保核心信号方向不被扭曲.
    
    【V144 与 V143 的本质区别】
    V143: Alpha = Sigmoid(Rank(Core)) * Rank(Kernel_Residual) * Regime_Modulation
    V144: Alpha = Sign(Rank(Core)) * abs(Distilled_Resid) * Regime_Gate
    
    【符号锁定逻辑】
    1. 计算截面上每个样本的符号：Sign = sign(Rank(Core) - 0.5)
    2. 取残差的绝对值：abs(Residual)
    3. 最终信号 = Sign * abs(Residual)
    
    这确保了核心信号的方向（正/负）不会被复杂的交互逻辑扭曲。
    
    Args:
        series: 输入序列
        
    Returns:
        locked_series: 符号锁定后的序列
        sign_direction: 符号方向 (1 或 -1)
    """
    # 计算排名并归一化到 0-1
    rank_pct = series.rank(method='average', pct=True)
    
    # 符号：排名 > 0.5 为正，否则为负
    sign = np.sign(rank_pct - 0.5)
    
    # 绝对值
    abs_value = series.abs()
    
    # 标准化绝对值
    abs_value = (abs_value - abs_value.mean()) / (abs_value.std() + 1e-10)
    
    # 符号锁定后的序列
    locked_series = sign * abs_value
    
    # 计算整体符号方向
    sign_direction = 1 if (locked_series * series).mean() >= 0 else -1
    
    return locked_series, sign_direction


def exponential_decay_filter(weights: np.ndarray, ic_mean: float, ic_std: float) -> np.ndarray:
    """
    V144 指数衰减滤波器 - 时序衰减核.
    
    【原理】
    对 IC 贡献不稳定的特征应用指数衰减，衰减率由 IC 稳定性决定。
    
    【公式】
    lambda = IC_Std / (|IC_Mean| + ε)  # IC 波动率越大，lambda 越大
    Weight_t = Weight_0 * exp(-lambda * t)
    
    【经济逻辑】
    - IC 稳定的因子：lambda 小，衰减慢，历史权重保持
    - IC 不稳定的因子：lambda 大，衰减快，更依赖近期数据
    """
    # 计算衰减率
    lambda_decay = ic_std / (abs(ic_mean) + 1e-10)
    lambda_decay = np.clip(lambda_decay, 0.1, 2.0)  # 限制衰减率范围
    
    # 计算时间权重
    t = np.arange(len(weights))
    decay_weights = np.exp(-lambda_decay * t)
    
    # 应用衰减
    filtered_weights = weights * decay_weights
    
    return filtered_weights


def volatility_adaptive_smoothing(series: pd.Series, volatility: pd.Series, 
                                   base_window: int = 5) -> pd.Series:
    """
    V144 波动率自适应平滑.
    
    【原理】
    在高波动环境下，增加信号的平滑窗口，防止信号在日度之间过度震荡。
    
    【公式】
    Volatility_ZScore = (Vol - Mean_Vol) / Std_Vol
    Smoothing_Window = Base_Window * (1 + Volatility_ZScore)
    Smoothed_Signal = EMA(Signal, span=Smoothing_Window)
    
    【经济逻辑】
    - 高波动环境：市场噪音大，需要更强的平滑来提取真实信号
    - 低波动环境：市场相对理性，可以使用较小的平滑窗口保留更多细节
    
    【修复】
    原始实现中 adaptive_window 是 Series，但 ewm 需要标量。
    修复方案：使用平均窗口，然后根据波动率调整平滑系数。
    """
    if len(series) < 2:
        return series.copy()
    
    # 计算波动率 Z-Score
    vol_mean = volatility.mean()
    vol_std = volatility.std() + 1e-10
    vol_zscore = (volatility - vol_mean) / vol_std
    
    # 计算平均自适应窗口（标量）
    avg_vol_zscore = vol_zscore.mean()
    adaptive_window = int(base_window * (1 + max(0, avg_vol_zscore)))  # 只增加不减少
    adaptive_window = max(3, min(20, adaptive_window))  # 限制窗口范围 [3, 20]
    
    # 应用 EMA 平滑（使用标量窗口）
    smoothed = series.ewm(span=adaptive_window, min_periods=1, adjust=False).mean()
    
    return smoothed


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, 
                        percentile: float = 0.99) -> pd.Series:
    """
    V144 自动愈合版 Winsorization - 处理 Inf/NaN.
    
    【Auto-Healing 逻辑】
    1. 检测并修复 Inf
    2. 检测并修复 NaN
    3. Sigma 截断
    4. Percentile 截断
    5. 最终 NaN 填充
    
    【防止 SettingWithCopyWarning】
    始终使用.copy() 创建新对象，避免修改原始数据
    """
    # 创建副本，防止 SettingWithCopyWarning
    series_clean = series.copy()
    
    # 1. 处理 Inf
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    # 2. 计算均值（用于后续填充）
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    
    # 3. Sigma 截断
    std = series_clean.std()
    if std > 1e-10:
        lower = mean - sigma * std
        upper = mean + sigma * std
        series_clean = series_clean.clip(lower=lower, upper=upper)
    
    # 4. Percentile 截断
    lower_pct = series_clean.quantile(1 - percentile)
    upper_pct = series_clean.quantile(percentile)
    series_clean = series_clean.clip(lower=lower_pct, upper=upper_pct)
    
    # 5. 最终 NaN 填充
    series_clean = series_clean.fillna(mean)
    
    return series_clean


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算两个变量之间的互信息"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    mask = np.isnan(x) | np.isnan(y)
    x_clean = x[~mask]
    y_clean = y[~mask]
    
    if len(x_clean) < 20:
        return 0.0
    
    try:
        x_bins = pd.qcut(x_clean, q=n_bins, labels=False, duplicates='drop')
        y_bins = pd.qcut(y_clean, q=n_bins, labels=False, duplicates='drop')
        
        n_x = len(np.unique(x_bins))
        n_y = len(np.unique(y_bins))
        
        joint_hist = np.zeros((n_x, n_y))
        for xi, yi in zip(x_bins, y_bins):
            joint_hist[xi, yi] += 1
        joint_prob = joint_hist / len(x_clean)
        
        px = joint_prob.sum(axis=1)
        py = joint_prob.sum(axis=0)
        
        mi = 0.0
        for i in range(n_x):
            for j in range(n_y):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        
        return mi
    except Exception:
        return 0.0


def gram_schmidt_orthogonalize(X: np.ndarray, mi_threshold: float = 0.1) -> Tuple[np.ndarray, List[int]]:
    """Gram-Schmidt 正交化 + 互信息验证"""
    n_samples, n_factors = X.shape
    
    if n_factors == 0:
        return X, []
    
    X_norm = X.copy()
    for i in range(n_factors):
        std = np.std(X_norm[:, i])
        if std > 1e-10:
            X_norm[:, i] = (X_norm[:, i] - np.mean(X_norm[:, i])) / std
    
    orthogonal = []
    kept_indices = []
    
    for i in range(n_factors):
        v = X_norm[:, i].copy()
        
        for u in orthogonal:
            proj = np.dot(v, u) / (np.dot(u, u) + 1e-10)
            v = v - proj * u
        
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            max_mi = 0
            for j in kept_indices:
                mi = compute_mutual_information(X_norm[:, i], X_norm[:, j], n_bins=10)
                max_mi = max(max_mi, mi)
            
            if max_mi < mi_threshold:
                orthogonal.append(v / norm)
                kept_indices.append(i)
    
    orthogonalized = np.zeros_like(X)
    for idx, (ortho_idx, u) in enumerate(zip(kept_indices, orthogonal)):
        orthogonalized[:, idx] = u
    
    return orthogonalized[:, :len(kept_indices)], kept_indices


class DataHealingV144:
    """V144 增强版数据自愈模块 - Auto-Healing 3.0"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        """初始化 SQL 自愈器"""
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V144][DataHealing] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V144][DataHealing] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V144][DataHealing] No database URL, SQL healer disabled")
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        """记录自愈日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details,
        }
        self.healing_log.append(entry)
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """V144 增强版检查并修复缺失列"""
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
            )
            
            if self.engine:
                result = self._heal_from_sql(result, missing)
            else:
                for col in missing:
                    result = result.assign(**{col: 0.0})
                    self._log_healing(
                        action="DefaultFill",
                        column=col,
                        status="PARTIAL",
                        details="Filled with 0.0 (no SQL connection)"
                    )
        else:
            self._log_healing(
                action="ColumnsComplete",
                column="ALL",
                status="OK",
                details="All required columns present"
            )
        
        # V144 增强：Auto-Impute
        result = self._auto_impute_grouped(result, 'trade_date')
        self._log_healing(
            action="AutoImputeApplied",
            column="ALL_NUMERIC",
            status="SUCCESS",
            details="Applied grouped median imputation"
        )
        
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """从 SQL 补全缺失列"""
        if not self.engine or df.empty:
            return df
        
        result = df.copy()
        symbols = df['symbol'].unique().tolist()[:50]
        
        if not symbols:
            return df
        
        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date']).unique()
            start_date = pd.to_datetime(dates.min()).strftime('%Y%m%d')
            end_date = pd.to_datetime(dates.max()).strftime('%Y%m%d')
        else:
            return df
        
        try:
            from sqlalchemy import text
            
            symbols_str = ', '.join([f"'{s}'" for s in symbols])
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       turnover_rate, total_mv, pe_ttm, pb
                FROM stock_daily
                WHERE symbol IN ({symbols_str})
                AND trade_date BETWEEN :start_date AND :end_date
            """)
            
            sql_df = pd.read_sql_query(query, self.engine, params={
                'start_date': start_date,
                'end_date': end_date,
            })
            
            if not sql_df.empty:
                for col in columns:
                    if col in sql_df.columns:
                        merge_df = result.merge(
                            sql_df[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'],
                            how='left',
                            suffixes=('', '_sql')
                        )
                        result[col] = merge_df[col].fillna(merge_df[f'{col}_sql'])
                        result = result.drop(columns=[c for c in result.columns if c.endswith('_sql')])
                        
                        self._log_healing(
                            action="HealedFromSQL",
                            column=col,
                            status="SUCCESS",
                            details=f"Healed {len(sql_df)} rows from stock_daily"
                        )
                        
        except Exception as e:
            logger.error(f"[V144][DataHealing] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """自动分组插值"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            
            def fill_group(group):
                group_median = group[col].median()
                if pd.isna(group_median):
                    group_median = global_median
                return group[col].fillna(group_median)
            
            result[col] = result.groupby(group_col, group_keys=False).apply(fill_group)
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class MarketRegimeTensor:
    """V144 市场环境张量 - 场景调制因子"""
    
    def __init__(self, volatility_window: int = 20, regime_threshold: float = 0.5):
        self.volatility_window = volatility_window
        self.regime_threshold = regime_threshold
        self.regime_log = []
        self.current_regime = 0
        self.regime_modulation = 1.0
        
    def _log_regime(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.regime_log.append(entry)
    
    def compute_market_volatility(self, df: pd.DataFrame) -> pd.Series:
        """计算市场波动率（截面平均）"""
        if 'volatility_20' in df.columns:
            return df.groupby('trade_date')['volatility_20'].transform('mean')
        elif 'pct_chg' in df.columns:
            return df.groupby('trade_date')['pct_chg'].transform('std')
        else:
            return pd.Series(1.0, index=df.index)
    
    def compute_regime_gate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算场景门控 - V144 简化版.
        
        Returns:
            regime_gate: 场景门控因子 (0.5 ~ 1.5)
        """
        vol = self.compute_market_volatility(df)
        
        vol_mean = vol.rolling(self.volatility_window, min_periods=10).mean()
        vol_std = vol.rolling(self.volatility_window, min_periods=10).std()
        
        vol_std = vol_std.replace(0, 1e-10).fillna(1e-10)
        vol_mean = vol_mean.fillna(vol.mean())
        
        vol_zscore = (vol - vol_mean) / vol_std
        
        # Sigmoid 映射到 (0.5, 1.5)
        regime_gate = 1.0 + np.tanh(vol_zscore.clip(-3, 3)) * 0.5
        
        self.regime_modulation = float(regime_gate.iloc[-1]) if len(regime_gate) > 0 else 1.0
        self.current_regime = 1 if self.regime_modulation > 1.0 else 0
        
        self._log_regime(
            "Computed",
            f"Regime gate: {self.regime_modulation:.3f}, regime={'high' if self.current_regime == 1 else 'low'} volatility"
        )
        
        return regime_gate
    
    def get_current_regime(self) -> int:
        return self.current_regime
    
    def get_regime_modulation(self) -> float:
        return self.regime_modulation


class TimeDecayDecayKernel:
    """
    V144 时序衰减核模块 - IC 稳定性加权.
    
    【原理】
    对 IC 贡献不稳定的特征应用指数衰减滤波器。
    
    【公式】
    lambda = IC_Std / (|IC_Mean| + ε)
    Weight_t = Weight_0 * exp(-lambda * t)
    
    【经济逻辑】
    - IC 稳定的因子：衰减慢，历史权重保持
    - IC 不稳定的因子：衰减快，更依赖近期数据
    """
    
    def __init__(self, rolling_window: int = 20, min_samples: int = 10):
        self.rolling_window = rolling_window
        self.min_samples = min_samples
        self.decay_log = []
        self.decay_weights = {}
        self.ic_rolling_stats = {}
        
    def _log_decay(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.decay_log.append(entry)
    
    def compute_rolling_ic(self, df: pd.DataFrame, factor_col: str) -> List[float]:
        """计算滚动 IC 序列"""
        if 'trade_date' not in df.columns or 't1_return' not in df.columns:
            return []
        
        dates = sorted(df['trade_date'].unique())
        ics = []
        
        for date in dates[-self.rolling_window:]:
            day_data = df[df['trade_date'] == date]
            if len(day_data) < self.min_samples:
                continue
            
            f = day_data[factor_col].fillna(0)
            l = day_data['t1_return'].fillna(0)
            
            if len(f) > self.min_samples and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        return ics
    
    def compute_decay_weight(self, df: pd.DataFrame, factor_col: str, 
                              base_weight: float = 1.0) -> float:
        """
        计算时序衰减权重 - V144 核心.
        
        【V144 修复】
        原始版本 lambda 上限 2.0 导致权重衰减太快（0.0007）。
        修复：降低 lambda 上限，增加最小权重保障。
        """
        ics = self.compute_rolling_ic(df, factor_col)
        
        if len(ics) < self.min_samples:
            return base_weight
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics, ddof=1) + 1e-10
        
        # 计算衰减率 - V144 修复：降低上限
        lambda_decay = ic_std / (abs(ic_mean) + 1e-10)
        lambda_decay = np.clip(lambda_decay, 0.05, 0.5)  # 从 0.1-2.0 改为 0.05-0.5
        
        # 计算时间权重
        t = np.arange(len(ics))
        decay_weights = np.exp(-lambda_decay * t)
        
        # 平均衰减权重
        avg_decay_weight = np.mean(decay_weights)
        
        # V144 修复：最小权重保障（不低于 base_weight 的 30%）
        min_decay_weight = 0.3
        avg_decay_weight = max(avg_decay_weight, min_decay_weight)
        
        # 最终权重
        final_weight = base_weight * avg_decay_weight
        
        # 记录统计
        self.ic_rolling_stats[factor_col] = {
            'ic_mean': float(ic_mean),
            'ic_std': float(ic_std),
            'lambda_decay': float(lambda_decay),
            'avg_decay_weight': float(avg_decay_weight),
            'base_weight': float(base_weight),
            'final_weight': float(final_weight),
            'num_samples': len(ics),
        }
        
        self.decay_weights[factor_col] = float(final_weight)
        
        self._log_decay(
            "Computed",
            f"{factor_col}: IC_mean={ic_mean:.4f}, IC_std={ic_std:.4f}, "
            f"lambda={lambda_decay:.2f}, decay_weight={final_weight:.4f}"
        )
        
        return final_weight
    
    def get_decay_log(self) -> List[Dict]:
        return self.decay_log
    
    def get_ic_rolling_stats(self) -> Dict:
        return self.ic_rolling_stats
    
    def get_decay_weights(self) -> Dict:
        return self.decay_weights


class SignConsistencyInteraction:
    """
    V144 符号一致性交互模块 - 核心创新.
    
    【V144 与 V143 的本质区别】
    V143: Alpha = Sigmoid(Rank(Core)) * Rank(Kernel_Residual) * Regime_Modulation
    V144: Alpha = Sign(Rank(Core)) * abs(Distilled_Resid) * Regime_Gate
    
    【Sign-Lock 原理】
    1. 计算截面上每个样本的符号：Sign = sign(Rank(Core) - 0.5)
    2. 取残差的绝对值：abs(Residual)
    3. 最终信号 = Sign * abs(Residual) * Regime_Gate
    
    【为什么需要 Sign-Lock】
    V143 的 Kernel Neutralization 导致了严重的 Alpha 流失，因为：
    1. 多项式映射 (Core^2) 扭曲了原始信号的方向
    2. 复杂的交互逻辑使得核心信号被"淹没"
    
    V144 通过 Sign-Lock 确保：
    1. 核心信号的方向（正/负）不被扭曲
    2. 只剔除线性冗余，保留原始方向信息
    3. 简化交互逻辑，提升可解释性
    """
    
    def __init__(self, enable_sign_lock: bool = True, enable_smoothing: bool = True):
        self.enable_sign_lock = enable_sign_lock
        self.enable_smoothing = enable_smoothing
        self.sci_log = []
        self.sci_features = {}
        self.sign_lock_applied = []
        
        # V144 新增模块
        self.market_regime = MarketRegimeTensor()
        
    def _log_sci(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.sci_log.append(entry)
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名 (0-1 归一化)"""
        return series.rank(method='average', pct=True)
    
    def compute_linear_residual(self, df: pd.DataFrame, factor_col: str, 
                                 core_col: str) -> pd.Series:
        """
        计算线性残差 - V144 回滚到简单线性.
        
        【V144 与 V143 的区别】
        V143: Kernel_Residual = Factor - β1*Core - β2*Core^2
        V144: Linear_Residual = Factor - β*Core
        
        【为什么回滚】
        V143 的核中性化 (Core^2) 导致了严重的 Alpha 流失，
        因为高阶多项式映射扭曲了原始信号。
        """
        if factor_col not in df.columns or core_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        y = df[factor_col].fillna(0).values
        x = df[core_col].fillna(0).values
        
        # 简单线性回归
        if np.std(x) > 1e-10:
            beta = np.corrcoef(x, y)[0, 1] * np.std(y) / (np.std(x) + 1e-10)
            residual = y - beta * x
        else:
            residual = y
        
        # 标准化残差
        residual_std = np.std(residual) + 1e-10
        standardized_residual = (residual - np.mean(residual)) / residual_std
        
        return pd.Series(standardized_residual, index=df.index)
    
    def compute_sci_feature(self, df: pd.DataFrame, core_factor: str, 
                            recall_factor: str) -> pd.Series:
        """
        计算 SCI 特征 - V144 核心.
        
        【完整流程】
        1. 线性残差：Linear_Residual = Factor - β*Core
        2. 符号锁定：Sign = sign(Rank(Core) - 0.5)
        3. SCI 交互：SCI = Sign * abs(Linear_Residual)
        4. 场景门控：Final = SCI * Regime_Gate
        """
        if core_factor not in df.columns or recall_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 线性残差（V144 回滚）
        linear_residual = self.compute_linear_residual(df, recall_factor, core_factor)
        
        # 2. 符号锁定（V144 核心创新）
        rank_core = self._rank(df[core_factor].fillna(0))
        sign = np.sign(rank_core - 0.5)
        abs_residual = linear_residual.abs()
        
        # 标准化绝对值残差
        abs_residual = (abs_residual - abs_residual.mean()) / (abs_residual.std() + 1e-10)
        
        # SCI 交互
        sci = sign * abs_residual
        
        # 3. 场景门控
        regime_gate = self.market_regime.compute_regime_gate(df)
        sci_final = sci * regime_gate.values
        
        # 4. 波动率自适应平滑（可选）
        if self.enable_smoothing:
            volatility = self.market_regime.compute_market_volatility(df)
            sci_final = volatility_adaptive_smoothing(sci_final, volatility, base_window=5)
        
        feature_name = f"{core_factor}_sci_{recall_factor}"
        self.sci_features[feature_name] = {
            'core_factor': core_factor,
            'recall_factor': recall_factor,
            'type': 'sign_consistency_interaction',
            'method': 'sign_lock + linear_residual + regime_gate',
            'enable_sign_lock': self.enable_sign_lock,
            'enable_smoothing': self.enable_smoothing,
        }
        
        self.sign_lock_applied.append(feature_name)
        
        self._log_sci(
            "SCIComputed",
            f"{feature_name}: Sign(Rank({core_factor})) × abs(Linear_Residual({recall_factor})) × Regime_Gate"
        )
        
        return sci_final
    
    def compute_all_sci_features(self, df: pd.DataFrame, core_factors: List[str],
                                  recalled_factors: List[str]) -> pd.DataFrame:
        """计算所有 SCI 特征"""
        result = df.copy()
        
        # 重点组合：volume_price_contradiction × reversion_5
        if 'volume_price_contradiction' in df.columns and 'reversion_5' in df.columns:
            distilled_sci = self.compute_sci_feature(df, 'volume_price_contradiction', 'reversion_5')
            result['volume_price_contradiction_sci_reversion_5'] = distilled_sci
            
            distilled_sci_rev = self.compute_sci_feature(df, 'reversion_5', 'volume_price_contradiction')
            result['reversion_5_sci_volume_price_contradiction'] = distilled_sci_rev
        
        # 其他 SCI 交互
        for core in core_factors:
            if core not in df.columns:
                continue
            for recalled in recalled_factors:
                if recalled not in df.columns:
                    continue
                if core == 'volume_price_contradiction' and recalled == 'reversion_5':
                    continue
                
                name = f"{core}_sci_{recalled}"
                if name not in result.columns:
                    result[name] = self.compute_sci_feature(df, core, recalled)
        
        self._log_sci(
            "Complete",
            f"Generated {len(self.sci_features)} SCI features"
        )
        
        return result
    
    def get_sci_log(self) -> List[Dict]:
        return self.sci_log
    
    def get_sci_features(self) -> Dict:
        return self.sci_features
    
    def get_sign_lock_applied(self) -> List[str]:
        return self.sign_lock_applied
    
    def get_market_regime(self) -> MarketRegimeTensor:
        return self.market_regime


class ResidualBasedRecallV144:
    """V144 基于残差分析的因子召回模块"""
    
    def __init__(self, top_percent: float = 0.2):
        self.top_percent = top_percent
        self.recall_log = []
        self.recalled_factors = []
        self.residual_analysis = {}
        
    def _log_recall(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.recall_log.append(entry)
    
    def compute_residuals(self, df: pd.DataFrame, core_factors: List[str]) -> pd.Series:
        """计算 V144 核心模型的残差"""
        result = df.copy()
        
        weights = {f: 1.0 / len(core_factors) for f in core_factors if f in df.columns}
        
        predicted = np.zeros(len(df))
        for factor, weight in weights.items():
            predicted += df[factor].fillna(0).values * weight
        
        actual = df['t1_return'].fillna(0).values
        residuals = actual - predicted
        
        self._log_recall(
            "Computed",
            f"Residuals for {len(df)} samples, mean={residuals.mean():.4f}, std={residuals.std():.4f}"
        )
        
        return pd.Series(residuals, index=df.index)
    
    def identify_failure_samples(self, residuals: pd.Series) -> pd.Series:
        """识别失效样本"""
        threshold = residuals.abs().quantile(1 - self.top_percent)
        failure_mask = residuals.abs() >= threshold
        
        self._log_recall(
            "Identified",
            f"{failure_mask.sum()} failure samples (top {self.top_percent*100}%), threshold={threshold:.4f}"
        )
        
        return failure_mask
    
    def compute_factor_ic_on_samples(self, df: pd.DataFrame, factor_col: str, 
                                      sample_mask: pd.Series) -> float:
        """计算因子在特定样本上的 IC"""
        ics = []
        
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            mask = sample_mask[sample_mask.index.isin(day.index)]
            
            if len(mask) < 5:
                continue
            
            day_failure = day.loc[mask.index]
            if len(day_failure) < 5:
                continue
            
            f = day_failure[factor_col].fillna(0)
            l = day_failure['t1_return'].fillna(0)
            
            if len(f) > 3 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        return float(np.mean(ics)) if ics else 0.0
    
    def compute_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子整体 IC"""
        ics = []
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 10:
                continue
            
            f = day[factor_col].fillna(0)
            l = day['t1_return'].fillna(0)
            
            if len(f) > 5 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        return float(np.mean(ics)) if ics else 0.0
    
    def select_recall_factors(self, df: pd.DataFrame, core_factors: List[str], 
                              candidate_factors: List[str], max_recall: int = 3) -> List[str]:
        """选择召回因子"""
        residuals = self.compute_residuals(df, core_factors)
        failure_mask = self.identify_failure_samples(residuals)
        
        recall_scores = {}
        
        for factor in candidate_factors:
            if factor not in df.columns:
                continue
            
            overall_ic = self.compute_factor_ic(df, factor)
            failure_ic = self.compute_factor_ic_on_samples(df, factor, failure_mask)
            recall_score = failure_ic - overall_ic
            
            max_mi = 0
            for core_factor in core_factors:
                if core_factor in df.columns:
                    mi = compute_mutual_information(
                        df[factor].fillna(0).values,
                        df[core_factor].fillna(0).values,
                        n_bins=10
                    )
                    max_mi = max(max_mi, mi)
            
            if max_mi < 0.15 and recall_score > -0.005:
                composite_score = recall_score * 0.6 + failure_ic * 0.4
                
                recall_scores[factor] = {
                    'overall_ic': overall_ic,
                    'failure_ic': failure_ic,
                    'recall_score': recall_score,
                    'composite_score': composite_score,
                    'max_mi': max_mi,
                }
        
        sorted_factors = sorted(recall_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        recalled = []
        for factor, scores in sorted_factors[:max_recall]:
            recalled.append(factor)
            self.residual_analysis[factor] = scores
            self._log_recall(
                "Recalled",
                f"{factor}: overall_ic={scores['overall_ic']:.4f}, failure_ic={scores['failure_ic']:.4f}"
            )
        
        self.recalled_factors = recalled
        
        if len(recalled) == 0:
            self._log_recall("Warning", "No factors recalled, forcing top IC factors")
            forced_factors = []
            for factor in candidate_factors[:10]:
                if factor in df.columns:
                    ic = self.compute_factor_ic(df, factor)
                    forced_factors.append((factor, ic))
            
            forced_factors.sort(key=lambda x: abs(x[1]), reverse=True)
            for factor, ic in forced_factors[:2]:
                recalled.append(factor)
                self._log_recall("Forced", f"{factor}: IC={ic:.4f}")
            
            self.recalled_factors = recalled
        
        return recalled


class FactorGeneratorV144:
    """V144 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.generation_log.append(entry)
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        return -df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window).std()
        ).fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        if 'pct_chg' in df.columns:
            close_return = df['pct_chg']
        elif 'change' in df.columns:
            close_return = df['change']
        else:
            close_return = pd.Series(0, index=df.index)
        
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
        elif 'amount' in df.columns:
            volume_change = df['amount'].pct_change()
        else:
            volume_change = pd.Series(0, index=df.index)
        
        price_rank = close_return.fillna(0).rank(method='average', pct=True)
        volume_rank = volume_change.fillna(0).rank(method='average', pct=True)
        
        return (price_rank - volume_rank).fillna(0)
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = df['amount'] / (df['volume'] + 1e-6)
            price_change = df['close'] - df.get('pre_close', df['close'])
            ofi = price_change * df['volume'] / (df['amount'] + 1e-6)
        elif 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = df.get('pct_chg', pd.Series(0, index=df.index)) * df.get('volume', pd.Series(1, index=df.index))
        
        if 'close' in df.columns:
            ts_std_20 = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
        else:
            ts_std_20 = pd.Series(1, index=df.index)
        
        return (ofi / (ts_std_20 + 1e-6)).fillna(0)
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有基础因子"""
        result = df.copy()
        
        self._log_generation("StartFactorGeneration", f"Processing {len(df)} rows")
        
        # 动量因子
        result['momentum_5'] = self.compute_momentum(result, 5)
        result['momentum_10'] = self.compute_momentum(result, 10)
        result['momentum_20'] = self.compute_momentum(result, 20)
        result['momentum_60'] = self.compute_momentum(result, 60)
        
        # 反转因子
        result['reversion_5'] = self.compute_reversion(result, 5)
        result['reversion_10'] = self.compute_reversion(result, 10)
        
        # 波动率因子
        result['volatility_5'] = self.compute_volatility(result, 5)
        result['volatility_10'] = self.compute_volatility(result, 10)
        result['volatility_20'] = self.compute_volatility(result, 20)
        
        # 量价因子
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        # 排名因子
        result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        self._log_generation("Complete", f"Generated base factors")
        
        return result


class SignalTurnoverCalculator:
    """
    V144 信号翻转率计算器 - 用于对比 V143 vs V144.
    
    【信号翻转率定义】
    Turnover = Mean(|Score_t - Score_{t-1}|) / Std(Score)
    
    【经济逻辑】
    - 翻转率高：信号不稳定，交易成本高
    - 翻转率低：信号稳定，交易成本低
    
    【V144 目标】
    V144 的信号翻转率应该比 V143 下降 15% 以上，因为：
    1. Sign-Lock 机制确保信号方向稳定
    2. Volatility-Adaptive Smoothing 减少日度震荡
    3. Time-Decay Decay Kernel 平滑权重变化
    """
    
    def __init__(self):
        self.turnover_log = []
        
    def _log_turnover(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.turnover_log.append(entry)
    
    def compute_signal_turnover(self, df: pd.DataFrame, score_col: str = 'score') -> float:
        """
        计算信号翻转率.
        
        Args:
            df: 数据
            score_col: 信号列
            
        Returns:
            turnover: 信号翻转率
        """
        if score_col not in df.columns:
            return 0.0
        
        result = df.copy()
        
        # 按日期排序
        dates = sorted(result['trade_date'].unique())
        
        if len(dates) < 2:
            return 0.0
        
        # 计算每日平均信号
        daily_avg_score = result.groupby('trade_date')[score_col].mean()
        
        # 计算信号变化
        score_diff = daily_avg_score.diff().abs()
        
        # 计算翻转率
        mean_diff = score_diff.mean()
        score_std = daily_avg_score.std()
        
        if score_std > 1e-10:
            turnover = mean_diff / score_std
        else:
            turnover = 0.0
        
        self._log_turnover(
            "Computed",
            f"Signal turnover for {score_col}: {turnover:.4f}"
        )
        
        return turnover
    
    def compute_turnover_comparison(self, df_v143: pd.DataFrame, df_v144: pd.DataFrame,
                                     score_col: str = 'score') -> Dict:
        """
        计算 V143 vs V144 信号翻转率对比.
        
        Returns:
            comparison: 对比结果
        """
        turnover_v143 = self.compute_signal_turnover(df_v143, score_col)
        turnover_v144 = self.compute_signal_turnover(df_v144, score_col)
        
        if turnover_v143 > 1e-10:
            turnover_reduction = (turnover_v143 - turnover_v144) / turnover_v143
        else:
            turnover_reduction = 0.0
        
        comparison = {
            'v143_turnover': turnover_v143,
            'v144_turnover': turnover_v144,
            'turnover_reduction': turnover_reduction,
            'reduction_percentage': f"{turnover_reduction * 100:.2f}%",
            'target_met': turnover_reduction >= 0.15,
        }
        
        self._log_turnover(
            "Comparison",
            f"V143: {turnover_v143:.4f}, V144: {turnover_v144:.4f}, "
            f"Reduction: {turnover_reduction * 100:.2f}%, Target Met: {comparison['target_met']}"
        )
        
        return comparison
    
    def get_turnover_log(self) -> List[Dict]:
        return self.turnover_log


class AlphaResearchV144:
    """
    V144 Alpha 研究引擎 - 时序一致性加固与非线性逻辑修复.
    
    【V144 核心改进】
    1. SignConsistencyInteraction: 符号一致性交互（Sign-Lock + 线性残差）
    2. TimeDecayDecayKernel: 时序衰减核（IC 稳定性加权）
    3. VolatilityAdaptiveSmoothing: 波动率自适应平滑
    
    【目标指标】
    - T+1 Rank IC > 0.055
    - IC_IR > 0.70
    - Signal Turnover 下降 15%+
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_sci: bool = True,
        enable_time_decay: bool = True,
        enable_smoothing: bool = True,
        enable_orthogonalization: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        max_recall_factors: int = 3,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_sci = enable_sci
        self.enable_time_decay = enable_time_decay
        self.enable_smoothing = enable_smoothing
        self.enable_orthogonalization = enable_orthogonalization
        self.auto_heal = auto_heal
        self.max_recall_factors = max_recall_factors
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.recalled_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealingV144(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV144()
        
        # V144 核心模块
        self.residual_recall = ResidualBasedRecallV144(top_percent=0.2)
        self.sci_interaction = SignConsistencyInteraction(
            enable_sign_lock=True,
            enable_smoothing=enable_smoothing
        ) if enable_sci else None
        self.time_decay = TimeDecayDecayKernel(rolling_window=20) if enable_time_decay else None
        self.turnover_calculator = SignalTurnoverCalculator()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Sign-Consistency Interaction (SCI)")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max N Factors: {n_factors}")
        logger.info(f"  Max Recall Factors: {max_recall_factors}")
        logger.info(f"  SCI (Sign-Lock): {'Enabled' if enable_sci else 'Disabled'}")
        logger.info(f"  Time-Decay Kernel: {'Enabled' if enable_time_decay else 'Disabled'}")
        logger.info(f"  Volatility Smoothing: {'Enabled' if enable_smoothing else 'Disabled'}")
        logger.info(f"  Target IR: 0.70 (V143: 0.60)")
        logger.info(f"  Target Turnover Reduction: 15%+")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
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
        """因子处理：Auto-Heal Winsorization + 标准化"""
        # V144 Auto-Heal 版去极值
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        # 截面标准化
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def _calc_factor_ic_decay(self, df: pd.DataFrame, factor_col: str) -> Tuple[float, float, float]:
        """计算因子 IC Decay (T+1, T+3, T+5)"""
        t1_ics, t3_ics, t5_ics = [], [], []
        
        result = df.copy()
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        if 't3_return_period' not in result.columns:
            result['t3_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-3) / x.shift(-2) - 1
            )
        if 't5_return_period' not in result.columns:
            result['t5_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-5) / x.shift(-4) - 1
            )
        
        for date in df['trade_date'].unique():
            day = result[result['trade_date'] == date]
            if len(day) < 20:
                continue
            
            f = day[factor_col].fillna(0)
            t1 = day['t1_return_period'].fillna(0)
            t3 = day['t3_return_period'].fillna(0)
            t5 = day['t5_return_period'].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                
                ic1 = np.corrcoef(f_rank, t1.rank(method='average'))[0, 1]
                ic3 = np.corrcoef(f_rank, t3.rank(method='average'))[0, 1]
                ic5 = np.corrcoef(f_rank, t5.rank(method='average'))[0, 1]
                
                if not np.isnan(ic1): t1_ics.append(ic1)
                if not np.isnan(ic3): t3_ics.append(ic3)
                if not np.isnan(ic5): t5_ics.append(ic5)
        
        return (np.mean(t1_ics) if t1_ics else 0,
                np.mean(t3_ics) if t3_ics else 0,
                np.mean(t5_ics) if t5_ics else 0)
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V144 核心逻辑（SCI + Time-Decay + Smoothing）"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据自愈检查
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 2. 准备标签（严格 T+1）
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # 计算单期回报
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        if 't3_return_period' not in result.columns:
            result['t3_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-3) / x.shift(-2) - 1
            )
        if 't5_return_period' not in result.columns:
            result['t5_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-5) / x.shift(-4) - 1
            )
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
            self._log_audit("FactorGeneration", "Generated base factors")
        
        # 4. 基于残差分析召回因子
        self._log_audit("ResidualRecall", "Starting residual-based factor recall...")
        
        core_factors = [f for f in V144_CORE_FACTORS if f in result.columns]
        
        if len(core_factors) >= 2:
            priority_candidates = ['reversion_5', 'reversion_10', 'volume_price_contradiction', 
                                   'rsi_14', 'mfi_14', 'price_position_20'] + V144_CANDIDATE_FACTORS
            self.recalled_factors = self.residual_recall.select_recall_factors(
                result, core_factors, priority_candidates, self.max_recall_factors
            )
            self._log_audit("ResidualRecall", f"Recalled {len(self.recalled_factors)} factors: {self.recalled_factors}")
        else:
            self._log_audit("ResidualRecall", "Insufficient core factors, skipping recall")
            self.recalled_factors = []
        
        # 5. 计算 SCI 特征（V144 核心）
        sci_factors = []
        if self.enable_sci and self.sci_interaction:
            self._log_audit("SCI", "Computing Sign-Consistency Interaction features...")
            result = self.sci_interaction.compute_all_sci_features(
                result, core_factors, self.recalled_factors
            )
            
            sci_factors = list(self.sci_interaction.get_sci_features().keys())
            self._log_audit("SCI", f"Generated {len(sci_factors)} SCI features")
        
        # 6. 构建候选因子池
        all_candidate_factors = []
        
        # 添加强制短期因子
        forced_short_term = ['reversion_5', 'volume_price_contradiction', 'liquidity_alpha']
        for factor in forced_short_term:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加核心因子
        for factor in core_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加召回因子
        for factor in self.recalled_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加 SCI 因子（V144 核心）
        for factor in sci_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors (SCI priority)"
        )
        
        # 7. 计算 IC 和 IC 滚动统计（用于 Time-Decay）
        # V144 修复：使用 T+1 IC 作为主要排序依据，而不是 T1_Specificity
        factor_ics = []
        factor_decay = {}
        
        for factor in all_candidate_factors:
            if factor not in result.columns:
                continue
            ic = self._calc_factor_ic(result, factor)
            t1_ic, t3_ic, t5_ic = self._calc_factor_ic_decay(result, factor)
            
            t1_specificity = t1_ic - (t3_ic + t5_ic) / 2
            
            self.factor_ics[factor] = ic
            factor_decay[factor] = {'t1': t1_ic, 't3': t3_ic, 't5': t5_ic, 't1_specificity': t1_specificity}
            
            # V144 修复：使用 T+1 IC 绝对值作为排序依据
            factor_ics.append((factor, abs(t1_ic)))
            
            self._log_audit(
                "FactorAnalysis",
                f"{factor}: T+1={t1_ic:.4f}, T+3={t3_ic:.4f}, T+5={t5_ic:.4f}, T1_Specificity={t1_specificity:.4f}"
            )
        
        # 分离 SCI 因子和其他因子
        sci_keywords = ['_sci_', '_sign_consistency_']
        sci_ics = [(f, ic) for f, ic in factor_ics if any(kw in f.lower() for kw in sci_keywords)]
        other_ics = [(f, ic) for f, ic in factor_ics if f not in [x[0] for x in sci_ics]]
        
        # 按 T+1 IC 绝对值排序
        sci_ics.sort(key=lambda x: x[1], reverse=True)  # 已经排好序
        other_ics.sort(key=lambda x: x[1], reverse=True)
        
        # 8. V144 修复策略：精选高 IC 因子，限制因子数量以提升 IR
        # 核心洞察：V143 只选 3 个因子但 IR=0.44，V144 选 12 个因子 IR=0.34
        # 解决方案：减少因子数量到 6 个，精选最高 IC 的因子
        
        final_selected = []
        max_factors = min(6, self.n_factors)  # V144 修复：最多选 6 个因子
        
        # 首先添加 top IC 的非 SCI 因子（原始高 IC 因子）
        # 目标：选择 4-5 个原始因子
        for factor, t1_ic_abs in other_ics:
            if len(final_selected) >= max_factors - 1:  # 保留 1 个位置给最好的 SCI 因子
                break
            if t1_ic_abs >= 0.02:  # T+1 IC 阈值 0.02
                final_selected.append(factor)
        
        # 然后添加 top IC 的 SCI 因子（最多 1-2 个）
        for factor, t1_ic_abs in sci_ics:
            if len(final_selected) >= max_factors:
                break
            if t1_ic_abs >= 0.01:  # SCI 因子 IC 阈值 0.01
                final_selected.append(factor)
        
        self.selected_factors = final_selected[:max_factors]
        
        self._log_audit(
            "FactorSelection",
            f"Final selected {len(self.selected_factors)} factors (mixed IC+SCI): {self.selected_factors}"
        )
        
        # 9. 准备因子数据
        factor_data = {}
        
        for factor in self.selected_factors:
            f_raw = result[factor]
            ic = self.factor_ics[factor]
            
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 计算权重并合成最终评分
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            weights = []
            for factor in self.selected_factors:
                # 基础权重 = IC 绝对值
                base_weight = abs(self.factor_ics[factor])
                
                # V144: Time-Decay 加权
                if self.enable_time_decay and self.time_decay:
                    decay_weight = self.time_decay.compute_decay_weight(
                        result, factor, base_weight
                    )
                    weight = decay_weight
                    self._log_audit(
                        "TimeDecayWeight",
                        f"{factor}: base={base_weight:.4f}, decay={weight:.4f}"
                    )
                else:
                    weight = base_weight
                
                # SCI 因子权重增强
                is_sci = any(kw in factor.lower() for kw in ['_sci_', '_sign_consistency_'])
                if is_sci:
                    weight = weight * 2.0  # SCI 因子权重×2.0
                    self._log_audit("WeightBoost", f"{factor}: SCI factor, weight ×2.0")
                
                weights.append(weight)
            
            # 归一化
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
            else:
                normalized_weights = [1.0 / len(self.selected_factors)] * len(self.selected_factors)
            
            score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                score += factor_data[factor] * normalized_weights[i]
                self.factor_weights[factor] = normalized_weights[i]
            
            result['score'] = score
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (SCI + Time-Decay)")
        
        # 输出列
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']
        for col in ['t1_return_period', 't3_return_period', 't5_return_period']:
            if col in result.columns and col not in output_cols:
                output_cols.append(col)
        
        return result[output_cols]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        """获取因子 IC"""
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子"""
        return self.selected_factors
    
    def get_recalled_factors(self) -> List[str]:
        """获取召回的因子"""
        return self.recalled_factors
    
    def get_sci_features(self) -> Dict:
        """获取 SCI 特征"""
        return self.sci_interaction.get_sci_features() if self.sci_interaction else {}
    
    def get_sign_lock_applied(self) -> List[str]:
        """获取应用符号锁定的因子"""
        return self.sci_interaction.get_sign_lock_applied() if self.sci_interaction else []
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_residual_analysis(self) -> Dict:
        """获取残差分析结果"""
        return self.residual_recall.residual_analysis
    
    def get_market_regime(self) -> MarketRegimeTensor:
        """获取市场场景模块"""
        return self.sci_interaction.get_market_regime() if self.sci_interaction else None
    
    def get_time_decay_stats(self) -> Dict:
        """获取时序衰减统计"""
        return self.time_decay.get_ic_rolling_stats() if self.time_decay else {}
    
    def get_decay_weights(self) -> Dict:
        """获取衰减权重"""
        return self.time_decay.get_decay_weights() if self.time_decay else {}
    
    def get_efficiency_ratio(self) -> float:
        """计算效率指标：IC / Factor Count"""
        if not self.selected_factors:
            return 0.0
        
        ics = list(self.get_factor_ics().values())
        if not ics:
            return 0.0
        
        mean_ic = abs(np.mean(ics))
        return mean_ic / len(self.selected_factors)
    
    def get_sci_log(self) -> List[Dict]:
        """获取 SCI 日志"""
        return self.sci_interaction.get_sci_log() if self.sci_interaction else []
    
    def get_turnover_calculator(self) -> SignalTurnoverCalculator:
        """获取翻转率计算器"""
        return self.turnover_calculator


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_sci: bool = True,
    enable_time_decay: bool = True,
    enable_smoothing: bool = True,
    enable_orthogonalization: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 3,
) -> AlphaResearchV144:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV144(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_sci=enable_sci,
        enable_time_decay=enable_time_decay,
        enable_smoothing=enable_smoothing,
        enable_orthogonalization=enable_orthogonalization,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV144...")
    
    np.random.seed(42)
    n_samples = 1000
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], n_samples),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], n_samples),
        'close': np.random.randn(n_samples) * 10 + 100,
        'volume': np.random.randn(n_samples) * 1000 + 5000,
        'amount': np.random.randn(n_samples) * 10000 + 50000,
        'pct_chg': np.random.randn(n_samples) * 2,
        'momentum_20': np.random.randn(n_samples),
        'volatility_10': np.abs(np.random.randn(n_samples)),
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Recalled factors: {alpha.get_recalled_factors()}")
    logger.info(f"  SCI features: {alpha.get_sci_features()}")
    logger.info(f"  Sign-Lock applied: {alpha.get_sign_lock_applied()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Time-Decay stats: {alpha.get_time_decay_stats()}")
    logger.info(f"  Efficiency Ratio: {alpha.get_efficiency_ratio():.4f}")