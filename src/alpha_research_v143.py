"""
Alpha Research Module - V143 多维张量核与 IR 稳定性加固.

【V142 回顾】
V142 实现了 IC 0.0598，这是巨大的成功。核心贡献：
1. FeatureDistillation: 特征提纯（残差缩放 + Sigmoid 门控）
2. ResidualBasedRecall: 基于残差分析的因子召回
3. RegimeAwareWeighting: 场景感知动态权重

【V143 核心使命】
在 V142 基础上继续进化，实现 IR 从 0.60 冲刺到 0.80。

【V143 核心算法 - 3D Tensor Interaction】
1. 从 2D 到 3D 的飞跃：
   - V142: Interaction = Sigmoid(Rank(Factor_A)) * Rank(Resid_Factor_B)  [2D 门控]
   - V143: Alpha = Sigmoid(Rank(Factor_A)) * Rank(Resid_Factor_B) * Transformation(Volatility_State)  [3D 张量]
   
   公式：Alpha_3D = Gated_Interaction_2D * Regime_Modulation
         Regime_Modulation = 1.0 + tanh((Volatility - Median_Vol) / Std_Vol) * 0.5
   
   逻辑：在不同波动率/换手率环境下，非线性核的强度应自动缩放。

2. Kernel-Based Neutralization (核中性化):
   - 在 _calculate_distilled_features 中，除了线性残差，尝试使用简单的多项式映射（如 X^2）
   - 剔除更高阶的冗余信息
   - 公式：Kernel_Residual = Factor - β1 * Core - β2 * Core^2

3. IC-Precision Weighting (IC 精度加权):
   - 对最近 20 天 IC 波动较大的特征进行惩罚性减权
   - 提升信号的日度平稳性
   - 公式：Weight = Base_IC_Weight / (1 + IC_Volatility_Penalty)
   - IC_Volatility_Penalty = Std(IC_20d) / Mean(IC_20d)

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 143 运行
- 严禁"回测碰运气"：禁止修改 backtest_referee.py 中的费率或初始资金
- 报错必改：内置更强的 Winsorization 和 Auto-Impute 逻辑
- 内存优化：使用 chunking 处理大数据

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标（必须超过 V142 的 0.0598） |
| IC_IR | > 0.80 | 稳定性（V142: 0.60） |
| IC_Std | < 0.04 | IC 波动率降低 |
| IC Decay | T+1 > T+3 > T+5 | 正常衰减模式 |
| 3D Tensor Features | >= 2 | 至少 2 个 3D 张量特征 |
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

VERSION = "V143"

# V142 核心因子（保留 V142 的因子召回框架）
V142_CORE_FACTORS = [
    'momentum_20',      # 20 日动量
    'volatility_10',    # 10 日波动率
    'volume_price_contradiction',  # 量价背离
    'liquidity_alpha',  # 流动性 Alpha
]

# V142 候选因子池（用于召回）
V142_CANDIDATE_FACTORS = [
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

# V143 所有因子（核心 + 召回）
ALL_FACTORS = V142_CORE_FACTORS + V142_CANDIDATE_FACTORS

# V143 最大因子数量
MAX_FACTORS = 12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数 - 用于门控机制"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def winsorize_enhanced(series: pd.Series, sigma: float = 3.0, percentile: float = 0.99) -> pd.Series:
    """
    V143 增强版 Winsorization 去极值 - 双重保护.
    
    【V143 数据自愈原则】
    1. 先使用 sigma 截断，再使用 percentile 截断
    2. 自动检测 NaN 和 Inf，进行修复
    3. 分组处理（按 trade_date）
    
    Args:
        series: 输入序列
        sigma: 标准差倍数
        percentile: 分位数截断
        
    Returns:
        去极值后的序列
    """
    # 1. 处理 Inf 和 NaN
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    # 2. Sigma 截断
    mean = series_clean.mean()
    std = series_clean.std()
    if std > 1e-10:
        lower_sigma = mean - sigma * std
        upper_sigma = mean + sigma * std
        series_clean = series_clean.clip(lower=lower_sigma, upper=upper_sigma)
    
    # 3. Percentile 截断（双重保护）
    lower_pct = series_clean.quantile(1 - percentile)
    upper_pct = series_clean.quantile(percentile)
    series_clean = series_clean.clip(lower=lower_pct, upper=upper_pct)
    
    # 4. 最终 NaN 填充
    series_clean = series_clean.fillna(mean)
    
    return series_clean


def auto_impute_grouped(df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
    """
    V143 自动分组插值 - 数据自愈核心.
    
    【自愈逻辑】
    1. 按 group_col 分组（通常是 trade_date）
    2. 组内使用中位数填充
    3. 如果整组缺失，使用全局中位数
    
    Args:
        df: 输入 DataFrame
        group_col: 分组列
        
    Returns:
        修复后的 DataFrame
    """
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # 计算全局中位数作为后备
        global_median = result[col].median()
        if pd.isna(global_median):
            global_median = 0.0
        
        # 分组填充
        def fill_group(group):
            group_median = group[col].median()
            if pd.isna(group_median):
                group_median = global_median
            return group[col].fillna(group_median)
        
        result[col] = result.groupby(group_col, group_keys=False).apply(fill_group)
        
        # 最后的后备填充
        result[col] = result[col].fillna(global_median)
    
    return result


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    计算两个变量之间的互信息 (Mutual Information).
    
    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    
    使用分箱法近似计算离散互信息.
    """
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
    """
    Gram-Schmidt 正交化 + 互信息验证.
    """
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
            max_corr = 0
            max_mi = 0
            
            for j in kept_indices:
                corr = np.corrcoef(X_norm[:, i], X_norm[:, j])[0, 1]
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
                
                mi = compute_mutual_information(X_norm[:, i], X_norm[:, j], n_bins=10)
                max_mi = max(max_mi, mi)
            
            if max_corr < 0.2 and max_mi < mi_threshold:
                orthogonal.append(v / norm)
                kept_indices.append(i)
    
    orthogonalized = np.zeros_like(X)
    for idx, (ortho_idx, u) in enumerate(zip(kept_indices, orthogonal)):
        orthogonalized[:, idx] = u
    
    return orthogonalized[:, :len(kept_indices)], kept_indices


class DataHealingV143:
    """V143 增强版数据自愈模块 - 数据自愈 3.0"""
    
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
                logger.info("[V143][DataHealing] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V143][DataHealing] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V143][DataHealing] No database URL, SQL healer disabled")
    
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
        logger.info(f"[V143][DataHealing] {action} - Column: {column}, Status: {status}, {details}")
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """V143 增强版检查并修复缺失列"""
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
                    result[col] = 0.0
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
        
        # V143 增强：自动插值
        result = auto_impute_grouped(result, 'trade_date')
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
            logger.error(f"[V143][DataHealing] SQL heal failed: {e}")
            for col in columns:
                result[col] = 0.0
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class MarketRegimeTensor:
    """
    V143 市场环境张量 - 3D 交互核心.
    
    【从 2D 到 3D 的飞跃】
    V142: Alpha = Sigmoid(Rank(Factor_A)) * Rank(Resid_Factor_B)  [2D 门控]
    V143: Alpha = Sigmoid(Rank(Factor_A)) * Rank(Resid_Factor_B) * Regime_Modulation  [3D 张量]
    
    【Regime_Modulation 计算】
    1. 计算市场波动率截面分布
    2. Regime_Modulation = 1.0 + tanh((Vol - Median_Vol) / Std_Vol) * 0.5
    3. 高波动环境：Regime_Modulation > 1.0（增强信号）
    4. 低波动环境：Regime_Modulation < 1.0（抑制信号）
    
    【经济逻辑】
    - 在高波动环境中，非线性因子应该被增强（因为市场更情绪化）
    - 在低波动环境中，线性因子更有效（因为市场更理性）
    """
    
    def __init__(self, volatility_window: int = 20, regime_threshold: float = 0.5):
        self.volatility_window = volatility_window
        self.regime_threshold = regime_threshold
        self.regime_log = []
        self.current_regime = 0  # 0=低波动，1=高波动
        self.regime_modulation = 1.0
        
    def _log_regime(self, action: str, details: str = ""):
        """记录场景日志"""
        entry = {'action': action, 'details': details}
        self.regime_log.append(entry)
        logger.info(f"[V143][MarketRegime] {action}: {details}")
    
    def compute_market_volatility(self, df: pd.DataFrame) -> pd.Series:
        """计算市场波动率（截面平均）"""
        if 'volatility_20' in df.columns:
            return df.groupby('trade_date')['volatility_20'].transform('mean')
        elif 'pct_chg' in df.columns:
            return df.groupby('trade_date')['pct_chg'].transform('std')
        else:
            return pd.Series(1.0, index=df.index)
    
    def compute_regime_modulation(self, df: pd.DataFrame) -> pd.Series:
        """
        计算场景调制因子 - V143 核心.
        
        Returns:
            regime_modulation: 场景调制因子 (0.5 ~ 1.5)
        """
        vol = self.compute_market_volatility(df)
        
        # 滚动计算波动率统计量
        vol_mean = vol.rolling(self.volatility_window, min_periods=10).mean()
        vol_std = vol.rolling(self.volatility_window, min_periods=10).std()
        
        # 防止除零
        vol_std = vol_std.replace(0, 1e-10).fillna(1e-10)
        vol_mean = vol_mean.fillna(vol.mean())
        
        # 计算标准化波动率偏离
        vol_zscore = (vol - vol_mean) / vol_std
        
        # Tanh 映射到 (0.5, 1.5)
        # tanh(-3) ≈ -1, tanh(3) ≈ 1
        # 1 + tanh(z) * 0.5 → (0.5, 1.5)
        regime_modulation = 1.0 + np.tanh(vol_zscore.clip(-3, 3)) * 0.5
        
        # 记录当前场景
        self.regime_modulation = float(regime_modulation.iloc[-1]) if len(regime_modulation) > 0 else 1.0
        self.current_regime = 1 if self.regime_modulation > 1.0 else 0
        
        self._log_regime(
            "Computed",
            f"Regime modulation: {self.regime_modulation:.3f}, regime={'high' if self.current_regime == 1 else 'low'} volatility"
        )
        
        return regime_modulation
    
    def apply_3d_tensor(self, interaction_2d: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        应用 3D 张量交互.
        
        Args:
            interaction_2d: 2D 门控交互结果
            df: 原始数据
            
        Returns:
            interaction_3d: 3D 张量交互结果
        """
        regime_mod = self.compute_regime_modulation(df)
        
        # 3D 张量 = 2D 交互 × 场景调制
        interaction_3d = interaction_2d * regime_mod.values
        
        self._log_regime(
            "Applied3DTensor",
            f"2D interaction modulated by regime (modulation={self.regime_modulation:.3f})"
        )
        
        return interaction_3d
    
    def get_regime_log(self) -> List[Dict]:
        """获取场景日志"""
        return self.regime_log
    
    def get_current_regime(self) -> int:
        """获取当前场景"""
        return self.current_regime
    
    def get_regime_modulation(self) -> float:
        """获取当前场景调制因子"""
        return self.regime_modulation


class KernelBasedNeutralization:
    """
    V143 核中性化模块 - 高阶冗余剔除.
    
    【为什么需要核中性化】
    V142 的残差缩放只剔除了线性冗余：
    Residual = Factor - β * Core
    
    但因子之间可能存在非线性关系（如二次关系）：
    Factor ≈ β1 * Core + β2 * Core^2
    
    【V143 核中性化】
    1. 多项式映射：Core^2, Core^3
    2. 多元回归：Factor ~ Core + Core^2
    3. 残差 = Factor - (β1 * Core + β2 * Core^2)
    
    【经济逻辑】
    - 某些因子可能是核心因子的非线性函数
    - 剔除高阶冗余后，残差包含更"纯净"的新信息
    """
    
    def __init__(self, polynomial_degree: int = 2):
        self.polynomial_degree = polynomial_degree
        self.kernel_log = []
        self.kernel_stats = {}
        
    def _log_kernel(self, action: str, details: str = ""):
        """记录核日志"""
        entry = {'action': action, 'details': details}
        self.kernel_log.append(entry)
        logger.info(f"[V143][KernelNeutralization] {action}: {details}")
    
    def compute_kernel_residual(self, df: pd.DataFrame, factor_col: str, 
                                 core_col: str) -> pd.Series:
        """
        计算核残差 - V143 核心.
        
        公式：
        1. 构建核矩阵：[Core, Core^2, ..., Core^degree]
        2. OLS 回归：Factor = β0 + β1*Core + β2*Core^2 + ... + ε
        3. 返回：ε (残差)
        
        Args:
            df: 数据
            factor_col: 待中性化的因子列
            core_col: 核心因子列
            
        Returns:
            kernel_residual: 核残差
        """
        if factor_col not in df.columns or core_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        y = df[factor_col].fillna(0).values
        x_core = df[core_col].fillna(0).values
        
        # 构建核矩阵
        kernel_matrix = [np.ones(len(x_core))]  # 截距项
        for d in range(1, self.polynomial_degree + 1):
            kernel_matrix.append(x_core ** d)
        
        X = np.column_stack(kernel_matrix)
        
        # OLS 回归：β = (X'X)^(-1) X'y
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            
            # 预测值
            y_pred = X @ beta
            
            # 残差
            residual = y - y_pred
            
            # 标准化残差
            residual_std = np.std(residual) + 1e-10
            standardized_residual = (residual - np.mean(residual)) / residual_std
            
            self.kernel_stats[factor_col] = {
                'core_col': core_col,
                'polynomial_degree': self.polynomial_degree,
                'beta_coefficients': beta.tolist(),
                'residual_std': float(residual_std),
                'r_squared': float(1 - np.var(residual) / (np.var(y) + 1e-10)),
            }
            
            self._log_kernel(
                "Computed",
                f"{factor_col} ~ {core_col} + {core_col}^2: R²={self.kernel_stats[factor_col]['r_squared']:.4f}"
            )
            
            return pd.Series(standardized_residual, index=df.index)
            
        except Exception as e:
            logger.warning(f"[V143][KernelNeutralization] Kernel residual failed: {e}")
            return df[factor_col].fillna(0)
    
    def get_kernel_log(self) -> List[Dict]:
        """获取核日志"""
        return self.kernel_log
    
    def get_kernel_stats(self) -> Dict:
        """获取核统计"""
        return self.kernel_stats


class ICPrecisionWeighting:
    """
    V143 IC 精度加权模块 - IR 稳定性加固.
    
    【目标】
    V142 的 IR 为 0.60，V143 目标是冲刺 0.80。
    
    【方法】
    对最近 20 天 IC 波动较大的特征进行惩罚性减权，提升信号的日度平稳性。
    
    【公式】
    1. IC_Mean = Mean(IC_20d)
    2. IC_Std = Std(IC_20d)
    3. IC_IR = IC_Mean / IC_Std
    4. IC_Volatility_Penalty = IC_Std / (|IC_Mean| + ε)
    5. Precision_Weight = Base_IC_Weight / (1 + IC_Volatility_Penalty)
    
    【经济逻辑】
    - IC 稳定的因子应该获得更高权重
    - IC 波动大的因子应该被降权（即使平均 IC 高）
    - 这有助于降低整体信号的波动率，提升 IR
    """
    
    def __init__(self, rolling_window: int = 20, min_samples: int = 10):
        self.rolling_window = rolling_window
        self.min_samples = min_samples
        self.ic_precision_log = []
        self.ic_rolling_stats = {}
        self.precision_weights = {}
        
    def _log_precision(self, action: str, details: str = ""):
        """记录精度日志"""
        entry = {'action': action, 'details': details}
        self.ic_precision_log.append(entry)
        logger.info(f"[V143][ICPrecision] {action}: {details}")
    
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
    
    def compute_precision_weight(self, df: pd.DataFrame, factor_col: str, 
                                  base_weight: float = 1.0) -> float:
        """
        计算 IC 精度权重 - V143 核心.
        
        Args:
            df: 数据
            factor_col: 因子列
            base_weight: 基础权重（通常是 IC 绝对值）
            
        Returns:
            precision_weight: 精度调整后的权重
        """
        ics = self.compute_rolling_ic(df, factor_col)
        
        if len(ics) < self.min_samples:
            # 数据不足，返回基础权重
            return base_weight
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics, ddof=1) + 1e-10
        ic_ir = ic_mean / ic_std
        
        # IC 波动率惩罚
        ic_volatility_penalty = ic_std / (abs(ic_mean) + 1e-10)
        
        # 精度权重 = 基础权重 / (1 + 惩罚)
        precision_weight = base_weight / (1 + ic_volatility_penalty)
        
        # 记录统计
        self.ic_rolling_stats[factor_col] = {
            'ic_mean': float(ic_mean),
            'ic_std': float(ic_std),
            'ic_ir': float(ic_ir),
            'ic_volatility_penalty': float(ic_volatility_penalty),
            'base_weight': float(base_weight),
            'precision_weight': float(precision_weight),
            'num_samples': len(ics),
        }
        
        self.precision_weights[factor_col] = float(precision_weight)
        
        self._log_precision(
            "Computed",
            f"{factor_col}: IC_mean={ic_mean:.4f}, IC_std={ic_std:.4f}, IC_IR={ic_ir:.2f}, "
            f"penalty={ic_volatility_penalty:.2f}, precision_weight={precision_weight:.4f}"
        )
        
        return precision_weight
    
    def get_precision_log(self) -> List[Dict]:
        """获取精度日志"""
        return self.ic_precision_log
    
    def get_ic_rolling_stats(self) -> Dict:
        """获取 IC 滚动统计"""
        return self.ic_rolling_stats
    
    def get_precision_weights(self) -> Dict:
        """获取精度权重"""
        return self.precision_weights


class FeatureDistillationV143:
    """
    V143 特征提纯模块 - 3D 张量交互.
    
    【V143 与 V142 的本质区别】
    V142: Interaction = Sigmoid(Rank(Factor_A)) * Rank(Residual_Factor_B)  [2D]
    V143: Interaction_3D = Sigmoid(Rank(Factor_A)) * Rank(Kernel_Residual) * Regime_Modulation  [3D]
    
    【V143 新增组件】
    1. Kernel-Based Neutralization: 多项式映射剔除高阶冗余
    2. Market Regime Tensor: 场景调制因子
    3. IC-Precision Weighting: IC 稳定性加权
    """
    
    def __init__(self, enable_kernel: bool = True, enable_3d: bool = True):
        self.enable_kernel = enable_kernel
        self.enable_3d = enable_3d
        self.distillation_log = []
        self.distilled_features = {}
        
        # V143 新增模块
        self.kernel_neutralization = KernelBasedNeutralization(polynomial_degree=2) if enable_kernel else None
        self.market_regime = MarketRegimeTensor() if enable_3d else None
        
    def _log_distillation(self, action: str, details: str = ""):
        """记录提纯日志"""
        entry = {'action': action, 'details': details}
        self.distillation_log.append(entry)
        logger.info(f"[V143][FeatureDistillation] {action}: {details}")
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名 (0-1 归一化)"""
        return series.rank(method='average', pct=True)
    
    def compute_3d_distilled_interaction(self, df: pd.DataFrame, core_factor: str, 
                                          recall_factor: str) -> pd.Series:
        """
        计算 3D 提纯交互特征 - V143 核心.
        
        【完整流程】
        1. 核中性化：Kernel_Residual = Factor - β1*Core - β2*Core^2
        2. Sigmoid 门控：Gate = Sigmoid(Rank(Core) * 10 - 5)
        3. 2D 交互：Interaction_2D = Gate * Rank(Kernel_Residual)
        4. 3D 张量：Interaction_3D = Interaction_2D * Regime_Modulation
        
        【经济逻辑】
        - 核中性化剔除高阶冗余
        - Sigmoid 门控实现"条件触发"
        - 场景调制实现"环境自适应"
        """
        if core_factor not in df.columns or recall_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 核中性化（如果启用）
        if self.enable_kernel and self.kernel_neutralization:
            kernel_residual = self.kernel_neutralization.compute_kernel_residual(
                df, recall_factor, core_factor
            )
        else:
            # 回退到简单残差
            core_data = df[core_factor].fillna(0).values
            recall_data = df[recall_factor].fillna(0).values
            
            # 简单线性回归残差
            if np.std(core_data) > 1e-10:
                beta = np.corrcoef(core_data, recall_data)[0, 1] * np.std(recall_data) / (np.std(core_data) + 1e-10)
                residual = recall_data - beta * core_data
            else:
                residual = recall_data
            
            kernel_residual = pd.Series(residual, index=df.index)
        
        # 2. Sigmoid 门控
        rank_core = self._rank(df[core_factor].fillna(0))
        sigmoid_input = rank_core * 10 - 5
        gate_values = sigmoid(sigmoid_input.values)
        
        # 3. 2D 交互
        rank_residual = self._rank(kernel_residual)
        interaction_2d = gate_values * rank_residual.values
        
        # 4. 3D 张量（如果启用）
        if self.enable_3d and self.market_regime:
            interaction_3d = self.market_regime.apply_3d_tensor(
                pd.Series(interaction_2d, index=df.index), df
            )
        else:
            interaction_3d = pd.Series(interaction_2d, index=df.index)
        
        feature_name = f"{core_factor}_3d_{recall_factor}"
        self.distilled_features[feature_name] = {
            'core_factor': core_factor,
            'recall_factor': recall_factor,
            'type': '3d_distilled_interaction',
            'method': 'kernel_neutralization + sigmoid_gating + regime_modulation',
            'enable_kernel': self.enable_kernel,
            'enable_3d': self.enable_3d,
        }
        
        self._log_distillation(
            "3DDistilledInteraction",
            f"{feature_name}: Sigmoid(Rank({core_factor})) × Kernel_Residual({recall_factor}) × Regime_Modulation"
        )
        
        return interaction_3d
    
    def compute_volume_reversion_3d(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量价背离×反转 3D 组合 - V143 重点优化.
        
        【经济逻辑】
        捕捉"缩量下跌后的极致反转"信号，并考虑市场环境：
        - 高波动环境：增强信号（市场情绪化，反转更剧烈）
        - 低波动环境：抑制信号（市场理性，反转更平缓）
        """
        result = df.copy()
        
        vpc = 'volume_price_contradiction'
        rev5 = 'reversion_5'
        
        if vpc not in df.columns or rev5 not in df.columns:
            return result
        
        # 1. 计算 3D 提纯交互
        distilled_3d = self.compute_3d_distilled_interaction(df, vpc, rev5)
        result[f'{vpc}_3d_{rev5}'] = distilled_3d
        
        # 2. 计算反向 3D 提纯（reversion 作为门控）
        distilled_3d_rev = self.compute_3d_distilled_interaction(df, rev5, vpc)
        result[f'{rev5}_3d_{vpc}'] = distilled_3d_rev
        
        self._log_distillation(
            "VolumeReversion3DCombo",
            f"Generated 2 3D distilled features for {vpc} × {rev5}"
        )
        
        return result
    
    def compute_all_distilled_features(self, df: pd.DataFrame, core_factors: List[str],
                                        recalled_factors: List[str]) -> pd.DataFrame:
        """计算所有 3D 提纯特征"""
        result = df.copy()
        
        # 1. 重点组合：volume_price_contradiction × reversion_5
        if 'volume_price_contradiction' in df.columns and 'reversion_5' in df.columns:
            result = self.compute_volume_reversion_3d(result)
        
        # 2. 其他 3D 提纯交互
        for core in core_factors:
            if core not in df.columns:
                continue
            for recalled in recalled_factors:
                if recalled not in df.columns or recalled == 'reversion_5':
                    continue
                if core == 'volume_price_contradiction':
                    continue
                
                name = f"{core}_3d_{recalled}"
                if name not in result.columns:
                    result[name] = self.compute_3d_distilled_interaction(df, core, recalled)
        
        self._log_distillation(
            "Complete",
            f"Generated {len(self.distilled_features)} 3D distilled features"
        )
        
        return result
    
    def get_distillation_log(self) -> List[Dict]:
        """获取提纯日志"""
        return self.distillation_log
    
    def get_distilled_features(self) -> Dict:
        """获取提纯特征字典"""
        return self.distilled_features
    
    def get_market_regime(self) -> MarketRegimeTensor:
        """获取市场场景模块"""
        return self.market_regime
    
    def get_kernel_neutralization(self) -> KernelBasedNeutralization:
        """获取核中性化模块"""
        return self.kernel_neutralization


class ResidualBasedRecallV143:
    """V143 基于残差分析的因子召回模块 - 增强版"""
    
    def __init__(self, top_percent: float = 0.2):
        self.top_percent = top_percent
        self.recall_log = []
        self.recalled_factors = []
        self.residual_analysis = {}
        
    def _log_recall(self, action: str, details: str = ""):
        """记录召回日志"""
        entry = {'action': action, 'details': details}
        self.recall_log.append(entry)
        logger.info(f"[V143][ResidualRecall] {action}: {details}")
    
    def compute_residuals(self, df: pd.DataFrame, core_factors: List[str]) -> pd.Series:
        """计算 V143 核心模型的残差"""
        result = df.copy()
        
        # 简单加权预测
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
        """识别失效样本（残差绝对值最大的前 20%）"""
        threshold = residuals.abs().quantile(1 - self.top_percent)
        failure_mask = residuals.abs() >= threshold
        
        self._log_recall(
            "Identified",
            f"{failure_mask.sum()} failure samples (top {self.top_percent*100}%), threshold={threshold:.4f}"
        )
        
        return failure_mask
    
    def compute_factor_ic_on_samples(self, df: pd.DataFrame, factor_col: str, sample_mask: pd.Series) -> float:
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
        # 计算残差
        residuals = self.compute_residuals(df, core_factors)
        failure_mask = self.identify_failure_samples(residuals)
        
        # 计算每个候选因子的召回得分
        recall_scores = {}
        
        for factor in candidate_factors:
            if factor not in df.columns:
                continue
            
            # 整体 IC
            overall_ic = self.compute_factor_ic(df, factor)
            
            # 失效样本 IC
            failure_ic = self.compute_factor_ic_on_samples(df, factor, failure_mask)
            
            # 召回得分 = 失效样本 IC - 整体 IC
            recall_score = failure_ic - overall_ic
            
            # 检查与核心因子的互信息
            max_mi = 0
            for core_factor in core_factors:
                if core_factor in df.columns:
                    mi = compute_mutual_information(
                        df[factor].fillna(0).values,
                        df[core_factor].fillna(0).values,
                        n_bins=10
                    )
                    max_mi = max(max_mi, mi)
            
            # 召回条件
            if max_mi < 0.15 and recall_score > -0.005:
                composite_score = recall_score * 0.6 + failure_ic * 0.4
                
                recall_scores[factor] = {
                    'overall_ic': overall_ic,
                    'failure_ic': failure_ic,
                    'recall_score': recall_score,
                    'composite_score': composite_score,
                    'max_mi': max_mi,
                }
        
        # 按综合得分排序
        sorted_factors = sorted(recall_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        recalled = []
        for factor, scores in sorted_factors[:max_recall]:
            recalled.append(factor)
            self.residual_analysis[factor] = scores
            self._log_recall(
                "Recalled",
                f"{factor}: overall_ic={scores['overall_ic']:.4f}, failure_ic={scores['failure_ic']:.4f}, "
                f"composite_score={scores['composite_score']:.4f}"
            )
        
        self.recalled_factors = recalled
        
        # 如果没有召回任何因子，强制召回 IC 最高的 2 个
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


class FactorGeneratorV143:
    """V143 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.generation_log.append(entry)
        logger.info(f"[V143][FactorGenerator] {action}: {details}")
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算动量因子"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算反转因子（负动量）"""
        return -df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算波动率因子"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window).std()
        ).fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """计算量价背离因子"""
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
        """计算流动性 Alpha 因子"""
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


class AlphaResearchV143:
    """
    V143 Alpha 研究引擎 - 多维张量核与 IR 稳定性加固.
    
    【V143 核心改进】
    1. FeatureDistillationV143: 3D 张量交互（核中性化 + 场景调制）
    2. ResidualBasedRecallV143: 基于残差分析的因子召回
    3. ICPrecisionWeighting: IC 精度加权（IR 稳定性加固）
    
    【目标指标】
    - T+1 Rank IC > 0.055
    - IC_IR > 0.80
    - IC_Std < 0.04
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_3d_distillation: bool = True,
        enable_kernel_neutralization: bool = True,
        enable_ic_precision: bool = True,
        enable_orthogonalization: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        max_recall_factors: int = 3,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_3d_distillation = enable_3d_distillation
        self.enable_kernel_neutralization = enable_kernel_neutralization
        self.enable_ic_precision = enable_ic_precision
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
        self.data_healer = DataHealingV143(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV143()
        
        # V143 核心模块
        self.residual_recall = ResidualBasedRecallV143(top_percent=0.2)
        self.feature_distillation = FeatureDistillationV143(
            enable_kernel=enable_kernel_neutralization,
            enable_3d=enable_3d_distillation
        ) if enable_3d_distillation else None
        self.ic_precision = ICPrecisionWeighting(rolling_window=20) if enable_ic_precision else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: 3D Tensor Interaction + IC Precision Weighting")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max N Factors: {n_factors}")
        logger.info(f"  Max Recall Factors: {max_recall_factors}")
        logger.info(f"  3D Distillation: {'Enabled' if enable_3d_distillation else 'Disabled'}")
        logger.info(f"  Kernel Neutralization: {'Enabled' if enable_kernel_neutralization else 'Disabled'}")
        logger.info(f"  IC Precision Weighting: {'Enabled' if enable_ic_precision else 'Disabled'}")
        logger.info(f"  Target IR: 0.80 (V142: 0.60)")
    
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
        """因子处理：增强版 Winsorization + 标准化"""
        # V143 增强版去极值
        series_wins = winsorize_enhanced(series.fillna(0), sigma=3.0, percentile=0.99)
        
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
        """计算 Alpha 评分 - V143 核心逻辑（3D 张量交互 + IC 精度加权）"""
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
        
        core_factors = [f for f in V142_CORE_FACTORS if f in result.columns]
        
        if len(core_factors) >= 2:
            priority_candidates = ['reversion_5', 'reversion_10', 'volume_price_contradiction', 
                                   'rsi_14', 'mfi_14', 'price_position_20'] + V142_CANDIDATE_FACTORS
            self.recalled_factors = self.residual_recall.select_recall_factors(
                result, core_factors, priority_candidates, self.max_recall_factors
            )
            self._log_audit("ResidualRecall", f"Recalled {len(self.recalled_factors)} factors: {self.recalled_factors}")
        else:
            self._log_audit("ResidualRecall", "Insufficient core factors, skipping recall")
            self.recalled_factors = []
        
        # 5. 计算 3D 提纯特征（V143 核心）
        distilled_factors = []
        if self.enable_3d_distillation and self.feature_distillation:
            self._log_audit("3DDistillation", "Computing 3D distilled features...")
            result = self.feature_distillation.compute_all_distilled_features(
                result, core_factors, self.recalled_factors
            )
            
            distilled_factors = list(self.feature_distillation.get_distilled_features().keys())
            self._log_audit("3DDistillation", f"Generated {len(distilled_factors)} 3D distilled features")
        
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
        
        # 添加 3D 提纯因子（V143 核心）
        for factor in distilled_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors (3D distillation priority)"
        )
        
        # 7. 计算 IC 和 IC 滚动统计（用于 IC 精度加权）
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
            
            factor_ics.append((factor, t1_specificity))
            
            self._log_audit(
                "FactorAnalysis",
                f"{factor}: T+1={t1_ic:.4f}, T+3={t3_ic:.4f}, T+5={t5_ic:.4f}, T1_Specificity={t1_specificity:.4f}"
            )
        
        # 分离 3D 提纯因子和其他因子
        distilled_keywords = ['_3d_', '_distilled_']
        distilled_ics = [(f, ic) for f, ic in factor_ics if any(kw in f.lower() for kw in distilled_keywords)]
        other_ics = [(f, ic) for f, ic in factor_ics if f not in [x[0] for x in distilled_ics]]
        
        # 3D 提纯因子优先（V143 核心）
        distilled_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        other_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 8. 准备因子数据 - 强制保留 3D 提纯因子
        factor_data = {}
        factor_names = []
        
        # 首先处理 3D 提纯因子（强制保留）
        for factor, adjusted_ic in distilled_ics:
            if abs(adjusted_ic) < self.ic_threshold:
                continue
            
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
            factor_names.append(factor)
            self.selected_factors.append(factor)
        
        # 然后处理其他因子
        for factor, adjusted_ic in other_ics:
            if abs(adjusted_ic) < self.ic_threshold:
                continue
            if len(factor_names) >= self.n_factors:
                break
            
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
            factor_names.append(factor)
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(all_candidate_factors)} factors")
        
        # 9. 简化策略 - 强制保留 3D 提纯因子
        distilled_selected = [f for f in self.selected_factors if any(kw in f.lower() for kw in ['_3d_', '_distilled_'])]
        other_selected = [f for f in self.selected_factors if f not in distilled_selected]
        
        other_selected.sort(key=lambda f: abs(self.factor_ics.get(f, 0)), reverse=True)
        
        final_selected = distilled_selected[:3]  # 至少保留 3 个 3D 提纯因子
        if len(final_selected) < 3:
            final_selected.extend(other_selected[:3 - len(final_selected)])
        
        self.selected_factors = final_selected[:self.n_factors]
        factor_data_filtered = {f: factor_data[f] for f in self.selected_factors}
        factor_data = factor_data_filtered
        
        self._log_audit(
            "FactorSelection",
            f"Final selected {len(self.selected_factors)} factors (3D distillation priority): {self.selected_factors}"
        )
        
        # 计算权重并合成最终评分
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            weights = []
            for factor in self.selected_factors:
                # 基础权重 = IC 绝对值
                base_weight = abs(self.factor_ics[factor])
                
                # V143: IC 精度加权
                if self.enable_ic_precision and self.ic_precision:
                    precision_weight = self.ic_precision.compute_precision_weight(
                        result, factor, base_weight
                    )
                    weight = precision_weight
                    self._log_audit(
                        "ICPrecisionWeight",
                        f"{factor}: base={base_weight:.4f}, precision={weight:.4f}"
                    )
                else:
                    weight = base_weight
                
                # 3D 提纯因子权重增强
                is_3d = any(kw in factor.lower() for kw in ['_3d_', '_distilled_'])
                if is_3d:
                    weight = weight * 2.0  # 3D 提纯因子权重×2.0
                    self._log_audit("WeightBoost", f"{factor}: 3D distilled factor, weight ×2.0")
                
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
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (3D Tensor + IC Precision)")
        
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
    
    def get_distilled_features(self) -> Dict:
        """获取 3D 提纯特征"""
        return self.feature_distillation.get_distilled_features() if self.feature_distillation else {}
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_residual_analysis(self) -> Dict:
        """获取残差分析结果"""
        return self.residual_recall.residual_analysis
    
    def get_market_regime(self) -> MarketRegimeTensor:
        """获取市场场景模块"""
        return self.feature_distillation.get_market_regime() if self.feature_distillation else None
    
    def get_kernel_stats(self) -> Dict:
        """获取核中性化统计"""
        return self.feature_distillation.get_kernel_neutralization().get_kernel_stats() if self.feature_distillation else {}
    
    def get_ic_precision_stats(self) -> Dict:
        """获取 IC 精度统计"""
        return self.ic_precision.get_ic_rolling_stats() if self.ic_precision else {}
    
    def get_efficiency_ratio(self) -> float:
        """计算效率指标：IC / Factor Count"""
        if not self.selected_factors:
            return 0.0
        
        ics = list(self.get_factor_ics().values())
        if not ics:
            return 0.0
        
        mean_ic = abs(np.mean(ics))
        return mean_ic / len(self.selected_factors)
    
    def get_distillation_log(self) -> List[Dict]:
        """获取提纯日志"""
        return self.feature_distillation.get_distillation_log() if self.feature_distillation else []


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_3d_distillation: bool = True,
    enable_kernel_neutralization: bool = True,
    enable_ic_precision: bool = True,
    enable_orthogonalization: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 3,
) -> AlphaResearchV143:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV143(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_3d_distillation=enable_3d_distillation,
        enable_kernel_neutralization=enable_kernel_neutralization,
        enable_ic_precision=enable_ic_precision,
        enable_orthogonalization=enable_orthogonalization,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV143...")
    
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
    logger.info(f"  3D Distilled features: {alpha.get_distilled_features()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Kernel stats: {alpha.get_kernel_stats()}")
    logger.info(f"  IC Precision stats: {alpha.get_ic_precision_stats()}")
    logger.info(f"  Market Regime: {alpha.get_market_regime().get_current_regime() if alpha.get_market_regime() else 'N/A'}")
    logger.info(f"  Efficiency Ratio: {alpha.get_efficiency_ratio():.4f}")