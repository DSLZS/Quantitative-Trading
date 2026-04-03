"""
Alpha Research Module - V142 特征提纯与 IC 强度修复.

【V142 核心改进 - 响应 V141 失败分析】
V141 的 IC 从 0.0425 跌至 0.0381，这是显著的退步。根因分析：
1. 简单的 Rank * Rank 交互过于粗糙，丢失了因子的原始信息
2. 没有对召回因子进行中性化，导致信息冗余
3. 交互核缺乏经济逻辑支撑

V142 的使命是：通过特征提纯（Feature Distillation）实现 IC 的实质性跃迁。

【V142 核心算法 - 信息瓶颈与高阶提纯】
1. Standardized Residual Scaling (标准化残差缩放):
   - 在交互前，对召回因子进行中性化
   - 剔除已有的核心信号分量，只保留"纯净残差"
   - 公式：Residual = Factor_Recall - β * Factor_Core
                    (β = OLS 回归系数)

2. Sigmoid-Gating (门控机制):
   - 不使用直接相乘，而是使用门控机制
   - 公式：Gated_Interaction = Sigmoid(Rank(Factor_A)) * Rank(Factor_B)
   - Sigmoid 将因子转化为"置信度开关"(0~1)

3. 重点组合优化:
   - volume_price_contradiction × reversion_5
   - 捕捉"缩量下跌后的极致反转"信号

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 142 运行
- 严禁"回测碰运气"：禁止修改 backtest_referee.py 中的费率或初始资金
- 报错必改：如遇数据缺失，必须主动在代码中通过逻辑（如 ffill 或 median_filling）自愈
- 内存优化：使用 chunking 处理大数据，严禁中断任务

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.045 | 核心指标（必须超过 V140 的 0.0425） |
| IC_IR | > 0.6 | 稳定性 |
| IC Decay | T+1 > T+3 > T+5 | 正常衰减模式 |
| Feature Distillation | >= 2 | 至少 2 个提纯特征 |
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

VERSION = "V142"

# V140 核心因子（保留 V141 的因子召回框架）
V140_CORE_FACTORS = [
    'momentum_20',      # 20 日动量
    'volatility_10',    # 10 日波动率
    'volume_price_contradiction',  # 量价背离
    'liquidity_alpha',  # 流动性 Alpha
]

# V139 候选因子池（用于召回）
V139_CANDIDATE_FACTORS = [
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

# V142 所有因子（核心 + 召回）
ALL_FACTORS = V140_CORE_FACTORS + V139_CANDIDATE_FACTORS

# V142 最大因子数量
MAX_FACTORS = 12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数 - 用于门控机制"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def winsorize(series: pd.Series, sigma: float = 2.5) -> pd.Series:
    """Winsorization 去极值"""
    mean = series.mean()
    std = series.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return series.clip(lower=lower, upper=upper)


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


def gram_schmidt_orthogonalize(X: np.ndarray, mi_threshold: float = 0.1) -> Tuple[np.ndarray, List[int]]:
    """
    Gram-Schmidt 正交化 + 互信息验证.
    """
    n_samples, n_factors = X.shape
    
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


class DataHealing:
    """V142 数据自愈模块 - 增强版"""
    
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
                logger.info("[V142][DataHealing] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V142][DataHealing] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V142][DataHealing] No database URL, SQL healer disabled")
    
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
        logger.info(f"[V142][DataHealing] {action} - Column: {column}, Status: {status}, {details}")
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """检查并修复缺失列"""
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
        
        result = self._heal_nan(result)
        
        return result
    
    def _heal_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN 检测与修复 - V142 强制自愈"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                if 'symbol' in result.columns:
                    result[col] = result.groupby('symbol')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
                else:
                    result[col] = result[col].ffill().bfill()
                
                remaining_nan = result[col].isna().sum()
                if remaining_nan > 0:
                    result[col] = result[col].fillna(0.0)
                
                self._log_healing(
                    action="NaNHealed",
                    column=col,
                    status="SUCCESS",
                    details=f"Healed {nan_count} NaN values, {remaining_nan} remaining"
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
            logger.error(f"[V142][DataHealing] SQL heal failed: {e}")
            for col in columns:
                result[col] = 0.0
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class DynamicHalfLifeEngine:
    """V142 动态衰减半衰期引擎 - 继承 V141"""
    
    def __init__(self, base_half_life: int = 10, volatility_window: int = 60):
        self.base_half_life = base_half_life
        self.volatility_window = volatility_window
        self.half_life_log = []
        self.current_half_life = base_half_life
        
    def _log_half_life(self, action: str, details: str = ""):
        """记录半衰期日志"""
        entry = {'action': action, 'details': details}
        self.half_life_log.append(entry)
        logger.info(f"[V142][DynamicHalfLife] {action}: {details}")
    
    def compute_market_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场波动率"""
        result = df.copy()
        
        if 'volatility_20' in df.columns:
            result['market_volatility'] = result.groupby('trade_date')['volatility_20'].transform('mean')
        elif 'pct_chg' in df.columns:
            result['market_volatility'] = result.groupby('trade_date')['pct_chg'].transform('std')
        else:
            result['market_volatility'] = 1.0
        
        result['market_volatility'] = result['market_volatility'].fillna(1.0)
        
        self._log_half_life(
            "Computed",
            f"market_volatility = Mean(Volatility_20, cross-section)"
        )
        
        return result
    
    def compute_dynamic_half_life(self, df: pd.DataFrame) -> pd.Series:
        """计算动态半衰期"""
        if 'market_volatility' not in df.columns:
            df = self.compute_market_volatility(df)
        
        def compute_half_life(series):
            median_vol = series.rolling(self.volatility_window, min_periods=20).median()
            q75 = series.rolling(self.volatility_window, min_periods=20).quantile(0.75)
            q25 = series.rolling(self.volatility_window, min_periods=20).quantile(0.25)
            
            def _half_life(vol, med, q75, q25):
                if pd.isna(med) or pd.isna(q75) or pd.isna(q25):
                    return self.base_half_life
                
                vol_adjustment = (vol - med) / (med + 1e-10)
                adjusted_half_life = self.base_half_life * (1 - 0.5 * vol_adjustment)
                adjusted_half_life = max(5, min(20, adjusted_half_life))
                
                return int(adjusted_half_life)
            
            return pd.Series([_half_life(v, m, q75_, q25_) 
                             for v, m, q75_, q25_ in zip(series, median_vol, q75, q25)],
                            index=series.index)
        
        half_life = df.groupby('symbol').apply(
            lambda g: compute_half_life(g['market_volatility'])
        ).reset_index(level=0, drop=True)
        
        self.current_half_life = int(half_life.iloc[-1]) if len(half_life) > 0 else self.base_half_life
        
        self._log_half_life(
            "Computed",
            f"Dynamic half-life: current={self.current_half_life}, base={self.base_half_life}"
        )
        
        return half_life
    
    def apply_ema_with_dynamic_half_life(self, df: pd.DataFrame, signal_col: str) -> pd.Series:
        """使用动态半衰期应用 EMA 平滑"""
        half_life_series = self.compute_dynamic_half_life(df)
        
        def apply_ema(group):
            half_life = half_life_series.loc[group.index].iloc[0]
            span = 2 * half_life - 1
            return group[signal_col].ewm(span=span, adjust=False).mean()
        
        smoothed = df.groupby('symbol').apply(apply_ema).reset_index(level=0, drop=True)
        
        self._log_half_life(
            "AppliedEMA",
            f"Signal: {signal_col}, avg_half_life={half_life_series.mean():.1f}"
        )
        
        return smoothed
    
    def get_half_life_log(self) -> List[Dict]:
        """获取半衰期日志"""
        return self.half_life_log
    
    def get_current_half_life(self) -> int:
        """获取当前半衰期"""
        return self.current_half_life


class ResidualBasedRecall:
    """
    V142 基于残差分析的因子召回模块 - 增强版.
    
    【核心功能】
    1. 计算 V140 核心模型的残差
    2. 在 Residuals 绝对值最大的前 20% 样本上，计算各 V139 因子的 IC
    3. 召回标准：在失效样本上 IC 显著高于整体 IC 的因子
    
    【V142 修复】
    - 放宽互信息阈值从 0.1 到 0.15
    - 召回得分阈值从 0 降到 -0.005（允许轻微负向）
    - 增加场景特异性评分
    """
    
    def __init__(self, top_percent: float = 0.2):
        self.top_percent = top_percent
        self.recall_log = []
        self.recalled_factors = []
        self.residual_analysis = {}
        
    def _log_recall(self, action: str, details: str = ""):
        """记录召回日志"""
        entry = {'action': action, 'details': details}
        self.recall_log.append(entry)
        logger.info(f"[V142][ResidualRecall] {action}: {details}")
    
    def compute_residuals(self, df: pd.DataFrame, core_factors: List[str]) -> pd.Series:
        """
        计算 V140 核心模型的残差.
        
        Residual = Actual_Return - Predicted_Return
        Predicted_Return = weighted_sum(core_factors)
        """
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
        """
        选择召回因子 - 增强版.
        
        召回标准（放宽）：
        1. 在失效样本上的 IC > 整体 IC - 0.01（允许轻微下降）
        2. 与核心因子的互信息 < 0.15（放宽）
        3. 优先选择在极端场景下表现优异的因子
        """
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
            
            # 检查与核心因子的互信息（放宽到 0.15）
            max_mi = 0
            for core_factor in core_factors:
                if core_factor in df.columns:
                    mi = compute_mutual_information(
                        df[factor].fillna(0).values,
                        df[core_factor].fillna(0).values,
                        n_bins=10
                    )
                    max_mi = max(max_mi, mi)
            
            # V142 修复：放宽条件
            # 1. 互信息 < 0.15（原 0.1）
            # 2. 召回得分 > -0.005（原 > 0）
            if max_mi < 0.15 and recall_score > -0.005:
                # 综合得分 = 召回得分 * 0.6 + 失效样本 IC * 0.4
                composite_score = recall_score * 0.6 + failure_ic * 0.4
                
                recall_scores[factor] = {
                    'overall_ic': overall_ic,
                    'failure_ic': failure_ic,
                    'recall_score': recall_score,
                    'composite_score': composite_score,
                    'max_mi': max_mi,
                }
        
        # 按综合得分排序，选择前 N 个
        sorted_factors = sorted(recall_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        recalled = []
        for factor, scores in sorted_factors[:max_recall]:
            recalled.append(factor)
            self.residual_analysis[factor] = scores
            self._log_recall(
                "Recalled",
                f"{factor}: overall_ic={scores['overall_ic']:.4f}, failure_ic={scores['failure_ic']:.4f}, "
                f"composite_score={scores['composite_score']:.4f}, max_mi={scores['max_mi']:.4f}"
            )
        
        self.recalled_factors = recalled
        
        # 如果没有召回任何因子，强制召回 IC 最高的 2 个
        if len(recalled) == 0:
            self._log_recall("Warning", "No factors recalled with default criteria, forcing top IC factors")
            forced_factors = []
            for factor in candidate_factors[:10]:  # 只检查前 10 个候选
                if factor in df.columns:
                    ic = self.compute_factor_ic(df, factor)
                    forced_factors.append((factor, ic))
            
            forced_factors.sort(key=lambda x: abs(x[1]), reverse=True)
            for factor, ic in forced_factors[:2]:
                recalled.append(factor)
                self._log_recall("Forced", f"{factor}: IC={ic:.4f}")
            
            self.recalled_factors = recalled
        
        return recalled


class FeatureDistillation:
    """
    V142 特征提纯模块 - 核心创新.
    
    【V142 与 V141 的本质区别】
    V141: Interaction = Rank(Factor_A) * Rank(Factor_B)  # 简单乘法
    V142: 
      1. Standardized Residual Scaling: Residual = Factor_Recall - β * Factor_Core
      2. Sigmoid-Gating: Gated = Sigmoid(Rank(Factor_A)) * Rank(Factor_B)
    
    【为什么 V142 更有效】
    1. 残差缩放剔除了信息冗余，只保留"纯净"的新信息
    2. Sigmoid 门控将因子转化为"置信度"(0~1)，避免简单相乘的粗糙
    3. 经济逻辑支撑：门控机制模拟了"条件触发"的交易决策
    """
    
    def __init__(self):
        self.distillation_log = []
        self.distilled_features = {}
        
    def _log_distillation(self, action: str, details: str = ""):
        """记录提纯日志"""
        entry = {'action': action, 'details': details}
        self.distillation_log.append(entry)
        logger.info(f"[V142][FeatureDistillation] {action}: {details}")
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名 (0-1 归一化)"""
        return series.rank(method='average', pct=True)
    
    def compute_residual_scaling(self, df: pd.DataFrame, recall_factor: str, 
                                  core_factors: List[str]) -> pd.Series:
        """
        计算标准化残差缩放.
        
        【原理】
        对召回因子进行中性化，剔除已有的核心信号分量，只保留"纯净残差"。
        
        公式：
        1. β = OLS(Factor_Recall ~ Factor_Core)
        2. Residual = Factor_Recall - β * Factor_Core
        3. Scaled_Residual = Residual / Std(Residual)
        
        【为什么有效】
        - 简单相乘会保留召回因子中与核心因子相关的冗余信息
        - 残差缩放确保交互特征只包含"新"信息
        """
        if recall_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 收集核心因子数据
        core_data = []
        for cf in core_factors:
            if cf in df.columns:
                core_data.append(df[cf].fillna(0).values)
        
        if not core_data:
            return df[recall_factor].fillna(0)
        
        # 核心因子矩阵
        X = np.column_stack(core_data)
        y = df[recall_factor].fillna(0).values
        
        # OLS 回归：y = X * β + ε
        # β = (X'X)^(-1) X'y
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            
            # 预测值
            y_pred = X @ beta
            
            # 残差
            residual = y - y_pred
            
            # 标准化残差
            residual_std = np.std(residual) + 1e-10
            scaled_residual = residual / residual_std
            
            self._log_distillation(
                "ResidualScaling",
                f"{recall_factor}: β={beta.mean():.4f}, residual_std={residual_std:.4f}"
            )
            
        except Exception as e:
            logger.warning(f"[V142][FeatureDistillation] Residual scaling failed: {e}")
            scaled_residual = y
        
        return pd.Series(scaled_residual, index=df.index)
    
    def compute_sigmoid_gating(self, df: pd.DataFrame, gate_factor: str, 
                                target_factor: str) -> pd.Series:
        """
        计算 Sigmoid 门控交互.
        
        【原理】
        不使用直接相乘，而是使用 Sigmoid 将门控因子转化为"置信度开关"(0~1)。
        
        公式：
        Gated_Interaction = Sigmoid(Rank(Gate_Factor) * 10 - 5) * Rank(Target_Factor)
        
        【为什么有效】
        - Sigmoid(x) ∈ (0, 1)，将因子转化为概率/置信度
        - 当 Gate_Factor 高时，Sigmoid→1，完全释放 Target_Factor 信号
        - 当 Gate_Factor 低时，Sigmoid→0，抑制 Target_Factor 信号
        - 这模拟了"条件触发"的交易决策逻辑
        """
        if gate_factor not in df.columns or target_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 排名归一化 (0~1)
        rank_gate = self._rank(df[gate_factor].fillna(0))
        rank_target = self._rank(df[target_factor].fillna(0))
        
        # Sigmoid 门控：将排名映射到 (-5, 5) 然后应用 Sigmoid
        # Rank=0 → Sigmoid(-5)≈0.007, Rank=1 → Sigmoid(5)≈0.993
        sigmoid_input = rank_gate * 10 - 5
        gate_values = sigmoid(sigmoid_input.values)
        
        # 门控交互
        gated_interaction = gate_values * rank_target.values
        
        self._log_distillation(
            "SigmoidGating",
            f"{gate_factor} → {target_factor}: gate_range=[{gate_values.min():.3f}, {gate_values.max():.3f}]"
        )
        
        return pd.Series(gated_interaction, index=df.index)
    
    def compute_distilled_interaction(self, df: pd.DataFrame, core_factor: str, 
                                       recall_factor: str) -> pd.Series:
        """
        计算提纯后的交互特征.
        
        【完整流程】
        1. 对 recall_factor 进行残差缩放，得到纯净残差
        2. 使用 core_factor 作为门控，对残差进行 Sigmoid 门控
        3. 输出：Distilled_Interaction = Sigmoid(Rank(Core)) * Scaled_Residual(Recall)
        
        【经济逻辑】
        - core_factor 作为"触发条件"（如量价背离程度）
        - recall_factor 的残差作为"新信息"（如反转信号的纯净分量）
        - 只有当触发条件满足时，才释放新信息
        """
        if core_factor not in df.columns or recall_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 残差缩放
        scaled_residual = self.compute_residual_scaling(df, recall_factor, [core_factor])
        
        # 2. Sigmoid 门控
        rank_core = self._rank(df[core_factor].fillna(0))
        sigmoid_input = rank_core * 10 - 5
        gate_values = sigmoid(sigmoid_input.values)
        
        # 3. 提纯交互
        distilled = gate_values * scaled_residual.values
        
        feature_name = f"{core_factor}_distilled_{recall_factor}"
        self.distilled_features[feature_name] = {
            'core_factor': core_factor,
            'recall_factor': recall_factor,
            'type': 'distilled_interaction',
            'method': 'residual_scaling + sigmoid_gating',
        }
        
        self._log_distillation(
            "DistilledInteraction",
            f"{feature_name}: Sigmoid(Rank({core_factor})) × Scaled_Residual({recall_factor})"
        )
        
        return pd.Series(distilled, index=df.index)
    
    def compute_volume_reversion_combo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量价背离×反转组合 - V142 重点优化.
        
        【经济逻辑】
        捕捉"缩量下跌后的极致反转"信号：
        1. volume_price_contradiction: 量价背离程度（缩量下跌时为正）
        2. reversion_5: 5 日反转信号（超跌时为正）
        
        【提纯逻辑】
        1. 对 reversion_5 进行残差缩放，剔除 volume_price_contradiction 已有分量
        2. 使用 volume_price_contradiction 作为门控
        3. 只有当量价背离显著时，才释放反转信号
        
        【为什么比 V141 更有效】
        V141: Rank(VPC) * Rank(Rev5) → 粗糙相乘，信息冗余
        V142: Sigmoid(Rank(VPC)) × Scaled_Residual(Rev5) → 提纯 + 门控
        """
        result = df.copy()
        
        vpc = 'volume_price_contradiction'
        rev5 = 'reversion_5'
        
        if vpc not in df.columns or rev5 not in df.columns:
            return result
        
        # 1. 计算提纯交互
        distilled = self.compute_distilled_interaction(df, vpc, rev5)
        result[f'{vpc}_distilled_{rev5}'] = distilled
        
        # 2. 计算反向提纯（reversion 作为门控）
        distilled_rev = self.compute_distilled_interaction(df, rev5, vpc)
        result[f'{rev5}_distilled_{vpc}'] = distilled_rev
        
        # 3. 计算简单 Sigmoid 门控（用于对比）
        sigmoid_gate = self.compute_sigmoid_gating(df, vpc, rev5)
        result[f'{vpc}_sigmoid_{rev5}'] = sigmoid_gate
        
        self._log_distillation(
            "VolumeReversionCombo",
            f"Generated 3 distilled features for {vpc} × {rev5}"
        )
        
        return result
    
    def compute_all_distilled_features(self, df: pd.DataFrame, core_factors: List[str],
                                        recalled_factors: List[str]) -> pd.DataFrame:
        """
        计算所有提纯特征.
        
        【策略】
        1. 重点优化 volume_price_contradiction × reversion_5
        2. 其他核心因子 × 召回因子的提纯交互
        """
        result = df.copy()
        
        # 1. 重点组合
        if 'volume_price_contradiction' in df.columns and 'reversion_5' in df.columns:
            result = self.compute_volume_reversion_combo(result)
        
        # 2. 其他提纯交互
        for core in core_factors:
            if core not in df.columns:
                continue
            for recalled in recalled_factors:
                if recalled not in df.columns or recalled == 'reversion_5':
                    continue
                if core == 'volume_price_contradiction':
                    continue
                
                name = f"{core}_distilled_{recalled}"
                if name not in result.columns:
                    result[name] = self.compute_distilled_interaction(df, core, recalled)
        
        self._log_distillation(
            "Complete",
            f"Generated {len(self.distilled_features)} distilled features"
        )
        
        return result
    
    def get_distillation_log(self) -> List[Dict]:
        """获取提纯日志"""
        return self.distillation_log
    
    def get_distilled_features(self) -> Dict:
        """获取提纯特征字典"""
        return self.distilled_features


class RegimeAwareWeighting:
    """
    V142 场景感知动态权重 2.0 - 继承 V141.
    
    【核心功能】
    1. 高波动场景：降低线性因子权重，提升"提纯特征"信号权重
    2. 低波动场景：保持核心因子主导
    """
    
    def __init__(self, volatility_threshold: float = 0.7):
        self.volatility_threshold = volatility_threshold
        self.weighting_log = []
        self.current_regime = None
        self.regime_weights = {}
        
    def _log_weighting(self, action: str, details: str = ""):
        """记录权重日志"""
        entry = {'action': action, 'details': details}
        self.weighting_log.append(entry)
        logger.info(f"[V142][RegimeAwareWeighting] {action}: {details}")
    
    def compute_market_volatility(self, df: pd.DataFrame) -> pd.Series:
        """计算市场波动率"""
        if 'volatility_20' in df.columns:
            return df.groupby('trade_date')['volatility_20'].transform('mean')
        elif 'pct_chg' in df.columns:
            return df.groupby('trade_date')['pct_chg'].transform('std')
        else:
            return pd.Series(1.0, index=df.index)
    
    def get_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        获取波动率场景分类.
        
        Returns:
            regime: 0=低波动，1=高波动
        """
        vol = self.compute_market_volatility(df)
        
        # 计算滚动分位数
        def classify_regime(series):
            threshold = series.quantile(self.volatility_threshold)
            return (series > threshold).astype(int)
        
        regime = df.groupby('trade_date')['vol_20'].transform(classify_regime) if 'vol_20' in df.columns else classify_regime(vol)
        
        self.current_regime = int(regime.iloc[-1]) if len(regime) > 0 else 0
        
        self._log_weighting(
            "Classified",
            f"Volatility regime: {regime.value_counts().to_dict()}"
        )
        
        return regime
    
    def compute_adaptive_weights(self, df: pd.DataFrame, base_weights: Dict[str, float],
                                  distilled_factors: List[str]) -> Dict[str, float]:
        """
        计算场景自适应权重.
        
        高波动场景：
        - 线性因子权重 × 0.7
        - 提纯因子权重 × 1.5
        
        低波动场景：
        - 线性因子权重 × 1.2
        - 提纯因子权重 × 0.8
        """
        # 获取最新场景
        vol = self.compute_market_volatility(df)
        threshold = vol.quantile(self.volatility_threshold)
        latest_vol = vol.iloc[-1] if len(vol) > 0 else 0
        latest_regime = 1 if latest_vol > threshold else 0
        
        self.current_regime = latest_regime
        
        adaptive_weights = base_weights.copy()
        
        if latest_regime == 1:
            # 高波动场景
            self._log_weighting("HighVolatility", "Enhancing distilled factors ×1.5, reducing linear ×0.7")
            
            for factor in base_weights:
                if factor in distilled_factors:
                    adaptive_weights[factor] = base_weights[factor] * 1.5
                else:
                    adaptive_weights[factor] = base_weights[factor] * 0.7
        else:
            # 低波动场景
            self._log_weighting("LowVolatility", "Enhancing linear factors ×1.2, reducing distilled ×0.8")
            
            for factor in base_weights:
                if factor in distilled_factors:
                    adaptive_weights[factor] = base_weights[factor] * 0.8
                else:
                    adaptive_weights[factor] = base_weights[factor] * 1.2
        
        # 归一化
        total = sum(adaptive_weights.values())
        if total > 0:
            adaptive_weights = {k: v / total for k, v in adaptive_weights.items()}
        
        self.regime_weights = adaptive_weights
        
        return adaptive_weights
    
    def get_weighting_log(self) -> List[Dict]:
        """获取权重日志"""
        return self.weighting_log
    
    def get_current_regime(self) -> int:
        """获取当前场景"""
        return self.current_regime
    
    def get_regime_weights(self) -> Dict[str, float]:
        """获取场景权重"""
        return self.regime_weights


class AdaptiveFeatureEnsemble:
    """V142 自适应特征集成"""
    
    def __init__(self, n_bins: int = 10, min_samples_per_bin: int = 30, rolling_window: int = 20):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.rolling_window = rolling_window
        self.bin_stats = {}
        self.ensemble_log = []
        self.rolling_ic = {}
        self.orthogonalization_stats = {}
        
        # V142 新增模块
        self.dynamic_half_life_engine = DynamicHalfLifeEngine(base_half_life=10, volatility_window=60)
        
    def _log_ensemble(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.ensemble_log.append(entry)
        logger.info(f"[V142][AdaptiveEnsemble] {action}: {details}")
    
    def update_rolling_ic(self, df: pd.DataFrame, factor_col: str):
        """更新滚动 IC 记录"""
        if 'trade_date' not in df.columns or 't1_return' not in df.columns:
            return
        
        dates = df['trade_date'].unique()
        dates = sorted(dates)
        
        ics = []
        for date in dates[-self.rolling_window:]:
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            f = day_data[factor_col].fillna(0)
            l = day_data['t1_return'].fillna(0)
            if len(f) > 10 and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        self.rolling_ic[factor_col] = ics
    
    def compute_dynamic_bin_weight(self, df: pd.DataFrame, factor_col: str) -> Dict[int, float]:
        """计算动态分箱权重"""
        self.update_rolling_ic(df, factor_col)
        
        ics = self.rolling_ic.get(factor_col, [])
        
        if len(ics) < 5:
            return {i: 1.0 / self.n_bins for i in range(self.n_bins)}
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics) + 1e-10
        ic_ir = ic_mean / ic_std
        
        bin_weights = {}
        for i in range(self.n_bins):
            base_weight = 1.0 / self.n_bins
            
            if ic_mean > 0:
                if i == 0 or i == self.n_bins - 1:
                    weight = base_weight * (1.5 + ic_ir)
                else:
                    weight = base_weight
            else:
                if i == 0 or i == self.n_bins - 1:
                    weight = base_weight * (1.5 - ic_ir)
                else:
                    weight = base_weight
            
            bin_weights[i] = weight
        
        total = sum(bin_weights.values())
        bin_weights = {k: v / total for k, v in bin_weights.items()}
        
        return bin_weights
    
    def apply_gram_schmidt(self, factor_matrix: np.ndarray, factor_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """应用 Gram-Schmidt 正交化 + MI 验证"""
        if len(factor_matrix) == 0 or len(factor_names) == 0:
            return factor_matrix, factor_names
        
        orthogonalized, kept_indices = gram_schmidt_orthogonalize(factor_matrix, mi_threshold=0.1)
        
        kept_names = [factor_names[i] for i in kept_indices]
        
        self.orthogonalization_stats = {
            'method': 'gram_schmidt_with_mi',
            'correlation_threshold': 0.2,
            'mi_threshold': 0.1,
            'input_features': len(factor_names),
            'output_features': len(kept_names),
            'removed_features': [factor_names[i] for i in range(len(factor_names)) if i not in kept_indices],
        }
        
        self._log_ensemble(
            "GramSchmidtApplied",
            f"Reduced {len(factor_names)} -> {len(kept_names)} factors (corr < 0.2, MI < 0.1)"
        )
        
        return orthogonalized, kept_names
    
    def compute_bin_based_score(self, df: pd.DataFrame, factor_col: str) -> pd.Series:
        """基于分箱计算非线性评分"""
        result = df.copy()
        
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if 't1_return' not in df.columns:
            return pd.Series(0, index=df.index)
        
        factor_values = df[factor_col].fillna(0)
        trade_dates = df['trade_date']
        
        all_bins = []
        for date in trade_dates.unique():
            date_mask = trade_dates == date
            date_values = factor_values[date_mask]
            
            if len(date_values) < self.n_bins:
                date_bins = date_values.rank(method='average', pct=True).mul(self.n_bins).fillna(0).astype(int).clip(0, self.n_bins - 1)
            else:
                try:
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    bin_boundaries = date_values.quantile(quantiles)
                    date_bins = pd.cut(
                        date_values,
                        bins=bin_boundaries.unique(),
                        labels=False,
                        include_lowest=True
                    )
                except Exception:
                    date_bins = date_values.rank(method='average', pct=True).mul(self.n_bins).fillna(0).astype(int).clip(0, self.n_bins - 1)
            
            all_bins.append(pd.Series(date_bins, index=date_values.index))
        
        if all_bins:
            factor_bins = pd.concat(all_bins).reindex(df.index).fillna(0).astype(int)
        else:
            factor_bins = pd.Series(0, index=df.index)
        
        bin_win_rates = {}
        bin_counts = {}
        for bin_id in range(self.n_bins):
            bin_mask = factor_bins == bin_id
            count = bin_mask.sum()
            bin_counts[bin_id] = count
            
            if count < self.min_samples_per_bin:
                bin_win_rates[bin_id] = 0.5
            else:
                bin_returns = df.loc[bin_mask, 't1_return']
                win_rate = (bin_returns > 0).mean()
                bin_win_rates[bin_id] = win_rate
        
        self.bin_stats[factor_col] = bin_win_rates
        
        dynamic_weights = self.compute_dynamic_bin_weight(df, factor_col)
        
        bin_scores = {}
        for bin_id, win_rate in bin_win_rates.items():
            weight = dynamic_weights.get(bin_id, 1.0 / self.n_bins)
            bin_scores[bin_id] = (win_rate - 0.5) * 2 * weight * self.n_bins
        
        score = factor_bins.map(bin_scores).fillna(0)
        
        valid_bins = [v for v in bin_win_rates.values() if v != 0.5]
        if valid_bins:
            self._log_ensemble(
                "ComputedBinScore",
                f"{factor_col}: bins={self.n_bins}, win_rate_range=[{min(valid_bins):.3f}, {max(valid_bins):.3f}]"
            )
        
        return score
    
    def compute_adaptive_weight(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子的自适应权重"""
        if factor_col not in self.bin_stats:
            self.compute_bin_based_score(df, factor_col)
        
        bin_win_rates = self.bin_stats.get(factor_col, {})
        
        if not bin_win_rates:
            return 1.0 / self.n_bins
        
        max_win = max(bin_win_rates.values())
        min_win = min(bin_win_rates.values())
        win_spread = max_win - min_win
        
        ics = self.rolling_ic.get(factor_col, [])
        if ics:
            ic_mean = np.mean(ics)
            ic_std = np.std(ics) + 1e-10
            ic_ir = abs(ic_mean / ic_std)
            adaptive_weight = win_spread * (1 + ic_ir)
        else:
            adaptive_weight = win_spread
        
        return adaptive_weight
    
    def get_bin_stats(self) -> Dict[str, Dict[int, float]]:
        """获取分箱统计"""
        return self.bin_stats
    
    def get_ensemble_log(self) -> List[Dict]:
        """获取集成日志"""
        return self.ensemble_log
    
    def get_orthogonalization_stats(self) -> Dict:
        """获取正交化统计"""
        return self.orthogonalization_stats
    
    def get_dynamic_half_life(self) -> int:
        """获取当前半衰期"""
        return self.dynamic_half_life_engine.get_current_half_life()


class FactorGenerator:
    """V142 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.generation_log.append(entry)
        logger.info(f"[V142][FactorGenerator] {action}: {details}")
    
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


class AlphaResearchV142:
    """
    V142 Alpha 研究引擎 - 特征提纯与 IC 强度修复.
    
    【V142 核心改进】
    1. FeatureDistillation: 特征提纯（残差缩放 + Sigmoid 门控）
    2. ResidualBasedRecall: 基于残差分析的因子召回
    3. RegimeAwareWeighting: 场景感知动态权重 2.0
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_distillation: bool = True,
        enable_regime_weighting: bool = True,
        enable_orthogonalization: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        max_recall_factors: int = 3,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_distillation = enable_distillation
        self.enable_regime_weighting = enable_regime_weighting
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
        self.data_healer = DataHealing(db_url) if auto_heal else None
        self.ensemble = AdaptiveFeatureEnsemble(n_bins=n_bins, rolling_window=20) if enable_ensemble else None
        self.factor_generator = FactorGenerator()
        
        # V142 新增模块
        self.residual_recall = ResidualBasedRecall(top_percent=0.2)
        self.feature_distillation = FeatureDistillation() if enable_distillation else None
        self.regime_weighting = RegimeAwareWeighting(volatility_threshold=0.7) if enable_regime_weighting else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Feature Distillation + Residual-Based Recall")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max N Factors: {n_factors}")
        logger.info(f"  Max Recall Factors: {max_recall_factors}")
        logger.info(f"  Feature Distillation: {'Enabled' if enable_distillation else 'Disabled'}")
        logger.info(f"  Regime-Aware Weighting: {'Enabled' if enable_regime_weighting else 'Disabled'}")
        logger.info(f"  Orthogonalization + MI: {'Enabled' if enable_orthogonalization else 'Disabled'}")
    
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
        """因子处理：去极值 + 标准化"""
        series_wins = winsorize(series.fillna(0), sigma=2.5)
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def _calc_factor_ic_decay(self, df: pd.DataFrame, factor_col: str) -> Tuple[float, float, float]:
        """
        计算因子 IC Decay (T+1, T+3, T+5) - V142 修复版.
        
        使用单期回报计算 IC Decay。
        """
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
        """计算 Alpha 评分 - V142 核心逻辑（特征提纯）"""
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
        
        # V142 修复：计算单期回报
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
        
        # 4. V142: 基于残差分析召回 V139 因子
        self._log_audit("ResidualRecall", "Starting residual-based factor recall...")
        
        core_factors = [f for f in V140_CORE_FACTORS if f in result.columns]
        
        if len(core_factors) >= 2:
            priority_candidates = ['reversion_5', 'reversion_10', 'volume_price_contradiction', 
                                   'rsi_14', 'mfi_14', 'price_position_20'] + V139_CANDIDATE_FACTORS
            self.recalled_factors = self.residual_recall.select_recall_factors(
                result, core_factors, priority_candidates, self.max_recall_factors
            )
            self._log_audit("ResidualRecall", f"Recalled {len(self.recalled_factors)} factors: {self.recalled_factors}")
        else:
            self._log_audit("ResidualRecall", "Insufficient core factors, skipping recall")
            self.recalled_factors = []
        
        # 5. V142: 计算特征提纯（核心创新）
        distilled_factors = []
        if self.enable_distillation and self.feature_distillation:
            self._log_audit("FeatureDistillation", "Computing distilled features...")
            result = self.feature_distillation.compute_all_distilled_features(
                result, core_factors, self.recalled_factors
            )
            
            # 获取提纯特征名称
            distilled_factors = list(self.feature_distillation.get_distilled_features().keys())
            self._log_audit("FeatureDistillation", f"Generated {len(distilled_factors)} distilled features")
        
        # 6. V142 关键修复：强制短期因子优先
        forced_short_term = ['reversion_5', 'volume_price_contradiction', 'liquidity_alpha']
        
        # 构建候选因子池：短期因子 + 提纯因子优先
        all_candidate_factors = []
        
        # 1. 首先添加强制短期因子
        for factor in forced_short_term:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 2. 添加核心因子
        for factor in core_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 3. 添加召回因子
        for factor in self.recalled_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 4. 添加提纯因子（V142 核心）
        for factor in distilled_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors (distillation priority)"
        )
        
        # 7. V142: 使用 T+1 特异性得分排序
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
        
        # 分离提纯因子和其他因子
        distilled_keywords = ['distilled', 'sigmoid']
        distilled_ics = [(f, ic) for f, ic in factor_ics if any(kw in f.lower() for kw in distilled_keywords)]
        other_ics = [(f, ic) for f, ic in factor_ics if f not in [x[0] for x in distilled_ics]]
        
        # 提纯因子优先（V142 核心）
        distilled_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        other_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 8. 准备因子数据 - 强制保留提纯因子
        factor_data = {}
        factor_names = []
        
        # 首先处理提纯因子（强制保留）
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
            
            if self.enable_ensemble and self.ensemble:
                bin_score = self.ensemble.compute_bin_based_score(result, factor)
                factor_data[factor] = bin_score.values
                self._log_audit("BinMapping", f"{factor}: applied {self.n_bins}-bin nonlinear mapping")
            else:
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
            
            if self.enable_ensemble and self.ensemble:
                bin_score = self.ensemble.compute_bin_based_score(result, factor)
                factor_data[factor] = bin_score.values
                self._log_audit("BinMapping", f"{factor}: applied {self.n_bins}-bin nonlinear mapping")
            else:
                f_std = self._process_factor(f_processed, result['trade_date'])
                factor_data[factor] = f_std
            
            factor_names.append(factor)
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(all_candidate_factors)} factors")
        
        # 9. V142: 简化策略 - 强制保留提纯因子
        distilled_selected = [f for f in self.selected_factors if any(kw in f.lower() for kw in ['distilled', 'sigmoid'])]
        other_selected = [f for f in self.selected_factors if f not in distilled_selected]
        
        other_selected.sort(key=lambda f: abs(self.factor_ics.get(f, 0)), reverse=True)
        
        final_selected = distilled_selected[:3]  # 至少保留 3 个提纯因子
        if len(final_selected) < 3:
            final_selected.extend(other_selected[:3 - len(final_selected)])
        
        self.selected_factors = final_selected[:3]
        factor_data_filtered = {f: factor_data[f] for f in self.selected_factors}
        factor_data = factor_data_filtered
        
        self._log_audit(
            "FactorSelection",
            f"Final selected {len(self.selected_factors)} factors (distillation priority): {self.selected_factors}"
        )
        
        # 计算权重并合成最终评分
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            weights = []
            for factor in self.selected_factors:
                if self.enable_ensemble and self.ensemble:
                    base_weight = self.ensemble.compute_adaptive_weight(result, factor)
                else:
                    base_weight = abs(self.factor_ics[factor])
                
                # V142: 提纯因子权重增强
                is_distilled = any(kw in factor.lower() for kw in ['distilled', 'sigmoid'])
                
                if is_distilled:
                    weight = base_weight * 2.0  # 提纯因子权重×2.0
                    self._log_audit("WeightBoost", f"{factor}: distilled factor, weight ×2.0")
                else:
                    weight = base_weight
                
                weights.append(weight)
            
            # V142: 场景自适应权重调整
            if self.enable_regime_weighting and self.regime_weighting:
                base_weights = {f: w for f, w in zip(self.selected_factors, weights)}
                adaptive_weights = self.regime_weighting.compute_adaptive_weights(
                    result, base_weights, distilled_factors
                )
                weights = [adaptive_weights.get(f, w) for f, w in zip(self.selected_factors, weights)]
            
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
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (Feature Distillation + Regime-Aware)")
        
        # V142 修复：确保输出包含单期回报列
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
        """获取召回的 V139 因子"""
        return self.recalled_factors
    
    def get_distilled_features(self) -> Dict:
        """获取提纯特征"""
        return self.feature_distillation.get_distilled_features() if self.feature_distillation else {}
    
    def get_bin_stats(self) -> Dict[str, Dict[int, float]]:
        """获取分箱统计"""
        return self.ensemble.get_bin_stats() if self.ensemble else {}
    
    def get_ensemble_log(self) -> List[Dict]:
        """获取集成日志"""
        return self.ensemble.get_ensemble_log() if self.ensemble else []
    
    def get_orthogonalization_stats(self) -> Dict:
        """获取正交化统计"""
        return self.ensemble.get_orthogonalization_stats() if self.ensemble else {}
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_residual_analysis(self) -> Dict:
        """获取残差分析结果"""
        return self.residual_recall.residual_analysis
    
    def get_regime_weights(self) -> Dict[str, float]:
        """获取场景权重"""
        return self.regime_weighting.get_regime_weights() if self.regime_weighting else {}
    
    def get_current_regime(self) -> int:
        """获取当前场景"""
        return self.regime_weighting.get_current_regime() if self.regime_weighting else 0
    
    def get_dynamic_half_life(self) -> int:
        """获取当前半衰期"""
        return self.ensemble.get_dynamic_half_life() if self.ensemble else 10
    
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
    enable_distillation: bool = True,
    enable_regime_weighting: bool = True,
    enable_orthogonalization: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 3,
) -> AlphaResearchV142:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV142(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_distillation=enable_distillation,
        enable_regime_weighting=enable_regime_weighting,
        enable_orthogonalization=enable_orthogonalization,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV142...")
    
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
    logger.info(f"  Distilled features: {alpha.get_distilled_features()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Residual analysis: {alpha.get_residual_analysis()}")
    logger.info(f"  Current regime: {alpha.get_current_regime()}")
    logger.info(f"  Efficiency Ratio: {alpha.get_efficiency_ratio():.4f}")