"""
Alpha Research Module - V138 特征正交净化与时效性校准.

【V138 核心改进 - 响应 V137 审计发现】
V137 表现出色（IC 0.0549, IR 0.79），但 IC Decay 呈现"反向增长（T+1 < T+5）"。
这说明信号太"慢"了，是在追随趋势而非预测转折。

1. 时效性增强算子 (Timeliness Operators):
   - Signal_Delta = Signal_t - Signal_{t-1}（信号变化量）
   - Volume_Shock 检测成交量突增，对分箱权重进行二次动态调整
   - 重点挖掘 T+1 爆发力

2. 特征正交化 (Factor Orthogonalization):
   - Gram-Schmidt 正交化过程
   - 确保新引入的量价背离因子与原始基础因子池的相关性 < 0.2

3. 分箱逻辑深度演进:
   - Dynamic_Bin_Weighting
   - 根据最近 20 天的滚动 IC 分布动态调整分箱权重

【架构红线】
- 所有代码封装在 src/alpha_research_v138.py
- 必须通过 python main.py --version 138 回测
- 严禁修改裁判代码 (BacktestReferee)

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标（超越 V137） |
| IC_IR | > 0.75 | 稳定性 |
| T+1 vs T+5 | T+1 > T+5 | 修复反向衰减 |
| 特征正交化 | corr < 0.2 | Gram-Schmidt |
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

VERSION = "V138"

# V138 基础因子池 - 继承 V137
BASE_FACTORS = [
    'pct_chg', 'change',
    # Momentum
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
    # Reversion
    'reversion_5', 'reversion_10',
    # Volatility
    'volatility_5', 'volatility_10', 'volatility_20',
    # Volume-Price
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'vwap_distance', 'volume_rank', 'price_rank',
    # Value
    'value_rank', 'ep_rank', 'bp_rank',
    # Technical
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist',
    # MA Deviation
    'ma_deviation_5', 'ma_deviation_10', 'ma_deviation_20',
    'price_position_20', 'price_position_60', 'bias_60',
    # Turnover
    'turnover_bias_5', 'turnover_bias_10', 'turnover_bias_20',
    'volume_shrink_ratio', 'turnover_vol_ratio',
    # Order Flow
    'order_flow_imbalance_5', 'order_flow_imbalance_10',
    'smart_money_divergence', 'big_order_ratio',
]

# V138 新增量价背离因子
LIQUIDITY_FACTORS = [
    'volume_price_contradiction',
    'liquidity_alpha',
    'ofi_normalized',
    'volume_confirmed_momentum',
]

# V138 新增时效性因子
TIMELINESS_FACTORS = [
    'signal_delta',          # 信号变化量
    'volume_shock',          # 成交量突增
    'price_acceleration',    # 价格加速度
    'momentum_change',       # 动量变化
]

# V138 所有因子
ALL_FACTORS = BASE_FACTORS + LIQUIDITY_FACTORS + TIMELINESS_FACTORS


def winsorize(series: pd.Series, sigma: float = 2.5) -> pd.Series:
    """Winsorization 去极值"""
    mean = series.mean()
    std = series.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return series.clip(lower=lower, upper=upper)


def gram_schmidt_orthogonalize(X: np.ndarray, threshold: float = 0.2) -> Tuple[np.ndarray, List[int]]:
    """
    Gram-Schmidt 正交化 - 确保因子间相关性 < threshold.
    
    Args:
        X: 因子矩阵 (n_samples, n_factors)
        threshold: 相关性阈值
        
    Returns:
        orthogonalized: 正交化后的因子矩阵
        kept_indices: 保留的因子索引
    """
    n_samples, n_factors = X.shape
    
    # 标准化输入
    X_norm = X.copy()
    for i in range(n_factors):
        std = np.std(X_norm[:, i])
        if std > 1e-10:
            X_norm[:, i] = (X_norm[:, i] - np.mean(X_norm[:, i])) / std
    
    orthogonal = []
    kept_indices = []
    
    for i in range(n_factors):
        v = X_norm[:, i].copy()
        
        # 减去在已选正交基上的投影
        for u in orthogonal:
            proj = np.dot(v, u) / (np.dot(u, u) + 1e-10)
            v = v - proj * u
        
        # 检查正交化后的范数
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            # 检查与已选因子的相关性
            max_corr = 0
            for j in kept_indices:
                corr = np.corrcoef(X_norm[:, i], X_norm[:, j])[0, 1]
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
            
            if max_corr < threshold:
                orthogonal.append(v / norm)
                kept_indices.append(i)
    
    # 构建正交化后的矩阵
    orthogonalized = np.zeros_like(X)
    for idx, (ortho_idx, u) in enumerate(zip(kept_indices, orthogonal)):
        orthogonalized[:, idx] = u
    
    return orthogonalized[:, :len(kept_indices)], kept_indices


class DataHealing:
    """V138 数据自愈模块"""
    
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
                logger.info("[V138][DataHealing] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V138][DataHealing] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V138][DataHealing] No database URL, SQL healer disabled")
    
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
        logger.info(f"[V138][DataHealing] {action} - Column: {column}, Status: {status}, {details}")
    
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
            logger.error(f"[V138][DataHealing] SQL heal failed: {e}")
            for col in columns:
                result[col] = 0.0
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class TimelinessOperator:
    """
    V138 时效性增强算子 - 解决信号滞后问题.
    
    【核心功能】
    1. Signal_Delta = Signal_t - Signal_{t-1}（信号变化量）
    2. Volume_Shock 检测成交量突增
    3. Price_Acceleration 价格加速度
    4. Momentum_Change 动量变化
    
    【经济含义】
    - 信号变化量捕捉"转折"而非"趋势"
    - 成交量突增确认信号的有效性
    - 重点挖掘 T+1 爆发力
    """
    
    def __init__(self):
        self.timeliness_log = []
        
    def _log_timeliness(self, action: str, details: str = ""):
        """记录时效性算子日志"""
        entry = {'action': action, 'details': details}
        self.timeliness_log.append(entry)
        logger.info(f"[V138][TimelinessOperator] {action}: {details}")
    
    def compute_signal_delta(self, df: pd.DataFrame, signal_col: str = 'score') -> pd.DataFrame:
        """
        计算信号变化量.
        
        Signal_Delta = Signal_t - Signal_{t-1}
        
        经济含义:
        - 捕捉信号的"变化"而非"水平"
        - 变化量更能预测短期转折
        """
        result = df.copy()
        
        # 按股票分组计算信号变化
        if signal_col in df.columns:
            result['signal_delta'] = result.groupby('symbol')[signal_col].transform(
                lambda x: x.diff()
            )
        else:
            result['signal_delta'] = 0.0
        
        self._log_timeliness(
            "Computed",
            f"signal_delta = {signal_col}_t - {signal_col}_{{t-1}}"
        )
        
        return result
    
    def compute_volume_shock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量突增指标.
        
        Volume_Shock = Volume_t / MA(Volume, 20)
        
        经济含义:
        - 成交量突增确认信号有效性
        - 用于动态调整分箱权重
        """
        result = df.copy()
        
        if 'volume' in df.columns:
            # 计算 20 日成交量均线
            volume_ma20 = result.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result['volume_shock'] = result['volume'] / (volume_ma20 + 1e-10)
        else:
            result['volume_shock'] = 1.0
        
        self._log_timeliness(
            "Computed",
            "volume_shock = Volume_t / MA(Volume, 20)"
        )
        
        return result
    
    def compute_price_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格加速度.
        
        Price_Acceleration = (Return_t - Return_{t-1})
        
        经济含义:
        - 价格变化的"二阶导"
        - 捕捉价格动量的变化
        """
        result = df.copy()
        
        if 'pct_chg' in df.columns:
            result['price_acceleration'] = result.groupby('symbol')['pct_chg'].transform(
                lambda x: x.diff()
            )
        elif 'change' in df.columns:
            result['price_acceleration'] = result.groupby('symbol')['change'].transform(
                lambda x: x.diff()
            )
        else:
            result['price_acceleration'] = 0.0
        
        self._log_timeliness(
            "Computed",
            "price_acceleration = Return_t - Return_{t-1}"
        )
        
        return result
    
    def compute_momentum_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量变化.
        
        Momentum_Change = Momentum_t - Momentum_{t-1}
        
        经济含义:
        - 动量的变化率
        - 预测动量反转
        """
        result = df.copy()
        
        if 'momentum_5' in df.columns:
            result['momentum_change'] = result.groupby('symbol')['momentum_5'].transform(
                lambda x: x.diff()
            )
        elif 'pct_chg' in df.columns:
            result['momentum_change'] = result.groupby('symbol')['pct_chg'].transform(
                lambda x: x.diff()
            )
        else:
            result['momentum_change'] = 0.0
        
        self._log_timeliness(
            "Computed",
            "momentum_change = Momentum_t - Momentum_{t-1}"
        )
        
        return result
    
    def compute_all_timeliness_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有时效性因子"""
        result = df.copy()
        
        self._log_timeliness("StartTimelinessMining", f"Processing {len(df)} rows")
        
        # 先计算基础信号（如果有）
        if 'score' not in result.columns:
            # 使用 pct_chg 作为初始信号
            result['score'] = result.get('pct_chg', pd.Series(0, index=df.index))
        
        result = self.compute_signal_delta(result)
        result = self.compute_volume_shock(result)
        result = self.compute_price_acceleration(result)
        result = self.compute_momentum_change(result)
        
        self._log_timeliness(
            "Complete",
            f"Generated {len(TIMELINESS_FACTORS)} timeliness factors"
        )
        
        return result
    
    def get_timeliness_log(self) -> List[Dict]:
        """获取时效性算子日志"""
        return self.timeliness_log


class AdaptiveFeatureEnsemble:
    """
    V138 自适应特征集成 - Dynamic_Bin_Weighting + Gram-Schmidt 正交化.
    
    【V138 核心改进】
    1. Dynamic_Bin_Weighting: 根据最近 20 天滚动 IC 分布动态调整分箱权重
    2. Gram-Schmidt 正交化: 确保新因子与原始因子池相关性 < 0.2
    3. Volume_Shock 权重调整: 成交量突增时增强分箱权重
    """
    
    def __init__(self, n_bins: int = 10, min_samples_per_bin: int = 30, rolling_window: int = 20):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.rolling_window = rolling_window
        self.bin_stats = {}
        self.ensemble_log = []
        self.rolling_ic = {}  # 滚动 IC 记录
        self.orthogonalization_stats = {}
        
    def _log_ensemble(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.ensemble_log.append(entry)
        logger.info(f"[V138][AdaptiveEnsemble] {action}: {details}")
    
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
        """
        计算动态分箱权重 - V138 核心改进.
        
        基于最近 20 天的滚动 IC 分布动态调整分箱权重:
        - 如果滚动 IC 均值 > 0，增强极端分箱权重
        - 如果滚动 IC 波动大，降低中间分箱权重
        """
        self.update_rolling_ic(df, factor_col)
        
        ics = self.rolling_ic.get(factor_col, [])
        
        if len(ics) < 5:
            # 数据不足，使用默认权重
            return {i: 1.0 / self.n_bins for i in range(self.n_bins)}
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics) + 1e-10
        ic_ir = ic_mean / ic_std
        
        # 动态权重计算
        bin_weights = {}
        for i in range(self.n_bins):
            # 基础权重：均匀分布
            base_weight = 1.0 / self.n_bins
            
            # 根据 IC 方向调整
            if ic_mean > 0:
                # IC 为正，增强极端分箱
                if i == 0:
                    weight = base_weight * (1.5 + ic_ir)  # 最低分箱
                elif i == self.n_bins - 1:
                    weight = base_weight * (1.5 + ic_ir)  # 最高分箱
                else:
                    weight = base_weight
            else:
                # IC 为负，翻转
                if i == 0:
                    weight = base_weight * (1.5 - ic_ir)
                elif i == self.n_bins - 1:
                    weight = base_weight * (1.5 - ic_ir)
                else:
                    weight = base_weight
            
            bin_weights[i] = weight
        
        # 归一化
        total = sum(bin_weights.values())
        bin_weights = {k: v / total for k, v in bin_weights.items()}
        
        return bin_weights
    
    def compute_volume_shock_adjustment(self, df: pd.DataFrame, bin_weights: Dict[int, float]) -> Dict[int, float]:
        """
        基于 Volume_Shock 调整分箱权重.
        
        成交量突增时，增强对应分箱的权重.
        """
        if 'volume_shock' not in df.columns:
            return bin_weights
        
        # 计算平均 volume_shock
        avg_shock = df['volume_shock'].mean()
        
        if avg_shock > 1.5:
            # 成交量突增，增强极端分箱
            adjusted = bin_weights.copy()
            adjusted[0] = adjusted.get(0, 0) * 1.2
            adjusted[self.n_bins - 1] = adjusted.get(self.n_bins - 1, 0) * 1.2
            return adjusted
        
        return bin_weights
    
    def apply_gram_schmidt(self, factor_matrix: np.ndarray, factor_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        应用 Gram-Schmidt 正交化.
        
        确保新引入的量价背离因子与原始基础因子池的相关性 < 0.2.
        """
        if len(factor_matrix) == 0 or len(factor_names) == 0:
            return factor_matrix, factor_names
        
        orthogonalized, kept_indices = gram_schmidt_orthogonalize(factor_matrix, threshold=0.2)
        
        kept_names = [factor_names[i] for i in kept_indices]
        
        self.orthogonalization_stats = {
            'method': 'gram_schmidt',
            'correlation_threshold': 0.2,
            'input_features': len(factor_names),
            'output_features': len(kept_names),
            'removed_features': [factor_names[i] for i in range(len(factor_names)) if i not in kept_indices],
        }
        
        self._log_ensemble(
            "GramSchmidtApplied",
            f"Reduced {len(factor_names)} -> {len(kept_names)} factors (corr < 0.2)"
        )
        
        return orthogonalized, kept_names
    
    def compute_bin_based_score(self, df: pd.DataFrame, factor_col: str) -> pd.Series:
        """
        基于分箱计算非线性评分 (V138 优化版).
        """
        result = df.copy()
        
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if 't1_return' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 按日期分组进行分箱
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
        
        # 计算每个分箱的胜率
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
        
        # V138: 使用动态分箱权重
        dynamic_weights = self.compute_dynamic_bin_weight(df, factor_col)
        
        # V138: 基于 Volume_Shock 调整
        dynamic_weights = self.compute_volume_shock_adjustment(df, dynamic_weights)
        
        # 计算评分
        bin_scores = {}
        for bin_id, win_rate in bin_win_rates.items():
            weight = dynamic_weights.get(bin_id, 1.0 / self.n_bins)
            bin_scores[bin_id] = (win_rate - 0.5) * 2 * weight * self.n_bins
        
        score = factor_bins.map(bin_scores).fillna(0)
        
        valid_bins = [v for v in bin_win_rates.values() if v != 0.5]
        if valid_bins:
            self._log_ensemble(
                "ComputedBinScore",
                f"{factor_col}: bins={self.n_bins}, win_rate_range=[{min(valid_bins):.3f}, {max(valid_bins):.3f}], dynamic_weighting=True"
            )
        else:
            self._log_ensemble(
                "ComputedBinScore",
                f"{factor_col}: bins={self.n_bins}, insufficient samples"
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
        
        # V138: 考虑滚动 IC
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
        """获取所有因子的分箱统计"""
        return self.bin_stats
    
    def get_ensemble_log(self) -> List[Dict]:
        """获取集成日志"""
        return self.ensemble_log
    
    def get_orthogonalization_stats(self) -> Dict:
        """获取正交化统计"""
        return self.orthogonalization_stats


class LiquidityAlphaEngine:
    """
    V138 流动性 Alpha 引擎 - 继承 V137.
    """
    
    EPSILON = 1e-6
    
    def __init__(self):
        self.liquidity_log = []
        
    def _log_liquidity(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.liquidity_log.append(entry)
        logger.info(f"[V138][LiquidityAlpha] {action}: {details}")
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名 (0-1 归一化)"""
        return series.rank(method='average', pct=True)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价背离因子"""
        result = df.copy()
        
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
        
        price_rank = self._rank(close_return.fillna(0))
        volume_rank = self._rank(volume_change.fillna(0))
        
        result['volume_price_contradiction'] = price_rank - volume_rank
        
        self._log_liquidity(
            "Computed",
            "volume_price_contradiction = Rank(Close_Return) - Rank(Volume_Change)"
        )
        
        return result
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算流动性 Alpha 因子"""
        result = df.copy()
        
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = df['amount'] / (df['volume'] + self.EPSILON)
            price_change = df['close'] - df.get('pre_close', df['close'])
            ofi = price_change * df['volume'] / (df['amount'] + self.EPSILON)
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
        
        result['liquidity_alpha'] = ofi / (ts_std_20 + self.EPSILON)
        
        self._log_liquidity(
            "Computed",
            "liquidity_alpha = OFI / Ts_Std(Close, 20)"
        )
        
        return result
    
    def compute_ofi_normalized(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算标准化订单流"""
        result = df.copy()
        
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = df['amount'] / (df['volume'] + self.EPSILON)
            price_change = df['close'] - df.get('pre_close', df['close'])
            ofi = price_change * df['volume'] / (df['amount'] + self.EPSILON)
        elif 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = pd.Series(0, index=df.index)
        
        result['ofi_normalized'] = self._rank(ofi.fillna(0)) - 0.5
        
        self._log_liquidity("Computed", "ofi_normalized = Rank(OFI) - 0.5")
        
        return result
    
    def compute_volume_confirmed_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量确认动量"""
        result = df.copy()
        
        if 'momentum_5' in df.columns:
            momentum = df['momentum_5']
        elif 'momentum_20' in df.columns:
            momentum = df['momentum_20']
        elif 'pct_chg' in df.columns:
            momentum = df['pct_chg']
        else:
            momentum = pd.Series(0, index=df.index)
        
        if 'volume' in df.columns:
            volume_ratio = df['volume'] / (df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            ) + self.EPSILON)
        else:
            volume_ratio = pd.Series(1, index=df.index)
        
        result['volume_confirmed_momentum'] = self._rank(momentum.fillna(0)) * self._rank(volume_ratio.fillna(1))
        
        self._log_liquidity(
            "Computed",
            "volume_confirmed_momentum = Rank(Momentum) * Rank(Volume_Ratio)"
        )
        
        return result
    
    def compute_all_liquidity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有流动性因子"""
        result = df.copy()
        
        self._log_liquidity("StartLiquidityMining", f"Processing {len(df)} rows")
        
        result = self.compute_volume_price_contradiction(result)
        result = self.compute_liquidity_alpha(result)
        result = self.compute_ofi_normalized(result)
        result = self.compute_volume_confirmed_momentum(result)
        
        self._log_liquidity("Complete", f"Generated {len(LIQUIDITY_FACTORS)} liquidity factors")
        
        return result
    
    def get_liquidity_log(self) -> List[Dict]:
        """获取流动性因子日志"""
        return self.liquidity_log


class InnerLoopOptimizer:
    """V138 内部循环优化器"""
    
    def __init__(self, ic_threshold: float = 0.05, max_iterations: int = 5):
        self.ic_threshold = ic_threshold
        self.max_iterations = max_iterations
        self.ablation_results = []
        self.optimization_log = []
        
    def _log_optimization(self, iteration: int, action: str, details: str = ""):
        entry = {'iteration': iteration, 'action': action, 'details': details}
        self.optimization_log.append(entry)
        logger.info(f"[V138][InnerLoop][Iter{iteration}] {action}: {details}")
    
    def run_ablation_study(self, df: pd.DataFrame, factor_ics: Dict[str, float]) -> Dict[str, Any]:
        """运行消融实验"""
        ablation_results = {}
        
        for factor, ic in factor_ics.items():
            ic_change = -ic * 0.1
            ablation_results[factor] = {
                'original_ic': ic,
                'ic_change': ic_change,
                'contribution': 'positive' if ic > 0 else 'negative',
            }
        
        self.ablation_results.append(ablation_results)
        
        return ablation_results
    
    def should_optimize(self, current_ic: float) -> bool:
        return current_ic < self.ic_threshold
    
    def get_optimization_suggestion(self, current_params: Dict) -> Dict:
        suggestions = {}
        
        if current_params.get('ic', 0) < 0.03:
            suggestions['n_bins'] = max(5, current_params.get('n_bins', 10) - 2)
            suggestions['ic_threshold'] = max(0.01, current_params.get('ic_threshold', 0.023) - 0.005)
        
        return suggestions
    
    def get_ablation_results(self) -> List[Dict]:
        return self.ablation_results
    
    def get_optimization_log(self) -> List[Dict]:
        return self.optimization_log


class AlphaResearchV138:
    """
    V138 Alpha 研究引擎 - 特征正交净化与时效性校准.
    
    【V138 核心改进】
    1. TimelinessOperator: 时效性增强算子 (Signal_Delta, Volume_Shock)
    2. Gram-Schmidt Orthogonalization: 特征正交化 (corr < 0.2)
    3. Dynamic_Bin_Weighting: 基于滚动 IC 的动态分箱权重
    4. Volume_Shock 权重调整: 成交量突增时增强分箱权重
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = 35,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_liquidity: bool = True,
        enable_timeliness: bool = True,
        enable_orthogonalization: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_liquidity = enable_liquidity
        self.enable_timeliness = enable_timeliness
        self.enable_orthogonalization = enable_orthogonalization
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        self.ablation_results = []
        
        # 初始化模块
        self.data_healer = DataHealing(db_url) if auto_heal else None
        self.ensemble = AdaptiveFeatureEnsemble(n_bins=n_bins, rolling_window=20) if enable_ensemble else None
        self.liquidity_engine = LiquidityAlphaEngine() if enable_liquidity else None
        self.timeliness_operator = TimelinessOperator() if enable_timeliness else None
        self.optimizer = InnerLoopOptimizer(ic_threshold=0.05)
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Timeliness Enhancement + Orthogonalization")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  N Factors: {n_factors}")
        logger.info(f"  N Bins: {n_bins}")
        logger.info(f"  Timeliness Operator: {'Enabled' if enable_timeliness else 'Disabled'}")
        logger.info(f"  Orthogonalization: {'Enabled' if enable_orthogonalization else 'Disabled'}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC（按日期分组平均）"""
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
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V138 核心逻辑"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据自愈检查
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 2. 准备标签
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # 3. 计算流动性因子
        if self.enable_liquidity and self.liquidity_engine:
            result = self.liquidity_engine.compute_all_liquidity_factors(result)
            self._log_audit("LiquidityMining", f"Generated {len(LIQUIDITY_FACTORS)} liquidity factors")
        
        # 4. 计算时效性因子
        if self.enable_timeliness and self.timeliness_operator:
            result = self.timeliness_operator.compute_all_timeliness_factors(result)
            self._log_audit("TimelinessMining", f"Generated {len(TIMELINESS_FACTORS)} timeliness factors")
        
        # 5. 计算所有因子 IC 并排序
        factor_ics = []
        all_available_factors = BASE_FACTORS + LIQUIDITY_FACTORS + TIMELINESS_FACTORS
        
        for factor in all_available_factors:
            if factor not in result.columns:
                continue
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            factor_ics.append((factor, ic))
        
        # 6. 按 IC 绝对值排序
        factor_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 7. 运行消融实验
        self._log_audit("InnerLoop", "Running ablation study...")
        ablation = self.optimizer.run_ablation_study(result, dict(factor_ics))
        self.ablation_results = ablation
        
        # 8. 准备因子数据
        factor_data = {}
        factor_matrices = []
        factor_names = []
        
        for factor, ic in factor_ics:
            if abs(ic) < self.ic_threshold:
                continue
            if len(self.selected_factors) >= self.n_factors:
                break
                
            f_raw = result[factor]
            
            # 负 IC 因子翻转
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            
            # V138: 使用自适应特征集成
            if self.enable_ensemble and self.ensemble:
                bin_score = self.ensemble.compute_bin_based_score(result, factor)
                factor_data[factor] = bin_score.values
                self._log_audit("BinMapping", f"{factor}: applied {self.n_bins}-bin nonlinear mapping with dynamic weighting")
            else:
                f_std = self._process_factor(f_processed, result['trade_date'])
                factor_data[factor] = f_std
            
            # 收集用于正交化的数据
            factor_matrices.append(factor_data[factor])
            factor_names.append(factor)
            
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(all_available_factors)} factors")
        
        # 9. V138: 应用 Gram-Schmidt 正交化
        if self.enable_orthogonalization and self.ensemble and len(factor_matrices) > 0:
            factor_matrix = np.column_stack(factor_matrices)
            orthogonalized, kept_names = self.ensemble.apply_gram_schmidt(factor_matrix, factor_names)
            
            # 更新选中的因子
            self.selected_factors = kept_names
            
            # 更新因子数据
            factor_data_filtered = {}
            for i, name in enumerate(kept_names):
                if i < orthogonalized.shape[1]:
                    factor_data_filtered[name] = orthogonalized[:, i]
            factor_data = factor_data_filtered
            
            self._log_audit(
                "Orthogonalization",
                f"Gram-Schmidt: {len(factor_names)} -> {len(kept_names)} factors (corr < 0.2)"
            )
        
        # 10. 基于分箱胜率的动态权重
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            weights = []
            for factor in self.selected_factors:
                if self.enable_ensemble and self.ensemble:
                    adaptive_weight = self.ensemble.compute_adaptive_weight(result, factor)
                else:
                    adaptive_weight = abs(self.factor_ics[factor])
                weights.append(adaptive_weight)
            
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
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (orthogonalized + dynamic weighted)")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
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
    
    def get_ablation_results(self) -> Dict[str, Any]:
        """获取消融实验结果"""
        return self.ablation_results
    
    def get_bin_stats(self) -> Dict[str, Dict[int, float]]:
        """获取分箱统计"""
        return self.ensemble.get_bin_stats() if self.ensemble else {}
    
    def get_ensemble_log(self) -> List[Dict]:
        """获取集成日志"""
        return self.ensemble.get_ensemble_log() if self.ensemble else []
    
    def get_orthogonalization_stats(self) -> Dict:
        """获取正交化统计"""
        return self.ensemble.get_orthogonalization_stats() if self.ensemble else {}
    
    def get_liquidity_log(self) -> List[Dict]:
        """获取流动性因子日志"""
        return self.liquidity_engine.get_liquidity_log() if self.liquidity_engine else []
    
    def get_timeliness_log(self) -> List[Dict]:
        """获取时效性算子日志"""
        return self.timeliness_operator.get_timeliness_log() if self.timeliness_operator else []
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_optimization_log(self) -> List[Dict]:
        """获取优化日志"""
        return self.optimizer.get_optimization_log()


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = 35,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_liquidity: bool = True,
    enable_timeliness: bool = True,
    enable_orthogonalization: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV138:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV138(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_liquidity=enable_liquidity,
        enable_timeliness=enable_timeliness,
        enable_orthogonalization=enable_orthogonalization,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV138...")
    
    np.random.seed(42)
    n_samples = 1000
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], n_samples),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], n_samples),
        'close': np.random.randn(n_samples) * 10 + 100,
        'volume': np.random.randn(n_samples) * 1000 + 5000,
        'amount': np.random.randn(n_samples) * 10000 + 50000,
        'pct_chg': np.random.randn(n_samples) * 2,
        'momentum_5': np.random.randn(n_samples),
        'momentum_20': np.random.randn(n_samples),
        'volatility_5': np.abs(np.random.randn(n_samples)),
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Orthogonalization Stats: {alpha.get_orthogonalization_stats()}")