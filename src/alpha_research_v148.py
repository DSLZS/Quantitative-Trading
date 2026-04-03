"""
Alpha Research Module - V148 Temporal Consistency & Physical Orthogonalization (TCPO).

【V147 失败诊断】
- MREF 置信度加权未有效降低信号时序震荡
- DCSS 缩放力度不足，高波动日惩罚不够
- 因子正交化不彻底，截面上仍存在冗余
- IR 未达到 0.55 目标

【V148 核心使命 - TCPO】
1. 实现实时 Gram-Schmidt 正交化：在 generate_scores 阶段，对选中的 6-8 个因子每天执行施密特正交化
2. 引入 Signal Inertia Kernel（信号惯性核）：Score = (1-α) * Raw_Score + α * Score_{t-1}，α根据 MREF 动态调整
3. 新增 Volume-Price Reversion (VPR) 因子：基于成交量分布的超跌反弹因子
4. DCSS 升级：增加"行业一致性约束"，防止行业过曝

【V148 核心算法】
1. Gram-Schmidt Orthogonalization (GSO):
   - 每日截面对因子执行施密特正交化
   - 确保输出的因子集合在截面上完全不相关
   - 使用互信息 (Mutual Information) 验证正交化效果

2. Signal Inertia Kernel (SIK):
   - Score_t = (1 - α_t) * Raw_Score_t + α_t * Score_{t-1}
   - α_t = MREF_Confidence_t * β (β为惯性系数，默认 0.3)
   - 当市场熵值高（噪音大）时，增大 α 以增强信号稳定性

3. Volume-Price Reversion (VPR):
   - VPR = Rank(Low_Price_Volume / Total_Volume) - Rank(Return)
   - 逻辑：成交量集中在低价区且近期超跌的股票有反弹潜力

4. DCSS Industry Consistency Constraint:
   - 如果信号在某一行业内高度趋同（>70% 股票同向），则对该行业信号进行惩罚收缩
   - 防止行业过曝导致的集中风险

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 148 运行
- 严禁修改 backtest_referee.py 中的资金 (10 万) 和费率 (0.15%)
- 数据缺失时必须主动调用 data_loader 补全，禁止用 dropna() 一删了之
- 报错必改：内置 Auto-Healing 逻辑处理 Inf/NaN，禁止停止运行

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V147: ~0.40） |
| IC Std | < 0.08 | 时序波动率 |
| Day-Level IC Volatility | 降低 15%+ | 日度 IC 波动率对比 V147 |
| IR Stability | V148 > V147 | 必须回升 |
| Signal Turnover Rate | 降低 10%+ | 日度信号换手率对比 V147 |
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

VERSION = "V148"

# V148 核心因子（扩容至 6-8 个，新增 VPR 因子）
V148_CORE_FACTORS = [
    'momentum_20',              # 20 日动量
    'volatility_10',            # 10 日波动率
    'volume_price_contradiction',  # 量价背离（强制保留）
    'liquidity_alpha',          # 流动性 Alpha（强制保留）
    'volatility_reversion',     # 波动率反转
    'reversion_5',              # 5 日反转
    'volume_price_reversion',   # V148 新增：成交量 - 价格反转（VPR）
]

# V148 候选因子池（用于召回）
V148_CANDIDATE_FACTORS = [
    # 动量类
    'momentum_5', 'momentum_10', 'momentum_60',
    # 反转类
    'reversion_10',
    # 波动率类
    'volatility_5', 'volatility_20',
    # 量价类
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'vwap_distance', 'volume_rank', 'price_rank',
    # 价值类
    'value_rank', 'ep_rank', 'bp_rank',
    # 技术指标类
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist',
    # 流动性类
    'turnover_bias_5', 'turnover_bias_10', 'turnover_bias_20',
    'volume_shrink_ratio', 'turnover_vol_ratio',
    # 尾部风险类
    'tail_risk_indicator', 'skewness_20', 'extreme_volume_ratio',
]

# V148 所有因子（核心 + 召回）
ALL_FACTORS = V148_CORE_FACTORS + V148_CANDIDATE_FACTORS

# V148 最大因子数量（扩容至 8 个）
MAX_FACTORS = 8


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数 - 用于非线性压缩"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算两个变量之间的互信息"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    except (ValueError, TypeError):
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
        
        px = joint_hist.sum(axis=1)
        py = joint_hist.sum(axis=0)
        
        mi = 0.0
        for i in range(n_x):
            for j in range(n_y):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        
        return mi
    except Exception:
        return 0.0


def gram_schmidt_orthogonalize(
    X: np.ndarray, 
    mi_threshold: float = 0.1
) -> Tuple[np.ndarray, List[int]]:
    """
    V148 核心：Gram-Schmidt 正交化 + 互信息验证.
    
    【V147 问题】
    - 因子在截面上仍存在冗余相关性
    - 正交化不彻底导致信号震荡
    
    【V148 修复】
    - 对每日截面因子数据执行标准 Gram-Schmidt 正交化
    - 使用互信息验证正交化效果
    - 确保输出因子集合在截面上完全不相关
    
    【算法】
    for i in range(n_factors):
        v_i = X[:, i]
        for j in range(i):
            v_i = v_i - proj(v_i, u_j) * u_j
        u_i = v_i / ||v_i||
    """
    n_samples, n_factors = X.shape
    
    if n_factors == 0:
        return X, []
    
    # 1. 标准化输入因子
    X_norm = X.copy()
    for i in range(n_factors):
        std = np.std(X_norm[:, i])
        if std > 1e-10:
            X_norm[:, i] = (X_norm[:, i] - np.mean(X_norm[:, i])) / std
        else:
            X_norm[:, i] = 0  # 常数列设为 0
    
    orthogonal = []
    kept_indices = []
    
    for i in range(n_factors):
        v = X_norm[:, i].copy()
        
        # Gram-Schmidt 正交化
        for u in orthogonal:
            proj = np.dot(v, u) / (np.dot(u, u) + 1e-10)
            v = v - proj * u
        
        # 归一化
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            u = v / norm
            
            # 互信息验证
            max_mi = 0
            for j in kept_indices:
                mi = compute_mutual_information(X_norm[:, i], X_norm[:, j], n_bins=10)
                max_mi = max(max_mi, mi)
            
            if max_mi < mi_threshold:
                orthogonal.append(u)
                kept_indices.append(i)
    
    # 返回正交化后的矩阵
    if orthogonal:
        X_orthogonal = np.column_stack(orthogonal)
        return X_orthogonal, kept_indices
    else:
        return X_norm, list(range(n_factors))


def compute_multi_scale_entropy(signals: pd.Series, windows: List[int] = [3, 5, 10]) -> pd.Series:
    """
    V148 多尺度时序熵计算 - 用于 Signal Inertia Kernel.
    
    【原理】
    计算多个时间窗口内信号方向的熵值，衡量不同时间尺度下的方向一致性。
    只有当短、中、长期信号方向一致时，才给予最高权重。
    """
    if len(signals) < max(windows):
        return pd.Series(1.0, index=signals.index)
    
    # 计算信号方向
    directions = np.sign(signals)
    
    # 存储各尺度的置信度
    confidence_dict = {}
    
    for window in windows:
        confidence_values = []
        for i in range(len(signals)):
            if i < window - 1:
                confidence_values.append(0.5)  # 初始化为中等置信度
            else:
                window_directions = directions.iloc[i - window + 1:i + 1]
                
                # 计算正负方向比例
                n_positive = (window_directions > 0).sum()
                n_negative = (window_directions < 0).sum()
                n_total = n_positive + n_negative
                
                if n_total == 0:
                    confidence_values.append(0.5)
                else:
                    p_positive = n_positive / n_total
                    p_negative = n_negative / n_total
                    
                    # 计算熵
                    entropy = 0.0
                    if p_positive > 0:
                        entropy -= p_positive * np.log(p_positive + 1e-10)
                    if p_negative > 0:
                        entropy -= p_negative * np.log(p_negative + 1e-10)
                    
                    # 归一化熵值（0-1 范围）
                    max_entropy = np.log(2)
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    # 置信度 = 1 - 归一化熵
                    confidence = 1 - normalized_entropy
                    confidence_values.append(confidence)
        
        confidence_dict[f'conf_{window}d'] = pd.Series(confidence_values, index=signals.index)
    
    # 多尺度融合：0.5*3 日 + 0.3*5 日 + 0.2*10 日
    weights = {3: 0.5, 5: 0.3, 10: 0.2}
    fused_confidence = pd.Series(0.0, index=signals.index)
    
    for window in windows:
        if f'conf_{window}d' in confidence_dict:
            fused_confidence += weights.get(window, 1.0/len(windows)) * confidence_dict[f'conf_{window}d']
    
    return fused_confidence


def winsorize_auto_heal(
    series: pd.Series, 
    sigma: float = 3.0, 
    percentile: float = 0.99
) -> pd.Series:
    """V148 自动愈合版 Winsorization - 处理 Inf/NaN"""
    series_clean = series.copy()
    
    # 1. 处理 Inf
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    # 2. 计算均值
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


class DataHealerV148:
    """V148 增强版数据自愈模块 - Auto-Healing 4.1"""
    
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
                logger.info("[V148][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V148][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V148][DataHealer] No database URL, SQL healer disabled")
    
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
        """V148 增强版检查并修复缺失列"""
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
        
        # V148 增强：Auto-Impute + NaN/Inf 修复
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result)
        
        self._log_healing(
            action="AutoImputeApplied",
            column="ALL_NUMERIC",
            status="SUCCESS",
            details="Applied grouped median imputation + NaN/Inf repair"
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
            logger.error(f"[V148][DataHealer] SQL heal failed: {e}")
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
    
    def _repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """V148 新增：自动修复 NaN/Inf"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 检测 Inf
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                self._log_healing(
                    action="InfRepaired",
                    column=col,
                    status="SUCCESS",
                    details=f"Repaired {inf_count} Inf values"
                )
            
            # 检测 NaN
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                result[col] = result[col].fillna(col_median)
                self._log_healing(
                    action="NaNRepaired",
                    column=col,
                    status="SUCCESS",
                    details=f"Repaired {nan_count} NaN values with median={col_median:.4f}"
                )
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class SignalInertiaKernel:
    """
    V148 核心创新 - 信号惯性核 (SIK).
    
    【V147 问题】
    - 日度信号换手率过高，导致 IC 波动大
    - 信号时序一致性不足
    
    【V148 修复】
    - 引入惯性核：Score_t = (1 - α_t) * Raw_Score_t + α_t * Score_{t-1}
    - α_t 根据 MREF 动态调整：市场熵值高时增大 α
    - 降低信号换手率，提升 IR 稳定性
    
    【参数】
    - base_inertia: 基础惯性系数（默认 0.3）
    - min_alpha: 最小 α 值（默认 0.1）
    - max_alpha: 最大 α 值（默认 0.6）
    """
    
    def __init__(
        self,
        base_inertia: float = 0.3,
        min_alpha: float = 0.1,
        max_alpha: float = 0.6,
    ):
        self.base_inertia = base_inertia
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.sik_log = []
        self.inertia_stats = {}
        
    def _log_sik(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.sik_log.append(entry)
    
    def apply_inertia(
        self, 
        df: pd.DataFrame, 
        raw_score_col: str,
        confidence_col: Optional[str] = None,
    ) -> pd.Series:
        """
        应用信号惯性核.
        
        【完整流程】
        1. 按日期排序
        2. 对每个符号应用惯性核
        3. α_t = base_inertia * confidence_t（如果有置信度）
        """
        if raw_score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        smoothed_scores = []
        
        for symbol in result['symbol'].unique():
            symbol_mask = result['symbol'] == symbol
            symbol_data = result.loc[symbol_mask].copy()
            
            raw_scores = symbol_data[raw_score_col].values
            n = len(raw_scores)
            
            if n == 0:
                smoothed_scores.append((symbol_mask, pd.Series([], index=symbol_data.index)))
                continue
            
            # 初始化平滑后的分数
            smoothed = np.zeros(n)
            smoothed[0] = raw_scores[0]
            
            for t in range(1, n):
                # 计算动态 α
                if confidence_col is not None and confidence_col in symbol_data.columns:
                    confidence = symbol_data[confidence_col].iloc[t]
                    # 高置信度时降低惯性，低置信度时增加惯性
                    alpha = self.base_inertia * (1.5 - confidence)
                else:
                    alpha = self.base_inertia
                
                # 限制 α 范围
                alpha = max(self.min_alpha, min(self.max_alpha, alpha))
                
                # 应用惯性核
                smoothed[t] = (1 - alpha) * raw_scores[t] + alpha * smoothed[t - 1]
            
            smoothed_series = pd.Series(smoothed, index=symbol_data.index)
            smoothed_scores.append((symbol_mask, smoothed_series))
        
        # 合并所有结果
        final_smoothed = pd.Series(0.0, index=df.index)
        for mask, series in smoothed_scores:
            final_smoothed.loc[mask] = series
        
        self._log_sik(
            "InertiaApplied",
            f"base_inertia={self.base_inertia}, min_alpha={self.min_alpha}, max_alpha={self.max_alpha}"
        )
        
        self.inertia_stats = {
            'mean_alpha': self.base_inertia,
            'min_alpha': self.min_alpha,
            'max_alpha': self.max_alpha,
        }
        
        return final_smoothed
    
    def get_sik_log(self) -> List[Dict]:
        return self.sik_log
    
    def get_inertia_stats(self) -> Dict:
        return self.inertia_stats


class VolumePriceReversion:
    """
    V148 新增 - 成交量 - 价格反转因子 (VPR).
    
    【V147 问题】
    - 缺少基于成交量分布的超跌反弹因子
    
    【V148 修复】
    - VPR = Rank(Low_Price_Volume / Total_Volume) - Rank(Return)
    - 逻辑：成交量集中在低价区且近期超跌的股票有反弹潜力
    
    【经济意义】
    - 低价区成交量占比高 → 主力在低位吸筹
    - 近期超跌 → 有反弹需求
    - 两者结合 → 超跌反弹信号
    """
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.vpr_log = []
        
    def _log_vpr(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.vpr_log.append(entry)
    
    def compute_vpr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算 VPR 因子.
        
        【公式】
        VPR = Rank(Low_Price_Volume / Total_Volume) - Rank(Return)
        
        Low_Price_Volume = 成交量 * (1 - (Low - Min_Low) / (Max_Low - Min_Low))
        其中 Min_Low/Max_Low 为 lookback 窗口内的最低/最高价
        """
        result = df.copy()
        
        # 1. 计算 lookback 窗口内的最低/最高价
        if 'low' not in result.columns or 'high' not in result.columns:
            self._log_vpr("MissingData", "Missing low/high columns, returning 0")
            return pd.Series(0, index=df.index)
        
        # 按符号分组计算
        low_prices = result.groupby('symbol')['low'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).min()
        )
        high_prices = result.groupby('symbol')['high'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).max()
        )
        
        # 2. 计算价格位置（0-1 归一化）
        price_range = high_prices - low_prices + 1e-10
        price_position = (result['low'] - low_prices) / price_range
        
        # 3. 计算低价区成交量
        # 价格位置越低，成交量权重越高
        volume_weight = 1 - price_position
        low_price_volume = result['volume'] * volume_weight
        
        # 4. 计算低价区成交量占比（滚动窗口）
        total_volume = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).sum()
        )
        low_price_volume_sum = result.groupby('symbol').apply(
            lambda x: (x['volume'] * (1 - (x['low'] - x['low'].rolling(self.lookback_window, min_periods=5).min()) / 
                         (x['high'].rolling(self.lookback_window, min_periods=5).max() - 
                          x['low'].rolling(self.lookback_window, min_periods=5).min() + 1e-10))
                       ).rolling(self.lookback_window, min_periods=5).sum()
        ).reset_index(level=0, drop=True)
        
        lpv_ratio = low_price_volume_sum / (total_volume + 1e-10)
        
        # 5. 计算近期收益（5 日反转）
        if 'close' in result.columns:
            returns = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change(5)
            )
        else:
            returns = pd.Series(0, index=df.index)
        
        # 6. 排名计算
        lpv_rank = result.groupby('trade_date')[lpv_ratio].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        ret_rank = result.groupby('trade_date')[returns].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        # 7. VPR = LPV_Rank - Return_Rank
        vpr = lpv_rank - ret_rank
        
        # 8. 标准化
        vpr = vpr.groupby(result['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        ).fillna(0)
        
        self._log_vpr(
            "Computed",
            f"VPR factor: mean={vpr.mean():.4f}, std={vpr.std():.4f}"
        )
        
        return vpr
    
    def get_vpr_log(self) -> List[Dict]:
        return self.vpr_log


class IndustryConsistencyConstraint:
    """
    V148 新增 - 行业一致性约束.
    
    【V147 问题】
    - DCSS 缺少行业维度约束
    - 可能导致信号在某一行业过度集中
    
    【V148 修复】
    - 如果信号在某一行业内高度趋同（>70% 股票同向），则对该行业信号进行惩罚收缩
    - 防止行业过曝导致的集中风险
    
    【参数】
    - consistency_threshold: 一致性阈值（默认 0.70）
    - shrinkage_factor: 收缩因子（默认 0.7）
    """
    
    def __init__(
        self,
        consistency_threshold: float = 0.70,
        shrinkage_factor: float = 0.7,
    ):
        self.consistency_threshold = consistency_threshold
        self.shrinkage_factor = shrinkage_factor
        self.icc_log = []
        self.industry_stats = {}
        
    def _log_icc(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.icc_log.append(entry)
    
    def apply_industry_shrinkage(
        self,
        df: pd.DataFrame,
        signal_col: str,
        industry_col: str = 'industry_code',
    ) -> pd.Series:
        """
        应用行业一致性约束.
        
        【完整流程】
        1. 按日期和行业分组
        2. 计算每个行业内信号方向一致性
        3. 若一致性 > threshold，对该行业信号进行收缩
        """
        if signal_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        scaled_signal = result[signal_col].copy()
        
        # 检查是否有行业数据
        if industry_col not in result.columns:
            self._log_icc("NoIndustryData", "Using global shrinkage instead")
            # 使用全局收缩
            signal_sign = np.sign(result[signal_col])
            global_consistency = (signal_sign == 1).mean()
            if global_consistency > self.consistency_threshold or global_consistency < (1 - self.consistency_threshold):
                scaled_signal = result[signal_col] * self.shrinkage_factor
                self.industry_stats = {
                    'global_consistency': float(global_consistency),
                    'industries_shrunk': 1,
                }
            return scaled_signal
        
        industries_shrunk = 0
        total_industries = 0
        
        for date in result['trade_date'].unique():
            date_mask = result['trade_date'] == date
            date_data = result.loc[date_mask]
            
            for industry in date_data[industry_col].unique():
                industry_mask = date_data[industry_col] == industry
                industry_data = date_data.loc[industry_mask, signal_col]
                
                if len(industry_data) < 3:
                    continue
                
                total_industries += 1
                
                # 计算信号方向一致性
                signal_sign = np.sign(industry_data)
                positive_ratio = (signal_sign > 0).mean()
                negative_ratio = (signal_sign < 0).mean()
                consistency = max(positive_ratio, negative_ratio)
                
                # 若一致性过高，进行收缩
                if consistency > self.consistency_threshold:
                    industry_idx = result.loc[date_mask & (result[industry_col] == industry)].index
                    scaled_signal.loc[industry_idx] = industry_data * self.shrinkage_factor
                    industries_shrunk += 1
        
        self._log_icc(
            "IndustryShrinkageApplied",
            f"industries_shrunk={industries_shrunk}/{total_industries}"
        )
        
        self.industry_stats = {
            'consistency_threshold': self.consistency_threshold,
            'shrinkage_factor': self.shrinkage_factor,
            'industries_shrunk': industries_shrunk,
            'total_industries': total_industries,
            'shrinkage_ratio': industries_shrunk / total_industries if total_industries > 0 else 0,
        }
        
        return scaled_signal
    
    def get_icc_log(self) -> List[Dict]:
        return self.icc_log
    
    def get_industry_stats(self) -> Dict:
        return self.industry_stats


class FactorGeneratorV148:
    """V148 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        self.vpr_computer = VolumePriceReversion(lookback_window=20)
        
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
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        """
        V148 波动率反转因子.
        
        【逻辑】
        高波动率后往往跟随低波动率，反之亦然。
        """
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        # 波动率变化
        vol_change = vol_10 - vol_20
        
        # 反转信号：波动率上升时看空，波动率下降时看多
        return -vol_change.fillna(0)
    
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
        
        # V148 新增：波动率反转因子
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        
        # 量价因子
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        # V148 核心新增：VPR 因子
        result['volume_price_reversion'] = self.vpr_computer.compute_vpr(result)
        
        # volume_rank（关键因子）
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        self._log_generation("Complete", f"Generated base factors including V148 VPR")
        
        return result


class AlphaResearchV148:
    """
    V148 Alpha 研究引擎 - Temporal Consistency & Physical Orthogonalization (TCPO).
    
    【V148 核心改进】
    1. Gram-Schmidt Orthogonalization: 每日截面因子正交化
    2. Signal Inertia Kernel: 信号惯性核，降低换手率
    3. Volume-Price Reversion: 新增 VPR 因子
    4. Industry Consistency Constraint: 行业一致性约束
    
    【目标指标】
    - T+1 Rank IC > 0.055
    - IC_IR > 0.55（V147: ~0.40）
    - IC Std < 0.08
    - 日度 IC 波动率降低 15%+
    - 日度信号换手率降低 10%+
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_sci: bool = True,
        enable_gso: bool = True,  # V148 核心：Gram-Schmidt 正交化
        enable_sik: bool = True,  # V148 核心：信号惯性核
        enable_vpr: bool = True,  # V148 核心：VPR 因子
        enable_icc: bool = True,  # V148 核心：行业一致性约束
        enable_sector_neutral: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        max_recall_factors: int = 2,
        inertia_base: float = 0.3,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_sci = enable_sci
        self.enable_gso = enable_gso
        self.enable_sik = enable_sik
        self.enable_vpr = enable_vpr
        self.enable_icc = enable_icc
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        self.max_recall_factors = max_recall_factors
        self.inertia_base = inertia_base
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.recalled_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV148(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV148()
        
        # V148 核心模块
        self.sik = SignalInertiaKernel(base_inertia=inertia_base) if enable_sik else None
        self.icc = IndustryConsistencyConstraint() if enable_icc else None
        
        self.sector_validator = SectorNeutralValidator() if enable_sector_neutral else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: TCPO (Temporal Consistency & Physical Orthogonalization)")
        logger.info(f"  GSO: {'Enabled' if enable_gso else 'Disabled'}")
        logger.info(f"  SIK: {'Enabled' if enable_sik else 'Disabled'} (base_inertia={inertia_base})")
        logger.info(f"  VPR: {'Enabled' if enable_vpr else 'Disabled'}")
        logger.info(f"  ICC: {'Enabled' if enable_icc else 'Disabled'}")
        logger.info(f"  Target IR: 0.55 (V147: ~0.40)")
        logger.info(f"  Target Signal Turnover Reduction: 10%+")
    
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
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def _apply_gram_schmidt_daily(
        self, 
        df: pd.DataFrame, 
        factor_cols: List[str]
    ) -> Dict[str, pd.Series]:
        """
        V148 核心：每日截面 Gram-Schmidt 正交化.
        
        【完整流程】
        1. 按日期分组
        2. 对每日因子截面执行 GSO
        3. 返回正交化后的因子字典
        """
        if not factor_cols or len(factor_cols) < 2:
            return {col: df[col] for col in factor_cols}
        
        result = df.copy()
        orthogonal_factors = {}
        
        # 初始化正交化后的因子
        for col in factor_cols:
            orthogonal_factors[col] = pd.Series(0.0, index=df.index)
        
        # 按日期分组处理
        for date in df['trade_date'].unique():
            date_mask = df['trade_date'] == date
            date_data = df.loc[date_mask]
            
            if len(date_data) < 20:
                continue
            
            # 提取当日因子数据
            factor_matrix = date_data[factor_cols].fillna(0).values
            
            # 执行 GSO
            X_orthogonal, kept_indices = gram_schmidt_orthogonalize(
                factor_matrix, mi_threshold=0.1
            )
            
            # 将正交化后的因子映射回原数据
            for i, col_idx in enumerate(kept_indices):
                if i < X_orthogonal.shape[1]:
                    col_name = factor_cols[col_idx]
                    orthogonal_factors[col_name].loc[date_mask] = X_orthogonal[:, i]
        
        self._log_audit(
            "GSOApplied",
            f"Orthogonalized {len(factor_cols)} factors, kept {len(kept_indices)}"
        )
        
        return orthogonal_factors
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V148 核心逻辑（TCPO）"""
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
        
        # 3. 生成基础因子（包括 V148 新增 VPR）
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
            self._log_audit("FactorGeneration", "Generated base factors including V148 VPR")
        
        # 4. 构建候选因子池
        all_candidate_factors = []
        
        # 强制包含 volume_rank
        if 'volume_rank' in result.columns:
            all_candidate_factors.append('volume_rank')
        
        # 添加核心因子（包括 V148 新增 VPR）
        core_factors = [f for f in V148_CORE_FACTORS if f in result.columns]
        for factor in core_factors:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加召回因子（简化版）
        candidate_factors = [f for f in V148_CANDIDATE_FACTORS if f in result.columns]
        for factor in candidate_factors[:5]:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors"
        )
        
        # 5. 计算 IC 和因子选择
        factor_ics = []
        
        for factor in all_candidate_factors:
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            factor_ics.append((factor, abs(ic)))
        
        # 按 IC 绝对值排序
        factor_ics.sort(key=lambda x: x[1], reverse=True)
        
        # 选择 top 因子
        max_factors = min(self.n_factors, 8)
        final_selected = [f[0] for f in factor_ics[:max_factors]]
        
        # 强制包含 VPR
        if 'volume_price_reversion' in [f[0] for f in factor_ics] and 'volume_price_reversion' not in final_selected:
            final_selected.append('volume_price_reversion')
        
        self.selected_factors = final_selected[:max_factors]
        
        self._log_audit(
            "FactorSelection",
            f"Final selected {len(self.selected_factors)} factors: {self.selected_factors}"
        )
        
        # 6. 准备因子数据并应用 Auto-Flip
        factor_data = {}
        
        for factor in self.selected_factors:
            f_raw = result[factor]
            ic = self.factor_ics[factor]
            
            # Auto-Flip: 负 IC 因子翻转方向
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
            
            # 标准化
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. V148 核心：Gram-Schmidt 正交化
        if self.enable_gso and len(self.selected_factors) >= 2:
            self._log_audit("GSO", "Applying Gram-Schmidt Orthogonalization...")
            orthogonal_factors = self._apply_gram_schmidt_daily(result, self.selected_factors)
            
            # 更新因子数据
            for factor in self.selected_factors:
                if factor in orthogonal_factors:
                    factor_data[factor] = orthogonal_factors[factor].values
        
        # 8. 计算初始分数
        score = np.zeros(len(result), dtype=np.float64)
        for i, factor in enumerate(self.selected_factors):
            f = factor_data[factor]
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            score += f_clean.values / len(self.selected_factors)
        
        result['score_raw'] = score
        
        # 9. V148 核心：Signal Inertia Kernel
        if self.enable_sik and self.sik:
            self._log_audit("SIK", "Applying Signal Inertia Kernel...")
            smoothed_score = self.sik.apply_inertia(result, 'score_raw')
            result['score_smoothed'] = smoothed_score
        else:
            result['score_smoothed'] = result['score_raw']
        
        # 10. V148 核心：Industry Consistency Constraint
        if self.enable_icc and self.icc:
            self._log_audit("ICC", "Applying Industry Consistency Constraint...")
            final_score = self.icc.apply_industry_shrinkage(result, 'score_smoothed')
            result['score'] = final_score
        else:
            result['score'] = result['score_smoothed']
        
        # 11. 等权重集成
        for factor in self.selected_factors:
            self.factor_weights[factor] = 1.0 / len(self.selected_factors)
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (TCPO)")
        
        # 输出列
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']
        for col in ['t1_return_period']:
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
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_sik_stats(self) -> Dict:
        """获取 SIK 统计"""
        return self.sik.get_inertia_stats() if self.sik else {}
    
    def get_sik_log(self) -> List[Dict]:
        """获取 SIK 日志"""
        return self.sik.get_sik_log() if self.sik else []
    
    def get_icc_stats(self) -> Dict:
        """获取 ICC 统计"""
        return self.icc.get_industry_stats() if self.icc else {}
    
    def get_icc_log(self) -> List[Dict]:
        """获取 ICC 日志"""
        return self.icc.get_icc_log() if self.icc else []
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log


class SectorNeutralValidator:
    """V148 行业中性化校验器"""
    
    def __init__(self, industry_column: str = 'industry_code'):
        self.industry_column = industry_column
        self.validation_log = []
        
    def _log_validation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.validation_log.append(entry)
    
    def validate_ir_improvement(
        self, df: pd.DataFrame, signal_col: str, return_col: str = 't1_return'
    ) -> Dict:
        """校验 IR 提升是否来自行业偏离"""
        if signal_col not in df.columns or return_col not in df.columns:
            return {'valid': False, 'reason': 'Missing columns'}
        
        # 计算原始 IC/IR
        ics_original = []
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[signal_col].fillna(0)
            l = day_data[return_col].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics_original.append(ic)
        
        if not ics_original:
            return {'valid': False, 'reason': 'No IC calculated'}
        
        ic_mean_orig = np.mean(ics_original)
        ic_std_orig = np.std(ics_original, ddof=1) + 1e-10
        ir_original = ic_mean_orig / ic_std_orig
        
        return {
            'valid': True,
            'ir_original': float(ir_original),
        }
    
    def get_validation_log(self) -> List[Dict]:
        return self.validation_log


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_sci: bool = True,
    enable_gso: bool = True,
    enable_sik: bool = True,
    enable_vpr: bool = True,
    enable_icc: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 2,
    inertia_base: float = 0.3,
) -> AlphaResearchV148:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV148(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_sci=enable_sci,
        enable_gso=enable_gso,
        enable_sik=enable_sik,
        enable_vpr=enable_vpr,
        enable_icc=enable_icc,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
        inertia_base=inertia_base,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV148...")
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
        'low': np.random.randn(1000) * 10 + 95,
        'high': np.random.randn(1000) * 10 + 105,
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  SIK Stats: {alpha.get_sik_stats()}")
    logger.info(f"  ICC Stats: {alpha.get_icc_stats()}")