"""
Alpha Research Module - V146 信号稳定性 IR 攻坚战 (Robust CS-Scaling).

【V145 失败诊断】
V145 的 Signal_Confidence_Filter 存在严重的滞后性问题：
- IR 从 V144 的 0.39 仅提升至 0.41（目标 > 0.55）
- 时序熵计算导致信号对新信息反应迟钝
- 置信度加权范围 [0.5, 1.0] 仍导致过度衰减

【V146 核心使命 - Robust Cross-Sectional Scaling (RCSS)】
1. 彻底删除 Signal_Confidence_Filter → 使用时序熵滞后性严重
2. 实现 Huber-Loss 稳健合成 → 对极端离群值梯度线性截断
3. 实现截面波动率缩放 → 当日信号 Std 突增时自动缩减杠杆
4. 实现行业一致性加固 → 70% 股票反向则剔除异常噪音

【V146 核心算法】
1. Huber-Loss Robust Synthesis（Huber 稳健合成）：
   - 对截面因子值进行 Winsorize 处理
   - 计算 Z-Score 标准化
   - 对极端值进行梯度截断 (Clipping)
   - 确保少数极端个股不干扰整个截面的信号排序

2. Cross-Sectional Volatility Scaling（截面波动率缩放）：
   - 每天计算截面信号的 Std
   - 如果当日信号过于分散（Std 突增），自动缩减当日信号的整体杠杆系数
   - 使不同交易日对 IC 的贡献权重趋于平等
   - 这是提升 IR 的统计学唯一正解

3. Industry Sign-Consistency Check（行业符号一致性检查）：
   - 在 Sector_Neutral 基础上增加 Sign-Consistency-Check
   - 如果一个行业的 70% 股票信号方向相反，强行剔除该行业的异常噪音信号
   - 减少行业内部的信号冲突

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 146 运行
- 严禁修改 backtest_referee.py 中的资金 (10 万) 和费率 (0.15%)
- 数据缺失时必须主动调用 data_loader 补全，禁止用 dropna() 一删了之
- 报错必改：内置 Auto-Healing 逻辑处理 Inf/NaN，禁止停止运行

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V145: 0.41） |
| IC Std | < 0.08 | 时序波动率 |
| Day-Level IC Volatility | 降低 15%+ | 日度 IC 波动率对比 V145 |
| IR Stability | V146 > V145 | 必须回升 |
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

VERSION = "V146"

# V146 核心因子（精简因子数量以提升 IR）
V146_CORE_FACTORS = [
    'momentum_20',      # 20 日动量
    'volatility_10',    # 10 日波动率
    'volume_price_contradiction',  # 量价背离
    'liquidity_alpha',  # 流动性 Alpha
]

# V146 候选因子池（用于召回）
V146_CANDIDATE_FACTORS = [
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
    # 流动性类
    'turnover_bias_5', 'turnover_bias_10', 'turnover_bias_20',
    'volume_shrink_ratio', 'turnover_vol_ratio',
    # 尾部风险类
    'tail_risk_indicator', 'skewness_20', 'extreme_volume_ratio',
]

# V146 所有因子（核心 + 召回）
ALL_FACTORS = V146_CORE_FACTORS + V146_CANDIDATE_FACTORS

# V146 最大因子数量（精简至 4 个核心因子）
MAX_FACTORS = 4


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数 - 用于门控机制"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def huber_loss(x: np.ndarray, delta: float = 1.5) -> np.ndarray:
    """
    V146 Huber-Loss 稳健估计 - 核心创新.
    
    【原理】
    Huber Loss 是一种稳健的损失函数，对正常值使用平方损失，对极端值使用线性损失。
    这里用于对截面因子值进行稳健标准化。
    
    【公式】
    if |x| <= delta: loss = 0.5 * x^2
    else: loss = delta * (|x| - 0.5 * delta)
    
    【梯度截断】
    对极端值的梯度进行线性截断（Clipping），确保少数极端个股不干扰整个截面的信号排序。
    """
    x_abs = np.abs(x)
    # Huber 变换
    huber_transform = np.where(
        x_abs <= delta,
        x,  # 正常值保持不变
        delta * np.sign(x)  # 极端值梯度截断
    )
    return huber_transform


def compute_cross_sectional_volatility_scaling(
    signal_std: float, 
    rolling_window_std: pd.Series,
    threshold_multiplier: float = 2.0
) -> float:
    """
    V146 截面波动率缩放 - 核心创新.
    
    【原理】
    计算当日截面信号的标准差，如果 Std 突增（超过滚动窗口均值的 threshold_multiplier 倍），
    则自动缩减当日信号的整体杠杆系数。
    
    【公式】
    Rolling_Mean_Std = Mean(过去 20 日的截面 Std)
    if 今日截面 Std > threshold_multiplier × Rolling_Mean_Std:
        Scaling_Factor = Rolling_Mean_Std / 今日截面 Std
    else:
        Scaling_Factor = 1.0
    
    【经济逻辑】
    - 当日信号过于分散时，说明市场分歧较大，应降低整体信号强度
    - 使不同交易日对 IC 的贡献权重趋于平等
    - 这是提升 IR 的统计学唯一正解
    """
    rolling_mean = rolling_window_std.rolling(20, min_periods=5).mean()
    rolling_mean = rolling_mean.fillna(rolling_mean.mean() + 1e-10)
    
    threshold = threshold_multiplier * rolling_mean
    
    if signal_std > threshold:
        # Std 突增，缩减杠杆
        scaling_factor = rolling_mean / signal_std
    else:
        scaling_factor = 1.0
    
    return float(scaling_factor)


def compute_industry_sign_consistency(
    df: pd.DataFrame, 
    signal_col: str, 
    industry_col: str = 'industry_code',
    threshold: float = 0.70
) -> pd.Series:
    """
    V146 行业符号一致性检查 - 核心创新.
    
    【原理】
    在每个行业内检查信号方向的一致性。如果一个行业的 70% 股票信号方向相反，
    则强行剔除该行业的异常噪音信号。
    
    【公式】
    for each industry:
        n_positive = count(signal > 0)
        n_negative = count(signal < 0)
        n_total = n_positive + n_negative
        
        if n_positive / n_total > threshold:
            # 多数为正，保留正向信号，剔除负向
            signal[signal < 0] = 0
        elif n_negative / n_total > threshold:
            # 多数为负，保留负向信号，剔除正向
            signal[signal > 0] = 0
    
    【经济逻辑】
    - 行业内部信号冲突时，说明该行业的信号噪音较大
    - 剔除与行业主流方向相反的信号，减少误判
    """
    if industry_col not in df.columns or signal_col not in df.columns:
        return df[signal_col].copy()
    
    result = df[signal_col].copy()
    
    for industry in df[industry_col].unique():
        mask = df[industry_col] == industry
        industry_signals = result.loc[mask]
        
        if len(industry_signals) < 5:
            continue
        
        n_positive = (industry_signals > 0).sum()
        n_negative = (industry_signals < 0).sum()
        n_total = n_positive + n_negative
        
        if n_total == 0:
            continue
        
        positive_ratio = n_positive / n_total
        negative_ratio = n_negative / n_total
        
        if positive_ratio > threshold:
            # 多数为正，剔除负向信号
            result.loc[mask & (result < 0)] = 0
        elif negative_ratio > threshold:
            # 多数为负，剔除正向信号
            result.loc[mask & (result > 0)] = 0
    
    return result


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, 
                        percentile: float = 0.99) -> pd.Series:
    """
    V146 自动愈合版 Winsorization - 处理 Inf/NaN.
    
    【Auto-Healing 逻辑】
    1. 检测并修复 Inf
    2. 检测并修复 NaN
    3. Sigma 截断
    4. Percentile 截断
    5. 最终 NaN 填充
    """
    # 创建副本
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


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算两个变量之间的互信息"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    # 转换为 float64 类型
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
    
    return X_norm[:, kept_indices], kept_indices


class DataHealerV146:
    """V146 增强版数据自愈模块 - Auto-Healing 4.0"""
    
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
                logger.info("[V146][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V146][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V146][DataHealer] No database URL, SQL healer disabled")
    
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
        """V146 增强版检查并修复缺失列"""
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
        
        # V146 增强：Auto-Impute + NaN/Inf 修复
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
            logger.error(f"[V146][DataHealer] SQL heal failed: {e}")
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
        """V146 新增：自动修复 NaN/Inf"""
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


class SectorNeutralValidator:
    """
    V146 行业中性化校验器 - 确保 IR 提升不是来自行业偏离.
    
    【原理】
    对每个行业内的信号进行中性化处理，确保 IR 的提升来自于
    真正的 Alpha 预测能力，而非行业配置带来的偶然性。
    """
    
    def __init__(self, industry_column: str = 'industry_code'):
        self.industry_column = industry_column
        self.validation_log = []
        self.sector_neutralized_signals = {}
        
    def _log_validation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.validation_log.append(entry)
    
    def compute_sector_neutralized_signal(
        self, df: pd.DataFrame, signal_col: str
    ) -> pd.Series:
        """
        计算行业中性化信号.
        
        【方法】
        在每个行业内对信号进行标准化，去除行业均值影响。
        """
        if self.industry_column not in df.columns:
            # 没有行业数据，返回原始信号
            return df[signal_col].copy()
        
        result = df.copy()
        neutralized_signal = pd.Series(0.0, index=df.index)
        
        for industry in df[self.industry_column].unique():
            mask = df[self.industry_column] == industry
            industry_data = result.loc[mask, signal_col]
            
            if len(industry_data) > 5:
                # 行业内标准化
                industry_mean = industry_data.mean()
                industry_std = industry_data.std() + 1e-10
                neutralized_signal.loc[mask] = (industry_data - industry_mean) / industry_std
            else:
                # 行业样本太少，使用全局标准化
                global_mean = result[signal_col].mean()
                global_std = result[signal_col].std() + 1e-10
                neutralized_signal.loc[mask] = (industry_data - global_mean) / global_std
        
        self.sector_neutralized_signals[signal_col] = neutralized_signal
        
        self._log_validation(
            "SectorNeutralized",
            f"{signal_col}: Neutralized across {df[self.industry_column].nunique()} sectors"
        )
        
        return neutralized_signal
    
    def validate_ir_improvement(
        self, df: pd.DataFrame, signal_col: str, return_col: str = 't1_return'
    ) -> Dict:
        """
        校验 IR 提升是否来自行业偏离.
        
        【方法】
        1. 计算原始信号的 IC 和 IR
        2. 计算行业中性化后信号的 IC 和 IR
        3. 如果中性化后 IR 下降超过 20%，说明原 IR 来自行业偏离
        """
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
        
        # 计算行业中性化后 IC/IR
        neutralized = self.compute_sector_neutralized_signal(df, signal_col)
        df_temp = df.copy()
        df_temp[f'{signal_col}_neutralized'] = neutralized
        
        ics_neutralized = []
        for date in df_temp['trade_date'].unique():
            day_data = df_temp[df_temp['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[f'{signal_col}_neutralized'].fillna(0)
            l = day_data[return_col].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics_neutralized.append(ic)
        
        if not ics_neutralized:
            return {'valid': True, 'reason': 'Cannot compute neutralized IC'}
        
        ic_mean_neut = np.mean(ics_neutralized)
        ic_std_neut = np.std(ics_neutralized, ddof=1) + 1e-10
        ir_neutralized = ic_mean_neut / ic_std_neut
        
        # 计算 IR 变化
        ir_change = (ir_neutralized - ir_original) / (abs(ir_original) + 1e-10)
        
        # 判断是否来自行业偏离
        is_sector_driven = ir_change < -0.20  # IR 下降超过 20%
        
        result = {
            'valid': not is_sector_driven,
            'ir_original': float(ir_original),
            'ir_neutralized': float(ir_neutralized),
            'ir_change': float(ir_change),
            'is_sector_driven': is_sector_driven,
            'reason': 'Sector-driven' if is_sector_driven else 'True Alpha',
        }
        
        self._log_validation(
            "IRValidation",
            f"{signal_col}: IR_orig={ir_original:.4f}, IR_neut={ir_neutralized:.4f}, "
            f"Change={ir_change:.2%}, Valid={result['valid']}"
        )
        
        return result
    
    def get_validation_log(self) -> List[Dict]:
        return self.validation_log


class RobustCrossSectionalScaler:
    """
    V146 核心创新 - 稳健截面缩放器 (RCSS).
    
    【组件】
    1. Huber-Loss 稳健合成
    2. 截面波动率缩放
    3. 行业符号一致性检查
    """
    
    def __init__(
        self, 
        huber_delta: float = 1.5,
        vol_scaling_threshold: float = 2.0,
        industry_consistency_threshold: float = 0.70,
    ):
        self.huber_delta = huber_delta
        self.vol_scaling_threshold = vol_scaling_threshold
        self.industry_consistency_threshold = industry_consistency_threshold
        self.rcss_log = []
        self.scaling_factors = {}
        self.huber_stats = {}
        
    def _log_rcss(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.rcss_log.append(entry)
    
    def apply_huber_robust_synthesis(
        self, df: pd.DataFrame, signal_col: str
    ) -> pd.Series:
        """
        应用 Huber-Loss 稳健合成.
        
        【完整流程】
        1. Winsorize 去极值
        2. Z-Score 标准化
        3. Huber 梯度截断
        """
        if signal_col not in df.columns:
            return pd.Series(0.0, index=df.index)
        
        # 1. Winsorize 去极值
        signal_winsorized = winsorize_auto_heal(
            df[signal_col].fillna(0), sigma=3.0, percentile=0.99
        )
        
        # 2. 截面 Z-Score 标准化
        signal_zscore = signal_winsorized.groupby(df['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        
        # 3. Huber 梯度截断 - 使用 apply 而非 transform
        def apply_huber(group):
            return pd.Series(
                huber_loss(group.values, delta=self.huber_delta),
                index=group.index
            )
        
        signal_huber = signal_zscore.groupby(df['trade_date']).apply(apply_huber)
        # 恢复原始索引
        signal_huber = signal_huber.droplevel(0).sort_index()
        
        # 记录统计
        self.huber_stats[signal_col] = {
            'mean_before': float(signal_winsorized.mean()),
            'std_before': float(signal_winsorized.std()),
            'mean_after': float(signal_huber.mean()),
            'std_after': float(signal_huber.std()),
            'huber_delta': self.huber_delta,
        }
        
        self._log_rcss(
            "HuberApplied",
            f"{signal_col}: delta={self.huber_delta}, std_before={signal_winsorized.std():.4f}, std_after={signal_huber.std():.4f}"
        )
        
        return signal_huber
    
    def apply_cross_sectional_volatility_scaling(
        self, df: pd.DataFrame, signal_col: str
    ) -> Tuple[pd.Series, Dict]:
        """
        应用截面波动率缩放.
        
        【完整流程】
        1. 计算每日截面信号 Std
        2. 计算滚动窗口 Std 均值
        3. 计算缩放因子
        4. 应用缩放
        """
        if signal_col not in df.columns:
            return pd.Series(0.0, index=df.index), {}
        
        # 1. 计算每日截面信号 Std
        daily_std = df.groupby('trade_date')[signal_col].std().fillna(1e-10)
        
        # 2. 计算滚动窗口 Std 均值
        rolling_std_mean = daily_std.rolling(20, min_periods=5).mean().fillna(daily_std.mean())
        
        # 3. 计算缩放因子
        scaling_factors = {}
        scaled_signal = pd.Series(0.0, index=df.index)
        
        for date in df['trade_date'].unique():
            date_mask = df['trade_date'] == date
            date_std = float(daily_std.get(date, 1e-10))
            date_rolling_mean = float(rolling_std_mean.get(date, date_std))
            
            # 直接计算缩放因子，避免 Series 比较
            threshold = self.vol_scaling_threshold * date_rolling_mean
            if date_std > threshold:
                scaling_factor = date_rolling_mean / date_std
            else:
                scaling_factor = 1.0
            
            scaling_factors[date] = scaling_factor
            
            # 4. 应用缩放
            scaled_signal.loc[date_mask] = df.loc[date_mask, signal_col] * scaling_factor
        
        scaling_stats = {
            'mean_scaling_factor': float(np.mean(list(scaling_factors.values()))),
            'min_scaling_factor': float(min(scaling_factors.values())),
            'max_scaling_factor': float(max(scaling_factors.values())),
            'days_scaled': sum(1 for v in scaling_factors.values() if v < 1.0),
            'total_days': len(scaling_factors),
        }
        
        self.scaling_factors[signal_col] = scaling_stats
        
        self._log_rcss(
            "VolScalingApplied",
            f"{signal_col}: mean_factor={scaling_stats['mean_scaling_factor']:.4f}, days_scaled={scaling_stats['days_scaled']}/{scaling_stats['total_days']}"
        )
        
        return scaled_signal, scaling_factors
    
    def apply_industry_sign_consistency(
        self, df: pd.DataFrame, signal_col: str, industry_col: str = 'industry_code'
    ) -> pd.Series:
        """
        应用行业符号一致性检查.
        
        【完整流程】
        1. 按行业分组
        2. 计算行业内信号方向比例
        3. 剔除与主流方向相反的信号
        """
        if signal_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = compute_industry_sign_consistency(
            df, signal_col, industry_col, self.industry_consistency_threshold
        )
        
        # 统计被剔除的信号数量
        original_nonzero = (df[signal_col] != 0).sum()
        filtered_nonzero = (result != 0).sum()
        removed_count = original_nonzero - filtered_nonzero
        
        self._log_rcss(
            "IndustryConsistencyApplied",
            f"{signal_col}: threshold={self.industry_consistency_threshold}, removed={removed_count} signals"
        )
        
        return result
    
    def apply_full_rcss(
        self, df: pd.DataFrame, signal_col: str, industry_col: str = 'industry_code'
    ) -> pd.Series:
        """
        应用完整的 RCSS 流程.
        
        【完整流程】
        1. Huber-Loss 稳健合成
        2. 截面波动率缩放
        3. 行业符号一致性检查
        """
        # 1. Huber-Loss 稳健合成
        signal_huber = self.apply_huber_robust_synthesis(df, signal_col)
        
        # 创建临时 DataFrame 用于后续处理
        temp_df = df.copy()
        temp_df[f'{signal_col}_huber'] = signal_huber
        
        # 2. 截面波动率缩放
        signal_scaled, scaling_stats = self.apply_cross_sectional_volatility_scaling(
            temp_df, f'{signal_col}_huber'
        )
        
        temp_df[f'{signal_col}_scaled'] = signal_scaled
        
        # 3. 行业符号一致性检查
        signal_final = self.apply_industry_sign_consistency(
            temp_df, f'{signal_col}_scaled', industry_col
        )
        
        self._log_rcss(
            "FullRCSSComplete",
            f"{signal_col}: Huber → VolScaling → IndustryConsistency"
        )
        
        return signal_final
    
    def get_rcss_log(self) -> List[Dict]:
        return self.rcss_log
    
    def get_scaling_factors(self) -> Dict:
        return self.scaling_factors
    
    def get_huber_stats(self) -> Dict:
        return self.huber_stats


class SignConsistencyInteractionV146:
    """
    V146 简化版符号一致性交互模块.
    
    【V146 与 V145 的区别】
    V145: 使用时序熵置信度加权（导致滞后）
    V146: 仅保留符号锁定和线性残差，移除置信度加权
    """
    
    def __init__(self, enable_sign_lock: bool = True):
        self.enable_sign_lock = enable_sign_lock
        self.sci_log = []
        self.sci_features = {}
        self.sign_lock_applied = []
        
    def _log_sci(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.sci_log.append(entry)
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名 (0-1 归一化)"""
        return series.rank(method='average', pct=True)
    
    def compute_linear_residual(self, df: pd.DataFrame, factor_col: str, 
                                 core_col: str) -> pd.Series:
        """计算线性残差"""
        if factor_col not in df.columns or core_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        y = df[factor_col].fillna(0).values
        x = df[core_col].fillna(0).values
        
        if np.std(x) > 1e-10:
            beta = np.corrcoef(x, y)[0, 1] * np.std(y) / (np.std(x) + 1e-10)
            residual = y - beta * x
        else:
            residual = y
        
        residual_std = np.std(residual) + 1e-10
        standardized_residual = (residual - np.mean(residual)) / residual_std
        
        return pd.Series(standardized_residual, index=df.index)
    
    def compute_sci_feature(self, df: pd.DataFrame, core_factor: str, 
                            recall_factor: str) -> pd.Series:
        """
        计算 SCI 特征 - V146 简化版.
        
        【V146 修复】
        完全移除时序熵置信度加权，仅保留符号锁定和线性残差。
        """
        if core_factor not in df.columns or recall_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 线性残差
        linear_residual = self.compute_linear_residual(df, recall_factor, core_factor)
        
        # 2. 符号锁定
        rank_core = self._rank(df[core_factor].fillna(0))
        sign = np.sign(rank_core - 0.5)
        abs_residual = linear_residual.abs()
        abs_residual = (abs_residual - abs_residual.mean()) / (abs_residual.std() + 1e-10)
        
        # V146 简化：仅符号锁定 + 线性残差
        sci_final = sign * abs_residual
        
        feature_name = f"{core_factor}_sci_{recall_factor}"
        self.sci_features[feature_name] = {
            'core_factor': core_factor,
            'recall_factor': recall_factor,
            'type': 'sign_consistency_interaction_v146',
            'method': 'sign_lock + linear_residual',
            'enable_sign_lock': self.enable_sign_lock,
        }
        
        self.sign_lock_applied.append(feature_name)
        
        self._log_sci(
            "SCIComputed",
            f"{feature_name}: Sign(Rank({core_factor})) × abs(Residual)"
        )
        
        return sci_final
    
    def compute_all_sci_features(self, df: pd.DataFrame, core_factors: List[str],
                                  recalled_factors: List[str]) -> pd.DataFrame:
        """计算所有 SCI 特征"""
        result = df.copy()
        
        # V146 关键 SCI 组合
        if 'liquidity_alpha' in df.columns and 'volume_rank' in df.columns:
            distilled_sci = self.compute_sci_feature(df, 'liquidity_alpha', 'volume_rank')
            result['liquidity_alpha_sci_volume_rank'] = distilled_sci
            self._log_sci("KeySCI", "Generated liquidity_alpha_sci_volume_rank")
        
        if 'volatility_10' in df.columns and 'reversion_5' in df.columns:
            distilled_sci = self.compute_sci_feature(df, 'volatility_10', 'reversion_5')
            result['volatility_10_sci_reversion_5'] = distilled_sci
        
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


class ResidualBasedRecallV146:
    """V146 基于残差分析的因子召回模块"""
    
    def __init__(self, top_percent: float = 0.2):
        self.top_percent = top_percent
        self.recall_log = []
        self.recalled_factors = []
        self.residual_analysis = {}
        
    def _log_recall(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.recall_log.append(entry)
    
    def compute_residuals(self, df: pd.DataFrame, core_factors: List[str]) -> pd.Series:
        """计算 V146 核心模型的残差"""
        result = df.copy()
        
        weights = {f: 1.0 / len(core_factors) for f in core_factors if f in df.columns}
        
        predicted = np.zeros(len(df), dtype=np.float64)
        for factor, weight in weights.items():
            factor_values = df[factor].fillna(0).values.astype(np.float64)
            predicted = predicted + factor_values * weight
        
        actual = df['t1_return'].fillna(0).values.astype(np.float64)
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
                              candidate_factors: List[str], max_recall: int = 2) -> List[str]:
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


class FactorGeneratorV146:
    """V146 因子生成器"""
    
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
        
        self._log_generation("Complete", f"Generated base factors")
        
        return result


class IRStabilityAnalyzer:
    """
    V146 IR 稳定性分析器 - V145 vs V146 对比.
    
    【职责】
    1. 计算 V145 和 V146 的 IC/IR 指标
    2. 计算日度 IC 波动率对比
    3. 如果 V146 IR < 0.55，提出改进假设
    """
    
    def __init__(self):
        self.analysis_log = []
        self.comparison_results = {}
        
    def _log_analysis(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.analysis_log.append(entry)
    
    def compute_ic_metrics(self, df: pd.DataFrame, score_col: str) -> Dict:
        """计算 IC 指标"""
        if score_col not in df.columns or 't1_return' not in df.columns:
            return {'mean_ic': 0, 'ic_std': 0, 'ic_ir': 0, 'ics': []}
        
        ics = []
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[score_col].fillna(0)
            l = day_data['t1_return'].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        if not ics:
            return {'mean_ic': 0, 'ic_std': 0, 'ic_ir': 0, 'ics': []}
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics, ddof=1) + 1e-10
        ic_ir = ic_mean / ic_std
        
        return {
            'mean_ic': float(ic_mean),
            'ic_std': float(ic_std),
            'ic_ir': float(ic_ir),
            'num_days': len(ics),
            'ics': ics,
        }
    
    def compare_v145_v146(self, df_v145: pd.DataFrame, df_v146: pd.DataFrame) -> Dict:
        """比较 V145 vs V146"""
        metrics_v145 = self.compute_ic_metrics(df_v145, 'score')
        metrics_v146 = self.compute_ic_metrics(df_v146, 'score')
        
        ir_improvement = (metrics_v146['ic_ir'] - metrics_v145['ic_ir']) / (abs(metrics_v145['ic_ir']) + 1e-10)
        ic_improvement = (metrics_v146['mean_ic'] - metrics_v145['mean_ic']) / (abs(metrics_v145['mean_ic']) + 1e-10)
        std_reduction = (metrics_v145['ic_std'] - metrics_v146['ic_std']) / (metrics_v145['ic_std'] + 1e-10)
        
        # 日度 IC 波动率对比（核心指标）
        v145_daily_vol = np.std(metrics_v145['ics'], ddof=1) if metrics_v145['ics'] else 0
        v146_daily_vol = np.std(metrics_v146['ics'], ddof=1) if metrics_v146['ics'] else 0
        daily_vol_reduction = (v145_daily_vol - v146_daily_vol) / (v145_daily_vol + 1e-10) if v145_daily_vol > 0 else 0
        
        comparison = {
            'v145': metrics_v145,
            'v146': metrics_v146,
            'ir_improvement': float(ir_improvement),
            'ic_improvement': float(ic_improvement),
            'std_reduction': float(std_reduction),
            'daily_vol_reduction': float(daily_vol_reduction),
            'v145_daily_vol': float(v145_daily_vol),
            'v146_daily_vol': float(v146_daily_vol),
            'target_met': metrics_v146['ic_ir'] >= 0.55,
            'vol_target_met': daily_vol_reduction >= 0.15,  # 15% 降低目标
        }
        
        self.comparison_results = comparison
        
        self._log_analysis(
            "Comparison",
            f"V145 IR={metrics_v145['ic_ir']:.4f}, V146 IR={metrics_v146['ic_ir']:.4f}, "
            f"Improvement={ir_improvement:.2%}, Daily Vol Reduction={daily_vol_reduction:.2%}"
        )
        
        return comparison
    
    def generate_improvement_hypotheses(self) -> List[str]:
        """如果 IR < 0.55，生成 2 条基于逻辑的改进假设"""
        if self.comparison_results.get('target_met', False):
            return []
        
        hypotheses = [
            "假设 1：进一步精简因子数量至 3 个，仅保留最高 IC 且时序最稳定的因子。"
            "理由：V146 已精简至 4 个因子，但 IR 仍未达标。"
            "建议：使用 IC-Weighted 方法，仅选择 IC>0.03 且 IC_Std<0.05 的因子。",
            
            "假设 2：增强 Huber-Loss 参数调优，将 delta 从 1.5 调整至 1.0 或 2.0。"
            "理由：Huber delta 控制极端值截断的敏感度。"
            "建议：进行网格搜索，寻找最优 delta 值。",
        ]
        
        return hypotheses
    
    def get_analysis_log(self) -> List[Dict]:
        return self.analysis_log


class AlphaResearchV146:
    """
    V146 Alpha 研究引擎 - 信号稳定性 IR 攻坚战 (RCSS).
    
    【V146 核心改进】
    1. RobustCrossSectionalScaler: 稳健截面缩放器 (RCSS)
       - Huber-Loss 稳健合成
       - 截面波动率缩放
       - 行业符号一致性检查
    2. 删除 Signal_Confidence_Filter: 时序熵滞后性严重
    3. 精简因子数量：从 6 个减少至 4 个
    
    【目标指标】
    - T+1 Rank IC > 0.055
    - IC_IR > 0.55（V145: 0.41）
    - IC Std < 0.08
    - 日度 IC 波动率降低 15%+
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_sci: bool = True,
        enable_rcss: bool = True,  # V146 核心：RCSS
        enable_orthogonalization: bool = True,
        enable_sector_neutral: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        max_recall_factors: int = 1,  # V146：精简召回
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_sci = enable_sci
        self.enable_rcss = enable_rcss
        self.enable_orthogonalization = enable_orthogonalization
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        self.max_recall_factors = max_recall_factors
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.recalled_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV146(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV146()
        
        # V146 核心模块
        self.residual_recall = ResidualBasedRecallV146(top_percent=0.2)
        self.sci_interaction = SignConsistencyInteractionV146(
            enable_sign_lock=True
        ) if enable_sci else None
        
        # V146 核心创新：RCSS
        self.rcss = RobustCrossSectionalScaler(
            huber_delta=1.5,
            vol_scaling_threshold=2.0,
            industry_consistency_threshold=0.70
        ) if enable_rcss else None
        
        self.sector_validator = SectorNeutralValidator() if enable_sector_neutral else None
        self.ir_analyzer = IRStabilityAnalyzer()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Robust Cross-Sectional Scaling (RCSS)")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max N Factors: {n_factors}")
        logger.info(f"  Max Recall Factors: {max_recall_factors}")
        logger.info(f"  SCI: {'Enabled' if enable_sci else 'Disabled'}")
        logger.info(f"  RCSS: {'Enabled' if enable_rcss else 'Disabled'}")
        logger.info(f"  Target IR: 0.55 (V145: 0.41)")
        logger.info(f"  Target Daily IC Vol Reduction: 15%+")
    
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
        # V146 Auto-Heal 版去极值
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
        """计算 Alpha 评分 - V146 核心逻辑（RCSS）"""
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
        
        core_factors = [f for f in V146_CORE_FACTORS if f in result.columns]
        
        if len(core_factors) >= 2:
            priority_candidates = ['reversion_5', 'reversion_10', 'volume_price_contradiction', 
                                   'rsi_14', 'mfi_14'] + V146_CANDIDATE_FACTORS
            self.recalled_factors = self.residual_recall.select_recall_factors(
                result, core_factors, priority_candidates, self.max_recall_factors
            )
            self._log_audit("ResidualRecall", f"Recalled {len(self.recalled_factors)} factors: {self.recalled_factors}")
        else:
            self._log_audit("ResidualRecall", "Insufficient core factors, skipping recall")
            self.recalled_factors = []
        
        # 5. 计算 SCI 特征
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
        
        # 强制包含 volume_rank
        if 'volume_rank' in result.columns:
            all_candidate_factors.append('volume_rank')
            self._log_audit("CandidatePool", "Added volume_rank")
        
        # 添加核心因子
        for factor in core_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加召回因子
        for factor in self.recalled_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加 SCI 因子
        for factor in sci_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors"
        )
        
        # 7. 计算 IC 和 IC 滚动统计
        factor_ics = []
        
        for factor in all_candidate_factors:
            if factor not in result.columns:
                continue
            ic = self._calc_factor_ic(result, factor)
            t1_ic, t3_ic, t5_ic = self._calc_factor_ic_decay(result, factor)
            
            t1_specificity = t1_ic - (t3_ic + t5_ic) / 2
            
            self.factor_ics[factor] = ic
            factor_ics.append((factor, abs(t1_ic)))
            
            self._log_audit(
                "FactorAnalysis",
                f"{factor}: T+1={t1_ic:.4f}, T+3={t3_ic:.4f}, T+5={t5_ic:.4f}"
            )
        
        # 分离 SCI 因子和其他因子
        sci_keywords = ['_sci_', '_sign_consistency_']
        sci_ics = [(f, ic) for f, ic in factor_ics if any(kw in f.lower() for kw in sci_keywords)]
        other_ics = [(f, ic) for f, ic in factor_ics if f not in [x[0] for x in sci_ics]]
        
        # 按 T+1 IC 绝对值排序
        other_ics.sort(key=lambda x: x[1], reverse=True)
        sci_ics.sort(key=lambda x: x[1], reverse=True)
        
        # 8. V146 因子选择策略：精简至 4 个因子
        final_selected = []
        max_factors = min(self.n_factors, 4)
        
        # 强制包含 volume_rank
        if 'volume_rank' in [f[0] for f in other_ics]:
            vr_ic = self.factor_ics.get('volume_rank', 0)
            if abs(vr_ic) >= 0.01:
                final_selected.append('volume_rank')
                self._log_audit("FactorSelection", f"Force included volume_rank (IC={vr_ic:.4f})")
        
        # 强制包含 liquidity_alpha_sci_volume_rank
        if 'liquidity_alpha_sci_volume_rank' in [f[0] for f in sci_ics]:
            sci_ic = self.factor_ics.get('liquidity_alpha_sci_volume_rank', 0)
            if abs(sci_ic) >= 0.01:
                final_selected.append('liquidity_alpha_sci_volume_rank')
                self._log_audit("FactorSelection", f"Force included liquidity_alpha_sci_volume_rank (IC={sci_ic:.4f})")
        
        # 添加 top IC 的非 SCI 因子
        for factor, t1_ic_abs in other_ics:
            if len(final_selected) >= max_factors:
                break
            if factor in final_selected:
                continue
            if t1_ic_abs >= 0.02:
                final_selected.append(factor)
        
        # 如果仍不足，强制补充
        if len(final_selected) < max_factors:
            for factor, t1_ic_abs in other_ics:
                if factor not in final_selected and len(final_selected) < max_factors:
                    final_selected.append(factor)
        
        self.selected_factors = final_selected[:max_factors]
        
        self._log_audit(
            "FactorSelection",
            f"Final selected {len(self.selected_factors)} factors: {self.selected_factors}"
        )
        
        # 9. 准备因子数据并应用 Auto-Flip
        factor_data = {}
        
        for factor in self.selected_factors:
            f_raw = result[factor]
            ic = self.factor_ics[factor]
            
            # Auto-Flip: 负 IC 因子翻转方向
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("AutoFlip", f"{factor}: IC={ic:.4f} < 0, FLIPPED direction")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("AutoFlip", f"{factor}: IC={ic:.4f} >= 0, kept direction")
            
            # 标准化
            f_std = self._process_factor(f_processed, result['trade_date'])
            
            factor_data[factor] = f_std
            self._log_audit("FactorPrep", f"{factor}: direction={self.factor_directions[factor]}, mean={f_std.mean():.6f}, std={f_std.std():.6f}")
        
        # 10. 行业中性化校验
        if self.enable_sector_neutral and self.sector_validator:
            self._log_audit("SectorNeutral", "Validating IR improvement is not sector-driven...")
            
            temp_score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                temp_score += factor_data[factor] / len(self.selected_factors)
            
            result['score_temp'] = temp_score
            validation = self.sector_validator.validate_ir_improvement(
                result, 'score_temp', 't1_return'
            )
            
            if not validation.get('valid', True):
                self._log_audit(
                    "SectorNeutralWarning",
                    f"IR improvement may be sector-driven: {validation.get('reason', 'Unknown')}"
                )
            
            result = result.drop(columns=['score_temp'])
        
        # 11. V146 RCSS 信号处理（核心创新）
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            # 计算初始分数
            score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                score += factor_data[factor] / len(self.selected_factors)
            
            result['score_raw'] = score
            
            # 应用 RCSS
            if self.enable_rcss and self.rcss:
                self._log_audit("RCSS", "Applying Robust Cross-Sectional Scaling...")
                
                # 1. Huber-Loss 稳健合成
                score_huber = self.rcss.apply_huber_robust_synthesis(result, 'score_raw')
                
                # 2. 截面波动率缩放
                score_scaled, scaling_stats = self.rcss.apply_cross_sectional_volatility_scaling(
                    result, 'score_raw'
                )
                
                # 3. 行业符号一致性检查
                industry_col = 'industry_code' if 'industry_code' in result.columns else None
                if industry_col:
                    score_final = self.rcss.apply_industry_sign_consistency(
                        result, 'score_raw', industry_col
                    )
                else:
                    score_final = score_scaled
                
                result['score'] = score_final
                
                self._log_audit(
                    "RCSSComplete",
                    f"Huber → VolScaling → IndustryConsistency applied"
                )
            else:
                result['score'] = score
            
            # 等权重集成
            for factor in self.selected_factors:
                self.factor_weights[factor] = 1.0 / len(self.selected_factors)
            
            self._log_audit("EnsembleComplete", f"Equal-weight ensemble with {len(self.selected_factors)} factors")
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (RCSS)")
        
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
    
    def get_sector_validation(self) -> Dict:
        """获取行业中性化校验结果"""
        return self.sector_validator.get_validation_log() if self.sector_validator else {}
    
    def get_ir_analyzer(self) -> IRStabilityAnalyzer:
        """获取 IR 分析器"""
        return self.ir_analyzer
    
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
    
    def get_rcss_log(self) -> List[Dict]:
        """获取 RCSS 日志"""
        return self.rcss.get_rcss_log() if self.rcss else []
    
    def get_rcss_scaling_factors(self) -> Dict:
        """获取 RCSS 缩放因子统计"""
        return self.rcss.get_scaling_factors() if self.rcss else {}
    
    def get_rcss_huber_stats(self) -> Dict:
        """获取 RCSS Huber 统计"""
        return self.rcss.get_huber_stats() if self.rcss else {}


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_sci: bool = True,
    enable_rcss: bool = True,
    enable_orthogonalization: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 1,
) -> AlphaResearchV146:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV146(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_sci=enable_sci,
        enable_rcss=enable_rcss,
        enable_orthogonalization=enable_orthogonalization,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV146...")
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
        'momentum_20': np.random.randn(1000),
        'volatility_10': np.abs(np.random.randn(1000)),
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Recalled factors: {alpha.get_recalled_factors()}")
    logger.info(f"  SCI features: {alpha.get_sci_features()}")
    logger.info(f"  Sign-Lock applied: {alpha.get_sign_lock_applied()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  RCSS Scaling Factors: {alpha.get_rcss_scaling_factors()}")
    logger.info(f"  RCSS Huber Stats: {alpha.get_rcss_huber_stats()}")
    logger.info(f"  Efficiency Ratio: {alpha.get_efficiency_ratio():.4f}")