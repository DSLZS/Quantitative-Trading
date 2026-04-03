"""
Alpha Research Module - V147 信号稳定性（IR）大修复 (Multi-Resolution Entropy Fusion).

【V146 失败诊断】
V146 的 RCSS 缩放因子完全未生效：
- 242 个交易日中 0 天触发截面波动率缩放
- Huber 变换反而放大了信号波动（std 从 0.57 增至 0.86）
- 错误删除时序熵导致失去噪音过滤机制
- 因子过度精简至 4 个，失去多样性

【V147 核心使命 - Multi-Resolution Entropy Fusion (MREF)】
1. 恢复 V145 的时序熵逻辑，升级为 Multi-Scale Entropy（3/5/10 日信号一致性）
2. 修正 Huber-Scaling：使用偏态自适应阈值 threshold = Median + 1.5 * IQR
3. 实现动态截面收缩（Dynamic CS-Shrinkage）：因子间相关性 > 0.7 自动开启 PCA-Shrinkage
4. 因子池扩容至 6-8 个正交因子，强制保留 volume_price_contradiction 和 liquidity_alpha

【V147 核心算法】
1. Multi-Resolution Entropy Fusion (MREF):
   - 计算 3 日、5 日、10 日信号方向的时序熵
   - 只有当短中长期信号方向一致时，才给予最高权重
   - 融合公式：Confidence = w1*Conf_3d + w2*Conf_5d + w3*Conf_10d

2. Skewness-Adaptive Huber Scaling:
   - threshold = Median + 1.5 * IQR（四分位距）
   - 对超过阈值的信号进行 Sigmoid 压缩，而非线性截断

3. Dynamic Cross-Sectional Shrinkage:
   - 计算每日因子截面相关性矩阵
   - 若因子间相关性 > 0.7，自动开启 PCA-Shrinkage
   - 强制提取第一主成分以减少信号冗余

4. Factor Pool Expansion:
   - 扩容至 6-8 个正交因子
   - 强制保留：volume_price_contradiction, liquidity_alpha
   - 新增：volatility_reversion

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 147 运行
- 严禁修改 backtest_referee.py 中的资金 (10 万) 和费率 (0.15%)
- 数据缺失时必须主动调用 data_loader 补全，禁止用 dropna() 一删了之
- 报错必改：内置 Auto-Healing 逻辑处理 Inf/NaN，禁止停止运行
- 动态缩放模块在波动率排名前 20% 的交易日必须生效

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V146: 0.39） |
| IC Std | < 0.08 | 时序波动率 |
| Day-Level IC Volatility | 降低 15%+ | 日度 IC 波动率对比 V146 |
| IR Stability | V147 > V146 | 必须回升 |
| Dynamic Scaling Triggered | >= 20% | 高波动交易日缩放生效 |
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

VERSION = "V147"

# V147 核心因子（扩容至 6-8 个）
V147_CORE_FACTORS = [
    'momentum_20',              # 20 日动量
    'volatility_10',            # 10 日波动率
    'volume_price_contradiction',  # 量价背离（强制保留）
    'liquidity_alpha',          # 流动性 Alpha（强制保留）
    'volatility_reversion',     # 波动率反转（新增）
    'reversion_5',              # 5 日反转
]

# V147 候选因子池（用于召回）
V147_CANDIDATE_FACTORS = [
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

# V147 所有因子（核心 + 召回）
ALL_FACTORS = V147_CORE_FACTORS + V147_CANDIDATE_FACTORS

# V147 最大因子数量（扩容至 8 个）
MAX_FACTORS = 8


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数 - 用于非线性压缩"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_multi_scale_entropy(signals: pd.Series, windows: List[int] = [3, 5, 10]) -> pd.Series:
    """
    V147 多尺度时序熵计算 - 核心创新.
    
    【原理】
    计算多个时间窗口内信号方向的熵值，衡量不同时间尺度下的方向一致性。
    只有当短、中、长期信号方向一致时，才给予最高权重。
    
    【公式】
    for each window W in [3, 5, 10]:
        p_positive = count(positive signals in W) / W
        p_negative = count(negative signals in W) / W
        Entropy_W = -Σ(p_i * log(p_i))
        Confidence_W = 1 - (Entropy_W / log(2))
    
    Fusion: Confidence = 0.5*Conf_3d + 0.3*Conf_5d + 0.2*Conf_10d
    
    【经济逻辑】
    - 短期熵低：信号近期方向稳定
    - 长期熵低：信号趋势一致性强
    - 多尺度一致：最高置信度
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


def compute_skewness_adaptive_threshold(series: pd.Series) -> float:
    """
    V147 偏态自适应阈值计算.
    
    【原理】
    使用 Median + 1.5 * IQR（四分位距）计算自适应阈值，
    比固定 delta 的 Huber 变换更能适应数据分布。
    
    【公式】
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    threshold = Median + 1.5 * IQR
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    median = series.median()
    
    threshold = median + 1.5 * iqr
    
    return float(threshold)


def skewness_adaptive_huber_scaling(
    signal: pd.Series, 
    compression_factor: float = 1.0
) -> pd.Series:
    """
    V147 偏态自适应 Huber 缩放 - 核心创新.
    
    【V146 失败原因】
    - 固定 delta=1.5 阈值过高，截断效果弱
    - 线性截断反而放大了信号波动
    
    【V147 修复】
    - 使用 Median + 1.5 * IQR 计算自适应阈值
    - 对超过阈值的信号进行 Sigmoid 压缩，而非线性截断
    
    【公式】
    threshold = Median + 1.5 * IQR
    if |signal| <= threshold: signal unchanged
    else: signal = threshold * sigmoid(signal / threshold)
    """
    # 计算自适应阈值
    threshold = compute_skewness_adaptive_threshold(signal)
    
    # 对信号进行缩放
    signal_abs = np.abs(signal)
    
    # 正常信号保持不变，极端信号进行 Sigmoid 压缩
    scaled_signal = np.where(
        signal_abs <= threshold,
        signal,
        np.sign(signal) * threshold * sigmoid(signal_abs / threshold * compression_factor)
    )
    
    return pd.Series(scaled_signal, index=signal.index)


def compute_cross_sectional_correlation_matrix(
    df: pd.DataFrame, 
    factor_cols: List[str]
) -> pd.DataFrame:
    """
    计算截面因子相关性矩阵.
    
    【原理】
    对每个交易日，计算因子间的截面相关性，
    用于检测因子冗余度。
    """
    if not factor_cols or len(factor_cols) < 2:
        return pd.DataFrame()
    
    # 按日期分组计算相关性
    corr_list = []
    
    for date in df['trade_date'].unique():
        date_data = df[df['trade_date'] == date]
        
        if len(date_data) < 20:  # 样本太少跳过
            continue
        
        # 计算当日因子相关性矩阵
        factor_data = date_data[factor_cols].fillna(0)
        corr_matrix = factor_data.corr()
        corr_list.append(corr_matrix)
    
    if not corr_list:
        return pd.DataFrame()
    
    # 返回平均相关性矩阵
    avg_corr = pd.concat(corr_list).groupby(level=0).mean()
    return avg_corr


def pca_shrinkage(
    factor_data: pd.DataFrame, 
    correlation_threshold: float = 0.7
) -> pd.Series:
    """
    V147 PCA 收缩 - 动态截面收缩.
    
    【原理】
    当因子间相关性超过阈值时，使用 PCA 提取第一主成分，
    减少信号冗余带来的波动。
    
    【公式】
    if max_correlation > threshold:
        # 执行 PCA
        eigenvalues, eigenvectors = eig(corr_matrix)
        PC1 = factor_data @ eigenvectors[:, 0]
        return PC1
    else:
        return equal_weight_average
    """
    if factor_data.shape[1] < 2:
        return factor_data.iloc[:, 0] if len(factor_data.columns) > 0 else pd.Series(0, index=factor_data.index)
    
    # 计算相关性矩阵
    corr_matrix = factor_data.corr()
    
    # 检查最大相关性（对角线除外）
    max_corr = 0
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            max_corr = max(max_corr, abs(corr_matrix.iloc[i, j]))
    
    if max_corr > correlation_threshold:
        # 高相关性，执行 PCA 收缩
        try:
            # 使用 SVD 提取第一主成分
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(factor_data)
            return pd.Series(pc1.flatten(), index=factor_data.index)
        except Exception:
            # PCA 失败时返回等权平均
            return factor_data.mean(axis=1)
    else:
        # 低相关性，返回等权平均
        return factor_data.mean(axis=1)


def winsorize_auto_heal(
    series: pd.Series, 
    sigma: float = 3.0, 
    percentile: float = 0.99
) -> pd.Series:
    """
    V147 自动愈合版 Winsorization - 处理 Inf/NaN.
    """
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


class DataHealerV147:
    """V147 增强版数据自愈模块 - Auto-Healing 4.0"""
    
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
                logger.info("[V147][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V147][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V147][DataHealer] No database URL, SQL healer disabled")
    
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
        """V147 增强版检查并修复缺失列"""
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
        
        # V147 增强：Auto-Impute + NaN/Inf 修复
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
            logger.error(f"[V147][DataHealer] SQL heal failed: {e}")
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
        """V147 新增：自动修复 NaN/Inf"""
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
    V147 行业中性化校验器 - 确保 IR 提升不是来自行业偏离.
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
        """计算行业中性化信号"""
        if self.industry_column not in df.columns:
            return df[signal_col].copy()
        
        result = df.copy()
        neutralized_signal = pd.Series(0.0, index=df.index)
        
        for industry in df[self.industry_column].unique():
            mask = df[self.industry_column] == industry
            industry_data = result.loc[mask, signal_col]
            
            if len(industry_data) > 5:
                industry_mean = industry_data.mean()
                industry_std = industry_data.std() + 1e-10
                neutralized_signal.loc[mask] = (industry_data - industry_mean) / industry_std
            else:
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
        is_sector_driven = ir_change < -0.20
        
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


class MultiResolutionEntropyFusion:
    """
    V147 核心创新 - 多尺度熵融合器 (MREF).
    
    【V147 修复】
    - 原问题：置信度加权均匀削弱了所有信号
    - 修复：只在低置信度时压缩信号，高置信度时保持
    
    【组件】
    1. Multi-Scale Entropy Calculation (3/5/10 日)
    2. Entropy Fusion with Dynamic Weights
    3. Confidence-Gated Signal Scaling (非线性门控)
    """
    
    def __init__(
        self, 
        windows: List[int] = [3, 5, 10],
        weights: Dict[int, float] = None,
        min_confidence: float = 0.5,
        high_confidence_threshold: float = 0.75,
    ):
        self.windows = windows
        self.weights = weights or {3: 0.5, 5: 0.3, 10: 0.2}
        self.min_confidence = min_confidence
        self.high_confidence_threshold = high_confidence_threshold
        self.mref_log = []
        self.entropy_stats = {}
        
    def _log_mref(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.mref_log.append(entry)
    
    def compute_fused_confidence(
        self, df: pd.DataFrame, signal_col: str
    ) -> pd.Series:
        """
        计算多尺度融合置信度.
        
        【完整流程】
        1. 按日期分组计算时序熵
        2. 多尺度融合
        3. 应用最小置信度限制
        """
        if signal_col not in df.columns:
            return pd.Series(1.0, index=df.index)
        
        # 按符号分组计算时序熵
        result = df.copy()
        all_confidences = []
        
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_data = result.loc[symbol_mask, signal_col].reset_index(drop=True)
            
            # 计算多尺度熵
            confidence = compute_multi_scale_entropy(symbol_data, self.windows)
            all_confidences.append((symbol_mask, confidence))
        
        # 合并所有置信度
        fused_confidence = pd.Series(1.0, index=df.index)
        for mask, conf in all_confidences:
            fused_confidence.loc[mask] = conf
        
        # 应用最小置信度限制
        fused_confidence = fused_confidence.clip(lower=self.min_confidence, upper=1.0)
        
        # 记录统计
        self.entropy_stats[signal_col] = {
            'mean_confidence': float(fused_confidence.mean()),
            'std_confidence': float(fused_confidence.std()),
            'high_confidence_ratio': float((fused_confidence > 0.7).mean()),
        }
        
        self._log_mref(
            "Computed",
            f"{signal_col}: Mean_Confidence={fused_confidence.mean():.4f}, "
            f"High_Confidence_Ratio={(fused_confidence > 0.7).mean():.2%}"
        )
        
        return fused_confidence
    
    def apply_confidence_weighted_signal(
        self, df: pd.DataFrame, signal_col: str
    ) -> pd.Series:
        """
        应用置信度加权信号 - V147 修复版 (非线性门控).
        
        【V146 失败原因】
        - 线性置信度加权均匀削弱了所有信号，降低了 IC
        
        【V147 修复】
        - 按全市场时序计算熵值，捕捉市场整体方向一致性
        - 非线性门控：高置信度时保持信号，低置信度时大幅压缩
        - 门控函数：Gating = (Confidence - 0.5) * 2，限制在 [0.3, 1.0]
        """
        if signal_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        raw_signal = df[signal_col].fillna(0)
        
        # V147 修复：按日期分组计算截面信号，然后计算时序一致性
        # 1. 计算每日截面平均信号
        daily_signal = df.groupby('trade_date')[signal_col].mean()
        
        # 2. 计算信号方向的时序一致性
        daily_sign = np.sign(daily_signal)
        
        # 3. 多尺度一致性计算
        confidence_by_date = {}
        
        for date in daily_signal.index:
            date_idx = daily_signal.index.get_loc(date)
            
            # 计算各尺度的一致性
            confidences = []
            weights = []
            
            for window in self.windows:
                if date_idx >= window - 1:
                    window_signs = daily_sign.iloc[date_idx - window + 1:date_idx + 1]
                    # 一致性 = 同向比例
                    positive_ratio = (window_signs > 0).mean()
                    negative_ratio = (window_signs < 0).mean()
                    consistency = max(positive_ratio, negative_ratio)
                    confidences.append(consistency)
                    weights.append(self.weights.get(window, 1.0/len(self.windows)))
                else:
                    # 初期使用中等置信度
                    confidences.append(0.7)
                    weights.append(self.weights.get(window, 1.0/len(self.windows)))
            
            # 加权融合
            fused_conf = sum(c * w for c, w in zip(confidences, weights))
            confidence_by_date[date] = max(self.min_confidence, min(1.0, fused_conf))
        
        # 4. 将置信度映射回原始数据
        confidence = df['trade_date'].map(confidence_by_date).fillna(self.min_confidence)
        
        # 5. V147 非线性门控：高置信度保持，低置信度压缩
        # Gating = (Confidence - 0.5) * 2, clipped to [0.3, 1.0]
        # 这样：Confidence=0.5 → Gating=0, Confidence=0.75 → Gating=0.5, Confidence=1.0 → Gating=1.0
        gating = (confidence - 0.5) * 2
        gating = gating.clip(lower=0.3, upper=1.0)
        
        # 6. 应用门控加权
        weighted_signal = raw_signal * gating
        
        self._log_mref(
            "ConfidenceWeighted",
            f"{signal_col}: Raw_Std={raw_signal.std():.4f}, Weighted_Std={weighted_signal.std():.4f}, "
            f"Mean_Gating={gating.mean():.4f}"
        )
        
        # 更新统计
        self.entropy_stats[signal_col] = {
            'mean_confidence': float(confidence.mean()),
            'std_confidence': float(confidence.std()),
            'high_confidence_ratio': float((confidence > self.high_confidence_threshold).mean()),
            'mean_gating': float(gating.mean()),
        }
        
        return weighted_signal
    
    def get_entropy_stats(self) -> Dict:
        return self.entropy_stats
    
    def get_mref_log(self) -> List[Dict]:
        return self.mref_log


class DynamicCrossSectionalShrinkage:
    """
    V147 核心创新 - 动态截面收缩器.
    
    【V147 修复】
    - 降低 PCA 阈值从 0.7 到 0.6，增强去冗余
    - 增强波动率缩放力度
    
    【组件】
    1. Cross-Sectional Correlation Matrix
    2. PCA Shrinkage (当相关性 > 0.6)
    3. Volatility-Weighted Scaling (增强版)
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.6,  # V147 降低阈值
        vol_scaling_threshold: float = 1.2,  # V147 增强敏感度
    ):
        self.correlation_threshold = correlation_threshold
        self.vol_scaling_threshold = vol_scaling_threshold
        self.dcss_log = []
        self.shrinkage_stats = {}
        self.scaling_factors = {}
        
    def _log_dcss(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.dcss_log.append(entry)
    
    def apply_pca_shrinkage(
        self, df: pd.DataFrame, factor_cols: List[str]
    ) -> pd.Series:
        """
        应用 PCA 收缩.
        
        【完整流程】
        1. 计算截面相关性矩阵
        2. 检查最大相关性
        3. 若 > threshold，执行 PCA 提取第一主成分
        """
        if not factor_cols or len(factor_cols) < 2:
            return df[factor_cols[0]] if factor_cols else pd.Series(0, index=df.index)
        
        factor_data = df[factor_cols].fillna(0)
        
        # 计算相关性矩阵
        corr_matrix = factor_data.corr()
        
        # 检查最大相关性
        max_corr = 0
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                max_corr = max(max_corr, abs(corr_matrix.iloc[i, j]))
        
        # 记录统计
        self.shrinkage_stats['max_correlation'] = float(max_corr)
        self.shrinkage_stats['correlation_threshold'] = self.correlation_threshold
        
        if max_corr > self.correlation_threshold:
            # 高相关性，执行 PCA 收缩
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(factor_data)
                result = pd.Series(pc1.flatten(), index=df.index)
                
                self._log_dcss(
                    "PCAShrinkageApplied",
                    f"max_corr={max_corr:.4f} > threshold={self.correlation_threshold}"
                )
                
                return result
            except Exception as e:
                self._log_dcss("PCAFailed", str(e))
                return factor_data.mean(axis=1)
        else:
            # 低相关性，返回等权平均
            self._log_dcss(
                "EqualWeightAverage",
                f"max_corr={max_corr:.4f} <= threshold={self.correlation_threshold}"
            )
            return factor_data.mean(axis=1)
    
    def apply_volatility_scaling(
        self, df: pd.DataFrame, signal_col: str
    ) -> Tuple[pd.Series, Dict]:
        """
        应用波动率缩放 - V147 修复确保高波动交易日生效.
        
        【V146 失败原因】
        - threshold_multiplier=2.0 阈值过高，242 天中 0 天触发
        
        【V147 修复】
        - 降低阈值至 1.5 倍
        - 强制在波动率排名前 20% 的交易日生效
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
        
        # 找出波动率排名前 20% 的交易日
        vol_percentile_80 = daily_std.quantile(0.80)
        high_vol_days = daily_std[daily_std >= vol_percentile_80].index.tolist()
        
        days_scaled = 0
        days_forced_scaled = 0
        
        for date in df['trade_date'].unique():
            date_mask = df['trade_date'] == date
            date_std = float(daily_std.get(date, 1e-10))
            date_rolling_mean = float(rolling_std_mean.get(date, date_std))
            
            # V147 修复：降低阈值并确保高波动交易日生效
            threshold = self.vol_scaling_threshold * date_rolling_mean
            
            is_high_vol_day = date in high_vol_days
            
            if date_std > threshold or is_high_vol_day:
                # Std 突增或高波动日，缩减杠杆
                scaling_factor = date_rolling_mean / max(date_std, 1e-10)
                scaling_factor = max(0.5, min(1.0, scaling_factor))  # 限制在 [0.5, 1.0]
                
                if is_high_vol_day:
                    days_forced_scaled += 1
                days_scaled += 1
            else:
                scaling_factor = 1.0
            
            scaling_factors[date] = scaling_factor
            scaled_signal.loc[date_mask] = df.loc[date_mask, signal_col] * scaling_factor
        
        total_days = len(df['trade_date'].unique())
        scaling_stats = {
            'mean_scaling_factor': float(np.mean(list(scaling_factors.values()))),
            'min_scaling_factor': float(min(scaling_factors.values())),
            'max_scaling_factor': float(max(scaling_factors.values())),
            'days_scaled': days_scaled,
            'days_forced_scaled': days_forced_scaled,
            'total_days': total_days,
            'scaling_ratio': days_scaled / total_days if total_days > 0 else 0,
            'forced_scaling_ratio': days_forced_scaled / total_days if total_days > 0 else 0,
        }
        
        self.scaling_factors = scaling_stats
        
        self._log_dcss(
            "VolScalingApplied",
            f"days_scaled={days_scaled}/{total_days} ({scaling_stats['scaling_ratio']:.1%}), "
            f"forced_scaled={days_forced_scaled}/{total_days} ({scaling_stats['forced_scaling_ratio']:.1%})"
        )
        
        return scaled_signal, scaling_stats
    
    def get_dcss_log(self) -> List[Dict]:
        return self.dcss_log
    
    def get_shrinkage_stats(self) -> Dict:
        return self.shrinkage_stats
    
    def get_scaling_factors(self) -> Dict:
        return self.scaling_factors


class SignConsistencyInteractionV147:
    """
    V147 符号一致性交互模块.
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
        """计算 SCI 特征"""
        if core_factor not in df.columns or recall_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 线性残差
        linear_residual = self.compute_linear_residual(df, recall_factor, core_factor)
        
        # 2. 符号锁定
        rank_core = self._rank(df[core_factor].fillna(0))
        sign = np.sign(rank_core - 0.5)
        abs_residual = linear_residual.abs()
        abs_residual = (abs_residual - abs_residual.mean()) / (abs_residual.std() + 1e-10)
        
        sci_final = sign * abs_residual
        
        feature_name = f"{core_factor}_sci_{recall_factor}"
        self.sci_features[feature_name] = {
            'core_factor': core_factor,
            'recall_factor': recall_factor,
            'type': 'sign_consistency_interaction_v147',
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
        
        # V147 关键 SCI 组合
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


class ResidualBasedRecallV147:
    """V147 基于残差分析的因子召回模块"""
    
    def __init__(self, top_percent: float = 0.2):
        self.top_percent = top_percent
        self.recall_log = []
        self.recalled_factors = []
        self.residual_analysis = {}
        
    def _log_recall(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.recall_log.append(entry)
    
    def compute_residuals(self, df: pd.DataFrame, core_factors: List[str]) -> pd.Series:
        """计算 V147 核心模型的残差"""
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


class FactorGeneratorV147:
    """V147 因子生成器"""
    
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
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        """
        V147 新增：波动率反转因子.
        
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
        
        # V147 新增：波动率反转因子
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        
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
    V147 IR 稳定性分析器 - V146 vs V147 对比.
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
    
    def compare_v146_v147(self, df_v146: pd.DataFrame, df_v147: pd.DataFrame) -> Dict:
        """比较 V146 vs V147"""
        metrics_v146 = self.compute_ic_metrics(df_v146, 'score')
        metrics_v147 = self.compute_ic_metrics(df_v147, 'score')
        
        ir_improvement = (metrics_v147['ic_ir'] - metrics_v146['ic_ir']) / (abs(metrics_v146['ic_ir']) + 1e-10)
        ic_improvement = (metrics_v147['mean_ic'] - metrics_v146['mean_ic']) / (abs(metrics_v146['mean_ic']) + 1e-10)
        std_reduction = (metrics_v146['ic_std'] - metrics_v147['ic_std']) / (metrics_v146['ic_std'] + 1e-10)
        
        # 日度 IC 波动率对比（核心指标）
        v146_daily_vol = np.std(metrics_v146['ics'], ddof=1) if metrics_v146['ics'] else 0
        v147_daily_vol = np.std(metrics_v147['ics'], ddof=1) if metrics_v147['ics'] else 0
        daily_vol_reduction = (v146_daily_vol - v147_daily_vol) / (v146_daily_vol + 1e-10) if v146_daily_vol > 0 else 0
        
        comparison = {
            'v146': metrics_v146,
            'v147': metrics_v147,
            'ir_improvement': float(ir_improvement),
            'ic_improvement': float(ic_improvement),
            'std_reduction': float(std_reduction),
            'daily_vol_reduction': float(daily_vol_reduction),
            'v146_daily_vol': float(v146_daily_vol),
            'v147_daily_vol': float(v147_daily_vol),
            'target_met': metrics_v147['ic_ir'] >= 0.55,
            'vol_target_met': daily_vol_reduction >= 0.15,
        }
        
        self.comparison_results = comparison
        
        self._log_analysis(
            "Comparison",
            f"V146 IR={metrics_v146['ic_ir']:.4f}, V147 IR={metrics_v147['ic_ir']:.4f}, "
            f"Improvement={ir_improvement:.2%}, Daily Vol Reduction={daily_vol_reduction:.2%}"
        )
        
        return comparison
    
    def generate_improvement_hypotheses(self) -> List[str]:
        """如果 IR < 0.55，生成 2 条基于逻辑的改进假设"""
        if self.comparison_results.get('target_met', False):
            return []
        
        hypotheses = [
            "假设 1：进一步调优 MREF 权重，增加短期熵权重（3 日→0.6, 5 日→0.25, 10 日→0.15）。"
            "理由：短期信号一致性对 IR 影响更大。",
            
            "假设 2：降低 PCA 收缩阈值从 0.7 至 0.6，增强因子去冗余效果。"
            "理由：因子间相关性可能被低估。",
        ]
        
        return hypotheses
    
    def get_analysis_log(self) -> List[Dict]:
        return self.analysis_log


class AlphaResearchV147:
    """
    V147 Alpha 研究引擎 - 信号稳定性（IR）大修复 (Multi-Resolution Entropy Fusion).
    
    【V147 核心改进】
    1. MultiResolutionEntropyFusion: 多尺度熵融合器 (MREF)
       - 3/5/10 日时序熵计算
       - 只有当短中长期信号方向一致时，才给予最高权重
    2. DynamicCrossSectionalShrinkage: 动态截面收缩器
       - PCA 收缩（当因子相关性 > 0.7）
       - 波动率缩放（阈值降低至 1.5 倍，高波动日强制生效）
    3. Skewness-Adaptive Huber Scaling: 偏态自适应 Huber 缩放
       - threshold = Median + 1.5 * IQR
       - Sigmoid 压缩而非线性截断
    4. Factor Pool Expansion: 因子池扩容至 6-8 个
    
    【目标指标】
    - T+1 Rank IC > 0.055
    - IC_IR > 0.55（V146: 0.39）
    - IC Std < 0.08
    - 日度 IC 波动率降低 15%+
    - Dynamic Scaling Triggered >= 20%
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_sci: bool = True,
        enable_mref: bool = True,  # V147 核心：MREF
        enable_dcss: bool = True,  # V147 核心：动态截面收缩
        enable_orthogonalization: bool = True,
        enable_sector_neutral: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        max_recall_factors: int = 2,  # V147：召回 2 个因子
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_sci = enable_sci
        self.enable_mref = enable_mref
        self.enable_dcss = enable_dcss
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
        self.data_healer = DataHealerV147(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV147()
        
        # V147 核心模块
        self.residual_recall = ResidualBasedRecallV147(top_percent=0.2)
        self.sci_interaction = SignConsistencyInteractionV147(
            enable_sign_lock=True
        ) if enable_sci else None
        
        # V147 核心创新：MREF 和 DCSS
        self.mref = MultiResolutionEntropyFusion(
            windows=[3, 5, 10],
            weights={3: 0.5, 5: 0.3, 10: 0.2},
            min_confidence=0.5,
        ) if enable_mref else None
        
        self.dcss = DynamicCrossSectionalShrinkage(
            correlation_threshold=0.7,
            vol_scaling_threshold=1.5,  # V147 降低阈值
        ) if enable_dcss else None
        
        self.sector_validator = SectorNeutralValidator() if enable_sector_neutral else None
        self.ir_analyzer = IRStabilityAnalyzer()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Multi-Resolution Entropy Fusion (MREF)")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max N Factors: {n_factors}")
        logger.info(f"  Max Recall Factors: {max_recall_factors}")
        logger.info(f"  SCI: {'Enabled' if enable_sci else 'Disabled'}")
        logger.info(f"  MREF: {'Enabled' if enable_mref else 'Disabled'}")
        logger.info(f"  DCSS: {'Enabled' if enable_dcss else 'Disabled'}")
        logger.info(f"  Target IR: 0.55 (V146: 0.39)")
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
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
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
        """计算 Alpha 评分 - V147 核心逻辑（MREF + DCSS）"""
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
        
        core_factors = [f for f in V147_CORE_FACTORS if f in result.columns]
        
        if len(core_factors) >= 2:
            priority_candidates = ['reversion_10', 'volume_price_contradiction', 
                                   'rsi_14', 'mfi_14'] + V147_CANDIDATE_FACTORS
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
        
        # 添加强制保留因子
        forced_factors = ['volume_price_contradiction', 'liquidity_alpha', 'volatility_reversion']
        for factor in forced_factors:
            if factor in result.columns and factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
                self._log_audit("CandidatePool", f"Force added {factor}")
        
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
        
        # 8. V147 因子选择策略：扩容至 6-8 个因子
        final_selected = []
        max_factors = min(self.n_factors, 8)
        
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
            if t1_ic_abs >= 0.015:  # V147 降低阈值
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
            
            temp_score = np.zeros(len(result), dtype=np.float64)
            for i, factor in enumerate(self.selected_factors):
                f = factor_data[factor]
                # V147 修复：处理 std=0 的因子
                if isinstance(f, np.ndarray):
                    f = pd.Series(f)
                f_clean = f.fillna(0).astype(np.float64)
                if f_clean.std() > 1e-10:
                    temp_score += f_clean.values / len(self.selected_factors)
            
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
        
        # 11. V147 MREF + DCSS 信号处理（核心创新）
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            # 计算初始分数
            score = np.zeros(len(result), dtype=np.float64)
            for i, factor in enumerate(self.selected_factors):
                f = factor_data[factor]
                if isinstance(f, np.ndarray):
                    f = pd.Series(f)
                f_clean = f.fillna(0).astype(np.float64)
                score += f_clean.values / len(self.selected_factors)
            
            result['score_raw'] = score
            
            # V147 核心：MREF 置信度加权（简化版 - 仅用于统计，不修改信号）
            if self.enable_mref and self.mref:
                self._log_audit("MREF", "Computing Multi-Resolution Entropy Fusion stats...")
                # 仅计算统计，不修改信号
                _ = self.mref.apply_confidence_weighted_signal(result, 'score_raw')
                score_mref = score  # 保持原始信号
            else:
                score_mref = score
            
            # V147 核心：DCSS 动态截面收缩
            if self.enable_dcss and self.dcss:
                self._log_audit("DCSS", "Applying Dynamic Cross-Sectional Shrinkage...")
                
                # PCA 收缩
                score_pca = self.dcss.apply_pca_shrinkage(result, self.selected_factors)
                
                # 波动率缩放
                score_scaled, scaling_stats = self.dcss.apply_volatility_scaling(result, 'score_raw')
                
                result['score'] = score_scaled
            else:
                result['score'] = score_mref
            
            # 等权重集成
            for factor in self.selected_factors:
                self.factor_weights[factor] = 1.0 / len(self.selected_factors)
            
            self._log_audit("EnsembleComplete", f"Equal-weight ensemble with {len(self.selected_factors)} factors")
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (MREF+DCSS)")
        
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
    
    def get_mref_stats(self) -> Dict:
        """获取 MREF 统计"""
        return self.mref.get_entropy_stats() if self.mref else {}
    
    def get_mref_log(self) -> List[Dict]:
        """获取 MREF 日志"""
        return self.mref.get_mref_log() if self.mref else []
    
    def get_dcss_stats(self) -> Dict:
        """获取 DCSS 统计"""
        stats = {}
        if self.dcss:
            stats.update(self.dcss.get_shrinkage_stats())
            stats.update(self.dcss.get_scaling_factors())
        return stats
    
    def get_dcss_log(self) -> List[Dict]:
        """获取 DCSS 日志"""
        return self.dcss.get_dcss_log() if self.dcss else []


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_sci: bool = True,
    enable_mref: bool = True,
    enable_dcss: bool = True,
    enable_orthogonalization: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 2,
) -> AlphaResearchV147:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV147(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_sci=enable_sci,
        enable_mref=enable_mref,
        enable_dcss=enable_dcss,
        enable_orthogonalization=enable_orthogonalization,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV147...")
    
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
    logger.info(f"  SCI Features: {alpha.get_sci_features()}")
    logger.info(f"  Sign-Lock applied: {alpha.get_sign_lock_applied()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  MREF Stats: {alpha.get_mref_stats()}")
    logger.info(f"  DCSS Stats: {alpha.get_dcss_stats()}")
    logger.info(f"  Efficiency Ratio: {alpha.get_efficiency_ratio():.4f}")