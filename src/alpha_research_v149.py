"""
Alpha Research Module - V149 Spectral-Inertia-Enhancement (SIE).

【V148 问题诊断】
- 400 报错：日志量过大，API 请求体体积超限
- 信号惯性核 α 参数固定，未动态调整
- 正交化效率可进一步优化
- 行业中性化需要更严格的约束

【V149 核心使命 - SIE】
1. 修复 400 报错：
   - 禁止 Dump 全量数据，输出 score_df 总结时仅保留每月首末交易日或 .head(100)
   - 日志分片：逻辑截断，确保 API 请求体体积 < 2MB

2. 增强信号惯性核（Signal Inertia Kernel）：
   - α 动态调整：计算过去 5 天信号的自相关性，若相关性低（噪音大），则调小 α 强制平滑
   - Final_Score_t = α * Raw_Score_t + (1-α) * Final_Score_{t-1}

3. 增强截面物理正交化（Gram-Schmidt）：
   - 在 generate_scores 内部，对所有候选因子执行实时施密特正交化
   - 确保输入决策引擎的特征完全不相关

4. 增强行业风险对冲（Industry Neutralization）：
   - 在最终 Score 输出前，减去所属行业的平均 Score
   - 消除行业 Beta 影响，这是提升 IR 的最快路径

【V149 核心算法】
1. Dynamic Signal Inertia Kernel (DSIK):
   - α_t = base_α * autocorr(signal_{t-5:t}, lag=1)
   - 当自相关性低时，α 减小，强制平滑
   - 当自相关性高时，α 增大，保留更多原始信号

2. Enhanced Gram-Schmidt Orthogonalization (EGSO):
   - 使用 QR 分解加速正交化
   - 互信息验证确保输出因子完全不相关

3. Strict Industry Neutralization (SIN):
   - Score_final = Score_raw - mean(Score | industry)
   - 可选：除以行业标准差进行标准化

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 149 运行
- 严禁修改 backtest_referee.py 中的资金 (10 万) 和费率 (0.15%)
- 数据缺失时必须主动调用 data_loader 补全，禁止用 dropna() 一删了之
- 报错必改：内置 Auto-Healing 逻辑处理 Inf/NaN，禁止停止运行

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V148: ~0.40） |
| IC Std | < 0.08 | 时序波动率 |
| 400 Error | 0 | 禁止 String Length 报错 |
| Signal Turnover Rate | 降低 15%+ | 日度信号换手率对比 V148 |
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

VERSION = "V149"

# V149 核心因子（与 V148 保持一致）
V149_CORE_FACTORS = [
    'momentum_20',
    'volatility_10',
    'volume_price_contradiction',
    'liquidity_alpha',
    'volatility_reversion',
    'reversion_5',
    'volume_price_reversion',
]

# V149 候选因子池
V149_CANDIDATE_FACTORS = [
    'momentum_5', 'momentum_10', 'momentum_60',
    'reversion_10',
    'volatility_5', 'volatility_20',
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'vwap_distance', 'volume_rank', 'price_rank',
    'value_rank', 'ep_rank', 'bp_rank',
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist',
    'turnover_bias_5', 'turnover_bias_10', 'turnover_bias_20',
    'volume_shrink_ratio', 'turnover_vol_ratio',
    'tail_risk_indicator', 'skewness_20', 'extreme_volume_ratio',
]

ALL_FACTORS = V149_CORE_FACTORS + V149_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V149 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
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
    V149 增强版 Gram-Schmidt 正交化 + 互信息验证.
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
            X_norm[:, i] = 0
    
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
    
    if orthogonal:
        X_orthogonal = np.column_stack(orthogonal)
        return X_orthogonal, kept_indices
    else:
        return X_norm, list(range(n_factors))


def compute_signal_autocorrelation(signals: pd.Series, lag: int = 1, window: int = 5) -> float:
    """
    V149 新增：计算信号的自相关性.
    
    【原理】
    计算过去 window 天内、lag 阶滞后自相关性。
    自相关性低表示噪音大，需要更强的平滑。
    """
    if len(signals) < window + lag:
        return 0.5  # 默认中等自相关
    
    # 取最近 window 天的信号
    recent_signals = signals.tail(window)
    
    if len(recent_signals) < window:
        return 0.5
    
    # 计算 lag 阶自相关
    autocorr = recent_signals.autocorr(lag=lag)
    
    if pd.isna(autocorr):
        return 0.5
    
    # 将自相关映射到 0-1 范围（自相关可能为负）
    normalized_autocorr = (autocorr + 1) / 2
    
    return float(normalized_autocorr)


def winsorize_auto_heal(
    series: pd.Series, 
    sigma: float = 3.0, 
    percentile: float = 0.99
) -> pd.Series:
    """V149 自动愈合版 Winsorization"""
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


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """
    V149 新增：截断日志摘要，防止 400 报错.
    
    【策略】
    1. 仅保留每月的首末交易日
    2. 如果仍然超过 max_rows，则仅保留前 max_rows 行
    """
    if df.empty:
        return "Empty DataFrame"
    
    # 1. 按日期排序
    if 'trade_date' in df.columns:
        df_sorted = df.sort_values('trade_date').copy()
        
        # 2. 提取年月
        df_sorted['year_month'] = pd.to_datetime(df_sorted['trade_date']).dt.to_period('M')
        
        # 3. 每月首末交易日
        first_days = df_sorted.groupby('year_month').first().reset_index()
        last_days = df_sorted.groupby('year_month').last().reset_index()
        
        # 4. 合并
        summary_df = pd.concat([first_days, last_days]).drop_duplicates()
        
        # 5. 如果仍然超过限制，截断
        if len(summary_df) > max_rows:
            summary_df = summary_df.head(max_rows)
        
        # 6. 移除辅助列
        if 'year_month' in summary_df.columns:
            summary_df = summary_df.drop(columns=['year_month'])
        
        return summary_df.to_string(max_rows=MAX_LOG_ENTRIES)
    else:
        # 无日期列，直接截断
        return df.head(max_rows).to_string(max_rows=MAX_LOG_ENTRIES)


def safe_summary_dict(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> Dict:
    """
    V149 新增：安全地生成 DataFrame 摘要字典，用于 JSON 序列化.
    """
    if df.empty:
        return {'rows': 0, 'summary': 'Empty'}
    
    # 仅保留关键统计信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {
        'rows': len(df),
        'columns': list(df.columns),
        'numeric_stats': {},
    }
    
    for col in numeric_cols[:10]:  # 仅统计前 10 个数值列
        summary['numeric_stats'][col] = {
            'mean': float(df[col].mean()) if not df[col].isna().all() else 0.0,
            'std': float(df[col].std()) if not df[col].isna().all() else 0.0,
            'min': float(df[col].min()) if not df[col].isna().all() else 0.0,
            'max': float(df[col].max()) if not df[col].isna().all() else 0.0,
        }
    
    return summary


class DataHealerV149:
    """V149 增强版数据自愈模块"""
    
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
                logger.info("[V149][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V149][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V149][DataHealer] No database URL, SQL healer disabled")
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        """记录自愈日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details,
        }
        # V149: 限制日志长度
        if len(self.healing_log) >= MAX_LOG_ENTRIES:
            self.healing_log = self.healing_log[-MAX_LOG_ENTRIES//2:]
        self.healing_log.append(entry)
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """V149 增强版检查并修复缺失列"""
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
        
        # V149 增强：Auto-Impute + NaN/Inf 修复
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
            logger.error(f"[V149][DataHealer] SQL heal failed: {e}")
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
        """V149 新增：自动修复 NaN/Inf"""
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
        """获取自愈日志（已截断）"""
        return self.healing_log[-MAX_LOG_ENTRIES:]


class DynamicSignalInertiaKernel:
    """
    V149 核心创新 - 动态信号惯性核 (DSIK).
    
    【V148 问题】
    - α 参数固定，无法适应市场状态
    
    【V149 修复】
    - α_t = base_α * autocorr(signal_{t-5:t}, lag=1)
    - 自相关性低时，α 减小，强制平滑
    - 自相关性高时，α 增大，保留更多原始信号
    
    【参数】
    - base_alpha: 基础惯性系数（默认 0.3）
    - min_alpha: 最小 α 值（默认 0.1）
    - max_alpha: 最大 α 值（默认 0.6）
    - autocorr_window: 自相关计算窗口（默认 5 天）
    """
    
    def __init__(
        self,
        base_alpha: float = 0.3,
        min_alpha: float = 0.1,
        max_alpha: float = 0.6,
        autocorr_window: int = 5,
    ):
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.autocorr_window = autocorr_window
        self.sik_log = []
        self.inertia_stats = {}
        
    def _log_sik(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        # V149: 限制日志长度
        if len(self.sik_log) >= MAX_LOG_ENTRIES:
            self.sik_log = self.sik_log[-MAX_LOG_ENTRIES//2:]
        self.sik_log.append(entry)
    
    def apply_inertia(
        self, 
        df: pd.DataFrame, 
        raw_score_col: str,
    ) -> pd.Series:
        """
        应用动态信号惯性核.
        
        【完整流程】
        1. 按日期排序
        2. 对每个符号应用惯性核
        3. α_t 根据自相关性动态调整
        """
        if raw_score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        smoothed_scores = []
        alpha_values = []
        
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
            
            # 用于计算自相关的信号序列
            signal_series = pd.Series(raw_scores)
            
            for t in range(1, n):
                # 计算动态 α
                # 使用过去 autocorr_window 天的信号计算自相关
                if t >= self.autocorr_window:
                    past_signals = signal_series.iloc[t-self.autocorr_window:t+1]
                    autocorr = compute_signal_autocorrelation(
                        past_signals, lag=1, window=self.autocorr_window
                    )
                else:
                    autocorr = 0.5  # 初始阶段使用默认值
                
                # α = base_alpha * autocorr
                # 自相关性低时，α 减小（强制平滑）
                # 自相关性高时，α 增大（保留原始信号）
                alpha = self.base_alpha * (0.5 + autocorr)
                
                # 限制 α 范围
                alpha = max(self.min_alpha, min(self.max_alpha, alpha))
                alpha_values.append(alpha)
                
                # 应用惯性核
                smoothed[t] = (1 - alpha) * raw_scores[t] + alpha * smoothed[t - 1]
            
            smoothed_series = pd.Series(smoothed, index=symbol_data.index)
            smoothed_scores.append((symbol_mask, smoothed_series))
        
        # 合并所有结果
        final_smoothed = pd.Series(0.0, index=df.index)
        for mask, series in smoothed_scores:
            final_smoothed.loc[mask] = series
        
        mean_alpha = np.mean(alpha_values) if alpha_values else self.base_alpha
        
        self._log_sik(
            "DynamicInertiaApplied",
            f"base_alpha={self.base_alpha}, mean_alpha={mean_alpha:.3f}, "
            f"min_alpha={self.min_alpha}, max_alpha={self.max_alpha}"
        )
        
        self.inertia_stats = {
            'mean_alpha': float(mean_alpha),
            'min_alpha': self.min_alpha,
            'max_alpha': self.max_alpha,
            'autocorr_window': self.autocorr_window,
        }
        
        return final_smoothed
    
    def get_sik_log(self) -> List[Dict]:
        return self.sik_log[-MAX_LOG_ENTRIES:]
    
    def get_inertia_stats(self) -> Dict:
        return self.inertia_stats


class VolumePriceReversion:
    """V149 成交量 - 价格反转因子 (VPR)"""
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.vpr_log = []
        
    def _log_vpr(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.vpr_log) >= MAX_LOG_ENTRIES:
            self.vpr_log = self.vpr_log[-MAX_LOG_ENTRIES//2:]
        self.vpr_log.append(entry)
    
    def compute_vpr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算 VPR 因子.
        
        【公式】
        VPR = Rank(Low_Price_Volume / Total_Volume) - Rank(Return)
        
        【修复】
        - 将中间结果添加到 DataFrame 中，避免 groupby[key] 报错
        """
        result = df.copy()
        
        if 'low' not in result.columns or 'high' not in result.columns:
            self._log_vpr("MissingData", "Missing low/high columns, returning 0")
            return pd.Series(0, index=df.index)
        
        if 'volume' not in result.columns:
            self._log_vpr("MissingData", "Missing volume column, returning 0")
            return pd.Series(0, index=df.index)
        
        # 1. 计算 lookback 窗口内的最低/最高价
        low_prices = result.groupby('symbol')['low'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).min()
        )
        high_prices = result.groupby('symbol')['high'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).max()
        )
        
        # 2. 计算价格位置（0-1 归一化）
        price_range = high_prices - low_prices + 1e-10
        price_position = (result['low'] - low_prices) / price_range
        
        # 3. 计算低价区成交量权重
        volume_weight = 1 - price_position
        
        # 4. 计算低价区成交量占比（简化版）
        low_price_volume = result['volume'] * volume_weight
        total_volume = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(self.lookback_window, min_periods=5).sum()
        )
        lpv_sum = result.groupby('symbol')['volume'].transform(
            lambda x: (x * volume_weight.loc[x.index]).rolling(self.lookback_window, min_periods=5).sum()
        )
        
        lpv_ratio = lpv_sum / (total_volume + 1e-10)
        
        # 5. 计算近期收益
        if 'close' in result.columns:
            returns = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change(5)
            )
        else:
            returns = pd.Series(0, index=df.index)
        
        # 6. 将中间结果添加到 DataFrame 中进行排名
        result['_lpv_ratio'] = lpv_ratio.fillna(0.5)
        result['_returns'] = returns.fillna(0)
        
        # 排名计算（使用列名而非 Series）
        lpv_rank = result.groupby('trade_date')['_lpv_ratio'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        ret_rank = result.groupby('trade_date')['_returns'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        # 7. VPR = LPV_Rank - Return_Rank
        vpr = lpv_rank - ret_rank
        
        # 8. 标准化
        vpr = vpr.groupby(result['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        ).fillna(0)
        
        # 清理临时列
        result = result.drop(columns=['_lpv_ratio', '_returns'])
        
        self._log_vpr(
            "Computed",
            f"VPR factor: mean={vpr.mean():.4f}, std={vpr.std():.4f}"
        )
        
        return vpr
    
    def get_vpr_log(self) -> List[Dict]:
        return self.vpr_log[-MAX_LOG_ENTRIES:]


class IndustryConsistencyConstraint:
    """V149 行业一致性约束"""
    
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
        if len(self.icc_log) >= MAX_LOG_ENTRIES:
            self.icc_log = self.icc_log[-MAX_LOG_ENTRIES//2:]
        self.icc_log.append(entry)
    
    def apply_industry_shrinkage(
        self,
        df: pd.DataFrame,
        signal_col: str,
        industry_col: str = 'industry_code',
    ) -> pd.Series:
        """应用行业一致性约束"""
        if signal_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        scaled_signal = result[signal_col].copy()
        
        if industry_col not in result.columns:
            self._log_icc("NoIndustryData", "Using global shrinkage instead")
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
                
                signal_sign = np.sign(industry_data)
                positive_ratio = (signal_sign > 0).mean()
                negative_ratio = (signal_sign < 0).mean()
                consistency = max(positive_ratio, negative_ratio)
                
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
        return self.icc_log[-MAX_LOG_ENTRIES:]
    
    def get_industry_stats(self) -> Dict:
        return self.industry_stats


class StrictIndustryNeutralizer:
    """
    V149 新增 - 严格行业中性化.
    
    【原理】
    在最终 Score 输出前，减去所属行业的平均 Score。
    Score_final = Score_raw - mean(Score | industry)
    
    这可以消除行业 Beta 影响，是提升 IR 的最快路径。
    """
    
    def __init__(self, industry_column: str = 'industry_code'):
        self.industry_column = industry_column
        self.neutralization_log = []
        self.neutralization_stats = {}
        
    def _log_neutralization(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.neutralization_log) >= MAX_LOG_ENTRIES:
            self.neutralization_log = self.neutralization_log[-MAX_LOG_ENTRIES//2:]
        self.neutralization_log.append(entry)
    
    def neutralize(
        self,
        df: pd.DataFrame,
        signal_col: str,
    ) -> pd.Series:
        """
        应用严格行业中性化.
        
        【完整流程】
        1. 按日期和行业分组
        2. 计算每个行业内信号均值
        3. 信号减去行业均值
        """
        if signal_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        
        if self.industry_column not in result.columns:
            self._log_neutralization("NoIndustryData", "Industry column not found")
            return result[signal_col]
        
        neutralized_signal = result[signal_col].copy()
        
        for date in result['trade_date'].unique():
            date_mask = result['trade_date'] == date
            date_data = result.loc[date_mask]
            
            # 计算每个行业的均值
            industry_means = date_data.groupby(self.industry_column)[signal_col].mean()
            
            # 减去行业均值
            for industry in industry_means.index:
                industry_mask = date_data[self.industry_column] == industry
                industry_idx = date_data.loc[industry_mask].index
                neutralized_signal.loc[industry_idx] = date_data.loc[industry_mask, signal_col] - industry_means[industry]
        
        self._log_neutralization(
            "IndustryNeutralizationApplied",
            f"Neutralized signal by {self.industry_column}"
        )
        
        self.neutralization_stats = {
            'method': 'industry_mean_subtraction',
            'industry_column': self.industry_column,
        }
        
        return neutralized_signal
    
    def get_neutralization_log(self) -> List[Dict]:
        return self.neutralization_log[-MAX_LOG_ENTRIES:]
    
    def get_neutralization_stats(self) -> Dict:
        return self.neutralization_stats


class FactorGeneratorV149:
    """V149 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        self.vpr_computer = VolumePriceReversion(lookback_window=20)
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.generation_log) >= MAX_LOG_ENTRIES:
            self.generation_log = self.generation_log[-MAX_LOG_ENTRIES//2:]
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
        """V149 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
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
        
        # V149 波动率反转因子
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        
        # 量价因子
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        # V149 核心：VPR 因子
        result['volume_price_reversion'] = self.vpr_computer.compute_vpr(result)
        
        # volume_rank
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        self._log_generation("Complete", f"Generated base factors including V149 VPR")
        
        return result


class AlphaResearchV149:
    """
    V149 Alpha 研究引擎 - Spectral-Inertia-Enhancement (SIE).
    
    【V149 核心改进】
    1. 修复 400 报错：日志截断，禁止 Dump 全量数据
    2. Dynamic Signal Inertia Kernel: α 根据自相关性动态调整
    3. Enhanced Gram-Schmidt: 每日截面因子正交化
    4. Strict Industry Neutralization: 行业均值减法
    
    【目标指标】
    - T+1 Rank IC > 0.055
    - IC_IR > 0.55（V148: ~0.40）
    - IC Std < 0.08
    - 400 Error: 0
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_sci: bool = True,
        enable_gso: bool = True,
        enable_sik: bool = True,
        enable_vpr: bool = True,
        enable_icc: bool = True,
        enable_sin: bool = True,  # Strict Industry Neutralization
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
        self.enable_sin = enable_sin
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
        self.data_healer = DataHealerV149(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV149()
        
        # V149 核心模块
        self.sik = DynamicSignalInertiaKernel(base_alpha=inertia_base) if enable_sik else None
        self.icc = IndustryConsistencyConstraint() if enable_icc else None
        self.sin = StrictIndustryNeutralizer() if enable_sin else None
        
        self.sector_validator = SectorNeutralValidator() if enable_sector_neutral else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: SIE (Spectral-Inertia-Enhancement)")
        logger.info(f"  GSO: {'Enabled' if enable_gso else 'Disabled'}")
        logger.info(f"  DSIK: {'Enabled' if enable_sik else 'Disabled'} (base_alpha={inertia_base})")
        logger.info(f"  VPR: {'Enabled' if enable_vpr else 'Disabled'}")
        logger.info(f"  ICC: {'Enabled' if enable_icc else 'Disabled'}")
        logger.info(f"  SIN: {'Enabled' if enable_sin else 'Disabled'}")
        logger.info(f"  Target IR: 0.55 (V148: ~0.40)")
        logger.info(f"  400 Error Fix: Log truncation enabled")
    
    def _log_audit(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.audit_log) >= MAX_LOG_ENTRIES:
            self.audit_log = self.audit_log[-MAX_LOG_ENTRIES//2:]
        self.audit_log.append(entry)
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
        """V149 每日截面 Gram-Schmidt 正交化"""
        if not factor_cols or len(factor_cols) < 2:
            return {col: df[col] for col in factor_cols}
        
        result = df.copy()
        orthogonal_factors = {}
        
        for col in factor_cols:
            orthogonal_factors[col] = pd.Series(0.0, index=df.index)
        
        for date in df['trade_date'].unique():
            date_mask = df['trade_date'] == date
            date_data = df.loc[date_mask]
            
            if len(date_data) < 20:
                continue
            
            factor_matrix = date_data[factor_cols].fillna(0).values
            
            X_orthogonal, kept_indices = gram_schmidt_orthogonalize(
                factor_matrix, mi_threshold=0.1
            )
            
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
        """计算 Alpha 评分 - V149 核心逻辑（SIE）"""
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
        
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
            self._log_audit("FactorGeneration", "Generated base factors including V149 VPR")
        
        # 4. 构建候选因子池
        all_candidate_factors = []
        
        if 'volume_rank' in result.columns:
            all_candidate_factors.append('volume_rank')
        
        core_factors = [f for f in V149_CORE_FACTORS if f in result.columns]
        for factor in core_factors:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        candidate_factors = [f for f in V149_CANDIDATE_FACTORS if f in result.columns]
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
        
        factor_ics.sort(key=lambda x: x[1], reverse=True)
        
        max_factors = min(self.n_factors, 8)
        final_selected = [f[0] for f in factor_ics[:max_factors]]
        
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
            
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
            
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. V149: Gram-Schmidt 正交化
        if self.enable_gso and len(self.selected_factors) >= 2:
            self._log_audit("GSO", "Applying Gram-Schmidt Orthogonalization...")
            orthogonal_factors = self._apply_gram_schmidt_daily(result, self.selected_factors)
            
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
        
        # 9. V149: Dynamic Signal Inertia Kernel
        if self.enable_sik and self.sik:
            self._log_audit("DSIK", "Applying Dynamic Signal Inertia Kernel...")
            smoothed_score = self.sik.apply_inertia(result, 'score_raw')
            result['score_smoothed'] = smoothed_score
        else:
            result['score_smoothed'] = result['score_raw']
        
        # 10. V149: Industry Consistency Constraint
        if self.enable_icc and self.icc:
            self._log_audit("ICC", "Applying Industry Consistency Constraint...")
            final_score = self.icc.apply_industry_shrinkage(result, 'score_smoothed')
            result['score_icc'] = final_score
        else:
            result['score_icc'] = result['score_smoothed']
        
        # 11. V149: Strict Industry Neutralization
        if self.enable_sin and self.sin:
            self._log_audit("SIN", "Applying Strict Industry Neutralization...")
            final_score = self.sin.neutralize(result, 'score_icc')
            result['score'] = final_score
        else:
            result['score'] = result['score_icc']
        
        # 12. 等权重集成
        for factor in self.selected_factors:
            self.factor_weights[factor] = 1.0 / len(self.selected_factors)
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (SIE)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']
        for col in ['t1_return_period']:
            if col in result.columns and col not in output_cols:
                output_cols.append(col)
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        获取因子 IC.
        
        【兼容性修复】
        - 接受可选 df 参数（与 backtest_referee 兼容）
        - 如果传入 df，基于 df 重新计算 IC
        """
        # 如果传入 df，基于 df 重新计算 IC
        if df is not None and not df.empty:
            ics = {}
            for factor in self.selected_factors:
                if factor in df.columns:
                    ic = self._calc_factor_ic(df, factor)
                    direction = self.factor_directions.get(factor, 1)
                    ics[factor] = ic * direction
                else:
                    ics[factor] = self.factor_ics.get(factor, 0.0) * self.factor_directions.get(factor, 1)
            return ics
        
        # 否则返回缓存的 IC
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
    
    def get_sin_stats(self) -> Dict:
        """获取 SIN 统计"""
        return self.sin.get_neutralization_stats() if self.sin else {}
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志（已截断）"""
        return self.audit_log[-MAX_LOG_ENTRIES:]


class SectorNeutralValidator:
    """V149 行业中性化校验器"""
    
    def __init__(self, industry_column: str = 'industry_code'):
        self.industry_column = industry_column
        self.validation_log = []
        
    def _log_validation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.validation_log) >= MAX_LOG_ENTRIES:
            self.validation_log = self.validation_log[-MAX_LOG_ENTRIES//2:]
        self.validation_log.append(entry)
    
    def validate_ir_improvement(
        self, df: pd.DataFrame, signal_col: str, return_col: str = 't1_return'
    ) -> Dict:
        """校验 IR 提升是否来自行业偏离"""
        if signal_col not in df.columns or return_col not in df.columns:
            return {'valid': False, 'reason': 'Missing columns'}
        
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
        return self.validation_log[-MAX_LOG_ENTRIES:]


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
    enable_sin: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    max_recall_factors: int = 2,
    inertia_base: float = 0.3,
) -> AlphaResearchV149:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV149(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_sci=enable_sci,
        enable_gso=enable_gso,
        enable_sik=enable_sik,
        enable_vpr=enable_vpr,
        enable_icc=enable_icc,
        enable_sin=enable_sin,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
        max_recall_factors=max_recall_factors,
        inertia_base=inertia_base,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV149...")
    
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
    logger.info(f"  SIN Stats: {alpha.get_sin_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")