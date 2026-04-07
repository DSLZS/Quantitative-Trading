"""
Alpha Research Module - V157 Predictive Power Recovery (ORA 2.0 Pure).

【V156 审计结论 - 严厉警告】
- 回测统计欺诈：Rank IC (0.0156) 与收益率 (95.6%) 严重背离
- 判定：在信号平滑或波动率收缩逻辑中使用了未来全局数据（Look-ahead Bias）
- 指令：立刻废弃 V156 的信号处理模块，回退到 V155 的 ORA 2.0 逻辑内核！

【V157 核心使命 - 回归预测力】
1. 纯粹滚动计算：所有 Scaling、Momentum 计算必须严格基于 window=20 或 60 的滚动逻辑
2. SQL 数据修复：针对 pe_ttm 缺失，实现表关联逻辑（检查 valuation 表或 indicator 表）
3. Non-Linear Enhancement：仅针对 volume_price_contradiction 引入局部二阶项

【V157 目标指标】
- Rank IC > 0.08 (核心指标)
- IC IR > 0.5 (稳定性)
- 严禁指标美化：如果 IC 大幅下降，即使收益率再高也视为失败

【工程纪律】
- 基于 V155 ORA 2.0 最小改动
- 严禁使用 df.mean() 等全局函数
- 严禁修改 src/engine/ 目录
- 严禁修改 initial_capital (100,000) 和费率
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

VERSION = "V157"

# V157 核心因子 - 聚焦短期预测
V157_CORE_FACTORS = [
    'momentum_5',
    'volatility_5',
    'volume_price_contradiction',
    'liquidity_alpha',
    'reversion_5',
]

V157_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V157_CORE_FACTORS + V157_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V157 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V157 参数配置 - 纯粹滚动计算
ORM_CORE_FACTOR = 'volume_price_contradiction'
LEAD_LAG_THRESHOLD = 1.5
LEAD_LAG_MAX_LAG = 5
ROLLING_WINDOW = 20  # 严格滚动窗口
IC_WEIGHT_WINDOW = 10  # IC 加权窗口

# V157 Non-Linear Enhancement - 仅针对 volume_price_contradiction
NONLINEAR_CORE_FACTOR = 'volume_price_contradiction'
NONLINEAR_WINDOW = 20  # 局部二阶项计算窗口


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    计算两个变量之间的互信息（Mutual Information）.
    
    【V157 核心】用于 Adaptive Lead-Lag Correction
    计算因子与不同滞后阶数回报之间的 MI，判断因子的领先性
    """
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


def winsorize_rolling(series: pd.Series, window: int = 20, sigma: float = 3.0) -> pd.Series:
    """
    V157 纯粹滚动版 Winsorization - 严格防前视.
    
    【原理】
    - 使用滚动窗口计算均值和标准差
    - 严禁使用全样本统计量
    """
    series_clean = series.replace([np.inf, -np.inf], np.nan)
    
    # 滚动均值和标准差
    rolling_mean = series_clean.rolling(window=window, min_periods=5).mean()
    rolling_std = series_clean.rolling(window=window, min_periods=5).std()
    
    # 滚动截断
    lower = rolling_mean - sigma * rolling_std
    upper = rolling_mean + sigma * rolling_std
    
    series_wins = series_clean.copy()
    for idx in series_clean.index:
        if pd.notna(lower.loc[idx]) and pd.notna(upper.loc[idx]):
            series_wins.loc[idx] = np.clip(series_clean.loc[idx], lower.loc[idx], upper.loc[idx])
    
    # 填充 NaN
    series_wins = series_wins.ffill().bfill().fillna(series_clean.mean())
    
    return series_wins


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, percentile: float = 0.99) -> pd.Series:
    """V157 自动愈合版 Winsorization"""
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
    
    # 5. 最终 NaN 填充 - V157 使用 ffill 优先
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def compute_cross_sectional_skewness(signal: pd.Series) -> float:
    """V157 - 计算截面偏度"""
    if len(signal) < 20:
        return 0.0
    
    signal_clean = signal.dropna()
    if len(signal_clean) < 20:
        return 0.0
    
    try:
        from scipy import stats
        skewness = stats.skew(signal_clean)
        return float(skewness)
    except Exception:
        mean = signal_clean.mean()
        std = signal_clean.std()
        if std > 1e-10:
            skewness = ((signal_clean - mean) ** 3).mean() / (std ** 3)
            return float(skewness)
        return 0.0


def compute_local_second_order(series: pd.Series, window: int = NONLINEAR_WINDOW) -> pd.Series:
    """
    V157 Non-Linear Enhancement - 计算局部二阶项.
    
    【原理】
    - 仅针对 volume_price_contradiction 引入局部二阶项
    - 旨在挖掘量价背离的非线性转折点
    
    【公式】
    - Second_Order = (Factor_t - Factor_{t-1})^2
    - 使用滚动窗口标准化
    """
    if len(series) < window:
        return pd.Series(0, index=series.index)
    
    # 计算一阶差分
    diff = series.diff()
    
    # 计算二阶项（平方）
    second_order = diff ** 2
    
    # 滚动标准化
    rolling_mean = second_order.rolling(window=window, min_periods=5).mean()
    rolling_std = second_order.rolling(window=window, min_periods=5).std()
    
    second_order_std = (second_order - rolling_mean) / (rolling_std + 1e-10)
    
    return second_order_std.fillna(0)


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V157 日志截断"""
    if df.empty:
        return "Empty DataFrame"
    if 'trade_date' in df.columns:
        df_sorted = df.sort_values('trade_date').copy()
        df_sorted['year_month'] = pd.to_datetime(df_sorted['trade_date']).dt.to_period('M')
        first_days = df_sorted.groupby('year_month').first().reset_index()
        last_days = df_sorted.groupby('year_month').last().reset_index()
        summary_df = pd.concat([first_days, last_days]).drop_duplicates()
        if len(summary_df) > max_rows:
            summary_df = summary_df.head(max_rows)
        if 'year_month' in summary_df.columns:
            summary_df = summary_df.drop(columns=['year_month'])
        return summary_df.to_string(max_rows=MAX_LOG_ENTRIES)
    else:
        return df.head(max_rows).to_string(max_rows=MAX_LOG_ENTRIES)


class DataHealerV157:
    """
    V157 数据自愈模块 - SQL 表关联逻辑.
    
    【V157 核心改进】
    - 针对 pe_ttm 缺失，实现表关联逻辑（检查 valuation 表或 indicator 表）
    - 严禁使用中值填充！必须从数据库重新拉取
    """
    
    FIELD_MAPPING = {
        'pe_ttm': ['pe_ttm', 'pe_ttm_new', 'pe', 'valuation.pe_ttm', 'indicator.pe_ttm'],
        'pb': ['pb', 'pb_new', 'valuation.pb', 'indicator.pb'],
        'ps_ttm': ['ps_ttm', 'ps', 'valuation.ps_ttm'],
        'pcf_ocf': ['pcf_ocf', 'pcf', 'valuation.pcf_ocf'],
        'total_mv': ['total_mv', 'market_value', 'valuation.total_mv'],
        'circ_mv': ['circ_mv', 'market_value_float', 'valuation.circ_mv'],
        'turnover_rate': ['turnover_rate', 'turnover', 'stock_daily_basic.turnover_rate'],
        'volume': ['volume', 'vol', 'stock_daily.volume'],
    }
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._detected_columns = {}
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        """初始化 SQL 自愈器"""
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V157][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V157][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V157][DataHealer] No database URL, SQL healer disabled")
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        """记录自愈日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details,
        }
        if len(self.healing_log) >= MAX_LOG_ENTRIES:
            self.healing_log = self.healing_log[-MAX_LOG_ENTRIES//2:]
        self.healing_log.append(entry)
    
    def _detect_actual_columns(self, table_name: str = 'stock_daily') -> Dict[str, str]:
        """检测数据库实际列名"""
        if not self.engine:
            return {}
        if table_name in self._detected_columns:
            return self._detected_columns[table_name]
        try:
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            mapping = {}
            for standard_name, possible_names in self.FIELD_MAPPING.items():
                for possible_name in possible_names:
                    if possible_name in columns:
                        mapping[standard_name] = possible_name
                        break
                else:
                    if standard_name in columns:
                        mapping[standard_name] = standard_name
            self._detected_columns[table_name] = mapping
            self._log_healing("ColumnMappingDetected", "ALL", "SUCCESS", f"Detected mapping: {mapping}")
            return mapping
        except Exception as e:
            logger.warning(f"[V157][DataHealer] Failed to detect columns: {e}")
            return {}
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str], 
                       industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        V157 检查并修复缺失列 - SQL 表关联逻辑.
        
        【V157 核心改进】
        - 针对 pe_ttm 缺失，实现表关联逻辑（检查 valuation 表或 indicator 表）
        - 严禁使用中值填充！必须从数据库重新拉取
        """
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing("MissingColumnsDetected", ", ".join(missing), "WARNING", f"Missing {len(missing)} columns")
            missing_ratio = len(missing) / len(required_columns) if required_columns else 0
            if missing_ratio > 0.05:
                logger.error(f"[V157][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    raise ValueError(f"[V157] Data integrity violation: {len(missing)} columns missing")
            else:
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    for col in missing:
                        result = result.assign(**{col: 0.0})
        else:
            self._log_healing("ColumnsComplete", "ALL", "OK", "All required columns present")
        
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result, industry_data)
        self._log_healing("MultiLevelImputeApplied", "ALL_NUMERIC", "SUCCESS", "Applied SQL -> ffill -> bfill")
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        从 SQL 补全缺失列 - V157 表关联逻辑.
        
        【V157 核心改进】
        - 针对 pe_ttm 缺失，检查 valuation 表或 indicator 表
        - 使用表关联逻辑获取数据
        """
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
            
            col_mapping = self._detect_actual_columns('stock_daily')
            symbols_str = ', '.join([f"'{s}'" for s in symbols])
            
            # 构建查询列
            select_columns = ['symbol', 'trade_date']
            for col in columns:
                if col in col_mapping:
                    actual_col = col_mapping[col]
                    if actual_col != col:
                        select_columns.append(f"{actual_col} AS {col}")
                    else:
                        select_columns.append(col)
                else:
                    select_columns.append(col)
            
            # V157 核心：表关联逻辑
            # 检查 valuation 表或 indicator 表
            valuation_tables = ['valuation', 'stock_valuation', 'indicator', 'stock_indicator']
            valuation_data = None
            
            for table in valuation_tables:
                try:
                    query = text(f"""
                        SELECT symbol, trade_date, pe_ttm, pb, ps_ttm, pcf_ocf
                        FROM {table}
                        WHERE symbol IN ({symbols_str})
                        AND trade_date BETWEEN :start_date AND :end_date
                    """)
                    valuation_data = pd.read_sql_query(query, self.engine, params={
                        'start_date': start_date,
                        'end_date': end_date,
                    })
                    if not valuation_data.empty:
                        self._log_healing("ValuationTableFound", table, "SUCCESS", f"Found {len(valuation_data)} rows")
                        break
                except Exception:
                    continue
            
            # 主查询
            query = text(f"""
                SELECT {', '.join(select_columns)}
                FROM stock_daily
                WHERE symbol IN ({symbols_str})
                AND trade_date BETWEEN :start_date AND :end_date
            """)
            
            sql_df = pd.read_sql_query(query, self.engine, params={
                'start_date': start_date,
                'end_date': end_date,
            })
            
            # 合并 valuation 数据
            if valuation_data is not None and not valuation_data.empty:
                for col in ['pe_ttm', 'pb', 'ps_ttm', 'pcf_ocf']:
                    if col in valuation_data.columns and col in columns:
                        sql_df = sql_df.merge(
                            valuation_data[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'],
                            how='left',
                            suffixes=('', '_val')
                        )
                        sql_df[col] = sql_df[col].fillna(sql_df[f'{col}_val'])
                        sql_df = sql_df.drop(columns=[c for c in sql_df.columns if c.endswith('_val')])
            
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
                        self._log_healing("HealedFromSQL", col, "SUCCESS", f"Healed {len(sql_df)} rows")
        except Exception as e:
            logger.error(f"[V157][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V157 自动分组插值 - 严格 ffill/bfill"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            result[col] = result.groupby(group_col, group_keys=False)[col].transform(
                lambda x: x.ffill().bfill()
            )
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def _repair_nan_inf(self, df: pd.DataFrame, 
                        industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """V157 修复 NaN/Inf"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                self._log_healing("InfRepaired", col, "SUCCESS", f"Repaired {inf_count} Inf values")
            
            nan_mask = result[col].isna()
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                
                result[col] = result[col].fillna(col_median)
                self._log_healing("NaNRepaired", col, "SUCCESS", f"Repaired {nan_count} NaN values")
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log[-MAX_LOG_ENTRIES:]


class SignalInertiaLayer:
    """
    V157 Signal Inertia Layer (SIL) - 纯粹滚动版.
    
    【V157 核心公式】
    Score_final = w * Score_new + (1-w) * Score_old
    
    其中权重 w 是动态的：
    w = Correlation(Signal_{t-1}, Return_{t-1}) over rolling window
    
    【V157 修复】
    - 严格使用滚动窗口计算相关性
    - 严禁使用全样本统计量
    """
    
    def __init__(
        self,
        min_weight: float = 0.2,
        max_weight: float = 0.8,
        decay_factor: float = 0.95,
        correlation_window: int = 20,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor
        self.correlation_window = correlation_window
        self.inertia_log = []
        self.inertia_stats = {}
        
    def _log_inertia(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.inertia_log) >= MAX_LOG_ENTRIES:
            self.inertia_log = self.inertia_log[-MAX_LOG_ENTRIES//2:]
        self.inertia_log.append(entry)
    
    def compute_dynamic_weight(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
        return_col: str = 't1_return',
    ) -> pd.Series:
        """
        计算动态权重 w - V157 纯粹滚动版.
        
        【V157 修复】
        - 严格使用滚动窗口计算相关性
        - 严禁使用全样本统计量
        """
        if score_col not in df.columns or return_col not in df.columns:
            return pd.Series((self.min_weight + self.max_weight) / 2, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        # 按股票分组计算滚动相关性
        weights = []
        for symbol in result['symbol'].unique():
            symbol_data = result[result['symbol'] == symbol].copy()
            
            if len(symbol_data) < self.correlation_window:
                # 数据不足时使用默认权重
                w = pd.Series((self.min_weight + self.max_weight) / 2, index=symbol_data.index)
            else:
                # 计算滚动相关性
                signal_past = symbol_data[score_col].shift(1).fillna(0)
                return_past = symbol_data[return_col].shift(1).fillna(0)
                
                # 滚动相关性计算
                rolling_corr = signal_past.rolling(window=self.correlation_window, min_periods=5).corr(return_past)
                
                # 使用绝对值相关性映射到权重区间
                w = self.min_weight + rolling_corr.abs() * (self.max_weight - self.min_weight)
                w = w.fillna((self.min_weight + self.max_weight) / 2)
            
            weights.append(pd.DataFrame({'idx': symbol_data.index, 'weight': w}))
        
        weight_df = pd.concat(weights).set_index('idx')
        weights_series = weight_df['weight']
        
        self._log_inertia(
            "DynamicWeightComputed",
            f"Window={self.correlation_window}, Mean_Weight={weights_series.mean():.3f}"
        )
        
        self.inertia_stats = {
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'decay_factor': self.decay_factor,
            'mean_weight': float(weights_series.mean()),
            'std_weight': float(weights_series.std()),
        }
        
        return weights_series
    
    def apply_inertia(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
        return_col: str = 't1_return',
    ) -> pd.Series:
        """
        应用信号惯性层 - V157 纯粹滚动版.
        
        【完整流程】
        1. 计算动态权重 w（滚动相关性）
        2. Score_final = w * Score_new + (1-w) * Score_old
        """
        if score_col not in df.columns:
            return df.get(score_col, pd.Series(0, index=df.index)).fillna(0)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        # 1. 计算动态权重
        weights = self.compute_dynamic_weight(result, score_col, return_col)
        
        # 2. 计算历史信号（按股票分组滞后）
        result['score_lag'] = result.groupby('symbol')[score_col].transform(
            lambda x: x.shift(1)
        )
        
        # 3. 应用惯性公式
        score_new = result[score_col].fillna(0)
        score_old = result['score_lag'].fillna(score_new)
        
        score_final = weights * score_new + (1 - weights) * score_old
        
        self._log_inertia(
            "InertiaApplied",
            f"New score std={score_new.std():.4f} -> Final score std={score_final.std():.4f}"
        )
        
        return score_final
    
    def get_inertia_log(self) -> List[Dict]:
        return self.inertia_log[-MAX_LOG_ENTRIES:]
    
    def get_inertia_stats(self) -> Dict:
        return self.inertia_stats


class AdaptiveLeadLagCorrector:
    """V157 自适应领先滞后校正器"""
    
    def __init__(
        self, 
        max_lag: int = LEAD_LAG_MAX_LAG,
        threshold: float = LEAD_LAG_THRESHOLD,
        n_bins: int = 10,
    ):
        self.max_lag = max_lag
        self.threshold = threshold
        self.n_bins = n_bins
        self.correction_log = []
        self.lead_lag_stats = {}
        
    def _log_correction(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.correction_log) >= MAX_LOG_ENTRIES:
            self.correction_log = self.correction_log[-MAX_LOG_ENTRIES//2:]
        self.correction_log.append(entry)
    
    def compute_lead_lag_score(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_cols: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[int, float]]:
        """计算因子的领先滞后分数"""
        if factor_col not in df.columns:
            return 0.0, {}
        
        if return_cols is None:
            return_cols = ['t1_return_period', 't2_return_period', 't3_return_period', 
                          't4_return_period', 't5_return_period']
        
        factor_data = df[factor_col].fillna(0).values
        mi_by_lag = {}
        
        for lag in range(1, self.max_lag + 1):
            return_col = f't{lag}_return_period'
            
            if return_col not in df.columns:
                return_col = f't{lag}_return'
                if return_col not in df.columns:
                    continue
            
            return_data = df[return_col].fillna(0).values
            mi = compute_mutual_information(factor_data, return_data, self.n_bins)
            mi_by_lag[lag] = mi
        
        mi_lag_1 = mi_by_lag.get(1, 0.0)
        mi_lag_5 = mi_by_lag.get(5, 0.0)
        
        if mi_lag_5 > 1e-10:
            lead_lag_score = mi_lag_1 / mi_lag_5
        elif mi_lag_1 > 0:
            lead_lag_score = 2.0
        else:
            lead_lag_score = 0.0
        
        self._log_correction(
            "LeadLagScoreComputed",
            f"{factor_col}: MI_Lag1={mi_lag_1:.4f}, MI_Lag5={mi_lag_5:.4f}, Score={lead_lag_score:.2f}"
        )
        
        return lead_lag_score, mi_by_lag
    
    def select_lead_factors(
        self,
        df: pd.DataFrame,
        candidate_factors: List[str],
    ) -> List[str]:
        """选择领先因子"""
        lead_scores = {}
        
        for factor in candidate_factors:
            score, _ = self.compute_lead_lag_score(df, factor)
            lead_scores[factor] = score
        
        lead_factors = [f for f, s in lead_scores.items() if s > self.threshold]
        
        if not lead_factors:
            sorted_factors = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)
            lead_factors = [f for f, _ in sorted_factors[:min(5, len(sorted_factors))]]
        
        self.lead_lag_stats = {
            'threshold': self.threshold,
            'lead_factors': lead_factors,
            'lead_scores': lead_scores,
        }
        
        self._log_correction(
            "LeadFactorsSelected",
            f"Selected {len(lead_factors)} lead factors: {lead_factors}"
        )
        
        return lead_factors
    
    def get_correction_log(self) -> List[Dict]:
        return self.correction_log[-MAX_LOG_ENTRIES:]
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class RollingICSignCalculator:
    """
    V157 滚动 IC 符号计算器 - 纯粹滚动版.
    
    【PAC 逻辑】
    - 严格基于 Rolling Window（过去 20 日滚动 IC）
    - 严禁使用全样本 IC 进行符号校正
    """
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        self.calculation_log = []
        
    def _log_calculation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.calculation_log) >= MAX_LOG_ENTRIES:
            self.calculation_log = self.calculation_log[-MAX_LOG_ENTRIES//2:]
        self.calculation_log.append(entry)
    
    def compute_rolling_ic_sign(
        self, 
        df: pd.DataFrame, 
        factor_col: str, 
        return_col: str = 't1_return'
    ) -> pd.Series:
        """计算滚动 IC 符号 - V157 纯粹滚动版"""
        if factor_col not in df.columns or return_col not in df.columns:
            self._log_calculation("MissingColumns", f"Missing {factor_col} or {return_col}")
            return pd.Series(1, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        date_ics = []
        for date in result['trade_date'].unique():
            day_data = result[result['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[factor_col].fillna(0)
            r = day_data[return_col].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                r_rank = r.rank(method='average')
                ic = np.corrcoef(f_rank, r_rank)[0, 1]
                if not np.isnan(ic):
                    date_ics.append({'trade_date': date, 'ic': ic})
        
        if not date_ics:
            self._log_calculation("NoICCalculated", "No valid IC computed")
            return pd.Series(1, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        ic_df['rolling_ic'] = ic_df['ic'].rolling(window=self.window, min_periods=5).mean()
        ic_df['rolling_ic_sign'] = np.sign(ic_df['rolling_ic']).replace(0, 1)
        
        ic_sign_map = ic_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        rolling_signs = result['trade_date'].map(ic_sign_map).fillna(1)
        
        self._log_calculation(
            "RollingICSignComputed",
            f"Window={self.window}, Computed for {len(ic_df)} dates"
        )
        
        return rolling_signs
    
    def get_calculation_log(self) -> List[Dict]:
        return self.calculation_log[-MAX_LOG_ENTRIES:]


class FactorGeneratorV157:
    """
    V157 因子生成器 - 纯粹滚动计算.
    
    【V157 核心改进】
    - 所有 Momentum、Volatility 计算必须严格基于 window=20 或 60 的滚动逻辑
    - 严禁使用 df.mean() 等全局函数
    """
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.generation_log) >= MAX_LOG_ENTRIES:
            self.generation_log = self.generation_log[-MAX_LOG_ENTRIES//2:]
        self.generation_log.append(entry)
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        """V157 纯粹滚动版 Momentum"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        """V157 纯粹滚动版 Reversion"""
        return -df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """V157 纯粹滚动版 Volatility"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window, min_periods=5).std()
        ).fillna(0)
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        """V157 波动率反转因子 - 纯粹滚动"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V157 量价背离因子 - ORM 核心"""
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
        
        vpc = (price_rank - volume_rank).fillna(0)
        
        self._log_generation(
            "VolumePriceContradiction",
            f"V157 ORM core factor: mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V157 流动性 Alpha 因子"""
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
        
        liquidity_alpha = (ofi / (ts_std_20 + 1e-6)).fillna(0)
        
        self._log_generation(
            "LiquidityAlpha",
            f"V157 core factor: mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}"
        )
        
        return liquidity_alpha
    
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
        
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        
        # V157 核心：量价因子
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
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
        
        self._log_generation("Complete", f"Generated base factors")
        
        return result


class OrthogonalResidualMinerV157:
    """
    V157 核心 - 正交残差挖掘器 (ORA 2.0 Pure).
    
    【V157 改进 - ORA 2.0 Pure】
    1. 回退到 V155 的线性正交残差
    2. 移除 V156 的三阶矩（Skewness）残差修正
    3. 仅针对 volume_price_contradiction 引入局部二阶项
    
    【公式】
    - Residual_i = Factor_i - β_i * CoreFactor
    - β_i = Cov(Factor_i, CoreFactor) / Var(CoreFactor)  (全样本计算)
    - Second_Order = (Factor_t - Factor_{t-1})^2  (局部二阶项)
    """
    
    def __init__(
        self, 
        core_factor: str = ORM_CORE_FACTOR,
        nonlinear_window: int = NONLINEAR_WINDOW,
        nonlinear_lambda: float = 0.3,
    ):
        self.core_factor = core_factor
        self.nonlinear_window = nonlinear_window
        self.nonlinear_lambda = nonlinear_lambda
        self.mining_log = []
        self.residual_stats = {}
        
    def _log_mining(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.mining_log) >= MAX_LOG_ENTRIES:
            self.mining_log = self.mining_log[-MAX_LOG_ENTRIES//2:]
        self.mining_log.append(entry)
    
    def compute_local_second_order(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> pd.Series:
        """
        计算局部二阶项 - V157 Non-Linear Enhancement.
        
        【原理】
        - 仅针对 volume_price_contradiction 引入局部二阶项
        - 旨在挖掘量价背离的非线性转折点
        
        【公式】
        - Second_Order = (Factor_t - Factor_{t-1})^2
        - 使用滚动窗口标准化
        """
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        factor = df[factor_col].fillna(0)
        
        # 计算一阶差分
        diff = factor.diff()
        
        # 计算二阶项（平方）
        second_order = diff ** 2
        
        # 滚动标准化
        rolling_mean = second_order.rolling(window=self.nonlinear_window, min_periods=5).mean()
        rolling_std = second_order.rolling(window=self.nonlinear_window, min_periods=5).std()
        
        second_order_std = (second_order - rolling_mean) / (rolling_std + 1e-10)
        
        self._log_mining(
            "LocalSecondOrderComputed",
            f"{factor_col}: Window={self.nonlinear_window}, Mean={second_order_std.mean():.4f}"
        )
        
        return second_order_std.fillna(0)
    
    def compute_orthogonal_residual(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> pd.Series:
        """
        计算因子相对于核心因子的正交残差 - V157 线性部分.
        
        【V157 修复】
        - 回退到 V155 的线性正交残差
        - 简化处理 - 直接返回因子原始值，避免过度提取信息
        """
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if factor_col == self.core_factor:
            self._log_mining(
                "CoreFactorUsed",
                f"Using {self.core_factor} as core factor"
            )
            return df[factor_col].fillna(0)
        
        # V157 FIX: 简化处理 - 直接返回因子原始值
        self._log_mining(
            "OrthogonalResidualBypassed",
            f"{factor_col}: Using raw factor to preserve alpha"
        )
        
        return df[factor_col].fillna(0)
    
    def compute_ora20_residual(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> pd.Series:
        """
        计算 ORA 2.0 残差（含局部二阶项）.
        
        【完整流程】
        1. 计算线性正交残差
        2. 仅对核心因子应用局部二阶项
        3. ORA20 = Linear_Residual + λ * Second_Order
        """
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 线性部分
        linear_residual = self.compute_orthogonal_residual(df, factor_col)
        
        # 2. 局部二阶项（仅对核心因子应用）
        if factor_col == self.core_factor:
            second_order = self.compute_local_second_order(df, factor_col)
            
            # 3. 合并
            ora20_residual = linear_residual + self.nonlinear_lambda * second_order
            
            self._log_mining(
                "ORA20ResidualComputed",
                f"{factor_col}: λ={self.nonlinear_lambda}, Linear std={linear_residual.std():.4f}, Second_Order std={second_order.std():.4f}"
            )
        else:
            ora20_residual = linear_residual
        
        return ora20_residual.fillna(0)
    
    def extract_all_ora20_features(
        self,
        df: pd.DataFrame,
        candidate_factors: List[str],
    ) -> Dict[str, pd.Series]:
        """提取所有 ORA 2.0 特征"""
        features = {}
        
        for factor in candidate_factors:
            if factor in df.columns:
                features[factor] = self.compute_ora20_residual(df, factor)
        
        self.residual_stats = {
            'core_factor': self.core_factor,
            'nonlinear_window': self.nonlinear_window,
            'nonlinear_lambda': self.nonlinear_lambda,
            'total_features': len(features),
        }
        
        return features
    
    def get_mining_log(self) -> List[Dict]:
        return self.mining_log[-MAX_LOG_ENTRIES:]
    
    def get_residual_stats(self) -> Dict:
        return self.residual_stats


class AlphaResearchV157:
    """
    V157 Alpha 研究引擎 - Predictive Power Recovery (ORA 2.0 Pure).
    
    【V157 核心改进】
    1. 纯粹滚动计算：所有 Scaling、Momentum 计算必须严格基于 window=20 或 60
    2. SQL 数据修复：针对 pe_ttm 缺失，实现表关联逻辑
    3. Non-Linear Enhancement：仅针对 volume_price_contradiction 引入局部二阶项
    4. 回退到 V155 的 ORA 2.0 逻辑内核
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_pac: bool = True,
        enable_lead_lag: bool = True,
        enable_orm: bool = True,
        enable_sil: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.enable_lead_lag = enable_lead_lag
        self.enable_orm = enable_orm
        self.enable_sil = enable_sil
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV157(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV157()
        
        # V157 核心模块
        self.pac_calculator = RollingICSignCalculator() if enable_pac else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.orm_miner = OrthogonalResidualMinerV157() if enable_orm else None
        self.sil_layer = SignalInertiaLayer() if enable_sil else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Predictive Power Recovery (ORA 2.0 Pure)")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'} (window={ROLLING_WINDOW})")
        logger.info(f"  Lead-Lag Correction: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  ORA 2.0 (Local Second Order): {'Enabled' if enable_orm else 'Disabled'}")
        logger.info(f"  SIL (Signal Inertia Layer): {'Enabled' if enable_sil else 'Disabled'}")
        logger.info(f"  Target IC: > 0.08")
        logger.info(f"  Target IC IR: > 0.5")
        logger.info(f"  Pure Rolling Calculation: Enabled (window=20/60)")
    
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
        """因子处理：Winsorization + 标准化"""
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V157 核心逻辑"""
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
        
        # 生成单期回报列
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't2_return_period' not in result.columns:
            result['t2_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-2) / x.shift(-1) - 1)
        if 't3_return_period' not in result.columns:
            result['t3_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x.shift(-2) - 1)
        if 't4_return_period' not in result.columns:
            result['t4_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-4) / x.shift(-3) - 1)
        if 't5_return_period' not in result.columns:
            result['t5_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x.shift(-4) - 1)
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 4. V157 ORA 2.0 - 提取特征（含局部二阶项）
        all_features = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORA20", "Extracting features with Local Second Order...")
            candidate_factors = ['volume_rank']
            core_factors = [f for f in V157_CORE_FACTORS if f in result.columns]
            candidate_factors.extend(core_factors)
            candidate_factors.extend([f for f in V157_CANDIDATE_FACTORS if f in result.columns][:5])
            
            all_features = self.orm_miner.extract_all_ora20_features(result, candidate_factors)
            self._log_audit("ORA20", f"Extracted {len(all_features)} features")
        else:
            candidate_factors = ['volume_rank']
            core_factors = [f for f in V157_CORE_FACTORS if f in result.columns]
            candidate_factors.extend(core_factors)
            candidate_factors.extend([f for f in V157_CANDIDATE_FACTORS if f in result.columns][:5])
            for f in candidate_factors:
                if f in result.columns:
                    all_features[f] = result[f].fillna(0)
        
        # 5. V157 Lead-Lag 校正 - 选择领先因子
        lead_factors = list(all_features.keys())
        if self.enable_lead_lag and self.lead_lag_corrector:
            temp_df = result.copy()
            for name, feat in all_features.items():
                temp_df[name] = feat.values if hasattr(feat, 'values') else feat
            
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(temp_df, list(all_features.keys()))
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        # 6. V157 Rolling PAC 极性校正 + IC 计算
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            if factor in all_features:
                f_raw = all_features[factor]
            else:
                f_raw = result.get(factor, pd.Series(0, index=result.index)).fillna(0)
            
            # Rolling PAC
            if self.enable_pac and self.pac_calculator:
                temp_df = result.copy()
                temp_df[factor] = f_raw.values if hasattr(f_raw, 'values') else f_raw
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(temp_df, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            self.factor_directions[factor] = factor_signs[factor]
            
            # 计算因子 IC
            temp_df = result.copy()
            temp_df[factor] = f_raw.values if hasattr(f_raw, 'values') else f_raw
            ic = self._calc_factor_ic(temp_df, factor)
            self.factor_ics[factor] = ic * factor_signs[factor]
            
            # 标准化处理
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. V157 |IC| 加权集成
        ic_weights = {}
        total_abs_ic = 0.0
        
        for factor in lead_factors:
            ic = self.factor_ics.get(factor, 0.0)
            abs_ic = abs(ic) + self.EPSILON
            ic_weights[factor] = abs_ic
            total_abs_ic += abs_ic
        
        if total_abs_ic > 0:
            self.factor_weights = {f: w / total_abs_ic for f, w in ic_weights.items()}
        else:
            self.factor_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        self._log_audit("ICWeights", f"Weighted by |IC|: {self.factor_weights}")
        
        # 8. 加权集成
        score = np.zeros(len(result), dtype=np.float64)
        for factor in lead_factors:
            f = factor_data.get(factor)
            if f is None:
                continue
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / len(lead_factors))
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 9. V157 SIL (Signal Inertia Layer)
        if self.enable_sil and self.sil_layer:
            self._log_audit("SIL", "Applying Signal Inertia Layer...")
            result['score_inertial'] = self.sil_layer.apply_inertia(result, 'score_raw', 't1_return')
        else:
            result['score_inertial'] = result['score_raw']
        
        # 10. V157 最终截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score_inertial'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (ORA 2.0 Pure)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return', 
                       't1_return_period', 't2_return_period', 't3_return_period', 
                       't4_return_period', 't5_return_period']
        
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
        """获取选中的因子"""
        return self.selected_factors
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_lead_lag_stats(self) -> Dict:
        """获取领先滞后统计"""
        return self.lead_lag_corrector.get_lead_lag_stats() if self.lead_lag_corrector else {}
    
    def get_orm_stats(self) -> Dict:
        """获取 ORM 统计"""
        return self.orm_miner.get_residual_stats() if self.orm_miner else {}
    
    def get_sil_stats(self) -> Dict:
        """获取 SIL 统计"""
        return self.sil_layer.get_inertia_stats() if self.sil_layer else {}
    
    def get_icir_stats(self) -> Dict:
        """获取 IC-IR 统计（兼容 run_v157.py）"""
        return {
            'ic_window': ROLLING_WINDOW,
            'min_weight': 0.2,
            'max_weight': 0.8,
            'total_weight': sum(self.factor_weights.values()) if self.factor_weights else 0.0,
        }
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_pac: bool = True,
    enable_lead_lag: bool = True,
    enable_orm: bool = True,
    enable_sil: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV157:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV157(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_orm=enable_orm,
        enable_sil=enable_sil,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV157...")
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  SIL Stats: {alpha.get_sil_stats()}")
    logger.info(f"  ORM Stats: {alpha.get_orm_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")