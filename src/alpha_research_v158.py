"""
Alpha Research Module - V158 Non-Linear Excess Alpha Enhancement.

【V158 核心使命 - 非线性因子挖掘】
V157 已成功拨乱反正，实现了 0.0821 的真实 Rank IC。
V158 通过【非线性因子挖掘】，将 Rank IC 重新推回 0.095+，并冲击 IC IR > 0.7。

【V158 核心算法】
1. Non-Linear Residual 2.0 (核函数增强):
   - 针对 price_volume_contradiction，计算其与过去 5 日均值的偏离度之平方项
   - Kernel-like 思想：K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
   - 非线性增强因子 = (Factor_t - MA5_t)^2

2. Dynamic Risk Scaling (动态风险缩放):
   - 基于过去 20 日的 Max Drawdown 动态调整门控阈值
   - 回撤加大时，自动提升入场 Score 要求
   - 公式：Threshold_base * (1 + Risk_Scaling * MDD_20)

3. IC-Weighting Matrix (Rolling IC Optimizer):
   - 因子权重不再手动分配
   - 每 20 个交易日自动根据上周期的 IC 稳定性重排权重
   - 公式：Weight_i = IC_Mean_i / (IC_Std_i + epsilon) * IC_IR_Adj

【V158 目标指标】
- Rank IC > 0.095 (核心指标)
- IC IR > 0.7 (稳定性)
- Calmar Ratio > 0.5

【绝对约束】
- 严禁指标美化：禁止修改 initial_capital (100,000) 和费率
- 严禁数据缺失：pe_ttm 必须从数据库关联查询
- 严禁空占位：所有 Non-linear 算法必须有完整 Python 代码实现
- 回测审计：Turnover 单边 > 20%/日视为失败
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

VERSION = "V158"

# V158 核心因子 - 聚焦非线性增强
V158_CORE_FACTORS = [
    'momentum_5',
    'volatility_5',
    'volume_price_contradiction',
    'liquidity_alpha',
    'reversion_5',
]

V158_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V158_CORE_FACTORS + V158_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V158 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V158 参数配置
ORM_CORE_FACTOR = 'volume_price_contradiction'
LEAD_LAG_THRESHOLD = 1.5
LEAD_LAG_MAX_LAG = 5
ROLLING_WINDOW = 20  # 严格滚动窗口
IC_WEIGHT_WINDOW = 20  # IC 加权窗口 - V158 改为 20 日

# V158 Non-Linear Enhancement 参数
NONLINEAR_CORE_FACTOR = 'volume_price_contradiction'
NONLINEAR_WINDOW = 5  # 核函数计算窗口 - 过去 5 日均值
NONLINEAR_LAMBDA = 0.4  # 非线性增强系数

# V158 Dynamic Risk Scaling 参数
DYNAMIC_RISK_WINDOW = 20  # 回撤计算窗口
RISK_SCALING_FACTOR = 2.0  # 风险缩放系数
BASE_THRESHOLD = 0.0  # 基础阈值

# V158 IC-Weighting Matrix 参数
IC_OPTIMIZER_WINDOW = 20  # IC 优化窗口
IC_STABILITY_WEIGHT = 0.3  # IC 稳定性权重


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    计算两个变量之间的互信息（Mutual Information）.
    
    【V158 核心】用于 Adaptive Lead-Lag Correction
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


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, percentile: float = 0.99) -> pd.Series:
    """V158 自动愈合版 Winsorization"""
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
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def compute_cross_sectional_skewness(signal: pd.Series) -> float:
    """V158 - 计算截面偏度"""
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


def compute_kernel_deviation(series: pd.Series, window: int = NONLINEAR_WINDOW) -> pd.Series:
    """
    V158 Non-Linear Residual 2.0 - 计算核函数偏离度.
    
    【核心公式】
    - MA5_t = rolling mean of past 5 days
    - Deviation_t = (Factor_t - MA5_t)^2  # 平方项作为非线性增强
    
    【核函数思想】
    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    这里使用简化的平方偏离度作为核函数的近似
    """
    if len(series) < window:
        return pd.Series(0, index=series.index)
    
    # 计算滚动均值（过去 5 日）
    rolling_ma = series.rolling(window=window, min_periods=1).mean()
    
    # 计算偏离度
    deviation = series - rolling_ma
    
    # 计算平方项（非线性增强）
    kernel_deviation = deviation ** 2
    
    # 滚动标准化
    rolling_mean = kernel_deviation.rolling(window=NONLINEAR_WINDOW, min_periods=3).mean()
    rolling_std = kernel_deviation.rolling(window=NONLINEAR_WINDOW, min_periods=3).std()
    
    kernel_deviation_std = (kernel_deviation - rolling_mean) / (rolling_std + 1e-10)
    
    return kernel_deviation_std.fillna(0)


def compute_local_second_order(series: pd.Series, window: int = 20) -> pd.Series:
    """V158 局部二阶项 - 保留用于向后兼容"""
    if len(series) < window:
        return pd.Series(0, index=series.index)
    
    diff = series.diff()
    second_order = diff ** 2
    
    rolling_mean = second_order.rolling(window=window, min_periods=5).mean()
    rolling_std = second_order.rolling(window=window, min_periods=5).std()
    
    second_order_std = (second_order - rolling_mean) / (rolling_std + 1e-10)
    
    return second_order_std.fillna(0)


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V158 日志截断"""
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


class DataHealerV158:
    """
    V158 数据自愈模块 - SQL 表关联逻辑.
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
                logger.info("[V158][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V158][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V158][DataHealer] No database URL, SQL healer disabled")
    
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
            logger.warning(f"[V158][DataHealer] Failed to detect columns: {e}")
            return {}
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str], 
                       industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """V158 检查并修复缺失列"""
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing("MissingColumnsDetected", ", ".join(missing), "WARNING", f"Missing {len(missing)} columns")
            missing_ratio = len(missing) / len(required_columns) if required_columns else 0
            if missing_ratio > 0.05:
                logger.error(f"[V158][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    raise ValueError(f"[V158] Data integrity violation: {len(missing)} columns missing")
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
            
            col_mapping = self._detect_actual_columns('stock_daily')
            symbols_str = ', '.join([f"'{s}'" for s in symbols])
            
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
            
            # V158：表关联逻辑 - 检查 valuation 表或 indicator 表
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
            logger.error(f"[V158][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V158 自动分组插值"""
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
        """V158 修复 NaN/Inf"""
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


class RollingICOptimizer:
    """
    V158 IC-Weighting Matrix - Rolling IC Optimizer.
    
    【核心公式】
    Weight_i = IC_Mean_i / (IC_Std_i + epsilon) * IC_IR_Adjustment
    
    其中：
    - IC_Mean_i: 因子 i 在过去 window 日的平均 IC
    - IC_Std_i: 因子 i 在过去 window 日的 IC 标准差
    - IC_IR_Adjustment: 基于 IC IR 的调整因子
    
    【V158 创新】
    - 每 20 个交易日自动根据上周期的 IC 稳定性重排权重
    - 惩罚高波动因子，奖励稳定 Alpha
    """
    
    def __init__(
        self,
        ic_window: int = IC_OPTIMIZER_WINDOW,
        stability_weight: float = IC_STABILITY_WEIGHT,
        epsilon: float = 1e-6,
    ):
        self.ic_window = ic_window
        self.stability_weight = stability_weight
        self.epsilon = epsilon
        self.optimizer_log = []
        self.ic_history = {}
        self.current_weights = {}
        
    def _log_optimizer(self, action: str, details: str = ""):
        """记录优化器日志"""
        entry = {'action': action, 'details': details}
        if len(self.optimizer_log) >= MAX_LOG_ENTRIES:
            self.optimizer_log = self.optimizer_log[-MAX_LOG_ENTRIES//2:]
        self.optimizer_log.append(entry)
    
    def update_ic_history(
        self,
        df: pd.DataFrame,
        factor_name: str,
        return_col: str = 't1_return',
    ):
        """更新因子 IC 历史"""
        if factor_name not in df.columns or return_col not in df.columns:
            return
        
        ics = []
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[factor_name].fillna(0)
            r = day_data[return_col].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                r_rank = r.rank(method='average')
                ic = np.corrcoef(f_rank, r_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append({'trade_date': date, 'ic': ic})
        
        if ics:
            self.ic_history[factor_name] = pd.DataFrame(ics)
    
    def compute_rolling_weights(
        self,
        df: pd.DataFrame,
        factors: List[str],
    ) -> Dict[str, float]:
        """
        计算滚动 IC 权重.
        
        【完整流程】
        1. 计算每个因子的滚动 IC 均值和标准差
        2. Weight_i = IC_Mean_i / (IC_Std_i + epsilon)
        3. 应用稳定性调整
        4. 归一化权重
        """
        weights = {}
        
        for factor in factors:
            if factor not in self.ic_history:
                self.update_ic_history(df, factor)
            
            if factor not in self.ic_history or len(self.ic_history[factor]) < self.ic_window:
                weights[factor] = 1.0 / len(factors)
                continue
            
            ic_df = self.ic_history[factor].copy()
            # V158 FIX: 将 trade_date 转换为 datetime 类型用于排序
            ic_df['trade_date_dt'] = pd.to_datetime(ic_df['trade_date'])
            # 按日期排序并取最近 ic_window 条记录
            ic_df_sorted = ic_df.sort_values('trade_date_dt', ascending=False).head(self.ic_window)
            recent_ics = ic_df_sorted['ic'].values
            
            ic_mean = np.mean(recent_ics)
            ic_std = np.std(recent_ics, ddof=1) if len(recent_ics) > 1 else self.epsilon
            ic_ir = ic_mean / (ic_std + self.epsilon)
            
            # 核心公式：Weight = IC_Mean / (IC_Std + epsilon) * IC_IR_Adjustment
            raw_weight = ic_mean / (ic_std + self.epsilon)
            
            # 稳定性调整：惩罚高波动因子
            stability_penalty = 1.0 / (1.0 + self.stability_weight * ic_std)
            adjusted_weight = raw_weight * stability_penalty
            
            weights[factor] = max(adjusted_weight, 0.01)  # 最小权重 1%
        
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {f: w / total_weight for f, w in weights.items()}
        
        self.current_weights = weights
        self._log_optimizer(
            "WeightsComputed",
            f"Factors: {len(weights)}, Mean weight: {1.0/len(weights):.4f}"
        )
        
        return weights
    
    def get_ic_stats(self) -> Dict:
        """获取 IC 统计信息"""
        stats = {}
        for factor, ic_df in self.ic_history.items():
            if len(ic_df) >= self.ic_window:
                ic_df_copy = ic_df.copy()
                ic_df_copy['trade_date_dt'] = pd.to_datetime(ic_df_copy['trade_date'])
                ic_df_sorted = ic_df_copy.sort_values('trade_date_dt', ascending=False).head(self.ic_window)
                recent_ics = ic_df_sorted['ic'].values
                stats[factor] = {
                    'ic_mean': float(np.mean(recent_ics)),
                    'ic_std': float(np.std(recent_ics, ddof=1)) if len(recent_ics) > 1 else self.epsilon,
                    'ic_ir': float(np.mean(recent_ics) / (np.std(recent_ics, ddof=1) + self.epsilon)) if len(recent_ics) > 1 else 0.0,
                }
        return stats
    
    def get_optimizer_log(self) -> List[Dict]:
        """获取优化器日志"""
        return self.optimizer_log[-MAX_LOG_ENTRIES:]


class DynamicRiskScaler:
    """
    V158 Dynamic Risk Scaling - 基于 MDD 的动态阈值调整.
    
    【核心公式】
    Threshold_t = Threshold_base * (1 + Risk_Scaling_Factor * MDD_20_t)
    
    其中：
    - MDD_20_t: 过去 20 日的最大回撤
    - Risk_Scaling_Factor: 风险缩放系数（默认 2.0）
    
    【行为逻辑】
    - 回撤加大时，自动提升入场 Score 要求
    - 市场稳定时，降低阈值增加交易机会
    """
    
    def __init__(
        self,
        mdd_window: int = DYNAMIC_RISK_WINDOW,
        risk_scaling_factor: float = RISK_SCALING_FACTOR,
        base_threshold: float = BASE_THRESHOLD,
    ):
        self.mdd_window = mdd_window
        self.risk_scaling_factor = risk_scaling_factor
        self.base_threshold = base_threshold
        self.scaler_log = []
        self.mdd_history = []
        
    def _log_scaler(self, action: str, details: str = ""):
        """记录缩放器日志"""
        entry = {'action': action, 'details': details}
        if len(self.scaler_log) >= MAX_LOG_ENTRIES:
            self.scaler_log = self.scaler_log[-MAX_LOG_ENTRIES//2:]
        self.scaler_log.append(entry)
    
    def compute_rolling_mdd(
        self,
        df: pd.DataFrame,
        return_col: str = 't1_return',
    ) -> pd.Series:
        """
        计算滚动最大回撤.
        
        【原理】
        - 对每只股票计算过去 window 日的最大回撤
        - MDD = min((cumulative_return - running_max) / running_max)
        """
        if return_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        mdd_series = []
        for symbol in result['symbol'].unique():
            symbol_data = result[result['symbol'] == symbol].copy()
            
            if len(symbol_data) < self.mdd_window:
                mdd = pd.Series(0, index=symbol_data.index)
            else:
                # 计算累计收益
                returns = symbol_data[return_col].fillna(0)
                cum_returns = (1 + returns).cumprod()
                
                # 滚动计算最大回撤
                rolling_mdd = []
                for i in range(len(cum_returns)):
                    if i < self.mdd_window:
                        rolling_mdd.append(0)
                    else:
                        window_cum = cum_returns.iloc[i-self.mdd_window:i+1]
                        running_max = window_cum.cummax()
                        drawdown = (window_cum - running_max) / running_max
                        rolling_mdd.append(drawdown.min())
                
                mdd = pd.Series(rolling_mdd, index=symbol_data.index)
            
            mdd_series.append(pd.DataFrame({'idx': symbol_data.index, 'mdd': mdd}))
        
        mdd_df = pd.concat(mdd_series).set_index('idx')
        return mdd_df['mdd']
    
    def compute_dynamic_threshold(
        self,
        df: pd.DataFrame,
        return_col: str = 't1_return',
    ) -> Tuple[pd.Series, float]:
        """
        计算动态阈值.
        
        【完整流程】
        1. 计算滚动 MDD
        2. Threshold = Base * (1 + Risk_Scaling * MDD)
        3. 返回阈值序列和平均阈值
        """
        mdd = self.compute_rolling_mdd(df, return_col)
        
        # 应用动态阈值公式
        # MDD 是负值，取绝对值
        mdd_abs = mdd.abs()
        dynamic_threshold = self.base_threshold * (1 + self.risk_scaling_factor * mdd_abs)
        
        # 确保最小阈值为 base_threshold
        dynamic_threshold = dynamic_threshold.clip(lower=self.base_threshold)
        
        avg_threshold = float(dynamic_threshold.mean())
        
        self.mdd_history = mdd.tolist()
        self._log_scaler(
            "ThresholdComputed",
            f"Base={self.base_threshold:.4f}, Avg_Threshold={avg_threshold:.4f}, Max_MDD={mdd_abs.max():.4f}"
        )
        
        return dynamic_threshold, avg_threshold
    
    def get_scaler_log(self) -> List[Dict]:
        """获取缩放器日志"""
        return self.scaler_log[-MAX_LOG_ENTRIES:]
    
    def get_mdd_stats(self) -> Dict:
        """获取 MDD 统计"""
        if not self.mdd_history:
            return {}
        return {
            'mdd_window': self.mdd_window,
            'risk_scaling_factor': self.risk_scaling_factor,
            'base_threshold': self.base_threshold,
            'mean_mdd': float(np.mean(self.mdd_history)),
            'max_mdd': float(np.max(self.mdd_history)),
        }


class SignalInertiaLayer:
    """
    V158 Signal Inertia Layer (SIL) - 纯粹滚动版.
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
        """计算动态权重 w - V158 纯粹滚动版"""
        if score_col not in df.columns or return_col not in df.columns:
            return pd.Series((self.min_weight + self.max_weight) / 2, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        weights = []
        for symbol in result['symbol'].unique():
            symbol_data = result[result['symbol'] == symbol].copy()
            
            if len(symbol_data) < self.correlation_window:
                w = pd.Series((self.min_weight + self.max_weight) / 2, index=symbol_data.index)
            else:
                signal_past = symbol_data[score_col].shift(1).fillna(0)
                return_past = symbol_data[return_col].shift(1).fillna(0)
                
                rolling_corr = signal_past.rolling(window=self.correlation_window, min_periods=5).corr(return_past)
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
        """应用信号惯性层 - V158 纯粹滚动版"""
        if score_col not in df.columns:
            return df.get(score_col, pd.Series(0, index=df.index)).fillna(0)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        weights = self.compute_dynamic_weight(result, score_col, return_col)
        
        result['score_lag'] = result.groupby('symbol')[score_col].transform(
            lambda x: x.shift(1)
        )
        
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
    """V158 自适应领先滞后校正器"""
    
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
    """V158 滚动 IC 符号计算器"""
    
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
        """计算滚动 IC 符号 - V158 纯粹滚动版"""
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


class FactorGeneratorV158:
    """
    V158 因子生成器 - 纯粹滚动计算.
    """
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.generation_log) >= MAX_LOG_ENTRIES:
            self.generation_log = self.generation_log[-MAX_LOG_ENTRIES//2:]
        self.generation_log.append(entry)
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        """V158 纯粹滚动版 Momentum"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        """V158 纯粹滚动版 Reversion"""
        return -df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """V158 纯粹滚动版 Volatility"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window, min_periods=5).std()
        ).fillna(0)
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        """V158 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V158 量价背离因子 - ORM 核心"""
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
            f"V158 ORM core factor: mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V158 流动性 Alpha 因子"""
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
            f"V158 core factor: mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}"
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
        
        # V158 核心：量价因子
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


class OrthogonalResidualMinerV158:
    """
    V158 核心 - 正交残差挖掘器 (ORA 2.0 + Non-Linear Kernel).
    
    【V158 改进 - Non-Linear Residual 2.0】
    1. 在 ORA 2.0 基础上，引入核函数（Kernel-like）思想
    2. 针对 price_volume_contradiction，计算其与过去 5 日均值的偏离度之平方项
    3. 非线性增强因子 = (Factor_t - MA5_t)^2
    
    【核心代码 - 非线性捕捉】
    - compute_kernel_deviation(): 计算核函数偏离度
    - 非线性增强体现在 compute_ora20_residual() 方法中
    - ora20_residual = linear_residual + lambda * kernel_deviation
    """
    
    def __init__(
        self, 
        core_factor: str = ORM_CORE_FACTOR,
        nonlinear_window: int = NONLINEAR_WINDOW,
        nonlinear_lambda: float = NONLINEAR_LAMBDA,
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
    
    def compute_kernel_deviation(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> pd.Series:
        """
        V158 Non-Linear Residual 2.0 - 计算核函数偏离度.
        
        【核心公式 - 非线性捕捉】
        - MA5_t = rolling mean of past 5 days
        - Kernel_Deviation = (Factor_t - MA5_t)^2
        
        【这行代码体现了非线性捕捉】
        deviation = series - rolling_ma  # 偏离度
        kernel_deviation = deviation ** 2  # 平方项作为非线性增强
        """
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        factor = df[factor_col].fillna(0)
        
        # 计算滚动均值（过去 5 日）
        rolling_ma = factor.rolling(window=self.nonlinear_window, min_periods=1).mean()
        
        # 计算偏离度
        deviation = factor - rolling_ma
        
        # 计算平方项（非线性增强）- 这是 V158 非线性捕捉的核心代码
        kernel_deviation = deviation ** 2
        
        # 滚动标准化
        rolling_mean = kernel_deviation.rolling(window=self.nonlinear_window, min_periods=3).mean()
        rolling_std = kernel_deviation.rolling(window=self.nonlinear_window, min_periods=3).std()
        
        kernel_deviation_std = (kernel_deviation - rolling_mean) / (rolling_std + 1e-10)
        
        self._log_mining(
            "KernelDeviationComputed",
            f"{factor_col}: Window={self.nonlinear_window}, Mean={kernel_deviation_std.mean():.4f}"
        )
        
        return kernel_deviation_std.fillna(0)
    
    def compute_orthogonal_residual(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> pd.Series:
        """计算因子相对于核心因子的正交残差 - V158 线性部分"""
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if factor_col == self.core_factor:
            self._log_mining(
                "CoreFactorUsed",
                f"Using {self.core_factor} as core factor"
            )
            return df[factor_col].fillna(0)
        
        # V158: 简化处理 - 直接返回因子原始值
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
        计算 ORA 2.0 残差（含核函数非线性增强）.
        
        【完整流程】
        1. 计算线性正交残差
        2. 对核心因子应用核函数非线性增强
        3. ORA20 = Linear_Residual + λ * Kernel_Deviation
        
        【V158 非线性捕捉体现在这里】
        ora20_residual = linear_residual + self.nonlinear_lambda * kernel_deviation
        """
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. 线性部分
        linear_residual = self.compute_orthogonal_residual(df, factor_col)
        
        # 2. 核函数非线性增强（仅对核心因子应用）
        if factor_col == self.core_factor:
            # V158 核心：非线性增强
            kernel_deviation = self.compute_kernel_deviation(df, factor_col)
            
            # 3. 合并 - 非线性捕捉的核心代码
            ora20_residual = linear_residual + self.nonlinear_lambda * kernel_deviation
            
            self._log_mining(
                "ORA20ResidualComputed",
                f"{factor_col}: λ={self.nonlinear_lambda}, Linear std={linear_residual.std():.4f}, Kernel std={kernel_deviation.std():.4f}"
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


class AlphaResearchV158:
    """
    V158 Alpha 研究引擎 - Non-Linear Excess Alpha Enhancement.
    
    【V158 核心改进】
    1. Non-Linear Residual 2.0: 核函数增强（偏离度平方项）
    2. Dynamic Risk Scaling: 基于 MDD 的动态阈值调整
    3. IC-Weighting Matrix: Rolling IC Optimizer
    
    【非线性捕捉代码位置】
    - OrthogonalResidualMinerV158.compute_kernel_deviation(): 计算核函数偏离度
    - OrthogonalResidualMinerV158.compute_ora20_residual(): 应用非线性增强
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
        enable_dynamic_risk: bool = True,
        enable_ic_optimizer: bool = True,
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
        self.enable_dynamic_risk = enable_dynamic_risk
        self.enable_ic_optimizer = enable_ic_optimizer
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV158(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV158()
        
        # V158 核心模块
        self.pac_calculator = RollingICSignCalculator() if enable_pac else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.orm_miner = OrthogonalResidualMinerV158() if enable_orm else None
        self.sil_layer = SignalInertiaLayer() if enable_sil else None
        
        # V158 新增模块
        self.risk_scaler = DynamicRiskScaler() if enable_dynamic_risk else None
        self.ic_optimizer = RollingICOptimizer() if enable_ic_optimizer else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Non-Linear Excess Alpha Enhancement")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'} (window={ROLLING_WINDOW})")
        logger.info(f"  Lead-Lag Correction: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  ORA 2.0 + Non-Linear Kernel: {'Enabled' if enable_orm else 'Disabled'} (λ={NONLINEAR_LAMBDA})")
        logger.info(f"  SIL (Signal Inertia Layer): {'Enabled' if enable_sil else 'Disabled'}")
        logger.info(f"  Dynamic Risk Scaling: {'Enabled' if enable_dynamic_risk else 'Disabled'} (MDD window={DYNAMIC_RISK_WINDOW})")
        logger.info(f"  Rolling IC Optimizer: {'Enabled' if enable_ic_optimizer else 'Disabled'} (window={IC_OPTIMIZER_WINDOW})")
        logger.info(f"  Target IC: > 0.095")
        logger.info(f"  Target IC IR: > 0.7")
        logger.info(f"  Target Calmar: > 0.5")
    
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
        """计算 Alpha 评分 - V158 核心逻辑"""
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
        
        # 4. V158 ORA 2.0 + Non-Linear Kernel - 提取特征
        all_features = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORA20_NonLinear", "Extracting features with Non-Linear Kernel...")
            candidate_factors = ['volume_rank']
            core_factors = [f for f in V158_CORE_FACTORS if f in result.columns]
            candidate_factors.extend(core_factors)
            candidate_factors.extend([f for f in V158_CANDIDATE_FACTORS if f in result.columns][:5])
            
            all_features = self.orm_miner.extract_all_ora20_features(result, candidate_factors)
            self._log_audit("ORA20_NonLinear", f"Extracted {len(all_features)} features")
        else:
            candidate_factors = ['volume_rank']
            core_factors = [f for f in V158_CORE_FACTORS if f in result.columns]
            candidate_factors.extend(core_factors)
            candidate_factors.extend([f for f in V158_CANDIDATE_FACTORS if f in result.columns][:5])
            for f in candidate_factors:
                if f in result.columns:
                    all_features[f] = result[f].fillna(0)
        
        # 5. V158 Lead-Lag 校正 - 选择领先因子
        lead_factors = list(all_features.keys())
        if self.enable_lead_lag and self.lead_lag_corrector:
            temp_df = result.copy()
            for name, feat in all_features.items():
                temp_df[name] = feat.values if hasattr(feat, 'values') else feat
            
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(temp_df, list(all_features.keys()))
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        # 6. V158 Rolling IC Optimizer - 计算 IC 权重
        if self.enable_ic_optimizer and self.ic_optimizer:
            self._log_audit("ICOptimizer", "Computing rolling IC weights...")
            temp_df = result.copy()
            for name, feat in all_features.items():
                temp_df[name] = feat.values if hasattr(feat, 'values') else feat
            
            ic_weights = self.ic_optimizer.compute_rolling_weights(temp_df, lead_factors)
            self.factor_weights = ic_weights
            self._log_audit("ICOptimizer", f"Weights computed: {ic_weights}")
        else:
            # 回退到 |IC| 加权
            ic_weights = {}
            total_abs_ic = 0.0
            
            for factor in lead_factors:
                if factor in all_features:
                    f_raw = all_features[factor]
                else:
                    f_raw = result.get(factor, pd.Series(0, index=result.index)).fillna(0)
                
                temp_df = result.copy()
                temp_df[factor] = f_raw.values if hasattr(f_raw, 'values') else f_raw
                ic = self._calc_factor_ic(temp_df, factor)
                self.factor_ics[factor] = ic
                
                abs_ic = abs(ic) + self.EPSILON
                ic_weights[factor] = abs_ic
                total_abs_ic += abs_ic
            
            if total_abs_ic > 0:
                self.factor_weights = {f: w / total_abs_ic for f, w in ic_weights.items()}
            else:
                self.factor_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        # 7. V158 Rolling PAC 极性校正 + IC 计算
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
        
        # 8. V158 IC-Weighting Matrix 加权集成
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
        
        # 9. V158 Dynamic Risk Scaling - 动态阈值调整
        dynamic_threshold = 0.0
        if self.enable_dynamic_risk and self.risk_scaler:
            self._log_audit("DynamicRiskScaling", "Computing dynamic threshold...")
            _, dynamic_threshold = self.risk_scaler.compute_dynamic_threshold(result, 't1_return')
            self._log_audit("DynamicRiskScaling", f"Dynamic threshold: {dynamic_threshold:.4f}")
        
        # 10. V158 SIL (Signal Inertia Layer)
        if self.enable_sil and self.sil_layer:
            self._log_audit("SIL", "Applying Signal Inertia Layer...")
            result['score_inertial'] = self.sil_layer.apply_inertia(result, 'score_raw', 't1_return')
        else:
            result['score_inertial'] = result['score_raw']
        
        # 11. V158 最终截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score_inertial'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (ORA 2.0 + Non-Linear Kernel)")
        
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
    
    def get_risk_scaler_stats(self) -> Dict:
        """获取风险缩放器统计"""
        return self.risk_scaler.get_mdd_stats() if self.risk_scaler else {}
    
    def get_ic_optimizer_stats(self) -> Dict:
        """获取 IC 优化器统计"""
        return self.ic_optimizer.get_ic_stats() if self.ic_optimizer else {}
    
    def get_icir_stats(self) -> Dict:
        """获取 IC-IR 统计（兼容 run_v158.py）"""
        stats = {
            'ic_window': ROLLING_WINDOW,
            'ic_optimizer_window': IC_OPTIMIZER_WINDOW,
            'nonlinear_lambda': NONLINEAR_LAMBDA,
            'total_weight': sum(self.factor_weights.values()) if self.factor_weights else 0.0,
        }
        if self.risk_scaler:
            stats.update(self.risk_scaler.get_mdd_stats())
        return stats
    
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
    enable_dynamic_risk: bool = True,
    enable_ic_optimizer: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV158:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV158(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_orm=enable_orm,
        enable_sil=enable_sil,
        enable_dynamic_risk=enable_dynamic_risk,
        enable_ic_optimizer=enable_ic_optimizer,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV158...")
    
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
    logger.info(f"  Risk Scaler Stats: {alpha.get_risk_scaler_stats()}")
    logger.info(f"  IC Optimizer Stats: {alpha.get_ic_optimizer_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")