"""
Alpha Research Module - V157 IC-IR Optimized Weighting.

【V156 审计结论】
- IC: 0.0924 (优秀)
- IR: 0.58 (未达到工业级 0.7)

【V157 核心使命 - IR 攻坚】
1. 保留 V156 的 Rolling PAC 极性校正
2. 保留 V156 的 Lead-Lag 校正
3. 保留 V156 的 |IC| 加权
4. 新增 IC-IR 权重调整：Weight = |IC| / Std(IC)  # 惩罚高波动因子

【V157 目标】
- IC: > 0.09 (保持 V156 水平)
- IR: > 0.7 (稳定性提升)
- Total Return: > 0 (严禁零交易)

【工程纪律】
- 基于 V156 最小改动
- 严禁修改 src/engine/ 目录
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

# V157 核心因子 - 聚焦短期预测 (同 V156)
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

# V157 参数 - 继承 V156
ORM_CORE_FACTOR = 'volume_price_contradiction'
LEAD_LAG_THRESHOLD = 1.5
LEAD_LAG_MAX_LAG = 5
ROLLING_WINDOW = 20
ORA3_INTERACTION_PAIRS = [
    ('volume_price_contradiction', 'momentum_5'),
    ('volume_price_contradiction', 'volatility_5'),
    ('volume_price_contradiction', 'reversion_5'),
    ('volume_price_contradiction', 'liquidity_alpha'),
    ('momentum_5', 'volatility_5'),
]

# V157 新增 - IC-IR 权重参数
ICIR_IC_WINDOW = 20
ICIR_MIN_WEIGHT = 0.01


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
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
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    std = series_clean.std()
    if std > 1e-10:
        lower = mean - sigma * std
        upper = mean + sigma * std
        series_clean = series_clean.clip(lower=lower, upper=upper)
    lower_pct = series_clean.quantile(1 - percentile)
    upper_pct = series_clean.quantile(percentile)
    series_clean = series_clean.clip(lower=lower_pct, upper=upper_pct)
    series_clean = series_clean.ffill().bfill().fillna(mean)
    return series_clean


def compute_cross_sectional_skewness(signal: pd.Series) -> float:
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


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
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
    """V157 数据自愈模块 - 动态字段映射"""
    
    FIELD_MAPPING = {
        'pe_ttm': ['pe_ttm', 'pe_ttm_new', 'pe', 'pe_ly', 'pe_static'],
        'pb': ['pb', 'pb_new', 'pb_ly', 'pb_static'],
        'ps_ttm': ['ps_ttm', 'ps', 'ps_ly'],
        'pcf_ocf': ['pcf_ocf', 'pcf', 'pcf_ly'],
        'total_mv': ['total_mv', 'market_value', 'mv_total'],
        'circ_mv': ['circ_mv', 'market_value_float', 'mv_float'],
        'turnover_rate': ['turnover_rate', 'turnover', 'turnover_rate_daily'],
    }
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._detected_columns = {}
        self._init_sql_healer()
        
    def _init_sql_healer(self):
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
        self._log_healing("MultiLevelImputeApplied", "ALL_NUMERIC", "SUCCESS", "Applied SQL -> Median -> Industry Mean")
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
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
            base_columns = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 
                           'volume', 'amount', 'turnover_rate', 'total_mv']
            select_columns = []
            for col in base_columns:
                actual_col = col_mapping.get(col, col)
                select_columns.append(f"{actual_col} AS {col}")
            for col in columns:
                if col in col_mapping:
                    actual_col = col_mapping[col]
                    if actual_col != col:
                        select_columns.append(f"{actual_col} AS {col}")
                    else:
                        select_columns.append(col)
                else:
                    select_columns.append(col)
            query = text(f"""
                SELECT {', '.join(select_columns)}
                FROM stock_daily
                WHERE symbol IN ({symbols_str})
                AND trade_date BETWEEN :start_date AND :end_date
            """)
            sql_df = pd.read_sql_query(query, self.engine, params={
                'start_date': start_date, 'end_date': end_date,
            })
            if not sql_df.empty:
                for col in columns:
                    if col in sql_df.columns:
                        merge_df = result.merge(
                            sql_df[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'], how='left', suffixes=('', '_sql')
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
    
    def _repair_nan_inf(self, df: pd.DataFrame, industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
                if industry_data is not None and 'industry' in industry_data.columns:
                    result = self._fill_with_industry_mean(result, col, industry_data)
                result[col] = result[col].fillna(col_median)
                self._log_healing("NaNRepaired_MultiLevel", col, "SUCCESS", f"Repaired {nan_count} NaN values")
        return result
    
    def _fill_with_industry_mean(self, df: pd.DataFrame, col: str, industry_data: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if 'symbol' in result.columns and 'symbol' in industry_data.columns:
            merged = result.merge(industry_data[['symbol', 'industry']], on='symbol', how='left')
            if 'industry' in merged.columns:
                industry_means = merged.groupby('industry')[col].transform('mean')
                nan_mask = result[col].isna()
                if nan_mask.any():
                    result.loc[nan_mask, col] = industry_means[nan_mask].fillna(result[col].median())
        return result
    
    def get_healing_log(self) -> List[Dict]:
        return self.healing_log[-MAX_LOG_ENTRIES:]


class AdaptiveLeadLagCorrector:
    """V157 自适应领先滞后校正器 - 同 V156"""
    
    def __init__(self, max_lag: int = LEAD_LAG_MAX_LAG, threshold: float = LEAD_LAG_THRESHOLD, n_bins: int = 10):
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
    
    def compute_lead_lag_score(self, df: pd.DataFrame, factor_col: str, 
                                return_cols: Optional[List[str]] = None) -> Tuple[float, Dict[int, float]]:
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
        
        self._log_correction("LeadLagScoreComputed", 
                            f"{factor_col}: MI_Lag1={mi_lag_1:.4f}, MI_Lag5={mi_lag_5:.4f}, Score={lead_lag_score:.2f}")
        
        return lead_lag_score, mi_by_lag
    
    def select_lead_factors(self, df: pd.DataFrame, candidate_factors: List[str]) -> List[str]:
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
        
        self._log_correction("LeadFactorsSelected", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        return lead_factors
    
    def get_correction_log(self) -> List[Dict]:
        return self.correction_log[-MAX_LOG_ENTRIES:]
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class RollingICSignCalculator:
    """V157 滚动 IC 符号计算器 - 同 V156"""
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        self.calculation_log = []
        
    def _log_calculation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.calculation_log) >= MAX_LOG_ENTRIES:
            self.calculation_log = self.calculation_log[-MAX_LOG_ENTRIES//2:]
        self.calculation_log.append(entry)
    
    def compute_rolling_ic_sign(self, df: pd.DataFrame, factor_col: str, 
                                 return_col: str = 't1_return') -> pd.Series:
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
        
        self._log_calculation("RollingICSignComputed", 
                             f"Window={self.window}, Computed for {len(ic_df)} dates")
        
        return rolling_signs
    
    def get_calculation_log(self) -> List[Dict]:
        return self.calculation_log[-MAX_LOG_ENTRIES:]


class FactorGeneratorV157:
    """V157 因子生成器 - 同 V156"""
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.generation_log) >= MAX_LOG_ENTRIES:
            self.generation_log = self.generation_log[-MAX_LOG_ENTRIES//2:]
        self.generation_log.append(entry)
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(lambda x: x.pct_change(window)).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        return -df.groupby('symbol')['close'].transform(lambda x: x.pct_change(window)).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(lambda x: x.pct_change().rolling(window).std()).fillna(0)
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        vol_10 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(10, min_periods=5).std()).fillna(0)
        vol_20 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=10).std()).fillna(0)
        return -(vol_10 - vol_20).fillna(0)
    
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
        vpc = (price_rank - volume_rank).fillna(0)
        self._log_generation("VolumePriceContradiction", f"mean={vpc.mean():.4f}, std={vpc.std():.4f}")
        return vpc
    
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
            ts_std_20 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=5).std())
        else:
            ts_std_20 = pd.Series(1, index=df.index)
        liquidity_alpha = (ofi / (ts_std_20 + 1e-6)).fillna(0)
        self._log_generation("LiquidityAlpha", f"mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}")
        return liquidity_alpha
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        self._log_generation("StartFactorGeneration", f"Processing {len(df)} rows")
        result['momentum_5'] = self.compute_momentum(result, 5)
        result['momentum_10'] = self.compute_momentum(result, 10)
        result['momentum_20'] = self.compute_momentum(result, 20)
        result['momentum_60'] = self.compute_momentum(result, 60)
        result['reversion_5'] = self.compute_reversion(result, 5)
        result['reversion_10'] = self.compute_reversion(result, 10)
        result['volatility_5'] = self.compute_volatility(result, 5)
        result['volatility_10'] = self.compute_volatility(result, 10)
        result['volatility_20'] = self.compute_volatility(result, 20)
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)).fillna(0.5)
        self._log_generation("Complete", "Generated base factors")
        return result


class AlphaResearchV157:
    """
    V157 Alpha 研究引擎 - IC-IR Optimized Weighting.
    
    【V157 核心改进】
    1. 保留 V156 Rolling PAC 极性校正
    2. 保留 V156 Lead-Lag 校正
    3. V157 IC-IR Weighting: Weight = |IC| / Std(IC)  # 惩罚高波动因子
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
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.ic_series = {}  # 保存 IC 序列用于 IC-IR 计算
        self.selected_factors = []
        self.audit_log = []
        
        self.data_healer = DataHealerV157(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV157()
        self.pac_calculator = RollingICSignCalculator() if enable_pac else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: IC-IR Optimized Weighting")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'}")
        logger.info(f"  Lead-Lag Correction: {'Enabled' if enable_lead_lag else 'Disabled'}")
        logger.info(f"  IC-IR Weighting: Enabled (|IC| / Std(IC))")
        logger.info(f"  Target IR: > 0.7")
        logger.info(f"  Target IC: > 0.09")
    
    def _log_audit(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.audit_log) >= MAX_LOG_ENTRIES:
            self.audit_log = self.audit_log[-MAX_LOG_ENTRIES//2:]
        self.audit_log.append(entry)
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
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
    
    def _calc_factor_ic_series(self, df: pd.DataFrame, factor_col: str) -> List[float]:
        """计算因子 IC 序列（按日期）"""
        ics_by_date = []
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
                    ics_by_date.append(ic)
        return ics_by_date
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_icir_weights(self, factor_ics: Dict[str, List[float]], 
                             factor_signs: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        V157 核心 - 计算 IC-IR 权重.
        
        【公式】
        Weight = |Mean(IC)| / Std(IC)  # 惩罚高波动因子
        Direction = sign(Mean(IC))  # 基于实际 IC 均值决定方向
        """
        raw_weights = {}
        factor_directions = {}
        
        for factor, ics in factor_ics.items():
            if not ics or len(ics) < 5:
                raw_weights[factor] = self.EPSILON
                factor_directions[factor] = 1.0  # 默认正方向
                continue
            
            ic_array = np.array(ics)
            ic_mean = np.mean(ic_array)
            ic_std = np.std(ic_array) + self.EPSILON
            
            # V157: 使用 |IC| / Std(IC)
            abs_ic = abs(ic_mean)
            raw_weight = abs_ic / ic_std
            
            # V157 FIX: 方向完全由 IC 均值决定，忽略 PAC 符号
            # 因为 PAC 可能导致错误的方向反转
            factor_directions[factor] = 1.0 if ic_mean >= 0 else -1.0
            raw_weights[factor] = max(raw_weight, self.EPSILON)
            
            logger.info(f"[V157][ICIR] {factor}: IC_mean={ic_mean:.4f}, IC_std={ic_std:.4f}, Weight={raw_weight:.4f}, Dir={factor_directions[factor]}")
        
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            normalized_weights = {f: w / total_weight for f, w in raw_weights.items()}
        else:
            n_factors = len(raw_weights)
            normalized_weights = {f: 1.0 / n_factors for f in raw_weights}
        
        return normalized_weights, factor_directions
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V157 核心逻辑"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        result = df.copy()
        
        # 1. 数据自愈
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
        
        # 4. 准备因子列表
        candidate_factors = ['volume_rank']
        core_factors = [f for f in V157_CORE_FACTORS if f in result.columns]
        candidate_factors.extend(core_factors)
        candidate_factors.extend([f for f in V157_CANDIDATE_FACTORS if f in result.columns][:5])
        
        # 5. V157 Lead-Lag 校正 - 选择领先因子
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        # 6. V157 Rolling PAC 极性校正 + IC 序列计算
        factor_data = {}
        factor_signs = {}
        factor_ic_series = {}
        
        for factor in lead_factors:
            if factor not in result.columns:
                continue
            f_raw = result[factor].fillna(0)
            
            # Rolling PAC
            if self.enable_pac and self.pac_calculator:
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            # V157 FIX: 计算 IC 序列时必须使用已乘以 rolling_sign 的因子数据
            # 否则 IC-IR 权重计算的方向与 PAC 校正后的因子方向不一致
            result_with_factor = result.copy()
            result_with_factor[f'{factor}_pac'] = f_processed
            ic_series = self._calc_factor_ic_series(result_with_factor, f'{factor}_pac')
            factor_ic_series[factor] = ic_series
            
            # 保存因子方向
            self.factor_directions[factor] = factor_signs[factor]
            
            # 保存因子 IC（带 PAC 符号）
            ic = self._calc_factor_ic(result_with_factor, f'{factor}_pac')
            self.factor_ics[factor] = ic
            
            # 标准化处理
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. V157 IC-IR Optimized Weighting
        self._log_audit("ICIR", "Computing IC-IR optimized weights...")
        self.factor_weights, final_directions = self.compute_icir_weights(factor_ic_series, factor_signs)
        self._log_audit("ICIRWeights", f"Weights: {self.factor_weights}, Directions: {final_directions}")
        
        # 8. 加权集成（使用 IC-IR 权重 + 因子方向）
        score = np.zeros(len(result), dtype=np.float64)
        
        for factor in lead_factors:
            if factor not in factor_data:
                continue
            f = factor_data[factor]
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / len(lead_factors))
            direction = final_directions.get(factor, 1.0)
            # V157 FIX: 使用 IC-IR 计算的方向
            score += f_clean.values * weight * direction
            self.factor_directions[factor] = direction
        
        result['score_raw'] = score
        
        # 9. 最终截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score_raw'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", "Final score computed")
        
        # V157 FIX: 保留 t1_return 列供 backtest_referee 使用
        # 但需要确保 backtest_referee 正确处理列冲突
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return',
                       't1_return_period', 't2_return_period', 't3_return_period',
                       't4_return_period', 't5_return_period']
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        return self.factor_ics
    
    def get_selected_factors(self) -> List[str]:
        return self.selected_factors
    
    def get_data_healing_log(self) -> List[Dict]:
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_corrector.get_lead_lag_stats() if self.lead_lag_corrector else {}
    
    def get_pac_stats(self) -> Dict:
        return self.pac_calculator.get_calculation_log() if hasattr(self.pac_calculator, 'get_calculation_log') else {}
    
    def get_icir_stats(self) -> Dict:
        """获取 IC-IR 权重统计信息"""
        return {
            'ic_window': ICIR_IC_WINDOW,
            'min_weight': ICIR_MIN_WEIGHT,
            'total_weight': sum(self.factor_weights.values()) if self.factor_weights else 0.0,
            'weights': self.factor_weights,
            'directions': self.factor_directions,
        }
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_pac: bool = True,
    enable_lead_lag: bool = True,
    enable_orm: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV157:
    return AlphaResearchV157(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_orm=enable_orm,
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
    logger.info(f"  ATG Stats: {alpha.get_icir_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")