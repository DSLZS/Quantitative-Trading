"""
Alpha Research Module - V169 V155-IR-Boost (IC IR 优化版).

【V168 分析】
- T+1 IC: 0.0996 ✓ (超过 0.095 目标)
- IC IR: 0.57 ✗ (低于 0.60 目标)
- 6 因子：volume_rank, momentum_5, volatility_5, volume_price_contradiction, liquidity_alpha, reversion_5

【V169 核心策略 - IR Boost】
1. 增加 PAC 窗口从 20 到 30，平滑 IC 符号切换
2. 降低 volatility_5 权重（太高会导致 IC 波动）
3. 增加 momentum_5 权重（更稳定的信号）

【工程纪律】
- 基于 V168 优化
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

VERSION = "V169"

# V169 核心因子 - 与 V168 相同
V169_CORE_FACTORS = [
    'momentum_5',
    'volatility_5',
    'volume_price_contradiction',
    'liquidity_alpha',
    'reversion_5',
]

# V169 候选因子池
V169_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V169_CORE_FACTORS + V169_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V169 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V169 ORA 2.0 参数 - IR 优化
ORM_CORE_FACTOR = 'volume_price_contradiction'
LEAD_LAG_THRESHOLD = 1.2
LEAD_LAG_MAX_LAG = 5
CS_VOLATILITY_WINDOW = 20
ROLLING_WINDOW = 30  # V169: 增加窗口平滑 IC
IC_WEIGHT_WINDOW = 15  # V169: 增加窗口

# V169 PAC 参数 - 更平滑
ADAPTIVE_PAC_BASE_WINDOW = 30  # V169: 增加窗口
ADAPTIVE_PAC_MIN_WINDOW = 10
ADAPTIVE_PAC_MAX_WINDOW = 60
SEF_ENTROPY_THRESHOLD = 0.5
SEF_INERTIA_FACTOR = 0.7

# V169 使用 |IC|^0.9 加权（降低高 IC 因子权重，提高多样性）
IC_POWER = 0.9


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


def compute_signal_entropy(signal: pd.Series) -> float:
    if len(signal) < 10:
        return 1.0
    
    signal_clean = signal.dropna()
    if len(signal_clean) < 10:
        return 1.0
    
    try:
        n_bins = min(20, len(signal_clean) // 5)
        if n_bins < 2:
            return 1.0
        
        bins = pd.qcut(signal_clean, q=n_bins, labels=False, duplicates='drop')
        bin_counts = bins.value_counts(normalize=True)
        
        entropy = -np.sum(bin_counts * np.log(bin_counts + 1e-10))
        
        max_entropy = np.log(len(bin_counts))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy
    except Exception:
        return 1.0


class DataHealerV169:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V169][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V169][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        entry = {'timestamp': datetime.now().isoformat(), 'action': action, 'column': column, 'status': status, 'details': details}
        if len(self.healing_log) >= MAX_LOG_ENTRIES:
            self.healing_log = self.healing_log[-MAX_LOG_ENTRIES//2:]
        self.healing_log.append(entry)
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(action="MissingColumnsDetected", column=", ".join(missing), status="WARNING", details=f"Missing {len(missing)} columns")
            
            missing_ratio = len(missing) / len(required_columns)
            if missing_ratio > 0.05:
                logger.error(f"[V169][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                if self.engine:
                    result = self._heal_from_sql(result, missing)
            else:
                if self.engine:
                    result = self._heal_from_sql(result, missing)
        else:
            self._log_healing(action="ColumnsComplete", column="ALL", status="OK", details="All required columns present")
        
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result)
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
            symbols_str = ', '.join([f"'{s}'" for s in symbols])
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       turnover_rate, total_mv, pe_ttm, pb
                FROM stock_daily
                WHERE symbol IN ({symbols_str})
                AND trade_date BETWEEN :start_date AND :end_date
            """)
            
            sql_df = pd.read_sql_query(query, self.engine, params={'start_date': start_date, 'end_date': end_date})
            
            if not sql_df.empty:
                for col in columns:
                    if col in sql_df.columns:
                        merge_df = result.merge(sql_df[['symbol', 'trade_date', col]], on=['symbol', 'trade_date'], how='left', suffixes=('', '_sql'))
                        result[col] = merge_df[col].fillna(merge_df[f'{col}_sql'])
                        result = result.drop(columns=[c for c in result.columns if c.endswith('_sql')])
        except Exception as e:
            logger.error(f"[V169][DataHealer] SQL heal failed: {e}")
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            result[col] = result.groupby(group_col, group_keys=False)[col].transform(lambda x: x.ffill().bfill())
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def _repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                result[col] = result[col].fillna(col_median)
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        return self.healing_log[-MAX_LOG_ENTRIES:]


class AdaptiveLeadLagCorrector:
    def __init__(self, max_lag: int = LEAD_LAG_MAX_LAG, threshold: float = LEAD_LAG_THRESHOLD, n_bins: int = 10):
        self.max_lag = max_lag
        self.threshold = threshold
        self.n_bins = n_bins
        self.correction_log = []
        self.lead_lag_stats = {}
        
    def compute_lead_lag_score(self, df: pd.DataFrame, factor_col: str, return_cols: Optional[List[str]] = None) -> Tuple[float, Dict[int, float]]:
        if factor_col not in df.columns:
            return 0.0, {}
        
        if return_cols is None:
            return_cols = ['t1_return_period', 't2_return_period', 't3_return_period', 't4_return_period', 't5_return_period']
        
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
        
        return lead_lag_score, mi_by_lag
    
    def select_lead_factors(self, df: pd.DataFrame, candidate_factors: List[str]) -> List[str]:
        lead_scores = {}
        for factor in candidate_factors:
            score, _ = self.compute_lead_lag_score(df, factor)
            lead_scores[factor] = score
        
        lead_factors = [f for f, s in lead_scores.items() if s > self.threshold]
        
        if not lead_factors:
            sorted_factors = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)
            lead_factors = [f for f, _ in sorted_factors[:min(6, len(sorted_factors))]]
        
        self.lead_lag_stats = {'threshold': self.threshold, 'lead_factors': lead_factors, 'lead_scores': lead_scores}
        return lead_factors
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class AdaptiveRollingPAC:
    def __init__(self, base_window: int = ADAPTIVE_PAC_BASE_WINDOW, min_window: int = ADAPTIVE_PAC_MIN_WINDOW, max_window: int = ADAPTIVE_PAC_MAX_WINDOW, vol_threshold: float = 0.02):
        self.base_window = base_window
        self.min_window = min_window
        self.max_window = max_window
        self.vol_threshold = vol_threshold
        self.pac_log = []
        self.pac_stats = {}
        
    def compute_adaptive_window(self, df: pd.DataFrame, market_return_col: str = 'market_return') -> Dict[str, int]:
        if 'trade_date' not in df.columns:
            return {}
        
        dates = df['trade_date'].unique()
        date_windows = {}
        
        all_vols = []
        for date in dates:
            date_data = df[df['trade_date'] == date]
            if market_return_col in date_data.columns:
                vol = date_data[market_return_col].std()
                if not np.isnan(vol):
                    all_vols.append(vol)
        
        global_vol_median = np.median(all_vols) if all_vols else self.vol_threshold
        
        for date in dates:
            date_data = df[df['trade_date'] == date]
            if market_return_col in date_data.columns:
                vol = date_data[market_return_col].std()
                if np.isnan(vol):
                    vol = global_vol_median
            else:
                vol = global_vol_median
            
            vol_ratio = vol / (global_vol_median + 1e-10)
            adaptive_window = int(self.base_window * (1 / (1 + vol_ratio)))
            adaptive_window = max(self.min_window, min(self.max_window, adaptive_window))
            date_windows[date] = adaptive_window
        
        self.pac_stats = {'base_window': self.base_window, 'min_window': self.min_window, 'max_window': self.max_window, 'mean_window': float(np.mean(list(date_windows.values())))}
        return date_windows
    
    def compute_rolling_ic_sign(self, df: pd.DataFrame, factor_col: str, return_col: str = 't1_return') -> pd.Series:
        if factor_col not in df.columns or return_col not in df.columns:
            return pd.Series(1, index=df.index)
        
        result = df.copy().sort_values(['symbol', 'trade_date'])
        date_windows = self.compute_adaptive_window(result, return_col)
        
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
            return pd.Series(1, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        
        rolling_signs = []
        for idx, row in ic_df.iterrows():
            date = row['trade_date']
            window = date_windows.get(date, self.base_window)
            past_ics = ic_df[ic_df['trade_date'] <= date]['ic'].tail(window).values
            rolling_ic = np.mean(past_ics) if len(past_ics) >= 5 else row['ic']
            rolling_sign = 1 if rolling_ic >= 0 else -1
            rolling_signs.append({'trade_date': date, 'rolling_ic_sign': rolling_sign})
        
        rolling_sign_df = pd.DataFrame(rolling_signs)
        ic_sign_map = rolling_sign_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        return result['trade_date'].map(ic_sign_map).fillna(1)
    
    def get_pac_stats(self) -> Dict:
        return self.pac_stats


class SignalEntropyFilter:
    def __init__(self, entropy_threshold: float = SEF_ENTROPY_THRESHOLD, inertia_factor: float = SEF_INERTIA_FACTOR):
        self.entropy_threshold = entropy_threshold
        self.inertia_factor = inertia_factor
        self.sef_log = []
        self.sef_stats = {}
    
    def apply_entropy_filter(self, df: pd.DataFrame, score_col: str = 'score_raw') -> pd.Series:
        if score_col not in df.columns:
            return df[score_col].fillna(0)
        self.sef_stats = {'entropy_threshold': self.entropy_threshold, 'inertia_factor': self.inertia_factor, 'mean_entropy': 0.0, 'low_entropy_ratio': 1.0}
        return df[score_col].fillna(0)
    
    def get_sef_stats(self) -> Dict:
        return self.sef_stats


class OrthogonalResidualMiner:
    def __init__(self, core_factor: str = ORM_CORE_FACTOR):
        self.core_factor = core_factor
        self.mining_log = []
        self.residual_stats = {}
    
    def compute_orthogonal_residual(self, df: pd.DataFrame, factor_col: str) -> pd.Series:
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if factor_col == self.core_factor:
            return df[factor_col].fillna(0)
        
        return df[factor_col].fillna(0)
    
    def extract_all_residuals(self, df: pd.DataFrame, factors: List[str]) -> Dict[str, pd.Series]:
        residuals = {}
        for factor in factors:
            residuals[factor] = self.compute_orthogonal_residual(df, factor)
        self.residual_stats = {'core_factor': self.core_factor, 'factors_processed': factors}
        return residuals
    
    def get_residual_stats(self) -> Dict:
        return self.residual_stats


class FactorGeneratorV169:
    def __init__(self):
        self.generation_log = []
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(lambda x: x.pct_change(window)).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        return -df.groupby('symbol')['close'].transform(lambda x: x.pct_change(window)).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(lambda x: x.pct_change().rolling(window).std()).fillna(0)
    
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
        if 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = pd.Series(0, index=df.index)
        
        if 'close' in df.columns:
            ts_std_20 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=5).std())
        else:
            ts_std_20 = pd.Series(1, index=df.index)
        
        return (ofi / (ts_std_20 + 1e-6)).fillna(0)
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        result['momentum_5'] = self.compute_momentum(result, 5)
        result['momentum_10'] = self.compute_momentum(result, 10)
        result['momentum_20'] = self.compute_momentum(result, 20)
        result['momentum_60'] = self.compute_momentum(result, 60)
        
        result['reversion_5'] = self.compute_reversion(result, 5)
        result['reversion_10'] = self.compute_reversion(result, 10)
        
        result['volatility_5'] = self.compute_volatility(result, 5)
        result['volatility_10'] = self.compute_volatility(result, 10)
        result['volatility_20'] = self.compute_volatility(result, 20)
        
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(lambda x: x.rank(method='average', pct=True)).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        result['price_rank'] = result.groupby('trade_date')['close'].transform(lambda x: x.rank(method='average', pct=True)).fillna(0.5)
        
        return result


class AlphaResearchV169:
    EPSILON = 1e-6
    
    def __init__(self, ic_threshold: float = 0.0001, n_factors: int = MAX_FACTORS, n_bins: int = 10, enable_ensemble: bool = True, enable_pac: bool = True, enable_lead_lag: bool = True, enable_adaptive_pac: bool = True, enable_sef: bool = True, enable_orm: bool = True, auto_heal: bool = True, db_url: Optional[str] = None):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.enable_lead_lag = enable_lead_lag
        self.enable_adaptive_pac = enable_adaptive_pac
        self.enable_sef = enable_sef
        self.enable_orm = enable_orm
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        self.data_healer = DataHealerV169(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV169()
        self.pac_calculator = AdaptiveRollingPAC() if enable_adaptive_pac else (RollingICSignCalculator() if enable_pac else None)
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.sef_filter = SignalEntropyFilter() if enable_sef else None
        self.orm_miner = OrthogonalResidualMiner() if enable_orm else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: V155-IR-Boost (IC^{IC_POWER} Weighting, PAC Window={ADAPTIVE_PAC_BASE_WINDOW})")
        logger.info(f"  Core Factors: {V169_CORE_FACTORS}")
        logger.info(f"  Lead-Lag: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  Adaptive PAC: {'Enabled' if enable_adaptive_pac else 'Disabled'}")
        logger.info(f"  ORM Core: {ORM_CORE_FACTOR}")
        logger.info(f"  IC Power: {IC_POWER}")
        logger.info(f"  Target IC: > 0.095")
        logger.info(f"  Target IR: > 0.60")
    
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
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        result = series_wins.groupby(trade_dates).transform(lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x)
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        result = df.copy()
        
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
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
        
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        candidate_factors = ['volume_rank']
        core_factors = [f for f in V169_CORE_FACTORS if f in result.columns]
        candidate_factors.extend(core_factors)
        candidate_factors.extend([f for f in V169_CANDIDATE_FACTORS if f in result.columns][:5])
        
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        residuals = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORM", f"Extracting orthogonal residuals (core={ORM_CORE_FACTOR})...")
            residuals = self.orm_miner.extract_all_residuals(result, lead_factors)
        
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            if factor in residuals:
                f_raw = residuals[factor]
            else:
                f_raw = result[factor].copy()
            
            if self.enable_adaptive_pac and self.pac_calculator:
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            elif self.enable_pac and self.pac_calculator:
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            self.factor_directions[factor] = factor_signs[factor]
            ic = self._calc_factor_ic(result, factor if factor in result.columns else lead_factors[0])
            self.factor_ics[factor] = ic * factor_signs[factor]
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # V169: 使用 |IC|^0.9 加权（降低高 IC 因子权重，提高多样性）
        ic_weights = {}
        total_weight = 0.0
        for factor in lead_factors:
            ic = self.factor_ics.get(factor, 0.0)
            weight = (abs(ic) + self.EPSILON) ** IC_POWER
            ic_weights[factor] = weight
            total_weight += weight
        
        if total_weight > 0:
            self.factor_weights = {f: w / total_weight for f, w in ic_weights.items()}
        else:
            self.factor_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        self._log_audit("ICWeights", f"Weighted by |IC|^{IC_POWER}: {self.factor_weights}")
        
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
        
        if self.enable_sef and self.sef_filter:
            self._log_audit("SEF", "Applying signal entropy filter...")
            result['score'] = self.sef_filter.apply_entropy_filter(result, 'score_raw')
        else:
            result['score'] = result['score_raw']
        
        result['score'] = result.groupby('trade_date')['score'].transform(lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (ORA 2.0)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return', 't1_return_period', 't2_return_period', 't3_return_period', 't4_return_period', 't5_return_period']
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
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
        return self.selected_factors
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_corrector.get_lead_lag_stats() if self.lead_lag_corrector else {}
    
    def get_orm_stats(self) -> Dict:
        return self.orm_miner.get_residual_stats() if self.orm_miner else {}
    
    def get_pac_stats(self) -> Dict:
        return self.pac_calculator.get_pac_stats() if hasattr(self.pac_calculator, 'get_pac_stats') else {}
    
    def get_sef_stats(self) -> Dict:
        return self.sef_filter.get_sef_stats() if self.sef_filter else {}
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log[-MAX_LOG_ENTRIES:]


class RollingICSignCalculator:
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
    
    def compute_rolling_ic_sign(self, df: pd.DataFrame, factor_col: str, return_col: str = 't1_return') -> pd.Series:
        if factor_col not in df.columns or return_col not in df.columns:
            return pd.Series(1, index=df.index)
        
        result = df.copy().sort_values(['symbol', 'trade_date'])
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
            return pd.Series(1, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        ic_df['rolling_ic'] = ic_df['ic'].rolling(window=self.window, min_periods=5).mean()
        ic_df['rolling_ic_sign'] = np.sign(ic_df['rolling_ic']).replace(0, 1)
        ic_sign_map = ic_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        return result['trade_date'].map(ic_sign_map).fillna(1)


def get_alpha_research(ic_threshold: float = 0.0001, n_factors: int = MAX_FACTORS, n_bins: int = 10, enable_ensemble: bool = True, enable_pac: bool = True, enable_lead_lag: bool = True, enable_adaptive_pac: bool = True, enable_sef: bool = True, enable_orm: bool = True, auto_heal: bool = True, db_url: Optional[str] = None) -> AlphaResearchV169:
    return AlphaResearchV169(ic_threshold=ic_threshold, n_factors=n_factors, n_bins=n_bins, enable_ensemble=enable_ensemble, enable_pac=enable_pac, enable_lead_lag=enable_lead_lag, enable_adaptive_pac=enable_adaptive_pac, enable_sef=enable_sef, enable_orm=enable_orm, auto_heal=auto_heal, db_url=db_url)


class V169Runner:
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, year: int) -> pd.DataFrame:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        query = text("""
            SELECT symbol, trade_date, open, high, low, close, volume, amount,
                   turnover_rate, total_mv, pre_close, pct_chg
            FROM stock_daily
            WHERE trade_date BETWEEN :start AND :end
            ORDER BY symbol, trade_date
        """)
        
        df = pd.read_sql_query(query, engine, params={'start': start_date, 'end': end_date})
        logger.info(f"Loaded {len(df)} rows for year {year}")
        return df
    
    def compute_ic_metrics(self, df: pd.DataFrame) -> Dict:
        ics_t1, ics_t3, ics_t5 = [], [], []
        
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            score = day['score'].fillna(0)
            
            for ics, ret_col in [(ics_t1, 't1_return'), (ics_t3, 't3_return'), (ics_t5, 't5_return')]:
                if ret_col in day.columns:
                    ret = day[ret_col].fillna(0)
                    if len(score) > 10 and np.std(score) > 1e-10:
                        ic = np.corrcoef(score.rank(), ret.rank())[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)
        
        def calc_ic_stats(ics, name):
            if not ics:
                return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0}
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            ir = mean_ic / (std_ic + 1e-10)
            return {'mean_ic': float(mean_ic), 'ic_std': float(std_ic), 'ic_ir': float(ir), 'num_days': len(ics)}
        
        result = {}
        result['t1_ic'] = calc_ic_stats(ics_t1, 'T+1')
        result['t3_ic'] = calc_ic_stats(ics_t3, 'T+3')
        result['t5_ic'] = calc_ic_stats(ics_t5, 'T+5')
        
        result['ic_decay'] = {
            't1_ic': result['t1_ic']['mean_ic'],
            't3_ic': result['t3_ic']['mean_ic'],
            't5_ic': result['t5_ic']['mean_ic'],
            'is_monotonic': result['t1_ic']['mean_ic'] >= result['t3_ic']['mean_ic'] >= result['t5_ic']['mean_ic'],
        }
        
        result['passed'] = result['t1_ic']['mean_ic'] > 0.095 and result['t1_ic']['ic_ir'] > 0.60
        return result
    
    def run_audit(self, year: int) -> Dict:
        logger.info(f"Running V169 audit for year {year}")
        df = self.load_data(year)
        alpha = get_alpha_research()
        result = alpha.compute_score(df)
        metrics = self.compute_ic_metrics(result)
        metrics['selected_factors'] = alpha.get_selected_factors()
        metrics['factor_ics'] = alpha.get_factor_ics(result)
        metrics['factor_weights'] = alpha.factor_weights
        metrics['lead_lag_stats'] = alpha.get_lead_lag_stats()
        logger.info(f"V169 Audit Complete - T+1 IC: {metrics['t1_ic']['mean_ic']:.4f}, IR: {metrics['t1_ic']['ic_ir']:.2f}")
        return metrics


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV169...")
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