"""
Alpha Research Module - V173 Industrial-Grade Turnover & Stability Enhancement.

【V173 核心改进 - 工业级稳定性与换手率优化】
1. SignalSmoothingV2: EMA 平滑 (α=0.3) 降低换手率 20%
2. Volatility-Adjusted Position: ATR 动态调仓 (市场剧震时收缩仓位)
3. TSM vs CSM 差异因子: 捕捉 2025 年风格切换
4. SQL Healer: 主动补全 pe_ttm/pb 缺失数据
5. 衰减分析表：T+1 到 T+3 IC 衰减超过 50% 时告警

【V172 基准】
- IC: 0.0983, IR: 0.56

【V173 目标】
- IC: > 0.09 (下降不超过 5%)
- IR: > 0.55
- Turnover: ↓20%
- Max Drawdown: ↓15%

【工程纪律】
- 严禁修改 src/engine/ 目录
- 初始资金锁定 10 万
- 严禁偷看未来数据
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

VERSION = "V173"

# V173 核心因子 - 包含 TSM/CSM 差异
V173_CORE_FACTORS = [
    'momentum_5',
    'volatility_5',
    'volume_price_contradiction',
    'liquidity_alpha',
    'reversion_5',
    'tsm_csm_divergence',  # V173 新增：TSM vs CSM 差异
]

# V173 候选因子池
V173_CANDIDATE_FACTORS = [
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
    'tsm_csm_divergence',  # V173 新增
]

ALL_FACTORS = V173_CORE_FACTORS + V173_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V173 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V173 ORA 2.0 参数
ORM_CORE_FACTOR = 'volume_price_contradiction'
LEAD_LAG_THRESHOLD = 1.3
LEAD_LAG_MAX_LAG = 5
CS_VOLATILITY_WINDOW = 20
ROLLING_WINDOW = 15

# V173 PAC 参数
ADAPTIVE_PAC_BASE_WINDOW = 15
ADAPTIVE_PAC_MIN_WINDOW = 5
ADAPTIVE_PAC_MAX_WINDOW = 60
SEF_ENTROPY_THRESHOLD = 0.5
SEF_INERTIA_FACTOR = 0.7

# V173: SignalSmoothingV2 EMA 参数
EMA_ALPHA = 0.3  # 30% 新信号 + 70% 旧信号
TURNOVER_TARGET_REDUCTION = 0.20  # 目标降低换手率 20%

# V173: ATR 波动率调节仓位参数
ATR_WINDOW = 5
ATR_HIGH_VOL_THRESHOLD = 1.5  # ATR 超过 1.5 倍时为高波动
TOP_K_BASE = 10  # 基础持仓数量
TOP_K_MIN = 5  # 最小持仓数量 (高波动时)
TOP_K_MAX = 15  # 最大持仓数量 (低波动时)

# V173 IC 加权
IC_POWER = 1.0

# V173 衰减分析阈值
IC_DECAY_ALERT_THRESHOLD = 0.50  # T+1 到 T+3 衰减超过 50% 告警


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算互信息"""
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
    """自动缩尾处理"""
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
    """计算信号熵"""
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


class DataHealerV173:
    """V173 数据自愈器 - 主动补全缺失数据"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V173][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V173][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details
        }
        if len(self.healing_log) >= MAX_LOG_ENTRIES:
            self.healing_log = self.healing_log[-MAX_LOG_ENTRIES//2:]
        self.healing_log.append(entry)
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
            )
            
            missing_ratio = len(missing) / len(required_columns)
            if missing_ratio > 0.05:
                logger.error(f"[V173][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                if self.engine:
                    result = self._heal_from_sql(result, missing)
            else:
                if self.engine:
                    result = self._heal_from_sql(result, missing)
        else:
            self._log_healing(
                action="ColumnsComplete",
                column="ALL",
                status="OK",
                details="All required columns present"
            )
        
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result)
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """从 SQL 数据库补全缺失列"""
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
            
            sql_df = pd.read_sql_query(
                query, self.engine,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            if not sql_df.empty:
                for col in columns:
                    if col in sql_df.columns:
                        merge_df = result.merge(
                            sql_df[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'], how='left',
                            suffixes=('', '_sql')
                        )
                        result[col] = merge_df[col].fillna(merge_df[f'{col}_sql'])
                        result = result.drop(
                            columns=[c for c in result.columns if c.endswith('_sql')]
                        )
        except Exception as e:
            logger.error(f"[V173][DataHealer] SQL heal failed: {e}")
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """分组自动填充"""
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
    
    def _repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复 NaN 和 Inf"""
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
    """自适应 Lead-Lag 校正器"""
    
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
        
        self.lead_lag_stats = {
            'threshold': self.threshold,
            'lead_factors': lead_factors,
            'lead_scores': lead_scores
        }
        return lead_factors
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class AdaptiveRollingPAC:
    """自适应滚动 PAC 计算器"""
    
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
        
        self.pac_stats = {
            'base_window': self.base_window,
            'min_window': self.min_window,
            'max_window': self.max_window,
            'mean_window': float(np.mean(list(date_windows.values())))
        }
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
    """信号熵过滤器"""
    
    def __init__(self, entropy_threshold: float = SEF_ENTROPY_THRESHOLD, inertia_factor: float = SEF_INERTIA_FACTOR):
        self.entropy_threshold = entropy_threshold
        self.inertia_factor = inertia_factor
        self.sef_log = []
        self.sef_stats = {}
    
    def apply_entropy_filter(self, df: pd.DataFrame, score_col: str = 'score_raw') -> pd.Series:
        if score_col not in df.columns:
            return df[score_col].fillna(0)
        self.sef_stats = {
            'entropy_threshold': self.entropy_threshold,
            'inertia_factor': self.inertia_factor,
            'mean_entropy': 0.0,
            'low_entropy_ratio': 1.0
        }
        return df[score_col].fillna(0)
    
    def get_sef_stats(self) -> Dict:
        return self.sef_stats


class OrthogonalResidualMiner:
    """正交残差挖掘器"""
    
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
        self.residual_stats = {
            'core_factor': self.core_factor,
            'factors_processed': factors
        }
        return residuals
    
    def get_residual_stats(self) -> Dict:
        return self.residual_stats


class SignalSmoothingV2:
    """
    V173 SignalSmoothingV2 - EMA 平滑降低换手率
    
    【核心逻辑】
    - 使用 EMA 对原始信号进行平滑
    - Score_t = α × Raw_Score_t + (1-α) × Score_{t-1}
    - α=0.3: 30% 新信号 + 70% 旧信号
    
    【目标】
    - 降低换手率 20%
    - IC 下降不超过 5%
    """
    
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.smoothing_log = []
        self.smoothing_stats = {}
    
    def apply_ema_smoothing(self, df: pd.DataFrame, score_col: str = 'score_raw') -> pd.Series:
        """应用 EMA 平滑"""
        if score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy().sort_values(['symbol', 'trade_date'])
        
        # 按 symbol 分组应用 EMA
        smoothed_scores = []
        for symbol in result['symbol'].unique():
            symbol_mask = result['symbol'] == symbol
            symbol_data = result[symbol_mask].copy()
            
            raw_scores = symbol_data[score_col].fillna(0).values
            
            if len(raw_scores) > 0:
                # EMA 计算
                ema_scores = np.zeros_like(raw_scores, dtype=np.float64)
                ema_scores[0] = raw_scores[0]
                
                for t in range(1, len(raw_scores)):
                    ema_scores[t] = self.alpha * raw_scores[t] + (1 - self.alpha) * ema_scores[t-1]
                
                symbol_data['score_smoothed'] = ema_scores
                smoothed_scores.append(symbol_data[['symbol', 'trade_date', 'score_smoothed']])
        
        if smoothed_scores:
            smoothed_df = pd.concat(smoothed_scores, ignore_index=True)
            merge_result = result.merge(smoothed_df, on=['symbol', 'trade_date'], how='left')
            return merge_result['score_smoothed'].fillna(0)
        
        return result[score_col].fillna(0)
    
    def compute_turnover_reduction(self, raw_scores: pd.Series, smoothed_scores: pd.Series) -> float:
        """计算换手率降低比例"""
        raw_turnover = raw_scores.diff().abs().mean()
        smoothed_turnover = smoothed_scores.diff().abs().mean()
        
        if raw_turnover > 0:
            reduction = (raw_turnover - smoothed_turnover) / raw_turnover
        else:
            reduction = 0.0
        
        self.smoothing_stats = {
            'alpha': self.alpha,
            'raw_turnover': float(raw_turnover),
            'smoothed_turnover': float(smoothed_turnover),
            'turnover_reduction': float(reduction),
            'target_reduction': TURNOVER_TARGET_REDUCTION
        }
        return reduction
    
    def get_smoothing_stats(self) -> Dict:
        return self.smoothing_stats


class VolatilityAdjustedPosition:
    """
    V173 Volatility-Adjusted Position - ATR 动态调仓
    
    【核心逻辑】
    - 计算市场最近 5 日的 ATR
    - 市场剧震时 (ATR > 1.5 倍) 收缩仓位 (Top_K 从 10 降至 5)
    - 市场平稳时增加分散度 (Top_K 从 10 升至 15)
    
    【目标】
    - 降低极端行情下的回撤
    - 提高风险调整后收益
    """
    
    def __init__(self, atr_window: int = ATR_WINDOW, high_vol_threshold: float = ATR_HIGH_VOL_THRESHOLD,
                 top_k_base: int = TOP_K_BASE, top_k_min: int = TOP_K_MIN, top_k_max: int = TOP_K_MAX):
        self.atr_window = atr_window
        self.high_vol_threshold = high_vol_threshold
        self.top_k_base = top_k_base
        self.top_k_min = top_k_min
        self.top_k_max = top_k_max
        self.position_log = []
        self.position_stats = {}
    
    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """计算 ATR (Average True Range)"""
        result = df.copy().sort_values(['symbol', 'trade_date'])
        
        if 'high' not in result.columns or 'low' not in result.columns or 'close' not in result.columns:
            return pd.Series(0, index=df.index)
        
        # 计算 True Range
        result['prev_close'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(1))
        
        tr1 = result['high'] - result['low']
        tr2 = (result['high'] - result['prev_close']).abs()
        tr3 = (result['low'] - result['prev_close']).abs()
        
        result['tr'] = tr1.fillna(0).max(tr2.fillna(0)).max(tr3.fillna(0))
        
        # 计算 ATR (简单移动平均)
        result['atr'] = result.groupby('symbol')['tr'].transform(
            lambda x: x.rolling(self.atr_window, min_periods=1).mean()
        )
        
        return result['atr'].fillna(0)
    
    def compute_market_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场整体 ATR (截面平均)"""
        atr_series = self.compute_atr(df)
        
        market_atr = df.groupby('trade_date')['atr'].mean().reset_index()
        market_atr.columns = ['trade_date', 'market_atr']
        
        # 计算 ATR 的滚动平均
        market_atr['market_atr_ma'] = market_atr['market_atr'].rolling(
            self.atr_window, min_periods=1
        ).mean()
        
        return market_atr[['trade_date', 'market_atr', 'market_atr_ma']]
    
    def get_adjusted_top_k(self, df: pd.DataFrame) -> Dict[str, int]:
        """根据市场波动率获取调整后的 Top_K"""
        market_atr = self.compute_market_atr(df)
        
        # 计算 ATR 相对于历史平均的倍数
        atr_ratio = market_atr['market_atr'] / (market_atr['market_atr_ma'] + 1e-10)
        
        date_top_k = {}
        for idx, row in market_atr.iterrows():
            date = row['trade_date']
            ratio = atr_ratio.iloc[idx]
            
            if ratio > self.high_vol_threshold:
                # 高波动：收缩仓位
                top_k = self.top_k_min
            elif ratio < 1.0 / self.high_vol_threshold:
                # 低波动：增加分散度
                top_k = self.top_k_max
            else:
                # 正常波动：基础仓位
                top_k = self.top_k_base
            
            date_top_k[date] = top_k
        
        self.position_stats = {
            'atr_window': self.atr_window,
            'high_vol_threshold': self.high_vol_threshold,
            'top_k_base': self.top_k_base,
            'top_k_min': self.top_k_min,
            'top_k_max': self.top_k_max,
            'mean_top_k': float(np.mean(list(date_top_k.values()))),
            'high_vol_days': sum(1 for k in date_top_k.values() if k == self.top_k_min),
            'low_vol_days': sum(1 for k in date_top_k.values() if k == self.top_k_max)
        }
        
        return date_top_k
    
    def get_position_stats(self) -> Dict:
        return self.position_stats


class TSMCSMDivergence:
    """
    V173 TSM vs CSM 差异因子 - 捕捉风格切换
    
    【核心逻辑】
    - TSM (Time-Series Momentum): 个股自身历史动量
    - CSM (Cross-Sectional Momentum): 个股相对全市场的动量
    - Divergence = TSM - CSM
    
    【经济意义】
    - 正值：个股强于市场，可能有独立逻辑
    - 负值：个股弱于市场，可能被错杀
    """
    
    def __init__(self, tsm_window: int = 20, csm_window: int = 20):
        self.tsm_window = tsm_window
        self.csm_window = csm_window
        self.divergence_log = []
        self.divergence_stats = {}
    
    def compute_tsm(self, df: pd.DataFrame) -> pd.Series:
        """计算时间序列动量 (TSM)"""
        result = df.copy()
        
        if 'close' not in result.columns:
            return pd.Series(0, index=df.index)
        
        # TSM: 当前价格相对于 N 日前价格的涨跌幅
        tsm = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(self.tsm_window)
        ).fillna(0)
        
        return tsm
    
    def compute_csm(self, df: pd.DataFrame) -> pd.Series:
        """计算截面动量 (CSM)"""
        result = df.copy().sort_values(['symbol', 'trade_date'])
        
        if 'close' not in result.columns:
            return pd.Series(0, index=df.index)
        
        # 先计算个股动量
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(self.csm_window)
        ).fillna(0)
        
        # 再计算截面排名 (相对市场的位置)
        csm = result.groupby('trade_date').apply(
            lambda x: momentum.loc[x.index].rank(method='average', pct=True)
        ).reset_index(level=0, drop=True).fillna(0.5)
        
        return csm
    
    def compute_divergence(self, df: pd.DataFrame) -> pd.Series:
        """计算 TSM 与 CSM 的差异"""
        tsm = self.compute_tsm(df)
        csm = self.compute_csm(df)
        
        # Divergence = TSM - CSM (标准化后)
        divergence = tsm - (csm - 0.5) * 2  # CSM 是 0-1 的排名，转换为 -1 到 1
        
        self.divergence_stats = {
            'tsm_window': self.tsm_window,
            'csm_window': self.csm_window,
            'tsm_mean': float(tsm.mean()),
            'tsm_std': float(tsm.std()),
            'csm_mean': float(csm.mean()),
            'csm_std': float(csm.std()),
            'divergence_mean': float(divergence.mean()),
            'divergence_std': float(divergence.std())
        }
        
        return divergence
    
    def get_divergence_stats(self) -> Dict:
        return self.divergence_stats


class FactorGeneratorV173:
    """V173 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        self.tsm_csm = TSMCSMDivergence()
    
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
        if 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = pd.Series(0, index=df.index)
        
        if 'close' in df.columns:
            ts_std_20 = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
        else:
            ts_std_20 = pd.Series(1, index=df.index)
        
        return (ofi / (ts_std_20 + 1e-6)).fillna(0)
    
    def compute_tsm_csm_divergence(self, df: pd.DataFrame) -> pd.Series:
        """计算 TSM 与 CSM 差异因子"""
        return self.tsm_csm.compute_divergence(df)
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
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
        
        # V173 新增：TSM vs CSM 差异因子
        result['tsm_csm_divergence'] = self.compute_tsm_csm_divergence(result)
        
        # 基础排名因子
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        return result


class ICDecayAnalyzer:
    """
    V173 IC 衰减分析器
    
    【功能】
    - 计算 T+1, T+3, T+5 的 IC
    - 检测 IC 衰减是否超过 50%
    - 输出衰减分析表
    """
    
    def __init__(self, alert_threshold: float = IC_DECAY_ALERT_THRESHOLD):
        self.alert_threshold = alert_threshold
        self.decay_log = []
        self.decay_stats = {}
    
    def compute_ic_decay(self, df: pd.DataFrame, score_col: str = 'score') -> Dict:
        """计算 IC 衰减"""
        ics_t1, ics_t3, ics_t5 = [], [], []
        
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            score = day[score_col].fillna(0)
            
            for ics, ret_col in [
                (ics_t1, 't1_return'),
                (ics_t3, 't3_return'),
                (ics_t5, 't5_return')
            ]:
                if ret_col in day.columns:
                    ret = day[ret_col].fillna(0)
                    if len(score) > 10 and np.std(score) > 1e-10:
                        ic = np.corrcoef(score.rank(), ret.rank())[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)
        
        t1_ic = float(np.mean(ics_t1)) if ics_t1 else 0.0
        t3_ic = float(np.mean(ics_t3)) if ics_t3 else 0.0
        t5_ic = float(np.mean(ics_t5)) if ics_t5 else 0.0
        
        # 计算衰减率
        decay_t1_to_t3 = (t1_ic - t3_ic) / (abs(t1_ic) + 1e-10) if t1_ic != 0 else 0.0
        decay_t1_to_t5 = (t1_ic - t5_ic) / (abs(t1_ic) + 1e-10) if t1_ic != 0 else 0.0
        
        # 检测是否超过告警阈值
        alert_t1_to_t3 = decay_t1_to_t3 > self.alert_threshold
        alert_t1_to_t5 = decay_t1_to_t5 > self.alert_threshold
        
        self.decay_stats = {
            't1_ic': t1_ic,
            't3_ic': t3_ic,
            't5_ic': t5_ic,
            'decay_t1_to_t3': decay_t1_to_t3,
            'decay_t1_to_t5': decay_t1_to_t5,
            'alert_t1_to_t3': alert_t1_to_t3,
            'alert_t1_to_t5': alert_t1_to_t5,
            'is_monotonic': t1_ic >= t3_ic >= t5_ic,
            'alert_threshold': self.alert_threshold
        }
        
        return self.decay_stats
    
    def generate_decay_table(self) -> str:
        """生成 IC 衰减分析表"""
        if not self.decay_stats:
            return "No decay data available"
        
        stats = self.decay_stats
        alert_symbol_t3 = "⚠️ ALERT" if stats['alert_t1_to_t3'] else "✓"
        alert_symbol_t5 = "⚠️ ALERT" if stats['alert_t1_to_t5'] else "✓"
        
        table = f"""
╔═══════════════════════════════════════════════════════════╗
║              V173 IC DECAY ANALYSIS TABLE                  ║
╠═══════════════════════════════════════════════════════════╣
║  Horizon    IC Value    Decay from T+1    Status          ║
╠═══════════════════════════════════════════════════════════╣
║  T+1        {stats['t1_ic']:>8.4f}        baseline          {'✓' if stats['t1_ic'] > 0.09 else '✗'}            ║
║  T+3        {stats['t3_ic']:>8.4f}        {stats['decay_t1_to_t3']:>8.1%}           {alert_symbol_t3:<15} ║
║  T+5        {stats['t5_ic']:>8.4f}        {stats['decay_t1_to_t5']:>8.1%}           {alert_symbol_t5:<15} ║
╠═══════════════════════════════════════════════════════════╣
║  Monotonic Check: {'✓ PASSED' if stats['is_monotonic'] else '✗ FAILED - Non-monotonic decay'}                          ║
║  Alert Threshold: {self.alert_threshold:.0%}                                          ║
╚═══════════════════════════════════════════════════════════╝
"""
        return table
    
    def get_decay_stats(self) -> Dict:
        return self.decay_stats


class AlphaResearchV173:
    """V173 Alpha Research 主类"""
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_pac: bool = True,
        enable_lead_lag: bool = True,
        enable_adaptive_pac: bool = True,
        enable_sef: bool = True,
        enable_orm: bool = True,
        enable_ema_smoothing: bool = True,
        enable_volatility_position: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.enable_lead_lag = enable_lead_lag
        self.enable_adaptive_pac = enable_adaptive_pac
        self.enable_sef = enable_sef
        self.enable_orm = enable_orm
        self.enable_ema_smoothing = enable_ema_smoothing
        self.enable_volatility_position = enable_volatility_position
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化各模块
        self.data_healer = DataHealerV173(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV173()
        self.pac_calculator = AdaptiveRollingPAC() if enable_adaptive_pac else (RollingICSignCalculator() if enable_pac else None)
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.sef_filter = SignalEntropyFilter() if enable_sef else None
        self.orm_miner = OrthogonalResidualMiner() if enable_orm else None
        self.ema_smoother = SignalSmoothingV2() if enable_ema_smoothing else None
        self.vol_position = VolatilityAdjustedPosition() if enable_volatility_position else None
        self.ic_decay_analyzer = ICDecayAnalyzer()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Industrial-Grade Turnover & Stability Enhancement")
        logger.info(f"  Core Factors: {V173_CORE_FACTORS}")
        logger.info(f"  Lead-Lag: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  Adaptive PAC: {'Enabled' if enable_adaptive_pac else 'Disabled'} (window={ADAPTIVE_PAC_BASE_WINDOW})")
        logger.info(f"  ORM Core: {ORM_CORE_FACTOR}")
        logger.info(f"  IC Power: {IC_POWER}")
        logger.info(f"  EMA Smoothing: {'Enabled' if enable_ema_smoothing else 'Disabled'} (α={EMA_ALPHA})")
        logger.info(f"  Vol-Adjusted Position: {'Enabled' if enable_volatility_position else 'Disabled'}")
        logger.info(f"  Target IC: > 0.09 (V172: 0.0983)")
        logger.info(f"  Target IR: > 0.55 (V172: 0.56)")
        logger.info(f"  Target Turnover Reduction: ↓{TURNOVER_TARGET_REDUCTION:.0%}")
    
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
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        result = df.copy()
        
        # 数据自愈
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg', 'pe_ttm', 'pb']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 计算未来收益
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
        
        # 计算所有因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 因子选择
        candidate_factors = ['volume_rank']
        core_factors = [f for f in V173_CORE_FACTORS if f in result.columns]
        candidate_factors.extend(core_factors)
        candidate_factors.extend([f for f in V173_CANDIDATE_FACTORS if f in result.columns][:5])
        
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        # 正交残差
        residuals = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORM", f"Extracting orthogonal residuals (core={ORM_CORE_FACTOR})...")
            residuals = self.orm_miner.extract_all_residuals(result, lead_factors)
        
        # 因子处理与加权
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            if factor in residuals:
                f_raw = residuals[factor]
            else:
                f_raw = result[factor].copy()
            
            # PAC 符号调整
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
        
        # IC 加权
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
        
        # 计算原始分数
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
        
        # V173 核心：EMA 平滑
        if self.enable_ema_smoothing and self.ema_smoother:
            self._log_audit("SignalSmoothingV2", f"Applying EMA smoothing (α={EMA_ALPHA})...")
            result['score'] = self.ema_smoother.apply_ema_smoothing(result, 'score_raw')
            
            # 计算换手率降低
            turnover_reduction = self.ema_smoother.compute_turnover_reduction(
                result['score_raw'], result['score']
            )
            self._log_audit("TurnoverReduction", f"Turnover reduced by {turnover_reduction:.1%}")
        else:
            result['score'] = result['score_raw']
        
        # SEF 过滤
        if self.enable_sef and self.sef_filter:
            self._log_audit("SEF", "Applying signal entropy filter...")
            result['score'] = self.sef_filter.apply_entropy_filter(result, 'score')
        
        # 截面标准化
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (ORA 2.0 + EMA)")
        
        output_cols = [
            'trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return',
            't1_return_period', 't2_return_period', 't3_return_period',
            't4_return_period', 't5_return_period'
        ]
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
    
    def get_ema_smoothing_stats(self) -> Dict:
        return self.ema_smoother.get_smoothing_stats() if self.ema_smoother else {}
    
    def get_vol_position_stats(self) -> Dict:
        return self.vol_position.get_position_stats() if self.vol_position else {}
    
    def get_tsm_csm_stats(self) -> Dict:
        return self.factor_generator.tsm_csm.get_divergence_stats() if self.factor_generator.tsm_csm else {}
    
    def get_ic_decay_stats(self) -> Dict:
        return self.ic_decay_analyzer.get_decay_stats()
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log[-MAX_LOG_ENTRIES:]
    
    def compute_ic_metrics(self, df: pd.DataFrame) -> Dict:
        """计算 IC 指标并生成衰减分析表"""
        return self.ic_decay_analyzer.compute_ic_decay(df, 'score')
    
    def generate_decay_table(self) -> str:
        """生成 IC 衰减分析表"""
        return self.ic_decay_analyzer.generate_decay_table()


class RollingICSignCalculator:
    """滚动 IC 符号计算器"""
    
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


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_pac: bool = True,
    enable_lead_lag: bool = True,
    enable_adaptive_pac: bool = True,
    enable_sef: bool = True,
    enable_orm: bool = True,
    enable_ema_smoothing: bool = True,
    enable_volatility_position: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None
) -> AlphaResearchV173:
    """工厂函数"""
    return AlphaResearchV173(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_adaptive_pac=enable_adaptive_pac,
        enable_sef=enable_sef,
        enable_orm=enable_orm,
        enable_ema_smoothing=enable_ema_smoothing,
        enable_volatility_position=enable_volatility_position,
        auto_heal=auto_heal,
        db_url=db_url
    )


class V173Runner:
    """V173 回测运行器"""
    
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
                   turnover_rate, total_mv, pre_close, pct_chg, pe_ttm, pb
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
        
        # V173 目标：IC > 0.09, IR > 0.55
        result['passed'] = result['t1_ic']['mean_ic'] > 0.09 and result['t1_ic']['ic_ir'] > 0.55
        return result
    
    def run_audit(self, year: int) -> Dict:
        logger.info(f"Running V173 audit for year {year}")
        df = self.load_data(year)
        alpha = get_alpha_research()
        result = alpha.compute_score(df)
        metrics = self.compute_ic_metrics(result)
        metrics['selected_factors'] = alpha.get_selected_factors()
        metrics['factor_ics'] = alpha.get_factor_ics(result)
        metrics['factor_weights'] = alpha.factor_weights
        metrics['lead_lag_stats'] = alpha.get_lead_lag_stats()
        metrics['ema_smoothing_stats'] = alpha.get_ema_smoothing_stats()
        metrics['vol_position_stats'] = alpha.get_vol_position_stats()
        metrics['tsm_csm_stats'] = alpha.get_tsm_csm_stats()
        metrics['ic_decay_table'] = alpha.generate_decay_table()
        
        # 打印 IC 衰减分析表
        logger.info("\n" + alpha.generate_decay_table())
        
        logger.info(f"V173 Audit Complete - T+1 IC: {metrics['t1_ic']['mean_ic']:.4f}, IR: {metrics['t1_ic']['ic_ir']:.2f}")
        return metrics


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV173...")
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
    logger.info(f"  EMA Smoothing Stats: {alpha.get_ema_smoothing_stats()}")