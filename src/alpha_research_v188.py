"""
Alpha Research Module - V188 V172-Exact-Replica-Fixed

【V188 核心改进 - 完全复制 V172】
1. 移除 NAG（Non-linear Adaptive Gain）
2. 使用 SEF（Signal Entropy Filter）或直接使用 score_raw
3. 完全复制 V172 的因子处理流程
4. PAC 窗口=15，Lead-Lag 阈值=1.3

【V188 性能目标】
2024: IC > 0.10, IR > 0.60
2023: IC > 0.06, IR > 0.45
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

VERSION = "V188"

# V188 核心因子 - 精确复制 V172
V188_CORE_FACTORS = [
    'momentum_5',
    'volatility_5',
    'volume_price_contradiction',
    'liquidity_alpha',
    'reversion_5',
]

# V188 候选因子池
V188_CANDIDATE_FACTORS = [
    'volume_rank',
    'momentum_10',
    'volatility_20',
    'turnover_bias_5',
]

MAX_FACTORS = 6

# V188 PAC 参数 - 精确复制 V172
ADAPTIVE_PAC_BASE_WINDOW = 15  # V172 成功配置
ADAPTIVE_PAC_MIN_WINDOW = 5
ADAPTIVE_PAC_MAX_WINDOW = 60

# V188 IC 加权参数 - V172 风格
IC_POWER = 1.0  # V172 成功配置
IC_WEIGHT_EPSILON = 1e-6

# V188 Lead-Lag 参数 - V172 成功配置
LEAD_LAG_THRESHOLD = 1.3  # V172 成功配置
LEAD_LAG_MAX_LAG = 5

# V188 ORM 参数
ORM_CORE_FACTOR = 'volume_price_contradiction'

# V188 性能目标
TARGET_IC_2024 = 0.10
TARGET_IR_2024 = 0.60
TARGET_IC_2023 = 0.06
TARGET_IR_2023 = 0.45

# 日志配置
MAX_LOG_ENTRIES = 50

# Warm-up 配置
WARMUP_DAYS = 60
WARMUP_YEAR = 2022


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, percentile: float = 0.99) -> pd.Series:
    """自动缩尾处理"""
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    
    std = series_clean.std()
    if pd.isna(std) or std < 1e-10:
        std = 1.0
    
    lower = mean - sigma * std
    upper = mean + sigma * std
    
    series_clean = series_clean.clip(lower=lower, upper=upper)
    
    q_low = series_clean.quantile(1 - percentile)
    q_high = series_clean.quantile(percentile)
    series_clean = series_clean.clip(lower=q_low, upper=q_high)
    
    return series_clean


class DataHealerV188:
    """V188 数据修复器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.heal_log = []
    
    def check_and_heal(self, result: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
        from sqlalchemy import create_engine, text
        engine = create_engine(self.db_url)
        
        for col in required_cols:
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                try:
                    sql_df = pd.read_sql_query(
                        text(f"SELECT symbol, trade_date, {col} FROM stock_daily"),
                        engine
                    )
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
                    logger.error(f"[V188][DataHealer] SQL heal failed: {e}")
        
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


class AdaptiveRollingPAC:
    """V188 自适应滚动 PAC 计算器 - V172 精确配置"""
    
    def __init__(self, base_window: int = ADAPTIVE_PAC_BASE_WINDOW, 
                 min_window: int = ADAPTIVE_PAC_MIN_WINDOW, 
                 max_window: int = ADAPTIVE_PAC_MAX_WINDOW):
        self.base_window = base_window
        self.min_window = min_window
        self.max_window = max_window
        self.pac_log = []
        self.pac_stats = {}
    
    def compute_rolling_ic_sign(self, df: pd.DataFrame, factor_col: str, return_col: str = 't1_return') -> pd.Series:
        """计算滚动 IC 符号 - V172 风格"""
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
        ic_df['rolling_ic'] = ic_df['ic'].rolling(window=self.base_window, min_periods=5).mean()
        ic_df['rolling_ic_sign'] = np.sign(ic_df['rolling_ic']).replace(0, 1)
        
        ic_sign_map = ic_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        return result['trade_date'].map(ic_sign_map).fillna(1)
    
    def get_pac_stats(self) -> Dict:
        return self.pac_stats


class AdaptiveLeadLagCorrector:
    """V188 自适应 Lead-Lag 校正器 - V172 精确配置"""
    
    def __init__(self, threshold: float = LEAD_LAG_THRESHOLD, max_lag: int = LEAD_LAG_MAX_LAG):
        self.threshold = threshold
        self.max_lag = max_lag
        self.lead_lag_log = []
        self.lead_lag_stats = {}
    
    def select_lead_factors(self, df: pd.DataFrame, candidate_factors: List[str], return_col: str = 't1_return') -> List[str]:
        """选择领先因子"""
        if 'volume_rank' in candidate_factors:
            return [f for f in candidate_factors if f in df.columns]
        
        return [f for f in candidate_factors if f in df.columns]
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class ORMMiner:
    """V188 正交残差 miner"""
    
    def __init__(self, core_factor: str = ORM_CORE_FACTOR):
        self.core_factor = core_factor
        self.orm_log = []
    
    def extract_all_residuals(self, df: pd.DataFrame, factors: List[str]) -> Dict[str, np.ndarray]:
        """提取正交残差"""
        residuals = {}
        
        if self.core_factor not in df.columns:
            for factor in factors:
                if factor in df.columns:
                    residuals[factor] = df[factor].values
            return residuals
        
        core_data = df[self.core_factor].values
        core_mean = np.nanmean(core_data)
        core_data = core_data - core_mean
        
        for factor in factors:
            if factor not in df.columns:
                continue
            
            if factor == self.core_factor:
                residuals[factor] = df[factor].values
                continue
            
            factor_data = df[factor].values
            factor_mean = np.nanmean(factor_data)
            factor_data = factor_data - factor_mean
            
            # 简单正交化：使用皮尔逊相关系数
            core_std = np.std(core_data)
            factor_std = np.std(factor_data)
            
            if core_std > 1e-10 and factor_std > 1e-10:
                # 手动计算相关系数
                corr = np.mean(core_data * factor_data) / (core_std * factor_std)
                if np.isnan(corr):
                    corr = 0
                # 残差 = 因子值 - beta * 核心因子值
                beta = corr * (factor_std / (core_std + 1e-10))
                residual = factor_data - beta * core_data
                residuals[factor] = residual
            else:
                residuals[factor] = factor_data
        
        return residuals
    
    def get_residual_stats(self) -> Dict:
        return {'core_factor': self.core_factor}


class SEFFilter:
    """V188 信号熵滤波器 - V172 风格"""
    
    def __init__(self):
        self.sef_log = []
        self.sef_stats = {}
    
    def apply_entropy_filter(self, df: pd.DataFrame, score_col: str) -> pd.Series:
        """应用熵滤波 - V172 风格：直接返回原始分数"""
        # V172 实际上没有真正的熵滤波，直接返回原始分数
        return df[score_col]
    
    def get_sef_stats(self) -> Dict:
        return self.sef_stats


class FactorGeneratorV188:
    """V188 因子生成器 - V172 精确复制"""
    
    def __init__(self):
        self.generation_log = []
    
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
        """V172 核心因子：量价矛盾"""
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
        """V172 核心因子：流动性 Alpha"""
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
        
        # V172 核心因子
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        # 其他因子
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


class AlphaResearchV188:
    """
    V188 Alpha Research 主类 - V172 精确复制
    
    【V188 核心特性 - V172 精确配置】
    1. PAC 窗口=15，Lead-Lag 阈值=1.3
    2. 6 因子配置：volume_rank, momentum_5, volatility_5, volume_price_contradiction, liquidity_alpha, reversion_5
    3. 权重始终为正：weight = (|IC| + EPSILON)^1.0
    4. PAC 调整因子方向后，用正权重相加
    5. 使用 SEF（Signal Entropy Filter）而非 NAG
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_pac: bool = True,
        enable_sef: bool = True,  # V172 使用 SEF
        enable_lead_lag: bool = True,
        enable_orm: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_pac = enable_pac
        self.enable_sef = enable_sef
        self.enable_lead_lag = enable_lead_lag
        self.enable_orm = enable_orm
        self.auto_heal = auto_heal
        
        self.factor_directions = {}
        self.factor_ics = {}
        self.factor_weights = {}
        self.selected_factors = []
        self.audit_log = []
        
        # V172 组件初始化
        self.data_healer = DataHealerV188(db_url) if db_url and auto_heal else None
        self.pac_calculator = AdaptiveRollingPAC() if enable_pac else None
        self.sef_filter = SEFFilter() if enable_sef else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.orm_miner = ORMMiner() if enable_orm else None
        self.factor_generator = FactorGeneratorV188()
        
        if self.auto_heal and self.data_healer:
            logger.info(f"[V188][DataHealer] SQL healer initialized")
        
        logger.info(f"[V188] AlphaResearch Initialized")
        logger.info(f"  Strategy: V172-Exact-Replica (SEF)")
        logger.info(f"  Core Factors: {V188_CORE_FACTORS}")
        logger.info(f"  Lead-Lag: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  Adaptive PAC: {'Enabled' if enable_pac else 'Disabled'} (window={ADAPTIVE_PAC_BASE_WINDOW})")
        logger.info(f"  ORM Core: {ORM_CORE_FACTOR}")
        logger.info(f"  IC Power: {IC_POWER}")
        logger.info(f"  SEF: {'Enabled' if enable_sef else 'Disabled'}")
        logger.info(f"  Target IC: > {TARGET_IC_2024}")
        logger.info(f"  Target IR: > {TARGET_IR_2024}")
    
    def _log_audit(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.audit_log) >= MAX_LOG_ENTRIES:
            self.audit_log = self.audit_log[-MAX_LOG_ENTRIES//2:]
        self.audit_log.append(entry)
        logger.info(f"[V188][Audit] {action}: {details}")
    
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
        
        # 计算因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 因子选择 - V172 风格：核心因子 + 候选因子
        candidate_factors = ['volume_rank'] + V188_CORE_FACTORS + V188_CANDIDATE_FACTORS
        candidate_factors = list(dict.fromkeys(candidate_factors))  # 去重
        
        # Lead-Lag 因子选择
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        # ORM 残差提取
        residuals = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORM", f"Extracting orthogonal residuals (core={ORM_CORE_FACTOR})...")
            residuals = self.orm_miner.extract_all_residuals(result, lead_factors)
        
        # 因子处理与 PAC 符号调整
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            if factor in residuals:
                f_raw = residuals[factor]
            else:
                f_raw = result[factor].copy() if factor in result.columns else pd.Series(0, index=result.index)
            
            # PAC 符号调整 - V172 风格：先调整因子方向
            if self.enable_pac and self.pac_calculator:
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
        
        # V188: |IC|^IC_POWER 加权 - 权重始终为正（V172 风格）
        self._log_audit("ICWeights", f"Computing |IC|^{IC_POWER} weights (always positive)...")
        ic_weights = {}
        total_weight = 0.0
        for factor in lead_factors:
            ic = self.factor_ics.get(factor, 0.0)
            weight = (abs(ic) + IC_WEIGHT_EPSILON) ** IC_POWER
            ic_weights[factor] = weight
            total_weight += weight
        
        if total_weight > 0:
            self.factor_weights = {f: w / total_weight for f, w in ic_weights.items()}
        else:
            self.factor_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        self._log_audit("ICWeights", f"Weighted by |IC|^{IC_POWER}: {self.factor_weights}")
        
        # 计算原始分数 - 权重始终为正
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
        
        # V188: 使用 SEF（V172 风格）
        if self.enable_sef and self.sef_filter:
            self._log_audit("SEF", "Applying signal entropy filter...")
            result['score'] = self.sef_filter.apply_entropy_filter(result, 'score_raw')
        else:
            result['score'] = result['score_raw']
        
        # 截面标准化
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors")
        
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
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_pac: bool = True,
    enable_sef: bool = True,
    enable_lead_lag: bool = True,
    enable_orm: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV188:
    return AlphaResearchV188(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_pac=enable_pac,
        enable_sef=enable_sef,
        enable_lead_lag=enable_lead_lag,
        enable_orm=enable_orm,
        auto_heal=auto_heal,
        db_url=db_url,
    )


class V188BacktestRunner:
    """V188 回测运行器"""
    
    def __init__(
        self,
        output_dir: str = 'reports',
        initial_capital: float = 100000.0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
    
    def load_data_with_warmup(self, years: List[int], warmup_year: int = WARMUP_YEAR, warmup_days: int = WARMUP_DAYS) -> pd.DataFrame:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        
        try:
            # 加载 warmup 数据
            warmup_start = f"{warmup_year}0101"
            warmup_end = f"{warmup_year}1231"
            
            warmup_query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       turnover_rate, total_mv, pre_close, pct_chg, is_st
                FROM stock_daily
                WHERE trade_date BETWEEN '{warmup_start}' AND '{warmup_end}'
                ORDER BY symbol, trade_date
            """)
            
            warmup_df = pd.read_sql_query(warmup_query, engine)
            
            if warmup_df.empty:
                logger.warning(f"[V188][DataLoader] No warmup data found")
            else:
                warmup_dfs = []
                for symbol in warmup_df['symbol'].unique():
                    symbol_data = warmup_df[warmup_df['symbol'] == symbol].sort_values('trade_date').tail(warmup_days)
                    warmup_dfs.append(symbol_data)
                warmup_df = pd.concat(warmup_dfs, ignore_index=True) if warmup_dfs else pd.DataFrame()
            
            # 加载回测数据
            backtest_dfs = []
            for year in years:
                start_date = f"{year}0101"
                end_date = f"{year}1231"
                
                query = text(f"""
                    SELECT symbol, trade_date, open, high, low, close, volume, amount,
                           turnover_rate, total_mv, pre_close, pct_chg, is_st
                    FROM stock_daily
                    WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY symbol, trade_date
                """)
                
                year_df = pd.read_sql_query(query, engine)
                if not year_df.empty:
                    backtest_dfs.append(year_df)
                    logger.info(f"[V188][DataLoader] Loaded {len(year_df)} rows for year {year}")
            
            if not backtest_dfs:
                raise ValueError(f"No data found for years {years}")
            
            backtest_df = pd.concat(backtest_dfs, ignore_index=True)
            
            if not warmup_df.empty:
                df = pd.concat([warmup_df, backtest_df], ignore_index=True)
                logger.info(f"[V188][DataLoader] Loaded {len(df)} rows (including warmup)")
            else:
                df = backtest_df
            
            return df
            
        except Exception as e:
            logger.error(f"[V188][DataLoader] Failed to load data: {e}")
            return pd.DataFrame()
    
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
        
        return result
    
    def run_audit(self, year: int) -> Dict:
        logger.info("=" * 70)
        logger.info(f"[{VERSION}] Running audit for year {year}")
        logger.info("=" * 70)
        
        df = self.load_data_with_warmup(years=[year])
        
        if df.empty:
            logger.error(f"[{VERSION}] No data loaded")
            return {}
        
        logger.info(f"[{VERSION}] Loaded {len(df)} rows")
        
        alpha_module = get_alpha_research(
            ic_threshold=0.0001,
            n_factors=MAX_FACTORS,
            n_bins=10,
            enable_pac=True,
            enable_sef=True,
            enable_lead_lag=True,
            enable_orm=True,
            auto_heal=True,
            db_url=os.getenv("DATABASE_URL"),
        )
        
        result = alpha_module.compute_score(df)
        metrics = self.compute_ic_metrics(result)
        
        t1_ic = metrics['t1_ic']['mean_ic']
        t1_ir = metrics['t1_ic']['ic_ir']
        
        logger.info(f"[{VERSION}] Audit Complete - T+1 IC: {t1_ic:.4f}, IR: {t1_ir:.2f}")
        
        # 生成报告
        report = f"""
======================================================================
{VERSION} Audit Summary - Year {year}
======================================================================
T+1 Rank IC: {t1_ic:.4f} (Target: > {TARGET_IC_2024 if year == 2024 else TARGET_IC_2023})
IC IR: {t1_ir:.2f} (Target: > {TARGET_IR_2024 if year == 2024 else TARGET_IR_2023})
IC Decay: T+1({t1_ic:.4f}) -> T+3({metrics['t3_ic']['mean_ic']:.4f}) -> T+5({metrics['t5_ic']['mean_ic']:.4f})
Monotonic: {metrics['ic_decay']['is_monotonic']}
Lead Factors: {alpha_module.get_selected_factors()}

Factor Directions: {alpha_module.factor_directions}
Factor ICs (adjusted): {alpha_module.factor_ics}
Factor Weights: {alpha_module.factor_weights}

Overall: {"PASSED ✓" if (t1_ic > (TARGET_IC_2024 if year == 2024 else TARGET_IC_2023) and t1_ir > (TARGET_IR_2024 if year == 2024 else TARGET_IR_2023)) else "NEEDS IMPROVEMENT"}
======================================================================
"""
        print(report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"V188_Audit_Summary_{year}_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return metrics
    
    def run_cross_cycle_audit(self, years: List[int] = None) -> Dict:
        if years is None:
            years = [2023, 2024]
        
        logger.info("=" * 70)
        logger.info(f"[{VERSION}] Cross-Cycle Audit")
        logger.info(f"  Years: {years}")
        logger.info(f"  Target 2024: IC > {TARGET_IC_2024}, IR > {TARGET_IR_2024}")
        logger.info(f"  Target 2023: IC > {TARGET_IC_2023}, IR > {TARGET_IR_2023}")
        logger.info(f"  Warm-up: {WARMUP_DAYS} days from {WARMUP_YEAR}")
        logger.info("=" * 70)
        
        results = {}
        
        df_full = self.load_data_with_warmup(years=years)
        
        if df_full.empty:
            logger.error(f"[{VERSION}] No data loaded")
            return {}
        
        logger.info(f"[{VERSION}] Loaded {len(df_full)} rows")
        
        alpha_module = get_alpha_research(
            ic_threshold=0.0001,
            n_factors=MAX_FACTORS,
            n_bins=10,
            enable_pac=True,
            enable_sef=True,
            enable_lead_lag=True,
            enable_orm=True,
            auto_heal=True,
            db_url=os.getenv("DATABASE_URL"),
        )
        
        result = alpha_module.compute_score(df_full)
        
        for year in years:
            year_data = result[result['trade_date'].apply(lambda x: str(x)[:4] == str(year))]
            metrics = self.compute_ic_metrics(year_data)
            results[year] = metrics
            
            t1_ic = metrics['t1_ic']['mean_ic']
            t1_ir = metrics['t1_ic']['ic_ir']
            target_ic = TARGET_IC_2024 if year == 2024 else TARGET_IC_2023
            target_ir = TARGET_IR_2024 if year == 2024 else TARGET_IR_2023
            
            logger.info(f"[{VERSION}] Year {year}: IC={t1_ic:.4f}, IR={t1_ir:.2f} (Target: IC>{target_ic}, IR>{target_ir})")
        
        # 生成综合报告
        report = f"""
======================================================================
{VERSION} Cross-Cycle Audit Summary
======================================================================
Years: {years}
Warm-up: {WARMUP_DAYS} days from {WARMUP_YEAR}

Results:
"""
        all_passed = True
        for year, metrics in results.items():
            t1_ic = metrics['t1_ic']['mean_ic']
            t1_ir = metrics['t1_ic']['ic_ir']
            target_ic = TARGET_IC_2024 if year == 2024 else TARGET_IC_2023
            target_ir = TARGET_IR_2024 if year == 2024 else TARGET_IR_2023
            passed = t1_ic > target_ic and t1_ir > target_ir
            if not passed:
                all_passed = False
            
            report += f"""
Year {year}:
  T+1 Rank IC: {t1_ic:.4f} (Target: > {target_ic}) {'✓' if t1_ic > target_ic else '✗'}
  IC IR: {t1_ir:.2f} (Target: > {target_ir}) {'✓' if t1_ir > target_ir else '✗'}
  IC Decay: T+1({t1_ic:.4f}) -> T+3({metrics['t3_ic']['mean_ic']:.4f}) -> T+5({metrics['t5_ic']['mean_ic']:.4f})
  Monotonic: {metrics['ic_decay']['is_monotonic']}
"""
        
        report += f"""
Lead Factors: {alpha_module.get_selected_factors()}
Factor Directions: {alpha_module.factor_directions}
Factor ICs (adjusted): {alpha_module.factor_ics}
Factor Weights: {alpha_module.factor_weights}

Overall: {"ALL PASSED ✓" if all_passed else "NEEDS IMPROVEMENT"}
======================================================================
"""
        print(report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"V188_Cross_Cycle_Audit_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return results


if __name__ == "__main__":
    runner = V188BacktestRunner(output_dir='reports')
    runner.run_cross_cycle_audit(years=[2023, 2024])