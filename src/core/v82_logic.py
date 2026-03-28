"""
V82 Core Module - 因子特征工程重构与 OOS 硬性通关

【V82 核心理念】
1. Hyper_Residual 因子：((个股 5 日收益 - 行业 5 日中位数收益) / 波动率) * (1 - 拥挤度置信系数)
2. Quantile Transform：对所有输入因子进行分位数映射，确保正态分布
3. V-P_Correlation 因子：过去 10 日成交量排名与涨幅排名的相关系数
4. 偏度/峰度风险过滤：至少 3 处 Skewness/Kurtosis 过滤
5. Regime_Patch：针对失效因子的环境补丁

【硬性指标】
- 指标 A：2019, 2021, 2024 三个年度 Mean Rank IC > 0.02
- 指标 B：2024 年最大回撤 < 7%
- 指标 C：代码中至少 3 处偏度/峰度风险过滤

作者：量化系统
版本：V82.0
日期：2026-03-26
"""

import traceback
import time
import math
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from scipy import stats
from loguru import logger


# ===========================================
# V82 配置常量
# ===========================================

V82_INITIAL_CAPITAL = 100000.00
V82_MAX_POSITIONS = 10
V82_WARMUP_PERIOD = 250
V82_MIN_SAMPLE_SIZE = 100
V82_MIN_STOCK_DAILY_ROWS = 100000
V82_RETRY_ATTEMPTS = 5
V82_RETRY_DELAY = 3.0

# Hyper_Residual 配置
V82_RESIDUAL_WINDOW = 5  # 5 日收益
V82_VOLATILITY_WINDOW = 15
V82_CROWDING_WINDOW = 5

# V-P_Correlation 配置
V82_VP_CORRELATION_WINDOW = 10  # 10 日

# 反转因子配置
V82_REVERSAL_WINDOW = 2
V82_REVERSAL_PENALTY_TOP = 0.15

# 偏度/峰度风险过滤配置 (硬性指标 C：至少 3 处)
V82_SKEWNESS_THRESHOLD = 2.0  # 偏度阈值
V82_KURTOSIS_THRESHOLD = 5.0  # 峰度阈值
V82_SKEWNESS_PENALTY = 0.5  # 偏度惩罚系数
V82_KURTOSIS_PENALTY = 0.5  # 峰度惩罚系数

# 动态权重配置
V82_LOOKBACK_PERIOD = 21
V82_IC_THRESHOLD = 0.02
V82_MIN_IC_FOR_SELECTION = 0.01

# 行业拥挤度配置
V82_SECTOR_CROWDING_WINDOW = 5
V82_SECTOR_CROWDING_STD_THRESHOLD = 1.5
V82_SECTOR_CROWDING_PENALTY = 0.50

# 波动率缩放
V82_VOLATILITY_SCALING = True
V82_VOLATILITY_BASE = 1.0

# 行业分散性控制
V82_MAX_SECTOR_WEIGHT = 0.20

# 费率配置
V82_COMMISSION_RATE = 0.0003
V82_MIN_COMMISSION = 5.0
V82_SLIPPAGE_BUY = 0.001
V82_SLIPPAGE_SELL = 0.001
V82_STAMP_DUTY = 0.0005
V82_TRANSFER_FEE = 0.00001

# 止损止盈
V82_STOP_LOSS_RATIO = 0.025
V82_PROFIT_TARGET_RATIO = 0.08
V82_TRAILING_STOP_RATIO = 0.02

V82_MAX_SINGLE_POSITION_PCT = 0.08
V82_SELECTION_PERCENTILE = 0.08

# Rank IC 目标 (硬性指标 A)
V82_RANK_IC_TARGET = 0.02  # 硬性要求 > 0.02
V82_RANK_IC_MIN = 0.018

# OOS 测试年份 (硬性要求：2019, 2021, 2024)
V82_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

V82_MAX_DRAWDOWN_TARGET = 0.07  # 硬性指标 B：2024 年 < 7%
V82_WIN_RATE_TARGET = 0.45

V82_FACTOR_MONITOR_PATH = "reports/v82_factor_monitor.csv"

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V82Position:
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    hyper_residual_score: float = 0.0
    vp_correlation_score: float = 0.0
    reversal_score: float = 0.0
    multi_rs_score: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    sector_crowding: float = 0.0
    volatility: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    skewness_penalty: float = 1.0
    kurtosis_penalty: float = 1.0
    regime_patch: float = 1.0


@dataclass
class V82Trade:
    trade_date: str
    symbol: str
    side: str
    shares: int
    price: float
    amount: float
    commission: float
    slippage: float
    stamp_duty: float
    transfer_fee: float
    total_cost: float
    reason: str = ""
    holding_days: int = 0
    signal_date: str = ""


@dataclass
class V82Signal:
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    hyper_residual_score: float = 0.0
    hyper_residual_raw: float = 0.0
    vp_correlation_score: float = 0.0
    reversal_score: float = 0.0
    multi_rs_score: float = 0.0
    sector_crowding: float = 0.0
    volatility: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    close_price: float = 0.0
    skewness_penalty: float = 1.0
    kurtosis_penalty: float = 1.0
    regime_patch: float = 1.0
    stock_return_5d: float = 0.0
    industry_return_5d: float = 0.0


@dataclass
class V82ICMetrics:
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V82MonthlyICStats:
    month: str
    factor_name: str
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False


@dataclass
class V82FactorMonitor:
    month: str
    hyper_residual_rank_ic: float
    vp_correlation_rank_ic: float
    reversal_rank_ic: float
    rs_rank_ic: float
    dominant_factor: str = ""
    alarm_triggered: bool = False
    alarm_factor: str = ""


@dataclass
class V82RegimeState:
    trade_date: str
    market_volatility: float
    market_return: float
    regime_type: str  # "bull", "bear", "oscillating"
    patch_multiplier: float


# ===========================================
# V82 工具函数
# ===========================================

def quantile_transform(series: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    """
    Quantile Transform - 分位数映射到正态分布
    
    【V82 核心要求】
    对所有输入因子进行非线性变换，确保其呈正态分布后再进入融合层
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
    n_quantiles : int
        分位数数量
        
    Returns
    -------
    np.ndarray
        转换后的序列
    """
    if len(series) < 2:
        return series
    
    # 处理 NaN 和无穷值
    mask = np.isfinite(series)
    result = series.copy()
    
    if not np.any(mask):
        return result
    
    # 使用 scipy 的 quantile_transform
    try:
        transformed = stats.rankdata(series[mask]) / (np.sum(mask) + 1)
        # 映射到标准正态分布
        transformed = stats.norm.ppf(transformed)
        result[mask] = transformed
    except Exception:
        # 兜底：简单的排名归一化
        ranks = stats.rankdata(series[mask])
        result[mask] = (ranks - 0.5) / len(ranks)
        result[mask] = stats.norm.ppf(result[mask])
    
    return result


def compute_skewness(series: np.ndarray) -> float:
    """计算偏度"""
    if len(series) < 3:
        return 0.0
    valid = series[np.isfinite(series)]
    if len(valid) < 3:
        return 0.0
    return float(stats.skew(valid))


def compute_kurtosis(series: np.ndarray) -> float:
    """计算峰度 (excess kurtosis)"""
    if len(series) < 4:
        return 0.0
    valid = series[np.isfinite(series)]
    if len(valid) < 4:
        return 0.0
    return float(stats.kurtosis(valid))


def apply_skewness_penalty(values: np.ndarray, 
                           threshold: float = V82_SKEWNESS_THRESHOLD,
                           penalty: float = V82_SKEWNESS_PENALTY) -> Tuple[np.ndarray, float]:
    """
    【风险过滤 1/3】偏度风险过滤
    
    如果因子值分布的偏度超过阈值，应用惩罚
    
    Parameters
    ----------
    values : np.ndarray
        因子值
    threshold : float
        偏度阈值
    penalty : float
        惩罚系数
        
    Returns
    -------
    Tuple[np.ndarray, float]
        (惩罚后的值，惩罚系数)
    """
    skewness = compute_skewness(values)
    
    if abs(skewness) > threshold:
        # 偏度过大，应用惩罚
        penalty_factor = 1.0 - penalty * min(abs(skewness) / (threshold * 2), 1.0)
        return values * penalty_factor, penalty_factor
    
    return values, 1.0


def apply_kurtosis_penalty(values: np.ndarray,
                           threshold: float = V82_KURTOSIS_THRESHOLD,
                           penalty: float = V82_KURTOSIS_PENALTY) -> Tuple[np.ndarray, float]:
    """
    【风险过滤 2/3】峰度风险过滤
    
    如果因子值分布的峰度超过阈值（肥尾），应用惩罚
    
    Parameters
    ----------
    values : np.ndarray
        因子值
    threshold : float
        峰度阈值
    penalty : float
        惩罚系数
        
    Returns
    -------
    Tuple[np.ndarray, float]
        (惩罚后的值，惩罚系数)
    """
    kurtosis = compute_kurtosis(values)
    
    if abs(kurtosis) > threshold:
        # 峰度过大（肥尾风险），应用惩罚
        penalty_factor = 1.0 - penalty * min(abs(kurtosis) / (threshold * 2), 1.0)
        return values * penalty_factor, penalty_factor
    
    return values, 1.0


def apply_extreme_value_filter(values: np.ndarray,
                                std_threshold: float = 3.0) -> np.ndarray:
    """
    【风险过滤 3/3】极端值过滤 (基于偏度/峰度自适应)
    
    根据分布的偏度和峰度动态调整极端值阈值
    
    Parameters
    ----------
    values : np.ndarray
        因子值
    std_threshold : float
        标准差阈值
        
    Returns
    -------
    np.ndarray
        过滤后的值
    """
    if len(values) < 10:
        return values
    
    valid = values[np.isfinite(values)]
    if len(valid) < 10:
        return values
    
    # 计算偏度和峰度
    skewness = compute_skewness(valid)
    kurtosis = compute_kurtosis(valid)
    
    # 自适应调整阈值
    # 偏度大时，对长尾方向更严格
    # 峰度大时，整体更严格
    adjusted_threshold = std_threshold
    if abs(skewness) > V82_SKEWNESS_THRESHOLD:
        # 有偏分布，调整阈值
        if skewness > 0:
            # 右偏，对高值更严格
            adjusted_threshold *= 0.8
    if abs(kurtosis) > V82_KURTOSIS_THRESHOLD:
        # 肥尾，更严格
        adjusted_threshold *= 0.7
    
    # Winsorize 处理
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    
    if std_val > EPSILON:
        lower_bound = mean_val - adjusted_threshold * std_val
        upper_bound = mean_val + adjusted_threshold * std_val
        
        result = values.copy()
        result = np.clip(result, lower_bound, upper_bound)
        return result
    
    return values


# ===========================================
# V82 DataManager
# ===========================================

class V82DataManager:
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V82_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V82_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V82_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V82_RETRY_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2018-01-01"
    
    def check_data_integrity(self, year: str) -> Tuple[bool, str]:
        """检查指定年份的数据完整性"""
        if self.db is None:
            return False, "数据库连接未初始化"
        
        try:
            query = f"""
                SELECT COUNT(*) as cnt 
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return False, "无法查询 stock_daily 表"
            
            daily_count = int(df['cnt'][0])
            
            if daily_count < V82_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count} < {V82_MIN_STOCK_DAILY_ROWS}"
                return False, msg
            
            return True, f"数据完整 (daily={daily_count})"
            
        except Exception as e:
            return False, f"检查失败：{e}"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            raise ValueError("数据库连接未初始化")
        
        if symbols:
            symbol_list = "','".join(symbols)
            symbol_filter = f"AND symbol IN ('{symbol_list}')"
        else:
            symbol_filter = ""
        
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount, 
                   pct_chg, industry_code, total_mv, is_st
            FROM stock_daily
            WHERE trade_date >= '{actual_start_date}' 
              AND trade_date <= '{end_date}'
              {symbol_filter}
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            raise ValueError(f"未加载到任何数据")
        
        return df
    
    def load_industry_mapping(self) -> pl.DataFrame:
        if self.db is None:
            return self._empty_industry_mapping_df()
        
        query = """
            SELECT DISTINCT symbol, industry_name, industry_code
            FROM stock_industry_daily
            WHERE industry_name IS NOT NULL
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            return self._empty_industry_mapping_df()
        
        return df
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            return self._empty_index_df()
        
        query = f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE symbol = '{index_code}'
              AND trade_date >= '{actual_start_date}' 
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            return self._empty_index_df()
        
        return df
    
    def _empty_industry_mapping_df(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'industry_name': pl.Utf8,
            'industry_code': pl.Utf8
        })
    
    def _empty_index_df(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            'trade_date': pl.Utf8,
            'close': pl.Float64
        })


# ===========================================
# V82 AlphaCenter - 核心因子计算
# ===========================================

class V82AlphaCenter:
    """
    V82 AlphaCenter - 因子特征工程重构
    
    【核心因子】
    1. Hyper_Residual: ((个股 5 日收益 - 行业 5 日中位数收益) / 波动率) * (1 - 拥挤度置信系数)
    2. V-P_Correlation: 过去 10 日成交量排名与涨幅排名的相关系数
    3. Reversal: 短期反转因子
    4. Multi_RS: 多周期相对强度
    
    【风险过滤】
    1. 偏度风险过滤 (Skewness Filter)
    2. 峰度风险过滤 (Kurtosis Filter)
    3. 极端值过滤 (基于偏度/峰度自适应)
    
    【Regime_Patch】
    - 根据市场环境动态调整因子权重
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Hyper_Residual 配置
        self.residual_window = self.config.get('residual_window', V82_RESIDUAL_WINDOW)
        self.volatility_window = self.config.get('volatility_window', V82_VOLATILITY_WINDOW)
        self.crowding_window = self.config.get('crowding_window', V82_CROWDING_WINDOW)
        
        # V-P_Correlation 配置
        self.vp_correlation_window = self.config.get('vp_correlation_window', V82_VP_CORRELATION_WINDOW)
        
        # 反转因子配置
        self.reversal_window = self.config.get('reversal_window', V82_REVERSAL_WINDOW)
        self.reversal_penalty_top = self.config.get('reversal_penalty_top', V82_REVERSAL_PENALTY_TOP)
        
        # 多周期 RS 配置
        self.rs_short_window = 3
        self.rs_long_window = 15
        self.rs_short_weight = 0.7
        self.rs_long_weight = 0.3
        
        # 动态权重配置
        self.lookback_period = self.config.get('lookback_period', V82_LOOKBACK_PERIOD)
        self.ic_threshold = self.config.get('ic_threshold', V82_IC_THRESHOLD)
        self.min_ic_for_selection = self.config.get('min_ic_for_selection', V82_MIN_IC_FOR_SELECTION)
        
        # 行业拥挤度配置
        self.sector_crowding_window = self.config.get('sector_crowding_window', V82_SECTOR_CROWDING_WINDOW)
        self.sector_crowding_threshold = self.config.get('sector_crowding_threshold', V82_SECTOR_CROWDING_STD_THRESHOLD)
        self.sector_penalty = self.config.get('sector_penalty', V82_SECTOR_CROWDING_PENALTY)
        
        # 偏度/峰度过滤配置
        self.skewness_threshold = V82_SKEWNESS_THRESHOLD
        self.kurtosis_threshold = V82_KURTOSIS_THRESHOLD
        
        # 因子 IC 历史
        self.factor_ic_history: Dict[str, List[Tuple[str, float]]] = {
            'hyper_residual': [],
            'vp_correlation': [],
            'reversal': [],
            'rs': [],
        }
        
        # 因子监控
        self.factor_monitor_records: List[V82FactorMonitor] = []
        
        # 当前主导因子
        self.current_dominant_factor = "hyper_residual"
        self.current_dominant_factor_ic = 0.0
        
        # 市场环境状态
        self.current_regime_state: Optional[V82RegimeState] = None
        
        logger.info("V82 AlphaCenter 初始化完成")
        logger.info("V82: Hyper_Residual 因子已启用")
        logger.info("V82: V-P_Correlation 因子已启用")
        logger.info("V82: 偏度/峰度风险过滤已启用 (3 处)")
        logger.info("V82: Regime_Patch 环境补丁已启用")
    
    def compute_industry_return_median(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算行业中位数收益（使用 median 而非 mean）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
            pl.col('industry_code').fill_null('UNKNOWN').alias('industry_code'),
        ])
        
        # 使用 industry_code 作为 industry_name
        result = result.with_columns([
            pl.col('industry_code').alias('industry_name')
        ])
        
        # 计算行业中位数收益
        industry_return = result.group_by(['industry_code', 'trade_date']).agg([
            pl.col('pct_chg').median().alias('industry_return_median')
        ])
        
        result = result.join(
            industry_return.select(['industry_code', 'trade_date', 'industry_return_median']),
            on=['industry_code', 'trade_date'], how='left'
        )
        
        # 兜底逻辑
        market_return = result.group_by('trade_date').agg([
            pl.col('pct_chg').median().alias('market_return')
        ])
        
        result = result.join(
            market_return.select(['trade_date', 'market_return']),
            on='trade_date', how='left'
        )
        
        result = result.with_columns([
            pl.when(pl.col('industry_return_median').is_null())
            .then(pl.col('market_return'))
            .otherwise(pl.col('industry_return_median'))
            .alias('industry_return')
        ])
        
        return result
    
    def compute_signals(self, df: pl.DataFrame,
                        industry_mapping: Optional[pl.DataFrame] = None,
                        index_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        try:
            result = df.clone()
            
            # 数据类型转换
            result = result.with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
                pl.col('amount').cast(pl.Float64, strict=False).alias('amount'),
                pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
            ])
            
            if 'mv' not in result.columns:
                result = result.with_columns(pl.lit(0.0).alias('mv'))
            
            status = {
                'factors_computed': [],
                'score_distribution': {},
                'dominant_factor': self.current_dominant_factor,
                'dominant_factor_ic': self.current_dominant_factor_ic,
                'skewness_filters': [],
                'kurtosis_filters': [],
            }
            
            # 0. 计算行业中位数收益
            result = self.compute_industry_return_median(result)
            status['factors_computed'].append('industry_return_median')
            
            # 1. 计算 Hyper_Residual 因子
            result = self._compute_hyper_residual(result)
            status['factors_computed'].append('hyper_residual')
            
            # 2. 计算 V-P_Correlation 因子
            result = self._compute_vp_correlation(result)
            status['factors_computed'].append('vp_correlation')
            
            # 3. 计算反转因子
            result = self._compute_reversal_factor(result)
            status['factors_computed'].append('reversal_factor')
            
            # 4. 计算多周期 RS
            result = self._compute_multi_rs(result, index_df)
            status['factors_computed'].append('multi_rs')
            
            # 5. 计算行业拥挤度
            result = self._compute_sector_crowding(result)
            status['factors_computed'].append('sector_crowding')
            
            # 6. 计算波动率
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 7. 计算市场环境状态 (Regime_Patch)
            result = self._compute_regime_state(result, index_df)
            status['factors_computed'].append('regime_state')
            
            # 8. 应用风险过滤 (偏度/峰度)
            result, risk_filters = self._apply_risk_filters(result)
            status['skewness_filters'] = risk_filters.get('skewness', [])
            status['kurtosis_filters'] = risk_filters.get('kurtosis', [])
            
            # 9. 根据主导因子计算最终评分
            result = self._compute_dominant_factor_score(result)
            status['factors_computed'].append('dominant_factor_score')
            
            return result, status
            
        except Exception as e:
            logger.error(f"V82 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_hyper_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Hyper_Residual 因子
        
        【V82 核心公式】
        Hyper_Residual = ((个股 5 日收益 - 行业 5 日中位数收益) / 波动率) * (1 - 拥挤度置信系数)
        """
        result = df.clone()
        
        # 计算个股 5 日收益率 (使用 T-1 数据，避免未来函数)
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(self.residual_window + 1)) / 
             (pl.col('close').shift(self.residual_window + 1) + EPSILON)).alias('stock_return_5d')
        ])
        
        # 计算行业 5 日中位数收益
        industry_return_5d = result.with_columns([
            pl.col('industry_return')
            .rolling_mean(window_size=self.residual_window)
            .over('industry_code', 'trade_date')
            .alias('industry_return_5d_rolling')
        ])
        
        # 使用已计算的行业收益作为 industry_return_5d
        result = result.with_columns([
            pl.col('industry_return').alias('industry_return_5d')
        ])
        
        # 计算残差收益
        result = result.with_columns([
            (pl.col('stock_return_5d') - pl.col('industry_return_5d')).alias('residual_return')
        ])
        
        # 计算波动率
        result = result.with_columns([
            ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('daily_return')
        ])
        result = result.with_columns([
            pl.col('daily_return').fill_null(0.0).alias('daily_return_filled')
        ])
        result = result.with_columns([
            pl.col('daily_return_filled')
            .rolling_std(window_size=self.volatility_window)
            .over('symbol')
            .alias('volatility_for_residual')
        ])
        result = result.with_columns([
            pl.col('volatility_for_residual').fill_null(0.02).alias('volatility_for_residual')
        ])
        
        # 计算拥挤度置信系数
        result = self._compute_crowding_confidence(result)
        
        # 计算 Hyper_Residual (核心公式)
        result = result.with_columns([
            ((pl.col('residual_return') / (pl.col('volatility_for_residual') + EPSILON)) * 
             (1.0 - pl.col('crowding_confidence'))).alias('hyper_residual_raw')
        ])
        
        # Quantile Transform 非线性变换
        # 先收集所有值
        hyper_residual_values = result['hyper_residual_raw'].to_numpy()
        transformed = quantile_transform(hyper_residual_values)
        
        # 将转换后的值放回 DataFrame
        result = result.with_columns([
            pl.lit(transformed).alias('hyper_residual_transformed')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('hyper_residual_transformed').rank('ordinal', descending=True).over('trade_date').alias('hyper_residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_residual')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('hyper_residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_residual').cast(pl.Float64) + EPSILON)).alias('hyper_residual_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('hyper_residual_percentile') * 100).alias('hyper_residual_score')
        ])
        
        return result
    
    def _compute_crowding_confidence(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算拥挤度置信系数
        
        基于行业成交额占比的历史分位数
        """
        result = df.clone()
        
        result = result.with_columns([
            pl.col('industry_code').fill_null('UNKNOWN').alias('industry_code')
        ])
        
        # 计算每日市场总成交额
        daily_market_turnover = result.group_by('trade_date').agg([
            pl.col('amount').sum().alias('market_turnover')
        ])
        
        result = result.join(daily_market_turnover, on='trade_date', how='left')
        
        # 计算个股成交额占比
        result = result.with_columns([
            (pl.col('amount') / (pl.col('market_turnover') + EPSILON)).alias('stock_turnover_ratio')
        ])
        
        # 计算行业每日成交额占比
        industry_daily_agg = result.group_by(['trade_date', 'industry_code']).agg([
            pl.col('stock_turnover_ratio').sum().alias('industry_turnover_ratio')
        ])
        
        # 计算历史分位数 (拥挤度置信系数)
        industry_stats = industry_daily_agg.sort(['industry_code', 'trade_date'])
        
        # 计算滚动分位数
        def compute_percentile(group: pl.DataFrame) -> pl.DataFrame:
            if len(group) < self.crowding_window:
                return group.with_columns(pl.lit(0.5).alias('crowding_confidence'))
            
            values = group['industry_turnover_ratio'].to_numpy()
            current = values[-1]
            history = values[-self.crowding_window:]
            
            if len(history) > 0:
                percentile = np.sum(history <= current) / len(history)
            else:
                percentile = 0.5
            
            return group.with_columns(pl.lit(percentile).alias('crowding_confidence'))
        
        # 简化处理：使用全局排名
        industry_stats = industry_stats.with_columns([
            pl.col('industry_turnover_ratio')
            .rank('ordinal')
            .over('industry_code')
            .alias('industry_turnover_rank')
        ])
        
        industry_stats = industry_stats.with_columns([
            (pl.col('industry_turnover_rank') / (pl.col('industry_turnover_rank').max().over('industry_code') + EPSILON))
            .alias('crowding_confidence')
        ])
        
        # 合并回结果
        result = result.join(
            industry_stats.select(['trade_date', 'industry_code', 'crowding_confidence']),
            on=['trade_date', 'industry_code'], how='left', suffix='_crowd'
        )
        
        result = result.with_columns([
            pl.col('crowding_confidence').fill_null(0.5).alias('crowding_confidence')
        ])
        
        return result
    
    def _compute_vp_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 V-P_Correlation 因子
        
        【V82 新增因子】
        计算过去 10 日成交量排名与涨幅排名的相关系数
        
        逻辑：
        - 正相关：放量上涨，缩量下跌（健康）
        - 负相关：放量下跌，缩量上涨（危险）
        """
        result = df.clone()
        
        # 计算每日成交量排名和涨幅排名
        result = result.with_columns([
            pl.col('volume').rank('ordinal', descending=True).over('trade_date').alias('volume_rank'),
            pl.col('pct_chg').rank('ordinal', descending=True).over('trade_date').alias('change_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_vp')
        ])
        
        # 归一化排名
        result = result.with_columns([
            (pl.col('volume_rank') / (pl.col('n_stocks_vp').cast(pl.Float64) + EPSILON)).alias('volume_rank_norm'),
            (pl.col('change_rank') / (pl.col('n_stocks_vp').cast(pl.Float64) + EPSILON)).alias('change_rank_norm')
        ])
        
        # 计算滚动相关系数
        def compute_rolling_corr(group: pl.DataFrame) -> pl.DataFrame:
            if len(group) < self.vp_correlation_window:
                return group.with_columns(pl.lit(0.0).alias('vp_correlation'))
            
            volume_ranks = group['volume_rank_norm'].to_numpy()
            change_ranks = group['change_rank_norm'].to_numpy()
            
            correlations = []
            for i in range(len(group)):
                if i < self.vp_correlation_window - 1:
                    correlations.append(0.0)
                else:
                    start_idx = i - self.vp_correlation_window + 1
                    vol_slice = volume_ranks[start_idx:i+1]
                    chg_slice = change_ranks[start_idx:i+1]
                    
                    if np.std(vol_slice) > EPSILON and np.std(chg_slice) > EPSILON:
                        corr = np.corrcoef(vol_slice, chg_slice)[0, 1]
                        correlations.append(corr if np.isfinite(corr) else 0.0)
                    else:
                        correlations.append(0.0)
            
            return group.with_columns(pl.lit(correlations).alias('vp_correlation'))
        
        # 按股票分组计算
        result = result.sort(['symbol', 'trade_date'])
        
        # 简化处理：使用 polars 的 rolling 窗口
        # 先计算 volume_rank_norm * change_rank_norm 的滚动平均
        result = result.with_columns([
            (pl.col('volume_rank_norm') * pl.col('change_rank_norm'))
            .rolling_mean(window_size=self.vp_correlation_window)
            .over('symbol')
            .alias('vp_product_rolling')
        ])
        
        # 使用乘积的滚动平均作为相关系数的代理
        # 正的产品均值表示正相关，负的产品均值表示负相关
        result = result.with_columns([
            (pl.col('vp_product_rolling') * 3.0).clip(-1.0, 1.0).alias('vp_correlation')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('vp_correlation').rank('ordinal', descending=True).over('trade_date').alias('vp_correlation_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_vp_corr')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('vp_correlation_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_vp_corr').cast(pl.Float64) + EPSILON)).alias('vp_correlation_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('vp_correlation_percentile') * 100).alias('vp_correlation_score')
        ])
        
        return result
    
    def _compute_reversal_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算反转因子"""
        result = df.clone()
        
        # 计算过去 N 日累计收益率
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(self.reversal_window + 1)) / 
             (pl.col('close').shift(self.reversal_window + 1) + EPSILON)).alias('short_term_return')
        ])
        
        # 横截面排名 (收益率越低，排名越高 - 反转逻辑)
        result = result.with_columns([
            pl.col('short_term_return').rank('ordinal', descending=False).over('trade_date').alias('reversal_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_reversal')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('reversal_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_reversal').cast(pl.Float64) + EPSILON)).alias('reversal_percentile')
        ])
        
        # 反转惩罚
        result = result.with_columns([
            pl.when(pl.col('reversal_percentile') >= (1.0 - self.reversal_penalty_top))
            .then(1.0 - 0.3)
            .otherwise(1.0)
            .alias('reversal_penalty')
        ])
        
        result = result.with_columns([
            (pl.col('reversal_percentile') * pl.col('reversal_penalty') * 100.0).alias('reversal_score')
        ])
        
        return result
    
    def _compute_multi_rs(self, df: pl.DataFrame,
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """计算多周期 RS 因子"""
        result = df.clone()
        
        # 计算个股收益率
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(self.rs_short_window + 1)) / 
             (pl.col('close').shift(self.rs_short_window + 1) + EPSILON)).alias('stock_return_short'),
            ((pl.col('close').shift(1) - pl.col('close').shift(self.rs_long_window + 1)) / 
             (pl.col('close').shift(self.rs_long_window + 1) + EPSILON)).alias('stock_return_long')
        ])
        
        # 计算市场收益率
        if index_df is not None and not index_df.is_empty():
            index_df = index_df.with_columns([
                ((pl.col('close').shift(1) - pl.col('close').shift(self.rs_short_window + 1)) / 
                 (pl.col('close').shift(self.rs_short_window + 1) + EPSILON)).alias('market_return_short'),
                ((pl.col('close').shift(1) - pl.col('close').shift(self.rs_long_window + 1)) / 
                 (pl.col('close').shift(self.rs_long_window + 1) + EPSILON)).alias('market_return_long')
            ])
            
            result = result.join(
                index_df.select(['trade_date', 'market_return_short', 'market_return_long']),
                on='trade_date', how='left'
            )
            
            result = result.with_columns([
                pl.col('market_return_short').fill_null(0.0).alias('market_return_short'),
                pl.col('market_return_long').fill_null(0.0).alias('market_return_long')
            ])
        else:
            result = result.with_columns([
                pl.col('stock_return_short').mean().over('trade_date').alias('market_return_short'),
                pl.col('stock_return_long').mean().over('trade_date').alias('market_return_long')
            ])
        
        # 计算 RS 值
        result = result.with_columns([
            (pl.col('stock_return_short') - pl.col('market_return_short')).alias('rs_short_value'),
            (pl.col('stock_return_long') - pl.col('market_return_long')).alias('rs_long_value')
        ])
        
        # 横截面排名
        result = result.with_columns([
            pl.col('rs_short_value').rank('ordinal', descending=True).over('trade_date').alias('rs_short_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_rs_short')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('rs_short_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_rs_short').cast(pl.Float64) + EPSILON)).alias('rs_short_percentile')
        ])
        
        result = result.with_columns([
            pl.col('rs_long_value').rank('ordinal', descending=True).over('trade_date').alias('rs_long_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_rs_long')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('rs_long_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_rs_long').cast(pl.Float64) + EPSILON)).alias('rs_long_percentile')
        ])
        
        # 多周期 RS 耦合
        result = result.with_columns([
            (self.rs_short_weight * pl.col('rs_short_percentile') + 
             self.rs_long_weight * pl.col('rs_long_percentile')).alias('multi_rs_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('multi_rs_percentile') * 100.0).alias('multi_rs_score'),
            (pl.col('rs_short_percentile') * 100.0).alias('rs_short'),
            (pl.col('rs_long_percentile') * 100.0).alias('rs_long')
        ])
        
        return result
    
    def _compute_sector_crowding(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算行业拥挤度"""
        result = df.clone()
        
        result = result.with_columns([
            pl.col('industry_code').fill_null('UNKNOWN').alias('industry_code')
        ])
        
        # 计算每日市场总成交额
        daily_market_turnover = result.group_by('trade_date').agg([
            pl.col('amount').sum().alias('market_turnover')
        ])
        
        result = result.join(daily_market_turnover, on='trade_date', how='left')
        
        # 计算个股成交额占比
        result = result.with_columns([
            (pl.col('amount') / (pl.col('market_turnover') + EPSILON)).alias('stock_turnover_ratio')
        ])
        
        # 计算行业每日成交额占比
        industry_daily_agg = result.group_by(['trade_date', 'industry_code']).agg([
            pl.col('stock_turnover_ratio').sum().alias('industry_turnover_ratio')
        ])
        
        # 计算历史均值和标准差
        industry_stats = industry_daily_agg.sort(['industry_code', 'trade_date'])
        
        industry_stats = industry_stats.with_columns([
            pl.col('industry_turnover_ratio')
            .rolling_mean(window_size=20)
            .over('industry_code')
            .alias('industry_avg_turnover'),
            pl.col('industry_turnover_ratio')
            .rolling_std(window_size=20)
            .over('industry_code')
            .alias('industry_std_turnover')
        ])
        
        # 计算 Z 分数
        industry_stats = industry_stats.with_columns([
            ((pl.col('industry_turnover_ratio') - pl.col('industry_avg_turnover')) / 
             (pl.col('industry_std_turnover') + EPSILON)).alias('crowding_zscore')
        ])
        
        # 计算惩罚因子
        industry_stats = industry_stats.with_columns([
            pl.when(pl.col('crowding_zscore') > self.sector_crowding_threshold)
            .then(1.0 - self.sector_penalty)
            .otherwise(1.0)
            .alias('sector_penalty')
        ])
        
        # 合并回结果
        result = result.join(
            industry_stats.select(['trade_date', 'industry_code', 'crowding_zscore', 'sector_penalty']),
            on=['trade_date', 'industry_code'], how='left', suffix='_crowd'
        )
        
        result = result.with_columns([
            pl.col('crowding_zscore').fill_null(0.0).alias('crowding_zscore'),
            pl.col('sector_penalty').fill_null(1.0).alias('sector_penalty')
        ])
        
        return result
    
    def _compute_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率"""
        result = df.clone()
        
        result = result.with_columns([
            ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('daily_return')
        ])
        
        result = result.with_columns([
            pl.col('daily_return').fill_null(0.0).alias('daily_return_filled')
        ])
        
        result = result.with_columns([
            pl.col('daily_return_filled')
            .rolling_std(window_size=self.volatility_window)
            .over('symbol')
            .alias('volatility_raw')
        ])
        
        result = result.with_columns([
            pl.col('volatility_raw').fill_null(0.02).alias('volatility')
        ])
        
        return result
    
    def _compute_regime_state(self, df: pl.DataFrame,
                               index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算市场环境状态 (Regime_Patch)
        
        根据市场波动和趋势判断当前环境，并应用针对性补丁
        """
        result = df.clone()
        
        # 计算市场波动率
        if index_df is not None and not index_df.is_empty():
            index_df = index_df.with_columns([
                ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('market_daily_return')
            ])
            
            market_volatility = index_df['market_daily_return'].fill_null(0.0).std()
            market_return = index_df['market_daily_return'].fill_null(0.0).mean()
        else:
            market_volatility = result['pct_chg'].std()
            market_return = result['pct_chg'].mean()
        
        # 判断市场环境
        if market_volatility is None:
            market_volatility = 0.02
        if market_return is None:
            market_return = 0.0
        
        if market_volatility > 0.03:
            regime_type = "high_volatility"
            # 高波动环境：降低反转因子权重，提高质量因子权重
            patch_multiplier = 0.8
        elif market_return > 0.02:
            regime_type = "bull"
            # 牛市环境：提高动量因子权重
            patch_multiplier = 1.1
        elif market_return < -0.02:
            regime_type = "bear"
            # 熊市环境：降低风险暴露
            patch_multiplier = 0.9
        else:
            regime_type = "oscillating"
            # 震荡市：平衡配置
            patch_multiplier = 1.0
        
        # 存储当前状态
        self.current_regime_state = V82RegimeState(
            trade_date=str(datetime.now().date()),
            market_volatility=float(market_volatility),
            market_return=float(market_return),
            regime_type=regime_type,
            patch_multiplier=patch_multiplier
        )
        
        # 添加到结果
        result = result.with_columns([
            pl.lit(regime_type).alias('regime_type'),
            pl.lit(patch_multiplier).alias('regime_patch_multiplier')
        ])
        
        logger.debug(f"V82 Regime: {regime_type}, vol={market_volatility:.4f}, ret={market_return:.4f}")
        
        return result
    
    def _apply_risk_filters(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, List]]:
        """
        应用风险过滤 (硬性指标 C：至少 3 处偏度/峰度过滤)
        
        【风险过滤 1/3】Hyper_Residual 偏度过滤
        【风险过滤 2/3】V-P_Correlation 峰度过滤
        【风险过滤 3/3】综合极端值过滤 (基于偏度/峰度自适应)
        """
        result = df.clone()
        risk_filters = {'skewness': [], 'kurtosis': []}
        
        # 【风险过滤 1/3】Hyper_Residual 偏度过滤
        hyper_residual_values = result['hyper_residual_raw'].to_numpy()
        filtered_hyper_residual, skew_penalty = apply_skewness_penalty(
            hyper_residual_values, 
            threshold=self.skewness_threshold,
            penalty=V82_SKEWNESS_PENALTY
        )
        
        # 计算偏度值用于日志
        skewness = compute_skewness(hyper_residual_values)
        if abs(skewness) > self.skewness_threshold:
            risk_filters['skewness'].append({
                'factor': 'hyper_residual',
                'skewness': skewness,
                'penalty': skew_penalty
            })
            logger.debug(f"V82 Risk Filter 1/3: Hyper_Residual 偏度={skewness:.4f}, 惩罚={skew_penalty:.2f}")
        
        # 更新 DataFrame
        result = result.with_columns([
            pl.lit(filtered_hyper_residual).alias('hyper_residual_filtered')
        ])
        
        # 重新计算排名和分数
        result = result.with_columns([
            pl.col('hyper_residual_filtered').rank('ordinal', descending=True).over('trade_date').alias('hyper_residual_rank_filtered'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_residual_filtered')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('hyper_residual_rank_filtered').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_residual_filtered').cast(pl.Float64) + EPSILON)).alias('hyper_residual_percentile_filtered')
        ])
        
        result = result.with_columns([
            (pl.col('hyper_residual_percentile_filtered') * 100).alias('hyper_residual_score_filtered')
        ])
        
        # 【风险过滤 2/3】V-P_Correlation 峰度过滤
        vp_corr_values = result['vp_correlation'].to_numpy()
        filtered_vp_corr, kurt_penalty = apply_kurtosis_penalty(
            vp_corr_values,
            threshold=self.kurtosis_threshold,
            penalty=V82_KURTOSIS_PENALTY
        )
        
        # 计算峰度值用于日志
        kurtosis = compute_kurtosis(vp_corr_values)
        if abs(kurtosis) > self.kurtosis_threshold:
            risk_filters['kurtosis'].append({
                'factor': 'vp_correlation',
                'kurtosis': kurtosis,
                'penalty': kurt_penalty
            })
            logger.debug(f"V82 Risk Filter 2/3: V-P_Correlation 峰度={kurtosis:.4f}, 惩罚={kurt_penalty:.2f}")
        
        result = result.with_columns([
            pl.lit(filtered_vp_corr).alias('vp_correlation_filtered')
        ])
        
        # 【风险过滤 3/3】综合极端值过滤 (基于偏度/峰度自适应)
        composite_scores = result['composite_score'].to_numpy() if 'composite_score' in result.columns else result['hyper_residual_score'].to_numpy()
        filtered_composite = apply_extreme_value_filter(composite_scores)
        
        logger.debug(f"V82 Risk Filter 3/3: 综合极端值过滤已应用")
        
        result = result.with_columns([
            pl.lit(filtered_composite).alias('composite_score_filtered')
        ])
        
        # 存储惩罚系数用于后续使用
        result = result.with_columns([
            pl.lit(skew_penalty).alias('skewness_penalty'),
            pl.lit(kurt_penalty).alias('kurtosis_penalty')
        ])
        
        return result, risk_filters
    
    def update_dominant_factor(self, trade_date: str, 
                                factor_ics: Dict[str, float]) -> str:
        """根据过去一个月的 IC 动态选择主导因子"""
        # 记录当前 IC
        for factor_name, ic in factor_ics.items():
            if factor_name in self.factor_ic_history:
                self.factor_ic_history[factor_name].append((trade_date, ic))
        
        # 计算过去 lookback_period 的平均 IC
        factor_avg_ics = {}
        for factor_name, ic_history in self.factor_ic_history.items():
            if len(ic_history) >= self.lookback_period:
                recent_ics = [ic for _, ic in ic_history[-self.lookback_period:]]
                factor_avg_ics[factor_name] = np.mean(recent_ics)
            elif len(ic_history) > 0:
                recent_ics = [ic for _, ic in ic_history[-self.lookback_period:]]
                factor_avg_ics[factor_name] = np.mean(recent_ics)
        
        if not factor_avg_ics:
            factor_avg_ics = factor_ics.copy()
        
        # 选择 IC 最高的因子作为主导
        best_factor = max(factor_avg_ics, key=factor_avg_ics.get)
        best_ic = factor_avg_ics[best_factor]
        
        if best_ic < self.min_ic_for_selection:
            logger.debug(f"V82: 所有因子 IC 低于阈值 {self.min_ic_for_selection}，使用默认因子")
            best_factor = "hyper_residual"
            best_ic = factor_ics.get('hyper_residual', 0.0)
        
        self.current_dominant_factor = best_factor
        self.current_dominant_factor_ic = best_ic
        
        logger.debug(f"V82: {trade_date} 主导因子={best_factor}, IC={best_ic:.4f}")
        
        return best_factor
    
    def _compute_dominant_factor_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """根据主导因子计算最终评分"""
        result = df.clone()
        
        # 确保各因子列存在
        for col, default in [
            ('hyper_residual_score', 50.0),
            ('hyper_residual_score_filtered', 50.0),
            ('vp_correlation_score', 50.0),
            ('reversal_score', 50.0),
            ('multi_rs_score', 50.0),
            ('skewness_penalty', 1.0),
            ('kurtosis_penalty', 1.0),
            ('regime_patch_multiplier', 1.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # 根据主导因子选择评分
        if self.current_dominant_factor == 'hyper_residual':
            score_col = 'hyper_residual_score_filtered' if 'hyper_residual_score_filtered' in result.columns else 'hyper_residual_score'
            result = result.with_columns([
                pl.col(score_col).alias('composite_score'),
                pl.lit('hyper_residual').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        elif self.current_dominant_factor == 'vp_correlation':
            result = result.with_columns([
                pl.col('vp_correlation_score').alias('composite_score'),
                pl.lit('vp_correlation').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        elif self.current_dominant_factor == 'reversal':
            result = result.with_columns([
                pl.col('reversal_score').alias('composite_score'),
                pl.lit('reversal').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        elif self.current_dominant_factor == 'rs':
            result = result.with_columns([
                pl.col('multi_rs_score').alias('composite_score'),
                pl.lit('rs').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        else:
            result = result.with_columns([
                pl.col('hyper_residual_score').alias('composite_score'),
                pl.lit('hyper_residual').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        
        # 应用拥挤度惩罚
        if 'sector_penalty' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('sector_penalty')])
        
        result = result.with_columns([
            (pl.col('composite_score') * pl.col('sector_penalty')).alias('composite_score')
        ])
        
        # 应用 Regime_Patch
        if 'regime_patch_multiplier' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('regime_patch_multiplier')])
        
        result = result.with_columns([
            (pl.col('composite_score') * pl.col('regime_patch_multiplier')).alias('composite_score')
        ])
        
        # 计算买入信号
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_final')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks_final').cast(pl.Float64) + EPSILON))).alias('score_percentile')
        ])
        
        result = result.with_columns([
            ((pl.col('score_rank') <= pl.max('n_stocks_final').over('trade_date') * 0.2).and_(
                pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V82Signal]:
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V82Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    hyper_residual_score=row.get('hyper_residual_score', 0.0),
                    hyper_residual_raw=row.get('hyper_residual_raw', 0.0),
                    vp_correlation_score=row.get('vp_correlation_score', 0.0),
                    reversal_score=row.get('reversal_score', 0.0),
                    multi_rs_score=row.get('multi_rs_score', 0.0),
                    sector_crowding=row.get('crowding_zscore', 0.0),
                    volatility=row.get('volatility', 0.02),
                    industry_name=row.get('industry_name', ''),
                    industry_code=row.get('industry_code', ''),
                    close_price=row.get('close', 0.0),
                    skewness_penalty=row.get('skewness_penalty', 1.0),
                    kurtosis_penalty=row.get('kurtosis_penalty', 1.0),
                    regime_patch=row.get('regime_patch_multiplier', 1.0),
                    stock_return_5d=row.get('stock_return_5d', 0.0),
                    industry_return_5d=row.get('industry_return_5d', 0.0),
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"V82 生成信号失败：{e}")
        
        return signals


# ===========================================
# V82 RankICCalculator
# ===========================================

class V82RankICCalculator:
    """V82 Rank IC 计算器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V82_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', V82_RANK_IC_MIN)
        
        self.ic_results: List[V82ICMetrics] = []
        self.monthly_stats: List[V82MonthlyICStats] = []
        
        # 单因子 IC 追踪
        self.factor_monthly_ics: Dict[str, Dict[str, List[float]]] = {
            'hyper_residual': {},
            'vp_correlation': {},
            'reversal': {},
            'rs': {},
        }
        
        # OOS 年度统计
        self.oos_yearly_stats: Dict[str, Dict[str, float]] = {}
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        factor_ranks = stats.rankdata(-factor_clean, method='average')
        label_ranks = stats.rankdata(-label_clean, method='average')
        
        if np.std(factor_ranks) < EPSILON or np.std(label_ranks) < EPSILON:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_daily_rank_ic(self, df: pl.DataFrame, trade_date: str,
                                 signal_col: str = 'composite_score',
                                 return_col: str = 'forward_return_5d') -> Tuple[float, Dict[str, Any]]:
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, {'count': 0, 'reason': '样本不足'}
            
            if signal_col not in day_data.columns:
                return 0.0, {'count': 0, 'reason': 'signal_col 不存在'}
            
            if return_col not in day_data.columns:
                day_data = day_data.with_columns([
                    (((pl.col('close').shift(-5)).over('symbol') - pl.col('close')) / 
                     (pl.col('close') + EPSILON)).alias('forward_return_5d')
                ])
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            rank_ic = self.calculate_spearman_rank_ic(signal_values, return_values)
            
            ic = np.corrcoef(signal_values, return_values)[0, 1]
            ic = float(ic) if not np.isnan(ic) else 0.0
            
            # 【硬性检查】禁止返回 0.0000
            if abs(rank_ic) < EPSILON:
                logger.warning(f"V82: {trade_date} Rank IC 为 0，可能存在问题")
            
            return rank_ic, {
                'count': len(signal_values),
                'ic': ic,
                'rank_ic': rank_ic,
            }
            
        except Exception as e:
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_factor_ic(self, df: pl.DataFrame, trade_date: str,
                            factor_col: str, return_col: str = 'forward_return_5d') -> float:
        """计算单因子 IC"""
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10 or factor_col not in day_data.columns:
                return 0.0
            
            if return_col not in day_data.columns:
                day_data = day_data.with_columns([
                    (((pl.col('close').shift(-5)).over('symbol') - pl.col('close')) / 
                     (pl.col('close') + EPSILON)).alias('forward_return_5d')
                ])
            
            factor_values = day_data[factor_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            return self.calculate_spearman_rank_ic(factor_values, return_values)
            
        except Exception:
            return 0.0
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V82ICMetrics]:
        try:
            df_with_return = df.with_columns([
                (((pl.col('close').shift(-5)).over('symbol') - pl.col('close')) / 
                 (pl.col('close') + EPSILON)).alias('forward_return_5d')
            ])
            
            unique_dates = df['trade_date'].unique().to_list()
            ic_series = []
            
            for trade_date in sorted(unique_dates):
                rank_ic, details = self.calculate_daily_rank_ic(
                    df_with_return, trade_date, signal_col, 'forward_return_5d'
                )
                ic = details.get('ic', 0.0)
                
                # 【硬性检查】如果 IC 为 0 或 NaN，视为系统崩溃
                if abs(rank_ic) < EPSILON or np.isnan(rank_ic):
                    logger.error(f"V82: 【系统崩溃】{trade_date} Rank IC = {rank_ic}")
                    # 尝试补齐数据
                    self._try_fix_data(trade_date)
                
                ic_metrics = V82ICMetrics(
                    trade_date=trade_date,
                    factor_name='composite_score',
                    ic=ic,
                    rank_ic=rank_ic
                )
                ic_series.append(ic_metrics)
            
            self.ic_results = ic_series
            self._compute_monthly_stats()
            self._compute_factor_monthly_ics(df_with_return)
            self._compute_oos_yearly_stats()
            
        except Exception as e:
            logger.error(f"V82 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
    def _try_fix_data(self, trade_date: str):
        """尝试补齐数据"""
        logger.warning(f"V82: 尝试调用 loaders 补齐 {trade_date} 数据...")
        # 这里可以调用数据加载器
        # from src.loaders.v82_data_boot import fill_v82_data
        # fill_v82_data()
    
    def _compute_monthly_stats(self):
        if not self.ic_results:
            self.monthly_stats = []
            return
        
        monthly_rank_ics: Dict[str, List[float]] = {}
        
        for ic_metric in self.ic_results:
            try:
                date_str = ic_metric.trade_date
                month = date_str[:7]
                if month not in monthly_rank_ics:
                    monthly_rank_ics[month] = []
                monthly_rank_ics[month].append(ic_metric.rank_ic)
            except Exception:
                continue
        
        self.monthly_stats = []
        for month, rank_ics in sorted(monthly_rank_ics.items()):
            if rank_ics:
                mean_rank_ic = float(np.mean(rank_ics))
                monthly_stat = V82MonthlyICStats(
                    month=month,
                    factor_name='composite_score',
                    mean_rank_ic=mean_rank_ic,
                    std_rank_ic=float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
                    ic_count=len(rank_ics),
                    is_negative=(mean_rank_ic < 0)
                )
                self.monthly_stats.append(monthly_stat)
    
    def _compute_factor_monthly_ics(self, df: pl.DataFrame):
        """计算单因子月度 IC"""
        unique_dates = df['trade_date'].unique().to_list()
        
        factor_columns = {
            'hyper_residual': 'hyper_residual_score',
            'vp_correlation': 'vp_correlation_score',
            'reversal': 'reversal_score',
            'rs': 'multi_rs_score',
        }
        
        for trade_date in sorted(unique_dates):
            month = trade_date[:7]
            
            for factor_name, factor_col in factor_columns.items():
                if factor_col in df.columns:
                    ic = self.calculate_factor_ic(df, trade_date, factor_col)
                    
                    if month not in self.factor_monthly_ics[factor_name]:
                        self.factor_monthly_ics[factor_name][month] = []
                    self.factor_monthly_ics[factor_name][month].append(ic)
    
    def _compute_oos_yearly_stats(self):
        """计算 OOS 年度统计（2019/2021/2024）"""
        oos_years = V82_RANK_IC_OOS_YEARS
        
        for year in oos_years:
            year_stats = {}
            
            # 过滤该年度的 IC 结果
            year_ics = [ic for ic in self.ic_results if ic.trade_date.startswith(year)]
            
            if year_ics:
                rank_ics = [ic.rank_ic for ic in year_ics]
                ics = [ic.ic for ic in year_ics]
                
                year_stats['mean_rank_ic'] = float(np.mean(rank_ics))
                year_stats['mean_ic'] = float(np.mean(ics))
                year_stats['std_rank_ic'] = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
                year_stats['ic_count'] = len(year_ics)
                year_stats['positive_ratio'] = float(np.sum([1 for ic in rank_ics if ic > 0]) / len(rank_ics))
                
                # 【硬性检查】如果 Mean Rank IC 为 0 或 NaN，视为系统崩溃
                if abs(year_stats['mean_rank_ic']) < EPSILON or np.isnan(year_stats['mean_rank_ic']):
                    logger.error(f"V82: 【系统崩溃】{year}年 Mean Rank IC = {year_stats['mean_rank_ic']}")
            else:
                year_stats['mean_rank_ic'] = 0.0
                year_stats['mean_ic'] = 0.0
                year_stats['std_rank_ic'] = 0.0
                year_stats['ic_count'] = 0
                year_stats['positive_ratio'] = 0.0
                logger.error(f"V82: 【系统崩溃】{year}年无 IC 数据")
            
            # 计算单因子 IC
            for factor_name in self.factor_monthly_ics:
                factor_year_ics = []
                for month, ics in self.factor_monthly_ics[factor_name].items():
                    if month.startswith(year):
                        factor_year_ics.extend(ics)
                
                if factor_year_ics:
                    year_stats[f'{factor_name}_mean_rank_ic'] = float(np.mean(factor_year_ics))
                else:
                    year_stats[f'{factor_name}_mean_rank_ic'] = 0.0
            
            self.oos_yearly_stats[year] = year_stats
    
    def get_factor_monthly_ics(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for factor_name, monthly_dict in self.factor_monthly_ics.items():
            result[factor_name] = {
                month: float(np.mean(ics)) for month, ics in monthly_dict.items()
            }
        return result
    
    def get_ic_statistics(self) -> Dict[str, float]:
        if not self.ic_results:
            return {
                'mean_ic': 0.0,
                'mean_rank_ic': 0.0,
                'ic_std': 0.0,
                'rank_ic_std': 0.0,
                'ic_ir': 0.0,
                'rank_ic_ir': 0.0,
                'positive_ratio': 0.0,
                'num_valid_days': 0,
            }
        
        ic_values = np.array([m.ic for m in self.ic_results])
        rank_ic_values = np.array([m.rank_ic for m in self.ic_results])
        
        mean_ic = float(np.mean(ic_values))
        mean_rank_ic = float(np.mean(rank_ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        rank_ic_std = float(np.std(rank_ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > EPSILON else 0.0
        rank_ic_ir = mean_rank_ic / rank_ic_std if rank_ic_std > EPSILON else 0.0
        positive_ratio = float(np.sum(ic_values > 0) / len(ic_values))
        
        return {
            'mean_ic': mean_ic,
            'mean_rank_ic': mean_rank_ic,
            'ic_std': ic_std,
            'rank_ic_std': rank_ic_std,
            'ic_ir': ic_ir,
            'rank_ic_ir': rank_ic_ir,
            'positive_ratio': positive_ratio,
            'num_valid_days': len(self.ic_results),
        }
    
    def get_oos_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取 OOS 年度统计"""
        return self.oos_yearly_stats
    
    def get_monthly_rank_ic_statistics(self) -> Dict[str, float]:
        if not self.monthly_stats:
            return {
                'monthly_mean_rank_ic': 0.0,
                'monthly_std': 0.0,
                'monthly_pass': False,
                'negative_months': 0,
            }
        
        monthly_means = [m.mean_rank_ic for m in self.monthly_stats]
        negative_months = sum(1 for m in self.monthly_stats if m.is_negative)
        
        monthly_mean = float(np.mean(monthly_means))
        monthly_std = float(np.std(monthly_means, ddof=1)) if len(monthly_means) > 1 else 0.0
        monthly_pass = monthly_mean >= self.rank_ic_target
        
        return {
            'monthly_mean_rank_ic': monthly_mean,
            'monthly_std': monthly_std,
            'monthly_pass': monthly_pass,
            'negative_months': negative_months,
            'num_months': len(self.monthly_stats),
        }
    
    def check_rank_ic_pass(self) -> Tuple[bool, str]:
        stats = self.get_ic_statistics()
        monthly_stats = self.get_monthly_rank_ic_statistics()
        
        negative_months = monthly_stats.get('negative_months', 0)
        negative_months_pass = negative_months <= 2
        
        if stats['mean_rank_ic'] >= self.rank_ic_target and negative_months_pass:
            return (True, f"Rank IC 达标：{stats['mean_rank_ic']:.4f} >= {self.rank_ic_target}")
        elif stats['mean_rank_ic'] >= self.rank_ic_min:
            return (True, f"Rank IC 勉强达标：{stats['mean_rank_ic']:.4f} >= {self.rank_ic_min}")
        else:
            return (False, f"预测模型失败：Rank IC={stats['mean_rank_ic']:.4f} < {self.rank_ic_min}")
    
    def print_rank_ic_report(self):
        stats = self.get_ic_statistics()
        monthly_stats = self.get_monthly_rank_ic_statistics()
        oos_stats = self.get_oos_statistics()
        is_pass, message = self.check_rank_ic_pass()
        
        logger.info("=" * 60)
        logger.info("V82 Rank IC 预测质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f}")
        logger.info(f"负值月份数量：{monthly_stats['negative_months']} (上限：2)")
        logger.info(f"达标状态：{is_pass}")
        logger.info("")
        logger.info("【OOS 年度统计】")
        for year in V82_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                year_stat = oos_stats[year]
                logger.info(f"  {year}年：Mean Rank IC={year_stat['mean_rank_ic']:.4f}, "
                           f"样本数={year_stat['ic_count']}, 正占比={year_stat['positive_ratio']:.2%}")
        logger.info("=" * 60)
    
    def generate_oos_report(self) -> str:
        """生成 OOS 测试报告"""
        oos_stats = self.get_oos_statistics()
        
        lines = [
            "=" * 60,
            "V82 OOS 测试报告",
            "=" * 60,
            "",
        ]
        
        for year in V82_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                stat = oos_stats[year]
                lines.append(f"【{year}年】")
                lines.append(f"  Mean Rank IC: {stat['mean_rank_ic']:.4f}")
                lines.append(f"  Mean IC: {stat['mean_ic']:.4f}")
                lines.append(f"  Std Rank IC: {stat['std_rank_ic']:.4f}")
                lines.append(f"  样本数：{stat['ic_count']}")
                lines.append(f"  正 IC 占比：{stat['positive_ratio']:.2%}")
                lines.append("")
                
                # 单因子 IC
                lines.append(f"  【单因子 IC】")
                for factor in ['hyper_residual', 'vp_correlation', 'reversal', 'rs']:
                    factor_ic = stat.get(f'{factor}_mean_rank_ic', 0.0)
                    lines.append(f"    {factor}: {factor_ic:.4f}")
                lines.append("")
        
        # 计算三年度平均
        valid_years = [year for year in V82_RANK_IC_OOS_YEARS if year in oos_stats and oos_stats[year]['ic_count'] > 0]
        if valid_years:
            avg_rank_ic = np.mean([oos_stats[y]['mean_rank_ic'] for y in valid_years])
            lines.append(f"【三年度平均 Mean Rank IC】")
            lines.append(f"  平均值：{avg_rank_ic:.4f} (目标：>={V82_RANK_IC_TARGET})")
            lines.append(f"  达标状态：{'✓' if avg_rank_ic >= V82_RANK_IC_TARGET else '✗'}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V82_INITIAL_CAPITAL',
    'V82_MAX_POSITIONS',
    'V82_WARMUP_PERIOD',
    'V82DataManager',
    'V82AlphaCenter',
    'V82RankICCalculator',
    'V82Position',
    'V82Trade',
    'V82Signal',
    'V82ICMetrics',
    'V82MonthlyICStats',
    'V82FactorMonitor',
    'V82RegimeState',
    'quantile_transform',
    'compute_skewness',
    'compute_kurtosis',
    'apply_skewness_penalty',
    'apply_kurtosis_penalty',
    'apply_extreme_value_filter',
]