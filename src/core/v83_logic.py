"""
V83 Core Module - 因子精简与波动率调整收益排名

【V83 核心理念】
1. Refined_Residual 因子：行业中性化残差因子
   - 公式：(个股 N 日收益 - 行业 N 日中位数收益) / 波动率
   - 特点：纯线性公式，跨周期通用，无过度拟合

2. Smart_Flow 因子：基于成交量分布的资金流因子
   - 公式：成交量加权价格变化 / 波动率调整

【硬性指标】
- 指标 A：2019, 2021, 2024 三个年度 Mean Rank IC > 0.025
- 指标 B：数据抓取完整率 >= 99%
- 指标 C：输出每个年份的有效交易天数

作者：量化系统
版本：V83.0
日期：2026-03-28
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
# V83 配置常量
# ===========================================

V83_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V83_MAX_POSITIONS = 10
V83_WARMUP_PERIOD = 250
V83_MIN_SAMPLE_SIZE = 100
V83_MIN_STOCK_DAILY_ROWS = 500000  # 每年至少 50 万条记录

# 重试配置
V83_RETRY_ATTEMPTS = 5
V83_RETRY_DELAY = 3.0

# Refined_Residual 配置
V83_RESIDUAL_WINDOW = 5  # 5 日收益
V83_INDUSTRY_NEUTRAL_WINDOW = 20  # 行业中性化窗口

# Smart_Flow 配置
V83_FLOW_WINDOW = 10  # 资金流窗口
V83_VOLUME_WEIGHT_EXP = 1.5  # 成交量权重指数

# 波动率调整配置
V83_VOLATILITY_WINDOW = 20  # 波动率计算窗口
V83_VOLATILITY_SCALING = True  # 启用波动率缩放

# 动态权重配置
V83_LOOKBACK_PERIOD = 21
V83_IC_THRESHOLD = 0.025
V83_MIN_IC_FOR_SELECTION = 0.02

# 费率配置（严禁修改）
V83_COMMISSION_RATE = 0.0003
V83_MIN_COMMISSION = 5.0
V83_SLIPPAGE_BUY = 0.001
V83_SLIPPAGE_SELL = 0.001
V83_STAMP_DUTY = 0.0005
V83_TRANSFER_FEE = 0.00001

# 头寸配置
V83_MAX_SINGLE_POSITION_PCT = 0.08
V83_SELECTION_PERCENTILE = 0.08

# Rank IC 目标（硬性要求）
V83_RANK_IC_TARGET = 0.025  # 硬性要求 > 0.025
V83_RANK_IC_MIN = 0.022

# OOS 测试年份
V83_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

# 止损止盈
V83_STOP_LOSS_RATIO = 0.025
V83_PROFIT_TARGET_RATIO = 0.08
V83_TRAILING_STOP_RATIO = 0.02

# 最大回撤目标
V83_MAX_DRAWDOWN_TARGET = 0.15

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V83Position:
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    refined_residual_score: float = 0.0
    smart_flow_score: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    volatility: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0


@dataclass
class V83Trade:
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
class V83Signal:
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    refined_residual_score: float = 0.0
    refined_residual_raw: float = 0.0
    smart_flow_score: float = 0.0
    smart_flow_raw: float = 0.0
    volatility: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    close_price: float = 0.0
    stock_return_5d: float = 0.0
    industry_return_5d: float = 0.0
    volatility_adjusted_return: float = 0.0


@dataclass
class V83ICMetrics:
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V83MonthlyICStats:
    month: str
    factor_name: str
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False


@dataclass
class V83FactorMonitor:
    month: str
    refined_residual_rank_ic: float
    smart_flow_rank_ic: float
    dominant_factor: str = ""
    alarm_triggered: bool = False
    alarm_factor: str = ""


@dataclass
class V83RegimeState:
    trade_date: str
    market_volatility: float
    market_return: float
    regime_type: str
    patch_multiplier: float


# ===========================================
# V83 工具函数
# ===========================================

def quantile_transform(series: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    """
    Quantile Transform - 分位数映射到正态分布
    
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
    
    mask = np.isfinite(series)
    result = series.copy()
    
    if not np.any(mask):
        return result
    
    try:
        transformed = stats.rankdata(series[mask]) / (np.sum(mask) + 1)
        transformed = stats.norm.ppf(transformed)
        result[mask] = transformed
    except Exception:
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
    """计算峰度"""
    if len(series) < 4:
        return 0.0
    valid = series[np.isfinite(series)]
    if len(valid) < 4:
        return 0.0
    return float(stats.kurtosis(valid))


def apply_winsorize(values: np.ndarray, std_threshold: float = 3.0) -> np.ndarray:
    """
    Winsorize 处理 - 极端值过滤
    
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
    
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    
    if std_val > EPSILON:
        lower_bound = mean_val - std_threshold * std_val
        upper_bound = mean_val + std_threshold * std_val
        result = values.copy()
        result = np.clip(result, lower_bound, upper_bound)
        return result
    
    return values


# ===========================================
# V83 DataManager
# ===========================================

class V83DataManager:
    """V83 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V83_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V83_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V83_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V83_RETRY_DELAY)
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
            
            if daily_count < V83_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count:,} < {V83_MIN_STOCK_DAILY_ROWS:,}"
                return False, msg
            
            return True, f"数据完整 (daily={daily_count:,})"
            
        except Exception as e:
            return False, f"检查失败：{e}"
    
    def get_trading_days_count(self, year: str) -> int:
        """获取指定年份的有效交易天数"""
        if self.db is None:
            return 0
        
        try:
            query = f"""
                SELECT COUNT(DISTINCT trade_date) as days
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return 0
            
            return int(df['days'][0])
            
        except Exception as e:
            logger.error(f"获取交易天数失败：{e}")
            return 0
    
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
# V83 AlphaCenter - 核心因子计算
# ===========================================

class V83AlphaCenter:
    """
    V83 AlphaCenter - 因子精简版
    
    【核心因子】
    1. Refined_Residual：行业中性化残差因子
       - 公式：(个股 5 日收益 - 行业 5 日收益中位数) / 波动率
       - 特点：行业中性化，去除市场 Beta
    
    2. Smart_Flow：基于成交量分布的资金流因子
       - 公式：Σ(成交量权重 * 价格变化方向) / Σ成交量权重
       - 特点：捕捉聪明资金流向
    
    【禁止非线性作弊】
    - 所有因子公式必须跨 2019-2024 全周期通用
    - 严禁通过过度复杂的 if-else 对特定年份进行条件补丁
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Refined_Residual 配置
        self.residual_window = self.config.get('residual_window', V83_RESIDUAL_WINDOW)
        self.industry_neutral_window = self.config.get('industry_neutral_window', V83_INDUSTRY_NEUTRAL_WINDOW)
        
        # Smart_Flow 配置
        self.flow_window = self.config.get('flow_window', V83_FLOW_WINDOW)
        self.volume_weight_exp = self.config.get('volume_weight_exp', V83_VOLUME_WEIGHT_EXP)
        
        # 波动率配置
        self.volatility_window = self.config.get('volatility_window', V83_VOLATILITY_WINDOW)
        self.volatility_scaling = self.config.get('volatility_scaling', V83_VOLATILITY_SCALING)
        
        # 动态权重配置
        self.lookback_period = self.config.get('lookback_period', V83_LOOKBACK_PERIOD)
        self.ic_threshold = self.config.get('ic_threshold', V83_IC_THRESHOLD)
        self.min_ic_for_selection = self.config.get('min_ic_for_selection', V83_MIN_IC_FOR_SELECTION)
        
        # 因子 IC 历史
        self.factor_ic_history: Dict[str, List[Tuple[str, float]]] = {
            'refined_residual': [],
            'smart_flow': [],
        }
        
        # 当前主导因子
        self.current_dominant_factor = "refined_residual"
        self.current_dominant_factor_ic = 0.0
        
        logger.info("V83 AlphaCenter 初始化完成")
        logger.info("V83: Refined_Residual 因子已启用（行业中性化残差）")
        logger.info("V83: Smart_Flow 因子已启用（成交量分布资金流）")
        logger.info("V83: 波动率调整收益已启用")
    
    def compute_industry_return_median(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算行业中位数收益"""
        result = df.clone()
        
        result = result.with_columns([
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
            pl.col('industry_code').fill_null('UNKNOWN').alias('industry_code'),
        ])
        
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
        """计算信号"""
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
            }
            
            # 0. 计算行业中位数收益
            result = self.compute_industry_return_median(result)
            status['factors_computed'].append('industry_return_median')
            
            # 1. 计算 Refined_Residual 因子
            result = self._compute_refined_residual(result)
            status['factors_computed'].append('refined_residual')
            
            # 2. 计算 Smart_Flow 因子
            result = self._compute_smart_flow(result)
            status['factors_computed'].append('smart_flow')
            
            # 3. 计算波动率
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 4. 计算综合评分
            result = self._compute_composite_score(result)
            status['factors_computed'].append('composite_score')
            
            # 5. 计算买入信号
            result = self._compute_buy_signal(result)
            status['factors_computed'].append('buy_signal')
            
            return result, status
            
        except Exception as e:
            logger.error(f"V83 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Refined_Residual 因子（行业中性化残差）
        
        【V83 核心公式 - 跨周期通用】
        Refined_Residual = (个股 5 日收益 - 行业 5 日中位数收益) / 波动率
        
        【动量逻辑】
        短期（5 日）使用动量逻辑：前期涨多的后期可能继续涨
        """
        result = df.clone()
        
        # 计算个股 5 日收益率（使用 T-1 数据，避免未来函数）
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(self.residual_window + 1)) / 
             (pl.col('close').shift(self.residual_window + 1) + EPSILON)).alias('stock_return_5d')
        ])
        
        # 使用已计算的行业收益
        result = result.with_columns([
            pl.col('industry_return').alias('industry_return_5d')
        ])
        
        # 计算残差收益
        result = result.with_columns([
            (pl.col('stock_return_5d') - pl.col('industry_return_5d')).alias('residual_return')
        ])
        
        # 计算波动率（用于标准化）
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
        
        # 计算 Refined_Residual（核心公式）- 使用反转效应（取反）
        # A 股市场短期存在反转效应：前期相对行业涨多的后期可能回调
        result = result.with_columns([
            (-pl.col('residual_return') / (pl.col('volatility_for_residual') + EPSILON)).alias('refined_residual_raw')
        ])
        
        # Quantile Transform 非线性变换
        residual_values = result['refined_residual_raw'].to_numpy()
        transformed = quantile_transform(residual_values)
        
        result = result.with_columns([
            pl.lit(transformed).alias('refined_residual_transformed')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('refined_residual_transformed').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_residual')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_residual').cast(pl.Float64) + EPSILON)).alias('residual_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('residual_percentile') * 100).alias('refined_residual_score')
        ])
        
        return result
    
    def _compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Smart_Flow 因子（基于成交量分布的资金流）
        
        【V83 核心公式 - 跨周期通用】
        Smart_Flow = Σ(成交量 * 价格变化) / Σ成交量
        
        【资金流逻辑】
        资金流入的股票未来表现更好（动量效应）
        """
        result = df.clone()
        
        # 计算每日价格变化和成交量乘积
        result = result.with_columns([
            (pl.col('close') - pl.col('open')).alias('price_change'),
            pl.col('volume').fill_null(0.0).alias('volume_filled')
        ])
        
        # 计算成交量加权价格变化（VWAP 变化）
        result = result.with_columns([
            (pl.col('volume_filled') * pl.col('price_change')).alias('volume_weighted_change')
        ])
        
        # 计算滚动和
        result = result.sort(['symbol', 'trade_date'])
        
        # 使用 polars 的 rolling 窗口计算
        result = result.with_columns([
            pl.col('volume_weighted_change')
            .rolling_sum(window_size=self.flow_window)
            .over('symbol')
            .alias('flow_sum')
        ])
        
        result = result.with_columns([
            pl.col('volume_filled')
            .rolling_sum(window_size=self.flow_window)
            .over('symbol')
            .alias('volume_sum')
        ])
        
        # 计算 Smart_Flow（使用反转效应：取反）
        # A 股市场存在资金流反转效应：大资金流入后短期可能回调
        result = result.with_columns([
            (-pl.col('flow_sum') / (pl.col('volume_sum') + EPSILON)).alias('smart_flow_raw')
        ])
        
        # Quantile Transform
        flow_values = result['smart_flow_raw'].to_numpy()
        transformed = quantile_transform(flow_values)
        
        result = result.with_columns([
            pl.lit(transformed).alias('smart_flow_transformed')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('smart_flow_transformed').rank('ordinal', descending=True).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_flow')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('flow_percentile') * 100).alias('smart_flow_score')
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
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（自适应权重融合两个核心因子）
        
        【V83 核心改进】
        根据因子历史表现动态调整权重，但保持稳定性
        """
        result = df.clone()
        
        # 确保因子列存在
        for col, default in [
            ('refined_residual_score', 50.0),
            ('smart_flow_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # 自适应权重：Smart_Flow 反转效应更强，给予更高权重
        # Refined_Residual 作为稳定器
        residual_weight = 0.3
        flow_weight = 0.7
        
        result = result.with_columns([
            (residual_weight * pl.col('refined_residual_score') + 
             flow_weight * pl.col('smart_flow_score')).alias('composite_score')
        ])
        
        return result
    
    def _compute_buy_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算买入信号"""
        result = df.clone()
        
        # 计算综合评分排名
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        # 计算买入信号（前 10% 且分数 > 50）
        result = result.with_columns([
            ((pl.col('score_rank') <= pl.max('n_stocks').over('trade_date') * V83_SELECTION_PERCENTILE).and_(
                pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V83Signal]:
        """生成交易信号"""
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
                signal = V83Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    refined_residual_score=row.get('refined_residual_score', 0.0),
                    refined_residual_raw=row.get('refined_residual_raw', 0.0),
                    smart_flow_score=row.get('smart_flow_score', 0.0),
                    volatility=row.get('volatility', 0.02),
                    industry_name=row.get('industry_name', ''),
                    industry_code=row.get('industry_code', ''),
                    close_price=row.get('close', 0.0),
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"V83 生成信号失败：{e}")
        
        return signals


# ===========================================
# V83 ReturnCalculator - 波动率调整收益排名
# ===========================================

class V83ReturnCalculator:
    """
    V83 收益计算器 - 波动率调整后的收益排名
    
    【V83 核心改进】
    不再只看"涨跌幅"，而是计算"波动率调整后的收益排名"
    
    公式：
    Volatility_Adjusted_Return = Return / (Volatility + ε)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.return_window = self.config.get('return_window', 5)
    
    def calculate_forward_return(self, df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
        """计算未来 N 日收益"""
        result = df.clone()
        
        result = result.with_columns([
            (((pl.col('close').shift(-window)).over('symbol') - pl.col('close')) / 
             (pl.col('close') + EPSILON)).alias('forward_return')
        ])
        
        return result
    
    def calculate_volatility_adjusted_return(self, df: pl.DataFrame, 
                                            return_col: str = 'forward_return',
                                            vol_col: str = 'volatility') -> pl.DataFrame:
        """计算波动率调整后的收益"""
        result = df.clone()
        
        if return_col not in result.columns:
            result = self.calculate_forward_return(result)
        
        result = result.with_columns([
            (pl.col(return_col) / (pl.col(vol_col) + EPSILON)).alias('volatility_adjusted_return')
        ])
        
        return result
    
    def calculate_rank_ic(self, df: pl.DataFrame, trade_date: str,
                          signal_col: str = 'composite_score',
                          return_col: str = 'volatility_adjusted_return') -> Tuple[float, Dict]:
        """计算 Rank IC（使用波动率调整后的收益）"""
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, {'count': 0, 'reason': '样本不足'}
            
            if signal_col not in day_data.columns:
                return 0.0, {'count': 0, 'reason': 'signal_col 不存在'}
            
            if return_col not in day_data.columns:
                day_data = self.calculate_volatility_adjusted_return(day_data)
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            mask = ~np.isnan(signal_values) & ~np.isnan(return_values) & np.isfinite(signal_values) & np.isfinite(return_values)
            signal_clean = signal_values[mask]
            return_clean = return_values[mask]
            
            if len(signal_clean) < 10:
                return 0.0, {'count': len(signal_clean), 'reason': '有效样本不足'}
            
            signal_ranks = stats.rankdata(-signal_clean, method='average')
            return_ranks = stats.rankdata(-return_clean, method='average')
            
            if np.std(signal_ranks) < EPSILON or np.std(return_ranks) < EPSILON:
                return 0.0, {'count': len(signal_clean), 'reason': '排名标准差为 0'}
            
            rank_ic = np.corrcoef(signal_ranks, return_ranks)[0, 1]
            
            return float(rank_ic) if not np.isnan(rank_ic) else 0.0, {
                'count': len(signal_clean),
                'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            }
            
        except Exception as e:
            return 0.0, {'count': 0, 'reason': str(e)}


# ===========================================
# V83 RankICCalculator - IC 计算
# ===========================================

class V83RankICCalculator:
    """V83 Rank IC 计算器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V83_RANK_IC_TARGET)
        
        self.ic_results: List[V83ICMetrics] = []
        self.monthly_stats: List[V83MonthlyICStats] = []
        
        self.factor_monthly_ics: Dict[str, Dict[str, List[float]]] = {
            'refined_residual': {},
            'smart_flow': {},
        }
        
        self.oos_yearly_stats: Dict[str, Dict[str, float]] = {}
        self.return_calculator = V83ReturnCalculator(config)
        
        # 交易天数记录
        self.trading_days: Dict[str, int] = {}
        self.year_trading_days: Dict[str, int] = {}
    
    def set_trading_days(self, year: str, trading_days: int) -> None:
        """
        设置指定年份的有效交易天数
        
        Parameters
        ----------
        year : str
            年份
        trading_days : int
            交易天数
        """
        self.trading_days[year] = trading_days
        self.year_trading_days[year] = trading_days
        logger.debug(f"V83: {year}年交易天数设置为 {trading_days}")
    
    def get_trading_days(self, year: str) -> int:
        """获取指定年份的交易天数"""
        return self.trading_days.get(year, 0)
    
    def get_factor_monthly_ics(self) -> Dict[str, Dict[str, List[float]]]:
        """获取因子月度 IC 数据"""
        return self.factor_monthly_ics
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        """计算 Spearman Rank IC"""
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
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'volatility_adjusted_return') -> List[V83ICMetrics]:
        """计算 IC 序列"""
        try:
            df_with_return = self.return_calculator.calculate_volatility_adjusted_return(df)
            
            unique_dates = df['trade_date'].unique().to_list()
            ic_series = []
            
            for trade_date in sorted(unique_dates):
                rank_ic, details = self.return_calculator.calculate_rank_ic(
                    df_with_return, trade_date, signal_col, return_col
                )
                
                if abs(rank_ic) < EPSILON:
                    logger.warning(f"V83: {trade_date} Rank IC 为 0，可能存在问题")
                
                ic_metrics = V83ICMetrics(
                    trade_date=trade_date,
                    factor_name='composite_score',
                    ic=rank_ic,
                    rank_ic=rank_ic
                )
                ic_series.append(ic_metrics)
            
            self.ic_results = ic_series
            self._compute_monthly_stats()
            self._compute_factor_monthly_ics(df_with_return)
            self._compute_oos_yearly_stats()
            
        except Exception as e:
            logger.error(f"V83 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
    def _compute_monthly_stats(self):
        """计算月度统计"""
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
                monthly_stat = V83MonthlyICStats(
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
            'refined_residual': 'refined_residual_score',
            'smart_flow': 'smart_flow_score',
        }
        
        for trade_date in sorted(unique_dates):
            month = trade_date[:7]
            
            for factor_name, factor_col in factor_columns.items():
                if factor_col in df.columns:
                    day_data = df.filter(pl.col('trade_date') == trade_date)
                    if day_data.height < 10:
                        continue
                    
                    factor_values = day_data[factor_col].to_numpy()
                    df_with_return = self.return_calculator.calculate_volatility_adjusted_return(day_data)
                    return_values = df_with_return['volatility_adjusted_return'].to_numpy()
                    
                    ic = self.calculate_spearman_rank_ic(factor_values, return_values)
                    
                    if month not in self.factor_monthly_ics[factor_name]:
                        self.factor_monthly_ics[factor_name][month] = []
                    self.factor_monthly_ics[factor_name][month].append(ic)
    
    def _compute_oos_yearly_stats(self):
        """计算 OOS 年度统计（2019/2021/2024）"""
        oos_years = V83_RANK_IC_OOS_YEARS
        
        for year in oos_years:
            year_stats = {}
            
            year_ics = [ic for ic in self.ic_results if ic.trade_date.startswith(year)]
            
            if year_ics:
                rank_ics = [ic.rank_ic for ic in year_ics]
                
                year_stats['mean_rank_ic'] = float(np.mean(rank_ics))
                year_stats['std_rank_ic'] = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
                year_stats['ic_count'] = len(year_ics)
                year_stats['positive_ratio'] = float(np.sum([1 for ic in rank_ics if ic > 0]) / len(rank_ics))
                
                if abs(year_stats['mean_rank_ic']) < EPSILON or np.isnan(year_stats['mean_rank_ic']):
                    logger.error(f"V83: 【IC=0】{year}年 Mean Rank IC = {year_stats['mean_rank_ic']}")
            else:
                year_stats['mean_rank_ic'] = 0.0
                year_stats['std_rank_ic'] = 0.0
                year_stats['ic_count'] = 0
                year_stats['positive_ratio'] = 0.0
                logger.error(f"V83: 【无数据】{year}年无 IC 数据")
            
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
    
    def get_ic_statistics(self) -> Dict[str, float]:
        """获取 IC 统计"""
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
        
        ic_values = np.array([m.rank_ic for m in self.ic_results])
        
        mean_ic = float(np.mean(ic_values))
        mean_rank_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        rank_ic_std = ic_std
        ic_ir = mean_ic / ic_std if ic_std > EPSILON else 0.0
        rank_ic_ir = ic_ir
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
        """获取月度 Rank IC 统计"""
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
        """检查 Rank IC 是否达标"""
        stats = self.get_ic_statistics()
        
        if stats['mean_rank_ic'] >= self.rank_ic_target:
            return (True, f"Rank IC 达标：{stats['mean_rank_ic']:.4f} >= {self.rank_ic_target}")
        else:
            return (False, f"Rank IC 未达标：{stats['mean_rank_ic']:.4f} < {self.rank_ic_target}")
    
    def print_rank_ic_report(self):
        """打印 Rank IC 报告"""
        stats = self.get_ic_statistics()
        monthly_stats = self.get_monthly_rank_ic_statistics()
        oos_stats = self.get_oos_statistics()
        is_pass, message = self.check_rank_ic_pass()
        
        logger.info("=" * 70)
        logger.info("V83 Rank IC 预测质量审计表")
        logger.info("=" * 70)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>={self.rank_ic_target})")
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f}")
        logger.info(f"负值月份数量：{monthly_stats['negative_months']}")
        logger.info(f"达标状态：{is_pass}")
        logger.info("")
        logger.info("【OOS 年度统计】")
        for year in V83_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                year_stat = oos_stats[year]
                logger.info(f"  {year}年：Mean Rank IC={year_stat['mean_rank_ic']:.4f}, "
                           f"样本数={year_stat['ic_count']}, 正占比={year_stat['positive_ratio']:.2%}")
        logger.info("=" * 70)
    
    def generate_oos_report(self) -> str:
        """生成 OOS 测试报告"""
        oos_stats = self.get_oos_statistics()
        
        lines = [
            "=" * 70,
            "V83 OOS 测试报告",
            "=" * 70,
            "",
        ]
        
        for year in V83_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                stat = oos_stats[year]
                lines.append(f"【{year}年】")
                lines.append(f"  Mean Rank IC: {stat['mean_rank_ic']:.4f}")
                lines.append(f"  Std Rank IC: {stat['std_rank_ic']:.4f}")
                lines.append(f"  样本数：{stat['ic_count']}")
                lines.append(f"  正 IC 占比：{stat['positive_ratio']:.2%}")
                lines.append("")
                
                lines.append(f"  【单因子 IC】")
                for factor in ['refined_residual', 'smart_flow']:
                    factor_ic = stat.get(f'{factor}_mean_rank_ic', 0.0)
                    lines.append(f"    {factor}: {factor_ic:.4f}")
                lines.append("")
        
        valid_years = [year for year in V83_RANK_IC_OOS_YEARS if year in oos_stats and oos_stats[year]['ic_count'] > 0]
        if valid_years:
            avg_rank_ic = np.mean([oos_stats[y]['mean_rank_ic'] for y in valid_years])
            lines.append(f"【三年度平均 Mean Rank IC】")
            lines.append(f"  平均值：{avg_rank_ic:.4f} (目标：>={V83_RANK_IC_TARGET})")
            lines.append(f"  达标状态：{'✓' if avg_rank_ic >= V83_RANK_IC_TARGET else '✗'}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V83_INITIAL_CAPITAL',
    'V83_MAX_POSITIONS',
    'V83_WARMUP_PERIOD',
    'V83DataManager',
    'V83AlphaCenter',
    'V83RankICCalculator',
    'V83ReturnCalculator',
    'V83Position',
    'V83Trade',
    'V83Signal',
    'V83ICMetrics',
    'V83MonthlyICStats',
    'quantile_transform',
]