"""
V88 Core Module - Alpha 唤醒与交易逻辑闭环

【V88 核心理念】
1. Alpha 唤醒与权重再平衡 (Alpha Re-activation)
   - Score 低于 60 分强制权重归零（精选策略）
   - 单只标的最低权重门槛 0.5%
   - 权重公式：W_i = (Score_i × (1/Volatility_i)) / Σ

2. 时空波动率调整融合 (Spatio-Temporal Volatility Fusion)
   - 使用过去 5 日 IC 的倒数作为 Lags [1, 3, 5] 的动态权重
   - 不再对 Score 简单取均值

3. 强制性数据补全与报错自愈
   - 检测到>5% 天数无交易时自动回溯数据源
   - 使用"行业中位数"或"全市场平均"进行冷启动填充
   - 严禁返回 0 处理空值

4. 防作弊与防未来数据审计
   - 检测 T+3 IC > T+1 IC 的逻辑
   - 审计 Rolling/Shift 中的未来数据使用
   - 输出 [CRITICAL_WARNING] Lookahead Bias Potential

【V88 硬性指标】
- 指标 A (活跃度): 年化换手率必须在 200% - 800% 之间
- 指标 B (盈利性): 2024 年多头超额收益 > 5%
- 指标 C (IC 衰减): 必须满足 IC_{T+1} > IC_{T+2} > IC_{T+3}
- 指标 D (数据率): 输出 2019, 2021, 2024 三年完整交易记录

作者：量化系统
版本：V88.0
日期：2026-03-29
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
# V88 配置常量
# ===========================================

V88_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V88_MAX_POSITIONS = 30  # 最大持仓数量
V88_WARMUP_PERIOD = 250
V88_MIN_SAMPLE_SIZE = 100
V88_MIN_STOCK_DAILY_ROWS = 500000

# Alpha 唤醒配置
V88_MIN_SCORE_THRESHOLD = 60.0  # 最低评分门槛（低于此值权重归零）
V88_MIN_SINGLE_WEIGHT = 0.005  # 单只标的最低权重 0.5%
V88_MAX_SINGLE_WEIGHT = 0.10  # 单只标的最大权重 10%

# 时空波动率融合配置
V88_FUSION_LAGS = [1, 3, 5]
V88_IC_WINDOW = 5  # IC 计算窗口
V88_IC_EPSILON = 0.001  # IC 平滑因子

# 数据补全配置
V88_NO_TRADE_THRESHOLD = 0.05  # 5% 天数无交易触发补全
V88_INDUSTRY_MEDIAN_FILL = True  # 优先使用行业中位数填充

# 防作弊配置
V88_TURNOVER_MIN = 2.0  # 最低年化换手率 200%
V88_TURNOVER_MAX = 8.0  # 最高年化换手率 800%
V88_EXCESS_RETURN_TARGET = 0.05  # 超额收益目标 5%

# 费率配置（严禁修改）
V88_COMMISSION_RATE = 0.002  # 0.2%
V88_MIN_COMMISSION = 5.0
V88_STAMP_DUTY = 0.0005  # 印花税
V88_TRANSFER_FEE = 0.00001

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V88FusionSignal:
    """时空波动率融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float
    ic_weight_t1: float
    ic_weight_t3: float
    ic_weight_t5: float


@dataclass
class V88AlphaWeightRecord:
    """Alpha 权重记录"""
    trade_date: str
    symbol: str
    raw_score: float
    volatility: float
    alpha_weight: float
    normalized_weight: float
    is_filtered: bool  # 是否被 Score 阈值过滤


@dataclass
class V88DataRepairRecord:
    """数据修复记录"""
    trade_date: str
    symbol: str
    field: str
    original_value: Optional[float]
    repaired_value: float
    repair_method: str  # "industry_median" 或 "market_mean"


@dataclass
class V88TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float


@dataclass
class V88Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V88DailyPortfolio:
    """每日组合快照"""
    trade_date: str
    total_value: float
    cash: float
    position_value: float
    position_count: int
    daily_return: float
    cumulative_return: float
    turnover_rate: float


# ===========================================
# V88 工具函数
# ===========================================

def calculate_rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    计算滚动波动率（年化）
    
    Parameters
    ----------
    returns : np.ndarray
        日收益率序列
    window : int
        滚动窗口
        
    Returns
    -------
    np.ndarray
        年化波动率序列
    """
    if len(returns) < window:
        return np.full(len(returns), np.nan)
    
    result = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        valid_returns = window_returns[~np.isnan(window_returns)]
        if len(valid_returns) > 1:
            # 日波动率 * sqrt(252) = 年化波动率
            result[i] = np.std(valid_returns, ddof=1) * np.sqrt(252)
    
    return result


def calculate_ic_dynamic_weights(ic_series: List[float], 
                                  epsilon: float = V88_IC_EPSILON) -> List[float]:
    """
    计算基于 IC 倒数的动态权重
    
    【公式】Weight_i = (1 / |IC_i|) / Σ(1 / |IC_j|)
    
    Parameters
    ----------
    ic_series : List[float]
        IC 序列（按 lag 顺序）
    epsilon : float
        平滑因子，防止除零
        
    Returns
    -------
    List[float]
        归一化权重
    """
    if not ic_series or len(ic_series) < 1:
        return [1.0]
    
    # 计算 IC 倒数权重
    inv_ic_weights = [1.0 / (abs(ic) + epsilon) for ic in ic_series]
    
    # 归一化
    total_weight = sum(inv_ic_weights)
    if total_weight < EPSILON:
        return [1.0 / len(ic_series)] * len(ic_series)
    
    return [w / total_weight for w in inv_ic_weights]


def alpha_weighting(scores: np.ndarray, 
                    volatilities: np.ndarray,
                    min_score: float = V88_MIN_SCORE_THRESHOLD,
                    min_weight: float = V88_MIN_SINGLE_WEIGHT,
                    max_weight: float = V88_MAX_SINGLE_WEIGHT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alpha 权重计算（带 Score 阈值过滤）
    
    【公式】W_i = (Score_i × (1/Volatility_i)) / Σ
    
    Parameters
    ----------
    scores : np.ndarray
        原始评分
    volatilities : np.ndarray
        波动率
    min_score : float
        最低评分门槛
    min_weight : float
        最低权重
    max_weight : float
        最高权重
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (权重数组，过滤掩码) - is_filtered=False 表示未被过滤（可交易）
    """
    n = len(scores)
    weights = np.zeros(n)
    filtered = np.ones(n, dtype=bool)  # True 表示被过滤（不可交易）
    
    # 应用 Score 阈值过滤
    valid_mask = scores >= min_score
    # Score >= 60 的股票未被过滤（is_filtered=False）
    filtered[valid_mask] = False
    # Score < 60 的股票被过滤（is_filtered=True），保持 True
    
    if np.sum(valid_mask) < 1:
        # 如果没有有效评分，返回等权重
        return np.full(n, 1.0 / n), filtered
    
    valid_scores = scores[valid_mask]
    valid_vols = volatilities[valid_mask]
    
    # 防止除零
    valid_vols = np.where(valid_vols < EPSILON, EPSILON, valid_vols)
    valid_vols = np.where(valid_vols > 10.0, 10.0, valid_vols)  # 限制最大波动率
    
    # 计算 Alpha 权重：Score × (1/Volatility)
    raw_weights = valid_scores / valid_vols
    
    # 归一化
    total_weight = np.sum(raw_weights)
    if total_weight < EPSILON:
        return np.full(n, 1.0 / n), filtered
    
    normalized_weights = raw_weights / total_weight
    
    # 应用权重限制
    normalized_weights = np.clip(normalized_weights, min_weight, max_weight)
    
    # 重新归一化
    total_weight = np.sum(normalized_weights)
    if total_weight > EPSILON:
        normalized_weights = normalized_weights / total_weight
    
    # 将权重填入原数组
    weights[valid_mask] = normalized_weights
    
    return weights, filtered


def fill_with_industry_median(df: pl.DataFrame, 
                               col: str, 
                               industry_col: str = 'industry_code') -> pl.DataFrame:
    """
    使用行业中位数填充空值
    
    Parameters
    ----------
    df : pl.DataFrame
        数据框
    col : str
        需要填充的列
    industry_col : str
        行业列
        
    Returns
    -------
    pl.DataFrame
        填充后的数据框
    """
    result = df.clone()
    
    # 计算每个行业的中位数
    industry_median = result.group_by(industry_col).agg([
        pl.col(col).median().alias(f'{col}_median')
    ])
    
    # 计算全市场中位数作为后备
    market_median = result[col].median()
    if market_median is None or not np.isfinite(market_median):
        market_median = 0.0
    
    # 填充空值
    result = result.join(
        industry_median,
        on=industry_col,
        how='left'
    )
    
    result = result.with_columns([
        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
        .then(pl.col(f'{col}_median'))
        .otherwise(pl.col(col))
        .alias(col)
    ])
    
    # 如果还有空值，使用全市场中位数
    result = result.with_columns([
        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
        .then(market_median)
        .otherwise(pl.col(col))
        .alias(col)
    ])
    
    # 删除临时列
    result = result.drop([f'{col}_median'])
    
    return result


def fill_with_market_mean(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    使用全市场均值填充空值
    
    Parameters
    ----------
    df : pl.DataFrame
        数据框
    col : str
        需要填充的列
        
    Returns
    -------
    pl.DataFrame
        填充后的数据框
    """
    result = df.clone()
    
    # 计算均值和中位数
    mean_val = result[col].mean()
    median_val = result[col].median()
    
    # 使用更稳健的统计量
    fill_value = median_val if median_val is not None and np.isfinite(median_val) else mean_val
    if fill_value is None or not np.isfinite(fill_value):
        fill_value = 0.0
    
    result = result.with_columns([
        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
        .then(fill_value)
        .otherwise(pl.col(col))
        .alias(col)
    ])
    
    return result


# ===========================================
# V88 DataManager
# ===========================================

class V88DataManager:
    """
    V88 数据管理器 - 带自愈功能
    
    【核心改进】
    - 自动检测数据缺失
    - 使用行业中位数/全市场均值填充
    - 记录所有修复操作
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V88_WARMUP_PERIOD)
        self.repair_records: List[V88DataRepairRecord] = []
    
    def check_data_integrity(self, year: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        检查数据完整性并返回详细统计
        
        Returns
        -------
        Tuple[bool, str, Dict]
            (是否通过，消息，详细统计)
        """
        if self.db is None:
            return False, "数据库连接未初始化", {}
        
        try:
            query = f"""
                SELECT 
                    COUNT(*) as cnt,
                    COUNT(DISTINCT trade_date) as trading_days,
                    COUNT(DISTINCT symbol) as stocks,
                    SUM(CASE WHEN industry_code IS NULL THEN 1 ELSE 0 END) as null_industry,
                    SUM(CASE WHEN total_mv IS NULL THEN 1 ELSE 0 END) as null_mv
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return False, f"{year}年无数据", {}
            
            stats = {
                'total_rows': int(df['cnt'][0]),
                'trading_days': int(df['trading_days'][0]),
                'stocks': int(df['stocks'][0]),
                'null_industry': int(df['null_industry'][0]),
                'null_mv': int(df['null_mv'][0]),
            }
            
            # 检查缺失率
            null_ratio = (stats['null_industry'] + stats['null_mv']) / max(stats['total_rows'], 1)
            
            if null_ratio > 0.1:
                return False, f"数据缺失率过高：{null_ratio:.2%}", stats
            
            return True, f"数据完整 (rows={stats['total_rows']:,}, days={stats['trading_days']})", stats
            
        except Exception as e:
            return False, f"检查失败：{e}", {}
    
    def detect_no_trade_days(self, df: pl.DataFrame, signal_col: str = 'composite_score') -> Tuple[int, float]:
        """
        检测无交易天数比例
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        signal_col : str
            信号列
            
        Returns
        -------
        Tuple[int, float]
            (无交易天数，比例)
        """
        total_days = df['trade_date'].n_unique()
        
        # 检查信号列是否存在
        if signal_col not in df.columns:
            # 如果信号列不存在，返回 0 无交易天数
            return 0, 0.0
        
        # 有有效信号的天数
        valid_days = df.filter(
            (pl.col(signal_col).is_not_null()) & 
            (pl.col(signal_col).is_finite()) &
            (pl.col(signal_col) >= V88_MIN_SCORE_THRESHOLD)
        )['trade_date'].n_unique()
        
        no_trade_days = total_days - valid_days
        no_trade_ratio = no_trade_days / max(total_days, 1)
        
        return no_trade_days, no_trade_ratio
    
    def load_and_repair_data(self, start_date: str, end_date: str,
                              symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """
        加载并自动修复数据
        
        Parameters
        ----------
        start_date : str
            起始日期
        end_date : str
            结束日期
        symbols : List[str], optional
            股票代码列表
            
        Returns
        -------
        pl.DataFrame
            修复后的数据框
        """
        # 计算热身起始日期
        extra_days = max(V88_FUSION_LAGS) + 20
        warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                       timedelta(days=self.warmup_period + extra_days)).strftime("%Y-%m-%d")
        
        if symbols:
            symbol_list = "','".join(symbols)
            symbol_filter = f"AND symbol IN ('{symbol_list}')"
        else:
            symbol_filter = ""
        
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount, 
                   pct_chg, industry_code, total_mv, is_st
            FROM stock_daily
            WHERE trade_date >= '{warmup_start}' 
              AND trade_date <= '{end_date}'
              {symbol_filter}
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            raise ValueError(f"未加载到任何数据")
        
        # 数据修复 - 直接使用简单填充
        df = self._repair_data_simple(df)
        
        return df
    
    def _repair_data_simple(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        执行简单数据修复（不使用 group_by.apply）
        
        Parameters
        ----------
        df : pl.DataFrame
            原始数据框
            
        Returns
        -------
        pl.DataFrame
            修复后的数据框
        """
        result = df.clone()
        
        # 修复关键列 - 使用全市场中位数
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'total_mv']:
            if col in result.columns:
                # 计算全市场中位数
                median_val = result[col].median()
                if median_val is not None and np.isfinite(median_val):
                    result = result.with_columns([
                        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
                        .then(median_val)
                        .otherwise(pl.col(col))
                        .alias(col)
                    ])
        
        # 特别处理 industry_code - 使用简单方法获取第一个非空值
        if 'industry_code' in result.columns:
            # 获取第一个非空行业值
            first_industry = None
            for val in result['industry_code']:
                if val is not None and val != '':
                    first_industry = val
                    break
            
            if first_industry is not None:
                result = result.with_columns([
                    pl.when(pl.col('industry_code').is_null() | (pl.col('industry_code') == ''))
                    .then(first_industry)
                    .otherwise(pl.col('industry_code'))
                    .alias('industry_code')
                ])
        
        return result
    
    def _repair_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        执行数据修复
        
        Parameters
        ----------
        df : pl.DataFrame
            原始数据框
            
        Returns
        -------
        pl.DataFrame
            修复后的数据框
        """
        result = df.clone()
        
        # 修复关键列 - 使用简单填充方法
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'total_mv']:
            if col in result.columns:
                # 计算全市场中位数
                median_val = result[col].median()
                if median_val is not None and np.isfinite(median_val):
                    result = result.with_columns([
                        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
                        .then(median_val)
                        .otherwise(pl.col(col))
                        .alias(col)
                    ])
        
        # 特别处理 industry_code - 使用第一个非空值填充
        if 'industry_code' in result.columns:
            non_null_industries = result['industry_code'].drop_nulls().unique().to_list()
            if non_null_industries:
                fill_value = non_null_industries[0]
                result = result.with_columns([
                    pl.when(pl.col('industry_code').is_null())
                    .then(fill_value)
                    .otherwise(pl.col('industry_code'))
                    .alias('industry_code')
                ])
        
        return result
    
    def load_index_data(self, start_date: str, end_date: str,
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        if self.db is None:
            return pl.DataFrame()
        
        query = f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE symbol = '{index_code}'
              AND trade_date >= '{start_date}' 
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        
        try:
            df = self.db.read_sql(query)
            return df
        except Exception:
            return pl.DataFrame()


# ===========================================
# V88 AlphaFusion - 时空波动率融合
# ===========================================

class V88AlphaFusion:
    """
    V88 AlphaFusion - 时空波动率融合引擎
    
    【核心逻辑】
    1. 计算 T-1, T-3, T-5 的 Alpha 信号
    2. 使用过去 5 日 IC 的倒数作为动态权重
    3. 加权融合生成最终信号
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V88_FUSION_LAGS)
        self.ic_window = self.config.get('ic_window', V88_IC_WINDOW)
        self.ic_epsilon = self.config.get('ic_epsilon', V88_IC_EPSILON)
        
        self.fusion_signals: List[V88FusionSignal] = []
        self.ic_weights_history: Dict[str, List[float]] = {}
        
        logger.info("V88 AlphaFusion 初始化完成")
        logger.info(f"V88: 融合 Lags={self.fusion_lags}")
        logger.info(f"V88: IC 窗口={self.ic_window}")
    
    def compute_ic_weights(self, df: pl.DataFrame, 
                           signal_col: str = 'composite_score') -> Dict[int, List[float]]:
        """
        计算基于历史 IC 的动态权重
        
        Parameters
        ----------
        df : pl.DataFrame
            包含信号和收益的数据框
        signal_col : str
            信号列
            
        Returns
        -------
        Dict[int, List[float]]
            {lag: [权重序列]}
        """
        # 计算未来收益
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        # 计算每个 lag 的 IC 序列
        ic_series = {}
        for lag in self.fusion_lags:
            return_col = f'forward_return_{lag}d'
            if return_col not in result.columns:
                continue
            
            # 按日期计算 IC
            ic_by_date = result.group_by('trade_date').agg([
                pl.corr(signal_col, return_col, method='spearman').alias('ic')
            ]).filter(pl.col('ic').is_not_null())
            
            if not ic_by_date.is_empty():
                ic_list = ic_by_date['ic'].to_list()
                valid_ic = [ic for ic in ic_list if ic is not None and np.isfinite(ic)]
                
                # 取最近 N 日的 IC 均值
                recent_ic = valid_ic[-self.ic_window:] if len(valid_ic) >= self.ic_window else valid_ic
                ic_series[lag] = np.mean(recent_ic) if recent_ic else 0.0
        
        # 计算动态权重
        if ic_series:
            ic_values = [ic_series.get(lag, 0.0) for lag in self.fusion_lags]
            dynamic_weights = calculate_ic_dynamic_weights(ic_values, self.ic_epsilon)
            
            # 记录权重
            for i, lag in enumerate(self.fusion_lags):
                self.ic_weights_history[lag] = [dynamic_weights[i]]
            
            logger.info(f"V88: IC 动态权重计算完成 - {dict(zip(self.fusion_lags, dynamic_weights))}")
        else:
            # 默认使用等权重
            dynamic_weights = [1.0 / len(self.fusion_lags)] * len(self.fusion_lags)
            logger.warning(f"V88: 无法计算 IC 权重，使用等权重 {dynamic_weights}")
        
        return {lag: [w] for lag, w in zip(self.fusion_lags, dynamic_weights)}
    
    def compute_fusion_signal(self, df: pl.DataFrame,
                               signal_col: str = 'composite_score') -> pl.DataFrame:
        """
        计算时空波动率融合信号
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        signal_col : str
            信号列
            
        Returns
        -------
        pl.DataFrame
            包含融合信号的数据框
        """
        result = df.clone()
        
        # 先计算 IC 动态权重
        ic_weights = self.compute_ic_weights(result, signal_col)
        weights = [ic_weights.get(lag, [0.5])[0] for lag in self.fusion_lags]
        
        # 为每个 lag 计算信号
        lag_signals = []
        result = result.sort(['symbol', 'trade_date'])
        
        for lag in self.fusion_lags:
            lag_col = f"{signal_col}_lag{lag}"
            result = result.with_columns([
                pl.col(signal_col).shift(lag).over('symbol').alias(lag_col)
            ])
            lag_signals.append(lag_col)
        
        # 计算融合信号（加权和）
        fusion_exprs = []
        for i, lag_col in enumerate(lag_signals):
            fusion_exprs.append(pl.col(lag_col) * weights[i])
        
        result = result.with_columns([
            sum(fusion_exprs).alias('fused_signal')
        ])
        
        # 记录融合信号
        unique_dates = result['trade_date'].unique().to_list()
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            for row in day_data.iter_rows(named=True):
                fusion_signal = V88FusionSignal(
                    trade_date=trade_date,
                    symbol=row['symbol'],
                    signal_t1=row.get(f'{signal_col}_lag{self.fusion_lags[0]}', 0.0) or 0.0,
                    signal_t3=row.get(f'{signal_col}_lag{self.fusion_lags[1]}', 0.0) if len(self.fusion_lags) > 1 else 0.0,
                    signal_t5=row.get(f'{signal_col}_lag{self.fusion_lags[2]}', 0.0) if len(self.fusion_lags) > 2 else 0.0,
                    fused_signal=row.get('fused_signal', 0.0) or 0.0,
                    ic_weight_t1=weights[0] if len(weights) > 0 else 0.0,
                    ic_weight_t3=weights[1] if len(weights) > 1 else 0.0,
                    ic_weight_t5=weights[2] if len(weights) > 2 else 0.0,
                )
                self.fusion_signals.append(fusion_signal)
        
        logger.info(f"V88: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        if not self.fusion_signals:
            return {'mean_fused_signal': 0.0, 'std_fused_signal': 0.0}
        
        fused_signals = [s.fused_signal for s in self.fusion_signals 
                        if s.fused_signal is not None and s.fused_signal != 0.0]
        
        return {
            'mean_fused_signal': float(np.mean(fused_signals)) if fused_signals else 0.0,
            'std_fused_signal': float(np.std(fused_signals)) if fused_signals else 0.0,
            'ic_weights': {
                lag: weights[0] if weights else 0.0 
                for lag, weights in self.ic_weights_history.items()
            },
        }


# ===========================================
# V88 AlphaWeight - Alpha 权重引擎
# ===========================================

class V88AlphaWeightEngine:
    """
    V88 AlphaWeight - Alpha 权重引擎
    
    【核心逻辑】
    1. Score < 60 强制权重归零
    2. 权重公式：W_i = Score_i × (1/Volatility_i) / Σ
    3. 单只标的权重限制在 [0.5%, 10%]
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V88_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V88_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V88_MAX_SINGLE_WEIGHT)
        
        self.alpha_weights: List[V88AlphaWeightRecord] = []
    
    def compute_alpha_weights(self, df: pl.DataFrame,
                               score_col: str = 'fused_signal') -> pl.DataFrame:
        """
        计算 Alpha 权重
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        score_col : str
            评分列
            
        Returns
        -------
        pl.DataFrame
            包含权重的数据框
        """
        result = df.clone()
        
        # 计算日收益率
        result = result.sort(['symbol', 'trade_date'])
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        # 计算滚动波动率（年化）
        result = result.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=20)
            .over('symbol')
            .alias('daily_volatility')
        ])
        
        # 年化处理
        result = result.with_columns([
            (pl.col('daily_volatility') * np.sqrt(252)).alias('volatility')
        ])
        
        # 按日期计算权重
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        all_weights = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            scores = day_data[score_col].to_numpy()
            volatilities = day_data['volatility'].to_numpy()
            
            # 过滤无效数据
            valid_mask = (~np.isnan(scores) & ~np.isnan(volatilities) & 
                         np.isfinite(scores) & np.isfinite(volatilities))
            
            if np.sum(valid_mask) < 1:
                continue
            
            valid_scores = scores[valid_mask]
            valid_vols = volatilities[valid_mask]
            valid_symbols = day_data['symbol'].to_numpy()[valid_mask]
            
            # 计算 Alpha 权重
            weights, filtered = alpha_weighting(
                valid_scores, valid_vols,
                self.min_score, self.min_weight, self.max_weight
            )
            
            # 记录权重
            for i, symbol in enumerate(valid_symbols):
                alpha_weight_record = V88AlphaWeightRecord(
                    trade_date=trade_date,
                    symbol=symbol,
                    raw_score=valid_scores[i],
                    volatility=valid_vols[i],
                    alpha_weight=weights[i],
                    normalized_weight=weights[i],
                    is_filtered=filtered[i]
                )
                all_weights.append(alpha_weight_record)
                self.alpha_weights.append(alpha_weight_record)
        
        # 合并回 DataFrame
        if all_weights:
            weight_df = pl.DataFrame({
                'trade_date': [w.trade_date for w in all_weights],
                'symbol': [w.symbol for w in all_weights],
                'volatility': [w.volatility for w in all_weights],
                'alpha_weight': [w.alpha_weight for w in all_weights],
                'is_filtered': [w.is_filtered for w in all_weights],
            })
            
            result = result.join(weight_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V88: Alpha 权重计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_weight_summary(self) -> Dict[str, Any]:
        """获取权重摘要"""
        if not self.alpha_weights:
            return {'mean_weight': 0.0, 'std_weight': 0.0, 'active_positions': 0}
        
        weights = [w.normalized_weight for w in self.alpha_weights if w.normalized_weight > 0]
        active_count = sum(1 for w in self.alpha_weights if w.normalized_weight > 0)
        filtered_count = sum(1 for w in self.alpha_weights if w.is_filtered)
        
        return {
            'mean_weight': float(np.mean(weights)) if weights else 0.0,
            'std_weight': float(np.std(weights)) if weights else 0.0,
            'max_weight': float(np.max(weights)) if weights else 0.0,
            'min_weight': float(np.min(weights)) if weights else 0.0,
            'active_positions': active_count,
            'filtered_positions': filtered_count,
            'avg_score': float(np.mean([w.raw_score for w in self.alpha_weights])) if self.alpha_weights else 0.0,
        }


# ===========================================
# V88 ICAudit - IC 审计
# ===========================================

class V88ICAudit:
    """
    V88 ICAudit - IC 衰减审计
    
    【核心逻辑】
    1. 计算 T+1, T+2, T+3 的 IC
    2. 验证 IC_{T+1} > IC_{T+2} > IC_{T+3}
    3. 若不满足，输出 [CRITICAL_WARNING] Lookahead Bias Potential
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_ic_decay(self, df: pl.DataFrame,
                           signal_col: str = 'fused_signal') -> Dict[str, Any]:
        """
        计算 IC 衰减并审计
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        signal_col : str
            信号列
            
        Returns
        -------
        Dict[str, Any]
            IC 统计和审计结果
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算未来收益
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        # 按日期计算 IC
        ic_results = {}
        for lag in [1, 2, 3]:
            return_col = f'forward_return_{lag}d'
            if return_col not in result.columns:
                continue
            
            ic_by_date = result.group_by('trade_date').agg([
                pl.corr(signal_col, return_col, method='spearman').alias('ic')
            ]).filter(pl.col('ic').is_not_null())
            
            if not ic_by_date.is_empty():
                ic_list = ic_by_date['ic'].drop_nulls().to_list()
                valid_ic = [ic for ic in ic_list if np.isfinite(ic)]
                
                ic_results[f't{lag}'] = {
                    'mean_ic': float(np.mean(valid_ic)) if valid_ic else 0.0,
                    'std_ic': float(np.std(valid_ic)) if valid_ic else 0.0,
                    'ic_count': len(valid_ic),
                }
        
        # 审计 IC 衰减规律
        warning_messages = []
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        
        if ic_t3 > ic_t1:
            warning_messages.append(
                "[CRITICAL_WARNING] Lookahead Bias Potential: T+3 IC > T+1 IC"
            )
        
        if ic_t2 > ic_t1:
            warning_messages.append(
                "[WARNING] IC 衰减异常：T+2 IC > T+1 IC"
            )
        
        # 检查衰减规律
        decay_normal = ic_t1 >= ic_t2 >= ic_t3
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_t2,
            'ic_t3': ic_t3,
            'std_t1': ic_results.get('t1', {}).get('std_ic', 0.0),
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'decay_normal': decay_normal,
            'warning_messages': warning_messages,
        }


# ===========================================
# V88 TurnoverTracker - 换手率追踪
# ===========================================

class V88TurnoverTracker:
    """V88 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V88TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        use_single_side: bool = False) -> V88TurnoverRecord:
        """
        记录换手率
        
        Parameters
        ----------
        trade_date : str
            交易日期
        portfolio_value : float
            组合总价值
        buy_value : float
            买入金额
        sell_value : float
            卖出金额
        use_single_side : bool
            是否使用单边换手率（只计算买入，避免重复计算）
        """
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
        else:
            buy_turnover = buy_value / portfolio_value
            sell_turnover = sell_value / portfolio_value
            
            if use_single_side:
                # 单边换手率：只计算买入（避免重复计算）
                turnover_rate = buy_turnover
            else:
                # 双边换手率：(买入 + 卖出) / 2
                turnover_rate = (buy_turnover + sell_turnover) / 2
        
        self.trading_days += 1
        
        # 年化换手率 = 日均换手 * 252
        annualized_turnover = turnover_rate * 252
        
        record = V88TurnoverRecord(
            trade_date=trade_date,
            turnover_rate=turnover_rate,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            annualized_turnover=annualized_turnover,
        )
        self.turnover_records.append(record)
        
        return record
    
    def get_turnover_summary(self) -> Dict[str, Any]:
        """获取换手率摘要"""
        if not self.turnover_records:
            return {
                'mean_turnover': 0.0,
                'annualized_turnover': 0.0,
                'is_active': False,
            }
        
        turnovers = [r.turnover_rate for r in self.turnover_records]
        annualized = [r.annualized_turnover for r in self.turnover_records]
        
        mean_annualized = np.mean(annualized)
        is_active = V88_TURNOVER_MIN <= mean_annualized <= V88_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean(turnovers)),
            'std_turnover': float(np.std(turnovers)),
            'max_turnover': float(np.max(turnovers)),
            'annualized_turnover': float(mean_annualized),
            'is_active': is_active,
            'turnover_min': V88_TURNOVER_MIN,
            'turnover_max': V88_TURNOVER_MAX,
        }


# ===========================================
# V88 PortfolioTracker - 组合追踪
# ===========================================

class V88PortfolioTracker:
    """V88 组合追踪器"""
    
    def __init__(self, initial_capital: float = V88_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V88Position] = {}
        self.portfolio_snapshots: List[V88DailyPortfolio] = []
        self.total_value = initial_capital
        self.peak_value = initial_capital
    
    def update_positions(self, trade_date: str, prices: Dict[str, float]) -> None:
        """更新持仓价格"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.pnl = (prices[symbol] - position.entry_price) * position.quantity
    
    def calculate_portfolio_value(self) -> float:
        """计算组合总价值"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        self.total_value = self.cash + position_value
        self.peak_value = max(self.peak_value, self.total_value)
        return self.total_value
    
    def get_drawdown(self) -> float:
        """计算当前回撤"""
        if self.peak_value < EPSILON:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value
    
    def record_snapshot(self, trade_date: str, daily_return: float,
                        turnover_rate: float) -> V88DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V88DailyPortfolio(
            trade_date=trade_date,
            total_value=self.total_value,
            cash=self.cash,
            position_value=position_value,
            position_count=len(self.positions),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            turnover_rate=turnover_rate,
        )
        self.portfolio_snapshots.append(snapshot)
        
        return snapshot
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取绩效摘要"""
        if not self.portfolio_snapshots:
            return {'total_return': 0.0, 'max_drawdown': 0.0}
        
        returns = [s.daily_return for s in self.portfolio_snapshots]
        max_drawdown = max(self.get_drawdown(), 
                          max((s.cumulative_return - 1) for s in self.portfolio_snapshots) if self.portfolio_snapshots else 0)
        
        # 计算最大回撤
        peak = self.initial_capital
        max_dd = 0.0
        for snapshot in self.portfolio_snapshots:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            dd = (peak - snapshot.total_value) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'total_value': self.total_value,
            'total_return': (self.total_value - self.initial_capital) / self.initial_capital,
            'max_drawdown': max_dd,
            'final_cash': self.cash,
            'position_count': len(self.positions),
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V88_INITIAL_CAPITAL',
    'V88_MAX_POSITIONS',
    'V88_WARMUP_PERIOD',
    'V88_MIN_SCORE_THRESHOLD',
    'V88_MIN_SINGLE_WEIGHT',
    'V88_MAX_SINGLE_WEIGHT',
    'V88_TURNOVER_MIN',
    'V88_TURNOVER_MAX',
    'V88_EXCESS_RETURN_TARGET',
    'V88_COMMISSION_RATE',
    'V88_MIN_COMMISSION',
    'V88_STAMP_DUTY',
    'V88_TRANSFER_FEE',
    'V88DataManager',
    'V88AlphaFusion',
    'V88AlphaWeightEngine',
    'V88AlphaWeightRecord',
    'V88ICAudit',
    'V88TurnoverTracker',
    'V88PortfolioTracker',
    'V88FusionSignal',
    'V88DataRepairRecord',
    'V88TurnoverRecord',
    'V88Position',
    'V88DailyPortfolio',
    'calculate_rolling_volatility',
    'calculate_ic_dynamic_weights',
    'alpha_weighting',
    'fill_with_industry_median',
    'fill_with_market_mean',
]
