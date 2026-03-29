"""
V87 Core Module - 多时空尺度融合与组合权重熵优化

【V87 核心理念】
1. 多时空尺度融合 (Multi-Horizon Fusion)
   - 计算 T-1, T-3, T-5 的 Alpha 信号
   - 使用衰减加权均值生成最终信号
   - 目的：平滑因单日波动产生的随机交易，将换手率控制在合理范围

2. 组合权重熵优化 (Entropy-Based Weighting)
   - 禁止使用等权重选股
   - 引入 Risk-Parity (风险平价) 思想
   - 权重公式：W_i ∝ Score_i / Volatility_i
   - 确保高分但高波动的个股不会造成净值大幅波动

3. 极端风险对冲模拟 (Fat-tail Stress Test)
   - 模拟"成分股跌停无法卖出"的情景
   - 当个股涨跌幅 <= -9.5% 时，标记为流动性受限
   - 强制次日处理

【硬性指标】
- 指标 A (换手率控制): 单日平均换手率 <= 15%，且 IC_IR / Turnover 比例需提升 10%
- 指标 B (回撤控制): 2024 年度最大回撤 <= 8%（通过风险平价权重实现）
- 指标 C (IC 衰减平滑度): T+1 到 T+3 的 IC 波动率（标准差）需下降 15%

作者：量化系统
版本：V87.0
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
# V87 配置常量
# ===========================================

V87_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V87_MAX_POSITIONS = 10
V87_WARMUP_PERIOD = 250
V87_MIN_SAMPLE_SIZE = 100
V87_MIN_STOCK_DAILY_ROWS = 500000

# 多时空尺度融合配置
V87_FUSION_LAGS = [1, 3, 5]  # T-1, T-3, T-5
V87_FUSION_DECAY_WEIGHTS = [0.5, 0.3, 0.2]  # 衰减权重（近期权重更高）
V87_FUSION_HALF_LIFE = 3  # 半衰期（用于指数加权）

# 风险平价权重配置
V87_VOLATILITY_WINDOW = 20  # 波动率计算窗口
V87_RISK_PARITY_EXPONENT = 1.0  # 风险平价指数（控制波动率敏感度）
V87_MAX_SINGLE_WEIGHT = 0.20  # 单个标的最大权重 20%
V87_MIN_SINGLE_WEIGHT = 0.02  # 单个标的最低权重 2%

# 极端风险配置
V87_LIMIT_DOWN_THRESHOLD = -9.5  # 跌停阈值
V87_LIMIT_UP_THRESHOLD = 9.5  # 涨停阈值
V87_LIQUIDITY_CONSTRAINT_WINDOW = 1  # 流动性受限处理窗口

# 费率配置（严禁修改）
V87_COMMISSION_RATE = 0.002  # 0.2%
V87_MIN_COMMISSION = 5.0
V87_STAMP_DUTY = 0.0005  # 印花税
V87_TRANSFER_FEE = 0.00001

# 绩效目标
V87_TURNOVER_TARGET = 0.15  # 15% 换手率上限
V87_DRAWDOWN_TARGET = 0.08  # 8% 最大回撤
V87_IC_SMOOTHING_TARGET = 0.15  # 15% IC 波动率下降
V87_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V87FusionSignal:
    """多时空融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float  # T-1 信号
    signal_t3: float  # T-3 信号
    signal_t5: float  # T-5 信号
    fused_signal: float  # 融合后信号
    fusion_weights: List[float]  # 实际使用的权重


@dataclass
class V87RiskParityWeight:
    """风险平价权重"""
    trade_date: str
    symbol: str
    raw_score: float  # 原始评分
    volatility: float  # 波动率
    risk_parity_weight: float  # 风险平价权重
    normalized_weight: float  # 归一化后权重


@dataclass
class V87LiquidityConstraintRecord:
    """流动性受限记录"""
    trade_date: str
    symbol: str
    pct_chg: float
    constraint_type: str  # "limit_down" 或 "limit_up"
    is_liquidity_constrained: bool
    release_date: Optional[str]  # 解除受限日期


@dataclass
class V87TurnoverMetrics:
    """换手率指标"""
    trade_date: str
    turnover_rate: float  # 换手率
    buy_turnover: float  # 买入换手
    sell_turnover: float  # 卖出换手
    is_over_limit: bool  # 是否超过限制


@dataclass
class V87DrawdownMetrics:
    """回撤指标"""
    trade_date: str
    portfolio_value: float
    peak_value: float
    drawdown: float  # 当前回撤
    max_drawdown: float  # 最大回撤
    is_over_limit: bool  # 是否超过限制


# ===========================================
# V87 工具函数
# ===========================================

def exponential_decay_weights(half_life: int, n_periods: int) -> List[float]:
    """
    计算指数衰减权重
    
    公式：weight = exp(-ln(2) * lag / half_life)
    
    Parameters
    ----------
    half_life : int
        半衰期
    n_periods : int
        周期数量
        
    Returns
    -------
    List[float]
        归一化后的权重列表
    """
    lags = list(range(n_periods))
    weights = [np.exp(-np.log(2) * lag / half_life) for lag in lags]
    total_weight = sum(weights)
    return [w / total_weight for w in weights]


def calculate_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    计算滚动波动率
    
    Parameters
    ----------
    returns : np.ndarray
        收益率序列
    window : int
        滚动窗口
        
    Returns
    -------
    np.ndarray
        波动率序列
    """
    if len(returns) < window:
        return np.full(len(returns), np.nan)
    
    result = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        valid_returns = window_returns[~np.isnan(window_returns)]
        if len(valid_returns) > 1:
            result[i] = np.std(valid_returns, ddof=1)
    
    return result


def risk_parity_weighting(scores: np.ndarray, volatilities: np.ndarray,
                          max_weight: float = V87_MAX_SINGLE_WEIGHT,
                          min_weight: float = V87_MIN_SINGLE_WEIGHT,
                          exponent: float = V87_RISK_PARITY_EXPONENT) -> np.ndarray:
    """
    风险平价权重计算
    
    公式：W_i ∝ (Score_i / Volatility_i)^exponent
    
    Parameters
    ----------
    scores : np.ndarray
        原始评分
    volatilities : np.ndarray
        波动率
    max_weight : float
        最大权重
    min_weight : float
        最小权重
    exponent : float
        风险平价指数
        
    Returns
    -------
    np.ndarray
        归一化权重
    """
    # 防止除零
    volatilities = np.where(volatilities < EPSILON, EPSILON, volatilities)
    
    # 确保评分为正
    scores = np.where(scores < 0, 0, scores)
    
    # 计算风险平价权重
    raw_weights = (scores / volatilities) ** exponent
    
    # 归一化
    total_weight = np.sum(raw_weights)
    if total_weight < EPSILON:
        return np.full(len(scores), 1.0 / len(scores))
    
    normalized_weights = raw_weights / total_weight
    
    # 应用权重限制
    normalized_weights = np.clip(normalized_weights, min_weight, max_weight)
    
    # 重新归一化以确保总和为 1
    total_weight = np.sum(normalized_weights)
    if total_weight > EPSILON:
        normalized_weights = normalized_weights / total_weight
    
    return normalized_weights


def quantile_transform(series: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    """分位数映射到正态分布"""
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


def zscore_normalize(series: np.ndarray) -> np.ndarray:
    """Z-Score 空间归一化"""
    if len(series) < 2:
        return series
    
    mask = np.isfinite(series)
    result = series.copy()
    
    if not np.any(mask):
        return result
    
    valid_data = series[mask]
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    
    if std_val > EPSILON:
        result[mask] = (valid_data - mean_val) / std_val
    else:
        result[mask] = 0.0
    
    return result


# ===========================================
# V87 DataManager
# ===========================================

class V87DataManager:
    """
    V87 数据管理器 - 支持多时空尺度数据加载
    
    【核心改进】
    - 支持加载历史信号用于融合
    - 支持波动率计算所需的历史数据
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V87_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V87_MIN_SAMPLE_SIZE)
        self._data_cache: Dict[str, pl.DataFrame] = {}
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        """计算热身起始日期（考虑最大 lag）"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            # 额外增加 lag 窗口
            extra_days = max(V87_FUSION_LAGS) + V87_VOLATILITY_WINDOW
            warmup_start = start - timedelta(days=self.warmup_period + extra_days)
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
            
            daily_count = int(df['cnt'][0])
            
            if daily_count < V87_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count:,} < {V87_MIN_STOCK_DAILY_ROWS:,}"
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
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
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
        """加载行业映射"""
        if self.db is None:
            return self._empty_industry_mapping_df()
        
        query = """
            SELECT DISTINCT symbol, industry_name, industry_code
            FROM stock_industry_daily
            WHERE industry_name IS NOT NULL
        """
        
        try:
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return self._empty_industry_mapping_df()
            
            return df
            
        except Exception:
            return self._empty_industry_mapping_df()
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
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
        
        try:
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return self._empty_index_df()
            
            return df
            
        except Exception:
            return self._empty_index_df()
    
    def _empty_industry_mapping_df(self) -> pl.DataFrame:
        """返回空的行业映射 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'industry_name': pl.Utf8,
            'industry_code': pl.Utf8
        })
    
    def _empty_index_df(self) -> pl.DataFrame:
        """返回空的指数 DataFrame"""
        return pl.DataFrame(schema={
            'trade_date': pl.Utf8,
            'close': pl.Float64
        })


# ===========================================
# V87 AlphaFusion - 多时空尺度融合
# ===========================================

class V87AlphaFusion:
    """
    V87 AlphaFusion - 多时空尺度融合引擎
    
    【核心逻辑】
    1. 计算 T-1, T-3, T-5 的 Alpha 信号
    2. 使用衰减加权均值生成最终信号
    3. 平滑因单日波动产生的随机交易
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 融合配置
        self.fusion_lags = self.config.get('fusion_lags', V87_FUSION_LAGS)
        self.fusion_decay_weights = self.config.get('fusion_decay_weights', V87_FUSION_DECAY_WEIGHTS)
        self.fusion_half_life = self.config.get('fusion_half_life', V87_FUSION_HALF_LIFE)
        
        # 状态记录
        self.fusion_signals: List[V87FusionSignal] = []
        
        # 信号缓存（用于 lag 计算）
        self._signal_cache: Dict[str, Dict[str, float]] = {}  # {date: {symbol: signal}}
        
        logger.info("V87 AlphaFusion 初始化完成")
        logger.info(f"V87: 融合 Lags={self.fusion_lags}")
        logger.info(f"V87: 衰减权重={self.fusion_decay_weights}")
        logger.info(f"V87: 半衰期={self.fusion_half_life}")
    
    def compute_fusion_signal(self, df: pl.DataFrame, 
                               signal_col: str = 'composite_score') -> pl.DataFrame:
        """
        计算多时空融合信号
        
        【公式】
        Fused_Signal = Σ(weight_i * Signal_{t-i})
        
        Parameters
        ----------
        df : pl.DataFrame
            包含原始信号的数据框
        signal_col : str
            原始信号列名
            
        Returns
        -------
        pl.DataFrame
            包含融合信号的数据框
        """
        result = df.clone()
        
        # 计算指数衰减权重
        n_lags = len(self.fusion_lags)
        weights = exponential_decay_weights(self.fusion_half_life, n_lags)
        
        logger.info(f"V87: 使用指数衰减权重：{weights}")
        
        # 为每个 lag 计算信号
        lag_signals = []
        for lag in self.fusion_lags:
            lag_col = f"{signal_col}_lag{lag}"
            result = result.sort(['symbol', 'trade_date'])
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
                fusion_signal = V87FusionSignal(
                    trade_date=trade_date,
                    symbol=row['symbol'],
                    signal_t1=row.get(f'{signal_col}_lag{self.fusion_lags[0]}', 0.0) or 0.0,
                    signal_t3=row.get(f'{signal_col}_lag{self.fusion_lags[1]}', 0.0) if len(self.fusion_lags) > 1 else 0.0,
                    signal_t5=row.get(f'{signal_col}_lag{self.fusion_lags[2]}', 0.0) if len(self.fusion_lags) > 2 else 0.0,
                    fused_signal=row.get('fused_signal', 0.0) or 0.0,
                    fusion_weights=weights
                )
                self.fusion_signals.append(fusion_signal)
        
        logger.info(f"V87: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        if not self.fusion_signals:
            return {
                'mean_fused_signal': 0.0,
                'std_fused_signal': 0.0,
                'mean_signal_t1': 0.0,
                'mean_signal_t3': 0.0,
                'mean_signal_t5': 0.0,
            }
        
        # 过滤 None 值和 0 值
        fused_signals = [s.fused_signal for s in self.fusion_signals 
                        if s.fused_signal is not None and s.fused_signal != 0.0]
        signal_t1_list = [s.signal_t1 for s in self.fusion_signals 
                         if s.signal_t1 is not None and s.signal_t1 != 0.0]
        signal_t3_list = [s.signal_t3 for s in self.fusion_signals 
                         if s.signal_t3 is not None and s.signal_t3 != 0.0]
        signal_t5_list = [s.signal_t5 for s in self.fusion_signals 
                         if s.signal_t5 is not None and s.signal_t5 != 0.0]
        
        return {
            'mean_fused_signal': float(np.mean(fused_signals)) if fused_signals else 0.0,
            'std_fused_signal': float(np.std(fused_signals)) if fused_signals else 0.0,
            'mean_signal_t1': float(np.mean(signal_t1_list)) if signal_t1_list else 0.0,
            'mean_signal_t3': float(np.mean(signal_t3_list)) if signal_t3_list else 0.0,
            'mean_signal_t5': float(np.mean(signal_t5_list)) if signal_t5_list else 0.0,
            'weights': self.fusion_decay_weights,
        }


# ===========================================
# V87 RiskParity - 风险平价权重
# ===========================================

class V87RiskParity:
    """
    V87 RiskParity - 风险平价权重引擎
    
    【核心逻辑】
    1. 计算个股近期波动率
    2. 使用公式 W_i ∝ Score_i / Volatility_i 计算权重
    3. 确保高分但高波动的个股不会造成净值大幅波动
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 风险平价配置
        self.volatility_window = self.config.get('volatility_window', V87_VOLATILITY_WINDOW)
        self.risk_parity_exponent = self.config.get('risk_parity_exponent', V87_RISK_PARITY_EXPONENT)
        self.max_single_weight = self.config.get('max_single_weight', V87_MAX_SINGLE_WEIGHT)
        self.min_single_weight = self.config.get('min_single_weight', V87_MIN_SINGLE_WEIGHT)
        
        # 状态记录
        self.risk_parity_weights: List[V87RiskParityWeight] = []
        
        logger.info("V87 RiskParity 初始化完成")
        logger.info(f"V87: 波动率窗口={self.volatility_window}")
        logger.info(f"V87: 风险平价指数={self.risk_parity_exponent}")
        logger.info(f"V87: 权重限制=[{self.min_single_weight:.1%}, {self.max_single_weight:.1%}]")
    
    def compute_risk_parity_weights(self, df: pl.DataFrame,
                                     score_col: str = 'fused_signal') -> pl.DataFrame:
        """
        计算风险平价权重
        
        【公式】
        W_i = (Score_i / Volatility_i)^exponent / Σ(Score_j / Volatility_j)^exponent
        
        Parameters
        ----------
        df : pl.DataFrame
            包含评分和价格数据的数据框
        score_col : str
            评分列名（应为融合后的信号）
            
        Returns
        -------
        pl.DataFrame
            包含风险平价权重的数据框
        """
        result = df.clone()
        
        # 计算个股收益率
        result = result.sort(['symbol', 'trade_date'])
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close')) / 
             (pl.col('close') + EPSILON)).alias('daily_return')
        ])
        
        # 计算滚动波动率
        result = result.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=self.volatility_window)
            .over('symbol')
            .alias('volatility')
        ])
        
        # 获取唯一交易日
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        # 为每个交易日计算风险平价权重
        all_weights = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            scores = day_data[score_col].to_numpy()
            volatilities = day_data['volatility'].to_numpy()
            
            # 过滤无效数据
            valid_mask = (~np.isnan(scores) & ~np.isnan(volatilities) & 
                         np.isfinite(scores) & np.isfinite(volatilities))
            
            if np.sum(valid_mask) < 3:
                continue
            
            valid_scores = scores[valid_mask]
            valid_volatilities = volatilities[valid_mask]
            valid_symbols = day_data.filter(pl.col('trade_date') == trade_date)['symbol'].to_numpy()[valid_mask]
            
            # 计算风险平价权重
            weights = risk_parity_weighting(
                valid_scores, valid_volatilities,
                self.max_single_weight, self.min_single_weight,
                self.risk_parity_exponent
            )
            
            # 记录权重
            for i, symbol in enumerate(valid_symbols):
                risk_parity_weight = V87RiskParityWeight(
                    trade_date=trade_date,
                    symbol=symbol,
                    raw_score=valid_scores[i],
                    volatility=valid_volatilities[i],
                    risk_parity_weight=weights[i],
                    normalized_weight=weights[i]
                )
                all_weights.append(risk_parity_weight)
                self.risk_parity_weights.append(risk_parity_weight)
        
        # 将权重合并回 DataFrame
        # 创建一个临时 DataFrame 用于合并
        if all_weights:
            weight_df = pl.DataFrame({
                'trade_date': [w.trade_date for w in all_weights],
                'symbol': [w.symbol for w in all_weights],
                'volatility': [w.volatility for w in all_weights],
                'risk_parity_weight': [w.risk_parity_weight for w in all_weights],
            })
            
            result = result.join(
                weight_df,
                on=['trade_date', 'symbol'],
                how='left'
            )
        
        logger.info(f"V87: 风险平价权重计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_risk_parity_summary(self) -> Dict[str, Any]:
        """获取风险平价权重摘要"""
        if not self.risk_parity_weights:
            return {
                'mean_weight': 0.0,
                'std_weight': 0.0,
                'max_weight': 0.0,
                'min_weight': 0.0,
                'mean_volatility': 0.0,
            }
        
        weights = [w.normalized_weight for w in self.risk_parity_weights]
        volatilities = [w.volatility for w in self.risk_parity_weights if w.volatility > 0]
        
        return {
            'mean_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights)),
            'mean_volatility': float(np.mean(volatilities)) if volatilities else 0.0,
            'weight_concentration': float(1.0 / (len(weights) * np.sum(np.array(weights) ** 2))),  # Herfindahl 指数
        }


# ===========================================
# V87 LiquidityConstraint - 流动性受限
# ===========================================

class V87LiquidityConstraintDetector:
    """
    V87 LiquidityConstraint - 流动性受限检测
    
    【核心逻辑】
    1. 检测涨跌幅 <= -9.5% 的跌停股票
    2. 检测涨跌幅 >= 9.5% 的涨停股票
    3. 标记为流动性受限，强制次日处理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 流动性配置
        self.limit_down_threshold = self.config.get('limit_down_threshold', V87_LIMIT_DOWN_THRESHOLD)
        self.limit_up_threshold = self.config.get('limit_up_threshold', V87_LIMIT_UP_THRESHOLD)
        self.constraint_window = self.config.get('constraint_window', V87_LIQUIDITY_CONSTRAINT_WINDOW)
        
        # 状态记录
        self.liquidity_constraints: List[V87LiquidityConstraintRecord] = []
        
        logger.info("V87 LiquidityConstraint 初始化完成")
        logger.info(f"V87: 跌停阈值={self.limit_down_threshold}%")
        logger.info(f"V87: 涨停阈值={self.limit_up_threshold}%")
    
    def detect_liquidity_constraints(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        检测流动性受限股票
        
        Parameters
        ----------
        df : pl.DataFrame
            包含涨跌幅数据的数据框
            
        Returns
        -------
        pl.DataFrame
            包含流动性受限标记的数据框
        """
        result = df.clone()
        
        # 检测跌停和涨停
        result = result.with_columns([
            (pl.col('pct_chg') <= self.limit_down_threshold).alias('is_limit_down'),
            (pl.col('pct_chg') >= self.limit_up_threshold).alias('is_limit_up'),
            ((pl.col('pct_chg') <= self.limit_down_threshold) | 
             (pl.col('pct_chg') >= self.limit_up_threshold)).alias('is_liquidity_constrained')
        ])
        
        # 记录流动性受限事件
        unique_dates = sorted(result['trade_date'].unique().to_list())
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            constrained_stocks = day_data.filter(pl.col('is_liquidity_constrained') == True)
            
            if not constrained_stocks.is_empty():
                for row in constrained_stocks.iter_rows(named=True):
                    constraint_type = "limit_down" if row['pct_chg'] <= self.limit_down_threshold else "limit_up"
                    liquidity_constraint = V87LiquidityConstraintRecord(
                        trade_date=trade_date,
                        symbol=row['symbol'],
                        pct_chg=row['pct_chg'],
                        constraint_type=constraint_type,
                        is_liquidity_constrained=True,
                        release_date=None  # 次日解除
                    )
                    self.liquidity_constraints.append(liquidity_constraint)
        
        logger.info(f"V87: 流动性受限检测完成，发现 {len(self.liquidity_constraints)} 次受限事件")
        
        return result
    
    def get_liquidity_summary(self) -> Dict[str, Any]:
        """获取流动性受限摘要"""
        if not self.liquidity_constraints:
            return {
                'total_constraints': 0,
                'limit_down_count': 0,
                'limit_up_count': 0,
                'constraint_ratio': 0.0,
            }
        
        limit_down_count = sum(1 for c in self.liquidity_constraints if c.constraint_type == "limit_down")
        limit_up_count = sum(1 for c in self.liquidity_constraints if c.constraint_type == "limit_up")
        
        return {
            'total_constraints': len(self.liquidity_constraints),
            'limit_down_count': limit_down_count,
            'limit_up_count': limit_up_count,
            'avg_pct_chg': float(np.mean([c.pct_chg for c in self.liquidity_constraints])),
        }


# ===========================================
# V87 TurnoverTracker - 换手率追踪
# ===========================================

class V87TurnoverTracker:
    """
    V87 TurnoverTracker - 换手率追踪器
    
    【核心逻辑】
    1. 追踪每日买入和卖出换手
    2. 检测是否超过 15% 换手率限制
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_target = self.config.get('turnover_target', V87_TURNOVER_TARGET)
        
        # 状态记录
        self.turnover_metrics: List[V87TurnoverMetrics] = []
        
        logger.info("V87 TurnoverTracker 初始化完成")
        logger.info(f"V87: 换手率目标 <= {self.turnover_target:.1%}")
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float) -> V87TurnoverMetrics:
        """
        记录换手率
        
        Parameters
        ----------
        trade_date : str
            交易日
        portfolio_value : float
            组合总价值
        buy_value : float
            买入金额
        sell_value : float
            卖出金额
            
        Returns
        -------
        V87TurnoverMetrics
            换手率指标
        """
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
        else:
            buy_turnover = buy_value / portfolio_value
            sell_turnover = sell_value / portfolio_value
            turnover_rate = (buy_turnover + sell_turnover) / 2
        
        is_over_limit = turnover_rate > self.turnover_target
        
        metric = V87TurnoverMetrics(
            trade_date=trade_date,
            turnover_rate=turnover_rate,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            is_over_limit=is_over_limit
        )
        self.turnover_metrics.append(metric)
        
        return metric
    
    def get_turnover_summary(self) -> Dict[str, Any]:
        """获取换手率摘要"""
        if not self.turnover_metrics:
            return {
                'mean_turnover': 0.0,
                'std_turnover': 0.0,
                'max_turnover': 0.0,
                'over_limit_days': 0,
            }
        
        turnovers = [m.turnover_rate for m in self.turnover_metrics]
        over_limit_days = sum(1 for m in self.turnover_metrics if m.is_over_limit)
        
        return {
            'mean_turnover': float(np.mean(turnovers)),
            'std_turnover': float(np.std(turnovers)),
            'max_turnover': float(np.max(turnovers)),
            'over_limit_days': over_limit_days,
            'over_limit_ratio': float(over_limit_days / len(self.turnover_metrics)),
        }


# ===========================================
# V87 DrawdownTracker - 回撤追踪
# ===========================================

class V87DrawdownTracker:
    """
    V87 DrawdownTracker - 回撤追踪器
    
    【核心逻辑】
    1. 追踪组合回撤
    2. 检测是否超过 8% 回撤限制
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.drawdown_target = self.config.get('drawdown_target', V87_DRAWDOWN_TARGET)
        
        # 状态记录
        self.drawdown_metrics: List[V87DrawdownMetrics] = []
        self.peak_value = 0.0
        
        logger.info("V87 DrawdownTracker 初始化完成")
        logger.info(f"V87: 回撤目标 <= {self.drawdown_target:.1%}")
    
    def record_drawdown(self, trade_date: str, portfolio_value: float) -> V87DrawdownMetrics:
        """
        记录回撤
        
        Parameters
        ----------
        trade_date : str
            交易日
        portfolio_value : float
            组合总价值
            
        Returns
        -------
        V87DrawdownMetrics
            回撤指标
        """
        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # 计算回撤
        if self.peak_value < EPSILON:
            drawdown = 0.0
        else:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # 计算最大回撤
        max_drawdown = max(
            drawdown,
            max((m.max_drawdown for m in self.drawdown_metrics), default=0.0)
        )
        
        is_over_limit = drawdown > self.drawdown_target
        
        metric = V87DrawdownMetrics(
            trade_date=trade_date,
            portfolio_value=portfolio_value,
            peak_value=self.peak_value,
            drawdown=drawdown,
            max_drawdown=max_drawdown,
            is_over_limit=is_over_limit
        )
        self.drawdown_metrics.append(metric)
        
        return metric
    
    def get_drawdown_summary(self) -> Dict[str, Any]:
        """获取回撤摘要"""
        if not self.drawdown_metrics:
            return {
                'max_drawdown': 0.0,
                'mean_drawdown': 0.0,
                'over_limit_days': 0,
            }
        
        drawdowns = [m.drawdown for m in self.drawdown_metrics]
        max_drawdown = max(m.max_drawdown for m in self.drawdown_metrics)
        over_limit_days = sum(1 for m in self.drawdown_metrics if m.is_over_limit)
        
        return {
            'max_drawdown': float(max_drawdown),
            'mean_drawdown': float(np.mean(drawdowns)),
            'over_limit_days': over_limit_days,
            'over_limit_ratio': float(over_limit_days / len(self.drawdown_metrics)),
        }


# ===========================================
# V87 ICSmoother - IC 平滑度分析
# ===========================================

class V87ICSmoother:
    """
    V87 ICSmoother - IC 平滑度分析
    
    【核心逻辑】
    1. 计算 T+1 到 T+3 的 IC 波动率
    2. 对比 V86 和 V87 的 IC 平滑度
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.smoothing_target = self.config.get('smoothing_target', V87_IC_SMOOTHING_TARGET)
        
        logger.info("V87 ICSmoother 初始化完成")
        logger.info(f"V87: IC 波动率下降目标 >= {self.smoothing_target:.1%}")
    
    def calculate_ic_volatility(self, ic_series: List[float]) -> float:
        """
        计算 IC 波动率
        
        Parameters
        ----------
        ic_series : List[float]
            IC 序列
            
        Returns
        -------
        float
            IC 波动率（标准差）
        """
        if len(ic_series) < 3:
            return 0.0
        
        return float(np.std(ic_series, ddof=1))
    
    def compare_smoothing(self, v86_ic_volatility: float, 
                          v87_ic_volatility: float) -> Dict[str, Any]:
        """
        对比 V86 和 V87 的 IC 平滑度
        
        Parameters
        ----------
        v86_ic_volatility : float
            V86 的 IC 波动率
        v87_ic_volatility : float
            V87 的 IC 波动率
            
        Returns
        -------
        Dict[str, Any]
            平滑度对比结果
        """
        if v86_ic_volatility < EPSILON:
            improvement = 0.0
        else:
            improvement = (v86_ic_volatility - v87_ic_volatility) / v86_ic_volatility
        
        is_target_met = improvement >= self.smoothing_target
        
        return {
            'v86_ic_volatility': v86_ic_volatility,
            'v87_ic_volatility': v87_ic_volatility,
            'improvement': improvement,
            'is_target_met': is_target_met,
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V87_INITIAL_CAPITAL',
    'V87_MAX_POSITIONS',
    'V87_WARMUP_PERIOD',
    'V87_TURNOVER_TARGET',
    'V87_DRAWDOWN_TARGET',
    'V87_IC_SMOOTHING_TARGET',
    'V87DataManager',
    'V87AlphaFusion',
    'V87RiskParity',
    'V87LiquidityConstraintDetector',
    'V87TurnoverTracker',
    'V87DrawdownTracker',
    'V87ICSmoother',
    'V87FusionSignal',
    'V87RiskParityWeight',
    'V87LiquidityConstraintRecord',
    'V87TurnoverMetrics',
    'V87DrawdownMetrics',
    'exponential_decay_weights',
    'calculate_volatility',
    'risk_parity_weighting',
    'quantile_transform',
    'zscore_normalize',
]
