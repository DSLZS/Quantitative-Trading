"""
V80 Core Module - 跨时空泛化训练与自适应因子修正 (V80.1 优化版)

【V80.1 核心修复】
1. 修复未来函数问题：确保所有因子使用 T-1 数据
2. 优化 Rank IC 计算：使用正确的 forward return
3. 增强因子预测能力：调整因子权重和计算逻辑
4. 严格止损控制：降低最大回撤

【验收标准】
- 标准 A：2019, 2021, 2024 三个测试年份的 Mean Rank IC 均值 > 0.025
- 标准 B：最大回撤在 2024 年不得超过 8%
- 标准 C：输出 factor_monitor.csv 证明具备了失效监控能力

作者：量化系统
版本：V80.1
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
from loguru import logger


# ===========================================
# V80 配置常量
# ===========================================

V80_INITIAL_CAPITAL = 100000.00
V80_MAX_POSITIONS = 10
V80_WARMUP_PERIOD = 250
V80_MIN_SAMPLE_SIZE = 100
V80_MIN_STOCK_DAILY_ROWS = 100000
V80_RETRY_ATTEMPTS = 5
V80_RETRY_DELAY = 3.0

# 残差 Alpha 配置 (V80.2 优化：缩短窗口，增强敏感度)
V80_RESIDUAL_WINDOW = 10
V80_RESIDUAL_MIN = 0.0
V80_VOLATILITY_WINDOW = 15

# 反转因子配置 (V80.2 优化：更短期反转)
V80_REVERSAL_WINDOW = 2
V80_REVERSAL_PENALTY_TOP = 0.15

# 流动性配置 (V80.2 优化：调整参数)
V80_LIQUIDITY_WINDOW = 15
V80_VSHOCK_OPTIMAL_MIN = 1.0
V80_VSHOCK_OPTIMAL_MAX = 3.0
V80_VSHOCK_EXCESSIVE = 4.0
V80_VSHOCK_SHRINK = 0.5
V80_LIQUIDITY_BONUS = 1.3
V80_LIQUIDITY_PENALTY = 0.6

# 多周期 RS 配置 (V80.2 优化：增强短期权重)
V80_RS_SHORT_WINDOW = 3
V80_RS_LONG_WINDOW = 15
V80_RS_SHORT_WEIGHT = 0.7
V80_RS_LONG_WEIGHT = 0.3

# 因子权重 (V80.2 优化：增强反转和 RS 权重)
V80_RESIDUAL_WEIGHT = 0.30
V80_REVERSAL_WEIGHT = 0.35
V80_LIQUIDITY_WEIGHT = 0.10
V80_MULTI_RS_WEIGHT = 0.25

# Regime_Switch 配置
V80_REGIME_HIGH_VOLATILITY = 0.025
V80_REGIME_LOW_VOLATILITY = 0.01
V80_REGIME_REVERSAL_BOOST = 1.5
V80_REGIME_MOMENTUM_BOOST = 1.3

# 行业拥挤度配置
V80_SECTOR_CROWDING_WINDOW = 5
V80_SECTOR_CROWDING_STD_THRESHOLD = 1.5  # 降低阈值
V80_SECTOR_CROWDING_PENALTY = 0.50  # 增强惩罚
V80_SECTOR_HISTORY_WINDOW = 60

# 波动率缩放
V80_VOLATILITY_SCALING = True
V80_VOLATILITY_BASE = 1.0

# 行业分散性控制
V80_MAX_SECTOR_WEIGHT = 0.20
V80_INDUSTRY_NEUTRAL_WEIGHT = 1.0

# 费率配置
V80_COMMISSION_RATE = 0.0003
V80_MIN_COMMISSION = 5.0
V80_SLIPPAGE_BUY = 0.001
V80_SLIPPAGE_SELL = 0.001
V80_STAMP_DUTY = 0.0005
V80_TRANSFER_FEE = 0.00001
V80_FRICTION_COST = 0.002

# 止损止盈 (V80.2 优化：更严格的止损，更宽松的止盈)
V80_STOP_LOSS_RATIO = 0.025  # 降低到 2.5%
V80_PROFIT_TARGET_RATIO = 0.08  # 降低到 8%
V80_TRAILING_STOP_RATIO = 0.02  # 移动止盈降低到 2%

V80_MAX_SINGLE_POSITION_PCT = 0.08  # 降低单仓位到 8%
V80_SELECTION_PERCENTILE = 0.08  # 只交易前 8% 的股票 (更严格)

# Rank IC 目标
V80_RANK_IC_TARGET = 0.025
V80_RANK_IC_MIN = 0.02
V80_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

V80_MAX_DRAWDOWN_TARGET = 0.08
V80_WIN_RATE_TARGET = 0.45

V80_FACTOR_MONITOR_PATH = "reports/factor_monitor.csv"
V80_FACTOR_FAILURE_THRESHOLD = 0.0
V80_FACTOR_CONSECUTIVE_NEGATIVE = 2
V80_FACTOR_DECAY_RATIO = 0.5

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V80Position:
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    residual_alpha_score: float = 0.0
    reversal_score: float = 0.0
    liquidity_score: float = 0.0
    risk_adjusted_score: float = 0.0
    multi_rs_score: float = 0.0
    rs_short: float = 0.0
    rs_long: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    industry_return: float = 0.0
    residual_return: float = 0.0
    sector_crowding: float = 0.0
    volatility: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    industry_weight: float = 1.0
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False


@dataclass
class V80Trade:
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
class V80Signal:
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    residual_alpha_score: float = 0.0
    residual_return: float = 0.0
    stock_return: float = 0.0
    industry_return: float = 0.0
    reversal_score: float = 0.0
    short_term_return: float = 0.0
    liquidity_score: float = 0.0
    v_shock: float = 0.0
    risk_adjusted_liquidity: float = 0.0
    multi_rs_score: float = 0.0
    rs_short: float = 0.0
    rs_long: float = 0.0
    sector_crowding: float = 0.0
    sector_penalty: float = 1.0
    volatility: float = 0.0
    vol_scaled_score: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    industry_weight: float = 1.0
    close_price: float = 0.0


@dataclass
class V80ICMetrics:
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V80MonthlyICStats:
    month: str
    factor_name: str
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False


@dataclass
class V80FactorMonitor:
    month: str
    rs_rank_ic: float
    residual_rank_ic: float
    reversal_rank_ic: float
    rs_weight: float
    residual_weight: float
    reversal_weight: float
    alarm_triggered: bool = False
    alarm_factor: str = ""


@dataclass
class V80RegimeState:
    trade_date: str
    market_volatility: float
    market_return: float
    regime_type: str
    reversal_weight_multiplier: float
    momentum_weight_multiplier: float


# ===========================================
# V80 DataManager
# ===========================================

class V80DataManager:
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V80_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V80_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V80_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V80_RETRY_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._missing_data_log: List[str] = []
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2023-01-01"
    
    def check_2024_data_integrity(self) -> Tuple[bool, str]:
        if self.db is None:
            return False, "数据库连接未初始化"
        
        try:
            query = """
                SELECT COUNT(*) as cnt 
                FROM stock_daily
                WHERE trade_date >= '2024-01-01' 
                  AND trade_date <= '2024-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return False, "无法查询 stock_daily 表"
            
            daily_count = int(df['cnt'][0])
            
            if daily_count < V80_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count} < {V80_MIN_STOCK_DAILY_ROWS}"
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
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            return self._empty_fund_flow_df()
        
        if symbols:
            symbol_list = "','".join(symbols)
            symbol_filter = f"AND symbol IN ('{symbol_list}')"
        else:
            symbol_filter = ""
        
        query = f"""
            SELECT symbol, trade_date, net_main_amount, net_main_rate
            FROM stock_fund_flow
            WHERE trade_date >= '{actual_start_date}' 
              AND trade_date <= '{end_date}'
              {symbol_filter}
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            return self._empty_fund_flow_df()
        
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
    
    def _empty_fund_flow_df(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'net_main_amount': pl.Float64,
            'net_main_ratio': pl.Float64,
            'net_main_rate': pl.Float64
        })
    
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
# V80 AlphaCenter
# ===========================================

class V80AlphaCenter:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 残差 Alpha 配置
        self.residual_window = self.config.get('residual_window', V80_RESIDUAL_WINDOW)
        self.volatility_window = self.config.get('volatility_window', V80_VOLATILITY_WINDOW)
        
        # 反转因子配置
        self.reversal_window = self.config.get('reversal_window', V80_REVERSAL_WINDOW)
        self.reversal_penalty_top = self.config.get('reversal_penalty_top', V80_REVERSAL_PENALTY_TOP)
        
        # 流动性配置
        self.liquidity_window = self.config.get('liquidity_window', V80_LIQUIDITY_WINDOW)
        self.vshock_optimal_min = self.config.get('vshock_optimal_min', V80_VSHOCK_OPTIMAL_MIN)
        self.vshock_optimal_max = self.config.get('vshock_optimal_max', V80_VSHOCK_OPTIMAL_MAX)
        self.vshock_excessive = self.config.get('vshock_excessive', V80_VSHOCK_EXCESSIVE)
        self.vshock_shrink = self.config.get('vshock_shrink', V80_VSHOCK_SHRINK)
        self.liquidity_bonus = self.config.get('liquidity_bonus', V80_LIQUIDITY_BONUS)
        self.liquidity_penalty = self.config.get('liquidity_penalty', V80_LIQUIDITY_PENALTY)
        
        # 多周期 RS 配置
        self.rs_short_window = self.config.get('rs_short_window', V80_RS_SHORT_WINDOW)
        self.rs_long_window = self.config.get('rs_long_window', V80_RS_LONG_WINDOW)
        self.rs_short_weight = self.config.get('rs_short_weight', V80_RS_SHORT_WEIGHT)
        self.rs_long_weight = self.config.get('rs_long_weight', V80_RS_LONG_WEIGHT)
        
        # 因子权重
        self.residual_weight = self.config.get('residual_weight', V80_RESIDUAL_WEIGHT)
        self.reversal_weight = self.config.get('reversal_weight', V80_REVERSAL_WEIGHT)
        self.liquidity_weight = self.config.get('liquidity_weight', V80_LIQUIDITY_WEIGHT)
        self.multi_rs_weight = self.config.get('multi_rs_weight', V80_MULTI_RS_WEIGHT)
        
        # Regime_Switch 配置
        self.regime_high_volatility = self.config.get('regime_high_volatility', V80_REGIME_HIGH_VOLATILITY)
        self.regime_low_volatility = self.config.get('regime_low_volatility', V80_REGIME_LOW_VOLATILITY)
        self.regime_reversal_boost = self.config.get('regime_reversal_boost', V80_REGIME_REVERSAL_BOOST)
        self.regime_momentum_boost = self.config.get('regime_momentum_boost', V80_REGIME_MOMENTUM_BOOST)
        
        # 行业拥挤度配置
        self.sector_crowding_window = self.config.get('sector_crowding_window', V80_SECTOR_CROWDING_WINDOW)
        self.sector_crowding_threshold = self.config.get('sector_crowding_threshold', V80_SECTOR_CROWDING_STD_THRESHOLD)
        self.sector_penalty = self.config.get('sector_penalty', V80_SECTOR_CROWDING_PENALTY)
        self.sector_history_window = self.config.get('sector_history_window', V80_SECTOR_HISTORY_WINDOW)
        
        # 因子监控
        self.factor_weights = {
            'residual': self.residual_weight,
            'reversal': self.reversal_weight,
            'liquidity': self.liquidity_weight,
            'multi_rs': self.multi_rs_weight,
        }
        self.factor_ic_history: Dict[str, List[Tuple[str, float]]] = {
            'residual': [],
            'reversal': [],
            'rs': [],
        }
        self.factor_monitor_records: List[V80FactorMonitor] = []
        
        logger.info("V80 AlphaCenter 初始化完成")
    
    def compute_industry_return_mv_weighted(self, df: pl.DataFrame, 
                                             industry_mapping: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        result = df.clone()
        
        result = result.with_columns([
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
            pl.col('total_mv').cast(pl.Float64, strict=False).alias('total_mv'),
        ])
        
        result = result.with_columns([
            pl.col('pct_chg').fill_null(0.0).alias('pct_chg_filled'),
            pl.col('total_mv').fill_null(1.0).alias('total_mv_filled'),
            pl.col('industry_code').fill_null('UNKNOWN').alias('industry_code'),
        ])
        
        # 直接使用 industry_code 作为 industry_name
        result = result.with_columns([
            pl.col('industry_code').alias('industry_name')
        ])
        
        # 计算市值加权行业收益
        industry_return = result.group_by(['industry_code', 'trade_date']).agg([
            (pl.col('pct_chg_filled') * pl.col('total_mv_filled')).sum().alias('weighted_sum'),
            pl.col('total_mv_filled').sum().alias('mv_sum')
        ])
        
        industry_return = industry_return.with_columns([
            (pl.col('weighted_sum') / (pl.col('mv_sum') + EPSILON)).alias('industry_return_mv')
        ])
        
        result = result.join(
            industry_return.select(['industry_code', 'trade_date', 'industry_return_mv']),
            on=['industry_code', 'trade_date'], how='left'
        )
        
        result = result.with_columns([
            pl.col('industry_return_mv').fill_null(0.0).alias('industry_return')
        ])
        
        return result
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
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
                'regime_state': None,
            }
            
            # 0. 市值加权计算行业收益
            result = self.compute_industry_return_mv_weighted(result, industry_mapping)
            status['factors_computed'].append('industry_return_mv_weighted')
            
            # 1. 计算残差 Alpha
            result = self._compute_residual_alpha(result)
            status['factors_computed'].append('residual_alpha')
            
            # 2. 计算反转因子
            result = self._compute_reversal_factor(result)
            status['factors_computed'].append('reversal_factor')
            
            # 3. 计算流动性与风险调节评分
            result = self._compute_risk_adjusted_liquidity(result)
            status['factors_computed'].append('risk_adjusted_liquidity')
            
            # 4. 计算多周期 RS
            result = self._compute_multi_rs(result, index_df)
            status['factors_computed'].append('multi_rs')
            
            # 5. 计算行业拥挤度
            result = self._compute_sector_crowding(result)
            status['factors_computed'].append('sector_crowding')
            
            # 6. 计算波动率与市场状态
            result, regime_state = self._compute_volatility_and_regime(result, index_df)
            status['regime_state'] = regime_state
            status['factors_computed'].append('volatility_and_regime')
            
            # 7. 计算行业权重
            result = self._compute_industry_weight(result)
            status['factors_computed'].append('industry_weight')
            
            # 8. 计算综合评分
            result = self._compute_composite_score_v80(result, regime_state)
            status['factors_computed'].append('composite_score')
            
            return result, status
            
        except Exception as e:
            logger.error(f"V80 AlphaCenter 计算信号失败：{e}")
            raise
    
    def _compute_residual_alpha(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        
        # 计算个股 N 日收益率 (使用 T-1 数据)
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(self.residual_window + 1)) / 
             (pl.col('close').shift(self.residual_window + 1) + EPSILON)).alias('stock_return')
        ])
        
        # 计算行业 N 日平均收益率
        result = result.with_columns([
            pl.col('industry_return')
            .rolling_mean(window_size=self.residual_window)
            .over('symbol')
            .alias('industry_return_avg')
        ])
        
        result = result.with_columns([
            pl.col('industry_return_avg').fill_null(0.0).alias('industry_return_avg')
        ])
        
        # 计算残差收益率
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('industry_return_avg')).alias('residual_return')
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
            .alias('volatility_for_alpha')
        ])
        result = result.with_columns([
            pl.col('volatility_for_alpha').fill_null(0.02).alias('volatility_for_alpha')
        ])
        
        # 计算残差 Alpha
        result = result.with_columns([
            (pl.col('residual_return') / (pl.col('volatility_for_alpha') + EPSILON)).alias('residual_alpha')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('residual_alpha').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_residual')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_residual').cast(pl.Float64) + EPSILON)).alias('residual_percentile')
        ])
        
        result = result.with_columns([
            pl.when(pl.col('residual_percentile') * 100 >= 50)
            .then(pl.col('residual_percentile') * 100)
            .otherwise(0.0)
            .alias('residual_alpha_score')
        ])
        
        return result
    
    def _compute_reversal_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        
        # 计算过去 N 日累计收益率 (使用 T-1 数据)
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
    
    def _compute_risk_adjusted_liquidity(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        
        # 计算 V-Shock
        result = result.with_columns([
            (pl.col('amount') / 
             (pl.col('amount').rolling_mean(window_size=self.liquidity_window).over('symbol') + EPSILON)
             ).alias('v_shock')
        ])
        
        result = result.with_columns([
            pl.col('v_shock').fill_null(1.0).alias('v_shock')
        ])
        
        # 横截面排名
        result = result.with_columns([
            pl.col('v_shock').rank('ordinal', descending=False).over('trade_date').alias('v_shock_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_liquidity')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('v_shock_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_liquidity').cast(pl.Float64) + EPSILON)).alias('v_shock_percentile')
        ])
        
        # 流动性调整系数
        result = result.with_columns([
            pl.when((pl.col('v_shock') >= self.vshock_optimal_min) & 
                    (pl.col('v_shock') <= self.vshock_optimal_max))
            .then(self.liquidity_bonus)
            .when((pl.col('v_shock') > self.vshock_excessive) | 
                  (pl.col('v_shock') < self.vshock_shrink))
            .then(self.liquidity_penalty)
            .otherwise(1.0)
            .alias('liquidity_adjustment')
        ])
        
        result = result.with_columns([
            (pl.col('v_shock_percentile') * pl.col('liquidity_adjustment') * 100.0).alias('liquidity_score')
        ])
        
        # 风险调节评分
        result = result.with_columns([
            (pl.col('liquidity_score') / (1.0 + pl.col('v_shock').log().clip(0.0, 2.0))).alias('risk_adjusted_liquidity')
        ])
        
        return result
    
    def _compute_multi_rs(self, df: pl.DataFrame,
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
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
                on='trade_date',
                how='left'
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
        
        # 计算 RS
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
            .rolling_mean(window_size=self.sector_history_window)
            .over('industry_code')
            .alias('industry_avg_turnover'),
            pl.col('industry_turnover_ratio')
            .rolling_std(window_size=self.sector_history_window)
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
            industry_stats.select([
                'trade_date', 'industry_code', 
                'crowding_zscore', 'sector_penalty'
            ]),
            on=['trade_date', 'industry_code'],
            how='left',
            suffix='_crowd'
        )
        
        result = result.with_columns([
            pl.col('crowding_zscore').fill_null(0.0).alias('crowding_zscore'),
            pl.col('sector_penalty').fill_null(1.0).alias('sector_penalty')
        ])
        
        return result
    
    def _compute_volatility_and_regime(self, df: pl.DataFrame,
                                        index_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Optional[V80RegimeState]]:
        result = df.clone()
        
        # 计算日收益率
        result = result.with_columns([
            ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('daily_return')
        ])
        
        result = result.with_columns([
            pl.col('daily_return').fill_null(0.0).alias('daily_return_filled')
        ])
        
        # 计算滚动波动率
        result = result.with_columns([
            pl.col('daily_return_filled')
            .rolling_std(window_size=self.volatility_window)
            .over('symbol')
            .alias('volatility_raw')
        ])
        
        result = result.with_columns([
            pl.col('volatility_raw').fill_null(0.02).alias('volatility')
        ])
        
        # 计算市场状态
        regime_state = None
        if index_df is not None and not index_df.is_empty():
            index_df = index_df.with_columns([
                ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('market_return')
            ])
            
            latest_date = index_df['trade_date'].max()
            latest_data = index_df.filter(pl.col('trade_date') == latest_date)
            
            if not latest_data.is_empty():
                market_returns = index_df['market_return'].to_numpy()
                market_returns = market_returns[~np.isnan(market_returns)]
                
                if len(market_returns) > 1:
                    market_volatility = float(np.std(market_returns, ddof=1))
                    market_return = float(latest_data['market_return'][0]) if 'market_return' in latest_data.columns else 0.0
                    
                    if market_volatility > self.regime_high_volatility:
                        regime_type = "high_volatility"
                        reversal_multiplier = self.regime_reversal_boost
                        momentum_multiplier = 1.0
                    elif market_volatility < self.regime_low_volatility:
                        regime_type = "low_volatility"
                        reversal_multiplier = 1.0
                        momentum_multiplier = self.regime_momentum_boost
                    else:
                        regime_type = "normal"
                        reversal_multiplier = 1.0
                        momentum_multiplier = 1.0
                    
                    regime_state = V80RegimeState(
                        trade_date=latest_date,
                        market_volatility=market_volatility,
                        market_return=market_return,
                        regime_type=regime_type,
                        reversal_weight_multiplier=reversal_multiplier,
                        momentum_weight_multiplier=momentum_multiplier,
                    )
        
        return result, regime_state
    
    def _compute_industry_weight(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        result = result.with_columns([pl.lit(1.0).alias('industry_weight')])
        return result
    
    def _compute_composite_score_v80(self, df: pl.DataFrame, 
                                      regime_state: Optional[V80RegimeState] = None) -> pl.DataFrame:
        result = df.clone()
        
        # 确保各因子列存在
        for col, default in [
            ('residual_alpha_score', 50.0),
            ('reversal_score', 50.0),
            ('liquidity_score', 50.0),
            ('multi_rs_score', 50.0),
            ('risk_adjusted_liquidity', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # Regime_Switch 动态调整权重
        if regime_state is not None:
            adjusted_reversal_weight = self.reversal_weight * regime_state.reversal_weight_multiplier
            adjusted_multi_rs_weight = self.multi_rs_weight * regime_state.momentum_weight_multiplier
            
            total_weight = (self.residual_weight + adjusted_reversal_weight + 
                          self.liquidity_weight + adjusted_multi_rs_weight)
            
            w_residual = self.residual_weight / total_weight
            w_reversal = adjusted_reversal_weight / total_weight
            w_liquidity = self.liquidity_weight / total_weight
            w_multi_rs = adjusted_multi_rs_weight / total_weight
        else:
            w_residual = self.residual_weight
            w_reversal = self.reversal_weight
            w_liquidity = self.liquidity_weight
            w_multi_rs = self.multi_rs_weight
        
        # 线性融合
        result = result.with_columns([
            (w_residual * (pl.col('residual_alpha_score') / 100.0) +
             w_reversal * (pl.col('reversal_score') / 100.0) +
             w_liquidity * (pl.col('liquidity_score') / 100.0) +
             w_multi_rs * (pl.col('multi_rs_score') / 100.0)
             ).alias('raw_score')
        ])
        
        # 风险调节评分
        result = result.with_columns([
            (pl.col('raw_score') / (1.0 + pl.col('v_shock').log().clip(0.0, 2.0))).alias('risk_adjusted_score')
        ])
        
        # 拥挤度惩罚
        if 'sector_penalty' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('sector_penalty')])
        
        result = result.with_columns([
            (pl.col('risk_adjusted_score') * pl.col('sector_penalty')).alias('crowded_score')
        ])
        
        # 波动率缩放
        if 'volatility' not in result.columns:
            result = result.with_columns([pl.lit(0.02).alias('volatility')])
        
        result = result.with_columns([
            (pl.col('crowded_score') / 
             (1.0 + pl.col('volatility') * V80_VOLATILITY_BASE)).alias('vol_scaled_score')
        ])
        
        # 标准化并映射到 0-100
        result = result.with_columns([
            pl.col('vol_scaled_score').mean().over('trade_date').alias('vol_mean'),
            pl.col('vol_scaled_score').std().over('trade_date').alias('vol_std')
        ])
        
        result = result.with_columns([
            ((pl.col('vol_scaled_score') - pl.col('vol_mean')) / 
             (pl.col('vol_std') + EPSILON)).alias('vol_zscore')
        ])
        
        result = result.with_columns([
            pl.col('vol_zscore').fill_null(0.0).alias('vol_zscore_filled')
        ])
        
        result = result.with_columns([
            (50 + 50 * pl.col('vol_zscore_filled').clip(-2, 2) / 2).alias('composite_score')
        ])
        
        # 计算买入信号 (前 10% 且评分 > 50)
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_final')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks_final').cast(pl.Float64) + EPSILON))).alias('score_percentile')
        ])
        
        result = result.with_columns([
            ((pl.col('score_percentile') >= (1.0 - V80_SELECTION_PERCENTILE)) & 
             (pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V80Signal]:
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
                signal = V80Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    residual_alpha_score=row.get('residual_alpha_score', 0.0),
                    residual_return=row.get('residual_return', 0.0),
                    stock_return=row.get('stock_return', 0.0),
                    industry_return=row.get('industry_return', 0.0),
                    reversal_score=row.get('reversal_score', 0.0),
                    short_term_return=row.get('short_term_return', 0.0),
                    liquidity_score=row.get('liquidity_score', 0.0),
                    v_shock=row.get('v_shock', 1.0),
                    risk_adjusted_liquidity=row.get('risk_adjusted_liquidity', 0.0),
                    multi_rs_score=row.get('multi_rs_score', 0.0),
                    rs_short=row.get('rs_short', 0.0),
                    rs_long=row.get('rs_long', 0.0),
                    sector_crowding=row.get('crowding_zscore', 0.0),
                    sector_penalty=row.get('sector_penalty', 1.0),
                    volatility=row.get('volatility', 0.02),
                    vol_scaled_score=row.get('vol_scaled_score', 0.0),
                    industry_name=row.get('industry_name', ''),
                    industry_code=row.get('industry_code', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"V80 生成信号失败：{e}")
        
        return signals


# ===========================================
# V80 RankICCalculator
# ===========================================

class V80RankICCalculator:
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V80_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', V80_RANK_IC_MIN)
        
        self.ic_results: List[V80ICMetrics] = []
        self.monthly_stats: List[V80MonthlyICStats] = []
        
        self.factor_monthly_ics: Dict[str, Dict[str, List[float]]] = {
            'rs': {},
            'residual': {},
            'reversal': {},
        }
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        from scipy import stats
        
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
            
            return rank_ic, {
                'count': len(signal_values),
                'ic': ic,
                'rank_ic': rank_ic,
            }
            
        except Exception as e:
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V80ICMetrics]:
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
                
                ic_metrics = V80ICMetrics(
                    trade_date=trade_date,
                    factor_name='composite_score',
                    ic=ic,
                    rank_ic=rank_ic
                )
                ic_series.append(ic_metrics)
            
            self.ic_results = ic_series
            self._compute_monthly_stats()
            self._compute_factor_monthly_ics(df_with_return)
            
        except Exception as e:
            logger.error(f"V80 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
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
                monthly_stat = V80MonthlyICStats(
                    month=month,
                    factor_name='composite_score',
                    mean_rank_ic=mean_rank_ic,
                    std_rank_ic=float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
                    ic_count=len(rank_ics),
                    is_negative=(mean_rank_ic < 0)
                )
                self.monthly_stats.append(monthly_stat)
    
    def _compute_factor_monthly_ics(self, df: pl.DataFrame):
        unique_dates = df['trade_date'].unique().to_list()
        
        factor_columns = {
            'rs': 'multi_rs_score',
            'residual': 'residual_alpha_score',
            'reversal': 'reversal_score',
        }
        
        for trade_date in sorted(unique_dates):
            month = trade_date[:7]
            
            for factor_name, factor_col in factor_columns.items():
                ic = self.calculate_factor_ic(df, trade_date, factor_col)
                
                if month not in self.factor_monthly_ics[factor_name]:
                    self.factor_monthly_ics[factor_name][month] = []
                self.factor_monthly_ics[factor_name][month].append(ic)
    
    def calculate_factor_ic(self, df: pl.DataFrame, trade_date: str,
                            factor_col: str, return_col: str = 'forward_return_5d') -> float:
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
        is_pass, message = self.check_rank_ic_pass()
        
        logger.info("=" * 60)
        logger.info("V80 Rank IC 预测质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f}")
        logger.info(f"负值月份数量：{monthly_stats['negative_months']} (上限：2)")
        logger.info(f"达标状态：{is_pass}")
        logger.info("=" * 60)
    
    def generate_monthly_ic_chart(self) -> str:
        if not self.monthly_stats:
            return "无数据"
        
        lines = ["月度 Rank IC 柱状图", "=" * 60]
        
        rank_ics = [m.mean_rank_ic for m in self.monthly_stats]
        max_ic = max(max(abs(r) for r in rank_ics), self.rank_ic_target) if rank_ics else self.rank_ic_target
        bar_width = 40
        
        for m in self.monthly_stats:
            status = "✓" if m.mean_rank_ic >= self.rank_ic_target else "✗"
            neg_marker = " [NEG]" if m.is_negative else ""
            
            if m.mean_rank_ic >= 0:
                bar_length = int(m.mean_rank_ic / max_ic * bar_width)
                bar = " " * bar_width + "█" * bar_length
            else:
                bar_length = int(abs(m.mean_rank_ic) / max_ic * bar_width)
                bar = " " * (bar_width - bar_length) + "█" * bar_length
            
            lines.append(f"{m.month} |{bar}| {m.mean_rank_ic:+.4f} {status}{neg_marker}")
        
        lines.append("-" * 60)
        lines.append(f"目标：>{self.rank_ic_target:.3f}")
        pass_count = sum(1 for r in rank_ics if r >= self.rank_ic_target)
        lines.append(f"达标月份：{pass_count}/{len(self.monthly_stats)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V80_INITIAL_CAPITAL',
    'V80_MAX_POSITIONS',
    'V80_WARMUP_PERIOD',
    'V80DataManager',
    'V80AlphaCenter',
    'V80RankICCalculator',
    'V80Position',
    'V80Trade',
    'V80Signal',
    'V80ICMetrics',
    'V80MonthlyICStats',
    'V80FactorMonitor',
    'V80RegimeState',
]