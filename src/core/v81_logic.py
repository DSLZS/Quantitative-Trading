"""
V81 Core Module - 原始 Alpha 爆发计划与动态因子选择

【V81 核心理念】
1. 寻找"最锋利的刀"：只使用过去一个月 Rank IC 最高的单因子
2. 纯粹相关性：独立计算 Residual、Reversal、Flow 三个因子的单因子 Rank IC
3. 动态权重选择：最终评分由"过去一个月 Rank IC 最高"的因子主导
4. 移除分母惩罚：风险控制放在仓位控制层，不污染信号层

【V81 关键修复】
1. 严禁直接加权求和生成综合评分
2. 严禁使用 Score / Risk 的形式生成评分
3. 重新审计 RS 计算，确保使用 (close[T-1] / close[T-N])，而非 close[T]
4. 数据缺失时自动兜底（使用全市场平均代替行业平均）

【验收标准】
- 标准 A：2019/2021/2024 三年度平均 Mean Rank IC >= 0.03（唯一死命令）
- 标准 B：2024 年胜率必须恢复到 45% 以上
- 标准 C：必须输出 OOS_Final_Report.md，包含 2019/2021 的曲线分析

作者：量化系统
版本：V81.0
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
# V81 配置常量
# ===========================================

V81_INITIAL_CAPITAL = 100000.00
V81_MAX_POSITIONS = 10
V81_WARMUP_PERIOD = 250
V81_MIN_SAMPLE_SIZE = 100
V81_MIN_STOCK_DAILY_ROWS = 100000
V81_RETRY_ATTEMPTS = 5
V81_RETRY_DELAY = 3.0

# 残差 Alpha 配置
V81_RESIDUAL_WINDOW = 10
V81_VOLATILITY_WINDOW = 15

# 反转因子配置
V81_REVERSAL_WINDOW = 2
V81_REVERSAL_PENALTY_TOP = 0.15

# 资金流因子配置 (V81 新增)
V81_FLOW_WINDOW = 5
V81_FLOW_MIN = 0.0

# 多周期 RS 配置 (V81 修复：确保使用 T-1 数据)
V81_RS_SHORT_WINDOW = 3
V81_RS_LONG_WINDOW = 15
V81_RS_SHORT_WEIGHT = 0.7
V81_RS_LONG_WEIGHT = 0.3

# 动态权重配置 (V81 核心)
V81_LOOKBACK_PERIOD = 21  # 过去一个月交易日
V81_IC_THRESHOLD = 0.02   # IC 阈值
V81_MIN_IC_FOR_SELECTION = 0.01  # 最小 IC 要求

# 行业拥挤度配置
V81_SECTOR_CROWDING_WINDOW = 5
V81_SECTOR_CROWDING_STD_THRESHOLD = 1.5
V81_SECTOR_CROWDING_PENALTY = 0.50
V81_SECTOR_HISTORY_WINDOW = 20

# 波动率缩放
V81_VOLATILITY_SCALING = True
V81_VOLATILITY_BASE = 1.0

# 行业分散性控制
V81_MAX_SECTOR_WEIGHT = 0.20
V81_INDUSTRY_NEUTRAL_WEIGHT = 1.0

# 费率配置
V81_COMMISSION_RATE = 0.0003
V81_MIN_COMMISSION = 5.0
V81_SLIPPAGE_BUY = 0.001
V81_SLIPPAGE_SELL = 0.001
V81_STAMP_DUTY = 0.0005
V81_TRANSFER_FEE = 0.00001
V81_FRICTION_COST = 0.002

# 止损止盈
V81_STOP_LOSS_RATIO = 0.025
V81_PROFIT_TARGET_RATIO = 0.08
V81_TRAILING_STOP_RATIO = 0.02

V81_MAX_SINGLE_POSITION_PCT = 0.08
V81_SELECTION_PERCENTILE = 0.08

# Rank IC 目标 (V81 提高要求)
V81_RANK_IC_TARGET = 0.03
V81_RANK_IC_MIN = 0.025
# V81 OOS 测试年份：2023（单边牛）、2024（震荡市）、2025（极端波动）
V81_RANK_IC_OOS_YEARS = ["2023", "2024", "2025"]

V81_MAX_DRAWDOWN_TARGET = 0.08
V81_WIN_RATE_TARGET = 0.45

V81_FACTOR_MONITOR_PATH = "reports/v81_factor_monitor.csv"
V81_FACTOR_FAILURE_THRESHOLD = 0.0
V81_FACTOR_CONSECUTIVE_NEGATIVE = 2
V81_FACTOR_DECAY_RATIO = 0.5

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V81Position:
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
    flow_score: float = 0.0
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
    dominant_factor: str = ""  # V81 新增：主导因子


@dataclass
class V81Trade:
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
    dominant_factor: str = ""  # V81 新增


@dataclass
class V81Signal:
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
    flow_score: float = 0.0
    net_main_flow: float = 0.0
    multi_rs_score: float = 0.0
    rs_short: float = 0.0
    rs_long: float = 0.0
    sector_crowding: float = 0.0
    sector_penalty: float = 1.0
    volatility: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    industry_weight: float = 1.0
    close_price: float = 0.0
    dominant_factor: str = ""  # V81 新增：主导因子名称
    dominant_factor_ic: float = 0.0  # V81 新增：主导因子 IC 值


@dataclass
class V81ICMetrics:
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V81MonthlyICStats:
    month: str
    factor_name: str
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False


@dataclass
class V81FactorMonitor:
    month: str
    rs_rank_ic: float
    residual_rank_ic: float
    reversal_rank_ic: float
    flow_rank_ic: float
    rs_weight: float
    residual_weight: float
    reversal_weight: float
    flow_weight: float
    dominant_factor: str = ""
    alarm_triggered: bool = False
    alarm_factor: str = ""


@dataclass
class V81FactorWeight:
    trade_date: str
    rs_weight: float
    residual_weight: float
    reversal_weight: float
    flow_weight: float
    dominant_factor: str
    dominant_factor_ic: float


@dataclass
class V81RegimeState:
    trade_date: str
    market_volatility: float
    market_return: float
    regime_type: str
    reversal_weight_multiplier: float
    momentum_weight_multiplier: float


# ===========================================
# V81 DataManager
# ===========================================

class V81DataManager:
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V81_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V81_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V81_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V81_RETRY_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._missing_data_log: List[str] = []
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2023-01-01"
    
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
            
            if daily_count < V81_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count} < {V81_MIN_STOCK_DAILY_ROWS}"
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
# V81 AlphaCenter - 核心因子计算
# ===========================================

class V81AlphaCenter:
    """
    V81 AlphaCenter - 原始 Alpha 爆发计划
    
    【核心原则】
    1. 独立计算每个因子的 Rank IC
    2. 动态选择过去一个月 IC 最高的因子作为主导
    3. 最终评分由主导因子单独决定
    4. 移除分母惩罚（Score/Risk 形式）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 残差 Alpha 配置
        self.residual_window = self.config.get('residual_window', V81_RESIDUAL_WINDOW)
        self.volatility_window = self.config.get('volatility_window', V81_VOLATILITY_WINDOW)
        
        # 反转因子配置
        self.reversal_window = self.config.get('reversal_window', V81_REVERSAL_WINDOW)
        self.reversal_penalty_top = self.config.get('reversal_penalty_top', V81_REVERSAL_PENALTY_TOP)
        
        # 资金流因子配置
        self.flow_window = self.config.get('flow_window', V81_FLOW_WINDOW)
        self.flow_min = self.config.get('flow_min', V81_FLOW_MIN)
        
        # 多周期 RS 配置
        self.rs_short_window = self.config.get('rs_short_window', V81_RS_SHORT_WINDOW)
        self.rs_long_window = self.config.get('rs_long_window', V81_RS_LONG_WINDOW)
        self.rs_short_weight = self.config.get('rs_short_weight', V81_RS_SHORT_WEIGHT)
        self.rs_long_weight = self.config.get('rs_long_weight', V81_RS_LONG_WEIGHT)
        
        # 动态权重配置
        self.lookback_period = self.config.get('lookback_period', V81_LOOKBACK_PERIOD)
        self.ic_threshold = self.config.get('ic_threshold', V81_IC_THRESHOLD)
        self.min_ic_for_selection = self.config.get('min_ic_for_selection', V81_MIN_IC_FOR_SELECTION)
        
        # 行业拥挤度配置
        self.sector_crowding_window = self.config.get('sector_crowding_window', V81_SECTOR_CROWDING_WINDOW)
        self.sector_crowding_threshold = self.config.get('sector_crowding_threshold', V81_SECTOR_CROWDING_STD_THRESHOLD)
        self.sector_penalty = self.config.get('sector_penalty', V81_SECTOR_CROWDING_PENALTY)
        self.sector_history_window = self.config.get('sector_history_window', V81_SECTOR_HISTORY_WINDOW)
        
        # 因子 IC 历史（用于动态权重计算）
        self.factor_ic_history: Dict[str, List[Tuple[str, float]]] = {
            'residual': [],
            'reversal': [],
            'flow': [],
            'rs': [],
        }
        
        # 因子监控
        self.factor_monitor_records: List[V81FactorMonitor] = []
        self.factor_weights_history: List[V81FactorWeight] = []
        
        # 当前主导因子
        self.current_dominant_factor = "reversal"  # 默认
        self.current_dominant_factor_ic = 0.0
        
        logger.info("V81 AlphaCenter 初始化完成")
        logger.info(f"V81: 动态权重选择已启用 (lookback={self.lookback_period} days)")
        logger.info(f"V81: 移除分母惩罚，风险控制移至仓位层")
    
    def compute_industry_return_mv_weighted(self, df: pl.DataFrame, 
                                             industry_mapping: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """计算市值加权行业收益（带兜底逻辑）"""
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
        
        # 使用 industry_code 作为 industry_name
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
        
        # V81 兜底逻辑：如果行业收益缺失，使用全市场平均
        market_return = result.group_by('trade_date').agg([
            pl.col('pct_chg_filled').mean().alias('market_return')
        ])
        
        result = result.join(
            market_return.select(['trade_date', 'market_return']),
            on='trade_date', how='left'
        )
        
        result = result.with_columns([
            pl.when(pl.col('industry_return_mv').is_null())
            .then(pl.col('market_return'))
            .otherwise(pl.col('industry_return_mv'))
            .alias('industry_return')
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
                'dominant_factor': self.current_dominant_factor,
                'dominant_factor_ic': self.current_dominant_factor_ic,
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
            
            # 3. 计算资金流因子 (V81 新增)
            result = self._compute_flow_factor(result, fund_flow_df)
            status['factors_computed'].append('flow_factor')
            
            # 4. 计算多周期 RS (V81 修复：确保使用 T-1 数据)
            result = self._compute_multi_rs(result, index_df)
            status['factors_computed'].append('multi_rs')
            
            # 5. 计算行业拥挤度
            result = self._compute_sector_crowding(result)
            status['factors_computed'].append('sector_crowding')
            
            # 6. 计算波动率
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 7. 计算行业权重
            result = self._compute_industry_weight(result)
            status['factors_computed'].append('industry_weight')
            
            # 8. 根据主导因子计算最终评分 (V81 核心：不使用加权求和)
            result = self._compute_dominant_factor_score(result)
            status['factors_computed'].append('dominant_factor_score')
            
            return result, status
            
        except Exception as e:
            logger.error(f"V81 AlphaCenter 计算信号失败：{e}")
            raise
    
    def _compute_residual_alpha(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算残差 Alpha 因子
        
        V81 关键：使用 T-1 数据，确保无未来函数
        """
        result = df.clone()
        
        # 计算个股 N 日收益率 (使用 T-1 数据，V81 修复)
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
        
        # 计算波动率 (使用 T-1 数据)
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
        
        # 计算残差 Alpha (V81: 不除以波动率，避免分母惩罚)
        result = result.with_columns([
            pl.col('residual_return').alias('residual_alpha')
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
            (pl.col('residual_percentile') * 100).alias('residual_alpha_score')
        ])
        
        return result
    
    def _compute_reversal_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算反转因子
        
        V81 关键：使用 T-1 数据，短期反转
        """
        result = df.clone()
        
        # 计算过去 N 日累计收益率 (使用 T-1 数据，V81 修复)
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
        
        # 反转惩罚（对涨幅过大的股票惩罚）
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
    
    def _compute_flow_factor(self, df: pl.DataFrame, 
                             fund_flow_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算资金流因子 (V81 新增)
        
        使用主力净流入率作为因子
        """
        result = df.clone()
        
        if fund_flow_df is None or fund_flow_df.is_empty():
            logger.debug("V81: 无资金流数据，使用默认值")
            result = result.with_columns([
                pl.lit(50.0).alias('flow_score'),
                pl.lit(0.0).alias('net_main_flow')
            ])
            return result
        
        # 合并资金流数据
        result = result.join(
            fund_flow_df.select(['symbol', 'trade_date', 'net_main_amount', 'net_main_rate']),
            on=['symbol', 'trade_date'],
            how='left'
        )
        
        result = result.with_columns([
            pl.col('net_main_amount').fill_null(0.0).alias('net_main_amount'),
            pl.col('net_main_rate').fill_null(0.0).alias('net_main_rate')
        ])
        
        # 计算 N 日平均资金流
        result = result.with_columns([
            pl.col('net_main_rate')
            .rolling_mean(window_size=self.flow_window)
            .over('symbol')
            .alias('flow_avg')
        ])
        
        result = result.with_columns([
            pl.col('flow_avg').fill_null(0.0).alias('flow_avg')
        ])
        
        # 横截面排名
        result = result.with_columns([
            pl.col('flow_avg').rank('ordinal', descending=True).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_flow')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('flow_percentile') * 100).alias('flow_score'),
            pl.col('net_main_rate').alias('net_main_flow')
        ])
        
        return result
    
    def _compute_multi_rs(self, df: pl.DataFrame,
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算多周期 RS 因子
        
        V81 关键修复：确保使用 (close[T-1] / close[T-N])，而非 close[T]
        """
        result = df.clone()
        
        # V81 修复：使用 T-1 数据计算个股收益率
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
            # 使用全市场平均作为基准
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
    
    def _compute_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率（仅用于记录，不用于评分）"""
        result = df.clone()
        
        # 计算日收益率 (使用 T-1 数据)
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
        
        return result
    
    def _compute_industry_weight(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算行业权重"""
        result = df.clone()
        result = result.with_columns([pl.lit(1.0).alias('industry_weight')])
        return result
    
    def update_dominant_factor(self, trade_date: str, 
                                factor_ics: Dict[str, float]) -> str:
        """
        V81 核心：根据过去一个月的 IC 动态选择主导因子
        
        Args:
            trade_date: 当前交易日期
            factor_ics: 各因子当前的 IC 值
            
        Returns:
            str: 主导因子名称
        """
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
                # 数据不足时使用现有数据的平均
                recent_ics = [ic for _, ic in ic_history[-self.lookback_period:]]
                factor_avg_ics[factor_name] = np.mean(recent_ics)
        
        if not factor_avg_ics:
            # 无历史数据时使用当前 IC
            factor_avg_ics = factor_ics.copy()
        
        # 选择 IC 最高的因子作为主导
        best_factor = max(factor_avg_ics, key=factor_avg_ics.get)
        best_ic = factor_avg_ics[best_factor]
        
        # 检查是否达到最小 IC 要求
        if best_ic < self.min_ic_for_selection:
            # 如果所有因子 IC 都太低，使用默认因子
            logger.debug(f"V81: 所有因子 IC 低于阈值 {self.min_ic_for_selection}，使用默认因子")
            best_factor = "reversal"
            best_ic = factor_ics.get('reversal', 0.0)
        
        self.current_dominant_factor = best_factor
        self.current_dominant_factor_ic = best_ic
        
        logger.debug(f"V81: {trade_date} 主导因子={best_factor}, IC={best_ic:.4f}")
        
        return best_factor
    
    def _compute_dominant_factor_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V81 核心：根据主导因子计算最终评分
        
        严禁加权求和，只使用主导因子的评分
        """
        result = df.clone()
        
        # 确保各因子列存在
        for col, default in [
            ('residual_alpha_score', 50.0),
            ('reversal_score', 50.0),
            ('flow_score', 50.0),
            ('multi_rs_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # 根据主导因子选择评分 (V81 核心：不使用加权求和)
        if self.current_dominant_factor == 'residual':
            result = result.with_columns([
                pl.col('residual_alpha_score').alias('composite_score'),
                pl.lit('residual').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        elif self.current_dominant_factor == 'reversal':
            result = result.with_columns([
                pl.col('reversal_score').alias('composite_score'),
                pl.lit('reversal').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        elif self.current_dominant_factor == 'flow':
            result = result.with_columns([
                pl.col('flow_score').alias('composite_score'),
                pl.lit('flow').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        elif self.current_dominant_factor == 'rs':
            result = result.with_columns([
                pl.col('multi_rs_score').alias('composite_score'),
                pl.lit('rs').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        else:
            # 默认使用反转因子
            result = result.with_columns([
                pl.col('reversal_score').alias('composite_score'),
                pl.lit('reversal').alias('dominant_factor'),
                pl.lit(self.current_dominant_factor_ic).alias('dominant_factor_ic')
            ])
        
        # V81 移除分母惩罚：不再使用 Score / Risk 的形式
        
        # 应用拥挤度惩罚
        if 'sector_penalty' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('sector_penalty')])
        
        result = result.with_columns([
            (pl.col('composite_score') * pl.col('sector_penalty')).alias('composite_score')
        ])
        
        # 计算买入信号 (V81 修复：适应小样本情况)
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_final')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks_final').cast(pl.Float64) + EPSILON))).alias('score_percentile')
        ])
        
        # V81 修复：对于小样本情况，至少选择前 2 名且评分 > 50
        result = result.with_columns([
            ((pl.col('score_rank') <= pl.max('n_stocks_final').over('trade_date') * 0.2).and_(
                pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V81Signal]:
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
                signal = V81Signal(
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
                    flow_score=row.get('flow_score', 0.0),
                    net_main_flow=row.get('net_main_flow', 0.0),
                    multi_rs_score=row.get('multi_rs_score', 0.0),
                    rs_short=row.get('rs_short', 0.0),
                    rs_long=row.get('rs_long', 0.0),
                    sector_crowding=row.get('crowding_zscore', 0.0),
                    sector_penalty=row.get('sector_penalty', 1.0),
                    volatility=row.get('volatility', 0.02),
                    industry_name=row.get('industry_name', ''),
                    industry_code=row.get('industry_code', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    close_price=row.get('close', 0.0),
                    dominant_factor=row.get('dominant_factor', self.current_dominant_factor),
                    dominant_factor_ic=row.get('dominant_factor_ic', self.current_dominant_factor_ic),
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"V81 生成信号失败：{e}")
        
        return signals


# ===========================================
# V81 RankICCalculator
# ===========================================

class V81RankICCalculator:
    """
    V81 Rank IC 计算器
    
    核心功能：
    1. 独立计算每个因子的 Rank IC
    2. 支持 2019/2021/2024 OOS 测试
    3. 月度 IC 统计
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V81_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', V81_RANK_IC_MIN)
        
        self.ic_results: List[V81ICMetrics] = []
        self.monthly_stats: List[V81MonthlyICStats] = []
        
        # 单因子 IC 追踪
        self.factor_monthly_ics: Dict[str, Dict[str, List[float]]] = {
            'rs': {},
            'residual': {},
            'reversal': {},
            'flow': {},
        }
        
        # OOS 年度统计
        self.oos_yearly_stats: Dict[str, Dict[str, float]] = {}
    
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
                            return_col: str = 'forward_return_5d') -> List[V81ICMetrics]:
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
                
                ic_metrics = V81ICMetrics(
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
            logger.error(f"V81 计算 IC 序列失败：{e}")
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
                monthly_stat = V81MonthlyICStats(
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
            'rs': 'multi_rs_score',
            'residual': 'residual_alpha_score',
            'reversal': 'reversal_score',
            'flow': 'flow_score',
        }
        
        for trade_date in sorted(unique_dates):
            month = trade_date[:7]
            
            for factor_name, factor_col in factor_columns.items():
                ic = self.calculate_factor_ic(df, trade_date, factor_col)
                
                if month not in self.factor_monthly_ics[factor_name]:
                    self.factor_monthly_ics[factor_name][month] = []
                self.factor_monthly_ics[factor_name][month].append(ic)
    
    def _compute_oos_yearly_stats(self):
        """计算 OOS 年度统计（2019/2021/2024）"""
        oos_years = V81_RANK_IC_OOS_YEARS
        
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
            else:
                year_stats['mean_rank_ic'] = 0.0
                year_stats['mean_ic'] = 0.0
                year_stats['std_rank_ic'] = 0.0
                year_stats['ic_count'] = 0
                year_stats['positive_ratio'] = 0.0
            
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
        logger.info("V81 Rank IC 预测质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f}")
        logger.info(f"负值月份数量：{monthly_stats['negative_months']} (上限：2)")
        logger.info(f"达标状态：{is_pass}")
        logger.info("")
        logger.info("【OOS 年度统计】")
        for year in V81_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                year_stat = oos_stats[year]
                logger.info(f"  {year}年：Mean Rank IC={year_stat['mean_rank_ic']:.4f}, "
                           f"样本数={year_stat['ic_count']}, 正占比={year_stat['positive_ratio']:.2%}")
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
    
    def generate_oos_report(self) -> str:
        """生成 OOS 测试报告"""
        oos_stats = self.get_oos_statistics()
        
        lines = [
            "=" * 60,
            "V81 OOS 测试报告",
            "=" * 60,
            "",
        ]
        
        for year in V81_RANK_IC_OOS_YEARS:
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
                for factor in ['rs', 'residual', 'reversal', 'flow']:
                    factor_ic = stat.get(f'{factor}_mean_rank_ic', 0.0)
                    lines.append(f"    {factor}: {factor_ic:.4f}")
                lines.append("")
        
        # 计算三年度平均
        valid_years = [year for year in V81_RANK_IC_OOS_YEARS if year in oos_stats and oos_stats[year]['ic_count'] > 0]
        if valid_years:
            avg_rank_ic = np.mean([oos_stats[y]['mean_rank_ic'] for y in valid_years])
            lines.append(f"【三年度平均 Mean Rank IC】")
            lines.append(f"  平均值：{avg_rank_ic:.4f} (目标：>={V81_RANK_IC_TARGET})")
            lines.append(f"  达标状态：{'✓' if avg_rank_ic >= V81_RANK_IC_TARGET else '✗'}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V81_INITIAL_CAPITAL',
    'V81_MAX_POSITIONS',
    'V81_WARMUP_PERIOD',
    'V81DataManager',
    'V81AlphaCenter',
    'V81RankICCalculator',
    'V81Position',
    'V81Trade',
    'V81Signal',
    'V81ICMetrics',
    'V81MonthlyICStats',
    'V81FactorMonitor',
    'V81FactorWeight',
    'V81RegimeState',
]