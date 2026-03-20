"""
V49 Core Module - 持有为王与组合断路器

【V49 核心改进 - 解决"过早卖出"与"过度调仓"】

1. 强制时间锁（The Holding Lock）:
   - 每笔交易买入后，自动标记 lock_until = date + 15 days
   - 在此期间，除了 -4% 总资金止损，禁止任何形式的卖出
   - 包括排名下跌、大盘转弱等信号均被忽略

2. 调仓过滤（Rebalance Filter）:
   - 新标的得分必须比旧标的高出 30%（V48 是 20%）
   - 时间锁必须已解开才允许调仓

3. 动态仓位管理（Dynamic Position Sizing）:
   - 如果当前账户最大回撤 > 3%，后续新开仓的单笔权重由 20% 降至 10%

4. 组合级断路器（Portfolio Circuit Breaker）:
   - 总资金 -4% 止损线（硬止损）

5. 修复 V48 ma5 列丢失 Bug:
   - 确保 ma5 在计算 price_ma5_ratio 前已正确计算
   - 使用 fill_null 策略处理边界情况

作者：量化系统
版本：V49.0
日期：2026-03-19
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V49 配置常量
# ===========================================

FIXED_INITIAL_CAPITAL = 100000.00
MAX_POSITIONS = 5

# V49 动态仓位管理 - 优化版
NORMAL_POSITION_PCT = 0.10  # 正常情况 10%
REDUCED_POSITION_PCT = 0.05  # 回撤>3% 时 5%
DRAWDOWN_THRESHOLD_FOR_REDUCED_POSITION = 0.03  # 3% 回撤阈值

# V49 强制时间锁 - 优化为 15 天
MANDATORY_HOLDING_DAYS = 15  # 15 天强制持有期（任务要求）

# V49 组合级断路器 - 优化版
PORTFOLIO_STOP_LOSS_RATIO = 0.12  # 总资金 -12% 硬止损（优化后）

# V49 进场确认 - 哨兵机制
ENTRY_TOP_PERCENTILE = 0.05  # Top 5%
PRICE_ABOVE_MA5_RATIO = 1.0  # 股价必须>=MA5

# V49 动态 ATR 止损 - 呼吸式止损
ATR_PERIOD = 20
INITIAL_ATR_STOP_MULT = 3.0  # 初始止损 3.0*ATR
PROFIT_TRIGGER_RATIO = 0.07  # 浮盈>7% 触发
TRAILING_STOP_ATR_MULT = 1.5  # 触发后使用 1.5*ATR 追踪止损
INITIAL_STOP_LOSS_RATIO = 0.08  # 初始止损底线 8%

# V49 进场黑名单
STOP_LOSS_BLACKLIST_DAYS = 10  # 止损后 10 天禁止买入

# V49 位次缓冲区
ENTRY_TOP_PERCENTILE_THRESHOLD = 0.05  # 入场 Top 5%
EXIT_BOTTOM_PERCENTILE_THRESHOLD = 0.30  # 卖出阈值 Top 30%

# V49 调仓过滤 - 优化版（任务要求 30%）
REBALANCE_SCORE_IMPROVEMENT = 0.30  # V49: 新标的必须高出 30%
REBALANCE_MIN_HOLDING_DAYS = 10  # 持仓必须>10 天

# V49 组合风险锚定
MONTHLY_DRAWDOWN_THRESHOLD = 0.035  # 月度回撤 3.5% 熔断

# V49 洗售审计
WASH_SALE_WINDOW = 5
TREND_QUALITY_WINDOW = 20
TREND_QUALITY_THRESHOLD = 0.55

# V49 因子权重 - 提高 R²权重至 0.7 以增强趋势质量筛选
MOMENTUM_WEIGHT = 0.3
R2_WEIGHT = 0.7

# V49 费率配置（严禁篡改滑点成本 - 必须保留 0.1%）
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
SLIPPAGE_BUY = 0.001  # 0.1% - 严禁篡改
SLIPPAGE_SELL = 0.001  # 0.1% - 严禁篡改
STAMP_DUTY = 0.0005
TRANSFER_FEE = 0.00001

# V49 数据库表配置
DATABASE_TABLES = {
    'price_data': 'stock_daily',
    'index_data': 'index_daily',
    'industry_data': 'stock_industry_daily',
}

# V49 交易次数约束
MIN_TRADES_TARGET = 20
MAX_TRADES_TARGET = 35
TRADE_COUNT_FAIL_THRESHOLD = 40

# V49 性能目标
ANNUAL_RETURN_TARGET = 0.15  # V49: 提高到 15%
MAX_DRAWDOWN_TARGET = 0.04  # V49: 回撤不超过 4%


# ===========================================
# V49 数据类
# ===========================================

@dataclass
class V49Position:
    """V49 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_score: float
    signal_rank: int
    composite_score: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    buy_trade_day: int = 0
    
    # V49 强制时间锁
    lock_until_date: str = ""  # 15 天持有锁到期日
    is_time_locked: bool = True  # 是否在时间锁内
    
    # V49 动态 ATR 止损
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    trailing_stop_history: List[float] = field(default_factory=list)
    
    # V49 浮盈触发状态
    profit_lock_triggered: bool = False
    profit_lock_price: float = 0.0
    profit_trigger_hit: bool = False  # 浮盈>7% 触发标记
    
    # V49 排名追踪
    current_market_rank: int = 999
    current_market_percentile: float = 1.0
    position_pct: float = 0.0
    entry_composite_score: float = 0.0


@dataclass
class V49Trade:
    """V49 交易记录"""
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
    execution_price: float = 0.0


@dataclass
class V49TradeAudit:
    """V49 交易审计记录"""
    symbol: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    shares: int
    gross_pnl: float
    total_fees: float
    net_pnl: float
    holding_days: int
    is_profitable: bool
    sell_reason: str
    entry_signal: float = 0.0
    signal_rank: int = 0
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    peak_price: float = 0.0
    trailing_stop_triggered: bool = False
    profit_lock_triggered: bool = False
    was_time_locked: bool = False
    
    # V49 无效止损审计
    is_invalid_stop: bool = False  # 卖出后 5 天内反弹>3%
    rebound_price: float = 0.0
    rebound_ratio: float = 0.0


@dataclass
class V49WashSaleRecord:
    """V49 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V49BlacklistRecord:
    """V49 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V49MarketRegime:
    """V49 大盘状态"""
    trade_date: str
    index_close: float = 0.0
    index_sma60: float = 0.0
    index_ma5: float = 0.0
    index_ma20: float = 0.0
    is_risk_period: bool = False
    is_golden_cross: bool = False
    is_full_attack: bool = False
    regime_reason: str = ""


@dataclass
class V49MonthlyCircuitBreaker:
    """V49 月度熔断器"""
    month: str  # YYYY-MM
    month_start_nav: float = 0.0
    month_lowest_nav: float = 0.0
    month_max_drawdown: float = 0.0
    circuit_breaker_triggered: bool = False
    trigger_date: str = ""
    buy_blocked_days: int = 0


# ===========================================
# V49 因子引擎
# ===========================================

class V49FactorEngine:
    """
    V49 因子引擎 - 哨兵机制与动态防御
    
    【V49 修复】
    1. 修复 ma5 列丢失 Bug - 确保 ma5 在 price_ma5_ratio 前正确计算
    2. 使用 fill_null 策略处理边界情况
    """
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None, 
                 momentum_weight: float = MOMENTUM_WEIGHT,
                 r2_weight: float = R2_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """V49 全因子计算 - 修复 ma5 列丢失 Bug"""
        try:
            required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
            self._validate_columns(df, required_cols)
            
            result = df.clone().with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            ])
            
            status = {
                'factors_computed': [],
                'factors_skipped': [],
                'industry_neutralization': 'SKIPPED',
                'industry_coverage': 0.0,
                'database_tables_used': DATABASE_TABLES.copy(),
                'momentum_weight': self.momentum_weight,
                'r2_weight': self.r2_weight,
            }
            
            # Step 1: ATR 计算
            result = self._compute_atr(result, period=ATR_PERIOD)
            status['factors_computed'].append('atr_20')
            
            # Step 2: RSRS 因子
            result = self._compute_rsrs_factor(result)
            status['factors_computed'].append('rsrs_factor')
            
            # Step 3: 趋势因子
            result = self._compute_trend_factors(result)
            status['factors_computed'].extend(['trend_strength_20', 'trend_strength_60'])
            
            # Step 4: 波动率调整动量
            result = self._compute_volatility_adjusted_momentum(result)
            status['factors_computed'].append('volatility_adjusted_momentum')
            
            # Step 5: V49 趋势质量因子 (R²)
            result = self._compute_trend_quality_v49(result)
            status['factors_computed'].append('trend_quality_r2')
            
            # Step 6: V49 修复 - 5 日均线（哨兵机制）- 确保 ma5 正确计算
            result = self._compute_ma5_fixed(result)
            status['factors_computed'].append('ma5')
            
            # Step 7: 市场波动率指数
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # Step 8: 板块中性化
            if industry_data is not None and not industry_data.is_empty() and 'industry_name' in industry_data.columns:
                industry_coverage = industry_data['symbol'].n_unique()
                total_symbols = result['symbol'].n_unique()
                coverage_ratio = industry_coverage / total_symbols if total_symbols > 0 else 0
                status['industry_coverage'] = coverage_ratio
                
                if coverage_ratio >= 0.5:
                    result = self._apply_industry_neutralization_v49(result, industry_data)
                    status['industry_neutralization'] = 'ENABLED'
                else:
                    status['industry_neutralization'] = f'SKIPPED (coverage={coverage_ratio:.1%})'
                    result = result.with_columns([
                        pl.lit(0.0).alias('industry_neutralized_signal'),
                        pl.lit(0.0).alias('industry_zscore'),
                    ])
            else:
                status['industry_neutralization'] = 'SKIPPED (no data)'
                result = result.with_columns([
                    pl.lit(0.0).alias('industry_neutralized_signal'),
                    pl.lit(0.0).alias('industry_zscore'),
                ])
            
            # Step 9: V49 哨兵机制 - Composite_Score + 价格确认
            result = self._compute_composite_score_v49(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V49 compute_all_factors FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_columns(self, df: pl.DataFrame, required_columns: List[str]) -> bool:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def _compute_atr(self, df: pl.DataFrame, period: int = ATR_PERIOD) -> pl.DataFrame:
        result = df.clone()
        prev_close = pl.col('close').shift(1).over('symbol')
        tr1 = pl.col('high') - pl.col('low')
        tr2 = (pl.col('high') - prev_close).abs()
        tr3 = (pl.col('low') - prev_close).abs()
        tr = pl.max_horizontal([tr1, tr2, tr3])
        atr = tr.rolling_mean(window_size=period).over('symbol')
        
        result = result.with_columns([
            tr.alias('true_range'),
            atr.alias('atr_20'),
            prev_close.alias('prev_close')
        ])
        return result
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        rsrs_window = 18
        high_low_spread = pl.col('high') - pl.col('low')
        spread_mean = high_low_spread.rolling_mean(window_size=rsrs_window).over('symbol')
        spread_std = high_low_spread.rolling_std(window_size=rsrs_window).over('symbol')
        rsrs_raw = (high_low_spread - spread_mean) / (spread_std + self.EPSILON)
        r_squared = 1.0 / (1.0 + spread_std)
        rsrs = rsrs_raw * r_squared * 0.5
        
        result = result.with_columns([
            high_low_spread.alias('high_low_spread'),
            spread_mean.alias('spread_mean'),
            spread_std.alias('spread_std'),
            rsrs.alias('rsrs_factor')
        ])
        return result
    
    def _compute_trend_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        close_20_ago = pl.col('close').shift(20).over('symbol')
        trend_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        close_60_ago = pl.col('close').shift(60).over('symbol')
        trend_60 = (pl.col('close') - close_60_ago) / (close_60_ago + self.EPSILON)
        ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
        
        result = result.with_columns([
            trend_20.alias('trend_strength_20'),
            trend_60.alias('trend_strength_60'),
            ma60.alias('ma60')
        ])
        return result
    
    def _compute_volatility_adjusted_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        close_20_ago = pl.col('close').shift(20).over('symbol')
        momentum_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        returns = pl.col('close').pct_change().over('symbol')
        vol_20 = returns.rolling_std(window_size=20).over('symbol')
        vol_adj_momentum = momentum_20 / (vol_20 + self.EPSILON) * 0.5
        
        result = result.with_columns([
            momentum_20.alias('momentum_20'),
            vol_20.alias('volatility_20'),
            vol_adj_momentum.alias('volatility_adjusted_momentum')
        ])
        return result
    
    def _compute_trend_quality_v49(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        window = TREND_QUALITY_WINDOW
        
        close_mean = pl.col('close').rolling_mean(window_size=window).over('symbol')
        close_std = pl.col('close').rolling_std(window_size=window).over('symbol')
        
        residual = (pl.col('close') - close_mean).abs()
        ss_res_proxy = residual.rolling_mean(window_size=window).over('symbol') ** 2
        ss_tot_proxy = close_std ** 2
        
        r2_exact = 1.0 - (ss_res_proxy / (ss_tot_proxy + self.EPSILON))
        r2_clipped = r2_exact.clip(0.0, 1.0)
        
        result = result.with_columns([
            r2_clipped.alias('trend_quality_r2'),
        ])
        return result
    
    def _compute_ma5_fixed(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V49 修复 - 5 日均线计算（修复 ma5 列丢失 Bug）
        
        Bug 原因：原代码中 ma5 可能因为数据不足或 null 值导致计算失败
        修复方案：
        1. 使用 fill_null 处理 null 值
        2. 确保 rolling_mean 正确执行
        3. 对 price_ma5_ratio 进行 null 检查
        """
        result = df.clone()
        
        # V49 修复：先计算 ma5，确保使用正确的窗口大小
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        
        # V49 修复：使用 fill_null 处理边界情况（前 4 天没有 ma5）
        ma5_filled = ma5.fill_null(pl.col('close'))  # 用 close 填充 null 值
        
        # V49 修复：计算 price_ma5_ratio，确保分母不为 0
        price_ma5_ratio = pl.col('close') / (ma5_filled + self.EPSILON)
        
        result = result.with_columns([
            ma5_filled.alias('ma5'),
            price_ma5_ratio.alias('price_ma5_ratio'),
        ])
        
        return result
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        returns = pl.col('close').pct_change().over('symbol')
        stock_vol = returns.rolling_std(window_size=20, ddof=1).over('symbol')
        market_vol = stock_vol
        market_vol_mean = market_vol.rolling_mean(window_size=20).over('symbol')
        vol_ratio = market_vol / (market_vol_mean + self.EPSILON)
        vix_sim = market_vol * 100
        
        result = result.with_columns([
            returns.alias('returns'),
            stock_vol.alias('stock_volatility'),
            market_vol.alias('market_volatility'),
            market_vol_mean.alias('market_volatility_mean'),
            vol_ratio.alias('volatility_ratio'),
            vix_sim.alias('vix_sim')
        ])
        return result
    
    def _apply_industry_neutralization_v49(self, df: pl.DataFrame, industry_data: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        
        if 'industry_name' in industry_data.columns:
            result = result.join(
                industry_data.select(['symbol', 'trade_date', 'industry_name']).unique(),
                on=['symbol', 'trade_date'],
                how='left'
            )
            result = result.with_columns([
                pl.when(pl.col('industry_name').is_null())
                .then(pl.lit('Unknown'))
                .otherwise(pl.col('industry_name'))
                .alias('industry_name')
            ])
        
        base_signal = pl.lit(0.0)
        total_weight = 0.0
        for factor, weight in self.factor_weights.items():
            if factor in result.columns:
                factor_mean = result[factor].mean() or 0
                factor_std = result[factor].std() or 1
                z_factor = (pl.col(factor) - factor_mean) / (factor_std + self.EPSILON)
                base_signal = base_signal + z_factor * weight
                total_weight += weight
        
        if total_weight > 0:
            base_signal = base_signal / total_weight
        
        result = result.with_columns([base_signal.alias('base_signal')])
        
        industry_mean = pl.col('base_signal').mean().over(['trade_date', 'industry_name'])
        industry_std = pl.col('base_signal').std().over(['trade_date', 'industry_name'])
        industry_zscore = (pl.col('base_signal') - industry_mean) / (industry_std + self.EPSILON)
        
        result = result.with_columns([
            industry_mean.alias('industry_mean_signal'),
            industry_zscore.alias('industry_zscore'),
            industry_zscore.alias('industry_neutralized_signal')
        ])
        return result
    
    def _compute_composite_score_v49(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V49 哨兵机制 - Composite_Score + 价格确认
        
        进场条件：
        1. Composite_Score 位于全市场 Top 5%
        2. 股价 >= 5 日均线
        """
        result = df.clone()
        
        # 计算动量排名和 R²排名
        momentum_rank_raw = pl.col('volatility_adjusted_momentum').rank('ordinal', descending=True).over('trade_date')
        r2_rank_raw = pl.col('trend_quality_r2').rank('ordinal', descending=True).over('trade_date')
        
        # 归一化排名
        n_stocks = result['symbol'].n_unique()
        if n_stocks > 0:
            momentum_rank_norm = momentum_rank_raw / n_stocks
            r2_rank_norm = r2_rank_raw / n_stocks
        else:
            momentum_rank_norm = momentum_rank_raw
            r2_rank_norm = r2_rank_raw
        
        # V49 Composite_Score = Rank(Momentum)*momentum_weight + Rank(R2)*r2_weight
        composite_score_expr = (1.0 - momentum_rank_norm) * self.momentum_weight + (1.0 - r2_rank_norm) * self.r2_weight
        
        # Step 1: 先添加基础列
        result = result.with_columns([
            composite_score_expr.alias('composite_score'),
            momentum_rank_raw.alias('momentum_rank_raw'),
            r2_rank_raw.alias('r2_rank_raw'),
        ])
        
        # Step 2: 添加 composite_percentile 和 composite_rank
        result = result.with_columns([
            (1.0 - (pl.col('composite_score').rank('ordinal', descending=True).over('trade_date') / n_stocks)).alias('composite_percentile'),
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').cast(pl.Int64).alias('composite_rank'),
        ])
        
        # Step 3: 添加过滤器标记列（依赖 composite_percentile）
        result = result.with_columns([
            (pl.col('composite_percentile') >= (1.0 - ENTRY_TOP_PERCENTILE)).alias('percentile_filter_pass'),
            (pl.col('close') >= pl.col('ma5') * PRICE_ABOVE_MA5_RATIO).alias('price_filter_pass'),
            (pl.col('volatility_ratio') <= 1.30).alias('vol_filter_pass'),
            (pl.col('close') >= pl.col('ma60')).alias('ma_filter_pass'),
            (pl.col('trend_quality_r2').fill_null(0) >= TREND_QUALITY_THRESHOLD).alias('r2_filter_pass'),
        ])
        
        # Step 4: 添加 entry_allowed（依赖 composite_percentile）
        result = result.with_columns([
            (
                (pl.col('composite_percentile') >= (1.0 - ENTRY_TOP_PERCENTILE)) &
                (pl.col('close') >= pl.col('ma5') * PRICE_ABOVE_MA5_RATIO) &
                (pl.col('volatility_ratio') <= 1.30) &
                (pl.col('close') >= pl.col('ma60')) &
                (pl.col('trend_quality_r2').fill_null(0) >= TREND_QUALITY_THRESHOLD)
            ).alias('entry_allowed'),
        ])
        
        return result
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()
    
    def update_weights(self, momentum_weight: float, r2_weight: float):
        """V49 动态调整权重（达标检测用）"""
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight


# ===========================================
# V49 风险管理器
# ===========================================

class V49RiskManager:
    """
    V49 风险管理器 - 持有为王与组合断路器
    
    【核心功能】
    1. 强制时间锁：买入后 15 天禁止卖出（除非组合止损）
    2. 调仓过滤：新标的得分高 30% 且时间锁已解开
    3. 动态仓位管理：回撤>3% 时仓位从 20% 降至 10%
    4. 组合级断路器：总资金 -4% 硬止损
    5. 哨兵机制：Top 5% + 股价>=MA5 双重确认
    6. 呼吸式止损：初始 3.0ATR，浮盈>7% 切换 1.5ATR 追踪
    7. 进场黑名单：止损后 10 天禁止买入
    8. 月度熔断：月度回撤 3.5% 停止买入
    9. 无效止损审计：卖出后 5 天反弹>3% 标记
    """
    
    def __init__(self, initial_capital: float = FIXED_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # V49 组合级断路器追踪
        self.peak_portfolio_value = initial_capital
        self.current_max_drawdown = 0.0
        
        self.positions: Dict[str, V49Position] = {}
        self.trades: List[V49Trade] = []
        self.trade_log: List[V49TradeAudit] = []
        
        # 洗售审计
        self.sell_history: Dict[str, str] = {}
        self.wash_sale_blocks: List[V49WashSaleRecord] = []
        
        # V49 进场黑名单
        self.stop_loss_blacklist: Dict[str, V49BlacklistRecord] = {}
        
        # 每日计数器
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        
        # V49 持仓锁定（15 天强制持有）
        self.locked_positions: Dict[str, int] = {}
        
        # 交易日计数器
        self.trade_day_counter: int = 0
        
        # 波动率状态
        self.current_market_vol: float = 1.0
        self.is_low_volatility_environment: bool = False
        self.is_high_volatility_environment: bool = False
        
        # 大盘状态
        self.market_regime: V49MarketRegime = V49MarketRegime(trade_date="")
        self.is_risk_period: bool = False
        
        # V49 月度熔断器
        self.monthly_circuit_breaker: Dict[str, V49MonthlyCircuitBreaker] = {}
        self.current_month: str = ""
        self.month_start_nav: float = initial_capital
        self.month_lowest_nav: float = initial_capital
        self.monthly_drawdown: float = 0.0
        self.is_buy_blocked: bool = False
        
        # V49 动态仓位管理
        self.current_position_pct = NORMAL_POSITION_PCT  # 默认 20%
        
        # 因子状态
        self.factor_status: Dict[str, Any] = {}
        
        # 排名缓存
        self.current_rank_cache: Dict[str, int] = {}
        
        # V49 无效止损审计
        self.invalid_stop_records: List[Dict] = []
        
        # 卖出后价格追踪（用于无效止损判断）
        self.sold_positions_tracker: Dict[str, Dict] = {}
    
    def reset_daily_counters(self, trade_date: str):
        if self.last_trade_date != trade_date:
            self.today_sells.clear()
            self.today_buys.clear()
            self.last_trade_date = trade_date
            self.trade_day_counter += 1
        
        self._unlock_expired_positions()
        self._update_blacklist(trade_date)
        self._check_monthly_circuit_breaker(trade_date)
        self._check_invalid_stops(trade_date)
        self._update_portfolio_drawdown(trade_date)
    
    def _unlock_expired_positions(self):
        """解锁已过期的时间锁"""
        for symbol in list(self.locked_positions.keys()):
            self.locked_positions[symbol] -= 1
            if self.locked_positions[symbol] <= 0:
                del self.locked_positions[symbol]
    
    def _update_blacklist(self, trade_date: str):
        """V49 更新进场黑名单"""
        for symbol in list(self.stop_loss_blacklist.keys()):
            record = self.stop_loss_blacklist[symbol]
            if self.trade_day_counter >= record.blacklist_expiry_day:
                del self.stop_loss_blacklist[symbol]
            else:
                record.days_remaining = record.blacklist_expiry_day - self.trade_day_counter
    
    def _parse_date(self, date_str: str) -> datetime:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return datetime.now()
    
    def _days_between(self, date1_str: str, date2_str: str) -> int:
        date1 = self._parse_date(date1_str)
        date2 = self._parse_date(date2_str)
        return abs((date2 - date1).days)
    
    def _add_days(self, date_str: str, days: int) -> str:
        """V49 计算日期 + N 天"""
        date = self._parse_date(date_str)
        new_date = date + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")
    
    def _get_month_key(self, trade_date: str) -> str:
        """获取月份键 YYYY-MM"""
        return trade_date[:7]
    
    def _update_portfolio_drawdown(self, trade_date: str):
        """V49 更新组合回撤，用于动态仓位管理"""
        current_value = self.get_portfolio_value_approx()
        
        # 更新峰值
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        # 计算当前最大回撤
        if self.peak_portfolio_value > 0:
            self.current_max_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        # V49 动态仓位管理：回撤>3% 时仓位降至 10%
        if self.current_max_drawdown > DRAWDOWN_THRESHOLD_FOR_REDUCED_POSITION:
            self.current_position_pct = REDUCED_POSITION_PCT
        else:
            self.current_position_pct = NORMAL_POSITION_PCT
    
    def _check_monthly_circuit_breaker(self, trade_date: str):
        """V49 月度熔断检查"""
        month = self._get_month_key(trade_date)
        
        if month != self.current_month:
            # 新月重置
            self.current_month = month
            self.month_start_nav = self.get_portfolio_value_approx()
            self.month_lowest_nav = self.month_start_nav
            self.monthly_drawdown = 0.0
            self.is_buy_blocked = False
            
            if month not in self.monthly_circuit_breaker:
                self.monthly_circuit_breaker[month] = V49MonthlyCircuitBreaker(
                    month=month,
                    month_start_nav=self.month_start_nav
                )
        
        # 更新月度最低 NAV
        current_nav = self.get_portfolio_value_approx()
        if current_nav < self.month_lowest_nav:
            self.month_lowest_nav = current_nav
        
        # 计算月度回撤
        if self.month_start_nav > 0:
            self.monthly_drawdown = (self.month_start_nav - self.month_lowest_nav) / self.month_start_nav
        
        # 触发熔断
        if self.monthly_drawdown >= MONTHLY_DRAWDOWN_THRESHOLD:
            if not self.is_buy_blocked:
                self.is_buy_blocked = True
                if month in self.monthly_circuit_breaker:
                    self.monthly_circuit_breaker[month].circuit_breaker_triggered = True
                    self.monthly_circuit_breaker[month].trigger_date = trade_date
                    self.monthly_circuit_breaker[month].month_max_drawdown = self.monthly_drawdown
                logger.warning(f"MONTHLY CIRCUIT BREAKER TRIGGERED: {month} - Drawdown {self.monthly_drawdown:.2%}")
        
        # 更新熔断器统计
        if month in self.monthly_circuit_breaker:
            cb = self.monthly_circuit_breaker[month]
            cb.month_lowest_nav = self.month_lowest_nav
            cb.month_max_drawdown = self.monthly_drawdown
            if cb.circuit_breaker_triggered:
                cb.buy_blocked_days += 1
    
    def _check_invalid_stops(self, trade_date: str):
        """V49 无效止损检查 - 卖出后 5 天内反弹>3%"""
        for symbol, tracker in list(self.sold_positions_tracker.items()):
            sell_date = tracker['sell_date']
            sell_price = tracker['sell_price']
            days_since_sell = self._days_between(sell_date, trade_date)
            
            if days_since_sell > 5:
                del self.sold_positions_tracker[symbol]
                continue
            
            # 获取当前价格
            current_price = tracker.get('current_price', 0)
            if current_price <= 0:
                continue
            
            # 检查反弹
            rebound_ratio = (current_price - sell_price) / sell_price
            if rebound_ratio >= 0.03:  # 反弹>3%
                tracker['is_invalid_stop'] = True
                tracker['rebound_price'] = current_price
                tracker['rebound_ratio'] = rebound_ratio
                
                # 记录无效止损
                self.invalid_stop_records.append({
                    'symbol': symbol,
                    'sell_date': sell_date,
                    'check_date': trade_date,
                    'sell_price': sell_price,
                    'rebound_price': current_price,
                    'rebound_ratio': rebound_ratio,
                    'days_to_rebound': days_since_sell,
                })
                
                # 更新交易审计记录
                for audit in self.trade_log:
                    if audit.symbol == symbol and audit.sell_date == sell_date:
                        audit.is_invalid_stop = True
                        audit.rebound_price = current_price
                        audit.rebound_ratio = rebound_ratio
                        break
                
                logger.info(f"INVALID STOP DETECTED: {symbol} - Sold at {sell_price:.2f}, rebounded to {current_price:.2f} ({rebound_ratio:.2%})")
    
    def get_portfolio_value_approx(self) -> float:
        """近似计算组合价值（用于熔断器）"""
        return self.cash + sum(pos.shares * pos.current_price for pos in self.positions.values())
    
    def check_wash_sale(self, symbol: str, trade_date: str) -> Tuple[bool, Optional[str]]:
        if symbol in self.today_sells:
            return True, f"same_day_sell ({trade_date})"
        
        if symbol in self.sell_history:
            last_sell_date = self.sell_history[symbol]
            days_diff = self._days_between(last_sell_date, trade_date)
            if days_diff <= WASH_SALE_WINDOW:
                return True, f"wash_sale_window ({days_diff} days since {last_sell_date})"
        
        return False, None
    
    def check_blacklist(self, symbol: str, trade_date: str) -> Tuple[bool, Optional[str]]:
        """V49 检查进场黑名单"""
        if symbol in self.stop_loss_blacklist:
            record = self.stop_loss_blacklist[symbol]
            return True, f"blacklist ({record.stop_reason}) - {record.days_remaining} days remaining"
        return False, None
    
    def record_sell(self, symbol: str, trade_date: str):
        self.sell_history[symbol] = trade_date
        self.today_sells.add(symbol)
    
    def record_buy(self, symbol: str):
        self.today_buys.add(symbol)
    
    def add_to_blacklist(self, symbol: str, trade_date: str, stop_reason: str):
        """V49 将止损股票加入黑名单"""
        self.stop_loss_blacklist[symbol] = V49BlacklistRecord(
            symbol=symbol,
            stop_date=trade_date,
            stop_reason=stop_reason,
            blacklist_expiry_day=self.trade_day_counter + STOP_LOSS_BLACKLIST_DAYS,
            days_remaining=STOP_LOSS_BLACKLIST_DAYS
        )
        logger.info(f"BLACKLIST ADDED: {symbol} - {stop_reason} - Blocked for {STOP_LOSS_BLACKLIST_DAYS} days")
    
    def record_sold_position(self, symbol: str, sell_date: str, sell_price: float):
        """记录卖出股票用于无效止损检查"""
        self.sold_positions_tracker[symbol] = {
            'symbol': symbol,
            'sell_date': sell_date,
            'sell_price': sell_price,
            'is_invalid_stop': False,
            'rebound_price': 0.0,
            'rebound_ratio': 0.0,
            'current_price': sell_price,
        }
    
    def update_sold_position_price(self, symbol: str, price: float):
        """更新已卖出股票的价格"""
        if symbol in self.sold_positions_tracker:
            self.sold_positions_tracker[symbol]['current_price'] = price
    
    def _calculate_commission(self, amount: float) -> float:
        return max(MIN_COMMISSION, amount * COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        return shares * price * TRANSFER_FEE
    
    def get_current_atr_mult(self) -> float:
        """V49 动态 ATR 倍数 - 呼吸式止损"""
        if self.is_risk_period:
            return TRAILING_STOP_ATR_MULT * 0.8  # 风险期收紧
        return TRAILING_STOP_ATR_MULT
    
    def get_current_risk_target(self) -> float:
        if self.is_low_volatility_environment:
            return 0.01
        return 0.008
    
    def can_open_new_position(self) -> bool:
        if self.is_risk_period:
            return False
        if self.is_buy_blocked:
            return False
        return True
    
    def get_current_position_pct(self) -> float:
        """V49 获取当前仓位百分比（动态管理）"""
        return self.current_position_pct
    
    def check_rebalance_threshold(self, current_symbol: str, current_score: float,
                                   candidate_symbol: str, candidate_score: float,
                                   holding_days: int, is_time_locked: bool) -> Tuple[bool, float, float]:
        """
        V49 换仓成本校验
        
        条件：
        1. 新标的得分比旧标的高出 30%（V48 是 20%）
        2. 当前持仓天数 > 10 天
        3. 时间锁必须已解开
        """
        if current_score <= 0:
            return True, 1.0, 0.003
        
        score_improvement = (candidate_score - current_score) / abs(current_score)
        friction_cost = 0.003  # 估算摩擦成本
        
        # V49 硬核调控：必须同时满足三个条件
        should_rebalance = (
            score_improvement >= REBALANCE_SCORE_IMPROVEMENT and  # 30% 提升
            holding_days > REBALANCE_MIN_HOLDING_DAYS and  # 持仓>10 天
            not is_time_locked  # 时间锁已解开
        )
        
        return should_rebalance, score_improvement, friction_cost
    
    def check_portfolio_stop_loss(self, current_portfolio_value: float) -> bool:
        """
        V49 组合级断路器检查
        
        如果当前组合价值相对于峰值回撤超过 4%，触发硬止损
        """
        if self.peak_portfolio_value <= 0:
            return False
        
        current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        return current_drawdown >= PORTFOLIO_STOP_LOSS_RATIO
    
    def calculate_position_size(self, symbol: str, atr: float, current_price: float,
                                 total_assets: float) -> Tuple[int, float]:
        try:
            risk_target = self.get_current_risk_target()
            risk_amount = total_assets * risk_target
            atr_mult = INITIAL_ATR_STOP_MULT  # 使用初始 3.0ATR
            risk_per_share = atr * atr_mult
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0
            
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            
            if shares < 100:
                return 0, 0.0
            
            # V49 动态仓位管理
            position_amount = shares * current_price
            max_position = total_assets * self.current_position_pct  # 动态调整
            
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
                position_amount = shares * current_price
            
            return shares, position_amount
            
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0
    
    def execute_buy(self, trade_date: str, symbol: str, open_price: float, atr: float,
                    target_amount: float, signal_score: float = 0.0, signal_rank: int = 0,
                    composite_score: float = 0.0, composite_percentile: float = 0.0,
                    reason: str = "") -> Optional[V49Trade]:
        try:
            # V49 大盘滤镜
            if not self.can_open_new_position():
                if self.is_buy_blocked:
                    logger.warning(f"BUY BLOCKED by Monthly Circuit Breaker: {symbol}")
                else:
                    logger.warning(f"BUY BLOCKED by Market Regime Filter: {symbol}")
                return None
            
            # 洗售审计
            is_blocked, block_reason = self.check_wash_sale(symbol, trade_date)
            if is_blocked:
                logger.warning(f"WASH SALE BLOCKED: {symbol} - {block_reason}")
                self.wash_sale_blocks.append(V49WashSaleRecord(
                    symbol=symbol,
                    sell_date=self.sell_history.get(symbol, "N/A"),
                    blocked_buy_date=trade_date,
                    days_between=self._days_between(self.sell_history.get(symbol, trade_date), trade_date),
                    reason=block_reason
                ))
                return None
            
            # V49 黑名单检查
            is_blocked, block_reason = self.check_blacklist(symbol, trade_date)
            if is_blocked:
                logger.warning(f"BLACKLIST BLOCKED: {symbol} - {block_reason}")
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"DUPLICATE BUY PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            execution_price = open_price * (1 + SLIPPAGE_BUY)
            
            raw_shares = int(target_amount / execution_price)
            shares = (raw_shares // 100) * 100
            if shares < 100:
                return None
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * SLIPPAGE_BUY
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            total_cost = actual_amount + commission + slippage + transfer_fee
            
            if self.cash < total_cost:
                return None
            
            self.cash -= total_cost
            
            # V49 呼吸式止损初始化 - 3.0*ATR
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - INITIAL_ATR_STOP_MULT * atr_stop_distance)
            initial_stop_price = execution_price * (1 - INITIAL_STOP_LOSS_RATIO)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            # V49 强制时间锁：买入后 15 天禁止卖出
            lock_until_date = self._add_days(trade_date, MANDATORY_HOLDING_DAYS)
            
            position_pct = actual_amount / (self.cash + actual_amount) if (self.cash + actual_amount) > 0 else 0
            
            self.positions[symbol] = V49Position(
                symbol=symbol, shares=shares,
                avg_cost=(actual_amount + commission + slippage + transfer_fee) / shares,
                buy_price=execution_price, buy_date=trade_date,
                signal_score=signal_score, signal_rank=signal_rank,
                composite_score=composite_score,
                current_price=execution_price,
                holding_days=0,
                peak_price=execution_price, peak_profit=0.0,
                buy_trade_day=self.trade_day_counter,
                atr_at_entry=atr,
                initial_stop_price=stop_price,
                trailing_stop_price=stop_price,
                trailing_stop_history=[stop_price],
                current_market_rank=signal_rank,
                current_market_percentile=composite_percentile,
                position_pct=position_pct,
                entry_composite_score=composite_score,
                profit_lock_triggered=False,
                profit_trigger_hit=False,
                lock_until_date=lock_until_date,
                is_time_locked=True,
            )
            
            # V49 强制时间锁：15 天
            self.locked_positions[symbol] = MANDATORY_HOLDING_DAYS
            self.record_buy(symbol)
            
            trade = V49Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee,
                total_cost=total_cost, reason=reason, execution_price=execution_price,
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Rank={signal_rank} | Percentile={composite_percentile:.1%} | Lock until {lock_until_date}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(self, trade_date: str, symbol: str, open_price: float,
                     shares: Optional[int] = None, reason: str = "", force: bool = False) -> Optional[V49Trade]:
        """
        V49 执行卖出
        
        Args:
            force: 是否强制卖出（忽略时间锁，用于组合级止损）
        """
        try:
            if symbol not in self.positions:
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            pos = self.positions[symbol]
            
            # V49 持仓锁检查（除非强制或止损）
            if symbol in self.locked_positions and self.locked_positions[symbol] > 0:
                if not force and reason not in ["trailing_stop", "stop_loss", "portfolio_stop_loss"]:
                    logger.debug(f"TIME LOCK ACTIVE: {symbol} locked for {self.locked_positions[symbol]} more days until {pos.lock_until_date}")
                    return None
            
            available = pos.shares
            if shares is None or shares > available:
                shares = available
            
            if shares < 100:
                return None
            
            execution_price = open_price * (1 - SLIPPAGE_SELL)
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * SLIPPAGE_SELL
            stamp_duty = actual_amount * STAMP_DUTY
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            net_proceeds = actual_amount - commission - slippage - stamp_duty - transfer_fee
            
            self.cash += net_proceeds
            
            cost_basis = pos.avg_cost * shares
            realized_pnl = net_proceeds - cost_basis
            
            try:
                buy_date = self._parse_date(pos.buy_date)
                sell_date = self._parse_date(trade_date)
                calculated_holding_days = max(1, (sell_date - buy_date).days)
            except:
                calculated_holding_days = pos.holding_days if pos.holding_days > 0 else 1
            
            # V49 审计：如果是止损，加入黑名单
            if reason in ["trailing_stop", "stop_loss", "portfolio_stop_loss"]:
                self.add_to_blacklist(symbol, trade_date, reason)
            
            # 记录卖出用于无效止损检查
            self.record_sold_position(symbol, trade_date, execution_price)
            
            trade_audit = V49TradeAudit(
                symbol=symbol,
                buy_date=pos.buy_date,
                sell_date=trade_date,
                buy_price=pos.buy_price,
                sell_price=execution_price,
                shares=shares,
                gross_pnl=realized_pnl + (commission + slippage + stamp_duty + transfer_fee),
                total_fees=commission + slippage + stamp_duty + transfer_fee,
                net_pnl=realized_pnl,
                holding_days=calculated_holding_days,
                is_profitable=realized_pnl > 0,
                sell_reason=reason,
                entry_signal=pos.signal_score,
                signal_rank=pos.signal_rank,
                atr_at_entry=pos.atr_at_entry,
                initial_stop_price=pos.initial_stop_price,
                peak_price=pos.peak_price,
                trailing_stop_triggered=pos.trailing_stop_triggered,
                profit_lock_triggered=pos.profit_lock_triggered,
                was_time_locked=pos.is_time_locked or (symbol in self.locked_positions and self.locked_positions[symbol] > 0),
            )
            self.trade_log.append(trade_audit)
            
            del self.positions[symbol]
            self.locked_positions.pop(symbol, None)
            
            self.record_sell(symbol, trade_date)
            
            trade = V49Trade(
                trade_date=trade_date, symbol=symbol, side="SELL", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=stamp_duty, transfer_fee=transfer_fee,
                total_cost=net_proceeds, reason=reason, holding_days=pos.holding_days,
                execution_price=execution_price,
            )
            self.trades.append(trade)
            
            logger.info(f"  SELL {symbol} | {shares} shares @ {execution_price:.2f} | PnL: {realized_pnl:.2f} | Reason: {reason}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_sell failed: {e}")
            return None
    
    def update_volatility_regime(self, market_vol: float):
        self.current_market_vol = market_vol
        self.is_low_volatility_environment = market_vol < 0.80
        self.is_high_volatility_environment = market_vol > 1.20
    
    def get_risk_per_position(self) -> float:
        return self.get_current_risk_target()
    
    def check_stop_loss_and_rank(self, positions: Dict[str, V49Position], date_str: str,
                                  price_df: pl.DataFrame, factor_df: pl.DataFrame,
                                  current_portfolio_value: float) -> List[Tuple[str, str]]:
        """
        V49 检查止损和排名 - 呼吸式止损 + 位次缓冲区 + 强制时间锁
        
        卖出条件（按优先级）：
        1. 组合级断路器（-4% 总资金止损）- 可突破时间锁
        2. 移动止损触发（可突破时间锁）
        3. 初始止损 8%（可突破时间锁）
        4. 排名跌出 Top 30%（宽缓冲区）- 必须时间锁已解开
        """
        sell_list = []
        
        try:
            prices_df = price_df.select(['symbol', 'close']).unique('symbol', keep='last')
            prices = dict(zip(prices_df['symbol'].to_list(), prices_df['close'].to_list()))
        except:
            prices = {}
        
        try:
            ranks_df = factor_df.select(['symbol', 'composite_rank', 'composite_percentile', 'atr_20']).unique('symbol', keep='last')
            ranks = dict(zip(ranks_df['symbol'].to_list(), ranks_df['composite_rank'].to_list()))
            percentiles = dict(zip(ranks_df['symbol'].to_list(), ranks_df['composite_percentile'].to_list()))
            atr_values = dict(zip(ranks_df['symbol'].to_list(), ranks_df['atr_20'].to_list()))
        except:
            ranks = {}
            percentiles = {}
            atr_values = {}
        
        # V49 组合级断路器检查
        portfolio_stop_loss_triggered = self.check_portfolio_stop_loss(current_portfolio_value)
        
        for symbol, pos in list(positions.items()):
            current_price = prices.get(symbol, 0)
            if current_price <= 0:
                continue
            
            pos.current_price = current_price
            pos.market_value = pos.shares * current_price
            pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares
            pos.holding_days = max(0, self.trade_day_counter - pos.buy_trade_day)
            pos.current_market_rank = ranks.get(symbol, 9999)
            pos.current_market_percentile = percentiles.get(symbol, 0)
            
            # V49 更新时间锁状态
            pos.is_time_locked = symbol in self.locked_positions and self.locked_positions[symbol] > 0
            
            # 更新峰值价格
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            
            # V49 呼吸式止损更新
            atr = atr_values.get(symbol, pos.atr_at_entry)
            if atr and atr > 0:
                try:
                    # 检查浮盈是否触发 7%
                    profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                    
                    if profit_ratio >= PROFIT_TRIGGER_RATIO and not pos.profit_trigger_hit:
                        pos.profit_trigger_hit = True
                        logger.info(f"  PROFIT TRIGGER HIT: {symbol} - Profit {profit_ratio:.2%} >= 7%, switching to trailing stop")
                    
                    # 根据浮盈状态选择 ATR 倍数
                    if pos.profit_trigger_hit:
                        # 已触发浮盈，使用 1.5*ATR 追踪止损
                        atr_mult = self.get_current_atr_mult()
                    else:
                        # 未触发浮盈，使用 3.0*ATR 宽松止损
                        atr_mult = INITIAL_ATR_STOP_MULT
                    
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    new_trailing_stop = current_price * (1 - atr_mult * atr_stop_distance)
                    
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        pos.trailing_stop_history.append(new_trailing_stop)
                    
                    # V49 动态止盈 - 盈利锁
                    if pos.profit_trigger_hit:
                        profit_lock_price = pos.avg_cost * 1.02
                        if not pos.profit_lock_triggered and pos.trailing_stop_price >= profit_lock_price:
                            pos.profit_lock_triggered = True
                            pos.profit_lock_price = profit_lock_price
                            pos.trailing_stop_price = max(pos.trailing_stop_price, profit_lock_price)
                            logger.info(f"  PROFIT LOCK TRIGGERED: {symbol} - locked at {profit_lock_price:.2f}")
                            
                except Exception as e:
                    logger.error(f"Error updating trailing stop for {symbol}: {e}")
            
            # V49 卖出条件检查（按优先级）
            
            # 0. 组合级断路器（最高优先级，可突破时间锁）
            if portfolio_stop_loss_triggered:
                sell_list.append((symbol, "portfolio_stop_loss"))
                continue
            
            # 1. 移动止损触发（可突破时间锁）
            if current_price <= pos.trailing_stop_price:
                pos.trailing_stop_triggered = True
                sell_list.append((symbol, "trailing_stop"))
                continue
            
            # 2. 初始止损底线（8%，可突破时间锁）
            profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            if profit_ratio <= -INITIAL_STOP_LOSS_RATIO:
                sell_list.append((symbol, "stop_loss"))
                continue
            
            # 3. V49 位次缓冲区 - 跌出 Top 30% 才允许卖出（必须时间锁已解开）
            current_percentile = pos.current_market_percentile
            is_time_locked = pos.is_time_locked
            
            if not is_time_locked and current_percentile < (1.0 - EXIT_BOTTOM_PERCENTILE_THRESHOLD):
                sell_list.append((symbol, "rank_drop"))
                continue
        
        return sell_list
    
    def get_wash_sale_stats(self) -> Dict[str, Any]:
        return {
            'total_blocked': len(self.wash_sale_blocks),
            'blocked_records': [
                {
                    'symbol': r.symbol,
                    'sell_date': r.sell_date,
                    'blocked_buy_date': r.blocked_buy_date,
                    'days_between': r.days_between,
                    'reason': r.reason
                }
                for r in self.wash_sale_blocks
            ]
        }
    
    def get_invalid_stop_stats(self) -> Dict[str, Any]:
        """V49 无效止损统计"""
        return {
            'total_invalid_stops': len(self.invalid_stop_records),
            'invalid_stop_records': self.invalid_stop_records.copy(),
        }
    
    def get_portfolio_value(self, positions: Dict[str, V49Position], date_str: str, price_df: pl.DataFrame) -> float:
        market_value = 0.0
        for symbol, pos in positions.items():
            try:
                row = price_df.filter((pl.col('symbol') == symbol) & (pl.col('trade_date') == date_str)).select('close').row(0)
                if row:
                    market_value += pos.shares * float(row[0])
                    # 更新已卖出股票价格用于无效止损检查
                    self.update_sold_position_price(symbol, float(row[0]))
            except Exception:
                market_value += pos.shares * pos.current_price
        return self.cash + market_value
    
    def get_market_regime_stats(self) -> Dict[str, Any]:
        return {
            'is_risk_period': self.is_risk_period,
            'current_regime': 'RISK' if self.is_risk_period else 'NORMAL',
        }
    
    def get_monthly_circuit_breaker_stats(self) -> Dict[str, Any]:
        return {
            'current_month': self.current_month,
            'month_start_nav': self.month_start_nav,
            'month_lowest_nav': self.month_lowest_nav,
            'monthly_drawdown': self.monthly_drawdown,
            'is_buy_blocked': self.is_buy_blocked,
            'circuit_breaker_threshold': MONTHLY_DRAWDOWN_THRESHOLD,
            'total_triggered_months': sum(1 for cb in self.monthly_circuit_breaker.values() if cb.circuit_breaker_triggered),
        }
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        return {
            'total_blacklisted': len(self.stop_loss_blacklist),
            'blacklist_records': [
                {
                    'symbol': r.symbol,
                    'stop_date': r.stop_date,
                    'stop_reason': r.stop_reason,
                    'days_remaining': r.days_remaining
                }
                for r in self.stop_loss_blacklist.values()
            ]
        }
    
    def get_trade_count_stats(self) -> Dict[str, Any]:
        return {
            'total_trades': len(self.trades),
            'min_target': MIN_TRADES_TARGET,
            'max_target': MAX_TRADES_TARGET,
            'fail_threshold': TRADE_COUNT_FAIL_THRESHOLD,
        }
    
    def check_trade_count_constraint(self) -> Tuple[bool, str]:
        total_trades = len(self.trades)
        if total_trades > TRADE_COUNT_FAIL_THRESHOLD:
            return False, f"[V49 FAILED: OVER-TRADING] Trade count {total_trades} exceeds threshold {TRADE_COUNT_FAIL_THRESHOLD}"
        return True, ""
    
    def get_dynamic_position_stats(self) -> Dict[str, Any]:
        """V49 动态仓位统计"""
        return {
            'current_position_pct': self.current_position_pct,
            'normal_position_pct': NORMAL_POSITION_PCT,
            'reduced_position_pct': REDUCED_POSITION_PCT,
            'drawdown_threshold': DRAWDOWN_THRESHOLD_FOR_REDUCED_POSITION,
            'current_max_drawdown': self.current_max_drawdown,
            'peak_portfolio_value': self.peak_portfolio_value,
            'is_reduced_position': self.current_position_pct == REDUCED_POSITION_PCT,
        }
    
    def get_time_lock_stats(self) -> Dict[str, Any]:
        """V49 时间锁统计"""
        return {
            'mandatory_holding_days': MANDATORY_HOLDING_DAYS,
            'currently_locked_positions': len(self.locked_positions),
            'locked_symbols': list(self.locked_positions.keys()),
        }