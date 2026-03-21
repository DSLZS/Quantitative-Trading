"""
V50 Core Module - 自适应动量与信号纯化

【V50 核心改进 - 废除 V49 枷锁，进化双重动量确认】

1. 逻辑大拆除（废除 V49 僵化枷锁）:
   - ❌ 废除"15 天时间锁" - 改为灵活持有期
   - ❌ 废除"30% 换仓阈值" - 改为位次缓冲带
   - ❌ 废除"回撤 3% 强制减仓" - 改为个股波动率头寸管理

2. 核心算法进化 - 双重动量确认 (Dual-Momentum Confirm):
   - 进场哨兵 2.0: Score = Rank(Momentum)*0.4 + Rank(R²)*0.6
   - 趋势确认：股价 > MA20 且 MA5 > MA20（趋势完全成型才入场）
   - 选股范围：全市场 Top 10

3. 分档动态止损 (Tiered-Exit):
   - 初段保护（浮盈 < 5%）: 2.5 * ATR 止损
   - 中段护航（浮盈 5%~15%）: 1.5 * ATR 追踪止盈
   - 高段奔跑（浮盈 > 15%）: 仅跌破 MA20 时清仓

4. 换仓逻辑 - 位次缓冲带 (Rank Buffer):
   - 入场：全市场 Top 10
   - 维持：只要位次在 Top 30 且 趋势未破 (MA20)，严禁调仓
   - 目标：自然将交易次数控制在 30 次左右

5. 个股波动率头寸管理:
   - 根据个股 ATR 动态计算头寸
   - 风险暴露 = 总资金 * 0.8% / (ATR * ATR_Mult)

6. 强制质量对赌（AI 执行约束）:
   - 对赌协议：年化收益 < 15% 或 MDD > 4% 视为失败
   - 透明审计：列出"单笔交易最大盈利"和"单笔交易最大亏损"
   - 盈亏比检查：必须 > 3:1

作者：量化系统
版本：V50.0
日期：2026-03-20
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V50 配置常量
# ===========================================

# 基础配置
FIXED_INITIAL_CAPITAL = 100000.00
MAX_POSITIONS = 10  # V50: Top 10 选股

# V50 双重动量权重 - 提升 R²权重
MOMENTUM_WEIGHT = 0.4
R2_WEIGHT = 0.6

# V50 进场确认 - 哨兵 2.0
ENTRY_TOP_N = 10  # Top 10 选股
ENTRY_MA5_ABOVE_MA20 = True  # MA5 > MA20 趋势确认
ENTRY_PRICE_ABOVE_MA20 = True  # 股价 > MA20

# V50 位次缓冲带
MAINTAIN_TOP_N = 30  # 只要位次在 Top 30 且趋势未破，严禁调仓

# V50 分档动态止损配置
TIERED_EXIT_CONFIG = {
    'initial_atr_mult': 2.5,  # 初段：2.5*ATR
    'profit_tier1_threshold': 0.05,  # 5% 浮盈阈值
    'profit_tier1_atr_mult': 1.5,  # 中段：1.5*ATR 追踪
    'profit_tier2_threshold': 0.15,  # 15% 浮盈阈值
    'profit_tier2_use_ma20': True,  # 高段：仅 MA20 清仓
}

# V50 初始止损
INITIAL_STOP_LOSS_RATIO = 0.08  # 8% 初始止损底线

# V50 个股波动率头寸管理
RISK_TARGET_PER_POSITION = 0.008  # 单只股票风险暴露 0.8%
MAX_SINGLE_POSITION_PCT = 0.15  # 单只股票最大 15%

# V50 洗售审计
WASH_SALE_WINDOW = 5  # 5 天洗售窗口

# V50 趋势质量
TREND_QUALITY_WINDOW = 20
TREND_QUALITY_THRESHOLD = 0.55

# V50 波动率过滤
VOLATILITY_FILTER_THRESHOLD = 1.30

# V50 费率配置（严禁篡改滑点成本 - 必须保留 0.1%）
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
SLIPPAGE_BUY = 0.001  # 0.1% - 严禁篡改
SLIPPAGE_SELL = 0.001  # 0.1% - 严禁篡改
STAMP_DUTY = 0.0005
TRANSFER_FEE = 0.00001

# V50 数据库表配置
DATABASE_TABLES = {
    'price_data': 'stock_daily',
    'index_data': 'index_daily',
    'industry_data': 'stock_industry_daily',
}

# V50 交易次数约束
MIN_TRADES_TARGET = 20
MAX_TRADES_TARGET = 35
TRADE_COUNT_FAIL_THRESHOLD = 40

# V50 性能目标（对赌协议）
ANNUAL_RETURN_TARGET = 0.15  # 年化收益 > 15%
MAX_DRAWDOWN_TARGET = 0.04  # MDD < 4%
PROFIT_LOSS_RATIO_TARGET = 3.0  # 盈亏比 > 3:1


# ===========================================
# V50 数据类
# ===========================================

@dataclass
class V50Position:
    """V50 持仓记录"""
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
    
    # V50 分档动态止损
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    trailing_stop_history: List[float] = field(default_factory=list)
    
    # V50 浮盈状态追踪
    profit_tier1_triggered: bool = False  # 5% 浮盈触发
    profit_tier2_triggered: bool = False  # 15% 浮盈触发
    current_profit_ratio: float = 0.0
    
    # V50 排名追踪
    current_market_rank: int = 999
    current_market_percentile: float = 1.0
    position_pct: float = 0.0
    entry_composite_score: float = 0.0
    
    # V50 趋势状态
    ma5_at_entry: float = 0.0
    ma20_at_entry: float = 0.0


@dataclass
class V50Trade:
    """V50 交易记录"""
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
class V50TradeAudit:
    """V50 交易审计记录"""
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
    profit_tier1_triggered: bool = False
    profit_tier2_triggered: bool = False
    exit_profit_ratio: float = 0.0


@dataclass
class V50WashSaleRecord:
    """V50 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V50BlacklistRecord:
    """V50 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V50MarketRegime:
    """V50 大盘状态"""
    trade_date: str
    index_close: float = 0.0
    index_sma60: float = 0.0
    index_ma5: float = 0.0
    index_ma20: float = 0.0
    is_risk_period: bool = False
    is_golden_cross: bool = False
    is_full_attack: bool = False
    regime_reason: str = ""


# ===========================================
# V50 因子引擎
# ===========================================

class V50FactorEngine:
    """
    V50 因子引擎 - 双重动量确认与信号纯化
    
    【V50 核心改进】
    1. Score = Rank(Momentum)*0.4 + Rank(R²)*0.6（提升 R²权重）
    2. 趋势确认：MA5 > MA20 且 股价 > MA20
    3. Top 10 选股 + Top 30 维持缓冲带
    """
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None,
                 momentum_weight: float = MOMENTUM_WEIGHT,
                 r2_weight: float = R2_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """V50 全因子计算"""
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
            result = self._compute_atr(result, period=20)
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
            
            # Step 5: V50 趋势质量因子 (R²)
            result = self._compute_trend_quality_v50(result)
            status['factors_computed'].append('trend_quality_r2')
            
            # Step 6: V50 均线系统（MA5, MA20）
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20'])
            
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
                    result = self._apply_industry_neutralization_v50(result, industry_data)
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
            
            # Step 9: V50 双重动量确认 - Composite_Score + 趋势确认
            result = self._compute_composite_score_v50(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V50 compute_all_factors FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_columns(self, df: pl.DataFrame, required_columns: List[str]) -> bool:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def _compute_atr(self, df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
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
    
    def _compute_trend_quality_v50(self, df: pl.DataFrame) -> pl.DataFrame:
        """V50 趋势质量因子（R²计算）"""
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
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        """V50 均线系统 - MA5, MA20"""
        result = df.clone()
        
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        
        # V50 趋势确认：MA5 > MA20
        ma_trend_bullish = ma5 > ma20
        
        # V50 价格确认：股价 > MA20
        price_above_ma20 = pl.col('close') > ma20
        
        result = result.with_columns([
            ma5.alias('ma5'),
            ma20.alias('ma20'),
            ma_trend_bullish.alias('ma_trend_bullish'),
            price_above_ma20.alias('price_above_ma20'),
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
    
    def _apply_industry_neutralization_v50(self, df: pl.DataFrame, industry_data: pl.DataFrame) -> pl.DataFrame:
        """V50 板块中性化"""
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
    
    def _compute_composite_score_v50(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V50 双重动量确认 - Composite_Score + 趋势确认
        
        进场条件：
        1. Composite_Score = Rank(Momentum)*0.4 + Rank(R²)*0.6
        2. Top 10 选股
        3. MA5 > MA20（趋势成型）
        4. 股价 > MA20
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
        
        # V50 核心：Composite_Score = Rank(Momentum)*0.4 + Rank(R²)*0.6
        # 先定义 composite_score 表达式
        composite_score_expr = (1.0 - momentum_rank_norm) * self.momentum_weight + (1.0 - r2_rank_norm) * self.r2_weight
        
        # 先计算 composite_score，然后再计算排名
        result = result.with_columns([
            composite_score_expr.alias('composite_score'),
        ])
        
        # 计算排名和百分位（基于已计算的 composite_score）
        composite_rank = pl.col('composite_score').rank('ordinal', descending=True).over('trade_date')
        composite_percentile = 1.0 - (composite_rank / n_stocks) if n_stocks > 0 else composite_rank
        
        # V50 趋势确认条件
        ma_trend_filter = pl.col('ma_trend_bullish') == True  # MA5 > MA20
        price_filter = pl.col('price_above_ma20') == True  # 股价 > MA20
        
        # V50 Top 10 选股（使用新计算的排名）
        top_n_filter = composite_rank <= ENTRY_TOP_N
        
        # 波动率过滤
        vol_filter = pl.col('volatility_ratio') <= VOLATILITY_FILTER_THRESHOLD
        
        # R²质量过滤
        r2_filter = pl.col('trend_quality_r2').fill_null(0) >= TREND_QUALITY_THRESHOLD
        
        # V50 综合入场条件
        entry_allowed = top_n_filter & ma_trend_filter & price_filter & vol_filter & r2_filter
        
        result = result.with_columns([
            momentum_rank_raw.alias('momentum_rank_raw'),
            r2_rank_raw.alias('r2_rank_raw'),
            composite_rank.cast(pl.Int64).alias('composite_rank'),
            composite_percentile.alias('composite_percentile'),
            ma_trend_filter.alias('ma_trend_filter_pass'),
            price_filter.alias('price_filter_pass'),
            top_n_filter.alias('top_n_filter_pass'),
            vol_filter.alias('vol_filter_pass'),
            r2_filter.alias('r2_filter_pass'),
            entry_allowed.alias('entry_allowed'),
        ])
        
        return result
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()
    
    def update_weights(self, momentum_weight: float, r2_weight: float):
        """V50 动态调整权重"""
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight


# ===========================================
# V50 风险管理器
# ===========================================

class V50RiskManager:
    """
    V50 风险管理器 - 分档动态止损与个股波动率头寸管理
    
    【核心功能】
    1. 废除 V49 时间锁 - 改为灵活持有期
    2. 废除 V49 30% 换仓阈值 - 改为位次缓冲带 (Top 30)
    3. 废除 V49 回撤 3% 强制减仓 - 改为个股波动率头寸管理
    4. 分档动态止损 (Tiered-Exit):
       - 初段（浮盈 < 5%）: 2.5*ATR 止损
       - 中段（浮盈 5%~15%）: 1.5*ATR 追踪
       - 高段（浮盈 > 15%）: 仅 MA20 清仓
    5. 洗售审计：5 天窗口
    6. 进场黑名单：止损后 5 天禁止买入
    """
    
    def __init__(self, initial_capital: float = FIXED_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, V50Position] = {}
        self.trades: List[V50Trade] = []
        self.trade_log: List[V50TradeAudit] = []
        
        # 洗售审计
        self.sell_history: Dict[str, str] = {}
        self.wash_sale_blocks: List[V50WashSaleRecord] = []
        
        # V50 进场黑名单
        self.stop_loss_blacklist: Dict[str, V50BlacklistRecord] = {}
        
        # 每日计数器
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        
        # 交易日计数器
        self.trade_day_counter: int = 0
        
        # 波动率状态
        self.current_market_vol: float = 1.0
        self.is_low_volatility_environment: bool = False
        self.is_high_volatility_environment: bool = False
        
        # 大盘状态
        self.market_regime: V50MarketRegime = V50MarketRegime(trade_date="")
        self.is_risk_period: bool = False
        
        # 因子状态
        self.factor_status: Dict[str, Any] = {}
        
        # 排名缓存
        self.current_rank_cache: Dict[str, int] = {}
        
        # V50 个股波动率头寸管理 - 无全局降仓
        self.current_position_pct = MAX_SINGLE_POSITION_PCT
    
    def reset_daily_counters(self, trade_date: str):
        if self.last_trade_date != trade_date:
            self.today_sells.clear()
            self.today_buys.clear()
            self.last_trade_date = trade_date
            self.trade_day_counter += 1
        
        self._update_blacklist(trade_date)
    
    def _update_blacklist(self, trade_date: str):
        """V50 更新进场黑名单"""
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
        date = self._parse_date(date_str)
        new_date = date + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")
    
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
        """V50 检查进场黑名单"""
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
        """V50 将止损股票加入黑名单"""
        self.stop_loss_blacklist[symbol] = V50BlacklistRecord(
            symbol=symbol,
            stop_date=trade_date,
            stop_reason=stop_reason,
            blacklist_expiry_day=self.trade_day_counter + 5,  # 5 天黑名单
            days_remaining=5
        )
        logger.info(f"BLACKLIST ADDED: {symbol} - {stop_reason} - Blocked for 5 days")
    
    def _calculate_commission(self, amount: float) -> float:
        return max(MIN_COMMISSION, amount * COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        return shares * price * TRANSFER_FEE
    
    def get_current_risk_target(self) -> float:
        return RISK_TARGET_PER_POSITION
    
    def can_open_new_position(self) -> bool:
        if self.is_risk_period:
            return False
        return True
    
    def calculate_position_size(self, symbol: str, atr: float, current_price: float,
                                 total_assets: float) -> Tuple[int, float]:
        """V50 个股波动率头寸管理"""
        try:
            risk_target = self.get_current_risk_target()
            risk_amount = total_assets * risk_target
            
            # V50 使用 2.5*ATR 初始止损
            atr_mult = TIERED_EXIT_CONFIG['initial_atr_mult']
            risk_per_share = atr * atr_mult
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0
            
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            
            if shares < 100:
                return 0, 0.0
            
            # V50 单只股票最大仓位
            position_amount = shares * current_price
            max_position = total_assets * MAX_SINGLE_POSITION_PCT
            
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
            
            return shares, shares * current_price
            
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0
    
    def execute_buy(self, trade_date: str, symbol: str, open_price: float, atr: float,
                    target_amount: float, signal_score: float = 0.0, signal_rank: int = 0,
                    composite_score: float = 0.0, composite_percentile: float = 0.0,
                    ma5: float = 0.0, ma20: float = 0.0,
                    reason: str = "") -> Optional[V50Trade]:
        try:
            # 大盘滤镜
            if not self.can_open_new_position():
                logger.warning(f"BUY BLOCKED by Market Regime Filter: {symbol}")
                return None
            
            # 洗售审计
            is_blocked, block_reason = self.check_wash_sale(symbol, trade_date)
            if is_blocked:
                logger.warning(f"WASH SALE BLOCKED: {symbol} - {block_reason}")
                self.wash_sale_blocks.append(V50WashSaleRecord(
                    symbol=symbol,
                    sell_date=self.sell_history.get(symbol, "N/A"),
                    blocked_buy_date=trade_date,
                    days_between=self._days_between(self.sell_history.get(symbol, trade_date), trade_date),
                    reason=block_reason
                ))
                return None
            
            # 黑名单检查
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
            
            # V50 初始止损 - 2.5*ATR
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - TIERED_EXIT_CONFIG['initial_atr_mult'] * atr_stop_distance)
            initial_stop_price = execution_price * (1 - INITIAL_STOP_LOSS_RATIO)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            position_pct = actual_amount / (self.cash + actual_amount) if (self.cash + actual_amount) > 0 else 0
            
            self.positions[symbol] = V50Position(
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
                profit_tier1_triggered=False,
                profit_tier2_triggered=False,
                current_profit_ratio=0.0,
                ma5_at_entry=ma5,
                ma20_at_entry=ma20,
            )
            
            self.record_buy(symbol)
            
            trade = V50Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee,
                total_cost=total_cost, reason=reason, execution_price=execution_price,
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Rank={signal_rank} | Tiered-Exit Enabled")
            return trade
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(self, trade_date: str, symbol: str, open_price: float,
                     shares: Optional[int] = None, reason: str = "", force: bool = False) -> Optional[V50Trade]:
        """V50 执行卖出"""
        try:
            if symbol not in self.positions:
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            pos = self.positions[symbol]
            
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
            
            # V50 审计：如果是止损，加入黑名单
            if reason in ["trailing_stop", "stop_loss", "ma20_break"]:
                self.add_to_blacklist(symbol, trade_date, reason)
            
            trade_audit = V50TradeAudit(
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
                profit_tier1_triggered=pos.profit_tier1_triggered,
                profit_tier2_triggered=pos.profit_tier2_triggered,
                exit_profit_ratio=pos.current_profit_ratio,
            )
            self.trade_log.append(trade_audit)
            
            del self.positions[symbol]
            
            self.record_sell(symbol, trade_date)
            
            trade = V50Trade(
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
    
    def check_tiered_exit(self, positions: Dict[str, V50Position], date_str: str,
                          price_df: pl.DataFrame, factor_df: pl.DataFrame) -> List[Tuple[str, str]]:
        """
        V50 分档动态止损 (Tiered-Exit)
        
        卖出条件（按优先级）：
        1. 移动止损触发（根据浮盈状态选择 ATR 倍数）
        2. 初始止损 8%
        3. 高段奔跑：浮盈 > 15% 时，跌破 MA20 清仓
        4. 位次缓冲带：跌出 Top 30 才允许卖出
        """
        sell_list = []
        
        try:
            prices_df = price_df.select(['symbol', 'close']).unique('symbol', keep='last')
            prices = dict(zip(prices_df['symbol'].to_list(), prices_df['close'].to_list()))
        except:
            prices = {}
        
        try:
            ranks_df = factor_df.select(['symbol', 'composite_rank', 'atr_20', 'ma20']).unique('symbol', keep='last')
            ranks = dict(zip(ranks_df['symbol'].to_list(), ranks_df['composite_rank'].to_list()))
            atr_values = dict(zip(ranks_df['symbol'].to_list(), ranks_df['atr_20'].to_list()))
            ma20_values = dict(zip(ranks_df['symbol'].to_list(), ranks_df['ma20'].to_list()))
        except:
            ranks = {}
            atr_values = {}
            ma20_values = {}
        
        for symbol, pos in list(positions.items()):
            current_price = prices.get(symbol, 0)
            if current_price <= 0:
                continue
            
            pos.current_price = current_price
            pos.market_value = pos.shares * current_price
            pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares
            pos.holding_days = max(0, self.trade_day_counter - pos.buy_trade_day)
            pos.current_market_rank = ranks.get(symbol, 9999)
            
            # 更新峰值价格
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            
            # 计算当前浮盈比例
            pos.current_profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            
            # V50 浮盈状态追踪
            if pos.current_profit_ratio >= TIERED_EXIT_CONFIG['profit_tier1_threshold'] and not pos.profit_tier1_triggered:
                pos.profit_tier1_triggered = True
                logger.info(f"  PROFIT TIER 1 HIT: {symbol} - Profit {pos.current_profit_ratio:.2%} >= 5%")
            
            if pos.current_profit_ratio >= TIERED_EXIT_CONFIG['profit_tier2_threshold'] and not pos.profit_tier2_triggered:
                pos.profit_tier2_triggered = True
                logger.info(f"  PROFIT TIER 2 HIT: {symbol} - Profit {pos.current_profit_ratio:.2%} >= 15% - Let it run!")
            
            # V50 分档动态止损逻辑
            atr = atr_values.get(symbol, pos.atr_at_entry)
            if atr and atr > 0:
                try:
                    # 根据浮盈状态选择 ATR 倍数
                    if pos.profit_tier2_triggered:
                        # 高段奔跑：浮盈 > 15%，仅 MA20 清仓
                        ma20 = ma20_values.get(symbol, 0)
                        if ma20 > 0 and current_price < ma20:
                            sell_list.append((symbol, "ma20_break"))
                            continue
                    
                    elif pos.profit_tier1_triggered:
                        # 中段护航：浮盈 5%~15%，1.5*ATR 追踪止盈
                        atr_mult = TIERED_EXIT_CONFIG['profit_tier1_atr_mult']
                        atr_stop_distance = atr / current_price if current_price > 0 else 0
                        new_trailing_stop = current_price * (1 - atr_mult * atr_stop_distance)
                        
                        if new_trailing_stop > pos.trailing_stop_price:
                            pos.trailing_stop_price = new_trailing_stop
                            pos.trailing_stop_history.append(new_trailing_stop)
                        
                        if current_price <= pos.trailing_stop_price:
                            pos.trailing_stop_triggered = True
                            sell_list.append((symbol, "trailing_stop"))
                            continue
                    
                    else:
                        # 初段保护：浮盈 < 5%，2.5*ATR 止损
                        atr_mult = TIERED_EXIT_CONFIG['initial_atr_mult']
                        atr_stop_distance = atr / current_price if current_price > 0 else 0
                        new_trailing_stop = current_price * (1 - atr_mult * atr_stop_distance)
                        
                        if new_trailing_stop > pos.trailing_stop_price:
                            pos.trailing_stop_price = new_trailing_stop
                            pos.trailing_stop_history.append(new_trailing_stop)
                        
                        if current_price <= pos.trailing_stop_price:
                            pos.trailing_stop_triggered = True
                            sell_list.append((symbol, "trailing_stop"))
                            continue
                            
                except Exception as e:
                    logger.error(f"Error updating trailing stop for {symbol}: {e}")
            
            # 初始止损底线（8%）
            if pos.current_profit_ratio <= -INITIAL_STOP_LOSS_RATIO:
                sell_list.append((symbol, "stop_loss"))
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
    
    def get_portfolio_value(self, positions: Dict[str, V50Position], date_str: str, price_df: pl.DataFrame) -> float:
        market_value = 0.0
        for symbol, pos in positions.items():
            try:
                row = price_df.filter((pl.col('symbol') == symbol) & (pl.col('trade_date') == date_str)).select('close').row(0)
                if row:
                    market_value += pos.shares * float(row[0])
            except Exception:
                market_value += pos.shares * pos.current_price
        return self.cash + market_value
    
    def get_market_regime_stats(self) -> Dict[str, Any]:
        return {
            'is_risk_period': self.is_risk_period,
            'current_regime': 'RISK' if self.is_risk_period else 'NORMAL',
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
            return False, f"[V50 FAILED: OVER-TRADING] Trade count {total_trades} exceeds threshold {TRADE_COUNT_FAIL_THRESHOLD}"
        return True, ""
    
    def get_position_sizing_stats(self) -> Dict[str, Any]:
        """V50 个股波动率头寸管理统计"""
        return {
            'risk_target_per_position': RISK_TARGET_PER_POSITION,
            'max_single_position_pct': MAX_SINGLE_POSITION_PCT,
            'initial_atr_mult': TIERED_EXIT_CONFIG['initial_atr_mult'],
            'position_sizing_method': 'Individual Stock Volatility',
        }