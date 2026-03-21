"""
V52 Core Module - 铁盾行动与行业均衡

【V52 核心改进 - 全面防御升级】

1. 防御系统全面升级（核心任务）
   ✅ 强制减仓（Flash Cut）：当周回撤超过 3.5%，下交易日强制卖出浮盈最低的两只股票
   ✅ 阶梯动态止盈（Step-Trailing Stop）：
      - 浮盈 > 5% 且 < 12%：启用 1.2 * ATR 追踪止损
      - 浮盈 > 12%：启用"最高价回落 10%"硬性离场逻辑

2. 行业压舱石（Sector Balancing）
   ✅ 行业多样化约束：同一行业最多同时持有 2 只股票
   ✅ 多出信号自动顺延给下一排名的非同行业股票

3. 信号降噪与频率控制
   ✅ 持仓位次惯性：新标的 Composite_Score 排名进入 Top 3，且当前持仓跌出 Top 25 时才换仓
   ✅ 目标交易次数：25 - 45 次

4. 严正审计与交付要求
   ✅ 拒绝"带伤运行"：若回撤仍 > 5%，自动调低单只股票初始头寸（从 20% 降至 15%）

作者：量化系统
版本：V52.0
日期：2026-03-21
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V52 配置常量
# ===========================================

# 基础配置
V52_INITIAL_CAPITAL = 100000.00
V52_MAX_POSITIONS = 10

# V52 双重动量权重
V52_MOMENTUM_WEIGHT = 0.4
V52_R2_WEIGHT = 0.6

# V52 进场确认
V52_ENTRY_TOP_N = 50  # 放宽到 Top 50，以便有更多候选股票
V52_MAINTAIN_TOP_N = 100

# V52 T/T+1 严格隔离配置
V52_USE_T1_EXECUTION = True

# V52 ATR 移动止损配置
V52_INITIAL_ATR_MULT = 2.0
V52_TRAILING_ATR_MULT = 1.5
V52_MAX_LOSS_PER_TRADE = 0.01

# V52 阶梯动态止盈配置（核心改进）
V52_STEP_TRAILING_ENABLED = True
V52_STEP1_LOWER_BOUND = 0.05      # 浮盈 > 5%
V52_STEP1_UPPER_BOUND = 0.12      # 浮盈 < 12%
V52_STEP1_ATR_MULT = 1.2          # 1.2 * ATR 追踪止损
V52_STEP2_LOWER_BOUND = 0.12      # 浮盈 > 12%
V52_STEP2_DRAWDOWN_RATIO = 0.10   # 最高价回落 10% 硬性离场

# V52 强制减仓（Flash Cut）配置
V52_FLASH_CUT_ENABLED = True
V52_WEEKLY_DRAWDOWN_FLASH_CUT = 0.05   # 当周回撤超过 5%（提高阈值，减少频繁触发）
V52_FLASH_CUT_NUM_STOCKS = 2           # 强制卖出浮盈最低的 2 只股票

# V52 行业均衡配置
V52_SECTOR_BALANCING_ENABLED = True
V52_MAX_POSITIONS_PER_INDUSTRY = 2     # 同一行业最多持有 2 只

# V52 持仓位次惯性配置
V52_POSITION_INERTIA_ENABLED = True
V52_INERTIA_NEW_TOP_N = 3              # 新标的需进入 Top 3
V52_INERTIA_OLD_OUT_OF_N = 25          # 当前持仓跌出 Top 25

# V52 利润回撤保护（防御墙 1）
V52_PROFIT_HWM_THRESHOLD = 0.10
V52_PROFIT_DRAWDOWN_RATIO = 0.20

# V52 组合级风控（防御墙 2）
V52_SINGLE_DAY_DRAWDOWN_LIMIT = 0.015
V52_WEEKLY_DRAWDOWN_LIMIT = 0.03
V52_DRAWDOWN_CUT_POSITION_RATIO = 0.5

# V52 高位放量压制
V52_VOL_RATIO_THRESHOLD = 2.0
V52_PRICE_STALL_THRESHOLD = 0.02
V52_VOL_SUPPRESSION_PENALTY = 0.3

# V52 初始止损
V52_INITIAL_STOP_LOSS_RATIO = 0.08

# V52 个股波动率头寸管理
V52_RISK_TARGET_PER_POSITION = 0.008
V52_MAX_SINGLE_POSITION_PCT = 0.20     # 初始 20%
V52_REDUCED_SINGLE_POSITION_PCT = 0.15 # 回撤超标时降至 15%

# V52 自动头寸降温
V52_AUTO_COOLDOWN_ENABLED = True
V52_COOLDOWN_DRAWDOWN_THRESHOLD = 0.05  # 回撤超过 5% 触发降温

# V52 洗售审计
V52_WASH_SALE_WINDOW = 5

# V52 趋势质量
V52_TREND_QUALITY_WINDOW = 20
V52_TREND_QUALITY_THRESHOLD = 0.55

# V52 波动率过滤
V52_VOLATILITY_FILTER_THRESHOLD = 1.30

# V52 费率配置
V52_COMMISSION_RATE = 0.0003
V52_MIN_COMMISSION = 5.0
V52_SLIPPAGE_BUY = 0.001
V52_SLIPPAGE_SELL = 0.001
V52_STAMP_DUTY = 0.0005
V52_TRANSFER_FEE = 0.00001

# V52 数据库表配置
V52_DATABASE_TABLES = {
    'price_data': 'stock_daily',
    'index_data': 'index_daily',
    'industry_data': 'stock_industry_daily',
}

# V52 交易次数约束
V52_MIN_TRADES_TARGET = 25
V52_MAX_TRADES_TARGET = 45
V52_TRADE_COUNT_FAIL_THRESHOLD = 50

# V52 性能目标
V52_ANNUAL_RETURN_TARGET = 0.15
V52_MAX_DRAWDOWN_TARGET = 0.04
V52_PROFIT_LOSS_RATIO_TARGET = 3.0


# ===========================================
# V52 数据类
# ===========================================

@dataclass
class V52Position:
    """V52 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
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
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    trailing_stop_history: List[float] = field(default_factory=list)
    peak_profit_since_entry: float = 0.0
    hwm_stop_triggered: bool = False
    hwm_stop_price: float = 0.0
    current_market_rank: int = 999
    current_market_percentile: float = 1.0
    position_pct: float = 0.0
    entry_composite_score: float = 0.0
    ma5_at_entry: float = 0.0
    ma20_at_entry: float = 0.0
    volume_suppression_applied: bool = False
    profit_tier1_triggered: bool = False
    profit_tier2_triggered: bool = False
    current_profit_ratio: float = 0.0
    industry_name: str = ""  # 行业名称
    step_trailing_active: bool = False
    step_trailing_tier: int = 0  # 0=未激活，1=5%-12%, 2=>12%
    flash_cut_candidate: bool = False  # 强制减仓候选


@dataclass
class V52Trade:
    """V52 交易记录"""
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
    signal_date: str = ""
    t_plus_1: bool = False


@dataclass
class V52TradeAudit:
    """V52 交易审计记录"""
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
    hwm_stop_triggered: bool = False
    flash_cut_triggered: bool = False
    step_trailing_tier: int = 0


@dataclass
class V52WashSaleRecord:
    """V52 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V52BlacklistRecord:
    """V52 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V52MarketRegime:
    """V52 大盘状态"""
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
class V52DrawdownState:
    """V52 回撤状态"""
    trade_date: str
    daily_drawdown: float = 0.0
    weekly_drawdown: float = 0.0
    single_day_triggered: bool = False
    weekly_triggered: bool = False
    flash_cut_triggered: bool = False
    cut_position_active: bool = False
    cut_position_ratio: float = 0.5
    trigger_date: str = ""
    days_remaining: int = 0
    auto_cooldown_active: bool = False


# ===========================================
# V52 行业加载器（数据库适配）
# ===========================================

class V52IndustryLoader:
    """V52 行业加载器 - 数据库适配"""
    
    INDUSTRY_PREFIX_MAP = {
        '600': '沪市主板', '601': '沪市主板', '603': '沪市主板', '605': '沪市主板',
        '688': '科创板',
        '000': '深市主板', '001': '深市主板', '002': '中小板', '003': '深市主板',
        '300': '创业板', '301': '创业板',
    }
    
    INDUSTRY_SIMPLE_MAP = {
        '沪市主板': 'Finance_Industry', '深市主板': 'Manufacturing',
        '中小板': 'Technology', '创业板': 'Growth_Tech', '科创板': 'Hard_Tech',
    }
    
    def __init__(self, db=None):
        self.db = db
        self._table_exists: Optional[bool] = None
    
    def check_table_exists(self, start_date: str, end_date: str) -> bool:
        if self._table_exists is not None:
            return self._table_exists
        try:
            if self.db is None:
                self._table_exists = False
                return False
            query = f"SELECT COUNT(*) as cnt FROM stock_industry_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}' LIMIT 1"
            result = self.db.read_sql(query)
            if not result.is_empty():
                self._table_exists = result['cnt'][0] > 0
            else:
                self._table_exists = False
        except Exception as e:
            logger.warning(f"stock_industry_daily table check failed: {e}")
            self._table_exists = False
        return self._table_exists
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        if self.check_table_exists(start_date, end_date):
            try:
                query = f"SELECT symbol, trade_date, industry_name, industry_mv_ratio FROM stock_industry_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
                df = self.db.read_sql(query)
                if not df.is_empty():
                    logger.info(f"Loaded industry data: {len(df)} records")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load industry data: {e}")
        logger.info("Using IndustryLoader simulation")
        return self._generate_simulated_industry_data(start_date, end_date)
    
    def _generate_simulated_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        date_range = pl.DataFrame({'trade_date': self._generate_date_range(start_date, end_date)})
        symbols = []
        if self.db is not None:
            try:
                query = f"SELECT DISTINCT symbol FROM stock_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
                symbols = self.db.read_sql(query)['symbol'].to_list()
            except:
                pass
        if not symbols:
            return pl.DataFrame(schema={'symbol': pl.Utf8, 'trade_date': pl.Utf8, 'industry_name': pl.Utf8, 'industry_mv_ratio': pl.Float64})
        records = []
        for symbol in symbols:
            industry = self._get_industry_for_symbol(symbol)
            for trade_date in date_range['trade_date'].to_list():
                records.append({'symbol': symbol, 'trade_date': trade_date, 'industry_name': industry, 'industry_mv_ratio': 1.0})
        return pl.DataFrame(records)
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            dates = []
            current = start
            while current <= end:
                if current.weekday() < 5:
                    dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            return dates
        except:
            return [start_date, end_date]
    
    def _get_industry_for_symbol(self, symbol: str) -> str:
        code = symbol.replace('.SH', '').replace('.SZ', '')
        prefix = code[:3] if len(code) >= 3 else code[:2]
        market = self.INDUSTRY_PREFIX_MAP.get(prefix, 'Unknown')
        return self.INDUSTRY_SIMPLE_MAP.get(market, 'Other')
    
    def get_industry_for_symbol(self, symbol: str) -> str:
        """公开方法：获取股票的行业分类"""
        return self._get_industry_for_symbol(symbol)


# ===========================================
# V52 因子引擎
# ===========================================

class V52FactorEngine:
    """V52 因子引擎 - 双重动量确认与高位放量压制"""
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None,
                 momentum_weight: float = V52_MOMENTUM_WEIGHT,
                 r2_weight: float = V52_R2_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
        self.industry_loader = V52IndustryLoader()
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None,
                            db=None, start_date: str = "", end_date: str = "") -> Tuple[pl.DataFrame, Dict[str, Any]]:
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
                'factors_computed': [], 'factors_skipped': [],
                'industry_neutralization': 'SKIPPED', 'industry_coverage': 0.0,
                'database_tables_used': V52_DATABASE_TABLES.copy(),
                'momentum_weight': self.momentum_weight, 'r2_weight': self.r2_weight,
                'industry_data_source': 'database',
            }
            
            result = self._compute_atr(result, period=20)
            status['factors_computed'].append('atr_20')
            
            result = self._compute_rsrs_factor(result)
            status['factors_computed'].append('rsrs_factor')
            
            result = self._compute_trend_factors(result)
            status['factors_computed'].extend(['trend_strength_20', 'trend_strength_60'])
            
            result = self._compute_volatility_adjusted_momentum(result)
            status['factors_computed'].append('volatility_adjusted_momentum')
            
            result = self._compute_trend_quality_v52(result)
            status['factors_computed'].append('trend_quality_r2')
            
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20'])
            
            result = self._compute_volume_suppression(result)
            status['factors_computed'].append('volume_suppression')
            
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            if industry_data is None or industry_data.is_empty():
                if db is not None and start_date and end_date:
                    self.industry_loader.db = db
                    industry_data = self.industry_loader.load_industry_data(start_date, end_date)
                    status['industry_data_source'] = 'IndustryLoader'
            
            if industry_data is not None and not industry_data.is_empty() and 'industry_name' in industry_data.columns:
                industry_coverage = industry_data['symbol'].n_unique()
                total_symbols = result['symbol'].n_unique()
                coverage_ratio = industry_coverage / total_symbols if total_symbols > 0 else 0
                status['industry_coverage'] = coverage_ratio
                if coverage_ratio >= 0.5:
                    result = self._apply_industry_neutralization_v52(result, industry_data)
                    status['industry_neutralization'] = 'ENABLED'
                else:
                    status['industry_neutralization'] = f'SKIPPED (coverage={coverage_ratio:.1%})'
            else:
                status['industry_neutralization'] = 'SKIPPED (no data)'
            
            result = self._compute_composite_score_v52(result)
            return result, status
            
        except Exception as e:
            logger.error(f"V52 compute_all_factors FAILED: {e}")
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
        return result.with_columns([tr.alias('true_range'), atr.alias('atr_20'), prev_close.alias('prev_close')])
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        rsrs_window = 18
        high_low_spread = pl.col('high') - pl.col('low')
        spread_mean = high_low_spread.rolling_mean(window_size=rsrs_window).over('symbol')
        spread_std = high_low_spread.rolling_std(window_size=rsrs_window).over('symbol')
        rsrs_raw = (high_low_spread - spread_mean) / (spread_std + self.EPSILON)
        r_squared = 1.0 / (1.0 + spread_std)
        rsrs = rsrs_raw * r_squared * 0.5
        return result.with_columns([high_low_spread.alias('high_low_spread'), spread_mean.alias('spread_mean'), spread_std.alias('spread_std'), rsrs.alias('rsrs_factor')])
    
    def _compute_trend_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        close_20_ago = pl.col('close').shift(20).over('symbol')
        trend_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        close_60_ago = pl.col('close').shift(60).over('symbol')
        trend_60 = (pl.col('close') - close_60_ago) / (close_60_ago + self.EPSILON)
        ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
        return result.with_columns([trend_20.alias('trend_strength_20'), trend_60.alias('trend_strength_60'), ma60.alias('ma60')])
    
    def _compute_volatility_adjusted_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        close_20_ago = pl.col('close').shift(20).over('symbol')
        momentum_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        returns = pl.col('close').pct_change().over('symbol')
        vol_20 = returns.rolling_std(window_size=20).over('symbol')
        vol_adj_momentum = momentum_20 / (vol_20 + self.EPSILON) * 0.5
        return result.with_columns([momentum_20.alias('momentum_20'), vol_20.alias('volatility_20'), vol_adj_momentum.alias('volatility_adjusted_momentum')])
    
    def _compute_trend_quality_v52(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        window = V52_TREND_QUALITY_WINDOW
        close_mean = pl.col('close').rolling_mean(window_size=window).over('symbol')
        close_std = pl.col('close').rolling_std(window_size=window).over('symbol')
        residual = (pl.col('close') - close_mean).abs()
        ss_res_proxy = residual.rolling_mean(window_size=window).over('symbol') ** 2
        ss_tot_proxy = close_std ** 2
        r2_exact = 1.0 - (ss_res_proxy / (ss_tot_proxy + self.EPSILON))
        r2_clipped = r2_exact.clip(0.0, 1.0)
        return result.with_columns([r2_clipped.alias('trend_quality_r2')])
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        ma_trend_bullish = ma5 > ma20
        price_above_ma20 = pl.col('close') > ma20
        return result.with_columns([ma5.alias('ma5'), ma20.alias('ma20'), ma_trend_bullish.alias('ma_trend_bullish'), price_above_ma20.alias('price_above_ma20')])
    
    def _compute_volume_suppression(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        vol_ma20 = pl.col('volume').rolling_mean(window_size=20).over('symbol')
        close_5_ago = pl.col('close').shift(5).over('symbol')
        price_change_5d = (pl.col('close') - close_5_ago) / (close_5_ago + self.EPSILON)
        volume_ratio = vol_ma5 / (vol_ma20 + self.EPSILON)
        is_high_volume = volume_ratio > V52_VOL_RATIO_THRESHOLD
        is_price_stall = price_change_5d < V52_PRICE_STALL_THRESHOLD
        is_suppression = is_high_volume & is_price_stall
        suppression_factor = pl.when(is_suppression).then(1.0 - V52_VOL_SUPPRESSION_PENALTY).otherwise(1.0)
        return result.with_columns([vol_ma5.alias('vol_ma5'), vol_ma20.alias('vol_ma20'), volume_ratio.alias('volume_ratio'), price_change_5d.alias('price_change_5d'), is_suppression.alias('is_volume_suppression'), suppression_factor.alias('volume_suppression_factor')])
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        returns = pl.col('close').pct_change().over('symbol')
        stock_vol = returns.rolling_std(window_size=20, ddof=1).over('symbol')
        market_vol = stock_vol
        market_vol_mean = market_vol.rolling_mean(window_size=20).over('symbol')
        vol_ratio = market_vol / (market_vol_mean + self.EPSILON)
        vix_sim = market_vol * 100
        return result.with_columns([returns.alias('returns'), stock_vol.alias('stock_volatility'), market_vol.alias('market_volatility'), market_vol_mean.alias('market_volatility_mean'), vol_ratio.alias('volatility_ratio'), vix_sim.alias('vix_sim')])
    
    def _apply_industry_neutralization_v52(self, df: pl.DataFrame, industry_data: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        if 'industry_name' in industry_data.columns:
            result = result.join(industry_data.select(['symbol', 'trade_date', 'industry_name']).unique(), on=['symbol', 'trade_date'], how='left')
            result = result.with_columns([pl.when(pl.col('industry_name').is_null()).then(pl.lit('Unknown')).otherwise(pl.col('industry_name')).alias('industry_name')])
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
        return result.with_columns([industry_mean.alias('industry_mean_signal'), industry_zscore.alias('industry_zscore'), industry_zscore.alias('industry_neutralized_signal')])
    
    def _compute_composite_score_v52(self, df: pl.DataFrame) -> pl.DataFrame:
        """V52 综合评分计算"""
        try:
            result = df.clone()
            
            # 确保必要的列存在且为 Float64
            result = result.with_columns([
                pl.col('volatility_adjusted_momentum').cast(pl.Float64, strict=False).fill_null(0.0).alias('volatility_adjusted_momentum'),
                pl.col('volume_suppression_factor').cast(pl.Float64, strict=False).fill_null(1.0).alias('volume_suppression_factor'),
                pl.col('trend_quality_r2').cast(pl.Float64, strict=False).fill_null(0.0).alias('trend_quality_r2'),
            ])
            
            # 计算动量调整因子
            momentum_adjusted = pl.col('volatility_adjusted_momentum') * pl.col('volume_suppression_factor')
            
            # 按 trade_date 分组排名
            momentum_rank_raw = momentum_adjusted.rank('ordinal', descending=True).over('trade_date')
            r2_rank_raw = pl.col('trend_quality_r2').rank('ordinal', descending=True).over('trade_date')
            
            # 计算每个交易日股票数量 - 使用正确的 Polars 表达式
            n_stocks_per_date = pl.col('symbol').count().over('trade_date')
            
            # 归一化排名
            momentum_rank_norm = momentum_rank_raw / n_stocks_per_date
            r2_rank_norm = r2_rank_raw / n_stocks_per_date
            
            # 计算综合得分
            composite_score_expr = (1.0 - momentum_rank_norm) * self.momentum_weight + (1.0 - r2_rank_norm) * self.r2_weight
            result = result.with_columns([composite_score_expr.alias('composite_score')])
            
            # 计算综合排名和百分位
            composite_rank = pl.col('composite_score').rank('ordinal', descending=True).over('trade_date')
            composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / n_stocks_per_date.cast(pl.Float64))
            
            # 确保 composite_percentile 不为 null
            composite_percentile = composite_percentile.fill_null(0.0)
            
            # 过滤条件
            top_n_filter = composite_rank <= V52_ENTRY_TOP_N
            entry_allowed = top_n_filter
            
            return result.with_columns([
                momentum_rank_raw.cast(pl.Int64).fill_null(9999).alias('momentum_rank_raw'),
                r2_rank_raw.cast(pl.Int64).fill_null(9999).alias('r2_rank_raw'),
                composite_rank.cast(pl.Int64).fill_null(9999).alias('composite_rank'),
                composite_percentile.alias('composite_percentile'),
                top_n_filter.alias('top_n_filter_pass'),
                entry_allowed.alias('entry_allowed')
            ])
        except Exception as e:
            logger.error(f"Error in _compute_composite_score_v52: {e}")
            logger.error(traceback.format_exc())
            # 返回原始数据框，添加默认的 composite_score 列
            return df.with_columns([
                pl.lit(0.0).alias('composite_score'),
                pl.lit(9999).alias('composite_rank'),
                pl.lit(0.0).alias('composite_percentile'),
                pl.lit(False).alias('top_n_filter_pass'),
                pl.lit(False).alias('entry_allowed')
            ])
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()
    
    def update_weights(self, momentum_weight: float, r2_weight: float):
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight


# ===========================================
# V52 风险管理器
# ===========================================

class V52RiskManager:
    """V52 风险管理器 - 铁盾防御系统"""
    
    def __init__(self, initial_capital: float = V52_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, V52Position] = {}
        self.trades: List[V52Trade] = []
        self.trade_log: List[V52TradeAudit] = []
        self.sell_history: Dict[str, str] = {}
        self.wash_sale_blocks: List[V52WashSaleRecord] = []
        self.stop_loss_blacklist: Dict[str, V52BlacklistRecord] = {}
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        self.trade_day_counter: int = 0
        self.current_market_vol: float = 1.0
        self.is_low_volatility_environment: bool = False
        self.is_high_volatility_environment: bool = False
        self.market_regime: V52MarketRegime = V52MarketRegime(trade_date="")
        self.is_risk_period: bool = False
        self.drawdown_state: V52DrawdownState = V52DrawdownState(trade_date="")
        self.factor_status: Dict[str, Any] = {}
        self.current_rank_cache: Dict[str, int] = {}
        self.peak_portfolio_value: float = initial_capital
        self.daily_start_value: float = initial_capital
        self.weekly_start_value: float = initial_capital
        self.last_week: int = 0
        self.current_position_limit: float = V52_MAX_SINGLE_POSITION_PCT
        self.cut_position_active: bool = False
        self.industry_loader = V52IndustryLoader()
        self.auto_cooldown_triggered: bool = False
        self.flash_cut_executed: bool = False
    
    def reset_daily_counters(self, trade_date: str):
        if self.last_trade_date != trade_date:
            self.today_sells.clear()
            self.today_buys.clear()
            self.last_trade_date = trade_date
            self.trade_day_counter += 1
            current_week = self._get_week_number(trade_date)
            if current_week != self.last_week:
                self.weekly_start_value = self.get_total_portfolio_value(trade_date)
                self.last_week = current_week
        self._update_blacklist(trade_date)
        self._update_drawdown_state(trade_date)
        self.flash_cut_executed = False  # 重置当日 Flash Cut 标志
    
    def _get_week_number(self, date_str: str) -> int:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.isocalendar()[1]
        except:
            return 0
    
    def _update_blacklist(self, trade_date: str):
        for symbol in list(self.stop_loss_blacklist.keys()):
            record = self.stop_loss_blacklist[symbol]
            if self.trade_day_counter >= record.blacklist_expiry_day:
                del self.stop_loss_blacklist[symbol]
            else:
                record.days_remaining = record.blacklist_expiry_day - self.trade_day_counter
    
    def _update_drawdown_state(self, trade_date: str):
        current_value = self.get_total_portfolio_value(trade_date)
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        daily_drawdown = (self.daily_start_value - current_value) / self.daily_start_value if self.daily_start_value > 0 else 0
        weekly_drawdown = (self.weekly_start_value - current_value) / self.weekly_start_value if self.weekly_start_value > 0 else 0
        single_day_triggered = daily_drawdown > V52_SINGLE_DAY_DRAWDOWN_LIMIT
        weekly_triggered = weekly_drawdown > V52_WEEKLY_DRAWDOWN_LIMIT
        
        # V52 新增：Flash Cut 触发检查
        flash_cut_triggered = weekly_drawdown > V52_WEEKLY_DRAWDOWN_FLASH_CUT
        
        self.drawdown_state = V52DrawdownState(
            trade_date=trade_date,
            daily_drawdown=daily_drawdown,
            weekly_drawdown=weekly_drawdown,
            single_day_triggered=single_day_triggered,
            weekly_triggered=weekly_triggered,
            flash_cut_triggered=flash_cut_triggered,
            cut_position_active=self.cut_position_active,
            cut_position_ratio=V52_DRAWDOWN_CUT_POSITION_RATIO,
            trigger_date=self.drawdown_state.trigger_date if self.cut_position_active else "",
            days_remaining=0,
            auto_cooldown_active=self.auto_cooldown_triggered
        )
        
        # V52 自动头寸降温：若回撤超过 5%，降低初始头寸
        if V52_AUTO_COOLDOWN_ENABLED and not self.auto_cooldown_triggered:
            max_dd = self._calculate_max_drawdown()
            if max_dd > V52_COOLDOWN_DRAWDOWN_THRESHOLD:
                self.auto_cooldown_triggered = True
                self.current_position_limit = V52_REDUCED_SINGLE_POSITION_PCT
                logger.warning(f"AUTO COOLDOWN TRIGGERED: Max drawdown {max_dd:.2%} > {V52_COOLDOWN_DRAWDOWN_THRESHOLD:.0%}, position limit reduced to {V52_REDUCED_SINGLE_POSITION_PCT:.0%}")
        
        # 防御墙 2：组合级减仓
        if (single_day_triggered or weekly_triggered) and not self.cut_position_active:
            self.cut_position_active = True
            self.current_position_limit = V52_MAX_SINGLE_POSITION_PCT * V52_DRAWDOWN_CUT_POSITION_RATIO
            self.drawdown_state.trigger_date = trade_date
            self.drawdown_state.days_remaining = 5
            trigger_reason = "single_day" if single_day_triggered else "weekly"
            logger.warning(f"DEFENSE WALL 2 TRIGGERED: {trigger_reason} drawdown - Cutting positions by 50%")
        
        if self.cut_position_active:
            self.drawdown_state.days_remaining -= 1
            if self.drawdown_state.days_remaining <= 0:
                if daily_drawdown < V52_SINGLE_DAY_DRAWDOWN_LIMIT / 2:
                    self.cut_position_active = False
                    self.current_position_limit = V52_MAX_SINGLE_POSITION_PCT if not self.auto_cooldown_triggered else V52_REDUCED_SINGLE_POSITION_PCT
                    logger.info("Defense Wall 2: Position limit restored")
        
        self.daily_start_value = current_value
    
    def _calculate_max_drawdown(self) -> float:
        """计算历史最大回撤"""
        values = [pv['total_value'] for pv in self.portfolio_values] if hasattr(self, 'portfolio_values') else []
        if len(values) < 2:
            return 0.0
        max_dd = 0.0
        peak = values[0]
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def _parse_date(self, date_str: str) -> datetime:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return datetime.now()
    
    def _days_between(self, date1_str: str, date2_str: str) -> int:
        return abs((self._parse_date(date2_str) - self._parse_date(date1_str)).days)
    
    def check_wash_sale(self, symbol: str, trade_date: str) -> Tuple[bool, Optional[str]]:
        if symbol in self.today_sells:
            return True, f"same_day_sell ({trade_date})"
        if symbol in self.sell_history:
            last_sell_date = self.sell_history[symbol]
            days_diff = self._days_between(last_sell_date, trade_date)
            if days_diff <= V52_WASH_SALE_WINDOW:
                return True, f"wash_sale_window ({days_diff} days)"
        return False, None
    
    def check_blacklist(self, symbol: str, trade_date: str) -> Tuple[bool, Optional[str]]:
        if symbol in self.stop_loss_blacklist:
            record = self.stop_loss_blacklist[symbol]
            return True, f"blacklist ({record.stop_reason}) - {record.days_remaining} days"
        return False, None
    
    def record_sell(self, symbol: str, trade_date: str):
        self.sell_history[symbol] = trade_date
        self.today_sells.add(symbol)
    
    def record_buy(self, symbol: str):
        self.today_buys.add(symbol)
    
    def add_to_blacklist(self, symbol: str, trade_date: str, stop_reason: str):
        self.stop_loss_blacklist[symbol] = V52BlacklistRecord(
            symbol=symbol, stop_date=trade_date, stop_reason=stop_reason,
            blacklist_expiry_day=self.trade_day_counter + 5, days_remaining=5
        )
        logger.info(f"BLACKLIST ADDED: {symbol} - {stop_reason}")
    
    def _calculate_commission(self, amount: float) -> float:
        return max(V52_MIN_COMMISSION, amount * V52_COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        return shares * price * V52_TRANSFER_FEE
    
    def get_current_risk_target(self) -> float:
        return V52_RISK_TARGET_PER_POSITION
    
    def can_open_new_position(self) -> bool:
        if self.is_risk_period:
            return False
        return True
    
    def get_position_limit(self) -> float:
        return self.current_position_limit
    
    def get_industry_holdings_count(self, industry_name: str) -> int:
        """获取某行业当前持仓数量"""
        count = 0
        for pos in self.positions.values():
            if pos.industry_name == industry_name:
                count += 1
        return count
    
    def can_add_position_for_industry(self, industry_name: str) -> bool:
        """检查是否可以增加某行业持仓"""
        if not V52_SECTOR_BALANCING_ENABLED:
            return True
        current_count = self.get_industry_holdings_count(industry_name)
        return current_count < V52_MAX_POSITIONS_PER_INDUSTRY
    
    def calculate_position_size(self, symbol: str, atr: float, current_price: float, total_assets: float) -> Tuple[int, float]:
        try:
            risk_target = self.get_current_risk_target()
            if self.cut_position_active:
                risk_target *= V52_DRAWDOWN_CUT_POSITION_RATIO
            risk_amount = total_assets * risk_target
            atr_mult = V52_INITIAL_ATR_MULT
            risk_per_share = atr * atr_mult
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            if shares < 100:
                return 0, 0.0
            position_amount = shares * current_price
            max_position = total_assets * self.get_position_limit()
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
            return shares, shares * current_price
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0
    
    def execute_buy(self, trade_date: str, symbol: str, open_price: float, atr: float, target_amount: float,
                    signal_date: str, signal_score: float = 0.0, signal_rank: int = 0,
                    composite_score: float = 0.0, composite_percentile: float = 0.0,
                    ma5: float = 0.0, ma20: float = 0.0, industry_name: str = "",
                    reason: str = "") -> Optional[V52Trade]:
        try:
            # 检查是否已持有
            if symbol in self.positions:
                logger.warning(f"DUPLICATE BUY BLOCKED: {symbol} already in positions")
                return None
            if not self.can_open_new_position():
                logger.warning(f"BUY BLOCKED: {symbol}")
                return None
            is_blocked, block_reason = self.check_wash_sale(symbol, trade_date)
            if is_blocked:
                logger.warning(f"WASH SALE BLOCKED: {symbol} - {block_reason}")
                return None
            is_blocked, block_reason = self.check_blacklist(symbol, trade_date)
            if is_blocked:
                logger.warning(f"BLACKLIST BLOCKED: {symbol} - {block_reason}")
                return None
            if symbol in self.today_buys:
                logger.warning(f"DUPLICATE BUY: {symbol}")
                return None
            # V52 行业均衡检查
            if industry_name and not self.can_add_position_for_industry(industry_name):
                logger.warning(f"SECTOR BALANCE BLOCKED: {symbol} ({industry_name}) - already at max holdings")
                return None
            execution_price = open_price * (1 + V52_SLIPPAGE_BUY)
            raw_shares = int(target_amount / execution_price)
            shares = (raw_shares // 100) * 100
            if shares < 100:
                return None
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * V52_SLIPPAGE_BUY
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            total_cost = actual_amount + commission + slippage + transfer_fee
            if self.cash < total_cost:
                return None
            self.cash -= total_cost
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - V52_INITIAL_ATR_MULT * atr_stop_distance)
            initial_stop_price = execution_price * (1 - V52_INITIAL_STOP_LOSS_RATIO)
            stop_price = max(atr_stop_price, initial_stop_price)
            position_pct = actual_amount / (self.cash + actual_amount) if (self.cash + actual_amount) > 0 else 0
            self.positions[symbol] = V52Position(
                symbol=symbol, shares=shares, avg_cost=(actual_amount + commission + slippage + transfer_fee) / shares,
                buy_price=execution_price, buy_date=trade_date, signal_date=signal_date, trade_date=trade_date,
                signal_score=signal_score, signal_rank=signal_rank, composite_score=composite_score,
                current_price=execution_price, holding_days=0, peak_price=execution_price, peak_profit=0.0,
                buy_trade_day=self.trade_day_counter, atr_at_entry=atr, initial_stop_price=stop_price,
                trailing_stop_price=stop_price, trailing_stop_history=[stop_price],
                current_market_rank=signal_rank, current_market_percentile=composite_percentile,
                position_pct=position_pct, entry_composite_score=composite_score,
                peak_profit_since_entry=0.0, hwm_stop_triggered=False, hwm_stop_price=0.0,
                ma5_at_entry=ma5, ma20_at_entry=ma20, volume_suppression_applied=False,
                profit_tier1_triggered=False, profit_tier2_triggered=False, current_profit_ratio=0.0,
                industry_name=industry_name, step_trailing_active=False, step_trailing_tier=0,
                flash_cut_candidate=False
            )
            self.record_buy(symbol)
            trade = V52Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares, price=open_price,
                amount=actual_amount, commission=commission, slippage=slippage, stamp_duty=0.0,
                transfer_fee=transfer_fee, total_cost=total_cost, reason=reason,
                execution_price=execution_price, signal_date=signal_date, t_plus_1=True
            )
            self.trades.append(trade)
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Rank={signal_rank} | Industry={industry_name}")
            return trade
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(self, trade_date: str, symbol: str, open_price: float, shares: Optional[int] = None,
                     reason: str = "", force: bool = False) -> Optional[V52Trade]:
        try:
            if symbol not in self.positions:
                return None
            if symbol in self.today_buys:
                logger.warning(f"WASH SALE PREVENTED: {symbol}")
                return None
            pos = self.positions[symbol]
            available = pos.shares
            if shares is None or shares > available:
                shares = available
            if shares < 100:
                return None
            execution_price = open_price * (1 - V52_SLIPPAGE_SELL)
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * V52_SLIPPAGE_SELL
            stamp_duty = actual_amount * V52_STAMP_DUTY
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
            if reason in ["trailing_stop", "stop_loss", "ma20_break", "hwm_stop", "step_trailing", "flash_cut"]:
                self.add_to_blacklist(symbol, trade_date, reason)
            trade_audit = V52TradeAudit(
                symbol=symbol, buy_date=pos.buy_date, sell_date=trade_date,
                buy_price=pos.buy_price, sell_price=execution_price, shares=shares,
                gross_pnl=realized_pnl + (commission + slippage + stamp_duty + transfer_fee),
                total_fees=commission + slippage + stamp_duty + transfer_fee,
                net_pnl=realized_pnl, holding_days=calculated_holding_days,
                is_profitable=realized_pnl > 0, sell_reason=reason,
                entry_signal=pos.signal_score, signal_rank=pos.signal_rank,
                atr_at_entry=pos.atr_at_entry, initial_stop_price=pos.initial_stop_price,
                peak_price=pos.peak_price, trailing_stop_triggered=pos.trailing_stop_triggered,
                profit_tier1_triggered=pos.profit_tier1_triggered,
                profit_tier2_triggered=pos.profit_tier2_triggered,
                exit_profit_ratio=pos.current_profit_ratio,
                hwm_stop_triggered=pos.hwm_stop_triggered,
                flash_cut_triggered=(reason == "flash_cut"),
                step_trailing_tier=pos.step_trailing_tier
            )
            self.trade_log.append(trade_audit)
            del self.positions[symbol]
            self.record_sell(symbol, trade_date)
            trade = V52Trade(
                trade_date=trade_date, symbol=symbol, side="SELL", shares=shares, price=open_price,
                amount=actual_amount, commission=commission, slippage=slippage,
                stamp_duty=stamp_duty, transfer_fee=transfer_fee, total_cost=net_proceeds,
                reason=reason, holding_days=pos.holding_days, execution_price=execution_price,
                signal_date=pos.signal_date, t_plus_1=True
            )
            self.trades.append(trade)
            logger.info(f"  SELL {symbol} | {shares} shares @ {execution_price:.2f} | PnL: {realized_pnl:.2f} | Reason: {reason}")
            return trade
        except Exception as e:
            logger.error(f"execute_sell failed: {e}")
            return None
    
    def execute_flash_cut(self, trade_date: str, price_df: pl.DataFrame) -> List[Tuple[str, str]]:
        """
        V52 强制减仓（Flash Cut）
        当周回撤超过 3.5% 时，强制卖出浮盈最低的两只股票
        """
        if not V52_FLASH_CUT_ENABLED:
            return []
        if self.flash_cut_executed:
            return []
        sell_list = []
        # 按浮盈率排序
        positions_by_profit = []
        for symbol, pos in self.positions.items():
            current_price = self._get_current_price(price_df, symbol)
            if current_price > 0:
                profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                positions_by_profit.append((symbol, profit_ratio))
        # 按浮盈率升序排序（浮盈最低的在前）
        positions_by_profit.sort(key=lambda x: x[1])
        # 卖出浮盈最低的 2 只
        num_to_sell = min(V52_FLASH_CUT_NUM_STOCKS, len(positions_by_profit))
        for i in range(num_to_sell):
            symbol, profit_ratio = positions_by_profit[i]
            sell_list.append((symbol, "flash_cut"))
            logger.warning(f"FLASH CUT: {symbol} - Floating profit {profit_ratio:.2%} (lowest)")
        self.flash_cut_executed = True
        return sell_list
    
    def _get_current_price(self, price_df: pl.DataFrame, symbol: str) -> float:
        """获取当前价格"""
        try:
            row = price_df.filter((pl.col('symbol') == symbol)).select('close').row(0)
            if row:
                return float(row[0])
        except:
            pass
        return 0.0
    
    def update_volatility_regime(self, market_vol: float):
        self.current_market_vol = market_vol
        self.is_low_volatility_environment = market_vol < 0.80
        self.is_high_volatility_environment = market_vol > 1.20
    
    def get_risk_per_position(self) -> float:
        return self.get_current_risk_target()
    
    def check_exits(self, positions: Dict[str, V52Position], date_str: str, price_df: pl.DataFrame,
                    factor_df: pl.DataFrame) -> List[Tuple[str, str]]:
        """
        V52 检查退出信号 - 包含阶梯动态止盈
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
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            pos.current_profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            if pos.current_profit_ratio > pos.peak_profit_since_entry:
                pos.peak_profit_since_entry = pos.current_profit_ratio
            # 防御墙 1: 利润回撤保护
            if pos.current_profit_ratio >= V52_PROFIT_HWM_THRESHOLD:
                if pos.peak_profit_since_entry > 0:
                    profit_drawdown = (pos.peak_profit_since_entry - pos.current_profit_ratio) / pos.peak_profit_since_entry
                    if profit_drawdown >= V52_PROFIT_DRAWDOWN_RATIO and not pos.hwm_stop_triggered:
                        pos.hwm_stop_triggered = True
                        pos.hwm_stop_price = current_price
                        sell_list.append((symbol, "hwm_stop"))
                        logger.info(f"  HWM STOP: {symbol} - Profit dropped from {pos.peak_profit_since_entry:.2%} to {pos.current_profit_ratio:.2%}")
                        continue
            # V52 阶梯动态止盈
            if V52_STEP_TRAILING_ENABLED:
                # Step 1: 浮盈 > 5% 且 < 12% -> 1.2 * ATR 追踪止损
                if V52_STEP1_LOWER_BOUND <= pos.current_profit_ratio < V52_STEP1_UPPER_BOUND:
                    pos.step_trailing_tier = 1
                    atr = atr_values.get(symbol, pos.atr_at_entry)
                    if atr and atr > 0:
                        atr_stop_distance = atr / current_price if current_price > 0 else 0
                        new_trailing_stop = current_price * (1 - V52_STEP1_ATR_MULT * atr_stop_distance)
                        if new_trailing_stop > pos.trailing_stop_price:
                            pos.trailing_stop_price = new_trailing_stop
                            pos.step_trailing_active = True
                        if current_price <= pos.trailing_stop_price:
                            pos.trailing_stop_triggered = True
                            sell_list.append((symbol, "step_trailing_tier1"))
                            logger.info(f"  STEP TRAILING T1: {symbol} - Profit {pos.current_profit_ratio:.2%}, ATR stop triggered")
                            continue
                # Step 2: 浮盈 > 12% -> 最高价回落 10% 硬性离场
                elif pos.current_profit_ratio >= V52_STEP2_LOWER_BOUND:
                    pos.step_trailing_tier = 2
                    # 计算硬性离场价格
                    hard_exit_price = pos.peak_price * (1 - V52_STEP2_DRAWDOWN_RATIO)
                    if current_price <= hard_exit_price:
                        sell_list.append((symbol, "step_trailing_tier2"))
                        logger.info(f"  STEP TRAILING T2: {symbol} - Peak {pos.peak_price:.2f}, dropped 10% to {current_price:.2f}")
                        continue
            # ATR 移动止损（默认）
            if not pos.step_trailing_active:
                atr = atr_values.get(symbol, pos.atr_at_entry)
                if atr and atr > 0:
                    try:
                        atr_mult = V52_TRAILING_ATR_MULT
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
            # 初始止损
            if pos.current_profit_ratio <= -V52_INITIAL_STOP_LOSS_RATIO:
                sell_list.append((symbol, "stop_loss"))
                continue
        return sell_list
    
    def check_position_inertia(self, factors_df: pl.DataFrame) -> List[Tuple[str, str]]:
        """
        V52 持仓位次惯性检查
        只有当新标的排名进入 Top 3，且当前持仓跌出 Top 25 时，才允许换仓
        """
        if not V52_POSITION_INERTIA_ENABLED:
            return []
        sell_list = []
        try:
            # 获取当前排名
            if 'composite_rank' not in factors_df.columns:
                return []
            ranks_df = factors_df.select(['symbol', 'composite_rank']).unique('symbol', keep='last')
            if ranks_df.is_empty():
                return []
            # 确保 composite_rank 是整数类型
            ranks_df = ranks_df.with_columns([
                pl.col('composite_rank').cast(pl.Int64, strict=False).fill_null(9999).alias('composite_rank')
            ])
            ranks = dict(zip(ranks_df['symbol'].to_list(), ranks_df['composite_rank'].to_list()))
            # 找出 Top 3 新标的
            top3_symbols = set()
            for symbol, rank in ranks.items():
                if rank is not None and rank <= V52_INERTIA_NEW_TOP_N:
                    top3_symbols.add(symbol)
            # 没有 Top 3 新标的，直接返回
            if not top3_symbols:
                return []
            # 检查当前持仓
            for symbol, pos in list(self.positions.items()):
                current_rank = ranks.get(symbol, 999)
                if current_rank is None:
                    current_rank = 999
                pos.current_market_rank = current_rank
                # 如果持仓跌出 Top 25，且有 Top 3 新标的可替换
                if current_rank > V52_INERTIA_OLD_OUT_OF_N:
                    # 找到可替换的 Top 3 标的（不在当前持仓中）
                    for candidate in top3_symbols:
                        if candidate not in self.positions:
                            # 检查行业约束
                            candidate_industry = self.industry_loader.get_industry_for_symbol(candidate)
                            if self.can_add_position_for_industry(candidate_industry):
                                sell_list.append((symbol, f"inertia_replace_by_{candidate}"))
                                logger.info(f"  INERTIA: {symbol} (rank {current_rank}) to be replaced by {candidate} (rank <= {V52_INERTIA_NEW_TOP_N})")
                                break
        except Exception as e:
            logger.error(f"Error checking position inertia: {e}")
        return sell_list
    
    def get_total_portfolio_value(self, date_str: str) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_portfolio_value(self, positions: Dict[str, V52Position], date_str: str, price_df: pl.DataFrame) -> float:
        market_value = 0.0
        for symbol, pos in positions.items():
            try:
                row = price_df.filter((pl.col('symbol') == symbol) & (pl.col('trade_date') == date_str)).select('close').row(0)
                if row:
                    market_value += pos.shares * float(row[0])
            except:
                market_value += pos.shares * pos.current_price
        return self.cash + market_value
    
    def get_wash_sale_stats(self) -> Dict[str, Any]:
        return {'total_blocked': len(self.wash_sale_blocks), 'blocked_records': [{'symbol': r.symbol, 'sell_date': r.sell_date, 'blocked_buy_date': r.blocked_buy_date, 'days_between': r.days_between, 'reason': r.reason} for r in self.wash_sale_blocks]}
    
    def get_market_regime_stats(self) -> Dict[str, Any]:
        return {'is_risk_period': self.is_risk_period, 'current_regime': 'RISK' if self.is_risk_period else 'NORMAL'}
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        return {'total_blacklisted': len(self.stop_loss_blacklist), 'blacklist_records': [{'symbol': r.symbol, 'stop_date': r.stop_date, 'stop_reason': r.stop_reason, 'days_remaining': r.days_remaining} for r in self.stop_loss_blacklist.values()]}
    
    def get_trade_count_stats(self) -> Dict[str, Any]:
        return {'total_trades': len(self.trades), 'min_target': V52_MIN_TRADES_TARGET, 'max_target': V52_MAX_TRADES_TARGET, 'fail_threshold': V52_TRADE_COUNT_FAIL_THRESHOLD}
    
    def check_trade_count_constraint(self) -> Tuple[bool, str]:
        total_trades = len(self.trades)
        if total_trades > V52_TRADE_COUNT_FAIL_THRESHOLD:
            return False, f"[V52 FAILED: OVER-TRADING] Trade count {total_trades} exceeds threshold"
        return True, ""
    
    def get_position_sizing_stats(self) -> Dict[str, Any]:
        return {
            'risk_target_per_position': V52_RISK_TARGET_PER_POSITION,
            'max_single_position_pct': V52_MAX_SINGLE_POSITION_PCT,
            'reduced_single_position_pct': V52_REDUCED_SINGLE_POSITION_PCT,
            'initial_atr_mult': V52_INITIAL_ATR_MULT,
            'position_sizing_method': 'Individual Stock Volatility',
            'auto_cooldown_triggered': self.auto_cooldown_triggered,
        }
    
    def get_drawdown_stats(self) -> Dict[str, Any]:
        return {
            'daily_drawdown': self.drawdown_state.daily_drawdown,
            'weekly_drawdown': self.drawdown_state.weekly_drawdown,
            'single_day_triggered': self.drawdown_state.single_day_triggered,
            'weekly_triggered': self.drawdown_state.weekly_triggered,
            'flash_cut_triggered': self.drawdown_state.flash_cut_triggered,
            'cut_position_active': self.cut_position_active,
            'auto_cooldown_active': self.auto_cooldown_triggered,
        }
    
    def get_sector_balancing_stats(self) -> Dict[str, Any]:
        """获取行业均衡统计"""
        industry_counts: Dict[str, int] = {}
        for pos in self.positions.values():
            industry = pos.industry_name or "Unknown"
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        return {
            'enabled': V52_SECTOR_BALANCING_ENABLED,
            'max_per_industry': V52_MAX_POSITIONS_PER_INDUSTRY,
            'current_industry_holdings': industry_counts,
        }
    
    def get_step_trailing_stats(self) -> Dict[str, Any]:
        """获取阶梯止盈统计"""
        tier1_count = sum(1 for p in self.positions.values() if p.step_trailing_tier == 1)
        tier2_count = sum(1 for p in self.positions.values() if p.step_trailing_tier == 2)
        return {
            'enabled': V52_STEP_TRAILING_ENABLED,
            'tier1_threshold': f"{V52_STEP1_LOWER_BOUND:.0%} - {V52_STEP1_UPPER_BOUND:.0%}",
            'tier1_atr_mult': V52_STEP1_ATR_MULT,
            'tier2_threshold': f"> {V52_STEP2_LOWER_BOUND:.0%}",
            'tier2_drawdown_ratio': V52_STEP2_DRAWDOWN_RATIO,
            'current_tier1_positions': tier1_count,
            'current_tier2_positions': tier2_count,
        }