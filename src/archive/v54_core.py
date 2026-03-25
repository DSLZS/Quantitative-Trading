"""
V54 Core Module - 逻辑唤醒与三级止损执行

【V54 核心改进 - 三级防御体系】

1. 诊断与修复（最高优先级）
   ✅ 修复：handle_exits 在每根 K 线结束时调用
   ✅ 修复：止损判定在因子计算之后、交易执行之前

2. 止损止盈体系重构（三级防御）
   ✅ 硬核止损：入场即挂单，跌破 max(1.5 * ATR, 6%) 强制平仓
   ✅ 动态止盈（Trailing Profit）：浮盈≥8% 激活，回撤 2.0 * ATR 平仓
   ✅ 均线保护：MA120 进场过滤改为 MA60，增加"跌破 MA20"趋势离场

3. 因子权重微调
   ✅ 保持 Score = Rank(Momentum)*0.4 + Rank(R²)*0.6
   ✅ 新增：20 日成交量未萎缩过滤器，防止买入"僵尸股"

4. 自迭代控制
   ✅ 内置参数微调：若回撤 > 10%，自动调整 ATR 止盈参数
   ✅ 逻辑统计审计：硬止损/动态止盈/均线离场触发次数统计

作者：量化系统
版本：V54.0
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
# V54 配置常量 - 三级防御体系
# ===========================================

# 基础配置
V54_INITIAL_CAPITAL = 100000.00
V54_MAX_POSITIONS = 10

# V54 双重动量权重（保持 V53）
V54_MOMENTUM_WEIGHT = 0.4
V54_R2_WEIGHT = 0.6

# V54 进场过滤 - 改为 MA60
V54_ENTRY_TOP_N = 15              # 入场：Composite_Score 全市场前 15
V54_MAINTAIN_TOP_N = 40           # 离场：跌出全市场前 40 名
V54_MA20_EXIT_ENABLED = True      # 跌破 MA20 均线离场
V54_MA60_FILTER = True            # 股价在 60 日均线之上（V54 改为 MA60）

# V54 T/T+1 严格隔离配置
V54_USE_T1_EXECUTION = True

# ===========================================
# V54 三级防御体系配置
# ===========================================

# 1. 硬核止损 - 入场即挂单
V54_HARD_STOP_LOSS_ATR_MULT = 1.5    # 1.5 * ATR
V54_HARD_STOP_LOSS_RATIO = 0.06      # 6% 固定止损
V54_HARD_STOP_LOSS_MODE = "max"      # 取 max(1.5*ATR, 6%)

# 2. 动态止盈（Trailing Profit）
V54_TRAILING_PROFIT_TRIGGER = 0.08   # 浮盈≥8% 激活追踪止盈
V54_TRAILING_PROFIT_ATR_MULT = 2.0   # 回撤 2.0 * ATR 平仓
V54_TRAILING_PROFIT_ENABLED = True

# 3. 均线保护
V54_MA20_TREND_EXIT_ENABLED = True   # 跌破 MA20 作为趋势离场信号

# V54 初始止损（被硬核止损替代，但保留作为后备）
V54_INITIAL_STOP_LOSS_RATIO = 0.06   # 与硬核止损一致

# V54 波动率适配头寸管理
V54_RISK_TARGET_PER_POSITION = 0.008
V54_MAX_SINGLE_POSITION_PCT = 0.20
V54_REDUCED_SINGLE_POSITION_PCT = 0.10
V54_ATR_VOLATILITY_THRESHOLD = 0.05

# V54 洗售审计
V54_WASH_SALE_WINDOW = 5

# V54 趋势质量
V54_TREND_QUALITY_WINDOW = 20
V54_TREND_QUALITY_THRESHOLD = 0.55

# V54 成交量萎缩过滤器（新增）
V54_VOLUME_FILTER_ENABLED = True
V54_VOLUME_MA_PERIOD = 20
V54_VOLUME_SHRINK_THRESHOLD = 0.5    # 成交量萎缩至 50% 以下视为"僵尸股"

# V54 费率配置
V54_COMMISSION_RATE = 0.0003
V54_MIN_COMMISSION = 5.0
V54_SLIPPAGE_BUY = 0.001
V54_SLIPPAGE_SELL = 0.001
V54_STAMP_DUTY = 0.0005
V54_TRANSFER_FEE = 0.00001

# V54 数据库表配置
V54_DATABASE_TABLES = {
    'price_data': 'stock_daily',
    'index_data': 'index_daily',
}

# V54 交易次数约束
V54_MAX_TRADES_THRESHOLD = 60

# V54 性能目标 - 自迭代控制
V54_ANNUAL_RETURN_TARGET = 0.15
V54_MAX_DRAWDOWN_TARGET = 0.08       # 调整为 8%（若>10% 自动调整参数）
V54_PROFIT_LOSS_RATIO_TARGET = 3.0
V54_DRAWDOWN_AUTO_ADJUST_THRESHOLD = 0.10  # 回撤>10% 触发自动调整


# ===========================================
# V54 数据类
# ===========================================

@dataclass
class V54Position:
    """V54 持仓记录 - 三级防御体系"""
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
    
    # V54 三级防御体系
    hard_stop_price: float = 0.0          # 硬核止损价
    hard_stop_triggered: bool = False     # 硬核止损是否触发
    
    trailing_profit_active: bool = False  # 动态止盈是否激活
    trailing_profit_stop: float = 0.0     # 动态止盈价
    trailing_profit_triggered: bool = False  # 动态止盈是否触发
    
    ma20_exit_triggered: bool = False     # MA20 跌破离场是否触发
    
    # 历史追踪
    hard_stop_history: List[float] = field(default_factory=list)
    trailing_stop_history: List[float] = field(default_factory=list)
    peak_price_history: List[float] = field(default_factory=list)
    
    # 位次追踪
    current_market_rank: int = 999
    current_market_percentile: float = 1.0
    position_pct: float = 0.0
    entry_composite_score: float = 0.0
    
    # 均线数据
    ma5_at_entry: float = 0.0
    ma20_at_entry: float = 0.0
    ma60_at_entry: float = 0.0  # V54 新增
    ma120_at_entry: float = 0.0  # V54 新增
    
    # 行业与成交量
    industry_name: str = ""
    entry_volatility_ratio: float = 0.0
    position_tier: str = "standard"
    volume_shrunk_at_entry: bool = False  # V54 新增：成交量萎缩标记


@dataclass
class V54Trade:
    """V54 交易记录"""
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
class V54TradeAudit:
    """V54 交易审计记录"""
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
    
    # V54 三级防御审计
    entry_signal: float = 0.0
    signal_rank: int = 0
    atr_at_entry: float = 0.0
    hard_stop_price: float = 0.0
    hard_stop_triggered: bool = False
    trailing_profit_active: bool = False
    trailing_profit_triggered: bool = False
    ma20_exit_triggered: bool = False
    peak_price: float = 0.0
    exit_profit_ratio: float = 0.0
    position_tier: str = "standard"
    volume_shrunk_at_entry: bool = False


@dataclass
class V54WashSaleRecord:
    """V54 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V54BlacklistRecord:
    """V54 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V54MarketRegime:
    """V54 大盘状态"""
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
class V54DrawdownState:
    """V54 回撤状态"""
    trade_date: str
    daily_drawdown: float = 0.0
    weekly_drawdown: float = 0.0
    single_day_triggered: bool = False
    weekly_triggered: bool = False


# ===========================================
# V54 行业加载器
# ===========================================

class V54IndustryLoader:
    """V54 行业加载器 - 数据库适配"""
    
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
# V54 因子引擎 - 包含成交量萎缩过滤器
# ===========================================

class V54FactorEngine:
    """V54 因子引擎 - 双重动量确认与成交量萎缩过滤"""
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None,
                 momentum_weight: float = V54_MOMENTUM_WEIGHT,
                 r2_weight: float = V54_R2_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
        self.industry_loader = V54IndustryLoader()
    
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
                'database_tables_used': V54_DATABASE_TABLES.copy(),
                'momentum_weight': self.momentum_weight, 'r2_weight': self.r2_weight,
                'industry_data_source': 'database',
                'volume_filter_enabled': V54_VOLUME_FILTER_ENABLED,
            }
            
            result = self._compute_atr(result, period=20)
            status['factors_computed'].append('atr_20')
            
            result = self._compute_rsrs_factor(result)
            status['factors_computed'].append('rsrs_factor')
            
            result = self._compute_trend_factors(result)
            status['factors_computed'].extend(['trend_strength_20', 'trend_strength_60'])
            
            result = self._compute_volatility_adjusted_momentum(result)
            status['factors_computed'].append('volatility_adjusted_momentum')
            
            result = self._compute_trend_quality_v54(result)
            status['factors_computed'].append('trend_quality_r2')
            
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20', 'ma60', 'ma120'])
            
            # V54 新增：成交量萎缩过滤器
            result = self._compute_volume_shrink_filter(result)
            status['factors_computed'].append('volume_shrink_filter')
            
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # V54 趋势过滤：股价在 60 日均线之上（改为 MA60）
            if V54_MA60_FILTER:
                result = self._apply_trend_filter(result)
                status['factors_computed'].append('trend_filter_pass')
            
            result = self._compute_composite_score_v54(result)
            return result, status
            
        except Exception as e:
            logger.error(f"V54 compute_all_factors FAILED: {e}")
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
    
    def _compute_trend_quality_v54(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        window = V54_TREND_QUALITY_WINDOW
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
        ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
        ma120 = pl.col('close').rolling_mean(window_size=120).over('symbol')
        ma_trend_bullish = ma5 > ma20
        price_above_ma20 = pl.col('close') > ma20
        price_above_ma60 = pl.col('close') > ma60
        price_above_ma120 = pl.col('close') > ma120
        return result.with_columns([
            ma5.alias('ma5'), ma20.alias('ma20'), ma60.alias('ma60'), ma120.alias('ma120'),
            ma_trend_bullish.alias('ma_trend_bullish'),
            price_above_ma20.alias('price_above_ma20'),
            price_above_ma60.alias('price_above_ma60'),
            price_above_ma120.alias('price_above_ma120')
        ])
    
    def _compute_volume_shrink_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V54 新增：成交量萎缩过滤器
        
        逻辑：若 20 日成交量均值低于过去 5 日均量的 50%，视为"僵尸股"
        防止买入流动性枯竭的股票
        """
        result = df.clone()
        
        vol_ma20 = pl.col('volume').rolling_mean(window_size=V54_VOLUME_MA_PERIOD).over('symbol')
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        
        # 成交量比率：近期成交量 / 长期成交量
        volume_ratio = vol_ma5 / (vol_ma20 + self.EPSILON)
        
        # 成交量萎缩标记：比率 < 阈值
        is_volume_shrunk = volume_ratio < V54_VOLUME_SHRINK_THRESHOLD
        
        # 成交量过滤器：通过 = 未萎缩
        volume_filter_pass = ~is_volume_shrunk
        
        return result.with_columns([
            vol_ma20.alias('vol_ma20'),
            vol_ma5.alias('vol_ma5'),
            volume_ratio.alias('volume_ratio'),
            is_volume_shrunk.alias('is_volume_shrunk'),
            volume_filter_pass.alias('volume_filter_pass')
        ])
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        returns = pl.col('close').pct_change().over('symbol')
        stock_vol = returns.rolling_std(window_size=20, ddof=1).over('symbol')
        market_vol = stock_vol
        market_vol_mean = market_vol.rolling_mean(window_size=20).over('symbol')
        vol_ratio = market_vol / (market_vol_mean + self.EPSILON)
        vix_sim = market_vol * 100
        return result.with_columns([
            returns.alias('returns'), stock_vol.alias('stock_volatility'),
            market_vol.alias('market_volatility'), market_vol_mean.alias('market_volatility_mean'),
            vol_ratio.alias('volatility_ratio'), vix_sim.alias('vix_sim')
        ])
    
    def _apply_trend_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """V54 趋势过滤：股价在 60 日均线之上（改为 MA60）"""
        result = df.clone()
        trend_filter_pass = pl.col('price_above_ma60')
        return result.with_columns([trend_filter_pass.alias('trend_filter_pass')])
    
    def _compute_composite_score_v54(self, df: pl.DataFrame) -> pl.DataFrame:
        """V54 综合评分计算 - 包含成交量萎缩过滤"""
        try:
            result = df.clone()
            
            # 确保必要的列存在且为 Float64
            result = result.with_columns([
                pl.col('volatility_adjusted_momentum').cast(pl.Float64, strict=False).fill_null(0.0).alias('volatility_adjusted_momentum'),
                pl.col('trend_quality_r2').cast(pl.Float64, strict=False).fill_null(0.0).alias('trend_quality_r2'),
            ])
            
            # V54 成交量萎缩过滤：若成交量萎缩，则降低动量得分
            # 成交量萎缩因子 = volume_ratio (近期成交量/长期成交量)
            # 若 volume_ratio < 0.5，视为僵尸股，动量得分打折
            volume_factor = pl.when(pl.col('volume_ratio') < V54_VOLUME_SHRINK_THRESHOLD) \
                .then(pl.lit(0.5)) \
                .otherwise(pl.lit(1.0))
            
            # 计算动量调整因子
            momentum_adjusted = pl.col('volatility_adjusted_momentum') * volume_factor
            
            # 按 trade_date 分组排名
            momentum_rank_raw = momentum_adjusted.rank('ordinal', descending=True).over('trade_date')
            r2_rank_raw = pl.col('trend_quality_r2').rank('ordinal', descending=True).over('trade_date')
            
            # 计算每个交易日股票数量
            n_stocks_per_date = pl.col('symbol').count().over('trade_date')
            
            # 归一化排名
            momentum_rank_norm = momentum_rank_raw / n_stocks_per_date
            r2_rank_norm = r2_rank_raw / n_stocks_per_date
            
            # V54 保持 Score = Rank(Momentum)*0.4 + Rank(R²)*0.6
            composite_score_expr = (1.0 - momentum_rank_norm) * self.momentum_weight + (1.0 - r2_rank_norm) * self.r2_weight
            result = result.with_columns([composite_score_expr.alias('composite_score')])
            
            # 计算综合排名和百分位
            composite_rank = pl.col('composite_score').rank('ordinal', descending=True).over('trade_date')
            composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / n_stocks_per_date.cast(pl.Float64))
            
            # 确保 composite_percentile 不为 null
            composite_percentile = composite_percentile.fill_null(0.0)
            
            # V54 进场过滤：趋势共振 + 排名过滤 + 成交量萎缩过滤
            top_n_filter = composite_rank <= V54_ENTRY_TOP_N
            trend_filter = pl.col('trend_filter_pass') if 'trend_filter_pass' in result.columns else pl.lit(True)
            volume_filter = pl.col('volume_filter_pass') if 'volume_filter_pass' in result.columns else pl.lit(True)
            entry_allowed = top_n_filter & trend_filter & volume_filter
            
            return result.with_columns([
                momentum_rank_raw.cast(pl.Int64).fill_null(9999).alias('momentum_rank_raw'),
                r2_rank_raw.cast(pl.Int64).fill_null(9999).alias('r2_rank_raw'),
                composite_rank.cast(pl.Int64).fill_null(9999).alias('composite_rank'),
                composite_percentile.alias('composite_percentile'),
                top_n_filter.alias('top_n_filter_pass'),
                trend_filter.alias('trend_filter_pass'),
                volume_filter.alias('volume_filter_pass'),
                entry_allowed.alias('entry_allowed')
            ])
        except Exception as e:
            logger.error(f"Error in _compute_composite_score_v54: {e}")
            logger.error(traceback.format_exc())
            return df.with_columns([
                pl.lit(0.0).alias('composite_score'),
                pl.lit(9999).alias('composite_rank'),
                pl.lit(0.0).alias('composite_percentile'),
                pl.lit(False).alias('top_n_filter_pass'),
                pl.lit(False).alias('trend_filter_pass'),
                pl.lit(False).alias('volume_filter_pass'),
                pl.lit(False).alias('entry_allowed')
            ])
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()
    
    def update_weights(self, momentum_weight: float, r2_weight: float):
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight


# ===========================================
# V54 风险管理器 - 三级防御体系
# ===========================================

class V54RiskManager:
    """V54 风险管理器 - 三级防御体系"""
    
    def __init__(self, initial_capital: float = V54_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, V54Position] = {}
        self.trades: List[V54Trade] = []
        self.trade_log: List[V54TradeAudit] = []
        self.sell_history: Dict[str, str] = {}
        self.wash_sale_blocks: List[V54WashSaleRecord] = []
        self.stop_loss_blacklist: Dict[str, V54BlacklistRecord] = {}
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        self.trade_day_counter: int = 0
        self.current_market_vol: float = 1.0
        self.is_low_volatility_environment: bool = False
        self.is_high_volatility_environment: bool = False
        self.market_regime: V54MarketRegime = V54MarketRegime(trade_date="")
        self.is_risk_period: bool = False
        self.drawdown_state: V54DrawdownState = V54DrawdownState(trade_date="")
        self.factor_status: Dict[str, Any] = {}
        self.current_rank_cache: Dict[str, int] = {}
        self.peak_portfolio_value: float = initial_capital
        self.daily_start_value: float = initial_capital
        self.weekly_start_value: float = initial_capital
        self.last_week: int = 0
        self.current_position_limit: float = V54_MAX_SINGLE_POSITION_PCT
        self.industry_loader = V54IndustryLoader()
        self.portfolio_values: List[Dict[str, Any]] = []
        
        # V54 三级防御统计
        self.hard_stop_count: int = 0
        self.trailing_profit_count: int = 0
        self.ma20_exit_count: int = 0
        self.rank_drop_count: int = 0
        self.initial_stop_count: int = 0
        
        # V54 自迭代控制
        self.auto_adjusted_params: bool = False
        self.current_trailing_atr_mult: float = V54_TRAILING_PROFIT_ATR_MULT
    
    def reset_daily_counters(self, trade_date: str):
        """重置每日计数器"""
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
    
    def _get_week_number(self, date_str: str) -> int:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.isocalendar()[1]
        except:
            return 0
    
    def _update_blacklist(self, trade_date: str):
        """更新黑名单"""
        for symbol in list(self.stop_loss_blacklist.keys()):
            record = self.stop_loss_blacklist[symbol]
            if self.trade_day_counter >= record.blacklist_expiry_day:
                del self.stop_loss_blacklist[symbol]
            else:
                record.days_remaining = record.blacklist_expiry_day - self.trade_day_counter
    
    def _update_drawdown_state(self, trade_date: str):
        """更新回撤状态"""
        current_value = self.get_total_portfolio_value(trade_date)
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        daily_drawdown = (self.daily_start_value - current_value) / self.daily_start_value if self.daily_start_value > 0 else 0
        weekly_drawdown = (self.weekly_start_value - current_value) / self.weekly_start_value if self.weekly_start_value > 0 else 0
        single_day_triggered = daily_drawdown > 0.015
        weekly_triggered = weekly_drawdown > 0.03
        
        self.drawdown_state = V54DrawdownState(
            trade_date=trade_date,
            daily_drawdown=daily_drawdown,
            weekly_drawdown=weekly_drawdown,
            single_day_triggered=single_day_triggered,
            weekly_triggered=weekly_triggered
        )
        
        self.daily_start_value = current_value
        
        self.portfolio_values.append({
            'trade_date': trade_date,
            'total_value': current_value
        })
    
    def _calculate_max_drawdown(self) -> float:
        """计算历史最大回撤"""
        if len(self.portfolio_values) < 2:
            return 0.0
        max_dd = 0.0
        peak = self.portfolio_values[0]['total_value']
        for pv in self.portfolio_values:
            v = pv['total_value']
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
        """检查洗售交易"""
        if symbol in self.today_sells:
            return True, f"same_day_sell ({trade_date})"
        if symbol in self.sell_history:
            last_sell_date = self.sell_history[symbol]
            days_diff = self._days_between(last_sell_date, trade_date)
            if days_diff <= V54_WASH_SALE_WINDOW:
                return True, f"wash_sale_window ({days_diff} days)"
        return False, None
    
    def check_blacklist(self, symbol: str, trade_date: str) -> Tuple[bool, Optional[str]]:
        """检查黑名单"""
        if symbol in self.stop_loss_blacklist:
            record = self.stop_loss_blacklist[symbol]
            return True, f"blacklist ({record.stop_reason}) - {record.days_remaining} days"
        return False, None
    
    def record_sell(self, symbol: str, trade_date: str):
        """记录卖出"""
        self.sell_history[symbol] = trade_date
        self.today_sells.add(symbol)
    
    def record_buy(self, symbol: str):
        """记录买入"""
        self.today_buys.add(symbol)
    
    def add_to_blacklist(self, symbol: str, trade_date: str, stop_reason: str):
        """添加到黑名单"""
        self.stop_loss_blacklist[symbol] = V54BlacklistRecord(
            symbol=symbol, stop_date=trade_date, stop_reason=stop_reason,
            blacklist_expiry_day=self.trade_day_counter + 5, days_remaining=5
        )
        logger.info(f"BLACKLIST ADDED: {symbol} - {stop_reason}")
    
    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        return max(V54_MIN_COMMISSION, amount * V54_COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        """计算过户费"""
        return shares * price * V54_TRANSFER_FEE
    
    def get_current_risk_target(self) -> float:
        """获取当前风险目标"""
        return V54_RISK_TARGET_PER_POSITION
    
    def can_open_new_position(self) -> bool:
        """是否可以开新仓"""
        if self.is_risk_period:
            return False
        return True
    
    def get_position_limit(self) -> float:
        """获取头寸限制"""
        return self.current_position_limit
    
    def calculate_position_size(self, symbol: str, atr: float, current_price: float, 
                                 total_assets: float, volatility_ratio: float = None) -> Tuple[int, float, str]:
        """V54 动态头寸计算 - 波动率适配仓位"""
        try:
            risk_target = self.get_current_risk_target()
            risk_amount = total_assets * risk_target
            atr_mult = V54_HARD_STOP_LOSS_ATR_MULT
            risk_per_share = atr * atr_mult
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0, "standard"
            
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            
            if shares < 100:
                return 0, 0.0, "standard"
            
            position_amount = shares * current_price
            
            volatility_ratio = atr / current_price if current_price > 0 else 0
            
            position_tier = "standard"
            if volatility_ratio > V54_ATR_VOLATILITY_THRESHOLD:
                position_tier = "reduced"
                max_position = total_assets * V54_REDUCED_SINGLE_POSITION_PCT
            else:
                max_position = total_assets * V54_MAX_SINGLE_POSITION_PCT
            
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
            
            return shares, shares * current_price, position_tier
            
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0, "standard"
    
    def _calculate_hard_stop_price(self, entry_price: float, atr: float) -> float:
        """
        V54 计算硬核止损价
        
        逻辑：max(1.5 * ATR, 6%) 强制平仓
        """
        # 1.5 * ATR 止损
        atr_stop_distance = atr / entry_price if entry_price > 0 else 0
        atr_stop_price = entry_price * (1 - V54_HARD_STOP_LOSS_ATR_MULT * atr_stop_distance)
        
        # 6% 固定止损
        fixed_stop_price = entry_price * (1 - V54_HARD_STOP_LOSS_RATIO)
        
        # 取 max（更宽松的那个，但都提供保护）
        if V54_HARD_STOP_LOSS_MODE == "max":
            # max 意味着取两者中较高的止损价（更严格的保护）
            stop_price = max(atr_stop_price, fixed_stop_price)
        else:
            stop_price = atr_stop_price
        
        return stop_price
    
    def execute_buy(self, trade_date: str, symbol: str, open_price: float, atr: float, 
                    target_amount: float, signal_date: str, signal_score: float = 0.0, 
                    signal_rank: int = 0, composite_score: float = 0.0, 
                    composite_percentile: float = 0.0, ma5: float = 0.0, 
                    ma20: float = 0.0, ma60: float = 0.0, ma120: float = 0.0, 
                    industry_name: str = "", volatility_ratio: float = 0.0, 
                    volume_shrunk: bool = False, reason: str = "") -> Optional[V54Trade]:
        """
        V54 执行买入 - 包含三级防御体系初始化
        """
        try:
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
            
            # V54 趋势过滤：检查股价是否在 60 日均线之上
            if V54_MA60_FILTER:
                if ma60 > 0 and open_price <= ma60:
                    logger.warning(f"TREND FILTER BLOCKED: {symbol} - Price {open_price:.2f} <= MA60 {ma60:.2f}")
                    return None
            
            execution_price = open_price * (1 + V54_SLIPPAGE_BUY)
            raw_shares = int(target_amount / execution_price)
            shares = (raw_shares // 100) * 100
            
            if shares < 100:
                return None
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * V54_SLIPPAGE_BUY
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            total_cost = actual_amount + commission + slippage + transfer_fee
            
            if self.cash < total_cost:
                return None
            
            self.cash -= total_cost
            
            # V54 核心：硬核止损 - 入场即挂单
            hard_stop_price = self._calculate_hard_stop_price(execution_price, atr)
            
            # 波动率适配头寸
            volatility_ratio = atr / execution_price if execution_price > 0 else 0
            position_tier = "reduced" if volatility_ratio > V54_ATR_VOLATILITY_THRESHOLD else "standard"
            position_pct = actual_amount / (self.cash + actual_amount) if (self.cash + actual_amount) > 0 else 0
            
            self.positions[symbol] = V54Position(
                symbol=symbol, shares=shares, 
                avg_cost=(actual_amount + commission + slippage + transfer_fee) / shares,
                buy_price=execution_price, buy_date=trade_date, signal_date=signal_date, 
                trade_date=trade_date, signal_score=signal_score, signal_rank=signal_rank, 
                composite_score=composite_score, current_price=execution_price, 
                holding_days=0, peak_price=execution_price, peak_profit=0.0,
                buy_trade_day=self.trade_day_counter, atr_at_entry=atr, 
                hard_stop_price=hard_stop_price, hard_stop_triggered=False,
                trailing_profit_active=False, trailing_profit_stop=hard_stop_price,
                trailing_profit_triggered=False, ma20_exit_triggered=False,
                hard_stop_history=[hard_stop_price], trailing_stop_history=[hard_stop_price],
                peak_price_history=[execution_price],
                current_market_rank=signal_rank, current_market_percentile=composite_percentile,
                position_pct=position_pct, entry_composite_score=composite_score,
                ma5_at_entry=ma5, ma20_at_entry=ma20, ma60_at_entry=ma60, ma120_at_entry=ma120,
                volume_shrunk_at_entry=volume_shrunk, industry_name=industry_name,
                entry_volatility_ratio=volatility_ratio, position_tier=position_tier
            )
            
            self.record_buy(symbol)
            
            trade = V54Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares, 
                price=open_price, amount=actual_amount, commission=commission, 
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee, 
                total_cost=total_cost, reason=reason, execution_price=execution_price, 
                signal_date=signal_date, t_plus_1=True
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | HardStop={hard_stop_price:.2f} | Rank={signal_rank} | Tier={position_tier}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(self, trade_date: str, symbol: str, open_price: float, 
                     shares: Optional[int] = None, reason: str = "", 
                     force: bool = False) -> Optional[V54Trade]:
        """V54 执行卖出"""
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
            
            execution_price = open_price * (1 - V54_SLIPPAGE_SELL)
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * V54_SLIPPAGE_SELL
            stamp_duty = actual_amount * V54_STAMP_DUTY
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
            
            # 卖出原因加入黑名单
            if reason in ["trailing_stop", "stop_loss", "ma20_break", "hwm_stop", "hard_stop", "trailing_profit"]:
                self.add_to_blacklist(symbol, trade_date, reason)
            
            trade_audit = V54TradeAudit(
                symbol=symbol, buy_date=pos.buy_date, sell_date=trade_date,
                buy_price=pos.buy_price, sell_price=execution_price, shares=shares,
                gross_pnl=realized_pnl + (commission + slippage + stamp_duty + transfer_fee),
                total_fees=commission + slippage + stamp_duty + transfer_fee,
                net_pnl=realized_pnl, holding_days=calculated_holding_days,
                is_profitable=realized_pnl > 0, sell_reason=reason,
                entry_signal=pos.signal_score, signal_rank=pos.signal_rank,
                atr_at_entry=pos.atr_at_entry, hard_stop_price=pos.hard_stop_price,
                hard_stop_triggered=pos.hard_stop_triggered,
                trailing_profit_active=pos.trailing_profit_active,
                trailing_profit_triggered=pos.trailing_profit_triggered,
                ma20_exit_triggered=pos.ma20_exit_triggered,
                peak_price=pos.peak_price, exit_profit_ratio=pos.current_profit_ratio,
                position_tier=pos.position_tier,
                volume_shrunk_at_entry=pos.volume_shrunk_at_entry
            )
            self.trade_log.append(trade_audit)
            
            del self.positions[symbol]
            self.record_sell(symbol, trade_date)
            
            trade = V54Trade(
                trade_date=trade_date, symbol=symbol, side="SELL", shares=shares, 
                price=open_price, amount=actual_amount, commission=commission, 
                slippage=slippage, stamp_duty=stamp_duty, transfer_fee=transfer_fee, 
                total_cost=net_proceeds, reason=reason, holding_days=pos.holding_days, 
                execution_price=execution_price, signal_date=pos.signal_date, t_plus_1=True
            )
            self.trades.append(trade)
            
            logger.info(f"  SELL {symbol} | {shares} shares @ {execution_price:.2f} | PnL: {realized_pnl:.2f} | Reason: {reason}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_sell failed: {e}")
            return None
    
    def update_volatility_regime(self, market_vol: float):
        """更新波动率状态"""
        self.current_market_vol = market_vol
        self.is_low_volatility_environment = market_vol < 0.80
        self.is_high_volatility_environment = market_vol > 1.20
    
    def get_risk_per_position(self) -> float:
        """获取每仓风险"""
        return self.get_current_risk_target()
    
    def check_exits(self, positions: Dict[str, V54Position], date_str: str, 
                    price_df: pl.DataFrame, factor_df: pl.DataFrame) -> List[Tuple[str, str]]:
        """
        V54 检查退出信号 - 三级防御体系
        
        离场条件（优先级从高到低）：
        1. 硬核止损：跌破 max(1.5 * ATR, 6%) 强制平仓
        2. 动态止盈：浮盈≥8% 激活，回撤 2.0 * ATR 平仓
        3. 均线保护：跌破 MA20 趋势离场
        4. 位次缓冲：跌出全市场前 40 名
        5. 初始止损：后备止损
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
            
            # 更新最高价
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                pos.peak_price_history.append(current_price)
            
            pos.current_profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            
            # ===========================================
            # V54 三级防御体系 - 按优先级检查
            # ===========================================
            
            # 1. 硬核止损 - 入场即挂单，跌破 max(1.5 * ATR, 6%) 强制平仓
            if current_price <= pos.hard_stop_price and not pos.hard_stop_triggered:
                pos.hard_stop_triggered = True
                sell_list.append((symbol, "hard_stop"))
                self.hard_stop_count += 1
                logger.info(f"  HARD STOP: {symbol} - Price {current_price:.2f} <= HardStop {pos.hard_stop_price:.2f}")
                continue
            
            # 2. 动态止盈（Trailing Profit）- 浮盈≥8% 激活，回撤 2.0 * ATR 平仓
            if V54_TRAILING_PROFIT_ENABLED:
                # 检查是否应该激活动态止盈
                if not pos.trailing_profit_active and pos.current_profit_ratio >= V54_TRAILING_PROFIT_TRIGGER:
                    pos.trailing_profit_active = True
                    # 初始动态止盈价设为当前价（最高点）
                    pos.trailing_profit_stop = current_price
                    logger.info(f"  TRAILING PROFIT ACTIVATED: {symbol} - Profit {pos.current_profit_ratio:.2%} >= {V54_TRAILING_PROFIT_TRIGGER:.0%}")
                
                # 如果已激活，检查是否触发
                if pos.trailing_profit_active:
                    atr = atr_values.get(symbol, pos.atr_at_entry)
                    if atr and atr > 0:
                        # 更新最高点
                        if current_price > pos.peak_price:
                            # 更新动态止盈价：最高点 - 2.0 * ATR
                            new_trailing_stop = current_price - self.current_trailing_atr_mult * atr
                            if new_trailing_stop > pos.trailing_profit_stop:
                                pos.trailing_profit_stop = new_trailing_stop
                                pos.trailing_stop_history.append(new_trailing_stop)
                        
                        # 检查是否触发
                        if current_price <= pos.trailing_profit_stop:
                            pos.trailing_profit_triggered = True
                            sell_list.append((symbol, "trailing_profit"))
                            self.trailing_profit_count += 1
                            logger.info(f"  TRAILING PROFIT: {symbol} - Price {current_price:.2f} <= TrailStop {pos.trailing_profit_stop:.2f}")
                            continue
            
            # 3. 均线保护 - 跌破 MA20 作为趋势离场信号
            if V54_MA20_TREND_EXIT_ENABLED:
                ma20 = ma20_values.get(symbol, pos.ma20_at_entry)
                if ma20 is None:
                    ma20 = pos.ma20_at_entry if pos.ma20_at_entry else 0
                if ma20 and ma20 > 0 and current_price < ma20:
                    pos.ma20_exit_triggered = True
                    sell_list.append((symbol, "ma20_break"))
                    self.ma20_exit_count += 1
                    logger.info(f"  MA20 BREAK: {symbol} - Price {current_price:.2f} < MA20 {ma20:.2f}")
                    continue
            
            # 4. 位次缓冲 - 跌出全市场前 40 名离场
            if V54_MAINTAIN_TOP_N > 0:
                current_rank = ranks.get(symbol, 9999)
                if current_rank > V54_MAINTAIN_TOP_N:
                    sell_list.append((symbol, "rank_drop"))
                    self.rank_drop_count += 1
                    logger.info(f"  RANK DROP: {symbol} - Rank {current_rank} > {V54_MAINTAIN_TOP_N}")
                    continue
            
            # 5. 初始止损（后备）
            if pos.current_profit_ratio <= -V54_INITIAL_STOP_LOSS_RATIO:
                sell_list.append((symbol, "stop_loss"))
                self.initial_stop_count += 1
                continue
        
        return sell_list
    
    def get_total_portfolio_value(self, date_str: str) -> float:
        """获取组合总价值"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_portfolio_value(self, positions: Dict[str, V54Position], date_str: str, 
                            price_df: pl.DataFrame) -> float:
        """获取组合价值"""
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
        """获取洗售统计"""
        return {
            'total_blocked': len(self.wash_sale_blocks),
            'blocked_records': [{
                'symbol': r.symbol, 'sell_date': r.sell_date,
                'blocked_buy_date': r.blocked_buy_date,
                'days_between': r.days_between, 'reason': r.reason
            } for r in self.wash_sale_blocks]
        }
    
    def get_market_regime_stats(self) -> Dict[str, Any]:
        """获取市场状态统计"""
        return {
            'is_risk_period': self.is_risk_period,
            'current_regime': 'RISK' if self.is_risk_period else 'NORMAL'
        }
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        """获取黑名单统计"""
        return {
            'total_blacklisted': len(self.stop_loss_blacklist),
            'blacklist_records': [{
                'symbol': r.symbol, 'stop_date': r.stop_date,
                'stop_reason': r.stop_reason, 'days_remaining': r.days_remaining
            } for r in self.stop_loss_blacklist.values()]
        }
    
    def get_trade_count_stats(self) -> Dict[str, Any]:
        """获取交易次数统计"""
        return {
            'total_trades': len(self.trades),
            'max_threshold': V54_MAX_TRADES_THRESHOLD
        }
    
    def check_trade_count_constraint(self) -> Tuple[bool, str]:
        """检查交易次数约束 - 严正审计"""
        total_trades = len(self.trades)
        if total_trades > V54_MAX_TRADES_THRESHOLD:
            return False, f"[V54 FAILED: OVER-TRADING] Trade count {total_trades} exceeds threshold {V54_MAX_TRADES_THRESHOLD}"
        return True, ""
    
    def get_position_sizing_stats(self) -> Dict[str, Any]:
        """获取头寸计算统计"""
        return {
            'risk_target_per_position': V54_RISK_TARGET_PER_POSITION,
            'max_single_position_pct': V54_MAX_SINGLE_POSITION_PCT,
            'reduced_single_position_pct': V54_REDUCED_SINGLE_POSITION_PCT,
            'atr_volatility_threshold': V54_ATR_VOLATILITY_THRESHOLD,
            'position_sizing_method': 'Volatility-Adaptive',
        }
    
    def get_drawdown_stats(self) -> Dict[str, Any]:
        """获取回撤统计"""
        return {
            'daily_drawdown': self.drawdown_state.daily_drawdown,
            'weekly_drawdown': self.drawdown_state.weekly_drawdown,
            'single_day_triggered': self.drawdown_state.single_day_triggered,
            'weekly_triggered': self.drawdown_state.weekly_triggered,
        }
    
    def get_trend_filter_stats(self) -> Dict[str, Any]:
        """获取趋势过滤统计"""
        return {
            'enabled': V54_MA60_FILTER,
            'ma_filter': 'MA60',  # V54 改为 MA60
            'description': 'Price above MA60 filter for trend confirmation'
        }
    
    def get_position_buffer_stats(self) -> Dict[str, Any]:
        """获取位次缓冲统计"""
        return {
            'entry_top_n': V54_ENTRY_TOP_N,
            'maintain_top_n': V54_MAINTAIN_TOP_N,
            'ma20_exit_enabled': V54_MA20_EXIT_ENABLED,
            'description': 'Enter at Top 15, exit when drop below Top 40 or break MA20'
        }
    
    def get_three_level_defense_stats(self) -> Dict[str, Any]:
        """V54 三级防御体系统计"""
        return {
            'hard_stop_count': self.hard_stop_count,
            'trailing_profit_count': self.trailing_profit_count,
            'ma20_exit_count': self.ma20_exit_count,
            'rank_drop_count': self.rank_drop_count,
            'initial_stop_count': self.initial_stop_count,
            'total_exit_count': self.hard_stop_count + self.trailing_profit_count + self.ma20_exit_count + self.rank_drop_count + self.initial_stop_count,
            'defense_config': {
                'hard_stop_atr_mult': V54_HARD_STOP_LOSS_ATR_MULT,
                'hard_stop_ratio': V54_HARD_STOP_LOSS_RATIO,
                'trailing_profit_trigger': V54_TRAILING_PROFIT_TRIGGER,
                'trailing_profit_atr_mult': self.current_trailing_atr_mult,
                'ma20_exit_enabled': V54_MA20_TREND_EXIT_ENABLED,
            },
            'auto_adjusted_params': self.auto_adjusted_params,
        }
    
    def adjust_trailing_params_if_needed(self, max_drawdown: float):
        """
        V54 自迭代控制：若回撤 > 10%，自动调整 ATR 止盈参数
        """
        if max_drawdown > V54_DRAWDOWN_AUTO_ADJUST_THRESHOLD and not self.auto_adjusted_params:
            # 收紧动态止盈参数
            self.current_trailing_atr_mult = 1.5  # 从 2.0 降至 1.5
            self.auto_adjusted_params = True
            logger.warning(f"V54 AUTO-ADJUST: Drawdown {max_drawdown:.2%} > {V54_DRAWDOWN_AUTO_ADJUST_THRESHOLD:.0%}, trailing ATR mult reduced to {self.current_trailing_atr_mult}")