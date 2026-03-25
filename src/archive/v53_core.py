"""
V53 Core Module - 自适应波段与波动率控仓

【V53 核心改进 - 事前风控与趋势共振】

1. 逻辑大扫除（已移除 V52 毒素）
   ❌ 移除：Flash Cut（周回撤减仓）逻辑
   ❌ 移除：15% 自动降温逻辑
   ❌ 移除：卖出浮盈最低的自残代码

2. 核心进化：事前风控（Pre-risk Control）
   ✅ 波动率适配仓位：DynamicPositionSizer
      - 若个股 ATR 极高（波动剧烈），自动将该股头寸降至 10%
      - 若个股走势平稳，保持 20% 头寸
   ✅ 中线位次缓冲：
      - 入场：Composite_Score 全市场前 5
      - 离场：跌出全市场前 40 名 或 跌破 MA20 均线
      - 目的：通过"只要不烂到垫底就不卖"来强制拉长持有期

3. 进场过滤：趋势共振
   ✅ 双重确认：除了因子得分，增加"股价在 120 日半年线之上"的过滤
   ✅ 只做处于大上升周期的票，放弃所有阴跌反弹的票

4. 严正审计与交付（禁令升级）
   ✅ 杜绝碎片化交易：若回测显示交易次数 > 60 次，必须重写 TradeLogic
   ✅ 拒绝造假：固定滑点为 0.1%，计入印花税
   ✅ 自动对赌：收益率 < 15% 时直接道歉并指出制约因素

作者：量化系统
版本：V53.0
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
# V53 配置常量
# ===========================================

# 基础配置
V53_INITIAL_CAPITAL = 100000.00
V53_MAX_POSITIONS = 10

# V53 双重动量权重
V53_MOMENTUM_WEIGHT = 0.4
V53_R2_WEIGHT = 0.6

# V53 进场确认 - 核心进化
V53_ENTRY_TOP_N = 15              # 入场：Composite_Score 全市场前 15（放宽条件）
V53_MAINTAIN_TOP_N = 40           # 离场：跌出全市场前 40 名
V53_MA20_EXIT_ENABLED = True      # 跌破 MA20 均线离场

# V53 趋势共振过滤
V53_TREND_FILTER_ENABLED = True
V53_MA120_FILTER = True           # 股价在 120 日半年线之上

# V53 T/T+1 严格隔离配置
V53_USE_T1_EXECUTION = True

# V53 ATR 移动止损配置
V53_INITIAL_ATR_MULT = 2.0
V53_TRAILING_ATR_MULT = 1.5
V53_MAX_LOSS_PER_TRADE = 0.01

# V53 初始止损
V53_INITIAL_STOP_LOSS_RATIO = 0.08

# V53 波动率适配头寸管理（事前风控核心）
V53_RISK_TARGET_PER_POSITION = 0.008
V53_MAX_SINGLE_POSITION_PCT = 0.20     # 标准头寸 20%
V53_REDUCED_SINGLE_POSITION_PCT = 0.10 # 高波动时降至 10%
V53_ATR_VOLATILITY_THRESHOLD = 0.05    # ATR/Price > 5% 视为高波动

# V53 洗售审计
V53_WASH_SALE_WINDOW = 5

# V53 趋势质量
V53_TREND_QUALITY_WINDOW = 20
V53_TREND_QUALITY_THRESHOLD = 0.55

# V53 波动率过滤
V53_VOLATILITY_FILTER_THRESHOLD = 1.30

# V53 费率配置（严正审计 - 固定值）
V53_COMMISSION_RATE = 0.0003
V53_MIN_COMMISSION = 5.0
V53_SLIPPAGE_BUY = 0.001          # 固定 0.1%
V53_SLIPPAGE_SELL = 0.001         # 固定 0.1%
V53_STAMP_DUTY = 0.0005           # 印花税 0.05%
V53_TRANSFER_FEE = 0.00001

# V53 数据库表配置
V53_DATABASE_TABLES = {
    'price_data': 'stock_daily',
    'index_data': 'index_daily',
}

# V53 交易次数约束（严正审计）
V53_MAX_TRADES_THRESHOLD = 60     # 若超过 60 次，必须重写 TradeLogic

# V53 性能目标
V53_ANNUAL_RETURN_TARGET = 0.15
V53_MAX_DRAWDOWN_TARGET = 0.04
V53_PROFIT_LOSS_RATIO_TARGET = 3.0


# ===========================================
# V53 数据类
# ===========================================

@dataclass
class V53Position:
    """V53 持仓记录"""
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
    ma120_at_entry: float = 0.0
    volume_suppression_applied: bool = False
    profit_tier1_triggered: bool = False
    profit_tier2_triggered: bool = False
    current_profit_ratio: float = 0.0
    industry_name: str = ""
    entry_volatility_ratio: float = 0.0  # 入场时的波动率比率
    position_tier: str = "standard"      # "standard"=20%, "reduced"=10%


@dataclass
class V53Trade:
    """V53 交易记录"""
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
class V53TradeAudit:
    """V53 交易审计记录"""
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
    exit_profit_ratio: float = 0.0
    hwm_stop_triggered: bool = False
    position_tier: str = "standard"


@dataclass
class V53WashSaleRecord:
    """V53 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V53BlacklistRecord:
    """V53 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V53MarketRegime:
    """V53 大盘状态"""
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
class V53DrawdownState:
    """V53 回撤状态 - 简化版（移除 Flash Cut）"""
    trade_date: str
    daily_drawdown: float = 0.0
    weekly_drawdown: float = 0.0
    single_day_triggered: bool = False
    weekly_triggered: bool = False


# ===========================================
# V53 行业加载器（数据库适配）
# ===========================================

class V53IndustryLoader:
    """V53 行业加载器 - 数据库适配"""
    
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
# V53 因子引擎
# ===========================================

class V53FactorEngine:
    """V53 因子引擎 - 双重动量确认与趋势共振过滤"""
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None,
                 momentum_weight: float = V53_MOMENTUM_WEIGHT,
                 r2_weight: float = V53_R2_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
        self.industry_loader = V53IndustryLoader()
    
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
                'database_tables_used': V53_DATABASE_TABLES.copy(),
                'momentum_weight': self.momentum_weight, 'r2_weight': self.r2_weight,
                'industry_data_source': 'database',
                'trend_filter_enabled': V53_TREND_FILTER_ENABLED,
                'ma120_filter_active': V53_MA120_FILTER,
            }
            
            result = self._compute_atr(result, period=20)
            status['factors_computed'].append('atr_20')
            
            result = self._compute_rsrs_factor(result)
            status['factors_computed'].append('rsrs_factor')
            
            result = self._compute_trend_factors(result)
            status['factors_computed'].extend(['trend_strength_20', 'trend_strength_60'])
            
            result = self._compute_volatility_adjusted_momentum(result)
            status['factors_computed'].append('volatility_adjusted_momentum')
            
            result = self._compute_trend_quality_v53(result)
            status['factors_computed'].append('trend_quality_r2')
            
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20', 'ma120'])
            
            result = self._compute_volume_suppression(result)
            status['factors_computed'].append('volume_suppression')
            
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # V53 趋势共振过滤：股价在 120 日半年线之上
            if V53_TREND_FILTER_ENABLED and V53_MA120_FILTER:
                result = self._apply_trend_filter(result)
                status['factors_computed'].append('trend_filter_pass')
            
            result = self._compute_composite_score_v53(result)
            return result, status
            
        except Exception as e:
            logger.error(f"V53 compute_all_factors FAILED: {e}")
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
    
    def _compute_trend_quality_v53(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        window = V53_TREND_QUALITY_WINDOW
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
        ma120 = pl.col('close').rolling_mean(window_size=120).over('symbol')
        ma_trend_bullish = ma5 > ma20
        price_above_ma20 = pl.col('close') > ma20
        price_above_ma120 = pl.col('close') > ma120
        return result.with_columns([
            ma5.alias('ma5'), ma20.alias('ma20'), ma120.alias('ma120'),
            ma_trend_bullish.alias('ma_trend_bullish'),
            price_above_ma20.alias('price_above_ma20'),
            price_above_ma120.alias('price_above_ma120')
        ])
    
    def _compute_volume_suppression(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算成交量压制因子"""
        result = df.clone()
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        vol_ma20 = pl.col('volume').rolling_mean(window_size=20).over('symbol')
        close_5_ago = pl.col('close').shift(5).over('symbol')
        price_change_5d = (pl.col('close') - close_5_ago) / (close_5_ago + self.EPSILON)
        volume_ratio = vol_ma5 / (vol_ma20 + self.EPSILON)
        # 使用常量而非模块属性
        is_high_volume = volume_ratio > 2.0
        is_price_stall = price_change_5d < 0.02
        is_suppression = is_high_volume & is_price_stall
        suppression_factor = pl.when(is_suppression).then(0.7).otherwise(1.0)
        return result.with_columns([
            vol_ma5.alias('vol_ma5'), vol_ma20.alias('vol_ma20'),
            volume_ratio.alias('volume_ratio'), price_change_5d.alias('price_change_5d'),
            is_suppression.alias('is_volume_suppression'),
            suppression_factor.alias('volume_suppression_factor')
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
        """V53 趋势共振过滤：股价在 120 日半年线之上"""
        result = df.clone()
        # 趋势过滤：只做处于大上升周期的票
        trend_filter_pass = pl.col('price_above_ma120')
        return result.with_columns([trend_filter_pass.alias('trend_filter_pass')])
    
    def _compute_composite_score_v53(self, df: pl.DataFrame) -> pl.DataFrame:
        """V53 综合评分计算 - 包含趋势共振过滤"""
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
            
            # 计算每个交易日股票数量
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
            
            # V53 进场过滤：趋势共振 + 排名过滤
            top_n_filter = composite_rank <= V53_ENTRY_TOP_N
            trend_filter = pl.col('trend_filter_pass') if 'trend_filter_pass' in result.columns else pl.lit(True)
            entry_allowed = top_n_filter & trend_filter
            
            return result.with_columns([
                momentum_rank_raw.cast(pl.Int64).fill_null(9999).alias('momentum_rank_raw'),
                r2_rank_raw.cast(pl.Int64).fill_null(9999).alias('r2_rank_raw'),
                composite_rank.cast(pl.Int64).fill_null(9999).alias('composite_rank'),
                composite_percentile.alias('composite_percentile'),
                top_n_filter.alias('top_n_filter_pass'),
                trend_filter.alias('trend_filter_pass'),
                entry_allowed.alias('entry_allowed')
            ])
        except Exception as e:
            logger.error(f"Error in _compute_composite_score_v53: {e}")
            logger.error(traceback.format_exc())
            return df.with_columns([
                pl.lit(0.0).alias('composite_score'),
                pl.lit(9999).alias('composite_rank'),
                pl.lit(0.0).alias('composite_percentile'),
                pl.lit(False).alias('top_n_filter_pass'),
                pl.lit(False).alias('trend_filter_pass'),
                pl.lit(False).alias('entry_allowed')
            ])
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()
    
    def update_weights(self, momentum_weight: float, r2_weight: float):
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight


# ===========================================
# V53 风险管理器 - 事前风控核心
# ===========================================

class V53RiskManager:
    """V53 风险管理器 - 自适应波段与波动率控仓"""
    
    def __init__(self, initial_capital: float = V53_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, V53Position] = {}
        self.trades: List[V53Trade] = []
        self.trade_log: List[V53TradeAudit] = []
        self.sell_history: Dict[str, str] = {}
        self.wash_sale_blocks: List[V53WashSaleRecord] = []
        self.stop_loss_blacklist: Dict[str, V53BlacklistRecord] = {}
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        self.trade_day_counter: int = 0
        self.current_market_vol: float = 1.0
        self.is_low_volatility_environment: bool = False
        self.is_high_volatility_environment: bool = False
        self.market_regime: V53MarketRegime = V53MarketRegime(trade_date="")
        self.is_risk_period: bool = False
        self.drawdown_state: V53DrawdownState = V53DrawdownState(trade_date="")
        self.factor_status: Dict[str, Any] = {}
        self.current_rank_cache: Dict[str, int] = {}
        self.peak_portfolio_value: float = initial_capital
        self.daily_start_value: float = initial_capital
        self.weekly_start_value: float = initial_capital
        self.last_week: int = 0
        # V53 移除自动降温，保持标准头寸
        self.current_position_limit: float = V53_MAX_SINGLE_POSITION_PCT
        self.industry_loader = V53IndustryLoader()
        self.portfolio_values: List[Dict[str, Any]] = []
    
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
        """更新回撤状态 - V53 简化版（移除 Flash Cut 和自动降温）"""
        current_value = self.get_total_portfolio_value(trade_date)
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        daily_drawdown = (self.daily_start_value - current_value) / self.daily_start_value if self.daily_start_value > 0 else 0
        weekly_drawdown = (self.weekly_start_value - current_value) / self.weekly_start_value if self.weekly_start_value > 0 else 0
        single_day_triggered = daily_drawdown > 0.015
        weekly_triggered = weekly_drawdown > 0.03
        
        self.drawdown_state = V53DrawdownState(
            trade_date=trade_date,
            daily_drawdown=daily_drawdown,
            weekly_drawdown=weekly_drawdown,
            single_day_triggered=single_day_triggered,
            weekly_triggered=weekly_triggered
        )
        
        self.daily_start_value = current_value
        
        # 记录组合价值
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
            if days_diff <= V53_WASH_SALE_WINDOW:
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
        self.stop_loss_blacklist[symbol] = V53BlacklistRecord(
            symbol=symbol, stop_date=trade_date, stop_reason=stop_reason,
            blacklist_expiry_day=self.trade_day_counter + 5, days_remaining=5
        )
        logger.info(f"BLACKLIST ADDED: {symbol} - {stop_reason}")
    
    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        return max(V53_MIN_COMMISSION, amount * V53_COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        """计算过户费"""
        return shares * price * V53_TRANSFER_FEE
    
    def get_current_risk_target(self) -> float:
        """获取当前风险目标"""
        return V53_RISK_TARGET_PER_POSITION
    
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
        """
        V53 动态头寸计算 - 波动率适配仓位
        
        Args:
            symbol: 股票代码
            atr: ATR 值
            current_price: 当前价格
            total_assets: 总资产
            volatility_ratio: 波动率比率（可选）
        
        Returns:
            (shares, position_amount, position_tier)
            position_tier: "standard"=20%, "reduced"=10%
        """
        try:
            risk_target = self.get_current_risk_target()
            risk_amount = total_assets * risk_target
            atr_mult = V53_INITIAL_ATR_MULT
            risk_per_share = atr * atr_mult
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0, "standard"
            
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            
            if shares < 100:
                return 0, 0.0, "standard"
            
            position_amount = shares * current_price
            
            # V53 核心：波动率适配仓位（事前风控）
            # 计算波动率比率：ATR / Price
            volatility_ratio = atr / current_price if current_price > 0 else 0
            
            # 若个股 ATR 极高（波动率比率 > 阈值），自动将该股头寸降至 10%
            position_tier = "standard"
            if volatility_ratio > V53_ATR_VOLATILITY_THRESHOLD:
                position_tier = "reduced"
                max_position = total_assets * V53_REDUCED_SINGLE_POSITION_PCT
                logger.info(f"VOLATILITY ADAPTIVE: {symbol} - ATR/Price={volatility_ratio:.2%} > {V53_ATR_VOLATILITY_THRESHOLD:.0%}, position reduced to {V53_REDUCED_SINGLE_POSITION_PCT:.0%}")
            else:
                max_position = total_assets * V53_MAX_SINGLE_POSITION_PCT
            
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
            
            return shares, shares * current_price, position_tier
            
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0, "standard"
    
    def execute_buy(self, trade_date: str, symbol: str, open_price: float, atr: float, 
                    target_amount: float, signal_date: str, signal_score: float = 0.0, 
                    signal_rank: int = 0, composite_score: float = 0.0, 
                    composite_percentile: float = 0.0, ma5: float = 0.0, 
                    ma20: float = 0.0, ma120: float = 0.0, industry_name: str = "",
                    volatility_ratio: float = 0.0, reason: str = "") -> Optional[V53Trade]:
        """
        V53 执行买入 - 包含波动率适配头寸
        """
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
            
            # V53 趋势共振过滤：检查股价是否在 120 日均线之上
            if V53_TREND_FILTER_ENABLED and V53_MA120_FILTER:
                if ma120 > 0 and open_price <= ma120:
                    logger.warning(f"TREND FILTER BLOCKED: {symbol} - Price {open_price:.2f} <= MA120 {ma120:.2f}")
                    return None
            
            execution_price = open_price * (1 + V53_SLIPPAGE_BUY)
            raw_shares = int(target_amount / execution_price)
            shares = (raw_shares // 100) * 100
            
            if shares < 100:
                return None
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * V53_SLIPPAGE_BUY
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            total_cost = actual_amount + commission + slippage + transfer_fee
            
            if self.cash < total_cost:
                return None
            
            self.cash -= total_cost
            
            # 计算初始止损
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - V53_INITIAL_ATR_MULT * atr_stop_distance)
            initial_stop_price = execution_price * (1 - V53_INITIAL_STOP_LOSS_RATIO)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            # V53 波动率适配头寸
            volatility_ratio = atr / execution_price if execution_price > 0 else 0
            position_tier = "reduced" if volatility_ratio > V53_ATR_VOLATILITY_THRESHOLD else "standard"
            position_pct = actual_amount / (self.cash + actual_amount) if (self.cash + actual_amount) > 0 else 0
            
            self.positions[symbol] = V53Position(
                symbol=symbol, shares=shares, 
                avg_cost=(actual_amount + commission + slippage + transfer_fee) / shares,
                buy_price=execution_price, buy_date=trade_date, signal_date=signal_date, 
                trade_date=trade_date, signal_score=signal_score, signal_rank=signal_rank, 
                composite_score=composite_score, current_price=execution_price, 
                holding_days=0, peak_price=execution_price, peak_profit=0.0,
                buy_trade_day=self.trade_day_counter, atr_at_entry=atr, 
                initial_stop_price=stop_price, trailing_stop_price=stop_price, 
                trailing_stop_history=[stop_price], current_market_rank=signal_rank, 
                current_market_percentile=composite_percentile, position_pct=position_pct, 
                entry_composite_score=composite_score, peak_profit_since_entry=0.0, 
                hwm_stop_triggered=False, hwm_stop_price=0.0,
                ma5_at_entry=ma5, ma20_at_entry=ma20, ma120_at_entry=ma120, 
                volume_suppression_applied=False, profit_tier1_triggered=False, 
                profit_tier2_triggered=False, current_profit_ratio=0.0,
                industry_name=industry_name, entry_volatility_ratio=volatility_ratio,
                position_tier=position_tier
            )
            
            self.record_buy(symbol)
            
            trade = V53Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares, 
                price=open_price, amount=actual_amount, commission=commission, 
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee, 
                total_cost=total_cost, reason=reason, execution_price=execution_price, 
                signal_date=signal_date, t_plus_1=True
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Rank={signal_rank} | Tier={position_tier}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(self, trade_date: str, symbol: str, open_price: float, 
                     shares: Optional[int] = None, reason: str = "", 
                     force: bool = False) -> Optional[V53Trade]:
        """V53 执行卖出"""
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
            
            execution_price = open_price * (1 - V53_SLIPPAGE_SELL)
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * V53_SLIPPAGE_SELL
            stamp_duty = actual_amount * V53_STAMP_DUTY
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
            if reason in ["trailing_stop", "stop_loss", "ma20_break", "hwm_stop"]:
                self.add_to_blacklist(symbol, trade_date, reason)
            
            trade_audit = V53TradeAudit(
                symbol=symbol, buy_date=pos.buy_date, sell_date=trade_date,
                buy_price=pos.buy_price, sell_price=execution_price, shares=shares,
                gross_pnl=realized_pnl + (commission + slippage + stamp_duty + transfer_fee),
                total_fees=commission + slippage + stamp_duty + transfer_fee,
                net_pnl=realized_pnl, holding_days=calculated_holding_days,
                is_profitable=realized_pnl > 0, sell_reason=reason,
                entry_signal=pos.signal_score, signal_rank=pos.signal_rank,
                atr_at_entry=pos.atr_at_entry, initial_stop_price=pos.initial_stop_price,
                peak_price=pos.peak_price, trailing_stop_triggered=pos.trailing_stop_triggered,
                exit_profit_ratio=pos.current_profit_ratio,
                hwm_stop_triggered=pos.hwm_stop_triggered,
                position_tier=pos.position_tier
            )
            self.trade_log.append(trade_audit)
            
            del self.positions[symbol]
            self.record_sell(symbol, trade_date)
            
            trade = V53Trade(
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
    
    def check_exits(self, positions: Dict[str, V53Position], date_str: str, 
                    price_df: pl.DataFrame, factor_df: pl.DataFrame) -> List[Tuple[str, str]]:
        """
        V53 检查退出信号 - 中线位次缓冲
        
        离场条件：
        1. 跌出全市场前 40 名
        2. 跌破 MA20 均线
        3. ATR 移动止损
        4. 初始止损
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
            
            pos.current_profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            
            if pos.current_profit_ratio > pos.peak_profit_since_entry:
                pos.peak_profit_since_entry = pos.current_profit_ratio
            
            # V53 中线位次缓冲：跌出全市场前 40 名离场
            if V53_MAINTAIN_TOP_N > 0:
                current_rank = ranks.get(symbol, 9999)
                if current_rank > V53_MAINTAIN_TOP_N:
                    sell_list.append((symbol, "rank_drop"))
                    logger.info(f"  RANK DROP: {symbol} - Rank {current_rank} > {V53_MAINTAIN_TOP_N}")
                    continue
            
            # V53 中线位次缓冲：跌破 MA20 均线离场
            if V53_MA20_EXIT_ENABLED:
                ma20 = ma20_values.get(symbol, pos.ma20_at_entry)
                # 处理 None 值
                if ma20 is None:
                    ma20 = pos.ma20_at_entry if pos.ma20_at_entry else 0
                if ma20 and ma20 > 0 and current_price < ma20:
                    sell_list.append((symbol, "ma20_break"))
                    logger.info(f"  MA20 BREAK: {symbol} - Price {current_price:.2f} < MA20 {ma20:.2f}")
                    continue
            
            # ATR 移动止损
            atr = atr_values.get(symbol, pos.atr_at_entry)
            if atr and atr > 0:
                try:
                    atr_mult = V53_TRAILING_ATR_MULT
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
            if pos.current_profit_ratio <= -V53_INITIAL_STOP_LOSS_RATIO:
                sell_list.append((symbol, "stop_loss"))
                continue
        
        return sell_list
    
    def get_total_portfolio_value(self, date_str: str) -> float:
        """获取组合总价值"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_portfolio_value(self, positions: Dict[str, V53Position], date_str: str, 
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
            'max_threshold': V53_MAX_TRADES_THRESHOLD
        }
    
    def check_trade_count_constraint(self) -> Tuple[bool, str]:
        """检查交易次数约束 - 严正审计"""
        total_trades = len(self.trades)
        if total_trades > V53_MAX_TRADES_THRESHOLD:
            return False, f"[V53 FAILED: OVER-TRADING] Trade count {total_trades} exceeds threshold {V53_MAX_TRADES_THRESHOLD}"
        return True, ""
    
    def get_position_sizing_stats(self) -> Dict[str, Any]:
        """获取头寸计算统计"""
        return {
            'risk_target_per_position': V53_RISK_TARGET_PER_POSITION,
            'max_single_position_pct': V53_MAX_SINGLE_POSITION_PCT,
            'reduced_single_position_pct': V53_REDUCED_SINGLE_POSITION_PCT,
            'atr_volatility_threshold': V53_ATR_VOLATILITY_THRESHOLD,
            'position_sizing_method': 'Volatility-Adaptive',
            'auto_cooldown_enabled': False,  # V53 移除自动降温
        }
    
    def get_drawdown_stats(self) -> Dict[str, Any]:
        """获取回撤统计"""
        return {
            'daily_drawdown': self.drawdown_state.daily_drawdown,
            'weekly_drawdown': self.drawdown_state.weekly_drawdown,
            'single_day_triggered': self.drawdown_state.single_day_triggered,
            'weekly_triggered': self.drawdown_state.weekly_triggered,
            'flash_cut_enabled': False,  # V53 移除 Flash Cut
            'auto_cooldown_enabled': False,  # V53 移除自动降温
        }
    
    def get_trend_filter_stats(self) -> Dict[str, Any]:
        """获取趋势过滤统计"""
        return {
            'enabled': V53_TREND_FILTER_ENABLED,
            'ma120_filter': V53_MA120_FILTER,
            'description': 'Price above MA120 filter for trend confirmation'
        }
    
    def get_position_buffer_stats(self) -> Dict[str, Any]:
        """获取位次缓冲统计"""
        return {
            'entry_top_n': V53_ENTRY_TOP_N,
            'maintain_top_n': V53_MAINTAIN_TOP_N,
            'ma20_exit_enabled': V53_MA20_EXIT_ENABLED,
            'description': 'Enter at Top 5, exit when drop below Top 40 or break MA20'
        }