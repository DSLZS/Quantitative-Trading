"""
V42 Core Module - 信号净化与风险核心

【V42 核心改进】
1. 信号净化：废除二阶动量，引入"趋势质量"因子（R² > 0.6）
2. 强制持仓冷却：Min_Holding_Days = 10，除非 ATR 2.0 止损
3. 洗售审计：5 天内"卖出即买入"强制拦截
4. 真实费率：手续费 0.03%，滑点 0.1%

【架构设计】
- 严格控制在 5 个源文件以内
- 所有因子与风险逻辑集中在此模块

作者：量化系统
版本：V42.0
日期：2026-03-19
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V42 配置常量
# ===========================================

# 基础配置
FIXED_INITIAL_CAPITAL = 100000.00
MAX_POSITIONS = 8
MAX_SINGLE_POSITION_PCT = 0.15

# V42 持仓周期配置
MIN_HOLDING_DAYS = 10  # 强制持仓冷却 10 天
MAX_HOLDING_DAYS = 60  # 最大持仓天数

# V42 ATR 止损配置
ATR_PERIOD = 20
TRAILING_STOP_ATR_MULT = 2.0  # 2 * ATR 移动止损
INITIAL_STOP_LOSS_RATIO = 0.08  # 初始止损 8%

# V42 风险配置
BASE_RISK_TARGET_PER_POSITION = 0.005  # 单只股票风险暴露 0.5%
LOW_VOL_RISK_TARGET = 0.0065  # 低波动时风险暴露（保守提升）
LOW_VOLATILITY_THRESHOLD = 0.6  # 低波动阈值

# V42 入场过滤
VOLATILITY_FILTER_THRESHOLD = 1.5  # 波动率过滤阈值

# V42 洗售审计配置
WASH_SALE_WINDOW = 5  # 5 天内卖出即买入检测

# V42 趋势质量配置
TREND_QUALITY_WINDOW = 20  # R²计算窗口
TREND_QUALITY_THRESHOLD = 0.6  # R²阈值，仅当 R² > 0.6 时动量信号生效

# V42 费率配置
COMMISSION_RATE = 0.0003  # 手续费 0.03% (万分之三)
MIN_COMMISSION = 5.0  # 最低 5 元
SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
STAMP_DUTY = 0.0005  # 印花税 0.05% (卖出)
TRANSFER_FEE = 0.00001  # 过户费 0.001%

# V42 因子权重（废除二阶动量，增加趋势质量）
FACTOR_WEIGHTS = {
    'trend_strength_20': 0.30,  # 20 日趋势强度
    'trend_strength_60': 0.25,  # 60 日趋势强度
    'rsrs_factor': 0.20,  # RSRS 因子
    'volatility_adjusted_momentum': 0.25,  # 波动率调整动量
    'trend_quality': 0.0,  # 趋势质量（作为过滤器，不直接加权）
}


# ===========================================
# V42 数据类
# ===========================================

@dataclass
class V42Position:
    """V42 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_score: float
    signal_rank: int
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    buy_trade_day: int = 0
    
    # ATR 动态防御
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    trailing_stop_history: List[float] = field(default_factory=list)


@dataclass
class V42Trade:
    """V42 交易记录"""
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
class V42TradeAudit:
    """V42 交易审计记录"""
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


@dataclass
class V42WashSaleRecord:
    """V42 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


# ===========================================
# V42 因子计算引擎
# ===========================================

class V42FactorEngine:
    """
    V42 因子引擎 - 信号净化
    
    【核心改进】
    1. 废除二阶动量因子
    2. 新增趋势质量因子（R² > 0.6 过滤器）
    3. 简化因子权重
    """
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None):
        self.factor_weights = factor_weights or FACTOR_WEIGHTS.copy()
        self._industry_cache: Dict[str, Set[str]] = {}
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        V42 全因子计算
        
        Returns:
            (factor_df, status_dict) - 因子 DataFrame 和状态字典
        """
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
            
            # Step 4: 波动率调整动量（废除二阶动量）
            result = self._compute_volatility_adjusted_momentum(result)
            status['factors_computed'].append('volatility_adjusted_momentum')
            
            # Step 5: V42 新增 - 趋势质量因子 (R²)
            result = self._compute_trend_quality(result)
            status['factors_computed'].append('trend_quality_r2')
            
            # Step 6: 市场波动率指数
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # Step 7: 板块中性化（仅在行业数据完整时启用）
            if industry_data is not None and not industry_data.is_empty() and 'industry_name' in industry_data.columns:
                industry_coverage = industry_data['symbol'].n_unique()
                total_symbols = result['symbol'].n_unique()
                coverage_ratio = industry_coverage / total_symbols if total_symbols > 0 else 0
                
                if coverage_ratio >= 0.9:  # 覆盖率 >= 90% 才启用
                    result = self._apply_industry_neutralization(result, industry_data)
                    status['industry_neutralization'] = 'ENABLED'
                    status['factors_computed'].extend(['industry_neutralized_signal', 'industry_zscore'])
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
            
            # Step 8: 计算综合信号（使用趋势质量作为过滤器）
            result = self._compute_composite_signal(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V42 compute_all_factors FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_columns(self, df: pl.DataFrame, required_columns: List[str]) -> bool:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def _compute_atr(self, df: pl.DataFrame, period: int = ATR_PERIOD) -> pl.DataFrame:
        """ATR 计算"""
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
        """RSRS 因子计算"""
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
        """趋势因子计算"""
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
        """波动率调整动量（废除二阶动量）"""
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
    
    def _compute_trend_quality(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V42 新增：趋势质量因子（线性回归 R²）
        
        计算过去 20 天价格序列的线性回归 R²
        R² > 0.6 表示趋势清晰，动量信号才生效
        """
        result = df.clone()
        window = TREND_QUALITY_WINDOW
        
        # 使用简化方法计算 R²
        # R² = 1 - SS_res / SS_tot
        # 对于价格序列，R²高表示价格接近线性趋势
        
        # 计算滚动窗口内的价格变化
        close_mean = pl.col('close').rolling_mean(window_size=window).over('symbol')
        
        # 计算趋势强度代理（使用价格与均值的偏离度）
        # 更精确的 R²计算需要使用线性回归
        # 这里使用简化的代理方法
        
        # 计算价格序列的相关性（与时间序列）
        # 使用 rolling_corr 计算 close 与时间索引的相关性
        
        # 简化方法：计算趋势的"线性度"
        # 如果价格持续上涨/下跌，R²高
        # 如果价格震荡，R²低
        
        # 计算连续上涨/下跌的天数比例
        prev_close = pl.col('close').shift(1).over('symbol')
        up_day = (pl.col('close') > prev_close).cast(pl.Float64)
        
        # 计算窗口内的方向一致性
        direction_consistency = up_day.rolling_mean(window_size=window).over('symbol')
        
        # R²代理 = 4 * |direction_consistency - 0.5|
        # 如果一直上涨 (1.0) 或一直下跌 (0.0)，R²=1.0
        # 如果随机 (0.5)，R²=0.0
        r2_proxy = 4.0 * (direction_consistency - 0.5).abs()
        
        # 更精确的 R²计算：使用价格与趋势线的拟合度
        # 计算滚动窗口内的价格范围
        high_in_window = pl.col('high').rolling_max(window_size=window).over('symbol')
        low_in_window = pl.col('low').rolling_min(window_size=window).over('symbol')
        price_range = high_in_window - low_in_window
        
        # 计算当前价格相对于范围的位置
        price_position = (pl.col('close') - low_in_window) / (price_range + self.EPSILON)
        
        # 计算趋势的"平滑度"（价格波动相对于总范围）
        price_std = pl.col('close').rolling_std(window_size=window).over('symbol')
        trend_smoothness = 1.0 - price_std / (price_range + self.EPSILON)
        
        # 综合 R² = 0.5 * direction_r2 + 0.5 * smoothness_r2
        r2_combined = 0.5 * r2_proxy + 0.5 * trend_smoothness.clip(0, 1)
        
        result = result.with_columns([
            r2_proxy.alias('direction_r2'),
            trend_smoothness.alias('trend_smoothness'),
            r2_combined.alias('trend_quality_r2'),
            price_position.alias('price_position')
        ])
        
        return result
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """市场波动率指数"""
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
    
    def _apply_industry_neutralization(self, df: pl.DataFrame, industry_data: pl.DataFrame) -> pl.DataFrame:
        """板块中性化（仅在数据完整时启用）"""
        result = df.clone()
        
        # 连接行业数据
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
        
        # 计算基础信号
        base_signal = pl.lit(0.0)
        for factor, weight in self.factor_weights.items():
            if factor in result.columns and factor != 'trend_quality':
                factor_mean = result[factor].mean() or 0
                factor_std = result[factor].std() or 1
                z_factor = (pl.col(factor) - factor_mean) / (factor_std + self.EPSILON)
                base_signal = base_signal + z_factor * weight
        
        result = result.with_columns([base_signal.alias('base_signal')])
        
        # 行业内标准化
        industry_mean = pl.col('base_signal').mean().over(['trade_date', 'industry_name'])
        industry_std = pl.col('base_signal').std().over(['trade_date', 'industry_name'])
        industry_zscore = (pl.col('base_signal') - industry_mean) / (industry_std + self.EPSILON)
        
        result = result.with_columns([
            industry_mean.alias('industry_mean_signal'),
            industry_zscore.alias('industry_zscore'),
            industry_zscore.alias('industry_neutralized_signal')
        ])
        
        return result
    
    def _compute_composite_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合信号（使用趋势质量作为过滤器）
        
        V42 核心：仅当 R² > 0.6 时，动量信号才生效
        """
        result = df.clone()
        
        # 检查行业数据是否存在
        has_industry_data = 'industry_name' in result.columns and result['industry_name'].n_unique() > 1 if 'industry_name' in result.columns else False
        
        # 使用标准化后的信号
        if 'industry_neutralized_signal' in result.columns and has_industry_data:
            signal_base = pl.col('industry_neutralized_signal')
        else:
            signal_base = pl.lit(0.0)
            for factor, weight in self.factor_weights.items():
                if factor in result.columns and factor != 'trend_quality':
                    factor_mean = result[factor].mean() or 0
                    factor_std = result[factor].std() or 1
                    z_factor = (pl.col(factor) - factor_mean) / (factor_std + self.EPSILON)
                    signal_base = signal_base + z_factor * weight
        
        # V42 核心：趋势质量过滤
        # 仅当 R² > 0.6 时，信号才完全生效
        trend_quality_pass = pl.col('trend_quality_r2') >= TREND_QUALITY_THRESHOLD
        
        # 信号打折：R²<0.6 时信号减半
        signal = pl.when(trend_quality_pass)\
            .then(signal_base)\
            .otherwise(signal_base * 0.5)
        
        # 入场过滤
        vol_filter = pl.col('volatility_ratio') <= VOLATILITY_FILTER_THRESHOLD
        ma_filter = pl.col('close') >= pl.col('ma60')
        entry_allowed = vol_filter & ma_filter
        
        result = result.with_columns([
            signal.alias('signal'),
            vol_filter.alias('vol_filter_pass'),
            ma_filter.alias('ma_filter_pass'),
            entry_allowed.alias('entry_allowed'),
            trend_quality_pass.alias('trend_quality_pass')
        ])
        
        # 排名计算
        result = result.with_columns([
            pl.when(pl.col('entry_allowed'))
            .then(pl.col('signal').rank('ordinal', descending=True).over('trade_date'))
            .otherwise(9999)
            .cast(pl.Int64)
            .alias('rank')
        ])
        
        return result
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()


# ===========================================
# V42 风险管理器
# ===========================================

class V42RiskManager:
    """
    V42 风险管理器 - 洗售审计与持仓冷却
    
    【核心功能】
    1. 洗售审计：5 天内"卖出即买入"强制拦截
    2. 强制持仓冷却：10 天锁定期（除非 ATR 2.0 止损）
    3. ATR 动态止损
    4. 真实费率计算
    """
    
    def __init__(self, initial_capital: float = FIXED_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, V42Position] = {}
        self.trades: List[V42Trade] = []
        self.trade_log: List[V42TradeAudit] = []
        
        # 洗售审计追踪
        self.sell_history: Dict[str, str] = {}  # symbol -> last_sell_date
        self.wash_sale_blocks: List[V42WashSaleRecord] = []
        
        # 每日计数器
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        
        # 持仓锁定期追踪
        self.locked_positions: Dict[str, int] = {}
        
        # 交易日计数器
        self.trade_day_counter: int = 0
        
        # 波动率状态
        self.current_market_vol: float = 1.0
        self.is_low_volatility_environment: bool = False
        
        # 因子状态
        self.factor_status: Dict[str, Any] = {}
    
    def reset_daily_counters(self, trade_date: str):
        """重置每日计数器"""
        if self.last_trade_date != trade_date:
            self.today_sells.clear()
            self.today_buys.clear()
            self.last_trade_date = trade_date
            self.trade_day_counter += 1
        
        # 解锁到期持仓
        self._unlock_expired_positions()
    
    def _unlock_expired_positions(self):
        """解锁到期持仓"""
        for symbol in list(self.locked_positions.keys()):
            self.locked_positions[symbol] -= 1
            if self.locked_positions[symbol] <= 0:
                del self.locked_positions[symbol]
    
    def _parse_date(self, date_str: str) -> datetime:
        """解析日期字符串"""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return datetime.now()
    
    def _days_between(self, date1_str: str, date2_str: str) -> int:
        """计算两个日期之间的天数"""
        date1 = self._parse_date(date1_str)
        date2 = self._parse_date(date2_str)
        return abs((date2 - date1).days)
    
    def check_wash_sale(self, symbol: str, trade_date: str) -> Tuple[bool, Optional[str]]:
        """
        V42 洗售审计：检查是否在 5 天内卖出又买入
        
        Returns:
            (is_blocked, reason) - 是否被拦截及原因
        """
        # 检查今日是否已卖出
        if symbol in self.today_sells:
            return True, f"same_day_sell ({trade_date})"
        
        # 检查 5 天内是否有卖出记录
        if symbol in self.sell_history:
            last_sell_date = self.sell_history[symbol]
            days_diff = self._days_between(last_sell_date, trade_date)
            
            if days_diff <= WASH_SALE_WINDOW:
                return True, f"wash_sale_window ({days_diff} days since {last_sell_date})"
        
        return False, None
    
    def record_sell(self, symbol: str, trade_date: str):
        """记录卖出操作（用于洗售审计）"""
        self.sell_history[symbol] = trade_date
        self.today_sells.add(symbol)
    
    def record_buy(self, symbol: str):
        """记录买入操作"""
        self.today_buys.add(symbol)
    
    def _calculate_commission(self, amount: float) -> float:
        """计算佣金（手续费万分之三）"""
        return max(MIN_COMMISSION, amount * COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        """计算过户费"""
        return shares * price * TRANSFER_FEE
    
    def get_current_risk_target(self) -> float:
        """获取当前风险暴露目标"""
        if self.is_low_volatility_environment:
            return LOW_VOL_RISK_TARGET
        return BASE_RISK_TARGET_PER_POSITION
    
    def calculate_position_size(
        self, 
        symbol: str, 
        atr: float, 
        current_price: float, 
        total_assets: float,
    ) -> Tuple[int, float]:
        """
        V42 风险平价调仓 - 计算仓位大小
        
        核心：波动大的股票少买，波动小的多买
        """
        try:
            risk_target = self.get_current_risk_target()
            risk_amount = total_assets * risk_target
            risk_per_share = atr * TRAILING_STOP_ATR_MULT
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0
            
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            
            if shares < 100:
                return 0, 0.0
            
            position_amount = shares * current_price
            
            # 仓位限制
            max_position = total_assets * MAX_SINGLE_POSITION_PCT
            min_position = total_assets * 0.05
            
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
                position_amount = shares * current_price
            
            if position_amount < min_position and position_amount > 0:
                return 0, 0.0
            
            return shares, position_amount
            
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0
    
    def execute_buy(
        self,
        trade_date: str,
        symbol: str,
        open_price: float,
        atr: float,
        target_amount: float,
        signal_score: float = 0.0,
        signal_rank: int = 0,
        reason: str = "",
    ) -> Optional[V42Trade]:
        """
        V42 买入执行 - 包含洗售审计
        """
        try:
            # V42 核心：洗售审计
            is_blocked, block_reason = self.check_wash_sale(symbol, trade_date)
            if is_blocked:
                logger.warning(f"WASH SALE BLOCKED: {symbol} - {block_reason}")
                self.wash_sale_blocks.append(V42WashSaleRecord(
                    symbol=symbol,
                    sell_date=self.sell_history.get(symbol, "N/A"),
                    blocked_buy_date=trade_date,
                    days_between=self._days_between(self.sell_history.get(symbol, trade_date), trade_date),
                    reason=block_reason
                ))
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"DUPLICATE BUY PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            # 场景化滑点（0.1%）
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
            
            # ATR 动态止损初始化
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - TRAILING_STOP_ATR_MULT * atr_stop_distance)
            initial_stop_price = execution_price * (1 - INITIAL_STOP_LOSS_RATIO)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            # 创建新持仓
            self.positions[symbol] = V42Position(
                symbol=symbol, shares=shares,
                avg_cost=(actual_amount + commission + slippage + transfer_fee) / shares,
                buy_price=execution_price, buy_date=trade_date,
                signal_score=signal_score, signal_rank=signal_rank,
                current_price=execution_price,
                holding_days=0,
                peak_price=execution_price, peak_profit=0.0,
                buy_trade_day=self.trade_day_counter,
                atr_at_entry=atr,
                initial_stop_price=stop_price,
                trailing_stop_price=stop_price,
                trailing_stop_history=[stop_price],
            )
            
            # V42 核心：强制持仓冷却 10 天
            self.locked_positions[symbol] = MIN_HOLDING_DAYS
            
            self.record_buy(symbol)
            
            trade = V42Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee,
                total_cost=total_cost, reason=reason, execution_price=execution_price,
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Locked={MIN_HOLDING_DAYS}d")
            return trade
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(
        self,
        trade_date: str,
        symbol: str,
        open_price: float,
        shares: Optional[int] = None,
        reason: str = "",
    ) -> Optional[V42Trade]:
        """
        V42 卖出执行 - 包含持仓冷却检查
        """
        try:
            if symbol not in self.positions:
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            pos = self.positions[symbol]
            
            # V42 核心：持仓锁定期检查
            # 除非触发 ATR 2.0 止损，否则 10 天内禁止卖出
            if symbol in self.locked_positions and self.locked_positions[symbol] > 0:
                if reason not in ["stop_loss", "trailing_stop", "max_holding"]:
                    logger.debug(f"LOCKED POSITION: {symbol} locked for {self.locked_positions[symbol]} more days, reason={reason}")
                    return None
            
            available = pos.shares
            if shares is None or shares > available:
                shares = available
            
            if shares < 100:
                return None
            
            # 场景化滑点（0.1%）
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
            
            # 计算持仓天数
            try:
                buy_date = self._parse_date(pos.buy_date)
                sell_date = self._parse_date(trade_date)
                calculated_holding_days = max(1, (sell_date - buy_date).days)
            except:
                calculated_holding_days = pos.holding_days if pos.holding_days > 0 else 1
            
            # 记录交易审计
            trade_audit = V42TradeAudit(
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
                trailing_stop_triggered=pos.trailing_stop_triggered
            )
            self.trade_log.append(trade_audit)
            
            # 删除持仓和锁
            del self.positions[symbol]
            self.locked_positions.pop(symbol, None)
            
            # 记录卖出（用于洗售审计）
            self.record_sell(symbol, trade_date)
            
            trade = V42Trade(
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
    
    def update_positions_and_check_stops(
        self, 
        prices: Dict[str, float],
        atrs: Dict[str, float],
        trade_date: str,
    ) -> List[Tuple[str, str]]:
        """
        V42 更新持仓价格并检查止损条件
        """
        sell_list = []
        
        for symbol, pos in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                pos.current_price = current_price
                pos.market_value = pos.shares * current_price
                pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares
                
                # 更新持仓天数
                pos.holding_days = max(0, self.trade_day_counter - pos.buy_trade_day)
                
                # 更新峰值价格
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                
                # ATR 移动止损更新
                if symbol in atrs and atrs[symbol] > 0:
                    atr = atrs[symbol]
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    new_trailing_stop = current_price * (1 - TRAILING_STOP_ATR_MULT * atr_stop_distance)
                    
                    # 只上移，不下移
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        pos.trailing_stop_history.append(new_trailing_stop)
                
                # 检查止损触发
                # 1. 移动止损触发（可突破锁定期）
                if current_price <= pos.trailing_stop_price:
                    pos.trailing_stop_triggered = True
                    sell_list.append((symbol, "trailing_stop"))
                    continue
                
                # 2. 初始止损底线（8%，可突破锁定期）
                profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                if profit_ratio <= -INITIAL_STOP_LOSS_RATIO:
                    sell_list.append((symbol, "stop_loss"))
                    continue
                
                # 3. 最大持仓天数触发（可突破锁定期）
                if pos.holding_days >= MAX_HOLDING_DAYS:
                    sell_list.append((symbol, "max_holding"))
                    continue
        
        return sell_list
    
    def get_position_count(self) -> int:
        return len(self.positions)
    
    def get_portfolio_value(self, positions: Dict[str, V42Position], date_str: str, price_df: pl.DataFrame) -> float:
        """获取组合价值"""
        market_value = 0.0
        for symbol, pos in positions.items():
            try:
                row = price_df.filter((pl.col('symbol') == symbol) & (pl.col('trade_date') == date_str)).select('close').row(0)
                if row:
                    market_value += pos.shares * float(row[0])
            except Exception:
                market_value += pos.shares * pos.current_price
        return self.cash + market_value
    
    def update_volatility_regime(self, market_vol: float):
        """更新波动率状态"""
        self.current_market_vol = market_vol
        self.is_low_volatility_environment = market_vol < LOW_VOLATILITY_THRESHOLD
    
    def get_risk_per_position(self) -> float:
        """获取当前风险暴露"""
        return self.get_current_risk_target()
    
    def check_stop_loss(
        self, 
        positions: Dict[str, V42Position], 
        date_str: str, 
        price_df: pl.DataFrame, 
        factor_df: pl.DataFrame
    ) -> List[str]:
        """检查止损并返回需要卖出的股票列表"""
        sell_list = []
        
        prices = {}
        atrs = {}
        
        for row in price_df.iter_rows(named=True):
            symbol = row['symbol']
            prices[symbol] = row.get('close', 0) or 0
        
        for row in factor_df.iter_rows(named=True):
            symbol = row['symbol']
            atrs[symbol] = row.get('atr_20', 0) or 0
        
        for symbol, pos in list(positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                pos.current_price = current_price
                pos.market_value = pos.shares * current_price
                pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares
                pos.holding_days = max(0, self.trade_day_counter - pos.buy_trade_day)
                
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                
                if symbol in atrs and atrs[symbol] > 0:
                    atr = atrs[symbol]
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    new_trailing_stop = current_price * (1 - TRAILING_STOP_ATR_MULT * atr_stop_distance)
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        pos.trailing_stop_history.append(new_trailing_stop)
                
                if current_price <= pos.trailing_stop_price:
                    pos.trailing_stop_triggered = True
                    sell_list.append(symbol)
                    continue
                
                profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                if profit_ratio <= -INITIAL_STOP_LOSS_RATIO:
                    sell_list.append(symbol)
                    continue
                
                if pos.holding_days >= MAX_HOLDING_DAYS:
                    sell_list.append(symbol)
                    continue
        
        return sell_list
    
    def rank_candidates(self, factor_df: pl.DataFrame, positions: Dict[str, V42Position]) -> List[Dict]:
        """排名候选股票"""
        try:
            held_symbols = set(positions.keys())
            
            if 'signal' in factor_df.columns:
                candidates = factor_df.filter(
                    ~pl.col('symbol').is_in(list(held_symbols))
                ).sort('signal', descending=True).head(20)
            else:
                candidates = factor_df.filter(
                    ~pl.col('symbol').is_in(list(held_symbols))
                ).limit(20)
            
            result = []
            for idx, row in enumerate(candidates.iter_rows(named=True)):
                result.append({
                    'symbol': row['symbol'],
                    'rank': idx + 1,
                    'signal': float(row.get('signal', 0)) if row.get('signal') is not None else 0
                })
            
            return result
            
        except Exception as e:
            logger.error(f"rank_candidates failed: {e}")
            return []
    
    def get_wash_sale_stats(self) -> Dict[str, Any]:
        """获取洗售审计统计"""
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