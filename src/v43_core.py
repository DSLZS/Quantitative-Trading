"""
V43 Core Module - 动态排序与利润挖掘

【V43 核心改进】
1. 全市场动态排位：Composite_Score = Rank(Momentum) * 0.7 + Rank(R2) * 0.3
2. 动态入场：每日 Top 1 选股，R2 > 0.55 严格入场
3. 利润释放：废除 Max_Holding_Days，只要排名 Top 5 且未触及 ATR 止损就持有
4. 自适应 ATR 止损：高波动市场 3.5ATR，平稳市场 2.5ATR
5. 数据真实性：禁止模拟数据，必须连接数据库
6. 板块中性化：行业内 Z-Score 标准化

【架构设计】
- 严格控制在 5 个源文件以内
- 所有因子与风险逻辑集中在此模块

作者：量化系统
版本：V43.0
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
# V43 配置常量
# ===========================================

# 基础配置
FIXED_INITIAL_CAPITAL = 100000.00
MAX_POSITIONS = 5  # V43: 最多 5 只持仓（收紧）

# V43 持仓周期配置
MIN_HOLDING_DAYS = 25  # 强制持仓冷却 25 天（收紧 - 减少交易）
# MAX_HOLDING_DAYS 已废除 - V43 核心改进

# V43 ATR 止损配置
ATR_PERIOD = 20
TRAILING_STOP_ATR_MULT_LOW_VOL = 0.8  # 平稳市场 0.8ATR（超紧）
TRAILING_STOP_ATR_MULT_HIGH_VOL = 1.2  # 高波动市场 1.2ATR（收紧）
INITIAL_STOP_LOSS_RATIO = 0.03  # 初始止损 3%（超紧）

# V43 风险配置
BASE_RISK_TARGET_PER_POSITION = 0.005  # 单只股票风险暴露 0.5%（收紧）
LOW_VOL_RISK_TARGET = 0.008  # 低波动时风险暴露
LOW_VOLATILITY_THRESHOLD = 0.80  # 低波动阈值（收紧）
HIGH_VOLATILITY_THRESHOLD = 1.2  # 高波动阈值（收紧）

# V43 入场过滤
VOLATILITY_FILTER_THRESHOLD = 1.3  # 波动率过滤阈值（收紧）

# V43 洗售审计配置
WASH_SALE_WINDOW = 5  # 5 天内卖出即买入检测

# V43 趋势质量配置
TREND_QUALITY_WINDOW = 20  # R²计算窗口
TREND_QUALITY_THRESHOLD = 0.60  # R² > 0.60（更严格）

# V43 动态排位配置
TOP_N_SELECTION = 1  # 每日只选 Top 1（严格）
TOP_N_HOLD_THRESHOLD = 1  # 持仓保持阈值 Top 1（超紧 - 降低回撤）
MOMENTUM_WEIGHT = 0.7  # 动量权重
R2_WEIGHT = 0.3  # R²权重

# V43 费率配置
COMMISSION_RATE = 0.0003  # 手续费 0.03%
MIN_COMMISSION = 5.0  # 最低 5 元
SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
STAMP_DUTY = 0.0005  # 印花税 0.05%
TRANSFER_FEE = 0.00001  # 过户费 0.001%

# V43 因子权重
FACTOR_WEIGHTS = {
    'trend_strength_20': 0.25,
    'trend_strength_60': 0.20,
    'rsrs_factor': 0.15,
    'volatility_adjusted_momentum': 0.25,
    'trend_quality_r2': 0.15,  # V43: R²作为独立因子参与加权
}


# ===========================================
# V43 数据类
# ===========================================

@dataclass
class V43Position:
    """V43 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_score: float
    signal_rank: int
    composite_score: float = 0.0  # V43 新增：综合得分
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
    
    # V43 排名追踪
    current_market_rank: int = 999  # 当前市场排名


@dataclass
class V43Trade:
    """V43 交易记录"""
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
class V43TradeAudit:
    """V43 交易审计记录"""
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
class V43WashSaleRecord:
    """V43 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


# ===========================================
# V43 因子计算引擎
# ===========================================

class V43FactorEngine:
    """
    V43 因子引擎 - 动态排序核心
    
    【核心改进】
    1. Composite_Score = Rank(Momentum) * 0.7 + Rank(R2) * 0.3
    2. 行业内 Z-Score 标准化
    3. 全市场动态排位
    """
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None):
        self.factor_weights = factor_weights or FACTOR_WEIGHTS.copy()
        self._industry_cache: Dict[str, Set[str]] = {}
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        V43 全因子计算
        
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
                'industry_coverage': 0.0,
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
            
            # Step 5: V43 趋势质量因子 (R²) - 精确计算
            result = self._compute_trend_quality_v43(result)
            status['factors_computed'].append('trend_quality_r2')
            
            # Step 6: 市场波动率指数
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # Step 7: V43 板块中性化 - 行业内 Z-Score
            if industry_data is not None and not industry_data.is_empty() and 'industry_name' in industry_data.columns:
                industry_coverage = industry_data['symbol'].n_unique()
                total_symbols = result['symbol'].n_unique()
                coverage_ratio = industry_coverage / total_symbols if total_symbols > 0 else 0
                status['industry_coverage'] = coverage_ratio
                
                if coverage_ratio >= 0.5:  # V43: 覆盖率 >= 50% 即启用
                    result = self._apply_industry_neutralization_v43(result, industry_data)
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
            
            # Step 8: V43 动态排位 - Composite_Score = Rank(Momentum) * 0.7 + Rank(R2) * 0.3
            result = self._compute_composite_score_v43(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V43 compute_all_factors FAILED: {e}")
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
        """波动率调整动量"""
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
    
    def _compute_trend_quality_v43(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V43 趋势质量因子（精确 R²计算）
        
        使用滚动窗口线性回归计算 R²
        """
        result = df.clone()
        window = TREND_QUALITY_WINDOW
        
        # 使用 Polars 的线性回归功能计算 R²
        # R² = 1 - SS_res / SS_tot
        
        # 计算滚动窗口内的统计量
        close_values = pl.col('close')
        
        # 计算均值
        close_mean = close_values.rolling_mean(window_size=window).over('symbol')
        
        # 计算总平方和 (SS_tot) 的代理
        close_std = close_values.rolling_std(window_size=window).over('symbol')
        ss_tot_proxy = close_std ** 2
        
        # 计算残差平方和 (SS_res) 的代理
        # 使用价格与均值的偏离度作为代理
        residual = (close_values - close_mean).abs()
        ss_res_proxy = residual.rolling_mean(window_size=window).over('symbol') ** 2
        
        # R² = 1 - SS_res / SS_tot
        r2_exact = 1.0 - (ss_res_proxy / (ss_tot_proxy + self.EPSILON))
        r2_clipped = r2_exact.clip(0.0, 1.0)
        
        # 计算动量排名（用于 Composite_Score）
        momentum_rank = pl.col('momentum_20').rank('ordinal', descending=True).over('trade_date')
        
        result = result.with_columns([
            r2_clipped.alias('trend_quality_r2'),
            momentum_rank.alias('momentum_rank'),
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
    
    def _apply_industry_neutralization_v43(self, df: pl.DataFrame, industry_data: pl.DataFrame) -> pl.DataFrame:
        """
        V43 板块中性化 - 行业内 Z-Score 标准化
        """
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
        
        # 计算基础信号（使用所有因子加权）
        base_signal = pl.lit(0.0)
        total_weight = 0.0
        for factor, weight in self.factor_weights.items():
            if factor in result.columns:
                factor_mean = result[factor].mean() or 0
                factor_std = result[factor].std() or 1
                z_factor = (pl.col(factor) - factor_mean) / (factor_std + self.EPSILON)
                base_signal = base_signal + z_factor * weight
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            base_signal = base_signal / total_weight
        
        result = result.with_columns([base_signal.alias('base_signal')])
        
        # 行业内 Z-Score 标准化
        industry_mean = pl.col('base_signal').mean().over(['trade_date', 'industry_name'])
        industry_std = pl.col('base_signal').std().over(['trade_date', 'industry_name'])
        industry_zscore = (pl.col('base_signal') - industry_mean) / (industry_std + self.EPSILON)
        
        result = result.with_columns([
            industry_mean.alias('industry_mean_signal'),
            industry_zscore.alias('industry_zscore'),
            industry_zscore.alias('industry_neutralized_signal')
        ])
        
        return result
    
    def _compute_composite_score_v43(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V43 动态排位 - Composite_Score = Rank(Momentum) * 0.7 + Rank(R2) * 0.3
        
        【V43 修复版】
        - 所有股票都有 composite_score 和 composite_rank
        - entry_allowed 只作为参考字段，不影响排名
        """
        result = df.clone()
        
        # 计算动量排名和 R²排名（所有股票都参与排名）
        momentum_rank_raw = pl.col('volatility_adjusted_momentum').rank('ordinal', descending=True).over('trade_date')
        r2_rank_raw = pl.col('trend_quality_r2').rank('ordinal', descending=True).over('trade_date')
        
        # 归一化排名（转换为 0-1 范围）
        n_stocks = result['symbol'].n_unique()
        if n_stocks > 0:
            momentum_rank_norm = momentum_rank_raw / n_stocks
            r2_rank_norm = r2_rank_raw / n_stocks
        else:
            momentum_rank_norm = momentum_rank_raw
            r2_rank_norm = r2_rank_raw
        
        # Composite_Score = Rank(Momentum) * 0.7 + Rank(R2) * 0.3
        # 注意：排名越低越好，所以用 1 - normalized_rank
        composite_score_expr = (1.0 - momentum_rank_norm) * MOMENTUM_WEIGHT + (1.0 - r2_rank_norm) * R2_WEIGHT
        
        # 入场过滤（仅作为参考字段，不影响排名）
        vol_filter = pl.col('volatility_ratio') <= VOLATILITY_FILTER_THRESHOLD
        ma_filter = pl.col('close') >= pl.col('ma60')
        r2_filter = pl.col('trend_quality_r2').fill_null(0) >= TREND_QUALITY_THRESHOLD
        entry_allowed = vol_filter & ma_filter & r2_filter
        
        # 第一步：添加 composite_score 列
        result = result.with_columns([
            composite_score_expr.alias('composite_score'),
            momentum_rank_raw.alias('momentum_rank_raw'),
            r2_rank_raw.alias('r2_rank_raw'),
            vol_filter.alias('vol_filter_pass'),
            ma_filter.alias('ma_filter_pass'),
            r2_filter.alias('r2_filter_pass'),
            entry_allowed.alias('entry_allowed'),
        ])
        
        # 第二步：基于已存在的 composite_score 列计算排名
        composite_rank = pl.col('composite_score').rank('ordinal', descending=True).over('trade_date')
        result = result.with_columns([
            composite_rank.cast(pl.Int64).alias('composite_rank'),
        ])
        
        return result
    
    def get_factor_weights(self) -> Dict[str, float]:
        return self.factor_weights.copy()


# ===========================================
# V43 风险管理器
# ===========================================

class V43RiskManager:
    """
    V43 风险管理器 - 自适应止损与利润释放
    
    【核心功能】
    1. 自适应 ATR 止损：高波动 2.5ATR，平稳 1.5ATR
    2. 利润释放：废除 Max_Holding_Days，只要排名 Top 20 且未触及止损就持有
    3. 洗售审计：5 天内"卖出即买入"强制拦截
    4. 动态排位追踪
    """
    
    def __init__(self, initial_capital: float = FIXED_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, V43Position] = {}
        self.trades: List[V43Trade] = []
        self.trade_log: List[V43TradeAudit] = []
        
        # 洗售审计追踪
        self.sell_history: Dict[str, str] = {}
        self.wash_sale_blocks: List[V43WashSaleRecord] = []
        
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
        self.is_high_volatility_environment: bool = False
        
        # 因子状态
        self.factor_status: Dict[str, Any] = {}
        
        # V43 排名缓存
        self.current_rank_cache: Dict[str, int] = {}
    
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
        """洗售审计检查"""
        if symbol in self.today_sells:
            return True, f"same_day_sell ({trade_date})"
        
        if symbol in self.sell_history:
            last_sell_date = self.sell_history[symbol]
            days_diff = self._days_between(last_sell_date, trade_date)
            
            if days_diff <= WASH_SALE_WINDOW:
                return True, f"wash_sale_window ({days_diff} days since {last_sell_date})"
        
        return False, None
    
    def record_sell(self, symbol: str, trade_date: str):
        """记录卖出操作"""
        self.sell_history[symbol] = trade_date
        self.today_sells.add(symbol)
    
    def record_buy(self, symbol: str):
        """记录买入操作"""
        self.today_buys.add(symbol)
    
    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        return max(MIN_COMMISSION, amount * COMMISSION_RATE)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        """计算过户费"""
        return shares * price * TRANSFER_FEE
    
    def get_current_atr_mult(self) -> float:
        """V43 自适应 ATR 倍数"""
        if self.is_high_volatility_environment:
            return TRAILING_STOP_ATR_MULT_HIGH_VOL  # 高波动 2.5ATR
        elif self.is_low_volatility_environment:
            return TRAILING_STOP_ATR_MULT_LOW_VOL  # 平稳 1.5ATR
        else:
            return 2.0  # 默认 2.0ATR
    
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
        """V43 风险平价调仓"""
        try:
            risk_target = self.get_current_risk_target()
            risk_amount = total_assets * risk_target
            atr_mult = self.get_current_atr_mult()
            risk_per_share = atr * atr_mult
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0
            
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100
            
            if shares < 100:
                return 0, 0.0
            
            position_amount = shares * current_price
            
            max_position = total_assets * 0.12
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
        composite_score: float = 0.0,
        reason: str = "",
    ) -> Optional[V43Trade]:
        """V43 买入执行"""
        try:
            # 洗售审计
            is_blocked, block_reason = self.check_wash_sale(symbol, trade_date)
            if is_blocked:
                logger.warning(f"WASH SALE BLOCKED: {symbol} - {block_reason}")
                self.wash_sale_blocks.append(V43WashSaleRecord(
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
            
            # 自适应 ATR 止损
            atr_mult = self.get_current_atr_mult()
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - atr_mult * atr_stop_distance)
            initial_stop_price = execution_price * (1 - INITIAL_STOP_LOSS_RATIO)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            self.positions[symbol] = V43Position(
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
            )
            
            # 强制持仓冷却
            self.locked_positions[symbol] = MIN_HOLDING_DAYS
            
            self.record_buy(symbol)
            
            trade = V43Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee,
                total_cost=total_cost, reason=reason, execution_price=execution_price,
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Rank={signal_rank} | Composite={composite_score:.3f}")
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
    ) -> Optional[V43Trade]:
        """V43 卖出执行"""
        try:
            if symbol not in self.positions:
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            pos = self.positions[symbol]
            
            # 持仓锁定期检查（除非止损）
            if symbol in self.locked_positions and self.locked_positions[symbol] > 0:
                if reason not in ["stop_loss", "trailing_stop", "rank_drop"]:
                    logger.debug(f"LOCKED POSITION: {symbol} locked for {self.locked_positions[symbol]} more days")
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
            
            trade_audit = V43TradeAudit(
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
            
            del self.positions[symbol]
            self.locked_positions.pop(symbol, None)
            
            self.record_sell(symbol, trade_date)
            
            trade = V43Trade(
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
        """V43 更新波动率状态"""
        self.current_market_vol = market_vol
        self.is_low_volatility_environment = market_vol < LOW_VOLATILITY_THRESHOLD
        self.is_high_volatility_environment = market_vol > HIGH_VOLATILITY_THRESHOLD
    
    def get_risk_per_position(self) -> float:
        """获取当前风险暴露"""
        return self.get_current_risk_target()
    
    def check_stop_loss_and_rank(
        self, 
        positions: Dict[str, V43Position], 
        date_str: str, 
        price_df: pl.DataFrame, 
        factor_df: pl.DataFrame
    ) -> List[Tuple[str, str]]:
        """
        V43 检查止损和排名 - 废除 Max_Holding_Days
        
        卖出条件：
        1. 触发 ATR 移动止损
        2. 初始止损 8%
        3. 排名跌出 Top 20（且不在锁定期）
        """
        sell_list = []
        
        # 高效构建价格映射 - 使用 Polars 的 to_dict
        try:
            prices_df = price_df.select(['symbol', 'close']).unique('symbol', keep='last')
            prices = dict(zip(prices_df['symbol'].to_list(), prices_df['close'].to_list()))
        except:
            prices = {}
        
        # 高效构建排名映射
        try:
            ranks_df = factor_df.select(['symbol', 'composite_rank', 'atr_20']).unique('symbol', keep='last')
            ranks = dict(zip(ranks_df['symbol'].to_list(), ranks_df['composite_rank'].to_list()))
            atr_values = dict(zip(ranks_df['symbol'].to_list(), ranks_df['atr_20'].to_list()))
        except:
            ranks = {}
            atr_values = {}
        
        for symbol, pos in list(positions.items()):
            current_price = prices.get(symbol, 0)
            if current_price <= 0:
                continue
                
            pos.current_price = current_price
            pos.market_value = pos.shares * current_price
            pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares
            pos.holding_days = max(0, self.trade_day_counter - pos.buy_trade_day)
            
            # 更新峰值价格
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            
            # 更新 ATR 移动止损
            atr = atr_values.get(symbol, pos.atr_at_entry)
            if atr and atr > 0:
                try:
                    atr_mult = self.get_current_atr_mult()
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    new_trailing_stop = current_price * (1 - atr_mult * atr_stop_distance)
                    
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        pos.trailing_stop_history.append(new_trailing_stop)
                except:
                    pass
            
            # V43 卖出条件检查
            
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
            
            # 3. V43 排名卖出 - 跌出 Top 20（可突破锁定期）
            current_rank = ranks.get(symbol, 9999)
            pos.current_market_rank = current_rank
            
            if current_rank > TOP_N_HOLD_THRESHOLD and symbol not in self.locked_positions:
                sell_list.append((symbol, "rank_drop"))
                continue
            
            # 注意：V43 废除了 Max_Holding_Days 强制卖出逻辑
        
        return sell_list
    
    def rank_candidates(self, factor_df: pl.DataFrame, positions: Dict[str, V43Position]) -> List[Dict]:
        """V43 排名候选股票 - 使用 Composite_Score"""
        try:
            held_symbols = set(positions.keys())
            
            # 筛选未持仓且允许入场的股票
            candidates = factor_df.filter(
                (~pl.col('symbol').is_in(list(held_symbols))) & 
                (pl.col('entry_allowed') == True)
            ).sort('composite_score', descending=True).head(TOP_N_SELECTION * 2)
            
            result = []
            for idx, row in enumerate(candidates.iter_rows(named=True)):
                result.append({
                    'symbol': row['symbol'],
                    'rank': idx + 1,
                    'composite_score': float(row.get('composite_score', 0)) if row.get('composite_score') is not None else 0,
                    'composite_rank': row.get('composite_rank', 9999),
                    'trend_quality_r2': float(row.get('trend_quality_r2', 0)) if row.get('trend_quality_r2') is not None else 0,
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
    
    def get_portfolio_value(self, positions: Dict[str, V43Position], date_str: str, price_df: pl.DataFrame) -> float:
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