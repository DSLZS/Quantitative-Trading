"""
V63 Core Module - VCP 波动率收缩与大盘熔断协议

【V63 核心改进 - 最高优先级】

1. 拒绝金融自杀（硬约束）
   ✅ 严禁放宽条件：如果全市场没有符合条件的股票，必须空仓！
   ✅ 严禁为了增加交易次数而私自修改 selection_percentile
   ✅ 交易频率管控：月均交易次数严禁超过 15 次

2. VCP (Volatility Contraction Pattern) 核心算法
   ✅ 趋势初筛：Close > MA50 > MA150 > MA200（确保个股处于大级别上升通道）
   ✅ 收缩确认：过去 10 个交易日的最高价与最低价振幅连续收窄，且当前振幅 < 8%
   ✅ 量能枯竭：成交量连续 3 日萎缩至均线的 60% 以下
   ✅ 突破买入：放量突破近 5 日最高价时，以 min(Trigger, Next_Open) 买入

3. 系统性风险保险丝
   ✅ 大盘择时：计算全市场 Close > MA20 的股票占比，若占比 < 25%，强制全局空仓
   ✅ 时间止损：买入 5 天不盈利强制离场

4. 工程一致性
   ✅ 维持与 V62 相同的 Schema 结构
   ✅ 所有常量定义在顶部，严禁 ImportError

作者：量化系统
版本：V63.0
日期：2026-03-23
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V63 配置常量 - VCP 波动率收缩策略
# ===========================================

# 基础配置
V63_INITIAL_CAPITAL = 100000.00
V63_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V63 频率熔断（硬约束）
V63_MONTHLY_TRADE_LIMIT = 15  # 月均交易次数上限（死命令！）
V63_WEEKLY_TRADE_LIMIT = 4    # 每周最多开仓 4 只
V63_GLOBAL_TRADE_LIMIT = 150  # 全场交易次数限制

# V63 数据预加载配置
V63_WARMUP_PERIOD = 250  # 预加载 250 天数据，确保 MA200 计算完整
V63_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V63 大盘择时熔断（系统性风险保险丝）
V63_MARKET_BREADTH_WINDOW = 20  # 计算 MA20 突破比例
V63_MARKET_BREADTH_THRESHOLD = 0.25  # 若占比 < 25%，强制空仓（死命令）

# V63 趋势初筛 - 大级别上升通道
V63_TREND_MA50_ABOVE_MA150 = True  # MA50 > MA150
V63_TREND_MA150_ABOVE_MA200 = True  # MA150 > MA200
V63_TREND_CLOSE_ABOVE_MA50 = True  # Close > MA50

# V63 VCP 收缩确认
V63_VCP_WINDOW = 10  # 过去 10 个交易日
V63_VCP_MAX_AMPLITUDE = 0.08  # 当前振幅 < 8%
V63_VCP_CONTRACTION_DAYS = 3  # 至少连续 3 天振幅收窄

# V63 量能枯竭
V63_VOLUME_MA_PERIOD = 10  # 10 日均量
V63_VOLUME_DRY_RATIO = 0.60  # 成交量 < 均量的 60%
V63_VOLUME_DRY_CONSECUTIVE_DAYS = 3  # 连续 3 日萎缩

# V63 突破买入
V63_BREAKOUT_WINDOW = 5  # 突破近 5 日最高价
V63_BREAKOUT_VOLUME_RATIO = 1.5  # 放量突破：成交量 > 均量的 1.5 倍

# V63 费率配置 - 总计 0.2%
V63_COMMISSION_RATE = 0.0003  # 佣金万 3
V63_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V63_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V63_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V63_STAMP_DUTY = 0.0005  # 印花税 0.05%
V63_TRANSFER_FEE = 0.00001  # 过户费 0.001%

# V63 摩擦成本总计（估算）
V63_FRICTION_COST = 0.002  # 0.2% 总计

# V63 止损配置
V63_HARD_STOP_LOSS_RATIO = 0.08  # 硬止损 8%
V63_HARD_STOP_LOSS_ATR_MULT = 2.5  # ATR 止损 2.5 倍
V63_HARD_STOP_LOSS_MODE = "ratio"  # 使用固定比例止损

# V63 时间止损（新！）
V63_TIME_STOP_DAYS = 5  # 买入 5 天不盈利强制离场
V63_TIME_STOP_ENABLED = True

# V63 止盈配置
V63_TRAILING_STOP_ENABLED = True
V63_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%
V63_PROFIT_TARGET_RATIO = 0.15  # 目标盈利 15%

# V63 仓位管理
V63_RISK_TARGET_PER_POSITION = 0.01  # 每仓风险 1%
V63_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V63 洗售审计
V63_WASH_SALE_WINDOW = 5  # 5 天洗售窗口

# V63 选股排名
V63_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票（严禁放宽！）


# ===========================================
# V63 数据类定义
# ===========================================

@dataclass
class V63Position:
    """V63 持仓记录"""
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
    
    # V63 VCP 状态
    is_vcp_entry: bool = False
    vcp_amplitude: float = 0.0
    vcp_contraction_days: int = 0
    volume_dry_days: int = 0
    breakout_confirmed: bool = False
    
    # V63 趋势状态
    ma_trend_aligned: bool = False  # MA50 > MA150 > MA200
    close_above_ma50: bool = False
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    profit_target_triggered: bool = False
    time_stop_triggered: bool = False
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0
    
    # 行业数据
    industry_name: str = ""


@dataclass
class V63Trade:
    """V63 交易记录"""
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
    
    # V63 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    min_trigger_open: float = 0.0
    slippage_applied: float = 0.0


@dataclass
class V63TradeAudit:
    """V63 交易审计记录"""
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
    
    # V63 VCP 状态
    is_vcp_entry: bool = False
    vcp_amplitude: float = 0.0
    vcp_contraction_days: int = 0
    volume_dry_days: int = 0
    breakout_confirmed: bool = False
    
    # V63 趋势状态
    ma_trend_aligned: bool = False
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V63Signal:
    """V63 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str  # 'buy' or 'sell'
    signal_score: float
    signal_rank: int
    composite_score: float
    
    # V63 VCP 状态
    is_vcp_entry: bool = False
    vcp_amplitude: float = 0.0
    vcp_contraction_days: int = 0
    volume_dry_days: int = 0
    breakout_confirmed: bool = False
    
    # V63 趋势状态
    ma_trend_aligned: bool = False
    close_above_ma50: bool = False
    
    # 价格数据
    close_price: float = 0.0
    next_open_price: float = 0.0
    ma50_price: float = 0.0
    ma150_price: float = 0.0
    ma200_price: float = 0.0


@dataclass
class V63WashSaleRecord:
    """V63 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V63MarketRegime:
    """V63 大盘状态"""
    trade_date: str
    market_breadth: float = 0.0  # Close > MA20 的股票占比
    is_safe_period: bool = True  # 是否安全期
    regime_reason: str = ""
    forced_empty: bool = False  # 是否强制空仓


# ===========================================
# V63 Schema 验证函数（最高优先级）
# ===========================================

def validate_factors(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """
    V63 Schema 强一致性验证函数
    
    【死命令】
    - 回测开始前必须验证所有必需列存在
    - 缺少 is_vcp_entry、vcp_amplitude 或 composite_score 立即报错
    - 严禁进入回测死锁
    
    Parameters
    ----------
    df : pl.DataFrame
        待验证的 DataFrame
    
    Returns
    -------
    Tuple[bool, List[str]]
        (是否通过验证，缺失列列表)
    """
    required_columns = [
        'symbol',
        'trade_date',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'is_vcp_entry',
        'vcp_amplitude',
        'composite_score',
        'ma_trend_aligned',
    ]
    
    missing_columns = []
    
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        error_msg = f"V63 Schema 验证失败！缺失列：{missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 额外验证：检查关键因子是否有有效值
    null_check_cols = ['is_vcp_entry', 'vcp_amplitude', 'composite_score']
    null_issues = []
    
    for col in null_check_cols:
        if col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                null_issues.append(f"{col} 有 {null_count} 个空值")
    
    if null_issues:
        logger.warning(f"V63 Schema 警告：{null_issues}")
    
    logger.info("V63 Schema 验证通过 ✅")
    return True, []


# ===========================================
# V63 DataManager - 数据获取与预处理
# ===========================================

class V63DataManager:
    """
    V63 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票数据
    2. 预加载逻辑：从 start_date - warmup_period 开始，确保 MA200 计算完整
    3. 数据清洗与格式化
    4. 样本量验证：必须 > 500 只股票
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V63_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V63_MIN_SAMPLE_SIZE)
        self._data_cache: Dict[str, pl.DataFrame] = {}
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """
        加载股票数据
        
        Parameters
        ----------
        start_date : str
            回测开始日期（如 2024-01-01）
        end_date : str
            回测结束日期
        symbols : Optional[List[str]]
            股票代码列表，None 表示加载全部
        
        Returns
        -------
        pl.DataFrame
            股票数据 DataFrame
        """
        # V63 核心：预加载逻辑 - 数据加载必须从 start_date - warmup_period 开始
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V63 数据加载：回测区间 [{start_date}, {end_date}]，预加载区间 [{actual_start_date}, {end_date}]")
        
        # 尝试从缓存加载
        cache_key = f"{actual_start_date}_{end_date}"
        if cache_key in self._data_cache:
            logger.info("V63 使用缓存数据")
            return self._data_cache[cache_key]
        
        # 从数据库加载
        if self.db is None:
            raise ValueError("V63 DataManager: 数据库连接未初始化")
        
        try:
            # 构建 SQL 查询
            if symbols:
                symbol_list = "','".join(symbols)
                symbol_filter = f"AND symbol IN ('{symbol_list}')"
            else:
                symbol_filter = ""
            
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount
                FROM stock_daily
                WHERE trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                  {symbol_filter}
                ORDER BY symbol, trade_date
            """
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V63 DataManager: 未加载到任何数据，查询区间 [{actual_start_date}, {end_date}]")
            
            # 数据验证
            self._validate_data(df, start_date)
            
            # 缓存数据
            self._data_cache[cache_key] = df
            
            logger.info(f"V63 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V63 DataManager 加载数据失败：{e}")
            raise
    
    def load_index_data(self, start_date: str, end_date: str,
                        index_symbol: str = "000001.SH") -> pl.DataFrame:
        """
        加载指数数据（用于计算市场广度）
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        index_symbol : str
            指数代码，默认上证指数
        
        Returns
        -------
        pl.DataFrame
            指数数据
        """
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        try:
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume
                FROM index_daily
                WHERE symbol = '{index_symbol}'
                  AND trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V63: 未加载到指数数据 {index_symbol}")
                return pl.DataFrame(schema={
                    'symbol': pl.Utf8,
                    'trade_date': pl.Utf8,
                    'open': pl.Float64,
                    'high': pl.Float64,
                    'low': pl.Float64,
                    'close': pl.Float64,
                    'volume': pl.Float64
                })
            
            return df
            
        except Exception as e:
            logger.warning(f"V63 加载指数数据失败：{e}")
            return pl.DataFrame(schema={
                'symbol': pl.Utf8,
                'trade_date': pl.Utf8,
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'volume': pl.Float64
            })
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name, industry_mv_ratio
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            df = self.db.read_sql(query)
            return df
            
        except Exception as e:
            logger.warning(f"V63 加载行业数据失败：{e}")
            return pl.DataFrame(schema={
                'symbol': pl.Utf8,
                'trade_date': pl.Utf8,
                'industry_name': pl.Utf8,
                'industry_mv_ratio': pl.Float64
            })
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        """计算预加载开始日期"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2023-01-01"
    
    def _validate_data(self, df: pl.DataFrame, start_date: str):
        """验证数据质量"""
        # 检查必需列
        required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"V63 DataManager: 缺失列 {missing}")
        
        # 检查样本量
        unique_symbols = df['symbol'].n_unique()
        if unique_symbols < self.min_sample_size:
            logger.warning(f"V63 警告：样本量 {unique_symbols} < {self.min_sample_size}")
        
        # 检查数据范围
        actual_start = df['trade_date'].min()
        actual_end = df['trade_date'].max()
        logger.info(f"V63 数据范围：[{actual_start}, {actual_end}]")
        
        # 检查空值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            null_count = df[col].null_count()
            if null_count > 0:
                logger.warning(f"V63 警告：{col} 有 {null_count} 个空值")
    
    def clear_cache(self):
        """清除缓存"""
        self._data_cache.clear()
        logger.info("V63 DataManager 缓存已清除")


# ===========================================
# V63 AlphaCenter - VCP 信号生成
# ===========================================

class V63AlphaCenter:
    """
    V63 AlphaCenter - VCP 波动率收缩信号生成
    
    【核心逻辑】
    1. 趋势初筛：Close > MA50 > MA150 > MA200（确保大级别上升通道）
    2. VCP 收缩：过去 10 日振幅连续收窄，当前振幅 < 8%
    3. 量能枯竭：成交量连续 3 日 < 均量 60%
    4. 突破确认：放量突破近 5 日最高价
    5. 大盘择时：市场广度 < 25% 强制空仓
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 趋势配置
        self.trend_aligned = self.config.get('trend_aligned', True)
        
        # VCP 配置
        self.vcp_window = self.config.get('vcp_window', V63_VCP_WINDOW)
        self.vcp_max_amplitude = self.config.get('vcp_max_amplitude', V63_VCP_MAX_AMPLITUDE)
        self.vcp_contraction_days = self.config.get('vcp_contraction_days', V63_VCP_CONTRACTION_DAYS)
        
        # 量能配置
        self.volume_ma_period = self.config.get('volume_ma_period', V63_VOLUME_MA_PERIOD)
        self.volume_dry_ratio = self.config.get('volume_dry_ratio', V63_VOLUME_DRY_RATIO)
        self.volume_dry_consecutive_days = self.config.get('volume_dry_consecutive_days', V63_VOLUME_DRY_CONSECUTIVE_DAYS)
        
        # 突破配置
        self.breakout_window = self.config.get('breakout_window', V63_BREAKOUT_WINDOW)
        self.breakout_volume_ratio = self.config.get('breakout_volume_ratio', V63_BREAKOUT_VOLUME_RATIO)
        
        # 大盘择时
        self.market_breadth_threshold = self.config.get('market_breadth_threshold', V63_MARKET_BREADTH_THRESHOLD)
    
    def compute_signals(self, df: pl.DataFrame, 
                        index_data: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
        
        Parameters
        ----------
        df : pl.DataFrame
            股票数据
        index_data : Optional[pl.DataFrame]
            指数数据
        
        Returns
        -------
        Tuple[pl.DataFrame, Dict[str, Any]]
            (包含信号列的 DataFrame，状态信息)
        """
        try:
            result = df.clone()
            
            # 数据类型转换
            result = result.with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            ])
            
            status = {'factors_computed': []}
            
            # 1. 计算均线系统（MA50/150/200）
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma50', 'ma150', 'ma200'])
            
            # 2. 计算趋势对齐
            result = self._compute_trend_alignment(result)
            status['factors_computed'].append('trend_alignment')
            
            # 3. 计算 VCP 收缩
            result = self._compute_vcp_contraction(result)
            status['factors_computed'].append('vcp_contraction')
            
            # 4. 计算量能枯竭
            result = self._compute_volume_dry(result)
            status['factors_computed'].append('volume_dry')
            
            # 5. 计算突破信号
            result = self._compute_breakout_signal(result)
            status['factors_computed'].append('breakout_signal')
            
            # 6. 计算综合评分
            result = self._compute_composite_score(result)
            status['factors_computed'].append('composite_score')
            
            # 7. Schema 验证
            validate_factors(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V63 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算均线系统（MA50/150/200）"""
        result = df.clone()
        
        ma50 = pl.col('close').rolling_mean(window_size=50).over('symbol')
        ma150 = pl.col('close').rolling_mean(window_size=150).over('symbol')
        ma200 = pl.col('close').rolling_mean(window_size=200).over('symbol')
        
        return result.with_columns([
            ma50.alias('ma50'),
            ma150.alias('ma150'),
            ma200.alias('ma200')
        ])
    
    def _compute_trend_alignment(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V63 核心：趋势对齐检测
        
        【核心逻辑】
        - Close > MA50 > MA150 > MA200
        - 确保个股处于大级别上升通道
        """
        result = df.clone()
        
        close = pl.col('close')
        ma50 = pl.col('ma50')
        ma150 = pl.col('ma150')
        ma200 = pl.col('ma200')
        
        # Close > MA50
        close_above_ma50 = close > ma50
        
        # MA50 > MA150
        ma50_above_ma150 = ma50 > ma150
        
        # MA150 > MA200
        ma150_above_ma200 = ma150 > ma200
        
        # 趋势完美对齐
        ma_trend_aligned = close_above_ma50 & ma50_above_ma150 & ma150_above_ma200
        
        return result.with_columns([
            close_above_ma50.alias('close_above_ma50'),
            ma50_above_ma150.alias('ma50_above_ma150'),
            ma150_above_ma200.alias('ma150_above_ma200'),
            ma_trend_aligned.alias('ma_trend_aligned')
        ])
    
    def _compute_vcp_contraction(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V63 核心：VCP 波动率收缩
        
        【核心逻辑】
        1. 计算过去 10 日每日振幅 = (High - Low) / Close
        2. 检测振幅是否连续收窄
        3. 当前振幅 < 8%
        """
        result = df.clone()
        vcp_window = self.vcp_window
        
        # 计算每日振幅
        daily_amplitude = (pl.col('high') - pl.col('low')) / (pl.col('close') + self.EPSILON)
        
        # 滚动计算平均振幅（过去 N 日）
        avg_amplitude = daily_amplitude.rolling_mean(window_size=vcp_window).over('symbol')
        
        # 计算振幅变化率（今日振幅 vs 前一日振幅）
        prev_amplitude = daily_amplitude.shift(1).over('symbol')
        amplitude_change = daily_amplitude - prev_amplitude
        
        # 振幅收窄检测
        amplitude_narrowing = amplitude_change < 0
        
        # 连续收窄天数检测
        # 连续 3 天收窄
        consec_1 = amplitude_narrowing
        consec_2 = amplitude_narrowing.shift(1).over('symbol')
        consec_3 = amplitude_narrowing.shift(2).over('symbol')
        
        consecutive_narrowing = consec_1 & consec_2 & consec_3
        
        # 当前振幅 < 8%
        current_amplitude_low = daily_amplitude < self.vcp_max_amplitude
        
        # VCP 收缩信号
        is_vcp_contraction = consecutive_narrowing & current_amplitude_low
        
        # 计算连续收窄天数
        vcp_contraction_days = pl.when(consec_1 & consec_2 & consec_3) \
            .then(3) \
            .otherwise(pl.when(consec_1 & consec_2).then(2).otherwise(0))
        
        return result.with_columns([
            daily_amplitude.alias('daily_amplitude'),
            avg_amplitude.alias('avg_amplitude'),
            amplitude_change.alias('amplitude_change'),
            amplitude_narrowing.alias('amplitude_narrowing'),
            consecutive_narrowing.alias('consecutive_narrowing'),
            current_amplitude_low.alias('current_amplitude_low'),
            vcp_contraction_days.alias('vcp_contraction_days'),
            is_vcp_contraction.alias('is_vcp_contraction')
        ])
    
    def _compute_volume_dry(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V63 核心：量能枯竭检测
        
        【核心逻辑】
        - 成交量连续 3 日 < 10 日均量的 60%
        """
        result = df.clone()
        
        vol_ma = pl.col('volume').rolling_mean(window_size=self.volume_ma_period).over('symbol')
        
        # 成交量萎缩检测
        is_volume_dry = pl.col('volume') < (vol_ma * self.volume_dry_ratio)
        
        # 连续萎缩天数检测
        dry_1 = is_volume_dry
        dry_2 = is_volume_dry.shift(1).over('symbol')
        dry_3 = is_volume_dry.shift(2).over('symbol')
        
        consecutive_dry = dry_1 & dry_2 & dry_3
        
        # 计算连续干涸天数
        volume_dry_days = pl.when(dry_1 & dry_2 & dry_3) \
            .then(3) \
            .otherwise(pl.when(dry_1 & dry_2).then(2).otherwise(pl.when(dry_1).then(1).otherwise(0)))
        
        # 成交量比率
        volume_ratio = pl.col('volume') / (vol_ma + self.EPSILON)
        
        return result.with_columns([
            vol_ma.alias('vol_ma10'),
            is_volume_dry.alias('is_volume_dry'),
            consecutive_dry.alias('consecutive_volume_dry'),
            volume_dry_days.alias('volume_dry_days'),
            volume_ratio.alias('volume_ratio')
        ])
    
    def _compute_breakout_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V63 核心：突破信号确认
        
        【核心逻辑】
        1. 价格突破近 5 日最高价
        2. 成交量放大至均量 1.5 倍以上
        """
        result = df.clone()
        
        # 近 5 日最高价
        high_5d = pl.col('high').rolling_max(window_size=self.breakout_window).over('symbol')
        prev_high_5d = high_5d.shift(1).over('symbol')
        
        # 突破检测：今日最高价 > 前一日计算的 5 日最高价
        breakout_price = pl.col('high') > prev_high_5d
        
        # 放量检测
        vol_ma = pl.col('vol_ma10') if 'vol_ma10' in df.columns else \
                 pl.col('volume').rolling_mean(window_size=self.volume_ma_period).over('symbol')
        breakout_volume = pl.col('volume') > (vol_ma * self.breakout_volume_ratio)
        
        # 突破确认
        breakout_confirmed = breakout_price & breakout_volume
        
        return result.with_columns([
            high_5d.alias('high_5d'),
            prev_high_5d.alias('prev_high_5d'),
            breakout_price.alias('breakout_price'),
            breakout_volume.alias('breakout_volume'),
            breakout_confirmed.alias('breakout_confirmed')
        ])
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V63 综合评分计算
        
        【核心逻辑】
        1. 趋势对齐 bonus（核心）
        2. VCP 收缩 bonus
        3. 量能枯竭 bonus
        4. 突破确认 bonus
        """
        result = df.clone()
        
        # 趋势对齐 bonus（最高权重）
        trend_bonus = pl.when(pl.col('ma_trend_aligned')) \
            .then(pl.lit(0.40)) \
            .otherwise(pl.lit(0.0))
        
        # VCP 收缩 bonus
        vcp_bonus = pl.when(pl.col('is_vcp_contraction')) \
            .then(pl.lit(0.25)) \
            .otherwise(pl.lit(0.0))
        
        # 量能枯竭 bonus
        volume_bonus = pl.when(pl.col('consecutive_volume_dry')) \
            .then(pl.lit(0.15)) \
            .otherwise(pl.lit(0.0))
        
        # 突破确认 bonus
        breakout_bonus = pl.when(pl.col('breakout_confirmed')) \
            .then(pl.lit(0.20)) \
            .otherwise(pl.lit(0.0))
        
        # 基础动量得分
        momentum = (pl.col('close') - pl.col('ma200')) / (pl.col('ma200') + self.EPSILON)
        momentum_rank = momentum.rank('ordinal', descending=True).over('trade_date')
        n_stocks = pl.col('symbol').count().over('trade_date')
        momentum_score = (1.0 - momentum_rank.cast(pl.Float64) / n_stocks.cast(pl.Float64)) * 0.3
        
        # 综合评分
        composite_score = trend_bonus + vcp_bonus + volume_bonus + breakout_bonus + momentum_score
        
        # 排名计算
        composite_rank = composite_score.rank('ordinal', descending=True).over('trade_date')
        composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / (n_stocks.cast(pl.Float64) + self.EPSILON))
        
        # V63 综合买入信号（死命令：所有条件必须满足）
        # 1. 趋势对齐
        # 2. VCP 收缩
        # 3. 量能枯竭
        # 4. 突破确认
        core_condition = pl.col('ma_trend_aligned') & \
                        pl.col('is_vcp_contraction') & \
                        pl.col('consecutive_volume_dry') & \
                        pl.col('breakout_confirmed')
        
        # 基础买入信号
        buy_signal = core_condition
        
        # VCP 入场信号（用于审计）
        is_vcp_entry = core_condition
        
        return result.with_columns([
            composite_score.alias('composite_score'),
            composite_rank.cast(pl.Int64).alias('composite_rank'),
            composite_percentile.alias('composite_percentile'),
            buy_signal.alias('buy_signal'),
            is_vcp_entry.alias('is_vcp_entry')
        ])
    
    def compute_market_breadth(self, df: pl.DataFrame) -> V63MarketRegime:
        """
        V63 核心：计算市场广度（大盘择时）
        
        【核心逻辑】
        - 计算全市场 Close > MA20 的股票占比
        - 若占比 < 25%，强制全局空仓
        """
        # 计算 MA20
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        df_with_ma20 = df.with_columns(ma20.alias('ma20'))
        
        # 获取最新交易日
        latest_date = df_with_ma20['trade_date'].max()
        
        # 过滤最新交易日数据
        latest_df = df_with_ma20.filter(pl.col('trade_date') == latest_date)
        
        if latest_df.is_empty():
            return V63MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        # 计算 Close > MA20 的股票数量
        above_ma20_count = latest_df.filter(pl.col('close') > pl.col('ma20')).height
        total_count = latest_df.height
        
        if total_count == 0:
            return V63MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        market_breadth = above_ma20_count / total_count
        is_safe = market_breadth >= self.market_breadth_threshold
        
        regime = V63MarketRegime(
            trade_date=latest_date,
            market_breadth=market_breadth,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"市场广度={market_breadth*100:.1f}%, 阈值={self.market_breadth_threshold*100:.0f}%"
        )
        
        logger.info(f"V63 市场广度：{market_breadth*100:.1f}% - {'安全' if is_safe else '危险 - 强制空仓'}")
        
        return regime
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str,
                         market_regime: Optional[V63MarketRegime] = None) -> List[V63Signal]:
        """
        生成交易信号
        
        Parameters
        ----------
        df : pl.DataFrame
            包含因子值的 DataFrame
        trade_date : str
            交易日期
        market_regime : Optional[V63MarketRegime]
            大盘状态
        
        Returns
        -------
        List[V63Signal]
            交易信号列表
        """
        signals = []
        
        # 大盘危险，强制空仓
        if market_regime and not market_regime.is_safe_period:
            logger.warning(f"V63: 大盘危险 ({market_regime.regime_reason})，强制空仓！")
            return signals
        
        try:
            # 过滤当日数据
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            # 过滤买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            # 按综合评分排序
            buy_df = buy_df.sort('composite_score', descending=True)
            
            # 只取前 15% 的股票（死命令！）
            top_n = max(1, int(buy_df.height * V63_SELECTION_PERCENTILE))
            buy_df = buy_df.head(top_n)
            
            # 生成信号
            for row in buy_df.iter_rows(named=True):
                signal = V63Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    signal_rank=row.get('composite_rank', 9999),
                    composite_score=row.get('composite_score', 0.0),
                    is_vcp_entry=row.get('is_vcp_entry', False),
                    vcp_amplitude=row.get('daily_amplitude', 0.0),
                    vcp_contraction_days=row.get('vcp_contraction_days', 0),
                    volume_dry_days=row.get('volume_dry_days', 0),
                    breakout_confirmed=row.get('breakout_confirmed', False),
                    ma_trend_aligned=row.get('ma_trend_aligned', False),
                    close_above_ma50=row.get('close_above_ma50', False),
                    close_price=row.get('close', 0.0),
                    ma50_price=row.get('ma50', 0.0),
                    ma150_price=row.get('ma150', 0.0),
                    ma200_price=row.get('ma200', 0.0)
                )
                signals.append(signal)
            
            logger.info(f"V63 生成 {len(signals)} 个 VCP 买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V63 生成信号失败：{e}")
        
        return signals


# ===========================================
# V63 TradeExec - 真实成交执行
# ===========================================

class V63TradeExec:
    """
    V63 TradeExec - 真实成交执行
    
    【核心功能】
    1. min(Trigger, Open) 成交规则
    2. 手续费 + 滑点总计 0.2% 扣除
    3. 持仓管理
    4. 时间止损（5 天不盈利强制离场）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', V63_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V63_MAX_POSITIONS)
        self.commission_rate = self.config.get('commission_rate', V63_COMMISSION_RATE)
        self.min_commission = self.config.get('min_commission', V63_MIN_COMMISSION)
        self.slippage_buy = self.config.get('slippage_buy', V63_SLIPPAGE_BUY)
        self.slippage_sell = self.config.get('slippage_sell', V63_SLIPPAGE_SELL)
        self.stamp_duty = self.config.get('stamp_duty', V63_STAMP_DUTY)
        self.transfer_fee = self.config.get('transfer_fee', V63_TRANSFER_FEE)
        self.friction_cost = self.config.get('friction_cost', V63_FRICTION_COST)
        
        # 持仓状态
        self.positions: Dict[str, V63Position] = {}
        self.cash = self.initial_capital
        self.trades: List[V63Trade] = []
        self.wash_sale_records: List[V63WashSaleRecord] = []
        self.sell_history: Dict[str, str] = {}  # symbol -> last sell date
        
        # 频率控制
        self.monthly_trades: Dict[str, int] = {}  # YYYY-MM -> count
        self.weekly_trades: Dict[str, int] = {}   # YYYY-WW -> count
        self.total_trades = 0
    
    def _get_month_key(self, date_str: str) -> str:
        """获取月份键"""
        return date_str[:7]  # YYYY-MM
    
    def _get_week_key(self, date_str: str) -> str:
        """获取周键"""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.year}-W{dt.isocalendar()[1]:02d}"
    
    def _check_trade_limit(self, trade_date: str) -> bool:
        """
        检查交易频率限制
        
        Returns
        -------
        bool
            是否可以继续交易
        """
        month_key = self._get_month_key(trade_date)
        week_key = self._get_week_key(trade_date)
        
        # 检查月度限制
        if self.monthly_trades.get(month_key, 0) >= V63_MONTHLY_TRADE_LIMIT:
            logger.warning(f"V63: 月度交易已达上限 ({self.monthly_trades[month_key]}/{V63_MONTHLY_TRADE_LIMIT})")
            return False
        
        # 检查周度限制
        if self.weekly_trades.get(week_key, 0) >= V63_WEEKLY_TRADE_LIMIT:
            logger.warning(f"V63: 周度交易已达上限 ({self.weekly_trades[week_key]}/{V63_WEEKLY_TRADE_LIMIT})")
            return False
        
        # 检查全局限制
        if self.total_trades >= V63_GLOBAL_TRADE_LIMIT:
            logger.warning(f"V63: 全局交易已达上限 ({self.total_trades}/{V63_GLOBAL_TRADE_LIMIT})")
            return False
        
        return True
    
    def _increment_trade_count(self, trade_date: str):
        """增加交易计数"""
        month_key = self._get_month_key(trade_date)
        week_key = self._get_week_key(trade_date)
        
        self.monthly_trades[month_key] = self.monthly_trades.get(month_key, 0) + 1
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1
        self.total_trades += 1
    
    def execute_buy(self, signal: V63Signal, next_open: float, 
                    trigger_price: float, capital: float) -> Optional[V63Trade]:
        """
        执行买入
        
        Parameters
        ----------
        signal : V63Signal
            买入信号
        next_open : float
            次日开盘价
        trigger_price : float
            触发价格（当日 Close）
        capital : float
            可用资金
        
        Returns
        -------
        Optional[V63Trade]
            成交记录，None 表示未成交
        """
        # V63 核心：min(Trigger, Open) 成交规则
        # 买入时取较小值（确保不追高）
        execution_price = min(trigger_price, next_open)
        
        # 应用滑点
        execution_price = execution_price * (1 + self.slippage_buy)
        
        # 计算可买数量
        max_position_value = capital * V63_MAX_SINGLE_POSITION_PCT
        shares = int(max_position_value / execution_price / 100) * 100
        
        if shares <= 0:
            return None
        
        # 计算费用
        amount = shares * execution_price
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer_fee = amount * self.transfer_fee
        total_cost = amount + commission + transfer_fee
        
        if total_cost > capital:
            shares = int((capital * 0.95) / execution_price / 100) * 100
            if shares <= 0:
                return None
            amount = shares * execution_price
            commission = max(amount * self.commission_rate, self.min_commission)
            transfer_fee = amount * self.transfer_fee
            total_cost = amount + commission + transfer_fee
        
        # 创建持仓
        position = V63Position(
            symbol=signal.symbol,
            shares=shares,
            avg_cost=execution_price,
            buy_price=execution_price,
            buy_date=signal.trade_date,
            signal_date=signal.trade_date,
            trade_date=signal.trade_date,
            signal_score=signal.signal_score,
            signal_rank=signal.signal_rank,
            composite_score=signal.composite_score,
            is_vcp_entry=signal.is_vcp_entry,
            vcp_amplitude=signal.vcp_amplitude,
            vcp_contraction_days=signal.vcp_contraction_days,
            volume_dry_days=signal.volume_dry_days,
            breakout_confirmed=signal.breakout_confirmed,
            ma_trend_aligned=signal.ma_trend_aligned,
            close_above_ma50=signal.close_above_ma50,
            stop_loss_price=execution_price * (1 - V63_HARD_STOP_LOSS_RATIO),
            trailing_stop_price=execution_price * (1 - V63_TRAILING_STOP_RATIO),
            trigger_price=trigger_price,
            next_open_price=next_open,
            execution_price=execution_price
        )
        
        self.positions[signal.symbol] = position
        
        # 创建成交记录
        trade = V63Trade(
            trade_date=signal.trade_date,
            symbol=signal.symbol,
            side='buy',
            shares=shares,
            price=execution_price,
            amount=amount,
            commission=commission,
            slippage=amount * self.slippage_buy,
            stamp_duty=0,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason='VCP 波动率收缩突破',
            signal_date=signal.trade_date,
            trigger_price=trigger_price,
            next_open_price=next_open,
            min_trigger_open=min(trigger_price, next_open),
            slippage_applied=self.slippage_buy
        )
        
        self.trades.append(trade)
        self.cash -= total_cost
        
        # 增加交易计数
        self._increment_trade_count(signal.trade_date)
        
        logger.info(f"V63 买入成交：{signal.symbol} @ {execution_price:.2f} x {shares}股")
        
        return trade
    
    def execute_sell(self, symbol: str, current_price: float, 
                     trade_date: str, reason: str) -> Optional[V63Trade]:
        """执行卖出"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # 应用滑点
        execution_price = current_price * (1 - self.slippage_sell)
        
        # 计算费用
        shares = position.shares
        amount = shares * execution_price
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        total_cost = commission + stamp_duty + transfer_fee
        
        # 创建成交记录
        trade = V63Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='sell',
            shares=shares,
            price=execution_price,
            amount=amount,
            commission=commission,
            slippage=amount * self.slippage_sell,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason=reason,
            holding_days=position.holding_days,
            signal_date=position.signal_date,
            trigger_price=position.trigger_price,
            next_open_price=position.next_open_price,
            min_trigger_open=min(position.trigger_price, position.next_open_price),
            slippage_applied=self.slippage_sell
        )
        
        self.trades.append(trade)
        self.cash += amount - total_cost
        
        # 记录卖出日期
        self.sell_history[symbol] = trade_date
        
        # 删除持仓
        del self.positions[symbol]
        
        logger.info(f"V63 卖出成交：{symbol} @ {execution_price:.2f} x {shares}股，盈亏：{(execution_price - position.avg_cost) * shares:.2f}")
        
        return trade
    
    def check_exit_conditions(self, symbol: str, current_price: float,
                              trade_date: str) -> Optional[Tuple[bool, str]]:
        """
        检查离场条件（含时间止损）
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        cost_price = position.avg_cost
        
        # 更新最高价和移动止盈
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.peak_profit = (current_price - cost_price) / cost_price
        
        if position.peak_price > 0:
            position.trailing_stop_price = position.peak_price * (1 - V63_TRAILING_STOP_RATIO)
        
        # 1. 硬止损检查
        if current_price <= position.stop_loss_price:
            position.stop_loss_triggered = True
            return True, f"硬止损 (亏损>{V63_HARD_STOP_LOSS_RATIO*100:.1f}%)"
        
        # 2. 移动止盈检查
        if V63_TRAILING_STOP_ENABLED and position.trailing_stop_price > 0:
            if current_price <= position.trailing_stop_price:
                position.trailing_stop_triggered = True
                return True, f"移动止盈 (回撤>{V63_TRAILING_STOP_RATIO*100:.1f}%)"
        
        # 3. 目标止盈检查
        current_profit = (current_price - cost_price) / cost_price
        if current_profit >= V63_PROFIT_TARGET_RATIO:
            position.profit_target_triggered = True
            return True, f"目标止盈 (盈利>{V63_PROFIT_TARGET_RATIO*100:.1f}%)"
        
        # 4. 时间止损检查（V63 新增！）
        if V63_TIME_STOP_ENABLED and position.holding_days >= V63_TIME_STOP_DAYS:
            if current_profit <= 0:
                position.time_stop_triggered = True
                return True, f"时间止损 (持有{position.holding_days}天不盈利)"
        
        return None
    
    def update_positions(self, market_data: Dict[str, Dict[str, float]], 
                         trade_date: str):
        """更新持仓状态"""
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            current_price = data.get('close', 0)
            
            if current_price <= 0:
                continue
            
            position.current_price = current_price
            position.market_value = current_price * position.shares
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            
            # 更新持仓天数
            try:
                buy_date = datetime.strptime(position.buy_date, "%Y-%m-%d")
                current = datetime.strptime(trade_date, "%Y-%m-%d")
                position.holding_days = (current - buy_date).days
            except Exception:
                pass
    
    def check_wash_sale(self, symbol: str, trade_date: str) -> bool:
        """检查洗售规则"""
        if symbol not in self.sell_history:
            return False
        
        last_sell_date = self.sell_history[symbol]
        
        try:
            sell_date = datetime.strptime(last_sell_date, "%Y-%m-%d")
            current = datetime.strptime(trade_date, "%Y-%m-%d")
            days_between = (current - sell_date).days
            
            if days_between <= V63_WASH_SALE_WINDOW:
                wash_record = V63WashSaleRecord(
                    symbol=symbol,
                    sell_date=last_sell_date,
                    blocked_buy_date=trade_date,
                    days_between=days_between
                )
                self.wash_sale_records.append(wash_record)
                return True
        except Exception:
            pass
        
        return False
    
    def get_portfolio_value(self) -> float:
        """获取组合总价值"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    def get_position_count(self) -> int:
        """获取持仓数量"""
        return len(self.positions)
    
    def can_buy_more(self) -> bool:
        """判断是否可以继续买入"""
        return self.get_position_count() < self.max_positions


# ===========================================
# __all__ 导出列表
# ===========================================

__all__ = [
    # 常量
    'V63_INITIAL_CAPITAL',
    'V63_MAX_POSITIONS',
    'V63_MONTHLY_TRADE_LIMIT',
    'V63_WEEKLY_TRADE_LIMIT',
    'V63_GLOBAL_TRADE_LIMIT',
    'V63_WARMUP_PERIOD',
    'V63_MIN_SAMPLE_SIZE',
    'V63_MARKET_BREADTH_THRESHOLD',
    'V63_TREND_MA50_ABOVE_MA150',
    'V63_TREND_MA150_ABOVE_MA200',
    'V63_TREND_CLOSE_ABOVE_MA50',
    'V63_VCP_WINDOW',
    'V63_VCP_MAX_AMPLITUDE',
    'V63_VCP_CONTRACTION_DAYS',
    'V63_VOLUME_MA_PERIOD',
    'V63_VOLUME_DRY_RATIO',
    'V63_VOLUME_DRY_CONSECUTIVE_DAYS',
    'V63_BREAKOUT_WINDOW',
    'V63_BREAKOUT_VOLUME_RATIO',
    'V63_COMMISSION_RATE',
    'V63_MIN_COMMISSION',
    'V63_SLIPPAGE_BUY',
    'V63_SLIPPAGE_SELL',
    'V63_STAMP_DUTY',
    'V63_TRANSFER_FEE',
    'V63_FRICTION_COST',
    'V63_HARD_STOP_LOSS_RATIO',
    'V63_HARD_STOP_LOSS_ATR_MULT',
    'V63_HARD_STOP_LOSS_MODE',
    'V63_TIME_STOP_DAYS',
    'V63_TIME_STOP_ENABLED',
    'V63_TRAILING_STOP_ENABLED',
    'V63_TRAILING_STOP_RATIO',
    'V63_PROFIT_TARGET_RATIO',
    'V63_RISK_TARGET_PER_POSITION',
    'V63_MAX_SINGLE_POSITION_PCT',
    'V63_WASH_SALE_WINDOW',
    'V63_SELECTION_PERCENTILE',
    
    # 函数
    'validate_factors',
    
    # 数据类
    'V63Position',
    'V63Trade',
    'V63TradeAudit',
    'V63Signal',
    'V63WashSaleRecord',
    'V63MarketRegime',
    
    # 核心类
    'V63DataManager',
    'V63AlphaCenter',
    'V63TradeExec',
]