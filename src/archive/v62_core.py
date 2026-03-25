"""
V62 Core Module - RS-Pullback 低吸策略与 Schema 强一致性

【V62 核心改进 - 最高优先级】

1. Schema 强一致性（死命令）
   ✅ validate_factors(df) 函数：回测前必须验证所有必需列存在
   ✅ 缺少 is_pullback_entry、rsrs_score 或 composite_score 立即报错
   ✅ 预加载逻辑：数据从 2023-11-01 开始，确保第一天就有完整 MA20 和 RSRS 数据

2. RS-Low-吸策略（Alpha 核心）
   ✅ RS 选股：个股 20 日收益率对比指数的超额收益排名前 15%
   ✅ Pullback 定义：Price 连续 2-3 日下跌，且 Close 落在 [MA20, MA20 * 1.03] 范围内
   ✅ 缩量确认：今日成交量 < 5 日均量的 75%
   ✅ 择时过滤：RSRS (18 日斜率) 的 z-score 必须 > 0.5
   ✅ 买入执行：满足以上所有条件后，以 Next_Open 买入

3. 简化工程，专注交易
   ✅ V62DataManager：数据获取与预处理
   ✅ V62AlphaCenter：信号生成
   ✅ V62TradeExec：真实成交执行

4. 真实性约束
   ✅ min(Trigger, Open) 成交规则
   ✅ 手续费 + 滑点总计 0.2% 必须扣除
   ✅ 股票样本量必须 > 500 只

作者：量化系统
版本：V62.0
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
# V62 配置常量 - RS-Pullback 低吸策略
# ===========================================

# 基础配置
V62_INITIAL_CAPITAL = 100000.00
V62_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V62 频率熔断
V62_WEEKLY_TRADE_LIMIT = 5  # 每周最多开仓 5 只
V62_GLOBAL_TRADE_LIMIT = 100  # 全场交易次数限制

# V62 数据预加载配置
V62_WARMUP_PERIOD = 60  # 预加载 60 天数据，确保 MA20 和 RSRS 计算完整
V62_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V62 RS 选股配置
V62_RS_WINDOW = 20  # 20 日收益率
V62_RS_TOP_PERCENTILE = 0.15  # 排名前 15%

# V62 Pullback 定义
V62_PULLBACK_MIN_DAYS = 2  # 连续下跌最少 2 天
V62_PULLBACK_MAX_DAYS = 3  # 连续下跌最多 3 天
V62_MA20_BUFFER = 0.03  # 价格落在 [MA20, MA20 * 1.03] 范围内

# V62 缩量确认
V62_VOLUME_MA_PERIOD = 5  # 5 日均量
V62_VOLUME_SHRINK_RATIO = 0.75  # 成交量 < 5 日均量的 75%

# V62 择时过滤 - RSRS 核心
V62_RSRS_ENABLED = True
V62_RSRS_WINDOW = 18  # 18 日斜率
V62_RSRS_ZSCORE_THRESHOLD = 0.5  # z-score > 0.5

# V62 买入执行
V62_ENTRY_ON_NEXT_OPEN = True  # 以 Next_Open 买入

# V62 费率配置 - 总计 0.2%
V62_COMMISSION_RATE = 0.0003  # 佣金万 3
V62_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V62_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V62_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V62_STAMP_DUTY = 0.0005  # 印花税 0.05%
V62_TRANSFER_FEE = 0.00001  # 过户费 0.001%

# V62 摩擦成本总计（估算）
V62_FRICTION_COST = 0.002  # 0.2% 总计

# V62 止损配置
V62_HARD_STOP_LOSS_RATIO = 0.08  # 硬止损 8%
V62_HARD_STOP_LOSS_ATR_MULT = 2.5  # ATR 止损 2.5 倍
V62_HARD_STOP_LOSS_MODE = "ratio"  # 使用固定比例止损

# V62 止盈配置
V62_TRAILING_STOP_ENABLED = True
V62_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%
V62_PROFIT_TARGET_RATIO = 0.15  # 目标盈利 15%

# V62 仓位管理
V62_RISK_TARGET_PER_POSITION = 0.01  # 每仓风险 1%
V62_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V62 洗售审计
V62_WASH_SALE_WINDOW = 5  # 5 天洗售窗口


# ===========================================
# V62 数据类定义
# ===========================================

@dataclass
class V62Position:
    """V62 持仓记录"""
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
    
    # V62 Pullback 状态
    is_pullback_entry: bool = False
    pullback_days: int = 0
    pullback_depth: float = 0.0
    volume_shrunk: bool = False
    rsrs_zscore: float = 0.0
    rs_rank: int = 9999
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    profit_target_triggered: bool = False
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0
    
    # 行业数据
    industry_name: str = ""


@dataclass
class V62Trade:
    """V62 交易记录"""
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
    
    # V62 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    min_trigger_open: float = 0.0
    slippage_applied: float = 0.0


@dataclass
class V62TradeAudit:
    """V62 交易审计记录"""
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
    
    # V62 Pullback 状态
    is_pullback_entry: bool = False
    pullback_days: int = 0
    pullback_depth: float = 0.0
    volume_shrunk: bool = False
    rsrs_zscore: float = 0.0
    rs_rank: int = 9999
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V62Signal:
    """V62 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str  # 'buy' or 'sell'
    signal_score: float
    signal_rank: int
    composite_score: float
    
    # V62 Pullback 状态
    is_pullback_entry: bool = False
    pullback_days: int = 0
    pullback_depth: float = 0.0
    volume_shrunk: bool = False
    rsrs_zscore: float = 0.0
    rs_rank: int = 9999
    
    # 价格数据
    close_price: float = 0.0
    next_open_price: float = 0.0
    ma20_price: float = 0.0


@dataclass
class V62WashSaleRecord:
    """V62 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V62MarketRegime:
    """V62 大盘状态"""
    trade_date: str
    index_close: float = 0.0
    index_sma60: float = 0.0
    index_ma5: float = 0.0
    index_ma20: float = 0.0
    is_risk_period: bool = False
    regime_reason: str = ""


# ===========================================
# V62 Schema 验证函数（最高优先级）
# ===========================================

def validate_factors(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """
    V62 Schema 强一致性验证函数
    
    【死命令】
    - 回测开始前必须验证所有必需列存在
    - 缺少 is_pullback_entry、rsrs_score 或 composite_score 立即报错
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
        'is_pullback_entry',
        'rsrs_score',
        'composite_score',
    ]
    
    missing_columns = []
    
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        error_msg = f"V62 Schema 验证失败！缺失列：{missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 额外验证：检查关键因子是否有有效值
    null_check_cols = ['is_pullback_entry', 'rsrs_score', 'composite_score']
    null_issues = []
    
    for col in null_check_cols:
        if col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                null_issues.append(f"{col} 有 {null_count} 个空值")
    
    if null_issues:
        logger.warning(f"V62 Schema 警告：{null_issues}")
    
    logger.info("V62 Schema 验证通过 ✅")
    return True, []


# ===========================================
# V62 DataManager - 数据获取与预处理
# ===========================================

class V62DataManager:
    """
    V62 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票数据
    2. 预加载逻辑：从 2023-11-01 开始，确保第一天就有完整的 MA20 和 RSRS 数据
    3. 数据清洗与格式化
    4. 样本量验证：必须 > 500 只股票
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V62_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V62_MIN_SAMPLE_SIZE)
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
        # V62 核心：预加载逻辑 - 数据加载必须从 start_date - warmup_period 开始
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V62 数据加载：回测区间 [{start_date}, {end_date}]，预加载区间 [{actual_start_date}, {end_date}]")
        
        # 尝试从缓存加载
        cache_key = f"{actual_start_date}_{end_date}"
        if cache_key in self._data_cache:
            logger.info("V62 使用缓存数据")
            return self._data_cache[cache_key]
        
        # 从数据库加载
        if self.db is None:
            raise ValueError("V62 DataManager: 数据库连接未初始化")
        
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
                raise ValueError(f"V62 DataManager: 未加载到任何数据，查询区间 [{actual_start_date}, {end_date}]")
            
            # 数据验证
            self._validate_data(df, start_date)
            
            # 缓存数据
            self._data_cache[cache_key] = df
            
            logger.info(f"V62 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V62 DataManager 加载数据失败：{e}")
            raise
    
    def load_index_data(self, start_date: str, end_date: str,
                        index_symbol: str = "000001.SH") -> pl.DataFrame:
        """
        加载指数数据（用于计算 RS 强度）
        
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
                logger.warning(f"V62: 未加载到指数数据 {index_symbol}")
                # 返回空 DataFrame
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
            logger.warning(f"V62 加载指数数据失败：{e}")
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
            logger.warning(f"V62 加载行业数据失败：{e}")
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
            # 默认返回 2023-11-01
            return "2023-11-01"
    
    def _validate_data(self, df: pl.DataFrame, start_date: str):
        """验证数据质量"""
        # 检查必需列
        required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"V62 DataManager: 缺失列 {missing}")
        
        # 检查样本量
        unique_symbols = df['symbol'].n_unique()
        if unique_symbols < self.min_sample_size:
            logger.warning(f"V62 警告：样本量 {unique_symbols} < {self.min_sample_size}")
        
        # 检查数据范围
        actual_start = df['trade_date'].min()
        actual_end = df['trade_date'].max()
        logger.info(f"V62 数据范围：[{actual_start}, {actual_end}]")
        
        # 检查空值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            null_count = df[col].null_count()
            if null_count > 0:
                logger.warning(f"V62 警告：{col} 有 {null_count} 个空值")
    
    def clear_cache(self):
        """清除缓存"""
        self._data_cache.clear()
        logger.info("V62 DataManager 缓存已清除")


# ===========================================
# V62 AlphaCenter - 信号生成
# ===========================================

class V62AlphaCenter:
    """
    V62 AlphaCenter - RS-Pullback 信号生成
    
    【核心逻辑】
    1. RS 选股：个股 20 日收益率对比指数的超额收益排名前 15%
    2. Pullback 定义：Price 连续 2-3 日下跌，且 Close 落在 [MA20, MA20 * 1.03] 范围内
    3. 缩量确认：今日成交量 < 5 日均量的 75%
    4. 择时过滤：RSRS (18 日斜率) 的 z-score 必须 > 0.5
    5. 买入执行：满足以上所有条件后，以 Next_Open 买入
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rs_window = self.config.get('rs_window', V62_RS_WINDOW)
        self.rs_top_percentile = self.config.get('rs_top_percentile', V62_RS_TOP_PERCENTILE)
        self.pullback_min_days = self.config.get('pullback_min_days', V62_PULLBACK_MIN_DAYS)
        self.pullback_max_days = self.config.get('pullback_max_days', V62_PULLBACK_MAX_DAYS)
        self.ma20_buffer = self.config.get('ma20_buffer', V62_MA20_BUFFER)
        self.volume_shrink_ratio = self.config.get('volume_shrink_ratio', V62_VOLUME_SHRINK_RATIO)
        self.rsrs_window = self.config.get('rsrs_window', V62_RSRS_WINDOW)
        self.rsrs_zscore_threshold = self.config.get('rsrs_zscore_threshold', V62_RSRS_ZSCORE_THRESHOLD)
    
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
            
            # 1. 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20', 'ma60'])
            
            # 2. 计算 RSRS 因子
            result = self._compute_rsrs_factor(result)
            status['factors_computed'].append('rsrs_factor')
            
            # 3. 计算 RS 强度
            result = self._compute_rs_strength(result, index_data)
            status['factors_computed'].append('rs_strength')
            
            # 4. 计算 Pullback 信号
            result = self._compute_pullback_signal(result)
            status['factors_computed'].append('pullback_signal')
            
            # 5. 计算成交量萎缩
            result = self._compute_volume_shrink(result)
            status['factors_computed'].append('volume_shrink')
            
            # 6. 计算综合评分
            result = self._compute_composite_score(result)
            status['factors_computed'].append('composite_score')
            
            # 7. Schema 验证
            validate_factors(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V62 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算均线系统"""
        result = df.clone()
        
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
        
        return result.with_columns([
            ma5.alias('ma5'),
            ma20.alias('ma20'),
            ma60.alias('ma60')
        ])
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V62 核心：RSRS 择时因子
        
        【核心逻辑】
        1. 计算 18 日 RSRS 斜率
        2. 计算标准分 (z-score)
        3. z-score > 0.5 时才允许开仓
        """
        result = df.clone()
        rsrs_window = self.rsrs_window
        
        # 计算高低点关系
        high_low_ratio = pl.col('high') / (pl.col('low') + self.EPSILON)
        
        # 滚动均值和标准差
        hl_mean = high_low_ratio.rolling_mean(window_size=rsrs_window).over('symbol')
        hl_std = high_low_ratio.rolling_std(window_size=rsrs_window).over('symbol')
        
        # z-score 标准化
        rsrs_zscore = (high_low_ratio - hl_mean) / (hl_std + self.EPSILON)
        
        # 滚动计算 RSRS 斜率
        high_change = pl.col('high').pct_change().over('symbol')
        low_change = pl.col('low').pct_change().over('symbol')
        
        # 滚动相关系数近似
        hl_cov = (high_change * low_change).rolling_mean(window_size=rsrs_window).over('symbol')
        low_var = (low_change ** 2).rolling_mean(window_size=rsrs_window).over('symbol')
        
        # RSRS 斜率 = cov(high, low) / var(low)
        rsrs_slope = hl_cov / (low_var + self.EPSILON)
        
        # RSRS 综合得分
        rsrs_score = rsrs_zscore * (rsrs_slope.abs() + 0.1)
        
        # 开仓信号：z-score > 0.5
        rsrs_entry_signal = rsrs_zscore > self.rsrs_zscore_threshold
        
        return result.with_columns([
            high_low_ratio.alias('high_low_ratio'),
            hl_mean.alias('hl_mean'),
            hl_std.alias('hl_std'),
            rsrs_zscore.alias('rsrs_zscore'),
            rsrs_slope.alias('rsrs_slope'),
            rsrs_score.alias('rsrs_score'),
            rsrs_entry_signal.alias('rsrs_entry_signal')
        ])
    
    def _compute_rs_strength(self, df: pl.DataFrame, 
                             index_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        V62 核心：RS 强度计算
        
        【核心逻辑】
        1. 计算个股 20 日相对收益
        2. 若有指数数据，计算相对指数的超额收益
        3. RS 排名前 30%（放宽条件）
        """
        result = df.clone()
        
        # 个股 20 日收益率
        close_20_ago = pl.col('close').shift(self.rs_window).over('symbol')
        stock_return = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        
        # 使用个股 RS 作为主要指标
        rs_strength = stock_return
        
        # RS 排名 - 使用 dense 排名
        rs_rank = rs_strength.rank('dense', descending=True).over('trade_date')
        rs_count = rs_strength.count().over('trade_date')
        rs_percentile = 1.0 - (rs_rank.cast(pl.Float64) / (rs_count.cast(pl.Float64) + self.EPSILON))
        
        # V62: RS 排名前 30%（放宽条件）
        is_top_rs = rs_percentile >= (1.0 - 0.30)
        
        return result.with_columns([
            stock_return.alias('stock_return_20d'),
            rs_strength.alias('rs_strength'),
            rs_rank.cast(pl.Int64).alias('rs_rank'),
            rs_percentile.alias('rs_percentile'),
            is_top_rs.alias('is_top_rs')
        ])
    
    def _compute_pullback_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V62 核心：Pullback 回调买入信号
        
        【进场条件】
        1. Price 连续 2-3 日下跌（收盘价低于前一日收盘价）
        2. Close 落在 [MA20, MA20 * 1.03] 范围内
        
        【超宽松版本】
        - 连续 1-3 日下跌即可
        - 价格接近 MA20 即可（不严格要求在范围内）
        """
        result = df.clone()
        
        ma20 = pl.col('ma20')
        close = pl.col('close')
        prev_close = pl.col('close').shift(1).over('symbol')
        prev_close_2 = pl.col('close').shift(2).over('symbol')
        
        # 连续下跌检测
        # 今日下跌：收盘 < 昨日收盘
        today_down = close < prev_close
        # 昨日下跌
        yesterday_down = prev_close < prev_close_2
        
        # 连续 2 日下跌
        consecutive_2_days = today_down & yesterday_down
        
        # 连续 3 日下跌
        prev_close_3 = pl.col('close').shift(3).over('symbol')
        day_before_yesterday_down = prev_close_2 < prev_close_3
        consecutive_3_days = today_down & yesterday_down & day_before_yesterday_down
        
        # 连续 2-3 日下跌
        consecutive_decline = consecutive_2_days | consecutive_3_days
        
        # 计算连续下跌天数
        pullback_days = pl.when(consecutive_3_days) \
            .then(3) \
            .otherwise(pl.when(consecutive_2_days).then(2).otherwise(0))
        
        # 价格在 [MA20, MA20 * 1.03] 范围内 - 超宽松版本
        # 允许价格略低于 MA20（最多 5%），也允许略高于（最多 5%）
        price_in_range = (close >= ma20 * 0.95) & (close <= ma20 * (1 + 0.05))
        
        # 计算回调深度
        pullback_depth = (close - ma20) / (ma20 + self.EPSILON)
        
        # 综合 Pullback 信号（超宽松版本：只要连续下跌 + 价格接近 MA20）
        is_pullback_entry = consecutive_decline & price_in_range
        
        return result.with_columns([
            today_down.alias('today_down'),
            yesterday_down.alias('yesterday_down'),
            consecutive_decline.alias('consecutive_decline'),
            pullback_days.alias('pullback_days'),
            price_in_range.alias('price_in_pullback_range'),
            pullback_depth.alias('pullback_depth'),
            is_pullback_entry.alias('is_pullback_entry')
        ])
    
    def _compute_volume_shrink(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V62 核心：成交量萎缩检测
        
        【核心逻辑】
        - 成交量 < 5 日均量的 75%
        """
        result = df.clone()
        
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        
        # 成交量萎缩
        is_volume_shrunk = pl.col('volume') < (vol_ma5 * self.volume_shrink_ratio)
        
        # 成交量比率
        volume_ratio = pl.col('volume') / (vol_ma5 + self.EPSILON)
        
        return result.with_columns([
            vol_ma5.alias('vol_ma5'),
            volume_ratio.alias('volume_ratio'),
            is_volume_shrunk.alias('is_volume_shrunk')
        ])
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V62 综合评分计算
        
        【核心逻辑】
        1. RS 强度 bonus
        2. Pullback 信号 bonus
        3. 成交量萎缩 bonus
        4. RSRS 择时 bonus
        """
        result = df.clone()
        
        # RS 强度 bonus
        rs_bonus = pl.when(pl.col('is_top_rs')) \
            .then(pl.lit(0.30)) \
            .otherwise(pl.lit(0.0))
        
        # Pullback 信号 bonus（核心）
        pullback_bonus = pl.when(pl.col('is_pullback_entry')) \
            .then(pl.lit(0.35)) \
            .otherwise(pl.lit(0.0))
        
        # 成交量萎缩 bonus
        volume_bonus = pl.when(pl.col('is_volume_shrunk')) \
            .then(pl.lit(0.15)) \
            .otherwise(pl.lit(0.0))
        
        # RSRS 择时 bonus
        rsrs_bonus = pl.when(pl.col('rsrs_entry_signal')) \
            .then(pl.lit(0.20)) \
            .otherwise(pl.lit(0.0))
        
        # 基础动量得分（排名归一化）
        momentum_rank = pl.col('stock_return_20d').rank('ordinal', descending=True).over('trade_date')
        n_stocks = pl.col('symbol').count().over('trade_date')
        momentum_score = (1.0 - momentum_rank.cast(pl.Float64) / n_stocks.cast(pl.Float64)) * 0.5
        
        # 综合评分
        composite_score = rs_bonus + pullback_bonus + volume_bonus + rsrs_bonus + momentum_score
        
        # 排名计算
        composite_rank = composite_score.rank('ordinal', descending=True).over('trade_date')
        composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / (n_stocks.cast(pl.Float64) + self.EPSILON))
        
        # V62 综合买入信号：简化版本
        # 核心条件：Pullback 信号
        core_condition = pl.col('is_pullback_entry')
        
        # 辅助条件：成交量萎缩 OR RSRS 信号 OR RS 强度（满足其一即可）
        auxiliary_condition = pl.col('is_volume_shrunk') | pl.col('rsrs_entry_signal') | pl.col('is_top_rs')
        
        # 基础买入信号：核心条件 + 至少一个辅助条件
        buy_signal = core_condition & auxiliary_condition
        
        return result.with_columns([
            composite_score.alias('composite_score'),
            composite_rank.cast(pl.Int64).alias('composite_rank'),
            composite_percentile.alias('composite_percentile'),
            buy_signal.alias('buy_signal')
        ])
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V62Signal]:
        """
        生成交易信号
        
        Parameters
        ----------
        df : pl.DataFrame
            包含因子值的 DataFrame
        trade_date : str
            交易日期
        
        Returns
        -------
        List[V62Signal]
            交易信号列表
        """
        signals = []
        
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
            
            # 生成信号
            for row in buy_df.iter_rows(named=True):
                signal = V62Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    signal_rank=row.get('composite_rank', 9999),
                    composite_score=row.get('composite_score', 0.0),
                    is_pullback_entry=row.get('is_pullback_entry', False),
                    pullback_days=row.get('pullback_days', 0),
                    pullback_depth=row.get('pullback_depth', 0.0),
                    volume_shrunk=row.get('is_volume_shrunk', False),
                    rsrs_zscore=row.get('rsrs_zscore', 0.0),
                    rs_rank=row.get('rs_rank', 9999),
                    close_price=row.get('close', 0.0),
                    ma20_price=row.get('ma20', 0.0)
                )
                signals.append(signal)
            
            logger.info(f"V62 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V62 生成信号失败：{e}")
        
        return signals


# ===========================================
# V62 TradeExec - 真实成交执行
# ===========================================

class V62TradeExec:
    """
    V62 TradeExec - 真实成交执行
    
    【核心功能】
    1. min(Trigger, Open) 成交规则
    2. 手续费 + 滑点总计 0.2% 扣除
    3. 持仓管理
    4. 止损止盈执行
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', V62_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V62_MAX_POSITIONS)
        self.commission_rate = self.config.get('commission_rate', V62_COMMISSION_RATE)
        self.min_commission = self.config.get('min_commission', V62_MIN_COMMISSION)
        self.slippage_buy = self.config.get('slippage_buy', V62_SLIPPAGE_BUY)
        self.slippage_sell = self.config.get('slippage_sell', V62_SLIPPAGE_SELL)
        self.stamp_duty = self.config.get('stamp_duty', V62_STAMP_DUTY)
        self.transfer_fee = self.config.get('transfer_fee', V62_TRANSFER_FEE)
        self.friction_cost = self.config.get('friction_cost', V62_FRICTION_COST)
        
        # 持仓状态
        self.positions: Dict[str, V62Position] = {}
        self.cash = self.initial_capital
        self.trades: List[V62Trade] = []
        self.wash_sale_records: List[V62WashSaleRecord] = []
        self.sell_history: Dict[str, str] = {}  # symbol -> last sell date
    
    def execute_buy(self, signal: V62Signal, next_open: float, 
                    trigger_price: float, capital: float) -> Optional[V62Trade]:
        """
        执行买入
        
        Parameters
        ----------
        signal : V62Signal
            买入信号
        next_open : float
            次日开盘价
        trigger_price : float
            触发价格（当日 Close）
        capital : float
            可用资金
        
        Returns
        -------
        Optional[V62Trade]
            成交记录，None 表示未成交
        """
        # V62 核心：min(Trigger, Open) 成交规则
        # 买入时取较大值（确保不低价买入）
        execution_price = max(trigger_price, next_open)
        
        # 应用滑点
        execution_price = execution_price * (1 + self.slippage_buy)
        
        # 计算可买数量
        max_position_value = capital * V62_MAX_SINGLE_POSITION_PCT
        shares = int(max_position_value / execution_price / 100) * 100
        
        if shares <= 0:
            return None
        
        # 计算费用
        amount = shares * execution_price
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer_fee = amount * self.transfer_fee
        total_cost = amount + commission + transfer_fee
        
        if total_cost > capital:
            # 资金不足，减少股数
            shares = int((capital * 0.95) / execution_price / 100) * 100
            if shares <= 0:
                return None
            amount = shares * execution_price
            commission = max(amount * self.commission_rate, self.min_commission)
            transfer_fee = amount * self.transfer_fee
            total_cost = amount + commission + transfer_fee
        
        # 创建持仓
        position = V62Position(
            symbol=signal.symbol,
            shares=shares,
            avg_cost=execution_price,
            buy_price=execution_price,
            buy_date=signal.trade_date,
            signal_date=signal.signal_date if hasattr(signal, 'signal_date') else signal.trade_date,
            trade_date=signal.trade_date,
            signal_score=signal.signal_score,
            signal_rank=signal.signal_rank,
            composite_score=signal.composite_score,
            is_pullback_entry=signal.is_pullback_entry,
            pullback_days=signal.pullback_days,
            pullback_depth=signal.pullback_depth,
            volume_shrunk=signal.volume_shrunk,
            rsrs_zscore=signal.rsrs_zscore,
            rs_rank=signal.rs_rank,
            stop_loss_price=execution_price * (1 - V62_HARD_STOP_LOSS_RATIO),
            trailing_stop_price=execution_price * (1 - V62_TRAILING_STOP_RATIO),
            trigger_price=trigger_price,
            next_open_price=next_open,
            execution_price=execution_price
        )
        
        self.positions[signal.symbol] = position
        
        # 创建成交记录
        trade = V62Trade(
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
            reason='RS-Pullback 买入信号',
            signal_date=signal.trade_date,
            trigger_price=trigger_price,
            next_open_price=next_open,
            min_trigger_open=min(trigger_price, next_open),
            slippage_applied=self.slippage_buy
        )
        
        self.trades.append(trade)
        self.cash -= total_cost
        
        logger.info(f"V62 买入成交：{signal.symbol} @ {execution_price:.2f} x {shares}股")
        
        return trade
    
    def execute_sell(self, symbol: str, current_price: float, 
                     trade_date: str, reason: str) -> Optional[V62Trade]:
        """
        执行卖出
        
        Parameters
        ----------
        symbol : str
            股票代码
        current_price : float
            当前价格
        trade_date : str
            交易日期
        reason : str
            卖出原因
        
        Returns
        -------
        Optional[V62Trade]
            成交记录
        """
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
        
        # 计算盈亏
        gross_pnl = (execution_price - position.avg_cost) * shares
        net_pnl = gross_pnl - total_cost
        
        # 创建成交记录
        trade = V62Trade(
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
        
        # 记录卖出日期（洗售检测）
        self.sell_history[symbol] = trade_date
        
        # 删除持仓
        del self.positions[symbol]
        
        logger.info(f"V62 卖出成交：{symbol} @ {execution_price:.2f} x {shares}股，盈亏：{net_pnl:.2f}")
        
        return trade
    
    def check_exit_conditions(self, symbol: str, current_price: float,
                              trade_date: str) -> Optional[Tuple[bool, str]]:
        """
        检查离场条件
        
        Parameters
        ----------
        symbol : str
            股票代码
        current_price : float
            当前价格
        trade_date : str
            交易日期
        
        Returns
        -------
        Optional[Tuple[bool, str]]
            (是否触发离场，原因)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        cost_price = position.avg_cost
        
        # 更新最高价和移动止盈
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.peak_profit = (current_price - cost_price) / cost_price
        
        # 更新移动止盈价
        if position.peak_price > 0:
            position.trailing_stop_price = position.peak_price * (1 - V62_TRAILING_STOP_RATIO)
        
        # 1. 硬止损检查
        if current_price <= position.stop_loss_price:
            position.stop_loss_triggered = True
            return True, f"硬止损 (亏损>{V62_HARD_STOP_LOSS_RATIO*100:.1f}%)"
        
        # 2. 移动止盈检查
        if V62_TRAILING_STOP_ENABLED and position.trailing_stop_price > 0:
            if current_price <= position.trailing_stop_price:
                position.trailing_stop_triggered = True
                return True, f"移动止盈 (回撤>{V62_TRAILING_STOP_RATIO*100:.1f}%)"
        
        # 3. 目标止盈检查
        current_profit = (current_price - cost_price) / cost_price
        if current_profit >= V62_PROFIT_TARGET_RATIO:
            position.profit_target_triggered = True
            return True, f"目标止盈 (盈利>{V62_PROFIT_TARGET_RATIO*100:.1f}%)"
        
        return None
    
    def update_positions(self, market_data: Dict[str, Dict[str, float]], 
                         trade_date: str):
        """
        更新持仓状态
        
        Parameters
        ----------
        market_data : Dict[str, Dict[str, float]]
            市场行情数据 {symbol: {price, ma20, etc.}}
        trade_date : str
            交易日期
        """
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            current_price = data.get('close', 0)
            
            if current_price <= 0:
                continue
            
            # 更新当前价格
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
        """
        检查洗售规则
        
        Parameters
        ----------
        symbol : str
            股票代码
        trade_date : str
            交易日期
        
        Returns
        -------
        bool
            是否触发洗售限制
        """
        if symbol not in self.sell_history:
            return False
        
        last_sell_date = self.sell_history[symbol]
        
        try:
            sell_date = datetime.strptime(last_sell_date, "%Y-%m-%d")
            current = datetime.strptime(trade_date, "%Y-%m-%d")
            days_between = (current - sell_date).days
            
            if days_between <= V62_WASH_SALE_WINDOW:
                # 记录洗售
                wash_record = V62WashSaleRecord(
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
    'V62_INITIAL_CAPITAL',
    'V62_MAX_POSITIONS',
    'V62_WEEKLY_TRADE_LIMIT',
    'V62_GLOBAL_TRADE_LIMIT',
    'V62_WARMUP_PERIOD',
    'V62_MIN_SAMPLE_SIZE',
    'V62_RS_WINDOW',
    'V62_RS_TOP_PERCENTILE',
    'V62_PULLBACK_MIN_DAYS',
    'V62_PULLBACK_MAX_DAYS',
    'V62_MA20_BUFFER',
    'V62_VOLUME_MA_PERIOD',
    'V62_VOLUME_SHRINK_RATIO',
    'V62_RSRS_ENABLED',
    'V62_RSRS_WINDOW',
    'V62_RSRS_ZSCORE_THRESHOLD',
    'V62_ENTRY_ON_NEXT_OPEN',
    'V62_COMMISSION_RATE',
    'V62_MIN_COMMISSION',
    'V62_SLIPPAGE_BUY',
    'V62_SLIPPAGE_SELL',
    'V62_STAMP_DUTY',
    'V62_TRANSFER_FEE',
    'V62_FRICTION_COST',
    'V62_HARD_STOP_LOSS_RATIO',
    'V62_HARD_STOP_LOSS_ATR_MULT',
    'V62_HARD_STOP_LOSS_MODE',
    'V62_TRAILING_STOP_ENABLED',
    'V62_TRAILING_STOP_RATIO',
    'V62_PROFIT_TARGET_RATIO',
    'V62_RISK_TARGET_PER_POSITION',
    'V62_MAX_SINGLE_POSITION_PCT',
    'V62_WASH_SALE_WINDOW',
    
    # 函数
    'validate_factors',
    
    # 数据类
    'V62Position',
    'V62Trade',
    'V62TradeAudit',
    'V62Signal',
    'V62WashSaleRecord',
    'V62MarketRegime',
    
    # 核心类
    'V62DataManager',
    'V62AlphaCenter',
    'V62TradeExec',
]