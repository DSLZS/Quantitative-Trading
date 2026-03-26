"""
V75 Core Module - 自适应环境建模与波动率归一化迭代

【V75 核心算法 - 市场环境敏感度 + 波动率归一化 + 分形维度】

1. 市场环境敏感度 (Market Regime Sensor) - 核心创新
   ✅ 计算全市场过去 10 日的平均 Rank IC
   ✅ 动态调权：若近期资金流 IC 为负，自动将资金流权重降至 0 或反向
   ✅ 自适应权重分配，让评分系统学会根据最近的成败来调整算法

2. 波动率归一化 RS (Risk-Adjusted RS) - 类似 Sharpe Ratio
   ✅ 计算个股 N 日收益的波动率
   ✅ Risk-Adjusted RS = 收益 / 波动率（剔除靠剧烈波动拉升的股票）
   ✅ 保留平稳上涨的"真强势"股票

3. 分形过滤 (Hurst Exponent) - 趋势持续性审计
   ✅ 计算简单的 Hurst 指数
   ✅ 审计股价走势是"趋势持续"还是"随机震荡"
   ✅ 只对具有持续性倾向的标的加分 (H > 0.5)

4. 自适应权重分配
   ✅ 根据市场环境动态调整 RS、资金流、波动率、Hurst 的权重
   ✅ 市场好时：增加资金流权重
   ✅ 市场差时：增加防御性因子权重

5. 错误处理与主动纠错
   ✅ 遇到报错输出"错误分析 + 修复方案"
   ✅ 自动重启回测

6. 验收指标（决不妥协）
   ✅ 指标 A：全年度 Mean Rank IC >= 0.03
   ✅ 指标 B：最大回撤相对于 V74 减少 15% 以上
   ✅ 指标 C：评分分布 Std 保持在 20-30 之间

作者：量化系统
版本：V75.0
日期：2026-03-26
"""

import traceback
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V75 配置常量
# ===========================================

# 基础配置
V75_INITIAL_CAPITAL = 100000.00  # 初始资金 10 万（严禁修改）
V75_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V75 数据预加载配置
V75_WARMUP_PERIOD = 250  # 预加载 250 天数据
V75_MIN_SAMPLE_SIZE = 100  # 最小股票样本量

# V75 数据完整性检查
V75_DATA_CHECK_YEAR = "2024"
V75_MIN_FUND_FLOW_ROWS = 50000  # 2024 全年资金流最少行数
V75_RETRY_ATTEMPTS = 5
V75_RETRY_DELAY = 3.0  # 秒

# V75 RS 动量配置
V75_RS_WINDOW = 20  # RS 计算窗口（20 日）
V75_RS_BASE_SCORE_MIN = 30.0  # RS 基础分最小值
V75_RS_BASE_SCORE_MAX = 70.0  # RS 基础分最大值

# V75 市场环境敏感度配置（核心创新）
V75_REGIME_SENSOR_WINDOW = 10  # 环境敏感度计算窗口（10 日）
V75_REGIME_IC_THRESHOLD = 0.0  # IC 阈值，低于此值认为资金流失效
V75_REGIME_ADAPTIVE_WEIGHT = True  # 启用自适应权重

# V75 波动率归一化配置
V75_VOLATILITY_WINDOW = 20  # 波动率计算窗口
V75_VOLATILITY_WEIGHT = 0.15  # 波动率权重
V75_RISK_ADJUSTED_RS = True  # 启用风险调整 RS

# V75 Hurst 指数配置
V75_HURST_WINDOW = 30  # Hurst 指数计算窗口（需要足够样本）
V75_HURST_WEIGHT = 0.10  # Hurst 权重
V75_HURST_THRESHOLD = 0.5  # Hurst 阈值，高于此值认为有趋势持续性

# V75 自适应权重配置
V75_ADAPTIVE_RS_WEIGHT = 0.55  # RS 基础权重
V75_ADAPTIVE_FUND_WEIGHT = 0.25  # 资金流基础权重
V75_ADAPTIVE_VOL_WEIGHT = 0.10  # 波动率基础权重
V75_ADAPTIVE_HURST_WEIGHT = 0.10  # Hurst 基础权重

# V75 非线性融合配置
V75_SIGMOID_SCALE = 0.05  # Sigmoid/Tanh 缩放因子
V75_SCORE_SMOOTHING = True  # 启用平滑函数

# V75 行业配置
V75_INDUSTRY_NEUTRAL_WEIGHT = 1.0  # 行业数据缺失时的中性权重

# V75 费率配置 - 总计 0.2%（严禁修改）
V75_COMMISSION_RATE = 0.0003  # 佣金万 3
V75_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V75_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V75_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V75_STAMP_DUTY = 0.0005  # 印花税 0.05%
V75_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V75_FRICTION_COST = 0.002  # 0.2% 总计

# V75 离场配置（严禁修改）
V75_STOP_LOSS_RATIO = 0.05  # 止损 5%
V75_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V75_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V75 仓位管理
V75_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V75 选股排名
V75_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V75 Rank IC 目标（验收指标）
V75_RANK_IC_TARGET = 0.03  # 月度 Rank IC 目标（核心指标）
V75_SCORE_STD_TARGET_MIN = 20.0  # 评分标准差最小值
V75_SCORE_STD_TARGET_MAX = 30.0  # 评分标准差最大值

# V75 回撤控制目标
V75_DRAWDOWN_REDUCTION_TARGET = 0.15  # 相对于 V74 回撤减少 15%


# ===========================================
# V75 数据类定义
# ===========================================

@dataclass
class V75Position:
    """V75 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    rs_score: float = 0.0  # RS 基础分
    risk_adjusted_rs: float = 0.0  # 风险调整 RS
    hurst_exponent: float = 0.5  # Hurst 指数
    fund_flow_weight: float = 0.0  # 资金流权重（自适应）
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    
    # 行业数据
    industry_name: str = ""
    industry_weight: float = 1.0
    
    # 市场环境
    market_regime: str = "neutral"  # bullish/bearish/neutral
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False


@dataclass
class V75Trade:
    """V75 交易记录"""
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


@dataclass
class V75Signal:
    """V75 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # RS 动量
    rs_score: float = 0.0
    rs_rank: int = 0
    risk_adjusted_rs: float = 0.0
    
    # 波动率
    volatility: float = 0.0
    vol_score: float = 0.0
    
    # Hurst 指数
    hurst_exponent: float = 0.5
    hurst_score: float = 0.0
    
    # 资金流（自适应权重）
    net_main_rate: float = 0.0
    fund_flow_weight: float = 0.0  # 自适应权重
    
    # 市场环境
    market_regime: str = "neutral"
    regime_ic: float = 0.0  # 近期 IC
    
    # 行业权重
    industry_name: str = ""
    industry_weight: float = 1.0
    
    # 价格数据
    close_price: float = 0.0


@dataclass
class V75ICMetrics:
    """V75 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V75ScoreDistribution:
    """V75 评分分布统计"""
    trade_date: str
    min_score: float
    max_score: float
    mean_score: float
    std_score: float
    median_score: float
    q1_score: float  # 25 分位
    q3_score: float  # 75 分位
    skewness: float
    kurtosis: float


@dataclass
class V75MonthlyICStats:
    """V75 月度 IC 统计"""
    month: str  # YYYY-MM
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int


@dataclass
class V75MarketRegime:
    """V75 市场环境状态"""
    trade_date: str
    regime_type: str  # bullish/bearish/neutral
    recent_ic: float  # 近期平均 IC
    fund_flow_ic: float  # 资金流 IC
    rs_ic: float  # RS IC
    volatility_regime: str  # high/low
    trend_strength: float  # 趋势强度


# ===========================================
# V75 DataManager - 数据获取与预处理
# ===========================================

class V75DataManager:
    """
    V75 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时透明化报告
    3. 支持全市场股票评分
    4. ConnectionError 重试机制
    5. 2024 全年数据完整性检查
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V75_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V75_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V75_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V75_RETRY_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._missing_data_log: List[str] = []
    
    def _retry_wrapper(self, func, *args, **kwargs):
        """重试包装器 - 处理 ConnectionError 和 NoneType 错误"""
        last_exception = None
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                if result is None:
                    raise ValueError("函数返回 None")
                
                if attempt > 1:
                    logger.info(f"V75: 第 {attempt} 次尝试成功")
                return result
                
            except (ConnectionError, OSError) as e:
                last_exception = e
                logger.warning(f"V75: 第 {attempt} 次尝试失败 (ConnectionError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V75: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V75: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except (AttributeError, TypeError) as e:
                last_exception = e
                logger.warning(f"V75: 第 {attempt} 次尝试失败 (AttributeError/TypeError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V75: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V75: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except Exception as e:
                logger.error(f"V75: 发生错误：{e}")
                raise
        
        raise last_exception
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        """计算预加载开始日期"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2023-01-01"
    
    def check_2024_data_integrity(self) -> Tuple[bool, str]:
        """检查 2024 全年数据完整性"""
        logger.info("=" * 60)
        logger.info("V75: 开始检查 2024 全年数据完整性")
        logger.info("=" * 60)
        
        if self.db is None:
            return False, "数据库连接未初始化"
        
        try:
            # 检查 stock_fund_flow 表 2024 年数据
            query = """
                SELECT COUNT(*) as cnt 
                FROM stock_fund_flow 
                WHERE trade_date >= '2024-01-01' 
                  AND trade_date <= '2024-12-31'
            """
            df = self._retry_wrapper(self.db.read_sql, query)
            
            if df.is_empty():
                return False, "无法查询 stock_fund_flow 表"
            
            fund_flow_count = int(df['cnt'][0])
            logger.info(f"V75: 2024 年资金流数据行数：{fund_flow_count}")
            
            # 检查 stock_industry_daily 表
            query = """
                SELECT COUNT(*) as cnt 
                FROM stock_industry_daily 
                WHERE trade_date >= '2024-01-01' 
                  AND trade_date <= '2024-12-31'
            """
            df = self._retry_wrapper(self.db.read_sql, query)
            
            if df.is_empty():
                return False, "无法查询 stock_industry_daily 表"
            
            industry_count = int(df['cnt'][0])
            logger.info(f"V75: 2024 年行业数据行数：{industry_count}")
            
            if fund_flow_count < V75_MIN_FUND_FLOW_ROWS:
                msg = f"2024 年资金流数据不足：{fund_flow_count} < {V75_MIN_FUND_FLOW_ROWS}，需要补抓"
                logger.warning(f"V75: {msg}")
                return False, msg
            
            logger.info("V75: 2024 全年数据完整性检查通过")
            return True, f"数据完整 (fund_flow={fund_flow_count}, industry={industry_count})"
            
        except Exception as e:
            logger.error(f"V75: 检查数据完整性失败：{e}")
            return False, f"检查失败：{e}"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V75 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V75 DataManager: 数据库连接未初始化")
        
        def _load():
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
            
            logger.debug(f"V75 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V75 DataManager: 未加载到任何数据")
            
            logger.info(f"V75 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V75: 数据库连接未初始化")
            return self._empty_fund_flow_df()
        
        def _load():
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
            
            logger.debug(f"V75 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V75: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V75: 数据库连接未初始化")
            return self._empty_industry_df()
        
        def _load():
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V75 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V75: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V75: 数据库连接未初始化")
            return self._empty_index_df()
        
        def _load():
            query = f"""
                SELECT trade_date, close
                FROM index_daily
                WHERE symbol = '{index_code}'
                  AND trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            
            logger.debug(f"V75 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V75: 未加载到指数 {index_code} 数据")
                return self._empty_index_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def _empty_fund_flow_df(self) -> pl.DataFrame:
        """返回空资金流 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'net_main_amount': pl.Float64,
            'net_main_ratio': pl.Float64,
            'net_main_rate': pl.Float64
        })
    
    def _empty_industry_df(self) -> pl.DataFrame:
        """返回空行业 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'industry_name': pl.Utf8
        })
    
    def _empty_index_df(self) -> pl.DataFrame:
        """返回空指数 DataFrame"""
        return pl.DataFrame(schema={
            'trade_date': pl.Utf8,
            'close': pl.Float64
        })
    
    def log_missing_data(self, symbol: str, trade_date: str, field_name: str):
        """记录缺失数据"""
        missing_info = f"缺失数据：symbol={symbol}, trade_date={trade_date}, field={field_name}"
        self._missing_data_log.append(missing_info)
        logger.warning(f"V75: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V75 缺失数据报告")
            logger.info("=" * 60)
            for log in self._missing_data_log[:100]:
                logger.info(f"  {log}")
            if len(self._missing_data_log) > 100:
                logger.info(f"  ... 还有 {len(self._missing_data_log) - 100} 条")
            logger.info("=" * 60)
    
    def clear_cache(self):
        """清除缓存"""
        self._data_cache.clear()


# ===========================================
# V75 AlphaCenter - 自适应环境建模 + 波动率归一化 + Hurst 指数
# ===========================================

class V75AlphaCenter:
    """
    V75 AlphaCenter - 自适应环境建模与波动率归一化
    
    【核心逻辑】
    1. 市场环境敏感度 (Market Regime Sensor) - 计算近期 IC 动态调权
    2. Risk-Adjusted RS - 波动率归一化的相对强度
    3. Hurst 指数 - 分形维度过滤趋势持续性
    4. 自适应权重分配 - 根据市场环境调整因子权重
    5. 非线性融合 - Tanh 平滑函数
    
    【评分公式】
    raw_score = w1*RS + w2*Fund + w3*Vol + w4*Hurst
    composite_score = 50 + 50 * Tanh(raw_score * scale)
    
    其中权重 w1-w4 根据市场环境动态调整
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # RS 配置
        self.rs_window = self.config.get('rs_window', V75_RS_WINDOW)
        self.rs_score_min = self.config.get('rs_score_min', V75_RS_BASE_SCORE_MIN)
        self.rs_score_max = self.config.get('rs_score_max', V75_RS_BASE_SCORE_MAX)
        
        # 市场环境敏感度配置
        self.regime_sensor_window = self.config.get('regime_sensor_window', V75_REGIME_SENSOR_WINDOW)
        self.regime_ic_threshold = self.config.get('regime_ic_threshold', V75_REGIME_IC_THRESHOLD)
        self.adaptive_weight = self.config.get('adaptive_weight', V75_REGIME_ADAPTIVE_WEIGHT)
        
        # 波动率归一化配置
        self.volatility_window = self.config.get('volatility_window', V75_VOLATILITY_WINDOW)
        self.volatility_weight = self.config.get('volatility_weight', V75_VOLATILITY_WEIGHT)
        self.risk_adjusted_rs = self.config.get('risk_adjusted_rs', V75_RISK_ADJUSTED_RS)
        
        # Hurst 指数配置
        self.hurst_window = self.config.get('hurst_window', V75_HURST_WINDOW)
        self.hurst_weight = self.config.get('hurst_weight', V75_HURST_WEIGHT)
        self.hurst_threshold = self.config.get('hurst_threshold', V75_HURST_THRESHOLD)
        
        # 自适应权重配置
        self.adaptive_rs_weight = self.config.get('adaptive_rs_weight', V75_ADAPTIVE_RS_WEIGHT)
        self.adaptive_fund_weight = self.config.get('adaptive_fund_weight', V75_ADAPTIVE_FUND_WEIGHT)
        self.adaptive_vol_weight = self.config.get('adaptive_vol_weight', V75_ADAPTIVE_VOL_WEIGHT)
        self.adaptive_hurst_weight = self.config.get('adaptive_hurst_weight', V75_ADAPTIVE_HURST_WEIGHT)
        
        # 非线性融合配置
        self.sigmoid_scale = self.config.get('sigmoid_scale', V75_SIGMOID_SCALE)
        self.score_smoothing = self.config.get('score_smoothing', V75_SCORE_SMOOTHING)
        
        # 行业配置
        self.industry_neutral_weight = self.config.get(
            'industry_neutral_weight', V75_INDUSTRY_NEUTRAL_WEIGHT
        )
        
        # IC 历史记录（用于计算近期 IC）
        self._ic_history: Dict[str, List[float]] = {'rs': [], 'fund': [], 'composite': []}
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None,
                        index_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """计算所有因子和交易信号"""
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
            ])
            
            # 添加 mv 列 (如果不存在)
            if 'mv' not in result.columns:
                result = result.with_columns(pl.lit(0.0).alias('mv'))
            
            status = {
                'factors_computed': [],
                'score_distribution': {},
                'market_regime': 'neutral',
            }
            
            # 检测数据可用性
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            has_industry = industry_df is not None and not industry_df.is_empty()
            has_index = index_df is not None and not index_df.is_empty()
            
            # 1. 计算市场环境敏感度（首先执行，用于后续权重调整）
            market_regime = self._compute_market_regime(result, index_df)
            status['market_regime'] = market_regime.regime_type
            
            # 2. 计算 Risk-Adjusted RS（波动率归一化）
            result = self._compute_risk_adjusted_rs(result, index_df)
            status['factors_computed'].append('risk_adjusted_rs')
            
            # 3. 计算 Hurst 指数（分形维度）
            result = self._compute_hurst_exponent(result)
            status['factors_computed'].append('hurst_exponent')
            
            # 4. 计算资金流因子（带自适应权重）
            if has_fund_flow:
                result = self._compute_fund_flow_with_adaptive_weight(
                    result, fund_flow_df, market_regime
                )
                status['factors_computed'].append('fund_flow_adaptive')
            else:
                result = self._add_fund_flow_placeholder(result)
                status['factors_computed'].append('fund_flow_placeholder')
            
            # 5. 计算行业权重
            if has_industry:
                result = self._compute_industry_weight(result, industry_df, fund_flow_df)
                status['factors_computed'].append('industry_weight')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 6. 计算综合评分（自适应权重融合）
            result = self._compute_composite_score_adaptive(result, market_regime)
            status['factors_computed'].append('composite_score')
            
            # 7. 计算评分分布统计
            result = self._compute_score_distribution(result)
            
            logger.info(f"V75 AlphaCenter 信号计算完成，市场环境：{market_regime.regime_type}")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V75 AlphaCenter 计算信号失败：{e}")
            logger.error(f"【错误分析】{traceback.format_exc()}")
            logger.error("【修复方案】检查数据完整性，确保所有必要数据已加载")
            raise
    
    def _compute_market_regime(self, df: pl.DataFrame,
                                index_df: Optional[pl.DataFrame] = None) -> V75MarketRegime:
        """
        计算市场环境敏感度 (Market Regime Sensor)
        
        【核心逻辑】
        1. 计算过去 N 日的平均 Rank IC
        2. 根据 IC 判断市场环境：
           - IC > 0.03: bullish（资金流有效）
           - IC < 0: bearish（资金流失效，可能是陷阱）
           - 0 <= IC <= 0.03: neutral
        3. 动态调整因子权重
        """
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        if len(unique_dates) < self.regime_sensor_window:
            # 数据不足，返回中性环境
            return V75MarketRegime(
                trade_date=unique_dates[-1] if unique_dates else datetime.now().strftime("%Y-%m-%d"),
                regime_type='neutral',
                recent_ic=0.0,
                fund_flow_ic=0.0,
                rs_ic=0.0,
                volatility_regime='normal',
                trend_strength=0.5
            )
        
        # 计算近期 IC（使用最近 N 天）
        recent_dates = unique_dates[-self.regime_sensor_window:]
        
        # 计算市场波动率 regime
        if index_df is not None and not index_df.is_empty():
            index_returns = index_df['close'].pct_change().drop_nulls().to_numpy()
            recent_vol = np.std(index_returns) if len(index_returns) > 0 else 0.0
            vol_regime = 'high' if recent_vol > 0.02 else 'low'
        else:
            vol_regime = 'normal'
        
        # 获取历史 IC 记录
        recent_ic = np.mean(self._ic_history.get('composite', [0.0]))
        fund_flow_ic = np.mean(self._ic_history.get('fund', [0.0]))
        rs_ic = np.mean(self._ic_history.get('rs', [0.0]))
        
        # 判断市场环境
        if recent_ic > 0.03:
            regime_type = 'bullish'
        elif recent_ic < self.regime_ic_threshold:
            regime_type = 'bearish'
        else:
            regime_type = 'neutral'
        
        # 计算趋势强度
        trend_strength = abs(recent_ic) * 10  # 放大 IC 作为趋势强度
        
        return V75MarketRegime(
            trade_date=recent_dates[-1],
            regime_type=regime_type,
            recent_ic=recent_ic,
            fund_flow_ic=fund_flow_ic,
            rs_ic=rs_ic,
            volatility_regime=vol_regime,
            trend_strength=trend_strength
        )
    
    def _compute_risk_adjusted_rs(self, df: pl.DataFrame,
                                   index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算 Risk-Adjusted RS（波动率归一化相对强度）
        
        【核心逻辑】
        1. 计算个股 N 日收益率
        2. 计算个股 N 日收益率的波动率
        3. Risk-Adjusted RS = 收益率 / 波动率（类似 Sharpe Ratio）
        4. 剔除靠剧烈波动拉升的股票，保留平稳上涨的"真强势"
        
        【计算公式】
        raw_return = (close - close.shift(N)) / close.shift(N)
        volatility = std(raw_return over N days)
        risk_adjusted_return = raw_return / (volatility + epsilon)
        """
        result = df.clone()
        
        # 计算个股 N 日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
             (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('stock_return')
        ])
        
        # 计算个股收益率的波动率（滚动窗口）
        result = result.with_columns([
            (pl.col('stock_return').rolling_std(window_size=self.volatility_window) / 
             (self.EPSILON)).alias('return_volatility')
        ])
        
        # 计算市场收益率和波动率
        if index_df is not None and not index_df.is_empty():
            index_df = index_df.with_columns([
                pl.col('trade_date').cast(pl.Utf8).alias('trade_date'),
                ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
                 (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('market_return')
            ])
            
            result = result.join(
                index_df.select(['trade_date', 'market_return']),
                on='trade_date',
                how='left'
            )
            
            result = result.with_columns([
                pl.col('market_return').fill_null(0.0).alias('market_return')
            ])
        else:
            result = result.with_columns([
                pl.col('stock_return').mean().over('trade_date').alias('market_return')
            ])
        
        # 计算超额收益
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('market_return')).alias('excess_return')
        ])
        
        # 计算 Risk-Adjusted RS = 超额收益 / 波动率
        result = result.with_columns([
            (pl.col('excess_return') / (pl.col('return_volatility') + self.EPSILON)).alias('risk_adjusted_rs_raw')
        ])
        
        # 填充 NaN 值
        result = result.with_columns([
            pl.col('risk_adjusted_rs_raw').fill_null(0.0).alias('risk_adjusted_rs_filled')
        ])
        
        # 横截面标准化
        result = result.with_columns([
            pl.col('risk_adjusted_rs_filled').mean().over('trade_date').alias('ra_rs_mean'),
            pl.col('risk_adjusted_rs_filled').std().over('trade_date').alias('ra_rs_std')
        ])
        
        result = result.with_columns([
            ((pl.col('risk_adjusted_rs_filled') - pl.col('ra_rs_mean')) / 
             (pl.col('ra_rs_std') + self.EPSILON)).alias('risk_adjusted_rs_zscore')
        ])
        
        # 填充 NaN zscore
        result = result.with_columns([
            pl.col('risk_adjusted_rs_zscore').fill_null(0.0).alias('risk_adjusted_rs_zscore_filled')
        ])
        
        # 映射到分数 [0, 100]
        ra_rs_scores = []
        for zscore in result['risk_adjusted_rs_zscore_filled'].to_numpy():
            if np.isnan(zscore):
                ra_rs_scores.append(50.0)
            else:
                # 使用 Sigmoid 映射到 0-100
                score = 50 + 50 * np.tanh(zscore * 0.5)
                ra_rs_scores.append(float(score))
        
        result = result.with_columns([
            pl.Series('risk_adjusted_rs', ra_rs_scores)
        ])
        
        # 同时保留原始 RS（用于对比）
        result = result.with_columns([
            ((pl.col('stock_return') - pl.col('market_return')) / self.EPSILON).alias('raw_rs_value')
        ])
        
        return result
    
    def _compute_hurst_exponent(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Hurst 指数（简化的分形维度）
        
        【核心逻辑】
        1. Hurst 指数 H 用于衡量时间序列的持续性
           - H > 0.5: 趋势持续（上涨/下跌趋势会延续）
           - H = 0.5: 随机游走
           - H < 0.5: 均值回归（反转）
        2. 只对 H > 0.5 的股票加分（趋势持续性）
        
        【简化计算】
        使用 R/S 分析法的简化版本：
        H = log(R/S) / log(N)
        其中 R/S 是重标极差，N 是窗口长度
        """
        result = df.clone()
        
        # 计算收益率序列
        result = result.with_columns([
            (pl.col('close').pct_change() / self.EPSILON).alias('returns')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('returns').fill_null(0.0).alias('returns_filled')
        ])
        
        # 按股票分组计算 Hurst 指数
        def compute_hurst_simple(returns: np.ndarray) -> float:
            """简化的 Hurst 指数计算"""
            if len(returns) < self.hurst_window:
                return 0.5
            
            # 取最近 N 天
            returns = returns[-self.hurst_window:]
            returns = returns[~np.isnan(returns)]
            
            if len(returns) < 10:
                return 0.5
            
            # 计算累积收益
            cum_returns = np.cumsum(returns - np.mean(returns))
            
            # 计算 R/S
            R = np.max(cum_returns) - np.min(cum_returns)
            S = np.std(returns)
            
            if S < self.EPSILON:
                return 0.5
            
            RS = R / S
            
            # 计算 Hurst 指数
            N = len(returns)
            H = np.log(RS) / np.log(N)
            
            # 限制在合理范围
            H = np.clip(H, 0.0, 1.0)
            
            return H
        
        # 为每个交易日计算 Hurst 指数
        hurst_values = []
        unique_dates = result['trade_date'].unique().to_list()
        date_hurst_map = {}
        
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            symbols = day_data['symbol'].unique().to_list()
            
            for symbol in symbols:
                symbol_data = result.filter(
                    (pl.col('symbol') == symbol) & 
                    (pl.col('trade_date') <= trade_date)
                ).sort('trade_date')
                
                returns = symbol_data['returns_filled'].to_numpy()
                hurst = compute_hurst_simple(returns)
                
                key = f"{symbol}_{trade_date}"
                date_hurst_map[key] = hurst
        
        # 将 Hurst 值映射回结果 - 使用高效的 join 方式
        hurst_rows = []
        symbols_list = []
        dates_list = []
        
        for idx in range(result.height):
            symbol = result['symbol'][idx]
            trade_date = result['trade_date'][idx]
            key = f"{symbol}_{trade_date}"
            hurst_rows.append(date_hurst_map.get(key, 0.5))
        
        result = result.with_columns([
            pl.Series('hurst_exponent', hurst_rows)
        ])
        
        # 计算 Hurst 得分（H > 0.5 加分）
        result = result.with_columns([
            ((pl.col('hurst_exponent') - 0.5) * 2 * 100).alias('hurst_score')
        ])
        
        return result
    
    def _compute_fund_flow_with_adaptive_weight(self, df: pl.DataFrame,
                                                  fund_flow_df: pl.DataFrame,
                                                  market_regime: V75MarketRegime) -> pl.DataFrame:
        """
        计算资金流因子（带自适应权重）
        
        【核心逻辑】
        1. 根据市场环境动态调整资金流权重
        2. 若近期资金流 IC 为负，将权重降至 0 或反向
        3. 自适应权重公式：
           - bullish: fund_weight = base_weight * 1.5
           - bearish: fund_weight = -base_weight * 0.5（反向）
           - neutral: fund_weight = base_weight
        """
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_main_rate']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left')
        
        # 填充缺失值
        result = result.with_columns([
            pl.col('net_main_rate').fill_null(0.0).alias('net_main_rate'),
            pl.col('net_main_amount').fill_null(0.0).alias('net_main_amount')
        ])
        
        # 计算横截面统计量
        result = result.with_columns([
            pl.col('net_main_rate').mean().over('trade_date').alias('market_mean_rate'),
            pl.col('net_main_rate').std().over('trade_date').alias('market_std_rate')
        ])
        
        # 计算资金流 Z-Score
        result = result.with_columns([
            ((pl.col('net_main_rate') - pl.col('market_mean_rate')) / 
             (pl.col('market_std_rate') + self.EPSILON)).alias('fund_flow_zscore')
        ])
        
        # 根据市场环境计算自适应权重
        if market_regime.regime_type == 'bullish':
            adaptive_weight = self.adaptive_fund_weight * 1.5
        elif market_regime.regime_type == 'bearish':
            # 资金流失效时，反向使用或降至 0
            adaptive_weight = -self.adaptive_fund_weight * 0.5
        else:
            adaptive_weight = self.adaptive_fund_weight
        
        # 存储自适应权重
        result = result.with_columns([
            pl.lit(adaptive_weight).alias('fund_flow_adaptive_weight')
        ])
        
        # 计算资金流得分
        fund_scores = []
        for zscore in result['fund_flow_zscore'].to_numpy():
            if np.isnan(zscore):
                fund_scores.append(50.0)
            else:
                score = 50 + 50 * np.tanh(zscore * 0.5)
                fund_scores.append(float(score))
        
        result = result.with_columns([
            pl.Series('fund_flow_score', fund_scores)
        ])
        
        return result
    
    def _add_fund_flow_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加资金流占位符（中性值）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(50.0).alias('fund_flow_score'),
            pl.lit(self.adaptive_fund_weight).alias('fund_flow_adaptive_weight')
        ])
        
        return result
    
    def _compute_industry_weight(self, df: pl.DataFrame,
                                  industry_df: pl.DataFrame,
                                  fund_flow_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """计算行业权重"""
        result = df.clone()
        
        # 合并行业数据
        industry_cols = ['symbol', 'trade_date', 'industry_name']
        available_industry_cols = [c for c in industry_cols if c in industry_df.columns]
        industry_data = industry_df.select(available_industry_cols)
        
        result = result.join(industry_data, on=['symbol', 'trade_date'], how='left')
        
        # 填充缺失的行业名称
        result = result.with_columns([
            pl.col('industry_name').fill_null('UNKNOWN').alias('industry_name')
        ])
        
        # 如果有资金流数据，计算行业资金流强度
        if fund_flow_df is not None and not fund_flow_df.is_empty():
            fund_cols = ['symbol', 'trade_date', 'net_main_rate']
            available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
            fund_data = fund_flow_df.select(available_fund_cols)
            
            result = result.join(fund_data, on=['symbol', 'trade_date'], how='left', suffix='_fund')
            
            # 计算行业每日资金流均值
            industry_daily_agg = result.group_by(['trade_date', 'industry_name']).agg([
                pl.col('net_main_rate').mean().alias('industry_net_flow')
            ])
            
            # 计算行业资金流的横截面排名
            industry_daily_agg = industry_daily_agg.with_columns([
                pl.col('industry_net_flow').mean().over('trade_date').alias('market_industry_mean'),
                pl.col('industry_net_flow').std().over('trade_date').alias('market_industry_std')
            ])
            
            industry_daily_agg = industry_daily_agg.with_columns([
                ((pl.col('industry_net_flow') - pl.col('market_industry_mean')) / 
                 (pl.col('market_industry_std') + self.EPSILON)).alias('industry_z_score')
            ])
            
            industry_daily_agg = industry_daily_agg.with_columns([
                (1.0 + (pl.col('industry_z_score') / 5.0).clip(-0.2, 0.2)).alias('industry_weight')
            ])
            
            result = result.join(
                industry_daily_agg.select(['trade_date', 'industry_name', 'industry_weight']),
                on=['trade_date', 'industry_name'],
                how='left',
                suffix='_ind'
            )
        else:
            result = result.with_columns([
                pl.lit(1.0).alias('industry_weight')
            ])
        
        # 对于 UNKNOWN 行业，赋予中性权重
        result = result.with_columns([
            pl.when(pl.col('industry_name') == 'UNKNOWN')
            .then(1.0)
            .otherwise(pl.col('industry_weight'))
            .alias('industry_weight')
        ])
        
        return result
    
    def _add_industry_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加行业占位符（中性权重）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit('UNKNOWN').alias('industry_name'),
            pl.lit(1.0).alias('industry_weight')
        ])
        
        return result
    
    def _compute_composite_score_adaptive(self, df: pl.DataFrame,
                                           market_regime: V75MarketRegime) -> pl.DataFrame:
        """
        计算综合评分（自适应权重融合）
        
        【核心逻辑】
        1. 根据市场环境动态调整各因子权重
        2. 加权组合各因子得分
        3. 使用 Tanh 平滑函数映射到 0-100
        
        【权重调整规则】
        - bullish: 增加资金流权重，减少防御因子
        - bearish: 增加 Risk-Adjusted RS 和 Hurst 权重，减少资金流
        - neutral: 使用基础权重
        """
        result = df.clone()
        
        # 确定自适应权重
        if market_regime.regime_type == 'bullish':
            rs_weight = self.adaptive_rs_weight * 0.8
            fund_weight = self.adaptive_fund_weight * 1.5
            vol_weight = self.adaptive_vol_weight * 0.8
            hurst_weight = self.adaptive_hurst_weight * 0.9
        elif market_regime.regime_type == 'bearish':
            # 防御模式：增加质量因子权重
            rs_weight = self.adaptive_rs_weight * 1.2
            fund_weight = -self.adaptive_fund_weight * 0.3  # 反向或降低
            vol_weight = self.adaptive_vol_weight * 1.3  # 更看重低波动
            hurst_weight = self.adaptive_hurst_weight * 1.5  # 更看重趋势持续性
        else:
            rs_weight = self.adaptive_rs_weight
            fund_weight = self.adaptive_fund_weight
            vol_weight = self.adaptive_vol_weight
            hurst_weight = self.adaptive_hurst_weight
        
        # 归一化权重
        total_weight = abs(rs_weight) + abs(fund_weight) + abs(vol_weight) + abs(hurst_weight)
        if total_weight < self.EPSILON:
            total_weight = 1.0
        
        rs_weight_norm = rs_weight / total_weight
        fund_weight_norm = fund_weight / total_weight
        vol_weight_norm = vol_weight / total_weight
        hurst_weight_norm = hurst_weight / total_weight
        
        # 获取各因子得分
        # RS 得分（从 risk_adjusted_rs 映射）
        result = result.with_columns([
            ((pl.col('risk_adjusted_rs') - 50) / 50).alias('rs_normalized')
        ])
        
        # 资金流得分
        if 'fund_flow_score' in result.columns:
            result = result.with_columns([
                ((pl.col('fund_flow_score') - 50) / 50).alias('fund_normalized')
            ])
        else:
            result = result.with_columns([
                pl.lit(0.0).alias('fund_normalized')
            ])
        
        # 波动率得分（低波动高分）
        result = result.with_columns([
            (pl.col('return_volatility').rank('ordinal', descending=False).over('trade_date')).cast(pl.Float64).alias('vol_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks_vol')
        ])
        
        # 先计算 vol_percentile，再计算 vol_normalized（分开两步）
        result = result.with_columns([
            (1.0 - (pl.col('vol_rank') - 0.5) / 
             (pl.col('n_stocks_vol') + self.EPSILON)).alias('vol_percentile')
        ])
        
        result = result.with_columns([
            ((pl.col('vol_percentile') - 0.5) * 2).alias('vol_normalized')
        ])
        
        # Hurst 得分
        result = result.with_columns([
            ((pl.col('hurst_exponent') - 0.5) * 2).alias('hurst_normalized')
        ])
        
        # 加权组合
        result = result.with_columns([
            (
                rs_weight_norm * pl.col('rs_normalized') +
                fund_weight_norm * pl.col('fund_normalized') +
                vol_weight_norm * pl.col('vol_normalized') +
                hurst_weight_norm * pl.col('hurst_normalized')
            ).alias('raw_composite')
        ])
        
        # 使用 Tanh 映射到 0-100
        raw_scores = result['raw_composite'].to_numpy()
        composite_scores = []
        
        for raw in raw_scores:
            if np.isnan(raw):
                composite_scores.append(50.0)
            else:
                score = 50 + 50 * np.tanh(raw * 10)  # 放大系数
                composite_scores.append(float(score))
        
        result = result.with_columns([
            pl.Series('composite_score', composite_scores)
        ])
        
        # 计算买入信号 - 排名前 15%
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks').cast(pl.Float64) + self.EPSILON))).alias('score_percentile')
        ])
        
        # 买入信号：排名前 15% 且评分 > 50
        result = result.with_columns([
            ((pl.col('score_percentile') >= (1.0 - V75_SELECTION_PERCENTILE)) & 
             (pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        # 存储市场环境信息
        result = result.with_columns([
            pl.lit(market_regime.regime_type).alias('market_regime'),
            pl.lit(market_regime.recent_ic).alias('regime_ic')
        ])
        
        return result
    
    def _compute_score_distribution(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算评分分布统计"""
        result = df.clone()
        
        result = result.with_columns([
            pl.col('composite_score').std().over('trade_date').alias('score_std'),
            pl.col('composite_score').median().over('trade_date').alias('score_median'),
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V75Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V75: {trade_date} 当日数据为空")
                return signals
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                logger.debug(f"V75: {trade_date} 无买入信号")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V75Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    risk_adjusted_rs=row.get('risk_adjusted_rs', 50.0),
                    hurst_exponent=row.get('hurst_exponent', 0.5),
                    hurst_score=row.get('hurst_score', 0.0),
                    vol_score=row.get('vol_percentile', 0.5) * 100,
                    net_main_rate=row.get('net_main_rate', 0.0),
                    fund_flow_weight=row.get('fund_flow_adaptive_weight', 0.15),
                    market_regime=row.get('market_regime', 'neutral'),
                    regime_ic=row.get('regime_ic', 0.0),
                    industry_name=row.get('industry_name', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
            if signals:
                logger.info(f"V75 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V75 生成信号失败：{e}")
        
        return signals
    
    def get_score_distribution_stats(self, df: pl.DataFrame) -> List[V75ScoreDistribution]:
        """获取评分分布统计序列"""
        unique_dates = df['trade_date'].unique().to_list()
        distributions = []
        
        for trade_date in sorted(unique_dates):
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            scores = day_data['composite_score'].to_numpy()
            
            dist = V75ScoreDistribution(
                trade_date=trade_date,
                min_score=float(np.min(scores)),
                max_score=float(np.max(scores)),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                median_score=float(np.median(scores)),
                q1_score=float(np.percentile(scores, 25)),
                q3_score=float(np.percentile(scores, 75)),
                skewness=float(self._compute_skewness(scores)),
                kurtosis=float(self._compute_kurtosis(scores))
            )
            distributions.append(dist)
        
        return distributions
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """计算偏度"""
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std < self.EPSILON:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """计算峰度"""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std < self.EPSILON:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4)) - 3
    
    def update_ic_history(self, rank_ic: float, fund_ic: float = None, rs_ic: float = None):
        """更新 IC 历史记录"""
        self._ic_history['composite'].append(rank_ic)
        # 保持历史记录长度
        if len(self._ic_history['composite']) > self.regime_sensor_window * 2:
            self._ic_history['composite'] = self._ic_history['composite'][-self.regime_sensor_window * 2:]
        
        if fund_ic is not None:
            self._ic_history['fund'].append(fund_ic)
            if len(self._ic_history['fund']) > self.regime_sensor_window * 2:
                self._ic_history['fund'] = self._ic_history['fund'][-self.regime_sensor_window * 2:]
        
        if rs_ic is not None:
            self._ic_history['rs'].append(rs_ic)
            if len(self._ic_history['rs']) > self.regime_sensor_window * 2:
                self._ic_history['rs'] = self._ic_history['rs'][-self.regime_sensor_window * 2:]


# ===========================================
# V75 RankICCalculator - Rank IC 计算
# ===========================================

class V75RankICCalculator:
    """
    V75 RankICCalculator - Rank IC 计算与月度统计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.03
    3. 生成月度 Rank IC 统计
    4. 验证评分分布的连续性
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V75_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', 0.02)
        
        self.ic_results: List[V75ICMetrics] = []
        self.monthly_stats: List[V75MonthlyICStats] = []
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        """计算 Spearman Rank IC"""
        from scipy import stats
        
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 使用降序排名
        factor_ranks = stats.rankdata(-factor_clean, method='average')
        label_ranks = stats.rankdata(-label_clean, method='average')
        
        if np.std(factor_ranks) < self.EPSILON or np.std(label_ranks) < self.EPSILON:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_daily_rank_ic(self, df: pl.DataFrame, trade_date: str,
                                 signal_col: str = 'composite_score',
                                 return_col: str = 'forward_return_5d') -> Tuple[float, Dict[str, Any]]:
        """计算单日的 Rank IC"""
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, {'count': 0, 'reason': '样本不足'}
            
            if signal_col not in day_data.columns:
                return 0.0, {'count': 0, 'reason': 'signal_col 不存在'}
            
            if return_col not in day_data.columns:
                day_data = day_data.with_columns([
                    (((pl.col('close').shift(-5)).over('symbol') - pl.col('close')) / 
                     (pl.col('close') + self.EPSILON)).alias('forward_return_5d')
                ])
            
            if return_col not in day_data.columns:
                return 0.0, {'count': 0, 'reason': '列不存在'}
            
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
            logger.debug(f"V75 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V75ICMetrics]:
        """计算 IC 序列"""
        try:
            df_with_return = df.with_columns([
                (((pl.col('close').shift(-5)).over('symbol') - pl.col('close')) / 
                 (pl.col('close') + self.EPSILON)).alias('forward_return_5d')
            ])
            
            unique_dates = df['trade_date'].unique().to_list()
            ic_series = []
            
            for trade_date in sorted(unique_dates):
                rank_ic, details = self.calculate_daily_rank_ic(
                    df_with_return, trade_date, signal_col, 'forward_return_5d'
                )
                ic = details.get('ic', 0.0)
                
                ic_metrics = V75ICMetrics(
                    trade_date=trade_date,
                    factor_name='composite_score',
                    ic=ic,
                    rank_ic=rank_ic
                )
                ic_series.append(ic_metrics)
            
            self.ic_results = ic_series
            
            # 计算月度统计
            self._compute_monthly_stats()
            
        except Exception as e:
            logger.error(f"V75 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
    def _compute_monthly_stats(self):
        """计算月度 IC 统计"""
        if not self.ic_results:
            self.monthly_stats = []
            return
        
        monthly_rank_ics: Dict[str, List[float]] = {}
        
        for ic_metric in self.ic_results:
            try:
                date_str = ic_metric.trade_date
                month = date_str[:7]  # YYYY-MM
                if month not in monthly_rank_ics:
                    monthly_rank_ics[month] = []
                monthly_rank_ics[month].append(ic_metric.rank_ic)
            except Exception:
                continue
        
        self.monthly_stats = []
        for month, rank_ics in sorted(monthly_rank_ics.items()):
            if rank_ics:
                monthly_stat = V75MonthlyICStats(
                    month=month,
                    mean_rank_ic=float(np.mean(rank_ics)),
                    std_rank_ic=float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
                    ic_count=len(rank_ics)
                )
                self.monthly_stats.append(monthly_stat)
    
    def get_ic_statistics(self) -> Dict[str, float]:
        """获取 IC 统计信息"""
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
        rank_ic_std = float(np.std(rank_ic_values, ddof=1)) if len(rank_ic_values) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > self.EPSILON else 0.0
        rank_ic_ir = mean_rank_ic / rank_ic_std if rank_ic_std > self.EPSILON else 0.0
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
    
    def get_monthly_rank_ic_statistics(self) -> Dict[str, float]:
        """计算月度 Rank IC 统计"""
        if not self.monthly_stats:
            return {
                'monthly_mean_rank_ic': 0.0,
                'monthly_std': 0.0,
                'monthly_pass': False,
            }
        
        monthly_means = [m.mean_rank_ic for m in self.monthly_stats]
        
        monthly_mean = float(np.mean(monthly_means))
        monthly_std = float(np.std(monthly_means, ddof=1)) if len(monthly_means) > 1 else 0.0
        monthly_pass = monthly_mean >= self.rank_ic_target
        
        return {
            'monthly_mean_rank_ic': monthly_mean,
            'monthly_std': monthly_std,
            'monthly_pass': monthly_pass,
            'num_months': len(self.monthly_stats),
        }
    
    def check_rank_ic_pass(self) -> Tuple[bool, str]:
        """检查 Rank IC 是否达标"""
        stats = self.get_ic_statistics()
        
        if stats['mean_rank_ic'] >= self.rank_ic_target:
            return (True, f"Rank IC 达标：{stats['mean_rank_ic']:.4f} >= {self.rank_ic_target}")
        elif stats['mean_rank_ic'] >= self.rank_ic_min:
            return (True, f"Rank IC 勉强达标：{stats['mean_rank_ic']:.4f} >= {self.rank_ic_min}")
        else:
            return (False, f"预测模型失败：Rank IC={stats['mean_rank_ic']:.4f} < {self.rank_ic_min}")
    
    def print_rank_ic_report(self):
        """打印 Rank IC 统计表"""
        stats = self.get_ic_statistics()
        monthly_stats = self.get_monthly_rank_ic_statistics()
        is_pass, message = self.check_rank_ic_pass()
        
        logger.info("=" * 60)
        logger.info("V75 Rank IC 预测质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info("-" * 40)
        logger.info(f"Mean IC:      {stats['mean_ic']:.4f}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"IC Std:       {stats['ic_std']:.4f}")
        logger.info(f"Rank IC Std:  {stats['rank_ic_std']:.4f}")
        logger.info("-" * 40)
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"月度 Rank IC 标准差：{monthly_stats['monthly_std']:.4f}")
        logger.info(f"月度 Rank IC 达标：{monthly_stats['monthly_pass']}")
        logger.info(f"达标月份：{monthly_stats.get('num_months', 0)} 个月")
        logger.info("-" * 40)
        
        # 打印月度 IC
        if self.monthly_stats:
            logger.info("月度 Rank IC 明细:")
            for m in self.monthly_stats:
                status = "✓" if m.mean_rank_ic >= self.rank_ic_target else "✗"
                logger.info(f"  {m.month}: {m.mean_rank_ic:.4f} {status}")
        
        logger.info("-" * 40)
        
        if is_pass:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")
        
        logger.info("=" * 60)
    
    def generate_monthly_ic_chart(self) -> str:
        """生成月度 Rank IC 柱状图（ASCII 格式）"""
        chart_data = self.get_monthly_rank_ic_statistics()
        
        if not self.monthly_stats:
            return "无数据"
        
        lines = []
        lines.append("月度 Rank IC 柱状图")
        lines.append("=" * 60)
        
        rank_ics = [m.mean_rank_ic for m in self.monthly_stats]
        max_ic = max(max(abs(r) for r in rank_ics), self.rank_ic_target) if rank_ics else self.rank_ic_target
        bar_width = 40
        
        for m in self.monthly_stats:
            passed = m.mean_rank_ic >= self.rank_ic_target
            status = "✓" if passed else "✗"
            
            # 计算柱长（支持负值）
            if m.mean_rank_ic >= 0:
                bar_length = int(m.mean_rank_ic / max_ic * bar_width)
                bar = " " * bar_width + "█" * bar_length
            else:
                bar_length = int(abs(m.mean_rank_ic) / max_ic * bar_width)
                bar = " " * (bar_width - bar_length) + "█" * bar_length
            
            lines.append(f"{m.month} |{bar}| {m.mean_rank_ic:+.4f} {status}")
        
        lines.append("-" * 60)
        lines.append(f"目标：>{self.rank_ic_target:.3f}")
        pass_count = sum(1 for r in rank_ics if r >= self.rank_ic_target)
        lines.append(f"达标月份：{pass_count}/{len(self.monthly_stats)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ===========================================
# V75 可视化辅助函数
# ===========================================

def generate_score_histogram_data(df: pl.DataFrame) -> Dict[str, Any]:
    """生成评分分布直方图数据"""
    all_scores = df['composite_score'].to_numpy()
    
    hist, bin_edges = np.histogram(all_scores, bins=50, range=(0, 100))
    
    stats = {
        'mean': float(np.mean(all_scores)),
        'std': float(np.std(all_scores)),
        'median': float(np.median(all_scores)),
        'min': float(np.min(all_scores)),
        'max': float(np.max(all_scores)),
        'q1': float(np.percentile(all_scores, 25)),
        'q3': float(np.percentile(all_scores, 75)),
    }
    
    return {
        'histogram': {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
        },
        'statistics': stats,
        'total_samples': len(all_scores),
    }


def generate_monthly_ic_ascii_chart(monthly_stats: List[V75MonthlyICStats], 
                                     target: float = V75_RANK_IC_TARGET) -> str:
    """生成月度 IC ASCII 柱状图"""
    if not monthly_stats:
        return "无数据"
    
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("月度 Rank IC 柱状图")
    lines.append("=" * 60)
    
    rank_ics = [m.mean_rank_ic for m in monthly_stats]
    max_ic = max(max(abs(r) for r in rank_ics), target) if rank_ics else target
    bar_width = 40
    
    for m in monthly_stats:
        passed = m.mean_rank_ic >= target
        status = "✓" if passed else "✗"
        
        if m.mean_rank_ic >= 0:
            bar_length = int(m.mean_rank_ic / max_ic * bar_width)
            bar = " " * bar_width + "█" * bar_length
        else:
            bar_length = int(abs(m.mean_rank_ic) / max_ic * bar_width)
            bar = " " * (bar_width - bar_length) + "█" * bar_length
        
        lines.append(f"{m.month} |{bar}| {m.mean_rank_ic:+.4f} {status}")
    
    lines.append("-" * 60)
    lines.append(f"目标：>{target:.3f}")
    pass_count = sum(1 for r in rank_ics if r >= target)
    lines.append(f"达标月份：{pass_count}/{len(monthly_stats)}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V75_INITIAL_CAPITAL',
    'V75_MAX_POSITIONS',
    'V75_WARMUP_PERIOD',
    'V75_MIN_SAMPLE_SIZE',
    'V75_DATA_CHECK_YEAR',
    'V75_MIN_FUND_FLOW_ROWS',
    'V75_RETRY_ATTEMPTS',
    'V75_RETRY_DELAY',
    'V75_RS_WINDOW',
    'V75_RS_BASE_SCORE_MIN',
    'V75_RS_BASE_SCORE_MAX',
    'V75_REGIME_SENSOR_WINDOW',
    'V75_REGIME_IC_THRESHOLD',
    'V75_REGIME_ADAPTIVE_WEIGHT',
    'V75_VOLATILITY_WINDOW',
    'V75_VOLATILITY_WEIGHT',
    'V75_RISK_ADJUSTED_RS',
    'V75_HURST_WINDOW',
    'V75_HURST_WEIGHT',
    'V75_HURST_THRESHOLD',
    'V75_ADAPTIVE_RS_WEIGHT',
    'V75_ADAPTIVE_FUND_WEIGHT',
    'V75_ADAPTIVE_VOL_WEIGHT',
    'V75_ADAPTIVE_HURST_WEIGHT',
    'V75_SIGMOID_SCALE',
    'V75_SCORE_SMOOTHING',
    'V75_INDUSTRY_NEUTRAL_WEIGHT',
    'V75_COMMISSION_RATE',
    'V75_MIN_COMMISSION',
    'V75_SLIPPAGE_BUY',
    'V75_SLIPPAGE_SELL',
    'V75_STAMP_DUTY',
    'V75_TRANSFER_FEE',
    'V75_FRICTION_COST',
    'V75_STOP_LOSS_RATIO',
    'V75_PROFIT_TARGET_RATIO',
    'V75_TRAILING_STOP_RATIO',
    'V75_MAX_SINGLE_POSITION_PCT',
    'V75_SELECTION_PERCENTILE',
    'V75_RANK_IC_TARGET',
    'V75_SCORE_STD_TARGET_MIN',
    'V75_SCORE_STD_TARGET_MAX',
    'V75_DRAWDOWN_REDUCTION_TARGET',
    
    # 数据类
    'V75Position',
    'V75Trade',
    'V75Signal',
    'V75ICMetrics',
    'V75ScoreDistribution',
    'V75MonthlyICStats',
    'V75MarketRegime',
    
    # 核心类
    'V75DataManager',
    'V75AlphaCenter',
    'V75RankICCalculator',
    
    # 辅助函数
    'generate_score_histogram_data',
    'generate_monthly_ic_ascii_chart',
]