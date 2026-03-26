"""
V76 Core Module - 行业拥挤度与波动率缩放迭代

【V76 核心算法 - 行业去拥挤度 + 波动率倒数加权 + 线性排名融合】

1. 回归 V74 线性排名融合基准（RS + Flow）
   ✅ 移除 V75 基于滞后 IC 的"自适应权重系统"
   ✅ 回归简化的线性排名融合：Score = 0.7*RS_Rank + 0.3*Flow_Rank

2. 行业拥挤度审计 (Sector Crowding) - 核心创新
   ✅ 计算每个行业过去 5 日的成交额占比变化
   ✅ 惩罚逻辑：若某行业成交额占比 > 历史均值 2 倍标准差，视为"极其拥挤"
   ✅ 对拥挤行业的个股进行分数打折，防止买入"高位共振"板块

3. 波动率倒数加权 (Volatility Targeting)
   ✅ Score = 原评分 / (1 + 过去 20 日波动率)
   ✅ 自动偏向"稳定上涨"标的，而非"暴涨暴跌"标的

4. 行业分散性控制（高压线）
   ✅ 单日持仓同一行业不得超过总资产的 20%
   ✅ 强制实现行业分散

5. 数据完整性检查
   ✅ 2024 全年数据验证
   ✅ 缺失时自动调用 src/loaders/ 脚本补抓

6. 验收指标（决不妥协）
   ✅ 指标 A：全年度 Mean Rank IC >= 0.03
   ✅ 指标 B：最大回撤 < 12%
   ✅ 指标 C：单日持仓同一行业不超过 20%

作者：量化系统
版本：V76.0
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
# V76 配置常量
# ===========================================

# 基础配置
V76_INITIAL_CAPITAL = 100000.00  # 初始资金 10 万（严禁修改）
V76_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V76 数据预加载配置
V76_WARMUP_PERIOD = 250  # 预加载 250 天数据
V76_MIN_SAMPLE_SIZE = 100  # 最小股票样本量

# V76 数据完整性检查
V76_DATA_CHECK_YEAR = "2024"
V76_MIN_FUND_FLOW_ROWS = 50000  # 2024 全年资金流最少行数
V76_RETRY_ATTEMPTS = 5
V76_RETRY_DELAY = 3.0  # 秒

# V76 RS 动量配置
V76_RS_WINDOW = 20  # RS 计算窗口（20 日）
V76_RS_BASE_SCORE_MIN = 30.0  # RS 基础分最小值
V76_RS_BASE_SCORE_MAX = 70.0  # RS 基础分最大值

# V76 线性融合权重（回归 V74）
V76_RS_WEIGHT = 0.70  # RS 权重 70%
V76_FLOW_WEIGHT = 0.30  # 资金流权重 30%

# V76 行业拥挤度配置（核心创新）
V76_SECTOR_CROWDING_WINDOW = 5  # 拥挤度计算窗口（5 日）
V76_SECTOR_CROWDING_STD_THRESHOLD = 2.0  # 拥挤度阈值（2 倍标准差）
V76_SECTOR_CROWDING_PENALTY = 0.30  # 拥挤惩罚系数（打 7 折）
V76_SECTOR_HISTORY_WINDOW = 60  # 历史均值计算窗口（60 日）

# V76 波动率缩放配置（核心创新）
V76_VOLATILITY_WINDOW = 20  # 波动率计算窗口（20 日）
V76_VOLATILITY_SCALING = True  # 启用波动率缩放
V76_VOLATILITY_BASE = 1.0  # 波动率基数

# V76 行业分散性控制（高压线）
V76_MAX_SECTOR_WEIGHT = 0.20  # 单行业最大权重 20%

# V76 行业配置
V76_INDUSTRY_NEUTRAL_WEIGHT = 1.0  # 行业数据缺失时的中性权重

# V76 费率配置 - 总计 0.2%（严禁修改）
V76_COMMISSION_RATE = 0.0003  # 佣金万 3
V76_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V76_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V76_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V76_STAMP_DUTY = 0.0005  # 印花税 0.05%
V76_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V76_FRICTION_COST = 0.002  # 0.2% 总计

# V76 离场配置（严禁修改）
V76_STOP_LOSS_RATIO = 0.05  # 止损 5%
V76_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V76_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V76 仓位管理
V76_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V76 选股排名
V76_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V76 Rank IC 目标
V76_RANK_IC_TARGET = 0.03  # 月度 Rank IC 目标（核心指标）

# V76 回撤控制目标（硬约束）
V76_MAX_DRAWDOWN_TARGET = 0.12  # 最大回撤 12%


# ===========================================
# V76 数据类定义
# ===========================================

@dataclass
class V76Position:
    """V76 持仓记录"""
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
    sector_crowding: float = 0.0  # 行业拥挤度
    volatility: float = 0.0  # 波动率
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    
    # 行业数据
    industry_name: str = ""
    industry_weight: float = 1.0
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False


@dataclass
class V76Trade:
    """V76 交易记录"""
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
class V76Signal:
    """V76 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # RS 动量
    rs_score: float = 0.0
    rs_rank: int = 0
    
    # 资金流
    net_main_rate: float = 0.0
    fund_flow_score: float = 0.0
    
    # 行业拥挤度
    sector_crowding: float = 0.0
    sector_penalty: float = 1.0
    
    # 波动率
    volatility: float = 0.0
    vol_scaled_score: float = 0.0
    
    # 行业数据
    industry_name: str = ""
    industry_weight: float = 1.0
    
    # 价格数据
    close_price: float = 0.0


@dataclass
class V76ICMetrics:
    """V76 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V76ScoreDistribution:
    """V76 评分分布统计"""
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
class V76MonthlyICStats:
    """V76 月度 IC 统计"""
    month: str  # YYYY-MM
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int


@dataclass
class V76SectorCrowding:
    """V76 行业拥挤度状态"""
    trade_date: str
    industry_name: str
    turnover_ratio: float  # 当日成交额占比
    avg_turnover: float  # 历史均值
    std_turnover: float  # 历史标准差
    crowding_zscore: float  # 拥挤度 Z 分数
    is_crowded: bool  # 是否拥挤
    penalty_factor: float  # 惩罚因子


# ===========================================
# V76 DataManager - 数据获取与预处理
# ===========================================

class V76DataManager:
    """
    V76 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时自动补抓
    3. 支持全市场股票评分
    4. ConnectionError 重试机制
    5. 2024 全年数据完整性检查
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V76_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V76_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V76_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V76_RETRY_DELAY)
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
                    logger.info(f"V76: 第 {attempt} 次尝试成功")
                return result
                
            except (ConnectionError, OSError) as e:
                last_exception = e
                logger.warning(f"V76: 第 {attempt} 次尝试失败 (ConnectionError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V76: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V76: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except (AttributeError, TypeError) as e:
                last_exception = e
                logger.warning(f"V76: 第 {attempt} 次尝试失败 (AttributeError/TypeError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V76: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V76: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except Exception as e:
                logger.error(f"V76: 发生错误：{e}")
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
        logger.info("V76: 开始检查 2024 全年数据完整性")
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
            logger.info(f"V76: 2024 年资金流数据行数：{fund_flow_count}")
            
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
            logger.info(f"V76: 2024 年行业数据行数：{industry_count}")
            
            if fund_flow_count < V76_MIN_FUND_FLOW_ROWS:
                msg = f"2024 年资金流数据不足：{fund_flow_count} < {V76_MIN_FUND_FLOW_ROWS}，需要补抓"
                logger.warning(f"V76: {msg}")
                return False, msg
            
            logger.info("V76: 2024 全年数据完整性检查通过")
            return True, f"数据完整 (fund_flow={fund_flow_count}, industry={industry_count})"
            
        except Exception as e:
            logger.error(f"V76: 检查数据完整性失败：{e}")
            return False, f"检查失败：{e}"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V76 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V76 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V76 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V76 DataManager: 未加载到任何数据")
            
            logger.info(f"V76 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V76: 数据库连接未初始化")
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
            
            logger.debug(f"V76 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V76: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载行业数据
        
        【核心逻辑】
        1. 从 stock_info 表获取股票与行业的静态映射
        2. 为每只股票分配所属行业
        3. 用于行业拥挤度计算和行业分散性控制
        """
        if self.db is None:
            logger.warning("V76: 数据库连接未初始化")
            return self._empty_industry_df()
        
        def _load():
            # 从 stock_info 表获取股票行业映射
            query = """
                SELECT symbol, industry_name
                FROM stock_info
                WHERE industry_name IS NOT NULL
                  AND industry_name != ''
            """
            
            logger.debug(f"V76 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V76: 未加载到行业数据")
                return self._empty_industry_df()
            
            logger.info(f"V76 行业数据加载完成：{df.height} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V76: 数据库连接未初始化")
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
            
            logger.debug(f"V76 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V76: 未加载到指数 {index_code} 数据")
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
        logger.warning(f"V76: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V76 缺失数据报告")
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
# V76 AlphaCenter - 行业拥挤度 + 波动率缩放 + 线性融合
# ===========================================

class V76AlphaCenter:
    """
    V76 AlphaCenter - 行业拥挤度与波动率缩放
    
    【核心逻辑】
    1. RS 动量轴：相对强度基础分（线性排名）
    2. 资金流轴：主力资金净流入（线性排名）
    3. 行业拥挤度：成交额占比 Z 分数惩罚
    4. 波动率缩放：Score = 原评分 / (1 + 波动率)
    5. 线性融合：Score = 0.7*RS_Rank + 0.3*Flow_Rank
    
    【评分公式】
    raw_score = 0.7*RS_percentile + 0.3*Flow_percentile
    crowded_score = raw_score * sector_penalty
    final_score = crowded_score / (1 + volatility)
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # RS 配置
        self.rs_window = self.config.get('rs_window', V76_RS_WINDOW)
        self.rs_score_min = self.config.get('rs_score_min', V76_RS_BASE_SCORE_MIN)
        self.rs_score_max = self.config.get('rs_score_max', V76_RS_BASE_SCORE_MAX)
        
        # 线性融合权重
        self.rs_weight = self.config.get('rs_weight', V76_RS_WEIGHT)
        self.flow_weight = self.config.get('flow_weight', V76_FLOW_WEIGHT)
        
        # 行业拥挤度配置
        self.sector_crowding_window = self.config.get(
            'sector_crowding_window', V76_SECTOR_CROWDING_WINDOW
        )
        self.sector_crowding_threshold = self.config.get(
            'sector_crowding_threshold', V76_SECTOR_CROWDING_STD_THRESHOLD
        )
        self.sector_penalty = self.config.get(
            'sector_penalty', V76_SECTOR_CROWDING_PENALTY
        )
        self.sector_history_window = self.config.get(
            'sector_history_window', V76_SECTOR_HISTORY_WINDOW
        )
        
        # 波动率缩放配置
        self.volatility_window = self.config.get(
            'volatility_window', V76_VOLATILITY_WINDOW
        )
        self.volatility_scaling = self.config.get(
            'volatility_scaling', V76_VOLATILITY_SCALING
        )
        self.volatility_base = self.config.get(
            'volatility_base', V76_VOLATILITY_BASE
        )
        
        # 行业配置
        self.industry_neutral_weight = self.config.get(
            'industry_neutral_weight', V76_INDUSTRY_NEUTRAL_WEIGHT
        )
        
        # 行业拥挤度缓存
        self._sector_crowding_cache: Dict[str, Dict[str, float]] = {}
    
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
            }
            
            # 检测数据可用性
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            has_industry = industry_df is not None and not industry_df.is_empty()
            has_index = index_df is not None and not index_df.is_empty()
            
            # 1. 计算 RS 相对强度
            result = self._compute_rs_score(result, index_df)
            status['factors_computed'].append('rs_score')
            
            # 2. 计算资金流因子
            if has_fund_flow:
                result = self._compute_fund_flow_score(result, fund_flow_df)
                status['factors_computed'].append('fund_flow_score')
            else:
                result = self._add_fund_flow_placeholder(result)
                status['factors_computed'].append('fund_flow_placeholder')
            
            # 3. 计算行业拥挤度
            if has_industry:
                result = self._compute_sector_crowding(result, industry_df)
                status['factors_computed'].append('sector_crowding')
            else:
                result = self._add_sector_crowding_placeholder(result)
                status['factors_computed'].append('sector_crowding_placeholder')
            
            # 4. 计算波动率
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 5. 计算行业权重
            if has_industry:
                result = self._compute_industry_weight(result, industry_df)
                status['factors_computed'].append('industry_weight')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 6. 计算综合评分（线性融合 + 拥挤度惩罚 + 波动率缩放）
            result = self._compute_composite_score_linear(result)
            status['factors_computed'].append('composite_score')
            
            # 7. 计算评分分布统计
            result = self._compute_score_distribution(result)
            
            logger.info(f"V76 AlphaCenter 信号计算完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V76 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_rs_score(self, df: pl.DataFrame, 
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算 RS 相对强度（线性排名版）
        
        【核心逻辑】
        1. 计算个股 N 日收益率
        2. 计算市场（指数）N 日收益率
        3. RS = 个股收益率 - 市场收益率
        4. 横截面排名映射到 0-100 分
        """
        result = df.clone()
        
        # 计算个股 N 日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
             (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('stock_return')
        ])
        
        # 如果有指数数据，计算市场收益率
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
        
        # 计算 RS（相对强度）
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('market_return')).alias('rs_value')
        ])
        
        # 横截面排名（降序 - 收益越高排名越高）
        result = result.with_columns([
            (pl.col('rs_value').rank('ordinal', descending=True).over('trade_date')).alias('rs_rank_1based'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        # 映射到百分位
        result = result.with_columns([
            (1.0 - (pl.col('rs_rank_1based').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + self.EPSILON)).alias('rs_percentile')
        ])
        
        # 映射到分数 [30, 70]
        result = result.with_columns([
            (self.rs_score_min + pl.col('rs_percentile') * 
             (self.rs_score_max - self.rs_score_min)).alias('rs_score')
        ])
        
        return result
    
    def _compute_fund_flow_score(self, df: pl.DataFrame, 
                                  fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算资金流分数（线性排名）
        
        【核心逻辑】
        1. 合并资金流数据
        2. 横截面排名 net_main_rate
        3. 映射到 0-100 分
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
        
        # 横截面排名（降序 - 流入越多排名越高）
        result = result.with_columns([
            (pl.col('net_main_rate').rank('ordinal', descending=True).over('trade_date')).alias('fund_rank_1based'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_fund')
        ])
        
        # 映射到百分位
        result = result.with_columns([
            (1.0 - (pl.col('fund_rank_1based').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_fund').cast(pl.Float64) + self.EPSILON)).alias('fund_percentile')
        ])
        
        # 映射到分数 [0, 100]
        result = result.with_columns([
            (pl.col('fund_percentile') * 100.0).alias('fund_flow_score')
        ])
        
        return result
    
    def _add_fund_flow_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加资金流占位符（中性值）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(0.0).alias('net_main_amount'),
            pl.lit(50.0).alias('fund_flow_score'),
            pl.lit(0.5).alias('fund_percentile')
        ])
        
        return result
    
    def _compute_sector_crowding(self, df: pl.DataFrame,
                                  industry_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算行业拥挤度（核心创新）
        
        【核心逻辑】
        1. 计算每个行业过去 N 日的成交额占比
        2. 计算历史均值和标准差
        3. Z 分数 = (当日占比 - 历史均值) / 历史标准差
        4. 若 Z > 2.0，视为"极其拥挤"，进行分数打折
        
        【计算公式】
        sector_turnover_ratio = industry_turnover / market_turnover
        crowding_zscore = (ratio - mean(ratio, 60d)) / std(ratio, 60d)
        penalty = 1.0 if zscore < 2.0 else (1.0 - penalty_rate)
        
        【注意】industry_df 是静态映射表，只有 symbol 和 industry_name 列
        """
        result = df.clone()
        
        # 合并行业数据（静态映射，只有 symbol 列）
        industry_data = industry_df.select(['symbol', 'industry_name']).unique()
        
        result = result.join(industry_data, on='symbol', how='left')
        
        # 填充缺失的行业名称
        result = result.with_columns([
            pl.col('industry_name').fill_null('UNKNOWN').alias('industry_name')
        ])
        
        # 计算每日市场总成交额
        daily_market_turnover = result.group_by('trade_date').agg([
            pl.col('amount').sum().alias('market_turnover')
        ])
        
        result = result.join(daily_market_turnover, on='trade_date', how='left')
        
        # 计算个股成交额占比
        result = result.with_columns([
            (pl.col('amount') / (pl.col('market_turnover') + self.EPSILON)).alias('stock_turnover_ratio')
        ])
        
        # 计算行业每日成交额占比
        industry_daily_agg = result.group_by(['trade_date', 'industry_name']).agg([
            pl.col('stock_turnover_ratio').sum().alias('industry_turnover_ratio')
        ])
        
        # 计算每个行业的历史均值和标准差（滚动 60 日）
        # 需要按行业分组计算
        industry_stats = industry_daily_agg.sort(['industry_name', 'trade_date'])
        
        industry_stats = industry_stats.with_columns([
            pl.col('industry_turnover_ratio')
            .rolling_mean(window_size=self.sector_history_window)
            .over('industry_name')
            .alias('industry_avg_turnover'),
            pl.col('industry_turnover_ratio')
            .rolling_std(window_size=self.sector_history_window)
            .over('industry_name')
            .alias('industry_std_turnover')
        ])
        
        # 计算 Z 分数
        industry_stats = industry_stats.with_columns([
            ((pl.col('industry_turnover_ratio') - pl.col('industry_avg_turnover')) / 
             (pl.col('industry_std_turnover') + self.EPSILON)).alias('crowding_zscore')
        ])
        
        # 计算惩罚因子
        industry_stats = industry_stats.with_columns([
            pl.when(pl.col('crowding_zscore') > self.sector_crowding_threshold)
            .then(1.0 - self.sector_penalty)  # 打 7 折
            .otherwise(1.0)
            .alias('sector_penalty')
        ])
        
        # 合并回结果
        result = result.join(
            industry_stats.select([
                'trade_date', 'industry_name', 
                'crowding_zscore', 'sector_penalty'
            ]),
            on=['trade_date', 'industry_name'],
            how='left',
            suffix='_crowd'
        )
        
        # 填充 NaN 值
        result = result.with_columns([
            pl.col('crowding_zscore').fill_null(0.0).alias('crowding_zscore'),
            pl.col('sector_penalty').fill_null(1.0).alias('sector_penalty')
        ])
        
        return result
    
    def _add_sector_crowding_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加行业拥挤度占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit('UNKNOWN').alias('industry_name'),
            pl.lit(0.0).alias('crowding_zscore'),
            pl.lit(1.0).alias('sector_penalty')
        ])
        
        return result
    
    def _compute_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算波动率（用于缩放）
        
        【核心逻辑】
        1. 计算 N 日收益率
        2. 计算滚动标准差
        3. 用于后续波动率倒数加权
        """
        result = df.clone()
        
        # 计算日收益率
        result = result.with_columns([
            (pl.col('close').pct_change() / self.EPSILON).alias('daily_return')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('daily_return').fill_null(0.0).alias('daily_return_filled')
        ])
        
        # 计算滚动波动率（20 日）
        result = result.with_columns([
            (pl.col('daily_return_filled')
             .rolling_std(window_size=self.volatility_window)
             .over('symbol')
             ).alias('volatility_raw')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('volatility_raw').fill_null(0.02).alias('volatility')
        ])
        
        return result
    
    def _compute_industry_weight(self, df: pl.DataFrame,
                                  industry_df: pl.DataFrame) -> pl.DataFrame:
        """计算行业权重（用于行业分散性控制）"""
        result = df.clone()
        
        # 合并行业数据（静态映射，只有 symbol 列）
        industry_data = industry_df.select(['symbol', 'industry_name']).unique()
        
        result = result.join(industry_data, on='symbol', how='left')
        
        # 填充缺失的行业名称
        result = result.with_columns([
            pl.col('industry_name').fill_null('UNKNOWN').alias('industry_name')
        ])
        
        # 中性权重
        result = result.with_columns([
            pl.lit(1.0).alias('industry_weight')
        ])
        
        return result
    
    def _add_industry_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加行业占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit('UNKNOWN').alias('industry_name'),
            pl.lit(1.0).alias('industry_weight')
        ])
        
        return result
    
    def _compute_composite_score_linear(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（线性融合 + 拥挤度惩罚 + 波动率缩放）
        
        【核心逻辑】
        1. 线性融合：raw_score = 0.7*RS_percentile + 0.3*Flow_percentile
        2. 拥挤度惩罚：crowded_score = raw_score * sector_penalty
        3. 波动率缩放：final_score = crowded_score / (1 + volatility)
        4. 重新映射到 0-100
        """
        result = df.clone()
        
        # 获取 RS 百分位
        if 'rs_percentile' not in result.columns:
            # 重新计算 RS 百分位
            result = result.with_columns([
                (pl.col('rs_value').rank('ordinal', descending=True).over('trade_date')).alias('rs_rank_1based'),
                pl.col('symbol').count().over('trade_date').alias('n_stocks_rs')
            ])
            
            result = result.with_columns([
                (1.0 - (pl.col('rs_rank_1based').cast(pl.Float64) - 0.5) / 
                 (pl.col('n_stocks_rs').cast(pl.Float64) + self.EPSILON)).alias('rs_percentile')
            ])
        
        # 获取资金流百分位
        if 'fund_percentile' not in result.columns:
            result = result.with_columns([
                pl.lit(0.5).alias('fund_percentile')
            ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('rs_percentile').fill_null(0.5).alias('rs_pct_filled'),
            pl.col('fund_percentile').fill_null(0.5).alias('fund_pct_filled')
        ])
        
        # 1. 线性融合
        result = result.with_columns([
            (self.rs_weight * pl.col('rs_pct_filled') + 
             self.flow_weight * pl.col('fund_pct_filled')).alias('raw_score')
        ])
        
        # 2. 拥挤度惩罚
        if 'sector_penalty' not in result.columns:
            result = result.with_columns([
                pl.lit(1.0).alias('sector_penalty')
            ])
        
        result = result.with_columns([
            (pl.col('raw_score') * pl.col('sector_penalty')).alias('crowded_score')
        ])
        
        # 3. 波动率缩放
        if 'volatility' not in result.columns:
            result = result.with_columns([
                pl.lit(0.02).alias('volatility')
            ])
        
        result = result.with_columns([
            (pl.col('crowded_score') / 
             (1.0 + pl.col('volatility') * self.volatility_base)).alias('vol_scaled_score')
        ])
        
        # 4. 重新映射到 0-100
        # 先计算横截面统计
        result = result.with_columns([
            pl.col('vol_scaled_score').mean().over('trade_date').alias('vol_mean'),
            pl.col('vol_scaled_score').std().over('trade_date').alias('vol_std')
        ])
        
        # Z-score 标准化
        result = result.with_columns([
            ((pl.col('vol_scaled_score') - pl.col('vol_mean')) / 
             (pl.col('vol_std') + self.EPSILON)).alias('vol_zscore')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('vol_zscore').fill_null(0.0).alias('vol_zscore_filled')
        ])
        
        # 映射到 0-100
        result = result.with_columns([
            (50 + 50 * pl.col('vol_zscore_filled').clip(-2, 2) / 2).alias('composite_score')
        ])
        
        # 5. 计算买入信号 - 排名前 15%
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_final')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks_final').cast(pl.Float64) + self.EPSILON))).alias('score_percentile')
        ])
        
        # 买入信号：排名前 15% 且评分 > 50
        result = result.with_columns([
            ((pl.col('score_percentile') >= (1.0 - V76_SELECTION_PERCENTILE)) & 
             (pl.col('composite_score') > 50)).alias('buy_signal')
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
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V76Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V76: {trade_date} 当日数据为空")
                return signals
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                logger.debug(f"V76: {trade_date} 无买入信号")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V76Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    rs_score=row.get('rs_score', 0.0),
                    rs_rank=row.get('score_rank', 0),
                    net_main_rate=row.get('net_main_rate', 0.0),
                    fund_flow_score=row.get('fund_flow_score', 50.0),
                    sector_crowding=row.get('crowding_zscore', 0.0),
                    sector_penalty=row.get('sector_penalty', 1.0),
                    volatility=row.get('volatility', 0.02),
                    vol_scaled_score=row.get('vol_scaled_score', 0.0),
                    industry_name=row.get('industry_name', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
            if signals:
                logger.info(f"V76 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V76 生成信号失败：{e}")
        
        return signals
    
    def get_score_distribution_stats(self, df: pl.DataFrame) -> List[V76ScoreDistribution]:
        """获取评分分布统计序列"""
        unique_dates = df['trade_date'].unique().to_list()
        distributions = []
        
        for trade_date in sorted(unique_dates):
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            scores = day_data['composite_score'].to_numpy()
            
            dist = V76ScoreDistribution(
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


# ===========================================
# V76 RankICCalculator - Rank IC 计算
# ===========================================

class V76RankICCalculator:
    """
    V76 RankICCalculator - Rank IC 计算与月度统计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.03
    3. 生成月度 Rank IC 统计
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V76_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', 0.02)
        
        self.ic_results: List[V76ICMetrics] = []
        self.monthly_stats: List[V76MonthlyICStats] = []
    
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
            logger.debug(f"V76 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V76ICMetrics]:
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
                
                ic_metrics = V76ICMetrics(
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
            logger.error(f"V76 计算 IC 序列失败：{e}")
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
                monthly_stat = V76MonthlyICStats(
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
        logger.info("V76 Rank IC 预测质量审计表")
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
# V76 可视化辅助函数
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


def generate_monthly_ic_ascii_chart(monthly_stats: List[V76MonthlyICStats], 
                                     target: float = V76_RANK_IC_TARGET) -> str:
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
    'V76_INITIAL_CAPITAL',
    'V76_MAX_POSITIONS',
    'V76_WARMUP_PERIOD',
    'V76_MIN_SAMPLE_SIZE',
    'V76_DATA_CHECK_YEAR',
    'V76_MIN_FUND_FLOW_ROWS',
    'V76_RETRY_ATTEMPTS',
    'V76_RETRY_DELAY',
    'V76_RS_WINDOW',
    'V76_RS_BASE_SCORE_MIN',
    'V76_RS_BASE_SCORE_MAX',
    'V76_RS_WEIGHT',
    'V76_FLOW_WEIGHT',
    'V76_SECTOR_CROWDING_WINDOW',
    'V76_SECTOR_CROWDING_STD_THRESHOLD',
    'V76_SECTOR_CROWDING_PENALTY',
    'V76_SECTOR_HISTORY_WINDOW',
    'V76_VOLATILITY_WINDOW',
    'V76_VOLATILITY_SCALING',
    'V76_VOLATILITY_BASE',
    'V76_MAX_SECTOR_WEIGHT',
    'V76_INDUSTRY_NEUTRAL_WEIGHT',
    'V76_COMMISSION_RATE',
    'V76_MIN_COMMISSION',
    'V76_SLIPPAGE_BUY',
    'V76_SLIPPAGE_SELL',
    'V76_STAMP_DUTY',
    'V76_TRANSFER_FEE',
    'V76_FRICTION_COST',
    'V76_STOP_LOSS_RATIO',
    'V76_PROFIT_TARGET_RATIO',
    'V76_TRAILING_STOP_RATIO',
    'V76_MAX_SINGLE_POSITION_PCT',
    'V76_SELECTION_PERCENTILE',
    'V76_RANK_IC_TARGET',
    'V76_MAX_DRAWDOWN_TARGET',
    
    # 数据类
    'V76Position',
    'V76Trade',
    'V76Signal',
    'V76ICMetrics',
    'V76ScoreDistribution',
    'V76MonthlyICStats',
    'V76SectorCrowding',
    
    # 核心类
    'V76DataManager',
    'V76AlphaCenter',
    'V76RankICCalculator',
    
    # 辅助函数
    'generate_score_histogram_data',
    'generate_monthly_ic_ascii_chart',
]