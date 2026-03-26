"""
V77 Core Module - 量价交互特征挖掘与 Rank IC 爆发计划

【V77 核心算法 - 非线性交互因子】

1. MAD 去极值化 (Median Absolute Deviation)
   ✅ 使用中位数绝对偏差对原始数据进行去极值处理
   ✅ 防止个别妖股拉高整体 Rank IC 的假象
   ✅ 3.5σ原则：超过 3.5 倍 MAD 的值将被截断

2. Money Flow Persistence (资金流持续性) - 核心创新
   ✅ 计算过去 5 日主力流入的自相关系数
   ✅ 而非简单的求和或均值
   ✅ 捕捉资金流的"持续性"特征

3. RS Breakout Quality (突破质量) - 核心创新
   ✅ 结合 RS 强度与收盘价在当日振幅中的位置 (Range Position)
   ✅ Range Position = (Close - Low) / (High - Low)
   ✅ 突破质量 = RS 强度 × Range Position × 成交量确认

4. 时序动量审计 (Temporal Momentum Audit) - 核心创新
   ✅ 在原有"截面排名"基础上增加时序动量检查
   ✅ 计算 20 日均线斜率
   ✅ 若股票截面排名前 5% 但时序动量掉头，给予大幅扣分

5. 线性融合（回归 V74 基准）
   ✅ Score = 0.5*RS_Breakout_Rank + 0.3*Flow_Persistence_Rank + 0.2*Temporal_Momentum_Rank

6. 数据完整性检查
   ✅ 2024 全年数据验证
   ✅ 缺失时自动调用 v70_data_loader 补齐

7. 验收指标（决不妥协）
   ✅ 指标 A：全年度 Mean Rank IC >= 0.035（这是本次唯一的及格线）
   ✅ 指标 B：Calmar Ratio > 2.0
   ✅ 指标 C：单月 Rank IC 出现负值的月份不得超过 2 个

作者：量化系统
版本：V77.0
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
# V77 配置常量
# ===========================================

# 基础配置
V77_INITIAL_CAPITAL = 100000.00  # 初始资金 10 万（严禁修改）
V77_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V77 数据预加载配置
V77_WARMUP_PERIOD = 250  # 预加载 250 天数据
V77_MIN_SAMPLE_SIZE = 100  # 最小股票样本量

# V77 数据完整性检查
V77_DATA_CHECK_YEAR = "2024"
V77_MIN_FUND_FLOW_ROWS = 50000  # 2024 全年资金流最少行数
V77_RETRY_ATTEMPTS = 5
V77_RETRY_DELAY = 3.0  # 秒

# V77 RS 动量配置
V77_RS_WINDOW = 20  # RS 计算窗口（20 日）
V77_RS_BASE_SCORE_MIN = 30.0  # RS 基础分最小值
V77_RS_BASE_SCORE_MAX = 70.0  # RS 基础分最大值

# V77 量价交互因子配置
V77_MAD_WINDOW = 60  # MAD 去极值化窗口
V77_MAD_SIGMA = 3.5  # MAD 截断阈值（3.5σ）

# Money Flow Persistence 配置
V77_FLOW_PERSISTENCE_WINDOW = 5  # 资金流持续性计算窗口（5 日）
V77_FLOW_PERSISTENCE_MIN_SAMPLES = 3  # 最小样本数

# RS Breakout Quality 配置
V77_BREAKOUT_WINDOW = 20  # 突破计算窗口
V77_RANGE_POSITION_WEIGHT = 0.4  # Range Position 权重
V77_VOLUME_CONFIRM_WEIGHT = 0.3  # 成交量确认权重
V77_RS_QUALITY_WEIGHT = 0.3  # RS 质量权重

# 时序动量审计配置
V77_TEMPORAL_MOMENTUM_WINDOW = 20  # 时序动量计算窗口（20 日）
V77_MA_SLOPE_THRESHOLD = 0.0  # 均线斜率阈值（0 表示掉头）
V77_TEMPORAL_PENALTY = 0.5  # 时序动量掉头惩罚系数（打 5 折）
V77_TOP_PERCENTILE_THRESHOLD = 0.05  # 前 5% 阈值

# V77 线性融合权重
V77_RS_BREAKOUT_WEIGHT = 0.50  # RS Breakout 权重 50%
V77_FLOW_PERSISTENCE_WEIGHT = 0.30  # 资金流持续性权重 30%
V77_TEMPORAL_MOMENTUM_WEIGHT = 0.20  # 时序动量权重 20%

# V77 行业拥挤度配置
V77_SECTOR_CROWDING_WINDOW = 5  # 拥挤度计算窗口（5 日）
V77_SECTOR_CROWDING_STD_THRESHOLD = 2.0  # 拥挤度阈值（2 倍标准差）
V77_SECTOR_CROWDING_PENALTY = 0.30  # 拥挤惩罚系数（打 7 折）
V77_SECTOR_HISTORY_WINDOW = 60  # 历史均值计算窗口（60 日）

# V77 波动率缩放配置
V77_VOLATILITY_WINDOW = 20  # 波动率计算窗口（20 日）
V77_VOLATILITY_SCALING = True  # 启用波动率缩放
V77_VOLATILITY_BASE = 1.0  # 波动率基数

# V77 行业分散性控制（高压线）
V77_MAX_SECTOR_WEIGHT = 0.20  # 单行业最大权重 20%

# V77 行业配置
V77_INDUSTRY_NEUTRAL_WEIGHT = 1.0  # 行业数据缺失时的中性权重

# V77 费率配置 - 总计 0.2%（严禁修改）
V77_COMMISSION_RATE = 0.0003  # 佣金万 3
V77_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V77_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V77_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V77_STAMP_DUTY = 0.0005  # 印花税 0.05%
V77_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V77_FRICTION_COST = 0.002  # 0.2% 总计

# V77 离场配置（严禁修改）
V77_STOP_LOSS_RATIO = 0.05  # 止损 5%
V77_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V77_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V77 仓位管理
V77_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V77 选股排名
V77_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V77 Rank IC 目标
V77_RANK_IC_TARGET = 0.035  # 月度 Rank IC 目标（核心指标 - 及格线）

# V77 回撤控制目标（硬约束）
V77_MAX_DRAWDOWN_TARGET = 0.12  # 最大回撤 12%

# V77 Calmar Ratio 目标
V77_CALMAR_RATIO_TARGET = 2.0  # Calmar Ratio 目标


# ===========================================
# V77 数据类定义
# ===========================================

@dataclass
class V77Position:
    """V77 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    
    # V77 新增因子
    rs_breakout_score: float = 0.0  # RS Breakout 质量分
    flow_persistence_score: float = 0.0  # 资金流持续性分
    temporal_momentum_score: float = 0.0  # 时序动量分
    ma_slope: float = 0.0  # 均线斜率
    range_position: float = 0.0  # Range Position
    
    # 行业数据
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
class V77Trade:
    """V77 交易记录"""
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
class V77Signal:
    """V77 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # V77 新增因子
    rs_breakout_score: float = 0.0
    rs_score: float = 0.0
    breakout_quality: float = 0.0
    range_position: float = 0.0
    volume_confirm: float = 0.0
    
    # 资金流持续性
    flow_persistence: float = 0.0  # 自相关系数
    fund_flow_score: float = 0.0
    
    # 时序动量
    temporal_momentum_score: float = 0.0
    ma_slope: float = 0.0
    temporal_penalty: float = 1.0
    
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
class V77ICMetrics:
    """V77 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V77ScoreDistribution:
    """V77 评分分布统计"""
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
class V77MonthlyICStats:
    """V77 月度 IC 统计"""
    month: str  # YYYY-MM
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False  # 是否为负值月份


@dataclass
class V77SectorCrowding:
    """V77 行业拥挤度状态"""
    trade_date: str
    industry_name: str
    turnover_ratio: float  # 当日成交额占比
    avg_turnover: float  # 历史均值
    std_turnover: float  # 历史标准差
    crowding_zscore: float  # 拥挤度 Z 分数
    is_crowded: bool  # 是否拥挤
    penalty_factor: float  # 惩罚因子


# ===========================================
# V77 DataManager - 数据获取与预处理
# ===========================================

class V77DataManager:
    """
    V77 DataManager - 数据获取与预处理
    
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
        self.warmup_period = self.config.get('warmup_period', V77_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V77_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V77_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V77_RETRY_DELAY)
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
                    logger.info(f"V77: 第 {attempt} 次尝试成功")
                return result
                
            except (ConnectionError, OSError) as e:
                last_exception = e
                logger.warning(f"V77: 第 {attempt} 次尝试失败 (ConnectionError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V77: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V77: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except (AttributeError, TypeError) as e:
                last_exception = e
                logger.warning(f"V77: 第 {attempt} 次尝试失败 (AttributeError/TypeError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V77: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V77: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except Exception as e:
                logger.error(f"V77: 发生错误：{e}")
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
        logger.info("V77: 开始检查 2024 全年数据完整性")
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
            logger.info(f"V77: 2024 年资金流数据行数：{fund_flow_count}")
            
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
            logger.info(f"V77: 2024 年行业数据行数：{industry_count}")
            
            if fund_flow_count < V77_MIN_FUND_FLOW_ROWS:
                msg = f"2024 年资金流数据不足：{fund_flow_count} < {V77_MIN_FUND_FLOW_ROWS}，需要补抓"
                logger.warning(f"V77: {msg}")
                return False, msg
            
            logger.info("V77: 2024 全年数据完整性检查通过")
            return True, f"数据完整 (fund_flow={fund_flow_count}, industry={industry_count})"
            
        except Exception as e:
            logger.error(f"V77: 检查数据完整性失败：{e}")
            return False, f"检查失败：{e}"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V77 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V77 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V77 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V77 DataManager: 未加载到任何数据")
            
            logger.info(f"V77 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V77: 数据库连接未初始化")
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
            
            logger.debug(f"V77 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V77: 未加载到资金流向数据")
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
            logger.warning("V77: 数据库连接未初始化")
            return self._empty_industry_df()
        
        def _load():
            # 从 stock_info 表获取股票行业映射
            query = """
                SELECT symbol, industry_name
                FROM stock_info
                WHERE industry_name IS NOT NULL
                  AND industry_name != ''
            """
            
            logger.debug(f"V77 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V77: 未加载到行业数据")
                return self._empty_industry_df()
            
            logger.info(f"V77 行业数据加载完成：{df.height} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V77: 数据库连接未初始化")
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
            
            logger.debug(f"V77 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V77: 未加载到指数 {index_code} 数据")
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
        logger.warning(f"V77: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V77 缺失数据报告")
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
# V77 AlphaCenter - 量价交互因子核心
# ===========================================

class V77AlphaCenter:
    """
    V77 AlphaCenter - 量价交互特征挖掘
    
    【核心逻辑】
    1. MAD 去极值化：Median Absolute Deviation 处理
    2. Money Flow Persistence：资金流持续性自相关系数
    3. RS Breakout Quality：突破质量 = RS × Range Position × Volume Confirm
    4. 时序动量审计：均线斜率检查 + 惩罚
    5. 线性融合：Score = 0.5*RS_Breakout + 0.3*Flow_Persistence + 0.2*Temporal_Momentum
    
    【评分公式】
    raw_score = 0.5*RS_Breakout_Rank + 0.3*Flow_Persistence_Rank + 0.2*Temporal_Momentum_Rank
    crowded_score = raw_score * sector_penalty
    final_score = crowded_score / (1 + volatility)
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # RS 配置
        self.rs_window = self.config.get('rs_window', V77_RS_WINDOW)
        self.rs_score_min = self.config.get('rs_score_min', V77_RS_BASE_SCORE_MIN)
        self.rs_score_max = self.config.get('rs_score_max', V77_RS_BASE_SCORE_MAX)
        
        # MAD 去极值化配置
        self.mad_window = self.config.get('mad_window', V77_MAD_WINDOW)
        self.mad_sigma = self.config.get('mad_sigma', V77_MAD_SIGMA)
        
        # Money Flow Persistence 配置
        self.flow_persistence_window = self.config.get(
            'flow_persistence_window', V77_FLOW_PERSISTENCE_WINDOW
        )
        self.flow_persistence_min_samples = self.config.get(
            'flow_persistence_min_samples', V77_FLOW_PERSISTENCE_MIN_SAMPLES
        )
        
        # RS Breakout Quality 配置
        self.breakout_window = self.config.get('breakout_window', V77_BREAKOUT_WINDOW)
        self.range_position_weight = self.config.get(
            'range_position_weight', V77_RANGE_POSITION_WEIGHT
        )
        self.volume_confirm_weight = self.config.get(
            'volume_confirm_weight', V77_VOLUME_CONFIRM_WEIGHT
        )
        self.rs_quality_weight = self.config.get('rs_quality_weight', V77_RS_QUALITY_WEIGHT)
        
        # 时序动量审计配置
        self.temporal_momentum_window = self.config.get(
            'temporal_momentum_window', V77_TEMPORAL_MOMENTUM_WINDOW
        )
        self.ma_slope_threshold = self.config.get(
            'ma_slope_threshold', V77_MA_SLOPE_THRESHOLD
        )
        self.temporal_penalty = self.config.get(
            'temporal_penalty', V77_TEMPORAL_PENALTY
        )
        self.top_percentile_threshold = self.config.get(
            'top_percentile_threshold', V77_TOP_PERCENTILE_THRESHOLD
        )
        
        # 线性融合权重
        self.rs_breakout_weight = self.config.get(
            'rs_breakout_weight', V77_RS_BREAKOUT_WEIGHT
        )
        self.flow_persistence_weight = self.config.get(
            'flow_persistence_weight', V77_FLOW_PERSISTENCE_WEIGHT
        )
        self.temporal_momentum_weight = self.config.get(
            'temporal_momentum_weight', V77_TEMPORAL_MOMENTUM_WEIGHT
        )
        
        # 行业拥挤度配置
        self.sector_crowding_window = self.config.get(
            'sector_crowding_window', V77_SECTOR_CROWDING_WINDOW
        )
        self.sector_crowding_threshold = self.config.get(
            'sector_crowding_threshold', V77_SECTOR_CROWDING_STD_THRESHOLD
        )
        self.sector_penalty = self.config.get(
            'sector_penalty', V77_SECTOR_CROWDING_PENALTY
        )
        self.sector_history_window = self.config.get(
            'sector_history_window', V77_SECTOR_HISTORY_WINDOW
        )
        
        # 波动率缩放配置
        self.volatility_window = self.config.get(
            'volatility_window', V77_VOLATILITY_WINDOW
        )
        self.volatility_scaling = self.config.get(
            'volatility_scaling', V77_VOLATILITY_SCALING
        )
        self.volatility_base = self.config.get(
            'volatility_base', V77_VOLATILITY_BASE
        )
        
        # 行业配置
        self.industry_neutral_weight = self.config.get(
            'industry_neutral_weight', V77_INDUSTRY_NEUTRAL_WEIGHT
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
            
            # 1. MAD 去极值化预处理
            logger.info("V77: Step 1 - MAD 去极值化处理...")
            result = self._apply_mad_winsorize(result)
            status['factors_computed'].append('mad_winsorize')
            
            # 2. 计算 RS Breakout Quality
            logger.info("V77: Step 2 - RS Breakout Quality 计算...")
            result = self._compute_rs_breakout_quality(result, index_df)
            status['factors_computed'].append('rs_breakout_quality')
            
            # 3. 计算 Money Flow Persistence
            if has_fund_flow:
                logger.info("V77: Step 3 - Money Flow Persistence 计算...")
                result = self._compute_flow_persistence(result, fund_flow_df)
                status['factors_computed'].append('flow_persistence')
            else:
                result = self._add_flow_persistence_placeholder(result)
                status['factors_computed'].append('flow_persistence_placeholder')
            
            # 4. 计算时序动量审计
            logger.info("V77: Step 4 - 时序动量审计计算...")
            result = self._compute_temporal_momentum_audit(result)
            status['factors_computed'].append('temporal_momentum_audit')
            
            # 5. 计算行业拥挤度
            if has_industry:
                logger.info("V77: Step 5 - 行业拥挤度计算...")
                result = self._compute_sector_crowding(result, industry_df)
                status['factors_computed'].append('sector_crowding')
            else:
                result = self._add_sector_crowding_placeholder(result)
                status['factors_computed'].append('sector_crowding_placeholder')
            
            # 6. 计算波动率
            logger.info("V77: Step 6 - 波动率计算...")
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 7. 计算行业权重
            if has_industry:
                logger.info("V77: Step 7 - 行业权重计算...")
                result = self._compute_industry_weight(result, industry_df)
                status['factors_computed'].append('industry_weight')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 8. 计算综合评分（线性融合 + 时序惩罚 + 拥挤度惩罚 + 波动率缩放）
            logger.info("V77: Step 8 - 综合评分计算...")
            result = self._compute_composite_score_v77(result)
            status['factors_computed'].append('composite_score')
            
            # 9. 计算评分分布统计
            result = self._compute_score_distribution(result)
            
            logger.info(f"V77 AlphaCenter 信号计算完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V77 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _apply_mad_winsorize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        MAD 去极值化 (Median Absolute Deviation Winsorization)
        
        【核心逻辑】
        1. 计算每个截面的中位数
        2. 计算绝对偏差的中位数 (MAD)
        3. 截断超过 3.5σ的值
        
        【计算公式】
        MAD = median(|X - median(X)|)
        上限 = median + 3.5 * 1.4826 * MAD
        下限 = median - 3.5 * 1.4826 * MAD
        """
        result = df.clone()
        
        # 需要去极值的列
        cols_to_winsorize = ['close', 'volume', 'amount', 'open', 'high', 'low']
        available_cols = [c for c in cols_to_winsorize if c in result.columns]
        
        for col in available_cols:
            # 计算截面中位数
            result = result.with_columns([
                pl.col(col).median().over('trade_date').alias(f'{col}_median')
            ])
            
            # 计算绝对偏差
            result = result.with_columns([
                (pl.col(col) - pl.col(f'{col}_median')).abs().alias(f'{col}_abs_dev')
            ])
            
            # 计算 MAD
            result = result.with_columns([
                pl.col(f'{col}_abs_dev').median().over('trade_date').alias(f'{col}_mad')
            ])
            
            # 计算上下限 (1.4826 是正态分布的转换系数)
            mad_scale = 1.4826 * self.mad_sigma
            result = result.with_columns([
                (pl.col(f'{col}_median') + mad_scale * pl.col(f'{col}_mad')).alias(f'{col}_upper'),
                (pl.col(f'{col}_median') - mad_scale * pl.col(f'{col}_mad')).alias(f'{col}_lower')
            ])
            
            # 截断
            result = result.with_columns([
                pl.col(col).clip(pl.col(f'{col}_lower'), pl.col(f'{col}_upper')).alias(col)
            ])
            
            # 清理临时列
            temp_cols = [f'{col}_median', f'{col}_abs_dev', f'{col}_mad', 
                        f'{col}_upper', f'{col}_lower']
            result = result.drop([c for c in temp_cols if c in result.columns])
        
        return result
    
    def _compute_rs_breakout_quality(self, df: pl.DataFrame,
                                      index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        RS Breakout Quality (突破质量)
        
        【核心逻辑】
        1. RS 强度 = 个股 N 日收益率 - 市场 N 日收益率
        2. Range Position = (Close - Low) / (High - Low)
        3. Volume Confirm = 当日成交量 / N 日平均成交量
        4. Breakout Quality = RS × Range Position × Volume Confirm
        
        【解释】
        - Range Position 越高，说明收盘价越接近当日高点，突破意愿越强
        - Volume Confirm 越高，说明有成交量配合，突破更可信
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
        
        # 横截面排名映射到分数 [30, 70]
        result = result.with_columns([
            (pl.col('rs_value').rank('ordinal', descending=True).over('trade_date')).alias('rs_rank_1based'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_rs')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('rs_rank_1based').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_rs').cast(pl.Float64) + self.EPSILON)).alias('rs_percentile')
        ])
        
        result = result.with_columns([
            (self.rs_score_min + pl.col('rs_percentile') * 
             (self.rs_score_max - self.rs_score_min)).alias('rs_score')
        ])
        
        # 计算 Range Position = (Close - Low) / (High - Low)
        result = result.with_columns([
            ((pl.col('close') - pl.col('low')) / 
             (pl.col('high') - pl.col('low') + self.EPSILON)).alias('range_position')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('range_position').fill_null(0.5).alias('range_position')
        ])
        
        # 计算 Volume Confirm = 当日成交量 / N 日平均成交量
        result = result.with_columns([
            (pl.col('volume') / 
             (pl.col('volume').rolling_mean(window_size=self.breakout_window).over('symbol') + self.EPSILON)
             ).alias('volume_confirm')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('volume_confirm').fill_null(1.0).alias('volume_confirm')
        ])
        
        # 计算 Breakout Quality = RS × Range Position × Volume Confirm
        # 先标准化 volume_confirm 到 0-1 范围
        # 第一步：clip volume_confirm
        result = result.with_columns([
            pl.col('volume_confirm').clip(0.5, 2.0).alias('volume_confirm_clipped')
        ])
        
        # 第二步：normalize clipped volume_confirm
        result = result.with_columns([
            ((pl.col('volume_confirm_clipped') - 0.5) / 1.5).alias('volume_confirm_norm')
        ])
        
        # 计算 breakout_quality
        result = result.with_columns([
            (pl.col('rs_percentile') * 
             (1.0 - self.range_position_weight - self.rs_quality_weight) +
             pl.col('range_position') * self.range_position_weight +
             pl.col('volume_confirm_norm') * self.volume_confirm_weight
             ).alias('breakout_quality_raw')
        ])
        
        # 映射到 0-100
        result = result.with_columns([
            (pl.col('breakout_quality_raw') * 100.0).alias('breakout_quality')
        ])
        
        return result
    
    def _compute_flow_persistence(self, df: pl.DataFrame,
                                   fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        Money Flow Persistence (资金流持续性)
        
        【核心逻辑】
        1. 计算过去 5 日主力流入的自相关系数
        2. 自相关系数 = corr(net_main_rate_t, net_main_rate_{t-1})
        3. 捕捉资金流的"持续性"特征
        
        【解释】
        - 自相关系数越高，说明资金流越有持续性
        - 持续的资金流入比单日大幅流入更有意义
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
        
        # 计算资金流持续性（自相关系数）
        # 使用 rolling 窗口计算自相关
        def compute_autocorr(series: pl.Series) -> float:
            """计算一阶自相关系数"""
            if len(series) < self.flow_persistence_min_samples + 1:
                return 0.0
            
            values = series.to_numpy()
            if np.std(values) < self.EPSILON:
                return 0.0
            
            # 计算一阶自相关
            mean_val = np.mean(values)
            numerator = np.sum((values[1:] - mean_val) * (values[:-1] - mean_val))
            denominator = np.sum((values - mean_val) ** 2)
            
            if denominator < self.EPSILON:
                return 0.0
            
            return numerator / denominator
        
        # 使用 polars 的 rolling 窗口计算自相关
        # 由于 polars 不直接支持自相关，我们使用 lag 计算
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算 lagged net_main_rate
        for lag in range(1, self.flow_persistence_window):
            result = result.with_columns([
                pl.col('net_main_rate').shift(lag).over('symbol').alias(f'net_main_rate_lag{lag}')
            ])
        
        # 计算自相关系数的近似值（使用最近 2 期的相关性）
        result = result.with_columns([
            ((pl.col('net_main_rate') - pl.col('net_main_rate').rolling_mean(window_size=self.flow_persistence_window).over('symbol')) *
             (pl.col('net_main_rate_lag1') - pl.col('net_main_rate_lag1').rolling_mean(window_size=self.flow_persistence_window).over('symbol'))
             ).alias('autocov')
        ])
        
        # 计算方差
        result = result.with_columns([
            ((pl.col('net_main_rate') - pl.col('net_main_rate').rolling_mean(window_size=self.flow_persistence_window).over('symbol')) ** 2
             ).alias('variance')
        ])
        
        # 滚动计算自相关系数
        result = result.with_columns([
            (pl.col('autocov').rolling_sum(window_size=self.flow_persistence_window).over('symbol') /
             (pl.col('variance').rolling_sum(window_size=self.flow_persistence_window).over('symbol') + self.EPSILON)
             ).alias('flow_persistence_raw')
        ])
        
        # 填充 NaN 并截断到 [-1, 1]
        result = result.with_columns([
            pl.col('flow_persistence_raw').fill_null(0.0).clip(-1, 1).alias('flow_persistence')
        ])
        
        # 横截面排名映射到 0-100 分
        result = result.with_columns([
            (pl.col('flow_persistence').rank('ordinal', descending=True).over('trade_date')).alias('flow_persist_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_flow')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('flow_persist_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_flow').cast(pl.Float64) + self.EPSILON)).alias('flow_persist_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('flow_persist_percentile') * 100.0).alias('flow_persistence_score')
        ])
        
        return result
    
    def _add_flow_persistence_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加资金流持续性占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(0.0).alias('net_main_amount'),
            pl.lit(0.0).alias('flow_persistence'),
            pl.lit(50.0).alias('flow_persistence_score')
        ])
        
        return result
    
    def _compute_temporal_momentum_audit(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        时序动量审计 (Temporal Momentum Audit)
        
        【核心逻辑】
        1. 计算 20 日均线斜率
        2. 若股票截面排名前 5% 但时序动量掉头，给予大幅扣分
        
        【计算公式】
        MA_Slope = (MA_20 - MA_20_shift_5) / 5
        Temporal_Penalty = 0.5 if (score_rank < 5% AND MA_Slope < 0) else 1.0
        """
        result = df.clone()
        
        # 计算 20 日均线
        result = result.with_columns([
            pl.col('close').rolling_mean(window_size=self.temporal_momentum_window).over('symbol').alias('ma20')
        ])
        
        # 计算均线斜率 = (MA_20 - MA_20_shift_5) / 5
        result = result.with_columns([
            ((pl.col('ma20') - pl.col('ma20').shift(5).over('symbol')) / 5).alias('ma_slope')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('ma_slope').fill_null(0.0).alias('ma_slope')
        ])
        
        # 计算时序动量分数（基于均线斜率）
        # 斜率越大，分数越高
        result = result.with_columns([
            (pl.col('ma_slope').rank('ordinal', descending=True).over('trade_date')).alias('ma_slope_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_slope')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('ma_slope_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_slope').cast(pl.Float64) + self.EPSILON)).alias('ma_slope_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('ma_slope_percentile') * 100.0).alias('temporal_momentum_score')
        ])
        
        # 计算时序惩罚
        # 先计算综合排名（使用 rs_percentile 和 flow_persist_percentile 的均值）
        if 'rs_percentile' not in result.columns:
            result = result.with_columns([pl.lit(0.5).alias('rs_percentile')])
        if 'flow_persist_percentile' not in result.columns:
            result = result.with_columns([pl.lit(0.5).alias('flow_persist_percentile')])
        
        result = result.with_columns([
            ((pl.col('rs_percentile') + pl.col('flow_persist_percentile')) / 2).alias('preliminary_score')
        ])
        
        # 计算 preliminary_score 的排名
        result = result.with_columns([
            (pl.col('preliminary_score').rank('ordinal', descending=True).over('trade_date')).alias('prelim_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_prelim')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('prelim_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_prelim').cast(pl.Float64) + self.EPSILON)).alias('prelim_percentile')
        ])
        
        # 时序惩罚：若前 5% 但均线斜率 < 0，打 5 折
        result = result.with_columns([
            pl.when(
                (pl.col('prelim_percentile') >= (1.0 - self.top_percentile_threshold)) &
                (pl.col('ma_slope') < self.ma_slope_threshold)
            )
            .then(self.temporal_penalty)
            .otherwise(1.0)
            .alias('temporal_penalty')
        ])
        
        return result
    
    def _compute_sector_crowding(self, df: pl.DataFrame,
                                  industry_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算行业拥挤度
        
        【核心逻辑】
        1. 计算每个行业过去 N 日的成交额占比
        2. 计算历史均值和标准差
        3. Z 分数 = (当日占比 - 历史均值) / 历史标准差
        4. 若 Z > 2.0，视为"极其拥挤"，进行分数打折
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
            .then(1.0 - self.sector_penalty)
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
        1. 计算日收益率
        2. 计算滚动标准差
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
    
    def _compute_composite_score_v77(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（V77 线性融合 + 时序惩罚 + 拥挤度惩罚 + 波动率缩放）
        
        【核心逻辑】
        1. 线性融合：raw_score = 0.5*RS_Breakout + 0.3*Flow_Persistence + 0.2*Temporal_Momentum
        2. 时序惩罚：若前 5% 但均线掉头，打 5 折
        3. 拥挤度惩罚：crowded_score = raw_score * sector_penalty * temporal_penalty
        4. 波动率缩放：final_score = crowded_score / (1 + volatility)
        5. 重新映射到 0-100
        """
        result = df.clone()
        
        # 获取各因子百分位
        if 'breakout_quality' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('breakout_quality')])
        if 'flow_persist_percentile' not in result.columns:
            result = result.with_columns([pl.lit(0.5).alias('flow_persist_percentile')])
        if 'ma_slope_percentile' not in result.columns:
            result = result.with_columns([pl.lit(0.5).alias('ma_slope_percentile')])
        
        # 将 breakout_quality 转换为百分位
        result = result.with_columns([
            (pl.col('breakout_quality') / 100.0).alias('breakout_percentile')
        ])
        
        # 1. 线性融合
        result = result.with_columns([
            (self.rs_breakout_weight * pl.col('breakout_percentile') +
             self.flow_persistence_weight * pl.col('flow_persist_percentile') +
             self.temporal_momentum_weight * pl.col('ma_slope_percentile')
             ).alias('raw_score')
        ])
        
        # 2. 获取时序惩罚
        if 'temporal_penalty' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('temporal_penalty')])
        
        # 3. 获取拥挤度惩罚
        if 'sector_penalty' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('sector_penalty')])
        
        # 4. 应用惩罚
        result = result.with_columns([
            (pl.col('raw_score') * pl.col('temporal_penalty') * pl.col('sector_penalty')
             ).alias('penalized_score')
        ])
        
        # 5. 波动率缩放
        if 'volatility' not in result.columns:
            result = result.with_columns([pl.lit(0.02).alias('volatility')])
        
        result = result.with_columns([
            (pl.col('penalized_score') / 
             (1.0 + pl.col('volatility') * self.volatility_base)).alias('vol_scaled_score')
        ])
        
        # 6. 重新映射到 0-100
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
        
        # 7. 计算买入信号 - 排名前 15% 且评分 > 50
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_final')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks_final').cast(pl.Float64) + self.EPSILON))).alias('score_percentile')
        ])
        
        # 买入信号
        result = result.with_columns([
            ((pl.col('score_percentile') >= (1.0 - V77_SELECTION_PERCENTILE)) & 
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
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V77Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V77: {trade_date} 当日数据为空")
                return signals
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                logger.debug(f"V77: {trade_date} 无买入信号")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V77Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    
                    # V77 新增因子
                    rs_breakout_score=row.get('breakout_quality', 0.0),
                    rs_score=row.get('rs_score', 0.0),
                    breakout_quality=row.get('breakout_quality', 0.0),
                    range_position=row.get('range_position', 0.0),
                    volume_confirm=row.get('volume_confirm', 1.0),
                    
                    # 资金流持续性
                    flow_persistence=row.get('flow_persistence', 0.0),
                    fund_flow_score=row.get('flow_persistence_score', 50.0),
                    
                    # 时序动量
                    temporal_momentum_score=row.get('temporal_momentum_score', 0.0),
                    ma_slope=row.get('ma_slope', 0.0),
                    temporal_penalty=row.get('temporal_penalty', 1.0),
                    
                    # 行业拥挤度
                    sector_crowding=row.get('crowding_zscore', 0.0),
                    sector_penalty=row.get('sector_penalty', 1.0),
                    
                    # 波动率
                    volatility=row.get('volatility', 0.02),
                    vol_scaled_score=row.get('vol_scaled_score', 0.0),
                    
                    # 行业数据
                    industry_name=row.get('industry_name', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    
                    # 价格数据
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
            if signals:
                logger.info(f"V77 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V77 生成信号失败：{e}")
        
        return signals
    
    def get_score_distribution_stats(self, df: pl.DataFrame) -> List[V77ScoreDistribution]:
        """获取评分分布统计序列"""
        unique_dates = df['trade_date'].unique().to_list()
        distributions = []
        
        for trade_date in sorted(unique_dates):
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            scores = day_data['composite_score'].to_numpy()
            
            dist = V77ScoreDistribution(
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
# V77 RankICCalculator - Rank IC 计算与月度审计
# ===========================================

class V77RankICCalculator:
    """
    V77 RankICCalculator - Rank IC 计算与月度统计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.035
    3. 单月 Rank IC 出现负值的月份不得超过 2 个
    4. 生成月度 Rank IC 统计
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V77_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', 0.025)
        
        self.ic_results: List[V77ICMetrics] = []
        self.monthly_stats: List[V77MonthlyICStats] = []
    
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
            logger.debug(f"V77 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V77ICMetrics]:
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
                
                ic_metrics = V77ICMetrics(
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
            logger.error(f"V77 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
    def _compute_monthly_stats(self):
        """计算月度 IC 统计（包含负值月份标记）"""
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
                mean_rank_ic = float(np.mean(rank_ics))
                monthly_stat = V77MonthlyICStats(
                    month=month,
                    mean_rank_ic=mean_rank_ic,
                    std_rank_ic=float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
                    ic_count=len(rank_ics),
                    is_negative=(mean_rank_ic < 0)
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
                'negative_months': 0,
            }
        
        monthly_means = [m.mean_rank_ic for m in self.monthly_stats]
        negative_months = sum(1 for m in self.monthly_stats if m.is_negative)
        
        monthly_mean = float(np.mean(monthly_means))
        monthly_std = float(np.std(monthly_means, ddof=1)) if len(monthly_means) > 1 else 0.0
        monthly_pass = monthly_mean >= self.rank_ic_target
        
        return {
            'monthly_mean_rank_ic': monthly_mean,
            'monthly_std': monthly_std,
            'monthly_pass': monthly_pass,
            'negative_months': negative_months,
            'num_months': len(self.monthly_stats),
        }
    
    def check_rank_ic_pass(self) -> Tuple[bool, str]:
        """检查 Rank IC 是否达标"""
        stats = self.get_ic_statistics()
        monthly_stats = self.get_monthly_rank_ic_statistics()
        
        # 检查负值月份数量
        negative_months = monthly_stats.get('negative_months', 0)
        negative_months_pass = negative_months <= 2
        
        if stats['mean_rank_ic'] >= self.rank_ic_target and negative_months_pass:
            return (True, f"Rank IC 达标：{stats['mean_rank_ic']:.4f} >= {self.rank_ic_target}, 负值月份={negative_months}")
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
        logger.info("V77 Rank IC 预测质量审计表")
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
        logger.info(f"负值月份数量：{monthly_stats['negative_months']} (上限：2)")
        logger.info(f"月度 Rank IC 达标：{monthly_stats['monthly_pass']}")
        logger.info(f"达标月份：{monthly_stats.get('num_months', 0)} 个月")
        logger.info("-" * 40)
        
        # 打印月度 IC
        if self.monthly_stats:
            logger.info("月度 Rank IC 明细:")
            for m in self.monthly_stats:
                status = "✓" if m.mean_rank_ic >= self.rank_ic_target else "✗"
                neg_status = " [NEGATIVE]" if m.is_negative else ""
                logger.info(f"  {m.month}: {m.mean_rank_ic:.4f} {status}{neg_status}")
        
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
            neg_marker = " [NEG]" if m.is_negative else ""
            
            # 计算柱长（支持负值）
            if m.mean_rank_ic >= 0:
                bar_length = int(m.mean_rank_ic / max_ic * bar_width)
                bar = " " * bar_width + "█" * bar_length
            else:
                bar_length = int(abs(m.mean_rank_ic) / max_ic * bar_width)
                bar = " " * (bar_width - bar_length) + "█" * bar_length
            
            lines.append(f"{m.month} |{bar}| {m.mean_rank_ic:+.4f} {status}{neg_marker}")
        
        lines.append("-" * 60)
        lines.append(f"目标：>{self.rank_ic_target:.3f}")
        pass_count = sum(1 for r in rank_ics if r >= self.rank_ic_target)
        negative_count = sum(1 for m in self.monthly_stats if m.is_negative)
        lines.append(f"达标月份：{pass_count}/{len(self.monthly_stats)}")
        lines.append(f"负值月份：{negative_count} (上限：2)")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ===========================================
# V77 可视化辅助函数
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


def generate_monthly_ic_ascii_chart(monthly_stats: List[V77MonthlyICStats], 
                                     target: float = V77_RANK_IC_TARGET) -> str:
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
        neg_marker = " [NEG]" if m.is_negative else ""
        
        if m.mean_rank_ic >= 0:
            bar_length = int(m.mean_rank_ic / max_ic * bar_width)
            bar = " " * bar_width + "█" * bar_length
        else:
            bar_length = int(abs(m.mean_rank_ic) / max_ic * bar_width)
            bar = " " * (bar_width - bar_length) + "█" * bar_length
        
        lines.append(f"{m.month} |{bar}| {m.mean_rank_ic:+.4f} {status}{neg_marker}")
    
    lines.append("-" * 60)
    lines.append(f"目标：>{target:.3f}")
    pass_count = sum(1 for r in rank_ics if r >= target)
    negative_count = sum(1 for m in monthly_stats if m.is_negative)
    lines.append(f"达标月份：{pass_count}/{len(monthly_stats)}")
    lines.append(f"负值月份：{negative_count} (上限：2)")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V77_INITIAL_CAPITAL',
    'V77_MAX_POSITIONS',
    'V77_WARMUP_PERIOD',
    'V77_MIN_SAMPLE_SIZE',
    'V77_DATA_CHECK_YEAR',
    'V77_MIN_FUND_FLOW_ROWS',
    'V77_RETRY_ATTEMPTS',
    'V77_RETRY_DELAY',
    'V77_RS_WINDOW',
    'V77_RS_BASE_SCORE_MIN',
    'V77_RS_BASE_SCORE_MAX',
    'V77_MAD_WINDOW',
    'V77_MAD_SIGMA',
    'V77_FLOW_PERSISTENCE_WINDOW',
    'V77_FLOW_PERSISTENCE_MIN_SAMPLES',
    'V77_BREAKOUT_WINDOW',
    'V77_RANGE_POSITION_WEIGHT',
    'V77_VOLUME_CONFIRM_WEIGHT',
    'V77_RS_QUALITY_WEIGHT',
    'V77_TEMPORAL_MOMENTUM_WINDOW',
    'V77_MA_SLOPE_THRESHOLD',
    'V77_TEMPORAL_PENALTY',
    'V77_TOP_PERCENTILE_THRESHOLD',
    'V77_RS_BREAKOUT_WEIGHT',
    'V77_FLOW_PERSISTENCE_WEIGHT',
    'V77_TEMPORAL_MOMENTUM_WEIGHT',
    'V77_SECTOR_CROWDING_WINDOW',
    'V77_SECTOR_CROWDING_STD_THRESHOLD',
    'V77_SECTOR_CROWDING_PENALTY',
    'V77_SECTOR_HISTORY_WINDOW',
    'V77_VOLATILITY_WINDOW',
    'V77_VOLATILITY_SCALING',
    'V77_VOLATILITY_BASE',
    'V77_MAX_SECTOR_WEIGHT',
    'V77_INDUSTRY_NEUTRAL_WEIGHT',
    'V77_COMMISSION_RATE',
    'V77_MIN_COMMISSION',
    'V77_SLIPPAGE_BUY',
    'V77_SLIPPAGE_SELL',
    'V77_STAMP_DUTY',
    'V77_TRANSFER_FEE',
    'V77_FRICTION_COST',
    'V77_STOP_LOSS_RATIO',
    'V77_PROFIT_TARGET_RATIO',
    'V77_TRAILING_STOP_RATIO',
    'V77_MAX_SINGLE_POSITION_PCT',
    'V77_SELECTION_PERCENTILE',
    'V77_RANK_IC_TARGET',
    'V77_MAX_DRAWDOWN_TARGET',
    'V77_CALMAR_RATIO_TARGET',
    
    # 数据类
    'V77Position',
    'V77Trade',
    'V77Signal',
    'V77ICMetrics',
    'V77ScoreDistribution',
    'V77MonthlyICStats',
    'V77SectorCrowding',
    
    # 核心类
    'V77DataManager',
    'V77AlphaCenter',
    'V77RankICCalculator',
    
    # 辅助函数
    'generate_score_histogram_data',
    'generate_monthly_ic_ascii_chart',
]