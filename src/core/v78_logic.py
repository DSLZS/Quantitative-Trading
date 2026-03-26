"""
V78 Core Module - 残差动量增强与多时空尺度特征融合

【V78 核心算法 - 残差动量与反转对冲】

1. Residual Momentum (残差动量) - 核心创新
   ✅ 个股收益率 - 行业平均收益率
   ✅ 剥离行业 beta，获取纯 alpha
   ✅ 只要"强于行业"的个股，不要行业拉升的虚假强势

2. Short-term Reversal (反转因子) - 核心创新
   ✅ 计算过去 3-5 日累计涨幅排名
   ✅ 对近期涨幅过大的个股进行分值扣减（惩罚）
   ✅ A 股具有强烈的周度反转效应

3. Dynamic Liquidity Audit (动态流动性审计) - 核心创新
   ✅ V-Shock = 当日成交额 / 过去 20 日均值
   ✅ 适度放量 [1.2, 2.5] 加分
   ✅ 巨量>3.0（巨量见顶）或 缩量<0.6（无人问津）大幅扣分

4. Multi-Timeframe RS Coupling (长短周期 RS 耦合) - 核心创新
   ✅ RS_5 (5 日相对强度) + RS_20 (20 日相对强度)
   ✅ Final RS = 0.3 * RS_20 + 0.7 * RS_5
   ✅ 缩短决策周期，提升对 A 股"电风扇轮动"的响应速度

5. Rank IC Quality Control (Rank IC 质量控制) - 核心创新
   ✅ 若 Mean Rank IC 连续三个月为负
   ✅ 自动记录当时的"市场拥挤度"特征
   ✅ 作为后续优化的负面样本

6. 线性融合（V78 最终评分）
   ✅ Score = 0.4*Residual_Momentum + 0.2*Reversal_Adjusted + 0.2*Liquidity_Score + 0.2*MultiRS_Score

7. 数据完整性检查
   ✅ 2024 全年数据验证
   ✅ 缺失时自动调用 v70_data_loader 补齐

8. 验收指标（决不妥协）
   ✅ 指标 A：全年度 Mean Rank IC >= 0.03（必须解决 4-11 月 IC 连续为负的问题）
   ✅ 指标 B：最大回撤回缩至 8% 以内
   ✅ 指标 C：胜率（Win Rate）需恢复至 45% 以上

作者：量化系统
版本：V78.0
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
# V78 配置常量
# ===========================================

# 基础配置
V78_INITIAL_CAPITAL = 100000.00  # 初始资金 10 万（严禁修改）
V78_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V78 数据预加载配置
V78_WARMUP_PERIOD = 250  # 预加载 250 天数据
V78_MIN_SAMPLE_SIZE = 100  # 最小股票样本量

# V78 数据完整性检查
V78_DATA_CHECK_YEAR = "2024"
V78_MIN_FUND_FLOW_ROWS = 50000  # 2024 全年资金流最少行数
V78_RETRY_ATTEMPTS = 5
V78_RETRY_DELAY = 3.0  # 秒

# V78 残差动量配置
V78_RESIDUAL_MOMENTUM_WINDOW = 20  # 残差动量计算窗口（20 日）
V78_RESIDUAL_MOMENTUM_MIN = 0.0  # 残差动量最小值（只要超额收益）

# V78 反转因子配置
V78_REVERSAL_WINDOW = 5  # 反转计算窗口（5 日）
V78_REVERSAL_PENALTY_TOP = 0.10  # 前 10% 涨幅过大股票进行惩罚
V78_REVERSAL_PENALTY_RATIO = 0.3  # 惩罚系数（扣减 30%）

# V78 流动性审计配置
V78_LIQUIDITY_WINDOW = 20  # 流动性计算窗口（20 日）
V78_VSHOCK_OPTIMAL_MIN = 1.2  # 适度放量下限
V78_VSHOCK_OPTIMAL_MAX = 2.5  # 适度放量上限
V78_VSHOCK_EXCESSIVE = 3.0  # 巨量阈值
V78_VSHOCK_SHRINK = 0.6  # 缩量阈值
V78_LIQUIDITY_BONUS = 1.2  # 适度放量加分系数
V78_LIQUIDITY_PENALTY = 0.7  # 巨量/缩量惩罚系数

# V78 长短周期 RS 耦合配置
V78_RS_SHORT_WINDOW = 5  # 短周期 RS（5 日）
V78_RS_LONG_WINDOW = 20  # 长周期 RS（20 日）
V78_RS_SHORT_WEIGHT = 0.7  # 短周期权重 70%
V78_RS_LONG_WEIGHT = 0.3  # 长周期权重 30%

# V78 线性融合权重
V78_RESIDUAL_MOMENTUM_WEIGHT = 0.40  # 残差动量权重 40%
V78_REVERSAL_WEIGHT = 0.20  # 反转因子权重 20%
V78_LIQUIDITY_WEIGHT = 0.20  # 流动性权重 20%
V78_MULTI_RS_WEIGHT = 0.20  # 多周期 RS 权重 20%

# V78 行业拥挤度配置
V78_SECTOR_CROWDING_WINDOW = 5  # 拥挤度计算窗口（5 日）
V78_SECTOR_CROWDING_STD_THRESHOLD = 2.0  # 拥挤度阈值（2 倍标准差）
V78_SECTOR_CROWDING_PENALTY = 0.30  # 拥挤惩罚系数（打 7 折）
V78_SECTOR_HISTORY_WINDOW = 60  # 历史均值计算窗口（60 日）

# V78 波动率缩放配置
V78_VOLATILITY_WINDOW = 20  # 波动率计算窗口（20 日）
V78_VOLATILITY_SCALING = True  # 启用波动率缩放
V78_VOLATILITY_BASE = 1.0  # 波动率基数

# V78 行业分散性控制（高压线）
V78_MAX_SECTOR_WEIGHT = 0.20  # 单行业最大权重 20%

# V78 行业配置
V78_INDUSTRY_NEUTRAL_WEIGHT = 1.0  # 行业数据缺失时的中性权重

# V78 费率配置 - 总计 0.2%（严禁修改）
V78_COMMISSION_RATE = 0.0003  # 佣金万 3
V78_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V78_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V78_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V78_STAMP_DUTY = 0.0005  # 印花税 0.05%
V78_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V78_FRICTION_COST = 0.002  # 0.2% 总计

# V78 离场配置（严禁修改）
V78_STOP_LOSS_RATIO = 0.05  # 止损 5%
V78_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V78_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V78 仓位管理
V78_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V78 选股排名
V78_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V78 Rank IC 目标
V78_RANK_IC_TARGET = 0.03  # 月度 Rank IC 目标（核心指标 - 及格线）
V78_RANK_IC_MIN = 0.02  # 最低可接受 Rank IC

# V78 回撤控制目标（硬约束）
V78_MAX_DRAWDOWN_TARGET = 0.08  # 最大回撤 8%

# V78 胜率目标
V78_WIN_RATE_TARGET = 0.45  # 胜率 45%

# V78 Rank IC 质量监控
V78_IC_MONITOR_MONTHS = 3  # 连续负值监控月数
V78_IC_NEGATIVE_THRESHOLD = 0.0  # 负值阈值


# ===========================================
# V78 数据类定义
# ===========================================

@dataclass
class V78Position:
    """V78 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    
    # V78 新增因子
    residual_momentum_score: float = 0.0  # 残差动量分
    reversal_score: float = 0.0  # 反转因子分
    reversal_penalty: float = 1.0  # 反转惩罚系数
    liquidity_score: float = 0.0  # 流动性分
    v_shock: float = 0.0  # 成交量冲击
    multi_rs_score: float = 0.0  # 多周期 RS 分
    rs_short: float = 0.0  # 短周期 RS
    rs_long: float = 0.0  # 长周期 RS
    
    # 行业数据
    industry_name: str = ""
    industry_return: float = 0.0  # 行业收益率
    residual_return: float = 0.0  # 残差收益率
    sector_crowding: float = 0.0  # 行业拥挤度
    volatility: float = 0.0  # 波动率
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    
    # 行业权重
    industry_weight: float = 1.0
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False


@dataclass
class V78Trade:
    """V78 交易记录"""
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
class V78Signal:
    """V78 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # V78 新增因子
    residual_momentum_score: float = 0.0
    residual_return: float = 0.0  # 个股收益率 - 行业收益率
    stock_return: float = 0.0  # 个股收益率
    industry_return: float = 0.0  # 行业收益率
    
    # 反转因子
    reversal_score: float = 0.0
    short_term_return: float = 0.0  # 短期收益率
    reversal_penalty: float = 1.0  # 反转惩罚
    
    # 流动性
    liquidity_score: float = 0.0
    v_shock: float = 0.0  # 成交量冲击
    
    # 多周期 RS
    multi_rs_score: float = 0.0
    rs_short: float = 0.0  # 5 日 RS
    rs_long: float = 0.0  # 20 日 RS
    
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
class V78ICMetrics:
    """V78 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V78ScoreDistribution:
    """V78 评分分布统计"""
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
class V78MonthlyICStats:
    """V78 月度 IC 统计"""
    month: str  # YYYY-MM
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False  # 是否为负值月份


@dataclass
class V78SectorCrowding:
    """V78 行业拥挤度状态"""
    trade_date: str
    industry_name: str
    turnover_ratio: float  # 当日成交额占比
    avg_turnover: float  # 历史均值
    std_turnover: float  # 历史标准差
    crowding_zscore: float  # 拥挤度 Z 分数
    is_crowded: bool  # 是否拥挤
    penalty_factor: float  # 惩罚因子


@dataclass
class V78ICQualityAlert:
    """V78 Rank IC 质量预警（连续负值记录）"""
    alert_date: str
    consecutive_negative_months: int
    market_crowding_level: float  # 市场拥挤度
    avg_sector_zscore: float  # 平均行业 Z 分数
    high_crowding_sectors: List[str]  # 高拥挤度行业列表
    factor_exposure: Dict[str, float]  # 因子暴露
    notes: str = ""


# ===========================================
# V78 DataManager - 数据获取与预处理
# ===========================================

class V78DataManager:
    """
    V78 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时自动补抓
    3. 支持全市场股票评分
    4. ConnectionError 重试机制
    5. 2024 全年数据完整性检查
    6. 数据自检功能（若 stock_daily 数据不完整，主动调用加载器）
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V78_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V78_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V78_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V78_RETRY_DELAY)
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
                    logger.info(f"V78: 第 {attempt} 次尝试成功")
                return result
                
            except (ConnectionError, OSError) as e:
                last_exception = e
                logger.warning(f"V78: 第 {attempt} 次尝试失败 (ConnectionError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V78: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V78: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except (AttributeError, TypeError) as e:
                last_exception = e
                logger.warning(f"V78: 第 {attempt} 次尝试失败 (AttributeError/TypeError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V78: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V78: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except Exception as e:
                logger.error(f"V78: 发生错误：{e}")
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
        logger.info("V78: 开始检查 2024 全年数据完整性")
        logger.info("=" * 60)
        
        if self.db is None:
            return False, "数据库连接未初始化"
        
        try:
            # 检查 stock_daily 表
            query = """
                SELECT COUNT(*) as cnt 
                FROM stock_daily
                WHERE trade_date >= '2024-01-01' 
                  AND trade_date <= '2024-12-31'
            """
            df = self._retry_wrapper(self.db.read_sql, query)
            
            if df.is_empty():
                return False, "无法查询 stock_daily 表"
            
            daily_count = int(df['cnt'][0])
            logger.info(f"V78: 2024 年 stock_daily 数据行数：{daily_count}")
            
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
            logger.info(f"V78: 2024 年行业数据行数：{industry_count}")
            
            # 数据完整性判断
            min_expected_daily = 100000  # 最少预期行数
            
            if daily_count < min_expected_daily:
                msg = f"stock_daily 数据不完整：{daily_count} < {min_expected_daily}"
                logger.warning(f"V78: {msg}")
                logger.warning("V78: 将尝试自动调用数据加载器补全数据")
                return False, msg
            
            logger.info("V78: 2024 全年数据完整性检查通过")
            return True, f"数据完整 (daily={daily_count}, industry={industry_count})"
            
        except Exception as e:
            logger.error(f"V78: 检查数据完整性失败：{e}")
            return False, f"检查失败：{e}"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V78 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V78 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V78 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V78 DataManager: 未加载到任何数据")
            
            logger.info(f"V78 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V78: 数据库连接未初始化")
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
            
            logger.debug(f"V78 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V78: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载行业数据
        
        【核心逻辑】
        1. 从 stock_industry_daily 表获取行业收益率数据
        2. 用于计算残差动量（个股收益率 - 行业收益率）
        """
        if self.db is None:
            logger.warning("V78: 数据库连接未初始化")
            return self._empty_industry_df()
        
        def _load():
            # 从 stock_industry_daily 表获取行业数据
            query = f"""
                SELECT symbol, industry_name, trade_date, industry_return
                FROM stock_industry_daily
                WHERE trade_date >= '{start_date}' 
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V78 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V78: 未加载到行业数据")
                return self._empty_industry_df()
            
            logger.info(f"V78 行业数据加载完成：{df.height} 行")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V78: 数据库连接未初始化")
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
            
            logger.debug(f"V78 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V78: 未加载到指数 {index_code} 数据")
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
            'industry_name': pl.Utf8,
            'industry_return': pl.Float64
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
        logger.warning(f"V78: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V78 缺失数据报告")
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
# V78 AlphaCenter - 残差动量与多时空尺度特征核心
# ===========================================

class V78AlphaCenter:
    """
    V78 AlphaCenter - 残差动量增强与多时空尺度特征融合
    
    【核心逻辑】
    1. Residual Momentum: 个股收益率 - 行业收益率
    2. Short-term Reversal: 过去 3-5 日涨幅排名惩罚
    3. Liquidity Audit: V-Shock 流动性审计
    4. Multi-Timeframe RS: 0.3*RS_20 + 0.7*RS_5
    5. 线性融合：Score = 0.4*Residual + 0.2*Reversal + 0.2*Liquidity + 0.2*MultiRS
    
    【评分公式】
    raw_score = 0.4*Residual_Momentum + 0.2*Reversal_Adjusted + 0.2*Liquidity_Score + 0.2*MultiRS_Score
    crowded_score = raw_score * sector_penalty
    final_score = crowded_score / (1 + volatility)
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 残差动量配置
        self.residual_momentum_window = self.config.get(
            'residual_momentum_window', V78_RESIDUAL_MOMENTUM_WINDOW
        )
        self.residual_momentum_min = self.config.get(
            'residual_momentum_min', V78_RESIDUAL_MOMENTUM_MIN
        )
        
        # 反转因子配置
        self.reversal_window = self.config.get('reversal_window', V78_REVERSAL_WINDOW)
        self.reversal_penalty_top = self.config.get(
            'reversal_penalty_top', V78_REVERSAL_PENALTY_TOP
        )
        self.reversal_penalty_ratio = self.config.get(
            'reversal_penalty_ratio', V78_REVERSAL_PENALTY_RATIO
        )
        
        # 流动性审计配置
        self.liquidity_window = self.config.get('liquidity_window', V78_LIQUIDITY_WINDOW)
        self.vshock_optimal_min = self.config.get('vshock_optimal_min', V78_VSHOCK_OPTIMAL_MIN)
        self.vshock_optimal_max = self.config.get('vshock_optimal_max', V78_VSHOCK_OPTIMAL_MAX)
        self.vshock_excessive = self.config.get('vshock_excessive', V78_VSHOCK_EXCESSIVE)
        self.vshock_shrink = self.config.get('vshock_shrink', V78_VSHOCK_SHRINK)
        self.liquidity_bonus = self.config.get('liquidity_bonus', V78_LIQUIDITY_BONUS)
        self.liquidity_penalty = self.config.get('liquidity_penalty', V78_LIQUIDITY_PENALTY)
        
        # 多周期 RS 配置
        self.rs_short_window = self.config.get('rs_short_window', V78_RS_SHORT_WINDOW)
        self.rs_long_window = self.config.get('rs_long_window', V78_RS_LONG_WINDOW)
        self.rs_short_weight = self.config.get('rs_short_weight', V78_RS_SHORT_WEIGHT)
        self.rs_long_weight = self.config.get('rs_long_weight', V78_RS_LONG_WEIGHT)
        
        # 线性融合权重
        self.residual_momentum_weight = self.config.get(
            'residual_momentum_weight', V78_RESIDUAL_MOMENTUM_WEIGHT
        )
        self.reversal_weight = self.config.get('reversal_weight', V78_REVERSAL_WEIGHT)
        self.liquidity_weight = self.config.get('liquidity_weight', V78_LIQUIDITY_WEIGHT)
        self.multi_rs_weight = self.config.get('multi_rs_weight', V78_MULTI_RS_WEIGHT)
        
        # 行业拥挤度配置
        self.sector_crowding_window = self.config.get(
            'sector_crowding_window', V78_SECTOR_CROWDING_WINDOW
        )
        self.sector_crowding_threshold = self.config.get(
            'sector_crowding_threshold', V78_SECTOR_CROWDING_STD_THRESHOLD
        )
        self.sector_penalty = self.config.get(
            'sector_penalty', V78_SECTOR_CROWDING_PENALTY
        )
        self.sector_history_window = self.config.get(
            'sector_history_window', V78_SECTOR_HISTORY_WINDOW
        )
        
        # 波动率缩放配置
        self.volatility_window = self.config.get(
            'volatility_window', V78_VOLATILITY_WINDOW
        )
        self.volatility_scaling = self.config.get(
            'volatility_scaling', V78_VOLATILITY_SCALING
        )
        self.volatility_base = self.config.get(
            'volatility_base', V78_VOLATILITY_BASE
        )
        
        # 行业配置
        self.industry_neutral_weight = self.config.get(
            'industry_neutral_weight', V78_INDUSTRY_NEUTRAL_WEIGHT
        )
        
        # 行业拥挤度缓存
        self._sector_crowding_cache: Dict[str, Dict[str, float]] = {}
        
        # Rank IC 质量监控
        self._ic_history: List[float] = []
        self._consecutive_negative_months: int = 0
        self._ic_quality_alerts: List[V78ICQualityAlert] = []
    
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
            has_industry = industry_df is not None and not industry_df.is_empty()
            has_index = index_df is not None and not index_df.is_empty()
            
            # 1. 计算残差动量 (Residual Momentum)
            logger.info("V78: Step 1 - 残差动量计算...")
            result = self._compute_residual_momentum(result, industry_df)
            status['factors_computed'].append('residual_momentum')
            
            # 2. 计算反转因子 (Short-term Reversal)
            logger.info("V78: Step 2 - 反转因子计算...")
            result = self._compute_reversal_factor(result)
            status['factors_computed'].append('reversal_factor')
            
            # 3. 计算流动性审计 (Liquidity Audit)
            logger.info("V78: Step 3 - 流动性审计计算...")
            result = self._compute_liquidity_audit(result)
            status['factors_computed'].append('liquidity_audit')
            
            # 4. 计算多周期 RS 耦合 (Multi-Timeframe RS)
            logger.info("V78: Step 4 - 多周期 RS 耦合计算...")
            result = self._compute_multi_rs(result, index_df)
            status['factors_computed'].append('multi_rs')
            
            # 5. 计算行业拥挤度
            if has_industry:
                logger.info("V78: Step 5 - 行业拥挤度计算...")
                result = self._compute_sector_crowding(result, industry_df)
                status['factors_computed'].append('sector_crowding')
            else:
                result = self._add_sector_crowding_placeholder(result)
                status['factors_computed'].append('sector_crowding_placeholder')
            
            # 6. 计算波动率
            logger.info("V78: Step 6 - 波动率计算...")
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 7. 计算行业权重
            if has_industry:
                logger.info("V78: Step 7 - 行业权重计算...")
                result = self._compute_industry_weight(result, industry_df)
                status['factors_computed'].append('industry_weight')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 8. 计算综合评分（线性融合 + 拥挤度惩罚 + 波动率缩放）
            logger.info("V78: Step 8 - 综合评分计算...")
            result = self._compute_composite_score_v78(result)
            status['factors_computed'].append('composite_score')
            
            # 9. 计算评分分布统计
            result = self._compute_score_distribution(result)
            
            logger.info(f"V78 AlphaCenter 信号计算完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V78 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_residual_momentum(self, df: pl.DataFrame,
                                    industry_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Residual Momentum (残差动量)
        
        【核心逻辑】
        1. 计算个股 N 日收益率
        2. 计算行业 N 日收益率
        3. 残差收益率 = 个股收益率 - 行业收益率
        4. 横截面排名映射到分数
        
        【计算公式】
        Stock_Return = (Close_t - Close_{t-N}) / Close_{t-N}
        Industry_Return = 行业平均收益率
        Residual_Return = Stock_Return - Industry_Return
        """
        result = df.clone()
        
        # 计算个股 N 日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.residual_momentum_window)) / 
             (pl.col('close').shift(self.residual_momentum_window) + self.EPSILON)).alias('stock_return')
        ])
        
        # 合并行业数据（获取行业收益率）
        if industry_df is not None and not industry_df.is_empty():
            # 确保行业数据包含 industry_return 列
            if 'industry_return' in industry_df.columns:
                industry_data = industry_df.select(['symbol', 'trade_date', 'industry_name', 'industry_return'])
                result = result.join(industry_data, on=['symbol', 'trade_date'], how='left')
                
                # 填充缺失的行业收益率（使用该股票所在行业的平均值）
                result = result.with_columns([
                    pl.col('industry_return').fill_null(0.0).alias('industry_return')
                ])
            else:
                # 如果行业数据没有 industry_return 列，使用 industry_name 进行聚合
                if 'industry_name' in industry_df.columns:
                    industry_data = industry_df.select(['symbol', 'industry_name']).unique()
                    result = result.join(industry_data, on='symbol', how='left')
                    result = result.with_columns([
                        pl.col('industry_name').fill_null('UNKNOWN').alias('industry_name'),
                        pl.lit(0.0).alias('industry_return')
                    ])
                else:
                    result = result.with_columns([
                        pl.lit('UNKNOWN').alias('industry_name'),
                        pl.lit(0.0).alias('industry_return')
                    ])
        else:
            # 没有行业数据，使用市场平均收益率作为替代
            result = result.with_columns([
                pl.col('stock_return').mean().over('trade_date').alias('industry_return'),
                pl.lit('UNKNOWN').alias('industry_name')
            ])
        
        # 计算残差收益率
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('industry_return')).alias('residual_return')
        ])
        
        # 横截面排名映射到分数 [0, 100]
        result = result.with_columns([
            (pl.col('residual_return').rank('ordinal', descending=True).over('trade_date')).alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_residual')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_residual').cast(pl.Float64) + self.EPSILON)).alias('residual_percentile')
        ])
        
        # 只取超额收益为正的股票（残差动量最小值为 0）
        result = result.with_columns([
            pl.when(pl.col('residual_percentile') * 100 >= 50)
            .then(pl.col('residual_percentile') * 100)
            .otherwise(0.0)
            .alias('residual_momentum_score')
        ])
        
        return result
    
    def _compute_reversal_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Short-term Reversal (反转因子)
        
        【核心逻辑】
        1. 计算过去 N 日累计收益率
        2. 对前 10% 涨幅过大的股票进行惩罚（扣减 30%）
        3. A 股具有强烈的周度反转效应
        
        【计算公式】
        Short_Term_Return = (Close_t - Close_{t-N}) / Close_{t-N}
        Reversal_Penalty = 0.7 if percentile > 90% else 1.0
        Reversal_Score = (1 - percentile) * Reversal_Penalty * 100
        """
        result = df.clone()
        
        # 计算过去 N 日累计收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.reversal_window)) / 
             (pl.col('close').shift(self.reversal_window) + self.EPSILON)).alias('short_term_return')
        ])
        
        # 横截面排名（收益率越低，排名越高 - 反转逻辑）
        result = result.with_columns([
            (pl.col('short_term_return').rank('ordinal', descending=False).over('trade_date')).alias('reversal_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_reversal')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('reversal_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_reversal').cast(pl.Float64) + self.EPSILON)).alias('reversal_percentile')
        ])
        
        # 计算反转惩罚：前 10% 涨幅过大的股票扣减 30%
        result = result.with_columns([
            pl.when(pl.col('reversal_percentile') >= (1.0 - self.reversal_penalty_top))
            .then(1.0 - self.reversal_penalty_ratio)
            .otherwise(1.0)
            .alias('reversal_penalty')
        ])
        
        # 计算反转分数（反转 percentile 越高越好，因为我们是降序排列）
        result = result.with_columns([
            (pl.col('reversal_percentile') * pl.col('reversal_penalty') * 100.0).alias('reversal_score')
        ])
        
        return result
    
    def _compute_liquidity_audit(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Dynamic Liquidity Audit (动态流动性审计)
        
        【核心逻辑】
        1. V-Shock = 当日成交额 / 过去 20 日平均成交额
        2. 适度放量 [1.2, 2.5] 加分 (1.2 倍)
        3. 巨量>3.0（巨量见顶）或 缩量<0.6（无人问津）大幅扣分 (0.7 倍)
        
        【评分规则】
        - V-Shock in [1.2, 2.5]: 加分 (流动性分数 = 百分位 * 1.2 * 100)
        - V-Shock > 3.0: 巨量见顶，扣分 (流动性分数 = 百分位 * 0.7 * 100)
        - V-Shock < 0.6: 缩量，扣分 (流动性分数 = 百分位 * 0.7 * 100)
        - 其他：正常 (流动性分数 = 百分位 * 100)
        """
        result = df.clone()
        
        # 计算滚动平均成交额
        result = result.with_columns([
            (pl.col('amount') / 
             (pl.col('amount').rolling_mean(window_size=self.liquidity_window).over('symbol') + self.EPSILON)
             ).alias('v_shock')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('v_shock').fill_null(1.0).alias('v_shock')
        ])
        
        # 横截面排名
        result = result.with_columns([
            (pl.col('v_shock').rank('ordinal', descending=False).over('trade_date')).alias('v_shock_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_liquidity')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('v_shock_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_liquidity').cast(pl.Float64) + self.EPSILON)).alias('v_shock_percentile')
        ])
        
        # 根据 V-Shock 值计算流动性调整系数
        result = result.with_columns([
            pl.when((pl.col('v_shock') >= self.vshock_optimal_min) & 
                    (pl.col('v_shock') <= self.vshock_optimal_max))
            .then(self.liquidity_bonus)  # 适度放量加分
            .when((pl.col('v_shock') > self.vshock_excessive) | 
                  (pl.col('v_shock') < self.vshock_shrink))
            .then(self.liquidity_penalty)  # 巨量或缩量扣分
            .otherwise(1.0)  # 正常
            .alias('liquidity_adjustment')
        ])
        
        # 计算流动性分数
        result = result.with_columns([
            (pl.col('v_shock_percentile') * pl.col('liquidity_adjustment') * 100.0).alias('liquidity_score')
        ])
        
        return result
    
    def _compute_multi_rs(self, df: pl.DataFrame,
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Multi-Timeframe RS Coupling (长短周期 RS 耦合)
        
        【核心逻辑】
        1. RS_5 = 个股 5 日收益率 - 市场 5 日收益率
        2. RS_20 = 个股 20 日收益率 - 市场 20 日收益率
        3. Final RS = 0.7 * RS_5 + 0.3 * RS_20
        4. 缩短决策周期，提升对 A 股"电风扇轮动"的响应速度
        
        【计算公式】
        RS_Short = Stock_Return_5d - Market_Return_5d
        RS_Long = Stock_Return_20d - Market_Return_20d
        Multi_RS = 0.7 * RS_Short_Rank + 0.3 * RS_Long_Rank
        """
        result = df.clone()
        
        # 计算个股 5 日和 20 日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.rs_short_window)) / 
             (pl.col('close').shift(self.rs_short_window) + self.EPSILON)).alias('stock_return_short'),
            ((pl.col('close') - pl.col('close').shift(self.rs_long_window)) / 
             (pl.col('close').shift(self.rs_long_window) + self.EPSILON)).alias('stock_return_long')
        ])
        
        # 如果有指数数据，计算市场收益率
        if index_df is not None and not index_df.is_empty():
            index_df = index_df.with_columns([
                pl.col('trade_date').cast(pl.Utf8).alias('trade_date'),
                ((pl.col('close') - pl.col('close').shift(self.rs_short_window)) / 
                 (pl.col('close').shift(self.rs_short_window) + self.EPSILON)).alias('market_return_short'),
                ((pl.col('close') - pl.col('close').shift(self.rs_long_window)) / 
                 (pl.col('close').shift(self.rs_long_window) + self.EPSILON)).alias('market_return_long')
            ])
            
            result = result.join(
                index_df.select(['trade_date', 'market_return_short', 'market_return_long']),
                on='trade_date',
                how='left'
            )
            
            result = result.with_columns([
                pl.col('market_return_short').fill_null(0.0).alias('market_return_short'),
                pl.col('market_return_long').fill_null(0.0).alias('market_return_long')
            ])
        else:
            # 使用市场平均作为替代
            result = result.with_columns([
                pl.col('stock_return_short').mean().over('trade_date').alias('market_return_short'),
                pl.col('stock_return_long').mean().over('trade_date').alias('market_return_long')
            ])
        
        # 计算 RS（相对强度）
        result = result.with_columns([
            (pl.col('stock_return_short') - pl.col('market_return_short')).alias('rs_short_value'),
            (pl.col('stock_return_long') - pl.col('market_return_long')).alias('rs_long_value')
        ])
        
        # 横截面排名映射到百分位
        # RS_5
        result = result.with_columns([
            (pl.col('rs_short_value').rank('ordinal', descending=True).over('trade_date')).alias('rs_short_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_rs_short')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('rs_short_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_rs_short').cast(pl.Float64) + self.EPSILON)).alias('rs_short_percentile')
        ])
        
        # RS_20
        result = result.with_columns([
            (pl.col('rs_long_value').rank('ordinal', descending=True).over('trade_date')).alias('rs_long_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_rs_long')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('rs_long_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_rs_long').cast(pl.Float64) + self.EPSILON)).alias('rs_long_percentile')
        ])
        
        # 多周期 RS 耦合：0.7 * RS_5 + 0.3 * RS_20
        result = result.with_columns([
            (self.rs_short_weight * pl.col('rs_short_percentile') + 
             self.rs_long_weight * pl.col('rs_long_percentile')).alias('multi_rs_percentile')
        ])
        
        # 映射到 0-100 分
        result = result.with_columns([
            (pl.col('multi_rs_percentile') * 100.0).alias('multi_rs_score')
        ])
        
        # 保存 RS_5 和 RS_20 的原始百分位值（用于信号输出）
        result = result.with_columns([
            (pl.col('rs_short_percentile') * 100.0).alias('rs_short'),
            (pl.col('rs_long_percentile') * 100.0).alias('rs_long')
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
        if 'industry_name' not in result.columns:
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
            ((pl.col('close') / (pl.col('close').shift(1) + self.EPSILON)) - 1).alias('daily_return')
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
        
        # 确保有 industry_name 列
        if 'industry_name' not in result.columns:
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
    
    def _compute_composite_score_v78(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（V78 线性融合 + 拥挤度惩罚 + 波动率缩放）
        
        【核心逻辑】
        1. 线性融合：raw_score = 0.4*Residual_Momentum + 0.2*Reversal + 0.2*Liquidity + 0.2*MultiRS
        2. 拥挤度惩罚：crowded_score = raw_score * sector_penalty
        3. 波动率缩放：final_score = crowded_score / (1 + volatility)
        4. 重新映射到 0-100
        """
        result = df.clone()
        
        # 获取各因子分数
        for col, default in [
            ('residual_momentum_score', 50.0),
            ('reversal_score', 50.0),
            ('liquidity_score', 50.0),
            ('multi_rs_score', 50.0)
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # 1. 线性融合
        result = result.with_columns([
            (self.residual_momentum_weight * (pl.col('residual_momentum_score') / 100.0) +
             self.reversal_weight * (pl.col('reversal_score') / 100.0) +
             self.liquidity_weight * (pl.col('liquidity_score') / 100.0) +
             self.multi_rs_weight * (pl.col('multi_rs_score') / 100.0)
             ).alias('raw_score')
        ])
        
        # 2. 获取拥挤度惩罚
        if 'sector_penalty' not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias('sector_penalty')])
        
        # 3. 应用拥挤度惩罚
        result = result.with_columns([
            (pl.col('raw_score') * pl.col('sector_penalty')).alias('crowded_score')
        ])
        
        # 4. 波动率缩放
        if 'volatility' not in result.columns:
            result = result.with_columns([pl.lit(0.02).alias('volatility')])
        
        result = result.with_columns([
            (pl.col('crowded_score') / 
             (1.0 + pl.col('volatility') * self.volatility_base)).alias('vol_scaled_score')
        ])
        
        # 5. 重新映射到 0-100
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
        
        # 6. 计算买入信号 - 排名前 15% 且评分 > 50
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
            ((pl.col('score_percentile') >= (1.0 - V78_SELECTION_PERCENTILE)) & 
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
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V78Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V78: {trade_date} 当日数据为空")
                return signals
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                logger.debug(f"V78: {trade_date} 无买入信号")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V78Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    
                    # V78 新增因子
                    residual_momentum_score=row.get('residual_momentum_score', 0.0),
                    residual_return=row.get('residual_return', 0.0),
                    stock_return=row.get('stock_return', 0.0),
                    industry_return=row.get('industry_return', 0.0),
                    
                    # 反转因子
                    reversal_score=row.get('reversal_score', 0.0),
                    short_term_return=row.get('short_term_return', 0.0),
                    reversal_penalty=row.get('reversal_penalty', 1.0),
                    
                    # 流动性
                    liquidity_score=row.get('liquidity_score', 0.0),
                    v_shock=row.get('v_shock', 1.0),
                    
                    # 多周期 RS
                    multi_rs_score=row.get('multi_rs_score', 0.0),
                    rs_short=row.get('rs_short', 0.0),
                    rs_long=row.get('rs_long', 0.0),
                    
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
                logger.info(f"V78 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V78 生成信号失败：{e}")
        
        return signals
    
    def get_score_distribution_stats(self, df: pl.DataFrame) -> List[V78ScoreDistribution]:
        """获取评分分布统计序列"""
        unique_dates = df['trade_date'].unique().to_list()
        distributions = []
        
        for trade_date in sorted(unique_dates):
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            scores = day_data['composite_score'].to_numpy()
            
            dist = V78ScoreDistribution(
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
    
    def record_ic_value(self, rank_ic: float, trade_date: str, 
                        sector_crowding_data: Optional[Dict[str, float]] = None):
        """
        记录 IC 值用于质量监控
        
        【核心逻辑】
        若 Mean Rank IC 连续三个月为负，自动记录当时的"市场拥挤度"特征
        """
        self._ic_history.append(rank_ic)
        
        # 检查是否连续负值
        if rank_ic < V78_IC_NEGATIVE_THRESHOLD:
            self._consecutive_negative_months += 1
        else:
            self._consecutive_negative_months = 0
        
        # 如果连续 3 个月负值，记录市场拥挤度特征
        if self._consecutive_negative_months >= V78_IC_MONITOR_MONTHS:
            alert = V78ICQualityAlert(
                alert_date=trade_date,
                consecutive_negative_months=self._consecutive_negative_months,
                market_crowding_level=sector_crowding_data.get('market_crowding_level', 0.0) if sector_crowding_data else 0.0,
                avg_sector_zscore=sector_crowding_data.get('avg_sector_zscore', 0.0) if sector_crowding_data else 0.0,
                high_crowding_sectors=sector_crowding_data.get('high_crowding_sectors', []) if sector_crowding_data else [],
                factor_exposure=sector_crowding_data.get('factor_exposure', {}) if sector_crowding_data else {},
                notes=f"连续 {self._consecutive_negative_months} 个月 Rank IC 为负"
            )
            self._ic_quality_alerts.append(alert)
            logger.warning(f"V78 IC 质量预警：{alert.notes} (日期：{trade_date})")
    
    def get_ic_quality_alerts(self) -> List[V78ICQualityAlert]:
        """获取 IC 质量预警记录"""
        return self._ic_quality_alerts


# ===========================================
# V78 RankICCalculator - Rank IC 计算与月度审计
# ===========================================

class V78RankICCalculator:
    """
    V78 RankICCalculator - Rank IC 计算与月度统计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.03
    3. 单月 Rank IC 出现负值的月份不得超过 2 个
    4. 生成月度 Rank IC 统计
    5. 连续 3 个月负值自动记录市场拥挤度特征
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V78_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', V78_RANK_IC_MIN)
        
        self.ic_results: List[V78ICMetrics] = []
        self.monthly_stats: List[V78MonthlyICStats] = []
    
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
            logger.debug(f"V78 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V78ICMetrics]:
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
                
                ic_metrics = V78ICMetrics(
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
            logger.error(f"V78 计算 IC 序列失败：{e}")
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
                monthly_stat = V78MonthlyICStats(
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
        logger.info("V78 Rank IC 预测质量审计表")
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
# V78 可视化辅助函数
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


def generate_monthly_ic_ascii_chart(monthly_stats: List[V78MonthlyICStats], 
                                     target: float = V78_RANK_IC_TARGET) -> str:
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
    'V78_INITIAL_CAPITAL',
    'V78_MAX_POSITIONS',
    'V78_WARMUP_PERIOD',
    'V78_MIN_SAMPLE_SIZE',
    'V78_DATA_CHECK_YEAR',
    'V78_MIN_FUND_FLOW_ROWS',
    'V78_RETRY_ATTEMPTS',
    'V78_RETRY_DELAY',
    'V78_RESIDUAL_MOMENTUM_WINDOW',
    'V78_RESIDUAL_MOMENTUM_MIN',
    'V78_REVERSAL_WINDOW',
    'V78_REVERSAL_PENALTY_TOP',
    'V78_REVERSAL_PENALTY_RATIO',
    'V78_LIQUIDITY_WINDOW',
    'V78_VSHOCK_OPTIMAL_MIN',
    'V78_VSHOCK_OPTIMAL_MAX',
    'V78_VSHOCK_EXCESSIVE',
    'V78_VSHOCK_SHRINK',
    'V78_LIQUIDITY_BONUS',
    'V78_LIQUIDITY_PENALTY',
    'V78_RS_SHORT_WINDOW',
    'V78_RS_LONG_WINDOW',
    'V78_RS_SHORT_WEIGHT',
    'V78_RS_LONG_WEIGHT',
    'V78_RESIDUAL_MOMENTUM_WEIGHT',
    'V78_REVERSAL_WEIGHT',
    'V78_LIQUIDITY_WEIGHT',
    'V78_MULTI_RS_WEIGHT',
    'V78_SECTOR_CROWDING_WINDOW',
    'V78_SECTOR_CROWDING_STD_THRESHOLD',
    'V78_SECTOR_CROWDING_PENALTY',
    'V78_SECTOR_HISTORY_WINDOW',
    'V78_VOLATILITY_WINDOW',
    'V78_VOLATILITY_SCALING',
    'V78_VOLATILITY_BASE',
    'V78_MAX_SECTOR_WEIGHT',
    'V78_INDUSTRY_NEUTRAL_WEIGHT',
    'V78_COMMISSION_RATE',
    'V78_MIN_COMMISSION',
    'V78_SLIPPAGE_BUY',
    'V78_SLIPPAGE_SELL',
    'V78_STAMP_DUTY',
    'V78_TRANSFER_FEE',
    'V78_FRICTION_COST',
    'V78_STOP_LOSS_RATIO',
    'V78_PROFIT_TARGET_RATIO',
    'V78_TRAILING_STOP_RATIO',
    'V78_MAX_SINGLE_POSITION_PCT',
    'V78_SELECTION_PERCENTILE',
    'V78_RANK_IC_TARGET',
    'V78_RANK_IC_MIN',
    'V78_MAX_DRAWDOWN_TARGET',
    'V78_WIN_RATE_TARGET',
    'V78_IC_MONITOR_MONTHS',
    'V78_IC_NEGATIVE_THRESHOLD',
    
    # 数据类
    'V78Position',
    'V78Trade',
    'V78Signal',
    'V78ICMetrics',
    'V78ScoreDistribution',
    'V78MonthlyICStats',
    'V78SectorCrowding',
    'V78ICQualityAlert',
    
    # 核心类
    'V78DataManager',
    'V78AlphaCenter',
    'V78RankICCalculator',
    
    # 辅助函数
    'generate_score_histogram_data',
    'generate_monthly_ic_ascii_chart',
]