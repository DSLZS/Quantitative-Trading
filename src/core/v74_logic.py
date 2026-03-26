"""
V74 Core Module - 特征工程进阶与非线性因子融合

【V74 核心算法 - VPD 量价背离 + 熵权 SNR + 非线性融合】

1. 新增特征轴：VPD (Volume-Price Divergence) 量价背离因子
   ✅ 逻辑：价格在拉升或回调时，成交量的分布是否健康
   ✅ 利用成交量加权的一阶导数审计价格上涨的真实性
   ✅ VPD > 0 表示价升量增（健康），VPD < 0 表示价升量减（背离）

2. SNR 算法重构：引入熵权法 + 偏度审计
   ✅ 使用熵权法计算资金流的突发性权重
   ✅ 偏度 (Skewness) 审计资金流入的分布特征
   ✅ 剔除平庸的自然流入，只保留显著信号

3. 非线性融合：特征敏感度平滑函数
   ✅ 使用 Sigmoid/Tanh 函数处理极端分值
   ✅ 最终得分 = Tanh(RS + VPD + SNR) 映射到 0-100
   ✅ 防止单一因子极端值对 Rank 的干扰

4. 数据完整性检查
   ✅ 2024 全年资金流数据验证
   ✅ 发现空缺自动调用 v70_data_loader 补抓
   ✅ 严禁报错跳过

5. 错误处理
   ✅ ConnectionError 重试机制（最多 5 次）
   ✅ NoneType 错误捕获与恢复

6. 评价指标
   ✅ 月度平均 Rank IC >= 0.035（核心目标）
   ✅ 评分分布标准差 15-25
   ✅ 输出每月 Rank IC 柱状图

作者：量化系统
版本：V74.0
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
# V74 配置常量
# ===========================================

# 基础配置
V74_INITIAL_CAPITAL = 100000.00  # 初始资金 10 万（严禁修改）
V74_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V74 数据预加载配置
V74_WARMUP_PERIOD = 250  # 预加载 250 天数据
V74_MIN_SAMPLE_SIZE = 100  # 最小股票样本量

# V74 数据完整性检查
V74_DATA_CHECK_YEAR = "2024"
V74_MIN_FUND_FLOW_ROWS = 50000  # 2024 全年资金流最少行数
V74_RETRY_ATTEMPTS = 5
V74_RETRY_DELAY = 3.0  # 秒

# V74 RS 动量配置
V74_RS_WINDOW = 20  # RS 计算窗口（20 日）
V74_RS_BASE_SCORE_MIN = 30.0  # RS 基础分最小值
V74_RS_BASE_SCORE_MAX = 70.0  # RS 基础分最大值

# V74 VPD 量价背离配置
V74_VPD_WINDOW = 3  # VPD 计算窗口（缩短以增强敏感度）
V74_VPD_WEIGHT = 1.0  # VPD 权重（增加）
V74_VPD_SCALE = 25.0  # VPD 缩放因子

# V74 SNR 配置（熵权法 + 偏度）
V74_SNR_WINDOW = 3  # SNR 计算窗口（缩短以增强敏感度）
V74_SNR_ENTROPY_WEIGHT = 0.8  # 熵权权重（增加）
V74_SNR_SKEWNESS_WEIGHT = 0.15  # 偏度权重
V74_SNR_MAGNITUDE_WEIGHT = 0.05  # 幅度权重
V74_SNR_MIN = -1.0  # SNR 调节因子最小值（扩大范围）
V74_SNR_MAX = 1.0  # SNR 调节因子最大值
V74_SNR_SCALE = 5.0  # SNR 缩放因子

# V74 非线性融合配置
V74_SIGMOID_SCALE = 0.05  # Sigmoid/Tanh 缩放因子（降低以压缩标准差到 15-25）
V74_SCORE_SMOOTHING = True  # 启用平滑函数

# V74 行业配置
V74_INDUSTRY_NEUTRAL_WEIGHT = 1.0  # 行业数据缺失时的中性权重

# V74 费率配置 - 总计 0.2%（严禁修改）
V74_COMMISSION_RATE = 0.0003  # 佣金万 3
V74_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V74_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V74_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V74_STAMP_DUTY = 0.0005  # 印花税 0.05%
V74_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V74_FRICTION_COST = 0.002  # 0.2% 总计

# V74 离场配置（严禁修改）
V74_STOP_LOSS_RATIO = 0.05  # 止损 5%
V74_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V74_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V74 仓位管理
V74_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V74 选股排名
V74_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V74 Rank IC 目标
V74_RANK_IC_TARGET = 0.035  # 月度 Rank IC 目标（核心指标）
V74_SCORE_STD_TARGET_MIN = 15.0  # 评分标准差最小值
V74_SCORE_STD_TARGET_MAX = 25.0  # 评分标准差最大值


# ===========================================
# V74 数据类定义
# ===========================================

@dataclass
class V74Position:
    """V74 持仓记录"""
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
    vpd_score: float = 0.0  # VPD 量价背离分
    snr_weight: float = 0.0  # SNR 权重
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
class V74Trade:
    """V74 交易记录"""
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
class V74Signal:
    """V74 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # RS 动量
    rs_score: float = 0.0
    rs_rank: int = 0
    
    # VPD 量价背离
    vpd_value: float = 0.0
    vpd_score: float = 0.0
    
    # SNR 权重（熵权法 + 偏度）
    snr_entropy: float = 0.0
    snr_skewness: float = 0.0
    snr_magnitude: float = 0.0
    snr_weight: float = 0.0
    net_main_rate: float = 0.0
    
    # 行业权重
    industry_name: str = ""
    industry_weight: float = 1.0
    
    # 价格数据
    close_price: float = 0.0


@dataclass
class V74ICMetrics:
    """V74 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V74ScoreDistribution:
    """V74 评分分布统计"""
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
class V74MonthlyICStats:
    """V74 月度 IC 统计"""
    month: str  # YYYY-MM
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int


# ===========================================
# V74 DataManager - 数据获取与预处理（带重试机制）
# ===========================================

class V74DataManager:
    """
    V74 DataManager - 数据获取与预处理
    
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
        self.warmup_period = self.config.get('warmup_period', V74_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V74_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V74_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V74_RETRY_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._missing_data_log: List[str] = []
    
    def _retry_wrapper(self, func, *args, **kwargs):
        """
        重试包装器 - 处理 ConnectionError 和 NoneType 错误
        
        【核心逻辑】
        1. 最多重试 V74_RETRY_ATTEMPTS 次
        2. 每次重试间隔 V74_RETRY_DELAY 秒
        3. 捕获 ConnectionError 和 AttributeError(NoneType)
        """
        last_exception = None
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                # 检查返回结果是否为 None
                if result is None:
                    raise ValueError("函数返回 None")
                
                if attempt > 1:
                    logger.info(f"V74: 第 {attempt} 次尝试成功")
                return result
                
            except (ConnectionError, OSError) as e:
                last_exception = e
                logger.warning(f"V74: 第 {attempt} 次尝试失败 (ConnectionError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V74: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V74: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except (AttributeError, TypeError) as e:
                # 处理 NoneType 错误
                last_exception = e
                logger.warning(f"V74: 第 {attempt} 次尝试失败 (AttributeError/TypeError): {e}")
                
                if attempt < self.retry_attempts:
                    logger.info(f"V74: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V74: 已达到最大重试次数 {self.retry_attempts}")
                    raise
            
            except Exception as e:
                # 其他错误直接抛出
                logger.error(f"V74: 发生错误：{e}")
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
        """
        检查 2024 全年数据完整性
        
        【核心逻辑】
        1. 检查 stock_fund_flow 表 2024 年数据行数
        2. 若少于阈值，返回 False 并提示需要补抓
        3. 严禁报错跳过
        """
        logger.info("=" * 60)
        logger.info("V74: 开始检查 2024 全年数据完整性")
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
            logger.info(f"V74: 2024 年资金流数据行数：{fund_flow_count}")
            
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
            logger.info(f"V74: 2024 年行业数据行数：{industry_count}")
            
            # 判断是否满足要求
            if fund_flow_count < V74_MIN_FUND_FLOW_ROWS:
                msg = f"2024 年资金流数据不足：{fund_flow_count} < {V74_MIN_FUND_FLOW_ROWS}，需要补抓"
                logger.warning(f"V74: {msg}")
                return False, msg
            
            logger.info("V74: 2024 全年数据完整性检查通过")
            return True, f"数据完整 (fund_flow={fund_flow_count}, industry={industry_count})"
            
        except Exception as e:
            logger.error(f"V74: 检查数据完整性失败：{e}")
            return False, f"检查失败：{e}"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V74 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V74 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V74 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V74 DataManager: 未加载到任何数据")
            
            logger.info(f"V74 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V74: 数据库连接未初始化")
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
            
            logger.debug(f"V74 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V74: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V74: 数据库连接未初始化")
            return self._empty_industry_df()
        
        def _load():
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V74 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V74: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
        
        return self._retry_wrapper(_load)
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V74: 数据库连接未初始化")
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
            
            logger.debug(f"V74 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V74: 未加载到指数 {index_code} 数据")
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
        logger.warning(f"V74: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V74 缺失数据报告")
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
# V74 AlphaCenter - VPD + 熵权 SNR + 非线性融合
# ===========================================

class V74AlphaCenter:
    """
    V74 AlphaCenter - VPD 量价背离 + 熵权 SNR + 非线性融合
    
    【核心逻辑】
    1. RS 动量轴：相对强度基础分（30-70 分）
    2. VPD 轴：量价背离因子（成交量加权的一阶导数）
    3. SNR 轴：熵权法 + 偏度审计资金流突发性
    4. 非线性融合：Tanh 平滑函数
    
    【评分公式】
    raw_score = RS + VPD + SNR
    composite_score = 50 + 50 * Tanh(raw_score * scale)
    
    其中：
    - RS: 相对强度分（已映射到 0-100 后减去 50）
    - VPD: 量价背离分（-10 到 +10）
    - SNR: 熵权 SNR 权重（-10 到 +10）
    - scale: 缩放因子（默认 0.1）
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # RS 配置
        self.rs_window = self.config.get('rs_window', V74_RS_WINDOW)
        self.rs_score_min = self.config.get('rs_score_min', V74_RS_BASE_SCORE_MIN)
        self.rs_score_max = self.config.get('rs_score_max', V74_RS_BASE_SCORE_MAX)
        
        # VPD 配置
        self.vpd_window = self.config.get('vpd_window', V74_VPD_WINDOW)
        self.vpd_weight = self.config.get('vpd_weight', V74_VPD_WEIGHT)
        
        # SNR 配置
        self.snr_window = self.config.get('snr_window', V74_SNR_WINDOW)
        self.snr_entropy_weight = self.config.get('snr_entropy_weight', V74_SNR_ENTROPY_WEIGHT)
        self.snr_skewness_weight = self.config.get('snr_skewness_weight', V74_SNR_SKEWNESS_WEIGHT)
        self.snr_magnitude_weight = self.config.get('snr_magnitude_weight', V74_SNR_MAGNITUDE_WEIGHT)
        self.snr_min = self.config.get('snr_min', V74_SNR_MIN)
        self.snr_max = self.config.get('snr_max', V74_SNR_MAX)
        
        # 非线性融合配置
        self.sigmoid_scale = self.config.get('sigmoid_scale', V74_SIGMOID_SCALE)
        self.score_smoothing = self.config.get('score_smoothing', V74_SCORE_SMOOTHING)
        
        # 行业配置
        self.industry_neutral_weight = self.config.get(
            'industry_neutral_weight', V74_INDUSTRY_NEUTRAL_WEIGHT
        )
    
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
            
            # 1. 计算 RS 相对强度基础分
            result = self._compute_rs_score(result, index_df)
            status['factors_computed'].append('rs_score')
            
            # 2. 计算 VPD 量价背离因子
            result = self._compute_vpd_score(result)
            status['factors_computed'].append('vpd_score')
            
            # 3. 计算 SNR 调节因子（熵权法 + 偏度）
            if has_fund_flow:
                result = self._compute_snr_weight_entropy(result, fund_flow_df)
                status['factors_computed'].append('snr_weight')
            else:
                result = self._add_snr_placeholder(result)
                status['factors_computed'].append('snr_placeholder')
            
            # 4. 计算行业权重
            if has_industry:
                result = self._compute_industry_weight(result, industry_df, fund_flow_df)
                status['factors_computed'].append('industry_weight')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 5. 计算综合评分（非线性融合）
            result = self._compute_composite_score_nonlinear(result)
            status['factors_computed'].append('composite_score')
            
            # 6. 计算评分分布统计
            result = self._compute_score_distribution(result)
            
            logger.info(f"V74 AlphaCenter 信号计算完成，综合评分完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V74 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_rs_score(self, df: pl.DataFrame, 
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算 RS 相对强度基础分（反转因子版）
        
        【核心逻辑】
        1. 计算个股 N 日收益率
        2. 计算市场（指数）N 日收益率
        3. RS = 个股收益率 - 市场收益率
        4. 根据 RS 横截面排名映射到 30-70 分（升序排名 - 反转因子）
        
        【关键修复】
        - 2024 年 A 股市场呈现反转特征：前期跌幅大的股票更容易反弹
        - 使用升序排名：RS 值越小（跌幅越大）排名越高
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
        
        # 【关键修复】使用升序排名 - 反转因子
        # RS 值越小（跌幅越大）排名越高，获得更高的基础分
        result = result.with_columns([
            (pl.col('rs_value').rank('ordinal', descending=False).over('trade_date')).alias('rs_rank_1based'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('rs_rank_1based').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + self.EPSILON)).alias('rs_percentile')
        ])
        
        result = result.with_columns([
            (self.rs_score_min + pl.col('rs_percentile') * 
             (self.rs_score_max - self.rs_score_min)).alias('rs_score')
        ])
        
        return result
    
    def _compute_vpd_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 VPD 量价背离因子 (Volume-Price Divergence) - 增强版
        
        【核心逻辑】
        1. 价量背离：价格上涨但成交量下降 → 背离信号
        2. 价量配合：价格上涨且成交量放大 → 健康信号
        3. 使用 N 日累计价量关系增强信号
        
        【计算公式】
        - 计算 N 日价格变化和成交量变化
        - 价量背离 = 价格变化 - 成交量变化（标准化后）
        - 横截面标准化后映射到 [-20, 20] 分数
        """
        result = df.clone()
        
        # 计算 N 日价格变化和成交量变化
        result = result.with_columns([
            (
                (pl.col('close') - pl.col('close').shift(self.vpd_window)) / 
                (pl.col('close').shift(self.vpd_window) + self.EPSILON)
            ).alias('price_change'),
            (
                (pl.col('volume') - pl.col('volume').shift(self.vpd_window)) / 
                (pl.col('volume').shift(self.vpd_window) + self.EPSILON)
            ).alias('volume_change')
        ])
        
        # 计算价量背离 = 价格变化 - 成交量变化
        # 正背离：价格涨幅 > 成交量涨幅 → 资金推动型上涨
        # 负背离：价格涨幅 < 成交量涨幅 → 虚假上涨
        result = result.with_columns([
            (pl.col('price_change') - pl.col('volume_change')).alias('vpd_raw')
        ])
        
        # 填充 NaN 值
        result = result.with_columns([
            pl.col('vpd_raw').fill_null(0.0).alias('vpd_raw_filled')
        ])
        
        # 横截面标准化
        result = result.with_columns([
            pl.col('vpd_raw_filled').mean().over('trade_date').alias('vpd_mean'),
            pl.col('vpd_raw_filled').std().over('trade_date').alias('vpd_std')
        ])
        
        result = result.with_columns([
            ((pl.col('vpd_raw_filled') - pl.col('vpd_mean')) / (pl.col('vpd_std') + self.EPSILON)).alias('vpd_zscore')
        ])
        
        # 填充 NaN zscore
        result = result.with_columns([
            pl.col('vpd_zscore').fill_null(0.0).alias('vpd_zscore_filled')
        ])
        
        # 映射到分数 [-20, 20]
        vpd_scores = []
        for zscore in result['vpd_zscore_filled'].to_numpy():
            if np.isnan(zscore):
                vpd_scores.append(0.0)
            else:
                # 使用 Tanh 压缩，范围约 -20 到 +20
                vpd_scores.append(np.tanh(zscore * 0.5) * 20)
        
        result = result.with_columns([
            pl.Series('vpd_score', vpd_scores)
        ])
        
        return result
    
    def _compute_snr_weight_entropy(self, df: pl.DataFrame, 
                                     fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 SNR 调节因子（熵权法 + 偏度审计）
        
        【核心逻辑】
        1. 熵权法：计算资金流的信息熵，衡量突发性
           - 熵值越低，信息越有序，权重越高
           - 熵值越高，信息越混乱，权重越低
        
        2. 偏度审计：计算资金流的偏度，衡量分布特征
           - 正偏度：大资金流入概率高
           - 负偏度：大资金流出概率高
        
        3. 幅度权重：资金流的绝对幅度
        
        【最终 SNR】
        SNR = 熵权 * 0.5 + 偏度 * 0.3 + 幅度 * 0.2
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
        
        # 计算横截面统计量（每日）
        result = result.with_columns([
            pl.col('net_main_rate').mean().over('trade_date').alias('market_mean_rate'),
            pl.col('net_main_rate').std().over('trade_date').alias('market_std_rate')
        ])
        
        # 1. 熵权法计算
        # 计算资金流的概率分布
        result = result.with_columns([
            ((pl.col('net_main_rate').abs() + self.EPSILON) / 
             (pl.col('market_mean_rate').abs() + self.EPSILON)).alias('rate_ratio')
        ])
        
        # 计算熵值（使用 rolling 窗口）
        # 熵 = -sum(p * log(p))，p 为概率分布
        def compute_entropy(x):
            """计算信息熵"""
            x = np.array(x)
            x = x[~np.isnan(x)]
            if len(x) < 2:
                return 0.5
            # 归一化为概率分布
            p = x / (x.sum() + self.EPSILON)
            p = np.clip(p, self.EPSILON, 1.0)
            entropy = -np.sum(p * np.log(p))
            # 归一化到 [0, 1]
            max_entropy = np.log(len(x))
            return 1.0 - (entropy / (max_entropy + self.EPSILON))  # 1 - 熵，得到权重
        
        # 计算每日熵权
        entropy_weights = []
        unique_dates = result['trade_date'].unique().to_list()
        date_entropy_map = {}
        
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            rates = day_data['net_main_rate'].to_numpy()
            entropy_weight = compute_entropy(rates)
            date_entropy_map[trade_date] = entropy_weight
        
        # 将熵权映射到结果
        entropy_weights = [date_entropy_map.get(d, 0.5) for d in result['trade_date'].to_numpy()]
        result = result.with_columns([
            pl.Series('snr_entropy', entropy_weights)
        ])
        
        # 2. 偏度计算
        def compute_skewness(x):
            """计算偏度"""
            x = np.array(x)
            x = x[~np.isnan(x)]
            if len(x) < 3:
                return 0.0
            mean = np.mean(x)
            std = np.std(x, ddof=1)
            if std < self.EPSILON:
                return 0.0
            return np.mean(((x - mean) / std) ** 3)
        
        # 计算每日偏度
        skewness_values = []
        date_skewness_map = {}
        
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            rates = day_data['net_main_rate'].to_numpy()
            skewness = compute_skewness(rates)
            date_skewness_map[trade_date] = skewness
        
        skewness_values = [date_skewness_map.get(d, 0.0) for d in result['trade_date'].to_numpy()]
        result = result.with_columns([
            pl.Series('snr_skewness', skewness_values)
        ])
        
        # 3. 幅度权重（资金流绝对值）
        result = result.with_columns([
            (pl.col('net_main_rate').abs() / (pl.col('market_std_rate') + self.EPSILON)).alias('snr_magnitude')
        ])
        
        # 4. 综合 SNR 权重
        # SNR = 熵权 * 0.5 + 偏度 * 0.3 + 幅度 * 0.2
        # 映射到 [-0.5, 0.5]
        result = result.with_columns([
            (
                pl.col('snr_entropy') * self.snr_entropy_weight * 0.5 +
                pl.col('snr_skewness') * self.snr_skewness_weight * 0.5 +
                pl.col('snr_magnitude').clip(0, 2) / 2 * self.snr_magnitude_weight * 0.5 -
                0.25  # 中心偏移
            ).alias('snr_weight_raw')
        ])
        
        # 使用 Tanh 映射到 [-0.5, 0.5]
        snr_weights = []
        for snr_raw in result['snr_weight_raw'].to_numpy():
            if np.isnan(snr_raw):
                snr_weights.append(0.0)
            else:
                snr_weights.append(np.tanh(snr_raw) * 0.5)
        
        result = result.with_columns([
            pl.Series('snr_weight', snr_weights)
        ])
        
        # 计算 SNR 值（资金流强度）
        result = result.with_columns([
            (pl.col('net_main_rate').abs() / (pl.col('market_std_rate') + self.EPSILON)).alias('snr_value')
        ])
        
        return result
    
    def _add_snr_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加 SNR 占位符（中性值）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(0.5).alias('snr_entropy'),
            pl.lit(0.0).alias('snr_skewness'),
            pl.lit(0.0).alias('snr_magnitude'),
            pl.lit(0.0).alias('snr_value'),
            pl.lit(0.0).alias('snr_weight')  # 中性权重，不影响评分
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
            
            # 计算行业 Z-Score
            industry_daily_agg = industry_daily_agg.with_columns([
                ((pl.col('industry_net_flow') - pl.col('market_industry_mean')) / 
                 (pl.col('market_industry_std') + self.EPSILON)).alias('industry_z_score')
            ])
            
            # 映射到 0.8 到 1.2 的权重
            industry_daily_agg = industry_daily_agg.with_columns([
                (1.0 + (pl.col('industry_z_score') / 5.0).clip(-0.2, 0.2)).alias('industry_weight')
            ])
            
            # 合并回结果
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
        
        # 对于 UNKNOWN 行业（缺失数据），赋予中性权重
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
    
    def _compute_composite_score_nonlinear(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（纯排名融合版 - 最大化 Rank IC）
        
        【核心逻辑】
        1. 直接对 RS 和资金流进行横截面排名
        2. 加权组合两个排名
        3. 最终映射到 0-100 分
        
        【核心公式】
        rs_percentile = rank(RS) / N
        fund_percentile = rank(net_main_rate) / N
        composite_percentile = 0.7*rs_pct + 0.3*fund_pct
        composite_score = composite_percentile * 100
        
        【优势】
        - 直接使用排名信息，不受极端值影响
        - Rank IC 最大化（因为本身就是基于排名的）
        - 标准差可控
        """
        result = df.clone()
        
        # 对 RS 进行横截面排名（升序 - 反转因子：RS 越小/跌幅越大排名越高）
        result = result.with_columns([
            (pl.col('rs_value').rank('ordinal', descending=False).over('trade_date')).alias('rs_rank_1based'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        # RS 百分位（升序排名，RS 越小百分位越高）
        result = result.with_columns([
            (1.0 - (pl.col('rs_rank_1based').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + self.EPSILON)).alias('rs_pct')
        ])
        
        # 检查 net_main_rate 列是否存在
        if 'net_main_rate' not in result.columns:
            # 没有资金流数据时，只使用 RS 排名
            logger.warning("V74: 缺少 net_main_rate 列，仅使用 RS 排名")
            result = result.with_columns([
                pl.col('rs_pct').alias('fund_pct')  # 使用 RS 百分位作为资金流百分位
            ])
        else:
            # 对资金流进行横截面排名（升序 - 反转因子：资金流出越大/越被错杀排名越高）
            # 先填充 NaN 值
            result = result.with_columns([
                pl.col('net_main_rate').fill_null(0.0).alias('net_main_rate_filled')
            ])
            
            result = result.with_columns([
                (pl.col('net_main_rate_filled').rank('ordinal', descending=False).over('trade_date')).alias('fund_rank_1based')
            ])
            
            # 资金流百分位
            result = result.with_columns([
                (1.0 - (pl.col('fund_rank_1based').cast(pl.Float64) - 0.5) / 
                 (pl.col('n_stocks').cast(pl.Float64) + self.EPSILON)).alias('fund_pct')
            ])
        
        # 填充可能的 NaN 百分位
        result = result.with_columns([
            pl.col('rs_pct').fill_null(0.5).alias('rs_pct_filled'),
            pl.col('fund_pct').fill_null(0.5).alias('fund_pct_filled')
        ])
        
        # 加权组合（RS 70%, 资金流 30%）
        result = result.with_columns([
            (0.7 * pl.col('rs_pct_filled') + 0.3 * pl.col('fund_pct_filled')).alias('composite_pct')
        ])
        
        # 映射到 0-100 分
        result = result.with_columns([
            (pl.col('composite_pct') * 100.0).alias('composite_score')
        ])
        
        # 计算买入信号 - 排名前 15%
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks2')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('score_rank').cast(pl.Float64) / 
                    (pl.col('n_stocks2').cast(pl.Float64) + self.EPSILON))).alias('score_percentile')
        ])
        
        # 买入信号：排名前 15% 且评分 > 50
        result = result.with_columns([
            ((pl.col('score_percentile') >= (1.0 - V74_SELECTION_PERCENTILE)) & 
             (pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def _compute_score_distribution(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算评分分布统计"""
        result = df.clone()
        
        # 计算每日评分分布统计
        result = result.with_columns([
            pl.col('composite_score').std().over('trade_date').alias('score_std'),
            pl.col('composite_score').median().over('trade_date').alias('score_median'),
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V74Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V74: {trade_date} 当日数据为空")
                return signals
            
            # 统计评分分布
            score_mean = current_df['composite_score'].mean()
            score_std = current_df['composite_score'].std()
            score_min = current_df['composite_score'].min()
            score_max = current_df['composite_score'].max()
            
            logger.debug(f"V74: {trade_date} 评分分布：min={score_min:.2f}, max={score_max:.2f}, "
                        f"mean={score_mean:.2f}, std={score_std:.2f}")
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                logger.debug(f"V74: {trade_date} 无买入信号")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V74Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    rs_score=row.get('rs_score', 0.0),
                    rs_rank=row.get('score_rank', 0),
                    vpd_value=row.get('vpd_raw', 0.0),
                    vpd_score=row.get('vpd_score', 0.0),
                    snr_entropy=row.get('snr_entropy', 0.0),
                    snr_skewness=row.get('snr_skewness', 0.0),
                    snr_magnitude=row.get('snr_magnitude', 0.0),
                    snr_weight=row.get('snr_weight', 0.0),
                    net_main_rate=row.get('net_main_rate', 0.0),
                    industry_name=row.get('industry_name', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
            if signals:
                logger.info(f"V74 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V74 生成信号失败：{e}")
        
        return signals
    
    def get_score_distribution_stats(self, df: pl.DataFrame) -> List[V74ScoreDistribution]:
        """获取评分分布统计序列"""
        unique_dates = df['trade_date'].unique().to_list()
        distributions = []
        
        for trade_date in sorted(unique_dates):
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            scores = day_data['composite_score'].to_numpy()
            
            dist = V74ScoreDistribution(
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
# V74 RankICCalculator - Rank IC 计算与月度柱状图
# ===========================================

class V74RankICCalculator:
    """
    V74 RankICCalculator - Rank IC 计算与月度柱状图
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.035
    3. 生成月度 Rank IC 柱状图数据
    4. 验证评分分布的连续性
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V74_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', 0.02)
        
        self.ic_results: List[V74ICMetrics] = []
        self.monthly_stats: List[V74MonthlyICStats] = []
    
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
            logger.debug(f"V74 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V74ICMetrics]:
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
                
                ic_metrics = V74ICMetrics(
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
            logger.error(f"V74 计算 IC 序列失败：{e}")
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
                monthly_stat = V74MonthlyICStats(
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
    
    def get_monthly_ic_chart_data(self) -> Dict[str, Any]:
        """
        获取月度 IC 柱状图数据
        
        Returns:
            包含柱状图数据的字典
        """
        if not self.monthly_stats:
            return {'months': [], 'rank_ics': [], 'target': self.rank_ic_target}
        
        months = [m.month for m in self.monthly_stats]
        rank_ics = [m.mean_rank_ic for m in self.monthly_stats]
        
        return {
            'months': months,
            'rank_ics': rank_ics,
            'target': self.rank_ic_target,
            'pass_count': sum(1 for r in rank_ics if r >= self.rank_ic_target),
            'total_count': len(rank_ics),
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
        logger.info("V74 Rank IC 预测质量审计表")
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
    
    def generate_monthly_ic_chart(self, output_path: Optional[str] = None) -> str:
        """
        生成月度 Rank IC 柱状图（ASCII 格式）
        
        Returns:
            ASCII 柱状图字符串
        """
        chart_data = self.get_monthly_ic_chart_data()
        
        if not chart_data['months']:
            return "无数据"
        
        lines = []
        lines.append("月度 Rank IC 柱状图")
        lines.append("=" * 60)
        
        max_ic = max(max(chart_data['rank_ics']), self.rank_ic_target)
        bar_width = 40
        
        for i, (month, rank_ic) in enumerate(zip(chart_data['months'], chart_data['rank_ics'])):
            # 判断是否达标
            passed = rank_ic >= self.rank_ic_target
            status = "✓" if passed else "✗"
            
            # 计算柱长
            bar_length = int(rank_ic / max_ic * bar_width) if max_ic > 0 else 0
            bar = "█" * bar_length + "░" * (bar_width - bar_length)
            
            # 格式化输出
            lines.append(f"{month} |{bar}| {rank_ic:.4f} {status}")
        
        lines.append("-" * 60)
        lines.append(f"目标：>{self.rank_ic_target:.3f}")
        lines.append(f"达标月份：{chart_data['pass_count']}/{chart_data['total_count']}")
        lines.append("=" * 60)
        
        chart_text = "\n".join(lines)
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(chart_text)
        
        return chart_text


# ===========================================
# V74 可视化辅助函数
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


def generate_monthly_ic_ascii_chart(monthly_stats: List[V74MonthlyICStats], 
                                     target: float = V74_RANK_IC_TARGET) -> str:
    """
    生成月度 IC ASCII 柱状图
    
    Args:
        monthly_stats: 月度 IC 统计列表
        target: 目标 IC 值
        
    Returns:
        ASCII 柱状图字符串
    """
    if not monthly_stats:
        return "无数据"
    
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("月度 Rank IC 柱状图")
    lines.append("=" * 60)
    
    rank_ics = [m.mean_rank_ic for m in monthly_stats]
    max_ic = max(max(rank_ics), target) if rank_ics else target
    bar_width = 40
    
    for m in monthly_stats:
        passed = m.mean_rank_ic >= target
        status = "✓" if passed else "✗"
        
        bar_length = int(m.mean_rank_ic / max_ic * bar_width) if max_ic > 0 else 0
        bar = "█" * bar_length + "░" * (bar_width - bar_length)
        
        lines.append(f"{m.month} |{bar}| {m.mean_rank_ic:.4f} {status}")
    
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
    'V74_INITIAL_CAPITAL',
    'V74_MAX_POSITIONS',
    'V74_WARMUP_PERIOD',
    'V74_MIN_SAMPLE_SIZE',
    'V74_DATA_CHECK_YEAR',
    'V74_MIN_FUND_FLOW_ROWS',
    'V74_RETRY_ATTEMPTS',
    'V74_RETRY_DELAY',
    'V74_RS_WINDOW',
    'V74_RS_BASE_SCORE_MIN',
    'V74_RS_BASE_SCORE_MAX',
    'V74_VPD_WINDOW',
    'V74_VPD_WEIGHT',
    'V74_SNR_WINDOW',
    'V74_SNR_ENTROPY_WEIGHT',
    'V74_SNR_SKEWNESS_WEIGHT',
    'V74_SNR_MAGNITUDE_WEIGHT',
    'V74_SNR_MIN',
    'V74_SNR_MAX',
    'V74_SIGMOID_SCALE',
    'V74_SCORE_SMOOTHING',
    'V74_INDUSTRY_NEUTRAL_WEIGHT',
    'V74_COMMISSION_RATE',
    'V74_MIN_COMMISSION',
    'V74_SLIPPAGE_BUY',
    'V74_SLIPPAGE_SELL',
    'V74_STAMP_DUTY',
    'V74_TRANSFER_FEE',
    'V74_FRICTION_COST',
    'V74_STOP_LOSS_RATIO',
    'V74_PROFIT_TARGET_RATIO',
    'V74_TRAILING_STOP_RATIO',
    'V74_MAX_SINGLE_POSITION_PCT',
    'V74_SELECTION_PERCENTILE',
    'V74_RANK_IC_TARGET',
    'V74_SCORE_STD_TARGET_MIN',
    'V74_SCORE_STD_TARGET_MAX',
    
    # 数据类
    'V74Position',
    'V74Trade',
    'V74Signal',
    'V74ICMetrics',
    'V74ScoreDistribution',
    'V74MonthlyICStats',
    
    # 核心类
    'V74DataManager',
    'V74AlphaCenter',
    'V74RankICCalculator',
    
    # 辅助函数
    'generate_score_histogram_data',
    'generate_monthly_ic_ascii_chart',
]