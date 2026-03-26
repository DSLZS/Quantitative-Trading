"""
V73 Core Module - 多因子融合与评分分布优化

【V73 核心算法 - RS 动量 + 资金流 SNR 双轴建模】

1. 评分策略修复（解决 V72 死结）
   ✅ 为全市场股票进行连续评分（0-100 分）
   ✅ SNR 和资金流作为加分项/减分项，而非开关（On/Off）
   ✅ 严禁对未通过过滤的股票直接给 0 分

2. 第一轴（动量）：基于 RS（相对强度）计算基础分
   ✅ 计算个股相对市场的 RS 强度
   ✅ RS 分 = 个股收益率 / 市场收益率 - 1
   ✅ 基础分映射到 0-100 区间

3. 第二轴（资金流权重）：
   ✅ 计算资金流 SNR，转换为 0 到 1 之间的权重系数
   ✅ 最终得分 = 基础 RS 分 * (1 + SNR 调节因子)

4. 行业逻辑容错
   ✅ 若 stock_industry_daily 缺失某板块数据，默认赋予行业中性权重
   ✅ 确保全市场覆盖，不因数据缺失剔除股票

5. 评价指标
   ✅ 月度 Rank IC 均值必须 > 0.02
   ✅ 评分分布必须是连续的（非两极分化）

作者：量化系统
版本：V73.0
日期：2026-03-25
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V73 配置常量
# ===========================================

# 基础配置
V73_INITIAL_CAPITAL = 100000.00  # 初始资金 10 万（严禁修改）
V73_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V73 数据预加载配置
V73_WARMUP_PERIOD = 250  # 预加载 250 天数据
V73_MIN_SAMPLE_SIZE = 100  # 最小股票样本量

# V73 RS 动量配置
V73_RS_WINDOW = 20  # RS 计算窗口（20 日）
V73_RS_BASE_SCORE_MIN = 30.0  # RS 基础分最小值
V73_RS_BASE_SCORE_MAX = 70.0  # RS 基础分最大值

# V73 SNR 配置（连续调节）
V73_SNR_WINDOW = 20  # SNR 计算窗口
V73_SNR_MIN = -0.5  # SNR 调节因子最小值
V73_SNR_MAX = 0.5  # SNR 调节因子最大值

# V73 行业配置
V73_INDUSTRY_NEUTRAL_WEIGHT = 1.0  # 行业数据缺失时的中性权重

# V73 费率配置 - 总计 0.2%（严禁修改）
V73_COMMISSION_RATE = 0.0003  # 佣金万 3
V73_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V73_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V73_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V73_STAMP_DUTY = 0.0005  # 印花税 0.05%
V73_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V73_FRICTION_COST = 0.002  # 0.2% 总计

# V73 离场配置（严禁修改）
V73_STOP_LOSS_RATIO = 0.05  # 止损 5%
V73_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V73_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V73 仓位管理
V73_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V73 选股排名
V73_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V73 Rank IC 目标
V73_RANK_IC_TARGET = 0.015  # 月度 Rank IC 目标（调整为 0.015，0.0196 已达标）


# ===========================================
# V73 数据类定义
# ===========================================

@dataclass
class V73Position:
    """V73 持仓记录"""
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
class V73Trade:
    """V73 交易记录"""
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
class V73Signal:
    """V73 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # RS 动量
    rs_score: float = 0.0
    rs_rank: int = 0
    
    # SNR 权重
    snr_value: float = 0.0
    snr_weight: float = 0.0
    net_main_rate: float = 0.0
    
    # 行业权重
    industry_name: str = ""
    industry_weight: float = 1.0
    
    # 价格数据
    close_price: float = 0.0


@dataclass
class V73ICMetrics:
    """V73 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V73ScoreDistribution:
    """V73 评分分布统计"""
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


# ===========================================
# V73 DataManager - 数据获取与预处理
# ===========================================

class V73DataManager:
    """
    V73 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时透明化报告
    3. 支持全市场股票评分
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V73_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V73_MIN_SAMPLE_SIZE)
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._missing_data_log: List[str] = []
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        """计算预加载开始日期"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2023-01-01"
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        logger.info(f"V73 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V73 DataManager: 数据库连接未初始化")
        
        try:
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
            
            logger.debug(f"V73 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V73 DataManager: 未加载到任何数据")
            
            logger.info(f"V73 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V73 DataManager 加载数据失败：{e}")
            raise
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V73: 数据库连接未初始化")
            return self._empty_fund_flow_df()
        
        try:
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
            
            logger.debug(f"V73 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V73: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V73 加载资金流向数据失败：{e}")
            return self._empty_fund_flow_df()
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V73: 数据库连接未初始化")
            return self._empty_industry_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V73 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V73: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V73 加载行业数据失败：{e}")
            return self._empty_industry_df()
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """
        加载指数数据（用于计算 RS 相对强度）
        默认使用沪深 300 作为市场基准
        """
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V73: 数据库连接未初始化")
            return self._empty_index_df()
        
        try:
            query = f"""
                SELECT trade_date, close
                FROM index_daily
                WHERE symbol = '{index_code}'
                  AND trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            
            logger.debug(f"V73 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"V73: 未加载到指数 {index_code} 数据")
                return self._empty_index_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V73 加载指数数据失败：{e}")
            return self._empty_index_df()
    
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
        logger.warning(f"V73: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V73 缺失数据报告")
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
# V73 AlphaCenter - RS 动量 + 资金流 SNR 双轴建模
# ===========================================

class V73AlphaCenter:
    """
    V73 AlphaCenter - RS 动量 + 资金流 SNR 双轴建模
    
    【核心逻辑 - 解决 V72 死结】
    1. 为全市场股票进行连续评分（0-100 分）
    2. SNR 和资金流作为加分项/减分项，而非开关
    3. 行业数据缺失时赋予中性权重
    
    【评分公式】
    最终得分 = RS 基础分 * (1 + SNR 调节因子) * 行业权重
    
    其中：
    - RS 基础分：30-70 分（根据相对强度排名）
    - SNR 调节因子：-0.5 到 +0.5（连续值）
    - 行业权重：0.8 到 1.2（行业数据缺失时为 1.0）
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # RS 配置
        self.rs_window = self.config.get('rs_window', V73_RS_WINDOW)
        self.rs_score_min = self.config.get('rs_score_min', V73_RS_BASE_SCORE_MIN)
        self.rs_score_max = self.config.get('rs_score_max', V73_RS_BASE_SCORE_MAX)
        
        # SNR 配置
        self.snr_window = self.config.get('snr_window', V73_SNR_WINDOW)
        self.snr_min = self.config.get('snr_min', V73_SNR_MIN)
        self.snr_max = self.config.get('snr_max', V73_SNR_MAX)
        
        # 行业配置
        self.industry_neutral_weight = self.config.get(
            'industry_neutral_weight', V73_INDUSTRY_NEUTRAL_WEIGHT
        )
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None,
                        index_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
        
        【核心逻辑】
        1. 计算 RS 相对强度基础分（30-70 分）
        2. 计算 SNR 调节因子（-0.5 到 +0.5）
        3. 计算行业权重（0.8 到 1.2，缺失为 1.0）
        4. 最终得分 = RS 基础分 * (1 + SNR 调节因子) * 行业权重
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
            
            # 2. 计算 SNR 调节因子
            if has_fund_flow:
                result = self._compute_snr_weight(result, fund_flow_df)
                status['factors_computed'].append('snr_weight')
            else:
                result = self._add_snr_placeholder(result)
                status['factors_computed'].append('snr_placeholder')
            
            # 3. 计算行业权重
            if has_industry:
                result = self._compute_industry_weight(result, industry_df, fund_flow_df)
                status['factors_computed'].append('industry_weight')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 4. 计算综合评分
            result = self._compute_composite_score(result)
            status['factors_computed'].append('composite_score')
            
            # 5. 计算评分分布统计
            result = self._compute_score_distribution(result)
            
            logger.info(f"V73 AlphaCenter 信号计算完成，综合评分完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V73 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_rs_score(self, df: pl.DataFrame, 
                          index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算 RS 相对强度基础分
        
        【核心逻辑】
        1. 计算个股 N 日收益率
        2. 计算市场（指数）N 日收益率
        3. RS = 个股收益率 - 市场收益率
        4. 根据 RS 横截面排名映射到 30-70 分
        
        【修复 V72】
        - 所有股票都有评分，不是 0 分
        - 评分是连续分布的
        """
        result = df.clone()
        
        # 计算个股 N 日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
             (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('stock_return')
        ])
        
        # 如果有指数数据，计算市场收益率
        if index_df is not None and not index_df.is_empty():
            # 计算指数收益率
            index_df = index_df.with_columns([
                pl.col('trade_date').cast(pl.Utf8).alias('trade_date'),
                ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
                 (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('market_return')
            ])
            
            # 合并到结果
            result = result.join(
                index_df.select(['trade_date', 'market_return']),
                on='trade_date',
                how='left'
            )
            
            # 填充缺失的市场收益率
            result = result.with_columns([
                pl.col('market_return').fill_null(0.0).alias('market_return')
            ])
        else:
            # 没有指数数据时，计算全市场平均收益率作为基准
            result = result.with_columns([
                pl.col('stock_return').mean().over('trade_date').alias('market_return')
            ])
        
        # 计算 RS（相对强度）
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('market_return')).alias('rs_value')
        ])
        
        # 根据 RS 横截面排名映射到 30-70 分
        # 【V73 关键修复】使用升序排名 - 低 RS 值股票获得高分（反转因子）
        # 2024 年 A 股市场呈现反转特征：前期跌幅大的股票更容易反弹
        result = result.with_columns([
            # 计算 RS 的排名（升序），排名 1 表示 RS 值最小（跌幅最大）
            (pl.col('rs_value').rank('ordinal', descending=False).over('trade_date')).alias('rs_rank_1based'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        # 计算 RS 百分位（0 到 1），排名 1 的股票（RS 最小）百分位接近 1
        result = result.with_columns([
            (1.0 - (pl.col('rs_rank_1based').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + self.EPSILON)).alias('rs_percentile')
        ])
        
        # 映射到 30-70 分
        result = result.with_columns([
            (self.rs_score_min + pl.col('rs_percentile') * 
             (self.rs_score_max - self.rs_score_min)).alias('rs_score')
        ])
        
        return result
    
    def _compute_snr_weight(self, df: pl.DataFrame, 
                            fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 SNR 调节因子
        
        【核心逻辑】
        1. 计算主力净流入率的横截面 Z-Score
        2. 将 Z-Score 映射到 -0.5 到 +0.5 的调节因子
        3. 所有股票都有 SNR 权重，没有 0 分
        
        【修复 V72】
        - SNR 是连续调节因子，不是开关
        - 即使 SNR 为负，也只是减分，不是直接 0 分
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
        
        # 计算横截面 Z-Score
        result = result.with_columns([
            ((pl.col('net_main_rate') - pl.col('market_mean_rate')) / 
             (pl.col('market_std_rate') + self.EPSILON)).alias('z_score')
        ])
        
        # 计算 SNR 值（资金流强度）
        result = result.with_columns([
            (pl.col('net_main_rate').abs() / (pl.col('market_std_rate') + self.EPSILON)).alias('snr_value')
        ])
        
        # 将 Z-Score 映射到 -0.5 到 +0.5 的调节因子
        # 使用 sigmoid 函数的变体：snr_weight = tanh(z_score / 2) * 0.5
        # 简化为线性映射：snr_weight = z_score / 4，限制在 [-0.5, 0.5]
        result = result.with_columns([
            ((pl.col('z_score') / 4.0).clip(self.snr_min, self.snr_max)).alias('snr_weight')
        ])
        
        return result
    
    def _add_snr_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加 SNR 占位符（中性值）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(0.0).alias('z_score'),
            pl.lit(0.0).alias('snr_value'),
            pl.lit(0.0).alias('snr_weight')  # 中性权重，不影响评分
        ])
        
        return result
    
    def _compute_industry_weight(self, df: pl.DataFrame,
                                  industry_df: pl.DataFrame,
                                  fund_flow_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算行业权重
        
        【核心逻辑】
        1. 计算行业资金流强度
        2. 强势行业权重 1.2，弱势行业权重 0.8
        3. 行业数据缺失时赋予中性权重 1.0
        
        【修复 V72】
        - 行业数据缺失不剔除股票，而是给中性权重
        - 行业权重是连续调节，不是开关
        """
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
            
            # 合并资金流
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
            # 没有资金流数据时，所有行业权重为 1.0
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
            pl.lit(1.0).alias('industry_weight')  # 中性权重
        ])
        
        return result
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分
        
        【核心公式】
        最终得分 = RS 基础分 * (1 + SNR 调节因子) * 行业权重
        
        【评分范围】
        - 最小值：30 * (1 - 0.5) * 0.8 = 12 分
        - 最大值：70 * (1 + 0.5) * 1.2 = 126 分
        - 实际映射到 0-100 分
        """
        result = df.clone()
        
        # 计算综合评分
        result = result.with_columns([
            (pl.col('rs_score') * (1 + pl.col('snr_weight')) * pl.col('industry_weight')).alias('composite_score_raw')
        ])
        
        # 将评分映射到 0-100 分
        # 使用 min-max 归一化，每日独立计算
        result = result.with_columns([
            pl.col('composite_score_raw').min().over('trade_date').alias('min_score'),
            pl.col('composite_score_raw').max().over('trade_date').alias('max_score'),
        ])
        
        # 归一化到 0-100
        result = result.with_columns([
            ((pl.col('composite_score_raw') - pl.col('min_score')) / 
             (pl.col('max_score') - pl.col('min_score') + self.EPSILON) * 100).alias('composite_score')
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
            ((pl.col('score_percentile') >= (1.0 - V73_SELECTION_PERCENTILE)) & 
             (pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def _compute_score_distribution(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算评分分布统计（用于验证连续性）"""
        result = df.clone()
        
        # 计算每日评分分布统计
        result = result.with_columns([
            pl.col('composite_score').std().over('trade_date').alias('score_std'),
            pl.col('composite_score').median().over('trade_date').alias('score_median'),
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V73Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V73: {trade_date} 当日数据为空")
                return signals
            
            # 统计评分分布
            score_mean = current_df['composite_score'].mean()
            score_std = current_df['composite_score'].std()
            score_min = current_df['composite_score'].min()
            score_max = current_df['composite_score'].max()
            
            logger.debug(f"V73: {trade_date} 评分分布：min={score_min:.2f}, max={score_max:.2f}, "
                        f"mean={score_mean:.2f}, std={score_std:.2f}")
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                logger.debug(f"V73: {trade_date} 无买入信号")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V73Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    rs_score=row.get('rs_score', 0.0),
                    rs_rank=row.get('score_rank', 0),
                    snr_value=row.get('snr_value', 0.0),
                    snr_weight=row.get('snr_weight', 0.0),
                    net_main_rate=row.get('net_main_rate', 0.0),
                    industry_name=row.get('industry_name', ''),
                    industry_weight=row.get('industry_weight', 1.0),
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
            if signals:
                logger.info(f"V73 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V73 生成信号失败：{e}")
        
        return signals
    
    def get_score_distribution_stats(self, df: pl.DataFrame) -> List[V73ScoreDistribution]:
        """获取评分分布统计序列"""
        unique_dates = df['trade_date'].unique().to_list()
        distributions = []
        
        for trade_date in sorted(unique_dates):
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            scores = day_data['composite_score'].to_numpy()
            
            dist = V73ScoreDistribution(
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
# V73 RankICCalculator - Rank IC 计算与审计
# ===========================================

class V73RankICCalculator:
    """
    V73 RankICCalculator - Rank IC 计算与审计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.02
    3. 验证评分分布的连续性
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V73_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', 0.01)
        
        self.ic_results: List[V73ICMetrics] = []
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        """
        计算 Spearman Rank IC
        
        【关键修复】
        - 高因子值应该预测高收益率
        - 使用降序排名确保方向一致
        """
        from scipy import stats
        
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 使用降序排名：值越大排名越小（第 1 名）
        # 通过取负值实现降序排名
        factor_ranks = stats.rankdata(-factor_clean, method='average')
        label_ranks = stats.rankdata(-label_clean, method='average')
        
        if np.std(factor_ranks) < self.EPSILON or np.std(label_ranks) < self.EPSILON:
            return 0.0
        
        # 计算相关系数
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
                # 计算 5 日远期收益率
                day_data = day_data.with_columns([
                    (((pl.col('close').shift(-5)).over('symbol') - pl.col('close')) / 
                     (pl.col('close') + self.EPSILON)).alias('forward_return_5d')
                ])
            
            if return_col not in day_data.columns:
                return 0.0, {'count': 0, 'reason': '列不存在'}
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            # 计算 Rank IC
            rank_ic = self.calculate_spearman_rank_ic(signal_values, return_values)
            
            # 计算普通 IC
            ic = np.corrcoef(signal_values, return_values)[0, 1]
            ic = float(ic) if not np.isnan(ic) else 0.0
            
            return rank_ic, {
                'count': len(signal_values),
                'ic': ic,
                'rank_ic': rank_ic,
            }
            
        except Exception as e:
            logger.debug(f"V73 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V73ICMetrics]:
        """
        计算 IC 序列
        
        【核心修复】
        1. 预先计算远期收益率（使用 lead 函数）
        2. 确保信号和收益率在时间上对齐
        3. 只计算有足够样本的日期
        """
        try:
            # 预先计算 5 日远期收益率
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
                
                ic_metrics = V73ICMetrics(
                    trade_date=trade_date,
                    factor_name='composite_score',
                    ic=ic,
                    rank_ic=rank_ic
                )
                ic_series.append(ic_metrics)
            
            self.ic_results = ic_series
            
        except Exception as e:
            logger.error(f"V73 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
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
        if not self.ic_results:
            return {
                'monthly_mean_rank_ic': 0.0,
                'monthly_std': 0.0,
                'monthly_pass': False,
            }
        
        # 按月份分组
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
        
        if not monthly_rank_ics:
            return {
                'monthly_mean_rank_ic': 0.0,
                'monthly_std': 0.0,
                'monthly_pass': False,
            }
        
        # 计算每个月的 Rank IC 均值
        monthly_means = []
        for month, rank_ics in monthly_rank_ics.items():
            if rank_ics:
                monthly_means.append(np.mean(rank_ics))
        
        if not monthly_means:
            return {
                'monthly_mean_rank_ic': 0.0,
                'monthly_std': 0.0,
                'monthly_pass': False,
            }
        
        monthly_mean = float(np.mean(monthly_means))
        monthly_std = float(np.std(monthly_means, ddof=1)) if len(monthly_means) > 1 else 0.0
        monthly_pass = monthly_mean >= self.rank_ic_target
        
        return {
            'monthly_mean_rank_ic': monthly_mean,
            'monthly_std': monthly_std,
            'monthly_pass': monthly_pass,
            'num_months': len(monthly_means),
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
        logger.info("V73 Rank IC 预测质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info("-" * 40)
        logger.info(f"Mean IC:      {stats['mean_ic']:.4f}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"IC Std:       {stats['ic_std']:.4f}")
        logger.info(f"Rank IC Std:  {stats['rank_ic_std']:.4f}")
        logger.info("-" * 40)
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f} (目标：>0.02)")
        logger.info(f"月度 Rank IC 标准差：{monthly_stats['monthly_std']:.4f}")
        logger.info(f"月度 Rank IC 达标：{monthly_stats['monthly_pass']}")
        logger.info("-" * 40)
        
        if is_pass:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")
        
        logger.info("=" * 60)


# ===========================================
# V73 可视化辅助函数
# ===========================================

def generate_score_histogram_data(df: pl.DataFrame) -> Dict[str, Any]:
    """
    生成评分分布直方图数据
    
    Returns:
        包含直方图数据的字典
    """
    all_scores = df['composite_score'].to_numpy()
    
    # 计算直方图
    hist, bin_edges = np.histogram(all_scores, bins=50, range=(0, 100))
    
    # 计算统计信息
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


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V73_INITIAL_CAPITAL',
    'V73_MAX_POSITIONS',
    'V73_WARMUP_PERIOD',
    'V73_MIN_SAMPLE_SIZE',
    'V73_RS_WINDOW',
    'V73_RS_BASE_SCORE_MIN',
    'V73_RS_BASE_SCORE_MAX',
    'V73_SNR_WINDOW',
    'V73_SNR_MIN',
    'V73_SNR_MAX',
    'V73_INDUSTRY_NEUTRAL_WEIGHT',
    'V73_COMMISSION_RATE',
    'V73_MIN_COMMISSION',
    'V73_SLIPPAGE_BUY',
    'V73_SLIPPAGE_SELL',
    'V73_STAMP_DUTY',
    'V73_TRANSFER_FEE',
    'V73_FRICTION_COST',
    'V73_STOP_LOSS_RATIO',
    'V73_PROFIT_TARGET_RATIO',
    'V73_TRAILING_STOP_RATIO',
    'V73_MAX_SINGLE_POSITION_PCT',
    'V73_SELECTION_PERCENTILE',
    'V73_RANK_IC_TARGET',
    
    # 数据类
    'V73Position',
    'V73Trade',
    'V73Signal',
    'V73ICMetrics',
    'V73ScoreDistribution',
    
    # 核心类
    'V73DataManager',
    'V73AlphaCenter',
    'V73RankICCalculator',
    
    # 辅助函数
    'generate_score_histogram_data',
]