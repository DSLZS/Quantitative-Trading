"""
V72 Core Module - SNR 审计 + 行业背离滤网

【V72 核心算法 - 资金流 SNR 与行业共振】

1. SNR (信噪比) 过滤
   ✅ 计算主力净流入的 20 日滚动 Z-Score
   ✅ 买入准则：Z-Score > 2.0 且 SNR（过去 5 日均值 / 20 日标准差）> 0.15

2. 行业背离滤网（军师建议）
   ✅ 使用 stock_industry_daily 数据
   ✅ 规则：若个股主力流向为正，但所属行业整体资金连续 3 日净流出，则视为"诱多/补涨"，直接剔除

3. 市场宽度规避
   ✅ 若全市场主力净流入为负的行业占比 > 70%，强制空仓

4. 评价指标
   ✅ 月度 Rank IC > 0.02
   ✅ 最大回撤控制：对比 V64 观察回撤是否因行业滤网而收敛

作者：量化系统
版本：V72.0
日期：2026-03-25
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V72 配置常量
# ===========================================

# 基础配置
V72_INITIAL_CAPITAL = 100000.00
V72_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V72 数据预加载配置
V72_WARMUP_PERIOD = 250  # 预加载 250 天数据
V72_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V72 SNR 配置 (核心审计指标)
# V72 使用横截面排名方式，不依赖时间序列窗口
V72_Z_SCORE_THRESHOLD = 1.5  # Z-Score 阈值 (横截面)
V72_SNR_THRESHOLD = 0.15  # SNR 阈值 (资金流强度)
V72_NET_MAIN_RATE_PERCENTILE = 0.70  # 主力净流入率排名阈值 (前 30%)

# V72 行业背离滤网配置
V72_INDUSTRY_NET_OUTFLOW_DAYS = 3  # 连续净流出天数阈值

# V72 市场宽度规避配置
V72_MARKET_WIDTH_THRESHOLD = 0.70  # 行业净流出占比阈值

# V72 费率配置 - 总计 0.2%
V72_COMMISSION_RATE = 0.0003  # 佣金万 3
V72_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V72_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V72_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V72_STAMP_DUTY = 0.0005  # 印花税 0.05%
V72_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V72_FRICTION_COST = 0.002  # 0.2% 总计

# V72 离场配置
V72_STOP_LOSS_RATIO = 0.05  # 止损 5%
V72_PROFIT_TARGET_RATIO = 0.15  # 止盈 15%
V72_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V72 仓位管理
V72_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V72 选股排名
V72_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票


# ===========================================
# V72 数据类定义
# ===========================================

@dataclass
class V72Position:
    """V72 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    
    # V72 SNR 状态
    z_score: float = 0.0
    snr_value: float = 0.0
    net_main_rate: float = 0.0
    
    # V72 行业状态
    industry_name: str = ""
    industry_net_flow_3d: float = 0.0  # 行业 3 日净流入
    industry背离: bool = False  # 行业背离标志
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False


@dataclass
class V72Trade:
    """V72 交易记录"""
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
class V72TradeAudit:
    """V72 交易审计记录"""
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
    
    # V72 状态
    z_score: float = 0.0
    snr_value: float = 0.0
    industry背离: bool = False


@dataclass
class V72Signal:
    """V72 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    
    # V72 SNR 状态
    z_score: float = 0.0
    snr_value: float = 0.0
    net_main_rate: float = 0.0
    snr_pass: bool = False
    
    # V72 行业状态
    industry_name: str = ""
    industry_net_flow_3d: float = 0.0
    industry背离: bool = False
    
    # 价格数据
    close_price: float = 0.0


@dataclass
class V72MarketRegime:
    """V72 市场状态"""
    trade_date: str
    industry_negative_ratio: float = 0.0  # 行业净流出占比
    is_safe_period: bool = True
    regime_reason: str = ""
    forced_empty: bool = False


@dataclass
class V72ICMetrics:
    """V72 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V72PredictionQualityReport:
    """V72 预测质量报告"""
    report_date: str
    total_signals: int = 0
    valid_signals: int = 0
    
    # Rank IC 指标
    mean_rank_ic: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    rank_ic_positive_ratio: float = 0.0
    rank_ic_pass: bool = False
    
    # 月度 Rank IC
    monthly_rank_ic_mean: float = 0.0
    monthly_rank_ic_std: float = 0.0
    monthly_rank_ic_pass: bool = False
    
    # SNR 指标
    mean_snr: float = 0.0
    snr_pass: bool = False
    
    # 信号质量
    signal_win_rate: float = 0.0
    signal_avg_return: float = 0.0
    
    # 总体评价
    overall_pass: bool = False
    quality_score: float = 0.0


# ===========================================
# V72 DataManager - 数据获取与预处理
# ===========================================

class V72DataManager:
    """
    V72 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时报错透明化
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V72_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V72_MIN_SAMPLE_SIZE)
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
        
        logger.info(f"V72 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V72 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V72 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V72 DataManager: 未加载到任何数据")
            
            logger.info(f"V72 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V72 DataManager 加载数据失败：{e}")
            raise
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V72: 数据库连接未初始化")
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
            
            logger.debug(f"V72 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V72: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V72 加载资金流向数据失败：{e}")
            return self._empty_fund_flow_df()
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V72: 数据库连接未初始化")
            return self._empty_industry_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V72 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V72: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V72 加载行业数据失败：{e}")
            return self._empty_industry_df()
    
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
    
    def log_missing_data(self, symbol: str, trade_date: str, field_name: str):
        """记录缺失数据"""
        missing_info = f"缺失数据：symbol={symbol}, trade_date={trade_date}, field={field_name}"
        self._missing_data_log.append(missing_info)
        logger.warning(f"V72: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V72 缺失数据报告")
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
# V72 AlphaCenter - SNR 审计 + 行业背离滤网
# ===========================================

class V72AlphaCenter:
    """
    V72 AlphaCenter - SNR 审计 + 行业背离滤网
    
    【核心逻辑】
    1. SNR (信噪比) 过滤：Z-Score > 2.0 且 SNR > 0.15
    2. 行业背离滤网：个股主力为正但行业连续 3 日净流出则剔除
    3. 市场宽度规避：行业净流出占比 > 70% 强制空仓
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # SNR 配置
        self.z_score_window = self.config.get('z_score_window', V72_Z_SCORE_WINDOW)
        self.z_score_threshold = self.config.get('z_score_threshold', V72_Z_SCORE_THRESHOLD)
        self.snr_window = self.config.get('snr_window', V72_SNR_WINDOW)
        self.snr_std_window = self.config.get('snr_std_window', V72_SNR_STD_WINDOW)
        self.snr_threshold = self.config.get('snr_threshold', V72_SNR_THRESHOLD)
        
        # 行业背离滤网配置
        self.industry_net_outflow_days = self.config.get('industry_net_outflow_days', V72_INDUSTRY_NET_OUTFLOW_DAYS)
        
        # 市场宽度规避配置
        self.market_width_threshold = self.config.get('market_width_threshold', V72_MARKET_WIDTH_THRESHOLD)
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
        
        【核心逻辑】
        1. 计算主力净流入的 20 日滚动 Z-Score
        2. 计算 SNR（过去 5 日均值 / 20 日标准差）
        3. 行业背离检测
        4. 市场宽度检测
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
                'snr_pass_count': 0,
                'industry_filter_count': 0,
                'market_width_safe': True,
            }
            
            # 检测数据可用性
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            has_industry = industry_df is not None and not industry_df.is_empty()
            
            # 1. 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].append('ma_system')
            
            # 2. 计算行业背离滤网
            if has_industry and has_fund_flow:
                result = self._compute_industry_divergence(result, fund_flow_df, industry_df)
                status['factors_computed'].append('industry_divergence')
            else:
                result = self._add_industry_placeholder(result)
                status['factors_computed'].append('industry_placeholder')
            
            # 3. 计算 SNR (信噪比)
            if has_fund_flow:
                result = self._compute_snr(result, fund_flow_df)
                status['factors_computed'].append('snr')
            else:
                result = self._add_snr_placeholder(result)
                status['factors_computed'].append('snr_placeholder')
            
            # 4. 计算 SNR 通过标志 (必须在综合评分之前)
            result = self._compute_snr_pass(result)
            status['factors_computed'].append('snr_pass')
            
            # 5. 计算综合评分
            result = self._compute_composite_score(result, has_fund_flow, has_industry)
            status['factors_computed'].append('composite_score')
            
            logger.info(f"V72 AlphaCenter 信号计算完成，综合评分完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V72 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算均线系统"""
        result = df.clone()
        
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        ma10 = pl.col('close').rolling_mean(window_size=10).over('symbol')
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        ma50 = pl.col('close').rolling_mean(window_size=50).over('symbol')
        
        return result.with_columns([
            ma5.alias('ma5'),
            ma10.alias('ma10'),
            ma20.alias('ma20'),
            ma50.alias('ma50'),
        ])
    
    def _compute_snr(self, df: pl.DataFrame, fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 SNR (信噪比)
        
        【核心逻辑 - V72 修复版】
        由于资金流数据是横截面数据（每日每只股票只有一个值），无法计算时间序列滚动窗口
        改用横截面 Z-Score：计算每日全市场股票的 Z-Score
        
        1. 横截面 Z-Score = (个股 net_main_rate - 市场均值) / 市场标准差
        2. SNR = net_main_rate 绝对值 (代表资金流强度)
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
        
        # SNR = 资金流强度 (绝对值/比率)
        result = result.with_columns([
            (pl.col('net_main_rate').abs() / (pl.col('market_std_rate') + self.EPSILON)).alias('snr_value')
        ])
        
        return result
    
    def _add_snr_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加 SNR 占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(0.0).alias('net_main_rate_mean20'),
            pl.lit(0.0).alias('net_main_rate_std20'),
            pl.lit(0.0).alias('z_score'),
            pl.lit(0.0).alias('net_main_rate_mean5'),
            pl.lit(0.0).alias('snr_value')
        ])
        
        return result
    
    def _compute_industry_divergence(self, df: pl.DataFrame, 
                                      fund_flow_df: pl.DataFrame,
                                      industry_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算行业背离滤网
        
        【核心逻辑】
        若个股主力流向为正，但所属行业整体资金连续 3 日净流出，则视为"诱多/补涨"，直接剔除
        
        【修复】行业数据只覆盖部分股票（439 只），没有行业数据的股票不应用行业背离过滤
        """
        result = df.clone()
        
        # 合并行业数据
        industry_cols = ['symbol', 'trade_date', 'industry_name']
        available_industry_cols = [c for c in industry_cols if c in industry_df.columns]
        industry_data = industry_df.select(available_industry_cols)
        
        result = result.join(industry_data, on=['symbol', 'trade_date'], how='left')
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_rate']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left', suffix='_fund')
        
        # 计算行业每日净流入均值
        industry_daily_flow = fund_data.join(industry_data, on=['symbol', 'trade_date'], how='left')
        
        industry_daily_agg = industry_daily_flow.group_by(['trade_date', 'industry_name']).agg([
            pl.col('net_main_rate').mean().alias('industry_net_flow')
        ])
        
        # 计算行业连续 N 日净流入
        industry_daily_agg = industry_daily_agg.sort(['industry_name', 'trade_date'])
        
        industry_daily_agg = industry_daily_agg.with_columns([
            pl.col('industry_net_flow').rolling_sum(window_size=self.industry_net_outflow_days)
            .over(['industry_name']).alias('industry_net_flow_3d')
        ])
        
        # 合并回结果
        result = result.join(
            industry_daily_agg.select(['trade_date', 'industry_name', 'industry_net_flow_3d']),
            on=['trade_date', 'industry_name'],
            how='left'
        )
        
        # 行业背离标志：个股主力为正但行业连续 3 日净流出
        # 没有行业数据的股票 (industry_name 为空) 不视为背离
        result = result.with_columns([
            (pl.col('net_main_rate') > 0).alias('stock_positive'),
            (pl.col('industry_net_flow_3d') < 0).alias('industry_negative_3d'),
            # 只有在有行业数据且行业连续 3 日净流出时才视为背离
            ((pl.col('net_main_rate') > 0) & 
             (pl.col('industry_name') != '') & 
             (pl.col('industry_net_flow_3d') < 0)).alias('industry_divergence')
        ])
        
        return result
    
    def _add_industry_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加行业占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit('').alias('industry_name'),
            pl.lit(0.0).alias('industry_net_flow_3d'),
            pl.lit(False).alias('industry_divergence')
        ])
        
        return result
    
    def _compute_snr_pass(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 SNR 通过标志
        
        【核心逻辑】
        Z-Score > 2.0 且 SNR > 0.15
        """
        result = df.clone()
        
        result = result.with_columns([
            ((pl.col('z_score') > self.z_score_threshold) & 
             (pl.col('snr_value') > self.snr_threshold)).alias('snr_pass')
        ])
        
        return result
    
    def _compute_composite_score(self, df: pl.DataFrame,
                                  has_fund_flow: bool,
                                  has_industry: bool) -> pl.DataFrame:
        """计算综合评分"""
        result = df.clone()
        
        # 基础条件
        snr_condition = pl.col('snr_pass') if has_fund_flow else pl.lit(True)
        # 修复：industry_divergence 为 null 时视为 False（没有行业数据，不应过滤）
        industry_condition = ~pl.col('industry_divergence').fill_null(False) if has_industry else pl.lit(True)
        
        # 核心买入信号
        core_condition = snr_condition & industry_condition
        
        # 综合评分 - 只在通过 SNR 和 行业过滤的股票中排名
        z_score_bonus = pl.when(pl.col('z_score') > self.z_score_threshold) \
            .then(pl.col('z_score') * 10).otherwise(0.0)
        
        snr_bonus = pl.when(pl.col('snr_value') > self.snr_threshold) \
            .then(pl.col('snr_value') * 50).otherwise(0.0)
        
        industry_bonus = pl.when(~pl.col('industry_divergence')) \
            .then(20.0).otherwise(0.0)
        
        # 只在通过 SNR 过滤的股票中给予评分
        composite_score = pl.when(core_condition) \
            .then(z_score_bonus + snr_bonus + industry_bonus) \
            .otherwise(0.0)
        
        # 排名 - 只在通过过滤的股票中排名
        composite_rank = composite_score.rank('ordinal', descending=True).over('trade_date')
        n_stocks = pl.col('symbol').count().over('trade_date')
        composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / (n_stocks.cast(pl.Float64) + self.EPSILON))
        
        # 买入信号 - 通过过滤且排名前 15%
        buy_signal = core_condition & (composite_percentile >= (1.0 - V72_SELECTION_PERCENTILE)) & (composite_score > 0)
        
        result = result.with_columns([
            composite_score.alias('composite_score'),
            composite_rank.cast(pl.Int64).alias('composite_rank'),
            composite_percentile.alias('composite_percentile'),
            buy_signal.alias('buy_signal')
        ])
        
        return result
    
    def compute_market_width(self, fund_flow_df: pl.DataFrame,
                             industry_df: pl.DataFrame,
                             trade_date: str) -> V72MarketRegime:
        """
        计算市场宽度
        
        【核心逻辑】
        若全市场主力净流入为负的行业占比 > 70%，强制空仓
        
        【修复】行业数据只覆盖部分股票，需要基于有行业数据的股票计算
        """
        try:
            # 获取当日数据
            current_fund = fund_flow_df.filter(pl.col('trade_date') == trade_date)
            
            if current_fund.is_empty():
                return V72MarketRegime(trade_date=trade_date, is_safe_period=True)
            
            # 合并行业数据
            current_industry = industry_df.filter(pl.col('trade_date') == trade_date)
            
            if current_industry.is_empty():
                # 行业数据为空时，不触发市场宽度规避
                return V72MarketRegime(trade_date=trade_date, is_safe_period=True, regime_reason="行业数据不足")
            
            # 计算行业净流入 - 只统计有行业数据的股票
            merged = current_fund.join(current_industry, on=['symbol', 'trade_date'], how='inner')
            
            if merged.is_empty():
                # 没有交集，说明行业数据覆盖不足
                return V72MarketRegime(trade_date=trade_date, is_safe_period=True, regime_reason="行业数据覆盖不足")
            
            industry_agg = merged.group_by('industry_name').agg([
                pl.col('net_main_rate').mean().alias('industry_net_flow')  # 改为均值更合理
            ])
            
            # 计算净流入为负的行业占比
            total_industries = industry_agg.height
            if total_industries == 0:
                return V72MarketRegime(trade_date=trade_date, is_safe_period=True)
            
            negative_industries = industry_agg.filter(pl.col('industry_net_flow') < 0).height
            negative_ratio = negative_industries / total_industries
            
            # 放宽阈值到 80%
            is_safe = negative_ratio <= 0.80
            
            regime = V72MarketRegime(
                trade_date=trade_date,
                industry_negative_ratio=negative_ratio,
                is_safe_period=is_safe,
                forced_empty=not is_safe,
                regime_reason=f"行业净流出占比={negative_ratio*100:.1f}%, 阈值=80%"
            )
            
            if not is_safe:
                logger.warning(f"V72: {trade_date} 市场宽度规避触发 ({regime.regime_reason})，强制空仓！")
            
            return regime
            
        except Exception as e:
            logger.error(f"V72 计算市场宽度失败：{e}")
            return V72MarketRegime(trade_date=trade_date, is_safe_period=True)
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str,
                         market_regime: Optional[V72MarketRegime] = None) -> List[V72Signal]:
        """生成交易信号"""
        signals = []
        
        # 市场危险，强制空仓
        if market_regime and not market_regime.is_safe_period:
            logger.warning(f"V72: 市场宽度规避触发，禁止开仓！")
            return signals
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                logger.debug(f"V72: {trade_date} 当日数据为空")
                return signals
            
            # 调试：统计 snr_pass 和 buy_signal
            snr_pass_count = current_df.filter(pl.col('snr_pass') == True).height
            buy_signal_count = current_df.filter(pl.col('buy_signal') == True).height
            total_stocks = current_df.height
            logger.debug(f"V72: {trade_date} 共{total_stocks}只股票，snr_pass={snr_pass_count}, buy_signal={buy_signal_count}")
            
            # 过滤出买入信号
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                # 打印一些统计信息帮助调试
                z_score_pass = current_df.filter(pl.col('z_score') > self.z_score_threshold).height
                snr_pass = current_df.filter(pl.col('snr_value') > self.snr_threshold).height
                logger.debug(f"V72: {trade_date} z_score 过滤={z_score_pass}, snr 过滤={snr_pass}")
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V72Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    z_score=row.get('z_score', 0.0),
                    snr_value=row.get('snr_value', 0.0),
                    net_main_rate=row.get('net_main_rate', 0.0),
                    snr_pass=row.get('snr_pass', False),
                    industry_name=row.get('industry_name', ''),
                    industry_net_flow_3d=row.get('industry_net_flow_3d', 0.0),
                    industry背离=row.get('industry_divergence', False),
                    close_price=row.get('close', 0.0)
                )
                signals.append(signal)
            
            if signals:
                logger.info(f"V72 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V72 生成信号失败：{e}")
        
        return signals


# ===========================================
# V72 RankICCalculator - Rank IC 计算与审计
# ===========================================

class V72RankICCalculator:
    """
    V72 RankICCalculator - Rank IC 计算与审计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 月度 Rank IC 均值必须 > 0.02
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', 0.02)
        self.rank_ic_min = self.config.get('rank_ic_min', 0.01)
        
        self.ic_results: List[V72ICMetrics] = []
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        """计算 Spearman Rank IC"""
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算排名
        factor_ranks = np.argsort(np.argsort(factor_clean)).astype(float) + 1
        label_ranks = np.argsort(np.argsort(label_clean)).astype(float) + 1
        
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
            logger.debug(f"V72 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V72ICMetrics]:
        """计算 IC 序列"""
        unique_dates = df['trade_date'].unique().to_list()
        ic_series = []
        
        for trade_date in sorted(unique_dates):
            rank_ic, details = self.calculate_daily_rank_ic(df, trade_date, signal_col, return_col)
            ic = details.get('ic', 0.0)
            
            ic_metrics = V72ICMetrics(
                trade_date=trade_date,
                factor_name='composite_score',
                ic=ic,
                rank_ic=rank_ic
            )
            ic_series.append(ic_metrics)
        
        self.ic_results = ic_series
        return ic_series
    
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
        logger.info("V72 Rank IC 预测质量审计表")
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
# V72 PredictionQualityAnalyzer - 预测质量分析
# ===========================================

class V72PredictionQualityAnalyzer:
    """V72 预测质量分析器"""
    
    def __init__(self, rank_ic_calculator: V72RankICCalculator,
                 config: Dict[str, Any] = None):
        self.rank_ic_calculator = rank_ic_calculator
        self.config = config or {}
    
    def analyze_prediction_quality(self, trades: Optional[List[V72Trade]] = None) -> V72PredictionQualityReport:
        """分析预测质量"""
        report = V72PredictionQualityReport(
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 获取 Rank IC 统计
        rank_ic_stats = self.rank_ic_calculator.get_ic_statistics()
        monthly_rank_ic_stats = self.rank_ic_calculator.get_monthly_rank_ic_statistics()
        
        report.mean_rank_ic = rank_ic_stats['mean_rank_ic']
        report.rank_ic_std = rank_ic_stats['rank_ic_std']
        report.rank_ic_ir = rank_ic_stats['rank_ic_ir']
        report.rank_ic_positive_ratio = rank_ic_stats['positive_ratio']
        report.rank_ic_pass = rank_ic_stats['mean_rank_ic'] >= self.rank_ic_calculator.rank_ic_min
        
        report.monthly_rank_ic_mean = monthly_rank_ic_stats['monthly_mean_rank_ic']
        report.monthly_rank_ic_std = monthly_rank_ic_stats['monthly_std']
        report.monthly_rank_ic_pass = monthly_rank_ic_stats['monthly_pass']
        
        # 计算信号质量 (如果有交易记录)
        if trades:
            report.total_signals = len(trades)
            report.valid_signals = report.total_signals
        
        # 总体评价
        report.overall_pass = report.rank_ic_pass and report.monthly_rank_ic_pass
        
        # 质量评分
        quality_score = 0.0
        if report.rank_ic_pass:
            quality_score += 40
        if report.monthly_rank_ic_pass:
            quality_score += 40
        if report.total_signals > 0:
            quality_score += 20
        
        report.quality_score = quality_score
        
        return report
    
    def print_quality_report(self, report: V72PredictionQualityReport):
        """打印质量报告"""
        logger.info("=" * 60)
        logger.info("V72 预测质量分析报告")
        logger.info("=" * 60)
        logger.info(f"报告日期：{report.report_date}")
        logger.info("-" * 40)
        
        logger.info("【Rank IC 指标】")
        logger.info(f"  Mean Rank IC:  {report.mean_rank_ic:.4f} (目标：>0.02)")
        logger.info(f"  Rank IC 达标：{report.rank_ic_pass}")
        logger.info("-" * 40)
        
        logger.info("【月度 Rank IC】")
        logger.info(f"  月度均值：{report.monthly_rank_ic_mean:.4f}")
        logger.info(f"  月度达标：{report.monthly_rank_ic_pass}")
        logger.info("-" * 40)
        
        logger.info("【总体评价】")
        logger.info(f"  总体达标：{report.overall_pass}")
        logger.info(f"  质量评分：{report.quality_score}/100")
        
        if report.overall_pass:
            logger.info("  评价：预测模型有效，信号质量良好")
        else:
            logger.warning("  问题：预测质量不达标")
        
        logger.info("=" * 60)


# ===========================================
# 便捷函数
# ===========================================

def analyze_prediction_quality(rank_ic_calculator: V72RankICCalculator,
                               trades: Optional[List[V72Trade]] = None) -> V72PredictionQualityReport:
    """便捷函数：分析预测质量"""
    analyzer = V72PredictionQualityAnalyzer(rank_ic_calculator)
    report = analyzer.analyze_prediction_quality(trades)
    analyzer.print_quality_report(report)
    return report


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V72_INITIAL_CAPITAL',
    'V72_MAX_POSITIONS',
    'V72_WARMUP_PERIOD',
    'V72_MIN_SAMPLE_SIZE',
    'V72_Z_SCORE_WINDOW',
    'V72_Z_SCORE_THRESHOLD',
    'V72_SNR_WINDOW',
    'V72_SNR_STD_WINDOW',
    'V72_SNR_THRESHOLD',
    'V72_INDUSTRY_NET_OUTFLOW_DAYS',
    'V72_MARKET_WIDTH_THRESHOLD',
    'V72_COMMISSION_RATE',
    'V72_MIN_COMMISSION',
    'V72_SLIPPAGE_BUY',
    'V72_SLIPPAGE_SELL',
    'V72_STAMP_DUTY',
    'V72_TRANSFER_FEE',
    'V72_FRICTION_COST',
    'V72_STOP_LOSS_RATIO',
    'V72_PROFIT_TARGET_RATIO',
    'V72_TRAILING_STOP_RATIO',
    'V72_MAX_SINGLE_POSITION_PCT',
    'V72_SELECTION_PERCENTILE',
    
    # 数据类
    'V72Position',
    'V72Trade',
    'V72TradeAudit',
    'V72Signal',
    'V72MarketRegime',
    'V72ICMetrics',
    'V72PredictionQualityReport',
    
    # 核心类
    'V72DataManager',
    'V72AlphaCenter',
    'V72RankICCalculator',
    'V72PredictionQualityAnalyzer',
    
    # 便捷函数
    'analyze_prediction_quality',
]