"""
V66 Core Module - 机构资金踪迹模型预测引擎 (IC 优化版)

【V66 核心算法 - Institutional-Trace with IC Optimization】

1. 入场逻辑 (Alpha Center)
   ✅ RS-Industry Z-Score：个股 RS 必须在其所属行业的横向分布中处于 Z > 1.2 的位置
   ✅ 资金共振：价格在 MA20 附近 Pullback 时，主力净流入占比高于过去 10 天均值
   ✅ 成交量特征：回调期间，主力资金/成交量的比值必须上升，识别"机构护盘"特征
   ✅ VCP 动态阈值：振幅收缩为行业动态标准差的 1.2 倍

2. 离场逻辑
   ✅ 趋势破坏止损：跌破 MA10 且主力资金净流出时离场
   ✅ 移动止盈：从最高点回撤 5% 离场
   ✅ 目标止盈：盈利 15% 离场

3. 评估指标
   ✅ AE (Alpha-Efficiency) = (Win_Rate × P/L_Ratio) / Max_Drawdown × √Trade_Count
   ✅ IC (信息系数)：2024 年 IC 均值 > 0.02

4. 数据可信度
   ✅ 每笔交易标注是基于"资金流 + 行业"双因子，还是单因子

作者：量化系统
版本：V66.0
日期：2026-03-24
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V66 配置常量 - 机构资金踪迹 (IC 优化版)
# ===========================================

# 基础配置
V66_INITIAL_CAPITAL = 100000.00
V66_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V66 频率熔断
V66_MONTHLY_TRADE_LIMIT = 15  # 月均交易次数上限
V66_WEEKLY_TRADE_LIMIT = 4    # 每周最多开仓 4 只
V66_GLOBAL_TRADE_LIMIT = 150  # 全场交易次数限制

# V66 数据预加载配置
V66_WARMUP_PERIOD = 250  # 预加载 250 天数据
V66_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V66 行业护城河 RS 配置 (IC 优化)
V66_RS_WINDOW = 20  # 计算 RS 的窗口
V66_RS_Z_SCORE_THRESHOLD = 1.2  # RS Z-Score 必须 > 1.2
V66_RS_PERCENTILE_THRESHOLD = 0.20  # 必须处于行业前 20%

# V66 资金共振配置
V66_NET_MAIN_RATE_WINDOW = 10  # 计算主力净流入均值的窗口
V66_PULLBACK_MA_PERIOD = 20  # 回调至 MA20
V66_PULLBACK_TOLERANCE = 0.02  # 回调容忍度 2%

# V66 成交量特征配置 (机构护盘识别)
V66_VOLUME_RATIO_WINDOW = 5  # 成交量比值计算窗口
V66_MAIN_FORCE_VOLUME_RATIO_THRESHOLD = 1.2  # 主力资金/成交量比值上升阈值

# V66 VCP 动态阈值配置
V66_VCP_WINDOW = 10  # 观察窗口
V66_VCP_VOLATILITY_MULTIPLIER = 1.2  # 行业动态标准差的 1.2 倍

# V66 大盘避坑配置
V66_MARKET_DECLINE_RATIO_THRESHOLD = 0.80  # 下跌家数占比 > 80% 强制空仓

# V66 费率配置 - 总计 0.2%
V66_COMMISSION_RATE = 0.0003  # 佣金万 3
V66_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V66_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V66_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V66_STAMP_DUTY = 0.0005  # 印花税 0.05%
V66_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V66_FRICTION_COST = 0.002  # 0.2% 总计

# V66 离场配置 - 趋势破坏止损
V66_TREND_BREAK_MA_PERIOD = 10  # 跌破 MA10
V66_PROFIT_TARGET_RATIO = 0.15  # 目标盈利 15%
V66_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V66 仓位管理
V66_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V66 选股排名
V66_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V66 IC 评估配置
V66_IC_TARGET_MEAN = 0.02  # 2024 年 IC 均值目标 > 0.02


# ===========================================
# V66 数据类定义
# ===========================================

@dataclass
class V66Position:
    """V66 持仓记录"""
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
    
    # V66 资金流状态
    net_main_rate: float = 0.0
    net_main_rate_vs_avg: float = 0.0
    main_force_volume_ratio: float = 0.0  # 主力资金/成交量比值
    
    # V66 行业护城河 RS
    rs_percentile: float = 0.0  # RS 行业百分位
    rs_z_score: float = 0.0  # RS Z-Score
    industry_name: str = ""
    
    # V66 VCP 动态阈值
    vcp_amplitude: float = 0.0
    vcp_industry_std: float = 0.0
    vcp_pass: bool = False
    
    # V66 数据可信度
    data_credibility: str = "dual"  # "dual" = 资金流 + 行业，"fund_flow" = 仅资金流，"industry" = 仅行业，"none" = 无
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    trend_break_triggered: bool = False
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V66Trade:
    """V66 交易记录"""
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
    
    # V66 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    
    # V66 数据可信度
    data_credibility: str = "dual"


@dataclass
class V66TradeAudit:
    """V66 交易审计记录"""
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
    
    # V66 状态
    net_main_rate: float = 0.0
    rs_percentile: float = 0.0
    rs_z_score: float = 0.0
    vcp_pass: bool = False
    
    # V66 数据可信度
    data_credibility: str = "dual"
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V66Signal:
    """V66 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    signal_rank: int
    composite_score: float
    
    # V66 状态
    net_main_rate: float = 0.0
    net_main_rate_vs_avg: float = 0.0
    main_force_volume_ratio: float = 0.0  # 主力资金/成交量比值
    rs_percentile: float = 0.0
    rs_z_score: float = 0.0
    vcp_amplitude: float = 0.0
    vcp_industry_std: float = 0.0
    vcp_pass: bool = False
    is_pullback_to_ma20: bool = False
    
    # V66 数据可信度
    data_credibility: str = "dual"
    
    # 价格数据
    close_price: float = 0.0
    ma20_price: float = 0.0


@dataclass
class V66MarketRegime:
    """V66 大盘状态"""
    trade_date: str
    decline_ratio: float = 0.0  # 下跌家数占比
    is_safe_period: bool = True
    regime_reason: str = ""
    forced_empty: bool = False


@dataclass
class V66ICMetrics:
    """V66 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


# ===========================================
# V66 数据可信度枚举
# ===========================================

class V66DataCredibility:
    """V66 数据可信度等级"""
    DUAL = "dual"  # 资金流 + 行业双因子
    FUND_FLOW = "fund_flow"  # 仅资金流
    INDUSTRY = "industry"  # 仅行业
    NONE = "none"  # 无数据


# ===========================================
# V66 DataManager - 数据获取与预处理
# ===========================================

class V66DataManager:
    """
    V66 DataManager - 数据获取与预处理
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V66_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V66_MIN_SAMPLE_SIZE)
        self._data_cache: Dict[str, pl.DataFrame] = {}
    
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
        
        logger.info(f"V66 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V66 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V66 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V66 DataManager: 未加载到任何数据")
            
            logger.info(f"V66 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V66 DataManager 加载数据失败：{e}")
            raise
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V66: 数据库连接未初始化")
            return self._empty_fund_flow_df()
        
        try:
            if symbols:
                symbol_list = "','".join(symbols)
                symbol_filter = f"AND symbol IN ('{symbol_list}')"
            else:
                symbol_filter = ""
            
            query = f"""
                SELECT symbol, trade_date, net_main_amount, net_main_ratio,
                       net_super_amount, net_large_amount, net_medium_amount, net_small_amount
                FROM stock_fund_flow
                WHERE trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                  {symbol_filter}
                ORDER BY symbol, trade_date
            """
            
            logger.debug(f"V66 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V66: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V66 加载资金流向数据失败：{e}")
            return self._empty_fund_flow_df()
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V66: 数据库连接未初始化")
            return self._empty_industry_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V66 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V66: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V66 加载行业数据失败：{e}")
            return self._empty_industry_df()
    
    def _empty_fund_flow_df(self) -> pl.DataFrame:
        """返回空资金流 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'net_main_amount': pl.Float64,
            'net_main_ratio': pl.Float64
        })
    
    def _empty_industry_df(self) -> pl.DataFrame:
        """返回空行业 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'industry_name': pl.Utf8
        })
    
    def clear_cache(self):
        """清除缓存"""
        self._data_cache.clear()


# ===========================================
# V66 AlphaCenter - 机构资金踪迹信号生成 (IC 优化)
# ===========================================

class V66AlphaCenter:
    """
    V66 AlphaCenter - 机构资金踪迹信号生成 (IC 优化版)
    
    【核心逻辑】
    1. RS-Industry Z-Score：个股 RS 必须在其所属行业的横向分布中处于 Z > 1.2 的位置
    2. 成交量特征：回调期间，主力资金/成交量的比值必须上升，识别"机构护盘"特征
    3. 资金共振：价格在 MA20 附近 Pullback 时，主力净流入占比高于过去 10 天均值
    4. VCP 动态阈值：振幅收缩为行业动态标准差的 1.2 倍
    5. 大盘避坑：下跌家数占比 > 80% 强制空仓
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 行业护城河配置
        self.rs_window = self.config.get('rs_window', V66_RS_WINDOW)
        self.rs_z_score_threshold = self.config.get('rs_z_score_threshold', V66_RS_Z_SCORE_THRESHOLD)
        self.rs_percentile_threshold = self.config.get('rs_percentile_threshold', V66_RS_PERCENTILE_THRESHOLD)
        
        # 资金共振配置
        self.net_main_rate_window = self.config.get('net_main_rate_window', V66_NET_MAIN_RATE_WINDOW)
        self.pullback_ma_period = self.config.get('pullback_ma_period', V66_PULLBACK_MA_PERIOD)
        self.pullback_tolerance = self.config.get('pullback_tolerance', V66_PULLBACK_TOLERANCE)
        
        # 成交量特征配置
        self.volume_ratio_window = self.config.get('volume_ratio_window', V66_VOLUME_RATIO_WINDOW)
        self.main_force_volume_ratio_threshold = self.config.get('main_force_volume_ratio_threshold', V66_MAIN_FORCE_VOLUME_RATIO_THRESHOLD)
        
        # VCP 动态阈值配置
        self.vcp_window = self.config.get('vcp_window', V66_VCP_WINDOW)
        self.vcp_volatility_multiplier = self.config.get('vcp_volatility_multiplier', V66_VCP_VOLATILITY_MULTIPLIER)
        
        # 大盘避坑配置
        self.decline_ratio_threshold = self.config.get('decline_ratio_threshold', V66_MARKET_DECLINE_RATIO_THRESHOLD)
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
        
        【数据可信度检测】
        - 检测是否有资金流数据和行业数据
        - 根据数据可用性确定可信度等级
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
            
            status = {'factors_computed': [], 'data_credibility': V66DataCredibility.DUAL}
            
            # 检测数据可用性
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            has_industry = industry_df is not None and not industry_df.is_empty()
            
            # 确定数据可信度
            if has_fund_flow and has_industry:
                status['data_credibility'] = V66DataCredibility.DUAL
            elif has_fund_flow:
                status['data_credibility'] = V66DataCredibility.FUND_FLOW
            elif has_industry:
                status['data_credibility'] = V66DataCredibility.INDUSTRY
            else:
                status['data_credibility'] = V66DataCredibility.NONE
            
            # 1. 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].append('ma_system')
            
            # 2. 计算行业护城河 RS (IC 优化：Z-Score)
            if has_industry:
                result = self._compute_industry_moat_rs_v2(result, industry_df)
                status['factors_computed'].append('industry_moat_rs_v2')
            else:
                result = self._compute_basic_rs(result)
                status['factors_computed'].append('basic_rs')
            
            # 3. 计算资金共振
            if has_fund_flow:
                result = self._compute_capital_resonance(result, fund_flow_df)
                status['factors_computed'].append('capital_resonance')
            else:
                result = self._add_fund_flow_placeholders(result)
                status['factors_computed'].append('fund_flow_placeholder')
            
            # 4. 计算成交量特征 (机构护盘识别)
            if has_fund_flow:
                result = self._compute_volume_characteristics(result, fund_flow_df)
                status['factors_computed'].append('volume_characteristics')
            else:
                result = self._add_volume_placeholders(result)
                status['factors_computed'].append('volume_placeholder')
            
            # 5. 计算价格回调至 MA20
            result = self._compute_pullback_to_ma20(result)
            status['factors_computed'].append('pullback_to_ma20')
            
            # 6. 计算 VCP 动态阈值
            result = self._compute_vcp_dynamic_threshold(result)
            status['factors_computed'].append('vcp_dynamic_threshold')
            
            # 7. 计算综合评分
            result = self._compute_composite_score(result, has_fund_flow, has_industry)
            status['factors_computed'].append('composite_score')
            
            logger.info(f"V66 AlphaCenter 信号计算完成，数据可信度：{status['data_credibility']}")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V66 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算均线系统"""
        result = df.clone()
        
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        ma10 = pl.col('close').rolling_mean(window_size=10).over('symbol')
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        ma50 = pl.col('close').rolling_mean(window_size=50).over('symbol')
        ma150 = pl.col('close').rolling_mean(window_size=150).over('symbol')
        ma200 = pl.col('close').rolling_mean(window_size=200).over('symbol')
        
        return result.with_columns([
            ma5.alias('ma5'),
            ma10.alias('ma10'),
            ma20.alias('ma20'),
            ma50.alias('ma50'),
            ma150.alias('ma150'),
            ma200.alias('ma200'),
        ])
    
    def _compute_industry_moat_rs_v2(self, df: pl.DataFrame, 
                                      industry_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算行业护城河 RS (IC 优化版：Z-Score)
        
        【核心逻辑】
        - 计算个股过去 N 日收益率
        - 计算所属行业指数同期收益率
        - 计算 RS 的 Z-Score，必须 Z > 1.2
        - 计算 RS 百分位，必须处于行业前 20%
        """
        result = df.clone()
        
        # 计算个股收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
             (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('stock_return')
        ])
        
        # 合并行业数据
        industry_cols = ['symbol', 'trade_date', 'industry_name']
        available_industry_cols = [c for c in industry_cols if c in industry_df.columns]
        industry_data = industry_df.select(available_industry_cols)
        
        result = result.join(industry_data, on=['symbol', 'trade_date'], how='left')
        
        # 计算行业收益率（使用行业平均）
        result = result.with_columns([
            pl.col('stock_return').mean().over(['trade_date', 'industry_name']).alias('industry_return')
        ])
        
        # 计算行业中性化 RS
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('industry_return')).alias('rs_vs_industry')
        ])
        
        # 计算 RS 行业百分位
        result = result.with_columns([
            (pl.col('rs_vs_industry').rank('ordinal', descending=True).over(['trade_date', 'industry_name']) /
             (pl.col('symbol').count().over(['trade_date', 'industry_name']) + self.EPSILON)).alias('rs_percentile')
        ])
        
        # 计算 RS Z-Score (IC 优化核心)
        # Z-Score = (RS - 行业均值) / 行业标准差
        result = result.with_columns([
            pl.col('rs_vs_industry').mean().over(['trade_date', 'industry_name']).alias('rs_industry_mean'),
            pl.col('rs_vs_industry').std().over(['trade_date', 'industry_name']).alias('rs_industry_std')
        ])
        
        result = result.with_columns([
            ((pl.col('rs_vs_industry') - pl.col('rs_industry_mean')) / 
             (pl.col('rs_industry_std') + self.EPSILON)).alias('rs_z_score')
        ])
        
        # 行业护城河：RS Z-Score > 1.2 且 处于前 20%
        result = result.with_columns([
            (pl.col('rs_z_score') > self.rs_z_score_threshold).alias('rs_z_score_pass'),
            (pl.col('rs_percentile') >= (1.0 - self.rs_percentile_threshold)).alias('rs_in_top_percentile'),
            (pl.col('rs_z_score') > self.rs_z_score_threshold) & (pl.col('rs_percentile') >= (1.0 - self.rs_percentile_threshold)).alias('rs_industry_moat_pass')
        ])
        
        return result
    
    def _compute_basic_rs(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算基础 RS（无行业数据时）"""
        result = df.clone()
        
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
             (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('stock_return'),
            ((pl.col('close') - pl.col('close').shift(self.rs_window)) / 
             (pl.col('close').shift(self.rs_window) + self.EPSILON)).alias('rs_vs_industry'),
            pl.lit(1.0).alias('rs_percentile'),
            pl.lit(0.0).alias('rs_z_score'),
            pl.lit(True).alias('rs_in_top_percentile'),
            pl.lit(True).alias('rs_industry_moat_pass')
        ])
        
        return result
    
    def _compute_capital_resonance(self, df: pl.DataFrame,
                                    fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算资金共振
        
        【核心逻辑】
        - 在价格回调至 MA20 期间
        - 主力净流入占比 (net_main_rate) 必须高于过去 10 天的均值
        """
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_main_ratio']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left')
        
        # 计算 net_main_rate（主力净流入占比）
        result = result.with_columns([
            pl.when(pl.col('net_main_ratio').is_not_null())
            .then(pl.col('net_main_ratio'))
            .otherwise(
                pl.col('net_main_amount') / (pl.col('amount') + self.EPSILON)
            ).alias('net_main_rate')
        ])
        
        # 计算过去 N 天的 net_main_rate 均值
        result = result.with_columns([
            pl.col('net_main_rate').rolling_mean(window_size=self.net_main_rate_window).over('symbol').alias('net_main_rate_avg')
        ])
        
        # 资金共振：net_main_rate 高于过去均值
        result = result.with_columns([
            (pl.col('net_main_rate') > pl.col('net_main_rate_avg')).alias('net_main_rate_above_avg'),
            (pl.col('net_main_rate') - pl.col('net_main_rate_avg')).alias('net_main_rate_vs_avg')
        ])
        
        return result
    
    def _compute_volume_characteristics(self, df: pl.DataFrame,
                                         fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算成交量特征 (机构护盘识别)
        
        【核心逻辑】
        - 回调期间，主力资金/成交量的比值必须上升
        - 识别"机构护盘"特征
        """
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount']
        available_fund_cols = [c for c in fund_cols if c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left')
        
        # 计算主力资金/成交量比值
        result = result.with_columns([
            (pl.col('net_main_amount').abs() / (pl.col('volume') + self.EPSILON)).alias('main_force_volume_ratio')
        ])
        
        # 计算过去 N 天的均值
        result = result.with_columns([
            pl.col('main_force_volume_ratio').rolling_mean(window_size=self.volume_ratio_window).over('symbol').alias('main_force_volume_ratio_avg')
        ])
        
        # 机构护盘：比值上升
        result = result.with_columns([
            (pl.col('main_force_volume_ratio') > pl.col('main_force_volume_ratio_avg') * self.main_force_volume_ratio_threshold).alias('main_force_volume_ratio_rising'),
            (pl.col('main_force_volume_ratio') / (pl.col('main_force_volume_ratio_avg') + self.EPSILON)).alias('main_force_volume_ratio_change')
        ])
        
        return result
    
    def _add_fund_flow_placeholders(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加资金流占位符（无数据时）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_amount'),
            pl.lit(0.0).alias('net_main_rate'),
            pl.lit(0.0).alias('net_main_rate_avg'),
            pl.lit(False).alias('net_main_rate_above_avg'),
            pl.lit(0.0).alias('net_main_rate_vs_avg')
        ])
        
        return result
    
    def _add_volume_placeholders(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加成交量特征占位符（无数据时）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('main_force_volume_ratio'),
            pl.lit(0.0).alias('main_force_volume_ratio_avg'),
            pl.lit(False).alias('main_force_volume_ratio_rising'),
            pl.lit(0.0).alias('main_force_volume_ratio_change')
        ])
        
        return result
    
    def _compute_pullback_to_ma20(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算价格回调至 MA20
        """
        result = df.clone()
        
        ma20 = pl.col('ma20')
        close = pl.col('close')
        
        # 回调至 MA20 附近（容忍度内）
        pullback_ratio = (close - ma20) / (ma20 + self.EPSILON)
        is_pullback = (pullback_ratio.abs() <= self.pullback_tolerance)
        
        # 确认是回调（之前价格高于 MA20）
        prev_close = close.shift(1).over('symbol')
        prev_ma20 = ma20.shift(1).over('symbol')
        was_above = prev_close > prev_ma20
        
        is_pullback_to_ma20 = is_pullback & was_above
        
        result = result.with_columns([
            pullback_ratio.alias('pullback_ratio'),
            is_pullback.alias('is_near_ma20'),
            is_pullback_to_ma20.alias('is_pullback_to_ma20')
        ])
        
        return result
    
    def _compute_vcp_dynamic_threshold(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 VCP 动态阈值
        """
        result = df.clone()
        
        # 1. 计算每日振幅
        daily_amplitude = (pl.col('high') - pl.col('low')) / (pl.col('close') + self.EPSILON)
        
        # 2. 滚动平均振幅
        avg_amplitude = daily_amplitude.rolling_mean(window_size=self.vcp_window).over('symbol')
        
        # 3. 计算振幅（如果行业数据存在）
        if 'industry_name' in df.columns:
            # 计算行业振幅标准差
            industry_amplitude_std = daily_amplitude.std().over(['trade_date', 'industry_name'])
            
            # 动态阈值
            dynamic_threshold = industry_amplitude_std * self.vcp_volatility_multiplier
            
            # VCP 通过条件：振幅 < 动态阈值
            vcp_pass = avg_amplitude < dynamic_threshold
            
            result = result.with_columns([
                industry_amplitude_std.alias('vcp_industry_std'),
                dynamic_threshold.alias('vcp_dynamic_threshold'),
                vcp_pass.alias('vcp_pass')
            ])
        else:
            # 无行业数据时使用固定阈值
            vcp_pass = avg_amplitude < 0.08  # 8% 固定阈值
            
            result = result.with_columns([
                pl.lit(0.0).alias('vcp_industry_std'),
                pl.lit(0.08).alias('vcp_dynamic_threshold'),
                vcp_pass.alias('vcp_pass')
            ])
        
        result = result.with_columns([
            daily_amplitude.alias('daily_amplitude'),
            avg_amplitude.alias('vcp_amplitude')
        ])
        
        return result
    
    def _compute_composite_score(self, df: pl.DataFrame,
                                  has_fund_flow: bool,
                                  has_industry: bool) -> pl.DataFrame:
        """
        计算综合评分
        """
        result = df.clone()
        
        # 基础条件
        rs_condition = pl.col('rs_industry_moat_pass') if has_industry else pl.col('rs_in_top_percentile')
        pullback_condition = pl.col('is_pullback_to_ma20')
        vcp_condition = pl.col('vcp_pass')
        
        # 资金流条件（如果有数据）
        if has_fund_flow:
            fund_flow_condition = pl.col('net_main_rate_above_avg')
            volume_condition = pl.col('main_force_volume_ratio_rising')
        else:
            fund_flow_condition = pl.lit(True)
            volume_condition = pl.lit(True)
        
        # 核心买入信号
        core_condition = rs_condition & pullback_condition & vcp_condition & fund_flow_condition & volume_condition
        
        # 综合评分
        rs_bonus = pl.when(pl.col('rs_in_top_percentile')) \
            .then(pl.col('rs_percentile') * 50).otherwise(0.0)
        
        rs_z_bonus = pl.when(pl.col('rs_z_score') > self.rs_z_score_threshold) \
            .then(pl.col('rs_z_score') * 20).otherwise(0.0)
        
        fund_flow_bonus = pl.when(pl.col('net_main_rate_vs_avg') > 0) \
            .then(pl.col('net_main_rate_vs_avg') * 100).otherwise(0.0)
        
        volume_bonus = pl.when(pl.col('main_force_volume_ratio_rising')) \
            .then(20.0).otherwise(0.0)
        
        vcp_bonus = pl.when(pl.col('vcp_pass')) \
            .then(30.0).otherwise(0.0)
        
        composite_score = rs_bonus + rs_z_bonus + fund_flow_bonus + volume_bonus + vcp_bonus
        
        # 排名
        composite_rank = composite_score.rank('ordinal', descending=True).over('trade_date')
        n_stocks = pl.col('symbol').count().over('trade_date')
        composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / (n_stocks.cast(pl.Float64) + self.EPSILON))
        
        # 买入信号
        buy_signal = core_condition
        
        result = result.with_columns([
            composite_score.alias('composite_score'),
            composite_rank.cast(pl.Int64).alias('composite_rank'),
            composite_percentile.alias('composite_percentile'),
            buy_signal.alias('buy_signal')
        ])
        
        return result
    
    def compute_market_decline_ratio(self, df: pl.DataFrame) -> V66MarketRegime:
        """
        计算市场下跌家数占比（大盘避坑）
        """
        latest_date = df['trade_date'].max()
        latest_df = df.filter(pl.col('trade_date') == latest_date)
        
        if latest_df.is_empty():
            return V66MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        # 计算下跌家数（收盘价 < 开盘价）
        decline_count = latest_df.filter(pl.col('close') < pl.col('open')).height
        total_count = latest_df.height
        
        if total_count == 0:
            return V66MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        decline_ratio = decline_count / total_count
        is_safe = decline_ratio <= self.decline_ratio_threshold
        
        regime = V66MarketRegime(
            trade_date=latest_date,
            decline_ratio=decline_ratio,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"下跌家数占比={decline_ratio*100:.1f}%, 阈值={self.decline_ratio_threshold*100:.0f}%"
        )
        
        if not is_safe:
            logger.warning(f"V66: {latest_date} 大盘避坑触发 ({regime.regime_reason})，强制空仓！")
        
        return regime
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str,
                         market_regime: Optional[V66MarketRegime] = None,
                         data_credibility: str = V66DataCredibility.DUAL) -> List[V66Signal]:
        """
        生成交易信号
        """
        signals = []
        
        # 大盘危险，强制空仓
        if market_regime and not market_regime.is_safe_period:
            logger.warning(f"V66: 大盘避坑触发，禁止开仓！")
            return signals
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            top_n = max(1, int(buy_df.height * V66_SELECTION_PERCENTILE))
            buy_df = buy_df.head(top_n)
            
            for row in buy_df.iter_rows(named=True):
                signal = V66Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    signal_rank=row.get('composite_rank', 9999),
                    composite_score=row.get('composite_score', 0.0),
                    net_main_rate=row.get('net_main_rate', 0.0),
                    net_main_rate_vs_avg=row.get('net_main_rate_vs_avg', 0.0),
                    main_force_volume_ratio=row.get('main_force_volume_ratio', 0.0),
                    rs_percentile=row.get('rs_percentile', 0.0),
                    rs_z_score=row.get('rs_z_score', 0.0),
                    vcp_amplitude=row.get('vcp_amplitude', 0.0),
                    vcp_industry_std=row.get('vcp_industry_std', 0.0),
                    vcp_pass=row.get('vcp_pass', False),
                    is_pullback_to_ma20=row.get('is_pullback_to_ma20', False),
                    data_credibility=data_credibility,
                    close_price=row.get('close', 0.0),
                    ma20_price=row.get('ma20', 0.0)
                )
                signals.append(signal)
            
            logger.info(f"V66 生成 {len(signals)} 个买入信号 ({trade_date}), 数据可信度：{data_credibility}")
            
        except Exception as e:
            logger.error(f"V66 生成信号失败：{e}")
        
        return signals


# ===========================================
# V66 TradeExec - 成交执行
# ===========================================

class V66TradeExec:
    """
    V66 TradeExec - 真实成交执行
    
    【V66 离场逻辑 - 趋势破坏止损】
    1. 趋势破坏止损：跌破 MA10 且主力资金净流出时离场
    2. 移动止盈：从最高点回撤 5% 离场
    3. 目标止盈：盈利 15% 离场
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', V66_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V66_MAX_POSITIONS)
        self.commission_rate = self.config.get('commission_rate', V66_COMMISSION_RATE)
        self.min_commission = self.config.get('min_commission', V66_MIN_COMMISSION)
        self.slippage_buy = self.config.get('slippage_buy', V66_SLIPPAGE_BUY)
        self.slippage_sell = self.config.get('slippage_sell', V66_SLIPPAGE_SELL)
        self.stamp_duty = self.config.get('stamp_duty', V66_STAMP_DUTY)
        self.transfer_fee = self.config.get('transfer_fee', V66_TRANSFER_FEE)
        
        self.positions: Dict[str, V66Position] = {}
        self.cash = self.initial_capital
        self.trades: List[V66Trade] = []
        self.sell_history: Dict[str, str] = {}
        
        self.monthly_trades: Dict[str, int] = {}
        self.weekly_trades: Dict[str, int] = {}
        self.total_trades = 0
    
    def _get_month_key(self, date_str: str) -> str:
        return date_str[:7]
    
    def _get_week_key(self, date_str: str) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.year}-W{dt.isocalendar()[1]:02d}"
    
    def _check_trade_limit(self, trade_date: str) -> bool:
        month_key = self._get_month_key(trade_date)
        week_key = self._get_week_key(trade_date)
        
        if self.monthly_trades.get(month_key, 0) >= V66_MONTHLY_TRADE_LIMIT:
            return False
        
        if self.weekly_trades.get(week_key, 0) >= V66_WEEKLY_TRADE_LIMIT:
            return False
        
        if self.total_trades >= V66_GLOBAL_TRADE_LIMIT:
            return False
        
        return True
    
    def _increment_trade_count(self, trade_date: str):
        month_key = self._get_month_key(trade_date)
        week_key = self._get_week_key(trade_date)
        
        self.monthly_trades[month_key] = self.monthly_trades.get(month_key, 0) + 1
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1
        self.total_trades += 1
    
    def execute_buy(self, signal: V66Signal, next_open: float,
                    trigger_price: float, capital: float) -> Optional[V66Trade]:
        """执行买入"""
        execution_price = min(trigger_price, next_open) * (1 + self.slippage_buy)
        
        max_position_value = capital * V66_MAX_SINGLE_POSITION_PCT
        shares = int(max_position_value / execution_price / 100) * 100
        
        if shares <= 0:
            return None
        
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
        
        position = V66Position(
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
            net_main_rate=signal.net_main_rate,
            net_main_rate_vs_avg=signal.net_main_rate_vs_avg,
            main_force_volume_ratio=signal.main_force_volume_ratio,
            rs_percentile=signal.rs_percentile,
            rs_z_score=signal.rs_z_score,
            vcp_amplitude=signal.vcp_amplitude,
            vcp_industry_std=signal.vcp_industry_std,
            vcp_pass=signal.vcp_pass,
            data_credibility=signal.data_credibility,
            stop_loss_price=execution_price * (1 - V66_TRAILING_STOP_RATIO),
            trailing_stop_price=execution_price * (1 - V66_TRAILING_STOP_RATIO),
            trigger_price=trigger_price,
            next_open_price=next_open,
            execution_price=execution_price
        )
        
        self.positions[signal.symbol] = position
        
        trade = V66Trade(
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
            reason='机构资金踪迹',
            signal_date=signal.trade_date,
            trigger_price=trigger_price,
            next_open_price=next_open,
            data_credibility=signal.data_credibility
        )
        
        self.trades.append(trade)
        self.cash -= total_cost
        self._increment_trade_count(signal.trade_date)
        
        logger.info(f"V66 买入成交：{signal.symbol} @ {execution_price:.2f} x {shares}股，数据可信度：{signal.data_credibility}")
        
        return trade
    
    def execute_sell(self, symbol: str, current_price: float,
                     trade_date: str, reason: str) -> Optional[V66Trade]:
        """执行卖出"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        execution_price = current_price * (1 - self.slippage_sell)
        
        shares = position.shares
        amount = shares * execution_price
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        total_cost = commission + stamp_duty + transfer_fee
        
        trade = V66Trade(
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
            data_credibility=position.data_credibility
        )
        
        self.trades.append(trade)
        self.cash += amount - total_cost
        self.sell_history[symbol] = trade_date
        del self.positions[symbol]
        
        logger.info(f"V66 卖出成交：{symbol} @ {execution_price:.2f}, 原因：{reason}")
        
        return trade
    
    def check_exit_conditions(self, symbol: str, current_price: float,
                              trade_date: str, current_net_main_rate: float = 0.0) -> Optional[Tuple[bool, str]]:
        """
        检查离场条件
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        cost_price = position.avg_cost
        
        # 更新最高价和移动止盈价
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.peak_profit = (current_price - cost_price) / cost_price
        
        if position.peak_price > 0:
            position.trailing_stop_price = position.peak_price * (1 - V66_TRAILING_STOP_RATIO)
        
        # 1. 移动止盈
        if position.trailing_stop_price > 0:
            if current_price <= position.trailing_stop_price:
                position.trailing_stop_triggered = True
                return True, f"移动止盈 (回撤>{V66_TRAILING_STOP_RATIO*100:.1f}%)"
        
        # 2. 目标止盈
        current_profit = (current_price - cost_price) / cost_price
        if current_profit >= V66_PROFIT_TARGET_RATIO:
            return True, f"目标止盈 (盈利>{V66_PROFIT_TARGET_RATIO*100:.1f}%)"
        
        return None
    
    def check_trend_break_exit(self, symbol: str, ma10_price: float,
                               net_main_rate: float) -> Optional[Tuple[bool, str]]:
        """
        检查趋势破坏止损
        
        条件：跌破 MA10 且主力资金净流出
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_price = position.current_price
        
        # 趋势破坏条件
        below_ma10 = current_price < ma10_price
        net_outflow = net_main_rate < 0
        
        if below_ma10 and net_outflow:
            position.trend_break_triggered = True
            return True, f"趋势破坏止损 (跌破 MA10:{ma10_price:.2f} 且资金净流出:{net_main_rate:.2%})"
        
        return None
    
    def update_positions(self, market_data: Dict[str, Dict[str, float]],
                         trade_date: str):
        """更新持仓状态"""
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            current_price = data.get('close', 0)
            ma10 = data.get('ma10', 0)
            
            if current_price <= 0:
                continue
            
            position.current_price = current_price
            position.market_value = current_price * position.shares
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            
            # 更新 MA10
            position.stop_loss_price = ma10 if ma10 > 0 else position.stop_loss_price
            
            try:
                buy_date = datetime.strptime(position.buy_date, "%Y-%m-%d")
                current = datetime.strptime(trade_date, "%Y-%m-%d")
                position.holding_days = (current - buy_date).days
            except Exception:
                pass
    
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
# V66 ICCalculator - IC 值计算
# ===========================================

class V66ICCalculator:
    """
    V66 IC 计算器 - 计算信号预测质量 IC 统计
    
    【核心功能】
    - 计算 Rank IC 值（Spearman 相关系数）
    - 计算 IC 序列统计信息
    - 输出《信号预测质量 IC 统计表》
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ic_results: List[V66ICMetrics] = []
    
    def calculate_rank_ic(self, factor_values: np.ndarray,
                          label_values: np.ndarray) -> float:
        """
        计算 Rank IC 值（Spearman 相关系数）
        
        Parameters
        ----------
        factor_values : np.ndarray
            因子值序列
        label_values : np.ndarray
            标签值序列（未来收益率）
            
        Returns
        -------
        float
            Rank IC 值
        """
        # 去除空值
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算秩
        factor_ranks = np.argsort(np.argsort(factor_clean)).astype(float) + 1
        label_ranks = np.argsort(np.argsort(label_clean)).astype(float) + 1
        
        # 计算 Pearson 相关系数（在秩上）
        if np.std(factor_ranks) < self.EPSILON or np.std(label_ranks) < self.EPSILON:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_daily_ic(self, df: pl.DataFrame, trade_date: str,
                           signal_col: str = 'composite_score',
                           return_col: str = 'future_return') -> Tuple[float, float]:
        """
        计算单日的 IC 值
        
        Parameters
        ----------
        df : pl.DataFrame
            包含信号和收益率的 DataFrame
        trade_date : str
            交易日期
        signal_col : str
            信号列名
        return_col : str
            收益率列名
            
        Returns
        -------
        Tuple[float, float]
            (IC, Rank IC)
        """
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, 0.0
            
            if signal_col not in day_data.columns or return_col not in day_data.columns:
                return 0.0, 0.0
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            # 计算 IC（Pearson 相关系数）
            ic = np.corrcoef(signal_values, return_values)[0, 1]
            
            # 计算 Rank IC
            rank_ic = self.calculate_rank_ic(signal_values, return_values)
            
            return (float(ic) if not np.isnan(ic) else 0.0, 
                    float(rank_ic) if not np.isnan(rank_ic) else 0.0)
            
        except Exception as e:
            logger.debug(f"V66 计算 {trade_date} IC 值失败：{e}")
            return 0.0, 0.0
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'future_return') -> List[V66ICMetrics]:
        """
        计算 IC 序列
        
        Parameters
        ----------
        df : pl.DataFrame
            包含信号和收益率的 DataFrame
        signal_col : str
            信号列名
        return_col : str
            收益率列名
            
        Returns
        -------
        List[V66ICMetrics]
            IC 指标列表
        """
        unique_dates = df['trade_date'].unique().to_list()
        ic_series = []
        
        for trade_date in sorted(unique_dates):
            ic, rank_ic = self.calculate_daily_ic(df, trade_date, signal_col, return_col)
            
            ic_metrics = V66ICMetrics(
                trade_date=trade_date,
                factor_name='composite_score',
                ic=ic,
                rank_ic=rank_ic
            )
            ic_series.append(ic_metrics)
        
        self.ic_results = ic_series
        return ic_series
    
    def get_ic_statistics(self) -> Dict[str, float]:
        """
        获取 IC 统计信息
        
        Returns
        -------
        Dict[str, float]
            IC 统计信息
        """
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
    
    def print_ic_report(self, target_mean_ic: float = V66_IC_TARGET_MEAN):
        """
        打印《信号预测质量 IC 统计表》
        
        Parameters
        ----------
        target_mean_ic : float
            目标 IC 均值
        """
        stats = self.get_ic_statistics()
        
        logger.info("=" * 60)
        logger.info("信号预测质量 IC 统计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info("-" * 40)
        logger.info(f"Mean IC:      {stats['mean_ic']:.4f} (目标：>{target_mean_ic})")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f}")
        logger.info(f"IC Std:       {stats['ic_std']:.4f}")
        logger.info(f"Rank IC Std:  {stats['rank_ic_std']:.4f}")
        logger.info("-" * 40)
        logger.info(f"IC IR:        {stats['ic_ir']:.2f}")
        logger.info(f"Rank IC IR:   {stats['rank_ic_ir']:.2f}")
        logger.info(f"Positive Ratio: {stats['positive_ratio']:.1%}")
        logger.info("-" * 40)
        
        if stats['mean_ic'] >= target_mean_ic:
            logger.info(f"✓ IC 达标：Mean IC ({stats['mean_ic']:.4f}) >= 目标 ({target_mean_ic})")
        else:
            logger.info(f"✗ IC 未达标：Mean IC ({stats['mean_ic']:.4f}) < 目标 ({target_mean_ic})")
        
        logger.info("=" * 60)


# ===========================================
# V66 评估指标计算
# ===========================================

def calculate_ae_metric(win_rate: float, profit_loss_ratio: float,
                        max_drawdown: float, trade_count: int) -> float:
    """
    计算 AE (Alpha-Efficiency) 指标
    
    公式：AE = (Win_Rate × P/L_Ratio) / Max_Drawdown × √Trade_Count
    
    Parameters
    ----------
    win_rate : float
        胜率
    profit_loss_ratio : float
        盈亏比
    max_drawdown : float
        最大回撤
    trade_count : int
        交易次数
        
    Returns
    -------
    float
        AE 值
    """
    if max_drawdown <= 0 or trade_count <= 0:
        return 0.0
    
    ae = (win_rate * profit_loss_ratio) / max_drawdown * np.sqrt(trade_count)
    return float(ae)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V66_INITIAL_CAPITAL',
    'V66_MAX_POSITIONS',
    'V66_MONTHLY_TRADE_LIMIT',
    'V66_WEEKLY_TRADE_LIMIT',
    'V66_GLOBAL_TRADE_LIMIT',
    'V66_WARMUP_PERIOD',
    'V66_MIN_SAMPLE_SIZE',
    'V66_RS_WINDOW',
    'V66_RS_Z_SCORE_THRESHOLD',
    'V66_RS_PERCENTILE_THRESHOLD',
    'V66_NET_MAIN_RATE_WINDOW',
    'V66_PULLBACK_MA_PERIOD',
    'V66_PULLBACK_TOLERANCE',
    'V66_VOLUME_RATIO_WINDOW',
    'V66_MAIN_FORCE_VOLUME_RATIO_THRESHOLD',
    'V66_VCP_WINDOW',
    'V66_VCP_VOLATILITY_MULTIPLIER',
    'V66_MARKET_DECLINE_RATIO_THRESHOLD',
    'V66_COMMISSION_RATE',
    'V66_MIN_COMMISSION',
    'V66_SLIPPAGE_BUY',
    'V66_SLIPPAGE_SELL',
    'V66_STAMP_DUTY',
    'V66_TRANSFER_FEE',
    'V66_FRICTION_COST',
    'V66_TREND_BREAK_MA_PERIOD',
    'V66_PROFIT_TARGET_RATIO',
    'V66_TRAILING_STOP_RATIO',
    'V66_MAX_SINGLE_POSITION_PCT',
    'V66_SELECTION_PERCENTILE',
    'V66_IC_TARGET_MEAN',
    
    # 数据类
    'V66Position',
    'V66Trade',
    'V66TradeAudit',
    'V66Signal',
    'V66MarketRegime',
    'V66ICMetrics',
    
    # 数据可信度
    'V66DataCredibility',
    
    # 核心类
    'V66DataManager',
    'V66AlphaCenter',
    'V66TradeExec',
    'V66ICCalculator',
    
    # 评估指标
    'calculate_ae_metric',
]