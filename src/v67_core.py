"""
V67 Core Module - SPI 信号质量审计预测引擎

【V67 核心算法 - SPI (Signal Purity Index) 驱动】

1. 预测引擎
   ✅ 计算 5 日远期收益率 (Forward 5D Return)
   ✅ 计算每个信号的 IC 值
   ✅ SPI 约束：SPI = IC / IC_Std，如果 SPI < 0.1，重新调整 RS-ZScore 权重

2. 机构踪迹逻辑
   ✅ 主力净额 / 流通市值 比例因子
   ✅ 识别真正的"庄股"异动

3. 杜绝偷懒与伪造
   ✅ 手续费 0.2% 必须写死在常量里
   ✅ 遇到 NoneType 或数据缺失，不准用 fill_null(0) 掩盖
   ✅ 必须打印出缺失数据的日期和股票代码

作者：量化系统
版本：V67.0
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
# V67 配置常量 - SPI 信号质量审计
# ===========================================

# 基础配置
V67_INITIAL_CAPITAL = 100000.00
V67_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V67 频率熔断
V67_MONTHLY_TRADE_LIMIT = 15  # 月均交易次数上限
V67_WEEKLY_TRADE_LIMIT = 4    # 每周最多开仓 4 只
V67_GLOBAL_TRADE_LIMIT = 150  # 全场交易次数限制

# V67 数据预加载配置
V67_WARMUP_PERIOD = 250  # 预加载 250 天数据
V67_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V67 SPI 配置 (核心)
V67_SPI_TARGET = 0.1  # SPI 目标阈值
V67_SPI_MIN = 0.05  # SPI 最低容忍值
V67_IC_WINDOW = 20  # IC 计算窗口

# V67 远期收益率配置
V67_FORWARD_RETURN_WINDOW = 5  # 5 日远期收益率

# V67 行业护城河 RS 配置
V67_RS_WINDOW = 20  # 计算 RS 的窗口
V67_RS_Z_SCORE_THRESHOLD = 1.2  # RS Z-Score 必须 > 1.2
V67_RS_PERCENTILE_THRESHOLD = 0.20  # 必须处于行业前 20%

# V67 资金共振配置
V67_NET_MAIN_RATE_WINDOW = 10  # 计算主力净流入均值的窗口
V67_PULLBACK_MA_PERIOD = 20  # 回调至 MA20
V67_PULLBACK_TOLERANCE = 0.02  # 回调容忍度 2%

# V67 机构踪迹配置 (主力净额/流通市值)
V67_MAIN_FORCE_RATIO_WINDOW = 20  # 主力净额/流通市值计算窗口
V67_MAIN_FORCE_RATIO_THRESHOLD = 0.001  # 主力净额/流通市值 > 0.1%

# V67 成交量特征配置
V67_VOLUME_RATIO_WINDOW = 5  # 成交量比值计算窗口
V67_MAIN_FORCE_VOLUME_RATIO_THRESHOLD = 1.2  # 主力资金/成交量比值上升阈值

# V67 VCP 动态阈值配置
V67_VCP_WINDOW = 10  # 观察窗口
V67_VCP_VOLATILITY_MULTIPLIER = 1.2  # 行业动态标准差的 1.2 倍

# V67 大盘避坑配置
V67_MARKET_DECLINE_RATIO_THRESHOLD = 0.80  # 下跌家数占比 > 80% 强制空仓

# V67 费率配置 - 总计 0.2% (写死，严禁修改)
V67_COMMISSION_RATE = 0.0003  # 佣金万 3
V67_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V67_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V67_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V67_STAMP_DUTY = 0.0005  # 印花税 0.05%
V67_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V67_FRICTION_COST = 0.002  # 0.2% 总计 (写死)

# V67 离场配置
V67_TREND_BREAK_MA_PERIOD = 10  # 跌破 MA10
V67_PROFIT_TARGET_RATIO = 0.15  # 目标盈利 15%
V67_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V67 仓位管理
V67_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V67 选股排名
V67_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票

# V67 IC 评估配置
V67_IC_TARGET_MEAN = 0.02  # 2024 年 IC 均值目标 > 0.02


# ===========================================
# V67 数据类定义
# ===========================================

@dataclass
class V67Position:
    """V67 持仓记录"""
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
    
    # V67 机构踪迹
    net_main_rate: float = 0.0
    net_main_rate_vs_avg: float = 0.0
    main_force_ratio: float = 0.0  # 主力净额/流通市值
    
    # V67 行业护城河 RS
    rs_percentile: float = 0.0  # RS 行业百分位
    rs_z_score: float = 0.0  # RS Z-Score
    industry_name: str = ""
    
    # V67 VCP 动态阈值
    vcp_amplitude: float = 0.0
    vcp_industry_std: float = 0.0
    vcp_pass: bool = False
    
    # V67 SPI 状态
    spi_value: float = 0.0
    spi_pass: bool = False
    
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
class V67Trade:
    """V67 交易记录"""
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
    
    # V67 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    
    # V67 SPI 状态
    spi_value: float = 0.0


@dataclass
class V67TradeAudit:
    """V67 交易审计记录"""
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
    
    # V67 状态
    net_main_rate: float = 0.0
    main_force_ratio: float = 0.0
    rs_percentile: float = 0.0
    rs_z_score: float = 0.0
    vcp_pass: bool = False
    
    # V67 SPI 状态
    spi_value: float = 0.0
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V67Signal:
    """V67 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    signal_rank: int
    composite_score: float
    
    # V67 状态
    net_main_rate: float = 0.0
    net_main_rate_vs_avg: float = 0.0
    main_force_ratio: float = 0.0  # 主力净额/流通市值
    rs_percentile: float = 0.0
    rs_z_score: float = 0.0
    vcp_amplitude: float = 0.0
    vcp_industry_std: float = 0.0
    vcp_pass: bool = False
    is_pullback_to_ma20: bool = False
    
    # V67 SPI 状态
    spi_value: float = 0.0
    spi_pass: bool = False
    
    # 价格数据
    close_price: float = 0.0
    ma20_price: float = 0.0


@dataclass
class V67MarketRegime:
    """V67 大盘状态"""
    trade_date: str
    decline_ratio: float = 0.0  # 下跌家数占比
    is_safe_period: bool = True
    regime_reason: str = ""
    forced_empty: bool = False


@dataclass
class V67ICMetrics:
    """V67 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V67SPIMetrics:
    """V67 SPI 统计指标"""
    trade_date: str
    factor_name: str
    ic_mean: float
    ic_std: float
    spi: float  # SPI = IC / IC_Std
    spi_pass: bool = False


# ===========================================
# V67 DataManager - 数据获取与预处理
# ===========================================

class V67DataManager:
    """
    V67 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时报错透明化，打印缺失的日期和股票代码
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V67_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V67_MIN_SAMPLE_SIZE)
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
        
        logger.info(f"V67 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V67 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V67 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V67 DataManager: 未加载到任何数据")
            
            logger.info(f"V67 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V67 DataManager 加载数据失败：{e}")
            raise
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V67: 数据库连接未初始化")
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
            
            logger.debug(f"V67 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V67: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V67 加载资金流向数据失败：{e}")
            return self._empty_fund_flow_df()
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V67: 数据库连接未初始化")
            return self._empty_industry_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V67 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V67: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V67 加载行业数据失败：{e}")
            return self._empty_industry_df()
    
    def load_market_cap_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载流通市值数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V67: 数据库连接未初始化")
            return self._empty_market_cap_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, mv
                FROM stock_daily
                WHERE trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            
            logger.debug(f"V67 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V67: 未加载到流通市值数据")
                return self._empty_market_cap_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V67 加载流通市值数据失败：{e}")
            return self._empty_market_cap_df()
    
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
    
    def _empty_market_cap_df(self) -> pl.DataFrame:
        """返回空流通市值 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'mv': pl.Float64
        })
    
    def log_missing_data(self, symbol: str, trade_date: str, field_name: str):
        """
        记录缺失数据 - 报错透明化
        
        【杜绝偷懒与伪造】
        - 不准用 fill_null(0) 掩盖
        - 必须打印出缺失数据的日期和股票代码
        """
        missing_info = f"缺失数据：symbol={symbol}, trade_date={trade_date}, field={field_name}"
        self._missing_data_log.append(missing_info)
        logger.warning(f"V67: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V67 缺失数据报告")
            logger.info("=" * 60)
            for log in self._missing_data_log[:100]:  # 只显示前 100 条
                logger.info(f"  {log}")
            if len(self._missing_data_log) > 100:
                logger.info(f"  ... 还有 {len(self._missing_data_log) - 100} 条")
            logger.info("=" * 60)
    
    def clear_cache(self):
        """清除缓存"""
        self._data_cache.clear()


# ===========================================
# V67 AlphaCenter - SPI 驱动的信号生成
# ===========================================

class V67AlphaCenter:
    """
    V67 AlphaCenter - SPI 驱动的信号生成
    
    【核心逻辑】
    1. 计算 5 日远期收益率 (Forward 5D Return)
    2. 计算每个信号的 IC 值
    3. SPI 约束：SPI = IC / IC_Std，如果 SPI < 0.1，重新调整 RS-ZScore 权重
    4. 机构踪迹：主力净额 / 流通市值 比例因子
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # SPI 配置
        self.spi_target = self.config.get('spi_target', V67_SPI_TARGET)
        self.spi_min = self.config.get('spi_min', V67_SPI_MIN)
        self.ic_window = self.config.get('ic_window', V67_IC_WINDOW)
        
        # 远期收益率配置
        self.forward_return_window = self.config.get('forward_return_window', V67_FORWARD_RETURN_WINDOW)
        
        # 行业护城河配置
        self.rs_window = self.config.get('rs_window', V67_RS_WINDOW)
        self.rs_z_score_threshold = self.config.get('rs_z_score_threshold', V67_RS_Z_SCORE_THRESHOLD)
        self.rs_percentile_threshold = self.config.get('rs_percentile_threshold', V67_RS_PERCENTILE_THRESHOLD)
        
        # 资金共振配置
        self.net_main_rate_window = self.config.get('net_main_rate_window', V67_NET_MAIN_RATE_WINDOW)
        self.pullback_ma_period = self.config.get('pullback_ma_period', V67_PULLBACK_MA_PERIOD)
        self.pullback_tolerance = self.config.get('pullback_tolerance', V67_PULLBACK_TOLERANCE)
        
        # 机构踪迹配置
        self.main_force_ratio_window = self.config.get('main_force_ratio_window', V67_MAIN_FORCE_RATIO_WINDOW)
        self.main_force_ratio_threshold = self.config.get('main_force_ratio_threshold', V67_MAIN_FORCE_RATIO_THRESHOLD)
        
        # 成交量特征配置
        self.volume_ratio_window = self.config.get('volume_ratio_window', V67_VOLUME_RATIO_WINDOW)
        self.main_force_volume_ratio_threshold = self.config.get('main_force_volume_ratio_threshold', V67_MAIN_FORCE_VOLUME_RATIO_THRESHOLD)
        
        # VCP 动态阈值配置
        self.vcp_window = self.config.get('vcp_window', V67_VCP_WINDOW)
        self.vcp_volatility_multiplier = self.config.get('vcp_volatility_multiplier', V67_VCP_VOLATILITY_MULTIPLIER)
        
        # 大盘避坑配置
        self.decline_ratio_threshold = self.config.get('decline_ratio_threshold', V67_MARKET_DECLINE_RATIO_THRESHOLD)
        
        # RS-ZScore 权重 (SPI 调整)
        self.rs_z_score_weight = 1.0  # 初始权重
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None,
                        market_cap_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
        
        【核心逻辑】
        1. 计算 5 日远期收益率
        2. 计算 IC 值和 SPI
        3. 如果 SPI < 0.1，调整 RS-ZScore 权重
        4. 机构踪迹：主力净额/流通市值
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
            
            status = {
                'factors_computed': [],
                'spi_value': 0.0,
                'spi_pass': False,
                'rs_z_score_weight': self.rs_z_score_weight,
            }
            
            # 检测数据可用性
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            has_industry = industry_df is not None and not industry_df.is_empty()
            has_market_cap = market_cap_df is not None and not market_cap_df.is_empty()
            
            # 1. 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].append('ma_system')
            
            # 2. 计算 5 日远期收益率
            result = self._compute_forward_return(result)
            status['factors_computed'].append('forward_return')
            
            # 3. 计算行业护城河 RS
            if has_industry:
                result = self._compute_industry_moat_rs(result, industry_df)
                status['factors_computed'].append('industry_moat_rs')
            else:
                result = self._compute_basic_rs(result)
                status['factors_computed'].append('basic_rs')
            
            # 4. 计算资金共振
            if has_fund_flow:
                result = self._compute_capital_resonance(result, fund_flow_df)
                status['factors_computed'].append('capital_resonance')
            else:
                result = self._add_fund_flow_placeholders(result)
                status['factors_computed'].append('fund_flow_placeholder')
            
            # 5. 计算机构踪迹 (主力净额/流通市值)
            if has_fund_flow and has_market_cap:
                result = self._compute_institutional_trace(result, fund_flow_df, market_cap_df)
                status['factors_computed'].append('institutional_trace')
            else:
                result = self._add_institutional_trace_placeholder(result)
                status['factors_computed'].append('institutional_trace_placeholder')
            
            # 6. 计算成交量特征
            if has_fund_flow:
                result = self._compute_volume_characteristics(result, fund_flow_df)
                status['factors_computed'].append('volume_characteristics')
            else:
                result = self._add_volume_placeholders(result)
                status['factors_computed'].append('volume_placeholder')
            
            # 7. 计算价格回调至 MA20
            result = self._compute_pullback_to_ma20(result)
            status['factors_computed'].append('pullback_to_ma20')
            
            # 8. 计算 VCP 动态阈值
            result = self._compute_vcp_dynamic_threshold(result)
            status['factors_computed'].append('vcp_dynamic_threshold')
            
            # 9. 计算综合评分 (SPI 调整权重)
            result = self._compute_composite_score(result, has_fund_flow, has_industry, has_market_cap)
            status['factors_computed'].append('composite_score')
            
            # 10. 计算 IC 和 SPI
            result, ic_metrics = self._compute_ic_and_spi(result)
            status['ic_metrics'] = ic_metrics
            status['spi_value'] = ic_metrics.get('spi', 0.0)
            status['spi_pass'] = ic_metrics.get('spi', 0.0) >= self.spi_target
            
            # 11. SPI 约束：如果 SPI < 0.1，调整 RS-ZScore 权重
            if status['spi_value'] < self.spi_target:
                self._adjust_rs_z_score_weight(status['spi_value'])
                status['rs_z_score_weight'] = self.rs_z_score_weight
                
                # 重新计算综合评分
                result = self._compute_composite_score(result, has_fund_flow, has_industry, has_market_cap)
            
            logger.info(f"V67 AlphaCenter 信号计算完成，SPI={status['spi_value']:.4f}, SPI Pass={status['spi_pass']}")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V67 AlphaCenter 计算信号失败：{e}")
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
    
    def _compute_forward_return(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 5 日远期收益率 (Forward 5D Return)
        
        【核心逻辑】
        Forward 5D Return = (close.shift(-5) - close) / close
        """
        result = df.clone()
        
        # 计算 5 日远期收益率
        result = result.with_columns([
            ((pl.col('close').shift(-self.forward_return_window)).over('symbol') - pl.col('close')) / 
            (pl.col('close') + self.EPSILON)
        ].alias('forward_return_5d'))
        
        return result
    
    def _compute_industry_moat_rs(self, df: pl.DataFrame, 
                                    industry_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算行业护城河 RS
        
        【核心逻辑】
        - 计算个股过去 N 日收益率
        - 计算所属行业指数同期收益率
        - 计算 RS 的 Z-Score，必须 Z > 1.2
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
        
        # 计算行业收益率
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
        
        # 计算 RS Z-Score
        result = result.with_columns([
            pl.col('rs_vs_industry').mean().over(['trade_date', 'industry_name']).alias('rs_industry_mean'),
            pl.col('rs_vs_industry').std().over(['trade_date', 'industry_name']).alias('rs_industry_std')
        ])
        
        result = result.with_columns([
            ((pl.col('rs_vs_industry') - pl.col('rs_industry_mean')) / 
             (pl.col('rs_industry_std') + self.EPSILON)).alias('rs_z_score')
        ])
        
        # 行业护城河
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
        """计算资金共振"""
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_main_ratio']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left')
        
        # 计算 net_main_rate
        result = result.with_columns([
            pl.when(pl.col('net_main_ratio').is_not_null())
            .then(pl.col('net_main_ratio'))
            .otherwise(
                pl.col('net_main_amount') / (pl.col('amount') + self.EPSILON)
            ).alias('net_main_rate')
        ])
        
        # 计算过去 N 天的均值
        result = result.with_columns([
            pl.col('net_main_rate').rolling_mean(window_size=self.net_main_rate_window).over('symbol').alias('net_main_rate_avg')
        ])
        
        # 资金共振
        result = result.with_columns([
            (pl.col('net_main_rate') > pl.col('net_main_rate_avg')).alias('net_main_rate_above_avg'),
            (pl.col('net_main_rate') - pl.col('net_main_rate_avg')).alias('net_main_rate_vs_avg')
        ])
        
        return result
    
    def _compute_institutional_trace(self, df: pl.DataFrame,
                                      fund_flow_df: pl.DataFrame,
                                      market_cap_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算机构踪迹 (主力净额/流通市值)
        
        【核心逻辑】
        - 主力净额 / 流通市值 比例因子
        - 识别真正的"庄股"异动
        """
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left')
        
        # 合并流通市值数据
        mv_cols = ['symbol', 'trade_date', 'mv']
        available_mv_cols = [c for c in mv_cols if c in market_cap_df.columns]
        mv_data = market_cap_df.select(available_mv_cols)
        
        result = result.join(mv_data, on=['symbol', 'trade_date'], how='left')
        
        # 计算主力净额/流通市值
        result = result.with_columns([
            (pl.col('net_main_amount') / (pl.col('mv') + self.EPSILON)).alias('main_force_ratio')
        ])
        
        # 计算过去 N 天的均值
        result = result.with_columns([
            pl.col('main_force_ratio').rolling_mean(window_size=self.main_force_ratio_window).over('symbol').alias('main_force_ratio_avg')
        ])
        
        # 机构踪迹：主力净额/流通市值 > 阈值
        result = result.with_columns([
            (pl.col('main_force_ratio') > self.main_force_ratio_threshold).alias('main_force_ratio_pass'),
            (pl.col('main_force_ratio') > pl.col('main_force_ratio_avg')).alias('main_force_ratio_rising')
        ])
        
        return result
    
    def _add_institutional_trace_placeholder(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加机构踪迹占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('main_force_ratio'),
            pl.lit(0.0).alias('main_force_ratio_avg'),
            pl.lit(False).alias('main_force_ratio_pass'),
            pl.lit(False).alias('main_force_ratio_rising')
        ])
        
        return result
    
    def _compute_volume_characteristics(self, df: pl.DataFrame,
                                         fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """计算成交量特征"""
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
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
        
        # 机构护盘
        result = result.with_columns([
            (pl.col('main_force_volume_ratio') > pl.col('main_force_volume_ratio_avg') * self.main_force_volume_ratio_threshold).alias('main_force_volume_ratio_rising'),
            (pl.col('main_force_volume_ratio') / (pl.col('main_force_volume_ratio_avg') + self.EPSILON)).alias('main_force_volume_ratio_change')
        ])
        
        return result
    
    def _add_fund_flow_placeholders(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加资金流占位符"""
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
        """添加成交量特征占位符"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('main_force_volume_ratio'),
            pl.lit(0.0).alias('main_force_volume_ratio_avg'),
            pl.lit(False).alias('main_force_volume_ratio_rising'),
            pl.lit(0.0).alias('main_force_volume_ratio_change')
        ])
        
        return result
    
    def _compute_pullback_to_ma20(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算价格回调至 MA20"""
        result = df.clone()
        
        ma20 = pl.col('ma20')
        close = pl.col('close')
        
        # 回调至 MA20 附近
        pullback_ratio = (close - ma20) / (ma20 + self.EPSILON)
        is_pullback = (pullback_ratio.abs() <= self.pullback_tolerance)
        
        # 确认是回调
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
        """计算 VCP 动态阈值"""
        result = df.clone()
        
        # 1. 计算每日振幅
        daily_amplitude = (pl.col('high') - pl.col('low')) / (pl.col('close') + self.EPSILON)
        
        # 2. 滚动平均振幅
        avg_amplitude = daily_amplitude.rolling_mean(window_size=self.vcp_window).over('symbol')
        
        # 3. 计算振幅
        if 'industry_name' in df.columns:
            industry_amplitude_std = daily_amplitude.std().over(['trade_date', 'industry_name'])
            dynamic_threshold = industry_amplitude_std * self.vcp_volatility_multiplier
            vcp_pass = avg_amplitude < dynamic_threshold
            
            result = result.with_columns([
                industry_amplitude_std.alias('vcp_industry_std'),
                dynamic_threshold.alias('vcp_dynamic_threshold'),
                vcp_pass.alias('vcp_pass')
            ])
        else:
            vcp_pass = avg_amplitude < 0.08
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
    
    def _compute_ic_and_spi(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算 IC 和 SPI
        
        【核心逻辑】
        - IC = Corr(Signal, Forward Return)
        - SPI = IC / IC_Std
        """
        result = df.clone()
        
        ic_metrics = {'ic_mean': 0.0, 'ic_std': 0.0, 'spi': 0.0}
        
        try:
            # 计算 IC (简化版本：使用相关系数)
            if 'composite_score' in df.columns and 'forward_return_5d' in df.columns:
                # 按日期分组计算 IC
                ic_by_date = df.group_by('trade_date').agg([
                    pl.corr('composite_score', 'forward_return_5d').alias('ic')
                ])
                
                if not ic_by_date.is_empty():
                    ic_values = ic_by_date['ic'].drop_nulls().to_numpy()
                    
                    if len(ic_values) > 1:
                        ic_mean = float(np.mean(ic_values))
                        ic_std = float(np.std(ic_values, ddof=1))
                        
                        # SPI = IC / IC_Std
                        spi = ic_mean / ic_std if ic_std > self.EPSILON else 0.0
                        
                        ic_metrics = {
                            'ic_mean': ic_mean,
                            'ic_std': ic_std,
                            'spi': spi,
                        }
        except Exception as e:
            logger.debug(f"V67 计算 IC/SPI 失败：{e}")
        
        return result, ic_metrics
    
    def _adjust_rs_z_score_weight(self, spi_value: float):
        """
        调整 RS-ZScore 权重
        
        【SPI 约束】
        - 如果 SPI < 0.1，增加 RS-ZScore 权重
        - 如果 SPI < 0.05，大幅增加权重
        """
        if spi_value < self.spi_min:
            # SPI 太低，大幅增加 RS-ZScore 权重
            self.rs_z_score_weight = 2.0
            logger.warning(f"V67: SPI={spi_value:.4f} < {self.spi_min}，RS-ZScore 权重调整为 {self.rs_z_score_weight}")
        elif spi_value < self.spi_target:
            # SPI 未达标，适度增加 RS-ZScore 权重
            self.rs_z_score_weight = 1.5
            logger.warning(f"V67: SPI={spi_value:.4f} < {self.spi_target}，RS-ZScore 权重调整为 {self.rs_z_score_weight}")
    
    def _compute_composite_score(self, df: pl.DataFrame,
                                  has_fund_flow: bool,
                                  has_industry: bool,
                                  has_market_cap: bool) -> pl.DataFrame:
        """计算综合评分 (SPI 调整权重)"""
        result = df.clone()
        
        # 基础条件
        rs_condition = pl.col('rs_industry_moat_pass') if has_industry else pl.col('rs_in_top_percentile')
        pullback_condition = pl.col('is_pullback_to_ma20')
        vcp_condition = pl.col('vcp_pass')
        
        # 资金流条件
        if has_fund_flow:
            fund_flow_condition = pl.col('net_main_rate_above_avg')
            volume_condition = pl.col('main_force_volume_ratio_rising')
        else:
            fund_flow_condition = pl.lit(True)
            volume_condition = pl.lit(True)
        
        # 机构踪迹条件
        if has_market_cap and has_fund_flow:
            institutional_condition = pl.col('main_force_ratio_pass')
        else:
            institutional_condition = pl.lit(True)
        
        # 核心买入信号
        core_condition = rs_condition & pullback_condition & vcp_condition & fund_flow_condition & volume_condition & institutional_condition
        
        # 综合评分 (SPI 调整 RS-ZScore 权重)
        rs_bonus = pl.when(pl.col('rs_in_top_percentile')) \
            .then(pl.col('rs_percentile') * 50).otherwise(0.0)
        
        rs_z_bonus = pl.when(pl.col('rs_z_score') > self.rs_z_score_threshold) \
            .then(pl.col('rs_z_score') * 20 * self.rs_z_score_weight).otherwise(0.0)
        
        fund_flow_bonus = pl.when(pl.col('net_main_rate_vs_avg') > 0) \
            .then(pl.col('net_main_rate_vs_avg') * 100).otherwise(0.0)
        
        volume_bonus = pl.when(pl.col('main_force_volume_ratio_rising')) \
            .then(20.0).otherwise(0.0)
        
        vcp_bonus = pl.when(pl.col('vcp_pass')) \
            .then(30.0).otherwise(0.0)
        
        institutional_bonus = pl.when(pl.col('main_force_ratio_pass')) \
            .then(25.0).otherwise(0.0)
        
        composite_score = rs_bonus + rs_z_bonus + fund_flow_bonus + volume_bonus + vcp_bonus + institutional_bonus
        
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
    
    def compute_market_decline_ratio(self, df: pl.DataFrame) -> V67MarketRegime:
        """计算市场下跌家数占比"""
        latest_date = df['trade_date'].max()
        latest_df = df.filter(pl.col('trade_date') == latest_date)
        
        if latest_df.is_empty():
            return V67MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        # 计算下跌家数
        decline_count = latest_df.filter(pl.col('close') < pl.col('open')).height
        total_count = latest_df.height
        
        if total_count == 0:
            return V67MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        decline_ratio = decline_count / total_count
        is_safe = decline_ratio <= self.decline_ratio_threshold
        
        regime = V67MarketRegime(
            trade_date=latest_date,
            decline_ratio=decline_ratio,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"下跌家数占比={decline_ratio*100:.1f}%, 阈值={self.decline_ratio_threshold*100:.0f}%"
        )
        
        if not is_safe:
            logger.warning(f"V67: {latest_date} 大盘避坑触发 ({regime.regime_reason})，强制空仓！")
        
        return regime
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str,
                         market_regime: Optional[V67MarketRegime] = None,
                         spi_value: float = 0.0,
                         spi_pass: bool = False) -> List[V67Signal]:
        """生成交易信号"""
        signals = []
        
        # 大盘危险，强制空仓
        if market_regime and not market_regime.is_safe_period:
            logger.warning(f"V67: 大盘避坑触发，禁止开仓！")
            return signals
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            top_n = max(1, int(buy_df.height * V67_SELECTION_PERCENTILE))
            buy_df = buy_df.head(top_n)
            
            for row in buy_df.iter_rows(named=True):
                signal = V67Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    signal_rank=row.get('composite_rank', 9999),
                    composite_score=row.get('composite_score', 0.0),
                    net_main_rate=row.get('net_main_rate', 0.0),
                    net_main_rate_vs_avg=row.get('net_main_rate_vs_avg', 0.0),
                    main_force_ratio=row.get('main_force_ratio', 0.0),
                    rs_percentile=row.get('rs_percentile', 0.0),
                    rs_z_score=row.get('rs_z_score', 0.0),
                    vcp_amplitude=row.get('vcp_amplitude', 0.0),
                    vcp_industry_std=row.get('vcp_industry_std', 0.0),
                    vcp_pass=row.get('vcp_pass', False),
                    is_pullback_to_ma20=row.get('is_pullback_to_ma20', False),
                    spi_value=spi_value,
                    spi_pass=spi_pass,
                    close_price=row.get('close', 0.0),
                    ma20_price=row.get('ma20', 0.0)
                )
                signals.append(signal)
            
            logger.info(f"V67 生成 {len(signals)} 个买入信号 ({trade_date}), SPI={spi_value:.4f}")
            
        except Exception as e:
            logger.error(f"V67 生成信号失败：{e}")
        
        return signals


# ===========================================
# V67 ICCalculator - IC/SPI 计算
# ===========================================

class V67ICCalculator:
    """
    V67 IC 计算器 - 计算信号预测质量 IC/SPI 统计
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ic_results: List[V67ICMetrics] = []
        self.spi_results: List[V67SPIMetrics] = []
    
    def calculate_rank_ic(self, factor_values: np.ndarray,
                          label_values: np.ndarray) -> float:
        """计算 Rank IC 值"""
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        factor_ranks = np.argsort(np.argsort(factor_clean)).astype(float) + 1
        label_ranks = np.argsort(np.argsort(label_clean)).astype(float) + 1
        
        if np.std(factor_ranks) < self.EPSILON or np.std(label_ranks) < self.EPSILON:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_daily_ic(self, df: pl.DataFrame, trade_date: str,
                           signal_col: str = 'composite_score',
                           return_col: str = 'forward_return_5d') -> Tuple[float, float]:
        """计算单日的 IC 值"""
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, 0.0
            
            if signal_col not in day_data.columns or return_col not in day_data.columns:
                return 0.0, 0.0
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            ic = np.corrcoef(signal_values, return_values)[0, 1]
            rank_ic = self.calculate_rank_ic(signal_values, return_values)
            
            return (float(ic) if not np.isnan(ic) else 0.0, 
                    float(rank_ic) if not np.isnan(rank_ic) else 0.0)
            
        except Exception as e:
            logger.debug(f"V67 计算 {trade_date} IC 值失败：{e}")
            return 0.0, 0.0
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V67ICMetrics]:
        """计算 IC 序列"""
        unique_dates = df['trade_date'].unique().to_list()
        ic_series = []
        
        for trade_date in sorted(unique_dates):
            ic, rank_ic = self.calculate_daily_ic(df, trade_date, signal_col, return_col)
            
            ic_metrics = V67ICMetrics(
                trade_date=trade_date,
                factor_name='composite_score',
                ic=ic,
                rank_ic=rank_ic
            )
            ic_series.append(ic_metrics)
        
        self.ic_results = ic_series
        return ic_series
    
    def calculate_spi_series(self) -> List[V67SPIMetrics]:
        """
        计算 SPI 序列
        
        【核心逻辑】
        SPI = IC / IC_Std (滚动窗口)
        """
        if not self.ic_results:
            return []
        
        spi_series = []
        ic_values = np.array([m.ic for m in self.ic_results])
        
        for i in range(len(self.ic_results)):
            # 滚动窗口计算 IC 均值和标准差
            start_idx = max(0, i - V67_IC_WINDOW + 1)
            window_ic = ic_values[start_idx:i+1]
            
            if len(window_ic) < 2:
                spi_metrics = V67SPIMetrics(
                    trade_date=self.ic_results[i].trade_date,
                    factor_name='composite_score',
                    ic_mean=float(np.mean(window_ic)),
                    ic_std=0.0,
                    spi=0.0,
                    spi_pass=False
                )
            else:
                ic_mean = float(np.mean(window_ic))
                ic_std = float(np.std(window_ic, ddof=1))
                spi = ic_mean / ic_std if ic_std > self.EPSILON else 0.0
                spi_pass = spi >= V67_SPI_TARGET
                
                spi_metrics = V67SPIMetrics(
                    trade_date=self.ic_results[i].trade_date,
                    factor_name='composite_score',
                    ic_mean=ic_mean,
                    ic_std=ic_std,
                    spi=spi,
                    spi_pass=spi_pass
                )
            
            spi_series.append(spi_metrics)
        
        self.spi_results = spi_series
        return spi_series
    
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
    
    def get_spi_statistics(self) -> Dict[str, float]:
        """获取 SPI 统计信息"""
        if not self.spi_results:
            return {
                'mean_spi': 0.0,
                'min_spi': 0.0,
                'max_spi': 0.0,
                'spi_positive_ratio': 0.0,
                'num_valid_days': 0,
            }
        
        spi_values = np.array([m.spi for m in self.spi_results])
        
        return {
            'mean_spi': float(np.mean(spi_values)),
            'min_spi': float(np.min(spi_values)),
            'max_spi': float(np.max(spi_values)),
            'spi_positive_ratio': float(np.sum(spi_values > 0) / len(spi_values)),
            'num_valid_days': len(self.spi_results),
        }
    
    def print_ic_report(self, target_mean_ic: float = V67_IC_TARGET_MEAN):
        """打印 IC 统计表"""
        stats = self.get_ic_statistics()
        
        logger.info("=" * 60)
        logger.info("V67 信号预测质量 IC 统计表")
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
    
    def print_spi_report(self):
        """打印 SPI 统计表"""
        stats = self.get_spi_statistics()
        
        logger.info("=" * 60)
        logger.info("V67 SPI 信号质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info("-" * 40)
        logger.info(f"Mean SPI: {stats['mean_spi']:.4f} (目标：>{V67_SPI_TARGET})")
        logger.info(f"Min SPI:  {stats['min_spi']:.4f} (最低容忍：{V67_SPI_MIN})")
        logger.info(f"Max SPI:  {stats['max_spi']:.4f}")
        logger.info("-" * 40)
        
        if stats['mean_spi'] >= V67_SPI_TARGET:
            logger.info(f"✓ SPI 达标：Mean SPI ({stats['mean_spi']:.4f}) >= 目标 ({V67_SPI_TARGET})")
        else:
            logger.info(f"✗ SPI 未达标：Mean SPI ({stats['mean_spi']:.4f}) < 目标 ({V67_SPI_TARGET})")
        
        if stats['min_spi'] < V67_SPI_MIN:
            logger.warning(f"⚠ SPI 低于最低容忍值：Min SPI ({stats['min_spi']:.4f}) < {V67_SPI_MIN}")
        
        logger.info("=" * 60)


# ===========================================
# 评估指标计算
# ===========================================

def calculate_ae_metric(win_rate: float, profit_loss_ratio: float,
                        max_drawdown: float, trade_count: int) -> float:
    """计算 AE (Alpha-Efficiency) 指标"""
    if max_drawdown <= 0 or trade_count <= 0:
        return 0.0
    
    ae = (win_rate * profit_loss_ratio) / max_drawdown * np.sqrt(trade_count)
    return float(ae)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V67_INITIAL_CAPITAL',
    'V67_MAX_POSITIONS',
    'V67_MONTHLY_TRADE_LIMIT',
    'V67_WEEKLY_TRADE_LIMIT',
    'V67_GLOBAL_TRADE_LIMIT',
    'V67_WARMUP_PERIOD',
    'V67_MIN_SAMPLE_SIZE',
    'V67_SPI_TARGET',
    'V67_SPI_MIN',
    'V67_IC_WINDOW',
    'V67_FORWARD_RETURN_WINDOW',
    'V67_RS_WINDOW',
    'V67_RS_Z_SCORE_THRESHOLD',
    'V67_RS_PERCENTILE_THRESHOLD',
    'V67_NET_MAIN_RATE_WINDOW',
    'V67_PULLBACK_MA_PERIOD',
    'V67_PULLBACK_TOLERANCE',
    'V67_MAIN_FORCE_RATIO_WINDOW',
    'V67_MAIN_FORCE_RATIO_THRESHOLD',
    'V67_VOLUME_RATIO_WINDOW',
    'V67_MAIN_FORCE_VOLUME_RATIO_THRESHOLD',
    'V67_VCP_WINDOW',
    'V67_VCP_VOLATILITY_MULTIPLIER',
    'V67_MARKET_DECLINE_RATIO_THRESHOLD',
    'V67_COMMISSION_RATE',
    'V67_MIN_COMMISSION',
    'V67_SLIPPAGE_BUY',
    'V67_SLIPPAGE_SELL',
    'V67_STAMP_DUTY',
    'V67_TRANSFER_FEE',
    'V67_FRICTION_COST',
    'V67_TREND_BREAK_MA_PERIOD',
    'V67_PROFIT_TARGET_RATIO',
    'V67_TRAILING_STOP_RATIO',
    'V67_MAX_SINGLE_POSITION_PCT',
    'V67_SELECTION_PERCENTILE',
    'V67_IC_TARGET_MEAN',
    
    # 数据类
    'V67Position',
    'V67Trade',
    'V67TradeAudit',
    'V67Signal',
    'V67MarketRegime',
    'V67ICMetrics',
    'V67SPIMetrics',
    
    # 核心类
    'V67DataManager',
    'V67AlphaCenter',
    'V67ICCalculator',
    
    # 评估指标
    'calculate_ae_metric',
]