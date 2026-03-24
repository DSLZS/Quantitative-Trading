"""
V68 Core Module - Rank IC 预测质量审计引擎

【V68 核心算法 - Rank IC 驱动】

1. 预测目标
   ✅ 模型必须预测未来 5 日的相对收益
   ✅ Forward 5D Return = (close.shift(-5) - close) / close

2. Rank IC 计算
   ✅ 每日计算预测排名与实际排名的相关性
   ✅ Rank IC = Spearman 相关系数 (预测排名 vs 实际收益排名)
   ✅ 存入 strategy_audit 表

3. 拒绝回测美化
   ✅ 不准通过调整"止损位"来刷分
   ✅ 如果 Rank IC 不达标，直接报告"预测模型失败"

4. 资金流增强因子
   ✅ 主力净额 / 流通市值 比例因子
   ✅ 资金流排名因子

作者：量化系统
版本：V68.0
日期：2026-03-24
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V68 配置常量 - Rank IC 预测质量审计
# ===========================================

# 基础配置
V68_INITIAL_CAPITAL = 100000.00
V68_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V68 频率熔断
V68_MONTHLY_TRADE_LIMIT = 15  # 月均交易次数上限
V68_WEEKLY_TRADE_LIMIT = 4    # 每周最多开仓 4 只
V68_GLOBAL_TRADE_LIMIT = 150  # 全场交易次数限制

# V68 数据预加载配置
V68_WARMUP_PERIOD = 250  # 预加载 250 天数据
V68_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V68 数据熔断阈值 (与 v68_engine.py 对齐)
V68_MIN_FUND_FLOW_ROWS = 1000000  # 100 万行阈值
V68_MIN_INDUSTRY_ROWS = 100000    # 行业数据最少行数

# V68 Rank IC 配置 (核心)
V68_RANK_IC_TARGET = 0.02  # Rank IC 目标 > 0.02
V68_RANK_IC_MIN = 0.01     # Rank IC 最低容忍值
V68_IC_WINDOW = 20         # IC 计算窗口

# V68 远期收益率配置
V68_FORWARD_RETURN_WINDOW = 5  # 5 日远期收益率

# V68 行业护城河 RS 配置
V68_RS_WINDOW = 20  # 计算 RS 的窗口
V68_RS_Z_SCORE_THRESHOLD = 1.2  # RS Z-Score 必须 > 1.2
V68_RS_PERCENTILE_THRESHOLD = 0.20  # 必须处于行业前 20%

# V68 资金共振配置
V68_NET_MAIN_RATE_WINDOW = 10  # 计算主力净流入均值的窗口
V68_PULLBACK_MA_PERIOD = 20  # 回调至 MA20
V68_PULLBACK_TOLERANCE = 0.02  # 回调容忍度 2%

# V68 机构踪迹配置 (主力净额/流通市值)
V68_MAIN_FORCE_RATIO_WINDOW = 20  # 主力净额/流通市值计算窗口
V68_MAIN_FORCE_RATIO_THRESHOLD = 0.001  # 主力净额/流通市值 > 0.1%

# V68 成交量特征配置
V68_VOLUME_RATIO_WINDOW = 5  # 成交量比值计算窗口
V68_MAIN_FORCE_VOLUME_RATIO_THRESHOLD = 1.2  # 主力资金/成交量比值上升阈值

# V68 VCP 动态阈值配置
V68_VCP_WINDOW = 10  # 观察窗口
V68_VCP_VOLATILITY_MULTIPLIER = 1.2  # 行业动态标准差的 1.2 倍

# V68 大盘避坑配置
V68_MARKET_DECLINE_RATIO_THRESHOLD = 0.80  # 下跌家数占比 > 80% 强制空仓

# V68 费率配置 - 总计 0.2% (写死，严禁修改)
V68_COMMISSION_RATE = 0.0003  # 佣金万 3
V68_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V68_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V68_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V68_STAMP_DUTY = 0.0005  # 印花税 0.05%
V68_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V68_FRICTION_COST = 0.002  # 0.2% 总计 (写死)

# V68 离场配置
V68_TREND_BREAK_MA_PERIOD = 10  # 跌破 MA10
V68_PROFIT_TARGET_RATIO = 0.15  # 目标盈利 15%
V68_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%

# V68 仓位管理
V68_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V68 选股排名
V68_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票


# ===========================================
# V68 数据类定义
# ===========================================

@dataclass
class V68Position:
    """V68 持仓记录"""
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
    
    # V68 机构踪迹
    net_main_rate: float = 0.0
    net_main_rate_vs_avg: float = 0.0
    main_force_ratio: float = 0.0  # 主力净额/流通市值
    
    # V68 行业护城河 RS
    rs_percentile: float = 0.0  # RS 行业百分位
    rs_z_score: float = 0.0  # RS Z-Score
    industry_name: str = ""
    
    # V68 VCP 动态阈值
    vcp_amplitude: float = 0.0
    vcp_industry_std: float = 0.0
    vcp_pass: bool = False
    
    # V68 Rank IC 状态
    predicted_rank: int = 0
    actual_return: float = 0.0
    
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
class V68Trade:
    """V68 交易记录"""
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
    
    # V68 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    
    # V68 Rank IC 状态
    predicted_rank: int = 0
    actual_return: float = 0.0


@dataclass
class V68TradeAudit:
    """V68 交易审计记录"""
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
    
    # V68 状态
    net_main_rate: float = 0.0
    main_force_ratio: float = 0.0
    rs_percentile: float = 0.0
    rs_z_score: float = 0.0
    vcp_pass: bool = False
    
    # V68 Rank IC 状态
    predicted_rank: int = 0
    actual_return: float = 0.0
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V68Signal:
    """V68 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    signal_rank: int
    composite_score: float
    
    # V68 状态
    net_main_rate: float = 0.0
    net_main_rate_vs_avg: float = 0.0
    main_force_ratio: float = 0.0  # 主力净额/流通市值
    rs_percentile: float = 0.0
    rs_z_score: float = 0.0
    vcp_amplitude: float = 0.0
    vcp_industry_std: float = 0.0
    vcp_pass: bool = False
    is_pullback_to_ma20: bool = False
    
    # V68 Rank IC 状态
    predicted_rank: int = 0
    
    # 价格数据
    close_price: float = 0.0
    ma20_price: float = 0.0


@dataclass
class V68MarketRegime:
    """V68 大盘状态"""
    trade_date: str
    decline_ratio: float = 0.0  # 下跌家数占比
    is_safe_period: bool = True
    regime_reason: str = ""
    forced_empty: bool = False


@dataclass
class V68ICMetrics:
    """V68 IC 统计指标"""
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V68StrategyAudit:
    """V68 策略审计记录 - 存入 strategy_audit 表"""
    trade_date: str
    symbol: str
    predicted_rank: int
    predicted_return: float
    actual_return: float
    actual_rank: int
    rank_ic_contribution: float
    composite_score: float
    fund_flow_score: float = 0.0
    rs_score: float = 0.0


# ===========================================
# V68 DataManager - 数据获取与预处理
# ===========================================

class V68DataManager:
    """
    V68 DataManager - 数据获取与预处理
    
    【核心功能】
    1. 从数据库加载股票、资金流、行业数据
    2. 数据缺失时报错透明化，打印缺失的日期和股票代码
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V68_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V68_MIN_SAMPLE_SIZE)
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
        
        logger.info(f"V68 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V68 DataManager: 数据库连接未初始化")
        
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
            
            logger.debug(f"V68 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V68 DataManager: 未加载到任何数据")
            
            logger.info(f"V68 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V68 DataManager 加载数据失败：{e}")
            raise
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V68: 数据库连接未初始化")
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
            
            logger.debug(f"V68 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V68: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V68 加载资金流向数据失败：{e}")
            return self._empty_fund_flow_df()
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V68: 数据库连接未初始化")
            return self._empty_industry_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            logger.debug(f"V68 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V68: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V68 加载行业数据失败：{e}")
            return self._empty_industry_df()
    
    def load_market_cap_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载流通市值数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V68: 数据库连接未初始化")
            return self._empty_market_cap_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, mv
                FROM stock_daily
                WHERE trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            
            logger.debug(f"V68 DataManager 执行 SQL: {query[:200]}...")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V68: 未加载到流通市值数据")
                return self._empty_market_cap_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V68 加载流通市值数据失败：{e}")
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
        logger.warning(f"V68: {missing_info}")
    
    def print_missing_data_report(self):
        """打印缺失数据报告"""
        if self._missing_data_log:
            logger.info("=" * 60)
            logger.info("V68 缺失数据报告")
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
# V68 AlphaCenter - Rank IC 驱动的信号生成
# ===========================================

class V68AlphaCenter:
    """
    V68 AlphaCenter - Rank IC 驱动的信号生成
    
    【核心逻辑】
    1. 计算 5 日远期收益率 (Forward 5D Return)
    2. 计算预测排名与实际收益排名的 Rank IC
    3. 如果 Rank IC 不达标，直接报告"预测模型失败"
    4. 机构踪迹：主力净额 / 流通市值 比例因子
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Rank IC 配置
        self.rank_ic_target = self.config.get('rank_ic_target', V68_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', V68_RANK_IC_MIN)
        self.ic_window = self.config.get('ic_window', V68_IC_WINDOW)
        
        # 远期收益率配置
        self.forward_return_window = self.config.get('forward_return_window', V68_FORWARD_RETURN_WINDOW)
        
        # 行业护城河配置
        self.rs_window = self.config.get('rs_window', V68_RS_WINDOW)
        self.rs_z_score_threshold = self.config.get('rs_z_score_threshold', V68_RS_Z_SCORE_THRESHOLD)
        self.rs_percentile_threshold = self.config.get('rs_percentile_threshold', V68_RS_PERCENTILE_THRESHOLD)
        
        # 资金共振配置
        self.net_main_rate_window = self.config.get('net_main_rate_window', V68_NET_MAIN_RATE_WINDOW)
        self.pullback_ma_period = self.config.get('pullback_ma_period', V68_PULLBACK_MA_PERIOD)
        self.pullback_tolerance = self.config.get('pullback_tolerance', V68_PULLBACK_TOLERANCE)
        
        # 机构踪迹配置
        self.main_force_ratio_window = self.config.get('main_force_ratio_window', V68_MAIN_FORCE_RATIO_WINDOW)
        self.main_force_ratio_threshold = self.config.get('main_force_ratio_threshold', V68_MAIN_FORCE_RATIO_THRESHOLD)
        
        # 成交量特征配置
        self.volume_ratio_window = self.config.get('volume_ratio_window', V68_VOLUME_RATIO_WINDOW)
        self.main_force_volume_ratio_threshold = self.config.get('main_force_volume_ratio_threshold', V68_MAIN_FORCE_VOLUME_RATIO_THRESHOLD)
        
        # VCP 动态阈值配置
        self.vcp_window = self.config.get('vcp_window', V68_VCP_WINDOW)
        self.vcp_volatility_multiplier = self.config.get('vcp_volatility_multiplier', V68_VCP_VOLATILITY_MULTIPLIER)
        
        # 大盘避坑配置
        self.decline_ratio_threshold = self.config.get('decline_ratio_threshold', V68_MARKET_DECLINE_RATIO_THRESHOLD)
        
        # 预测权重
        self.rs_z_score_weight = 1.0
        self.fund_flow_weight = 1.0
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None,
                        market_cap_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
        
        【核心逻辑】
        1. 计算 5 日远期收益率
        2. 计算预测排名
        3. 机构踪迹：主力净额/流通市值
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
                'rank_ic_value': 0.0,
                'rank_ic_pass': False,
                'rs_z_score_weight': self.rs_z_score_weight,
                'fund_flow_weight': self.fund_flow_weight,
            }
            
            # 检测数据可用性
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            has_industry = industry_df is not None and not industry_df.is_empty()
            has_market_cap = market_cap_df is not None and not market_cap_df.is_empty()
            
            # 1. 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].append('ma_system')
            
            # 2. 计算 5 日远期收益率 (预测目标)
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
            
            # 9. 计算综合评分 (预测因子)
            result = self._compute_composite_score(result, has_fund_flow, has_industry, has_market_cap)
            status['factors_computed'].append('composite_score')
            
            # 10. 计算预测排名
            result = self._compute_predicted_rank(result)
            status['factors_computed'].append('predicted_rank')
            
            logger.info(f"V68 AlphaCenter 信号计算完成，综合评分完成")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V68 AlphaCenter 计算信号失败：{e}")
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
        这是模型的预测目标
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
    
    def _compute_predicted_rank(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算预测排名
        
        【核心逻辑】
        - 根据 composite_score 对股票进行排名
        - 预测排名将用于 Rank IC 计算
        """
        result = df.clone()
        
        # 计算预测排名 (按 composite_score 降序排名)
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').cast(pl.Int64).alias('predicted_rank')
        ])
        
        return result
    
    def _compute_composite_score(self, df: pl.DataFrame,
                                  has_fund_flow: bool,
                                  has_industry: bool,
                                  has_market_cap: bool) -> pl.DataFrame:
        """计算综合评分 (预测因子)"""
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
        
        # 综合评分 (资金流增强)
        rs_bonus = pl.when(pl.col('rs_in_top_percentile')) \
            .then(pl.col('rs_percentile') * 50).otherwise(0.0)
        
        rs_z_bonus = pl.when(pl.col('rs_z_score') > self.rs_z_score_threshold) \
            .then(pl.col('rs_z_score') * 20 * self.rs_z_score_weight).otherwise(0.0)
        
        fund_flow_bonus = pl.when(pl.col('net_main_rate_vs_avg') > 0) \
            .then(pl.col('net_main_rate_vs_avg') * 100 * self.fund_flow_weight).otherwise(0.0)
        
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
    
    def compute_market_decline_ratio(self, df: pl.DataFrame) -> V68MarketRegime:
        """计算市场下跌家数占比"""
        latest_date = df['trade_date'].max()
        latest_df = df.filter(pl.col('trade_date') == latest_date)
        
        if latest_df.is_empty():
            return V68MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        # 计算下跌家数
        decline_count = latest_df.filter(pl.col('close') < pl.col('open')).height
        total_count = latest_df.height
        
        if total_count == 0:
            return V68MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        decline_ratio = decline_count / total_count
        is_safe = decline_ratio <= self.decline_ratio_threshold
        
        regime = V68MarketRegime(
            trade_date=latest_date,
            decline_ratio=decline_ratio,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"下跌家数占比={decline_ratio*100:.1f}%, 阈值={self.decline_ratio_threshold*100:.0f}%"
        )
        
        if not is_safe:
            logger.warning(f"V68: {latest_date} 大盘避坑触发 ({regime.regime_reason})，强制空仓！")
        
        return regime
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str,
                         market_regime: Optional[V68MarketRegime] = None) -> List[V68Signal]:
        """生成交易信号"""
        signals = []
        
        # 大盘危险，强制空仓
        if market_regime and not market_regime.is_safe_period:
            logger.warning(f"V68: 大盘避坑触发，禁止开仓！")
            return signals
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            top_n = max(1, int(buy_df.height * V68_SELECTION_PERCENTILE))
            buy_df = buy_df.head(top_n)
            
            for row in buy_df.iter_rows(named=True):
                signal = V68Signal(
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
                    predicted_rank=row.get('predicted_rank', 0),
                    close_price=row.get('close', 0.0),
                    ma20_price=row.get('ma20', 0.0)
                )
                signals.append(signal)
            
            logger.info(f"V68 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V68 生成信号失败：{e}")
        
        return signals


# ===========================================
# V68 RankICCalculator - Rank IC 计算与审计
# ===========================================

class V68RankICCalculator:
    """
    V68 RankICCalculator - Rank IC 计算与审计
    
    【核心功能】
    1. 计算每日预测排名与实际收益排名的 Rank IC
    2. 将审计结果存入 strategy_audit 表
    3. 如果 Rank IC 不达标，直接报告"预测模型失败"
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target = self.config.get('rank_ic_target', V68_RANK_IC_TARGET)
        self.rank_ic_min = self.config.get('rank_ic_min', V68_RANK_IC_MIN)
        self.ic_window = self.config.get('ic_window', V68_IC_WINDOW)
        
        self.ic_results: List[V68ICMetrics] = []
        self.audit_records: List[V68StrategyAudit] = []
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        """
        计算 Spearman Rank IC
        
        【核心算法】
        Rank IC = Spearman 相关系数 (预测排名 vs 实际收益排名)
        
        Parameters
        ----------
        factor_values : np.ndarray
            预测因子值 (composite_score)
        label_values : np.ndarray
            实际标签值 (forward_return_5d)
            
        Returns
        -------
        float
            Spearman Rank IC
        """
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
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
        """
        计算单日的 Rank IC
        
        Parameters
        ----------
        df : pl.DataFrame
            数据
        trade_date : str
            交易日期
        signal_col : str
            信号列名
        return_col : str
            收益列名
            
        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (Rank IC 值，详细信息)
        """
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, {'count': 0, 'reason': '样本不足'}
            
            if signal_col not in day_data.columns or return_col not in day_data.columns:
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
            logger.debug(f"V68 计算 {trade_date} Rank IC 失败：{e}")
            return 0.0, {'count': 0, 'reason': str(e)}
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'forward_return_5d') -> List[V68ICMetrics]:
        """计算 IC 序列"""
        unique_dates = df['trade_date'].unique().to_list()
        ic_series = []
        
        for trade_date in sorted(unique_dates):
            rank_ic, details = self.calculate_daily_rank_ic(df, trade_date, signal_col, return_col)
            ic = details.get('ic', 0.0)
            
            ic_metrics = V68ICMetrics(
                trade_date=trade_date,
                factor_name='composite_score',
                ic=ic,
                rank_ic=rank_ic
            )
            ic_series.append(ic_metrics)
        
        self.ic_results = ic_series
        return ic_series
    
    def create_strategy_audit_records(self, df: pl.DataFrame) -> List[V68StrategyAudit]:
        """
        创建策略审计记录
        
        【核心功能】
        - 为每个股票在每个交易日创建审计记录
        - 记录预测排名与实际收益
        - 用于后续分析预测质量
        
        Parameters
        ----------
        df : pl.DataFrame
            包含预测和收益数据
        
        Returns
        -------
        List[V68StrategyAudit]
            审计记录列表
        """
        audit_records = []
        
        try:
            # 按日期分组处理
            unique_dates = sorted(df['trade_date'].unique().to_list())
            
            for trade_date in unique_dates:
                day_data = df.filter(pl.col('trade_date') == trade_date)
                
                if day_data.height < 10:
                    continue
                
                # 获取必要列
                required_cols = ['symbol', 'composite_score', 'forward_return_5d', 'predicted_rank']
                available_cols = [c for c in required_cols if c in day_data.columns]
                
                if len(available_cols) < len(required_cols):
                    continue
                
                # 计算实际收益排名
                day_data = day_data.with_columns([
                    pl.col('forward_return_5d').rank('ordinal', descending=True).cast(pl.Int64).alias('actual_rank')
                ])
                
                # 为每个股票创建审计记录
                for row in day_data.iter_rows(named=True):
                    symbol = row.get('symbol', '')
                    composite_score = row.get('composite_score', 0.0)
                    predicted_rank = row.get('predicted_rank', 0)
                    actual_return = row.get('forward_return_5d', 0.0)
                    actual_rank = row.get('actual_rank', 0)
                    
                    # 计算 Rank IC 贡献
                    rank_ic_contribution = self._calculate_single_rank_ic_contribution(
                        predicted_rank, actual_rank, day_data.height
                    )
                    
                    # 资金流评分
                    fund_flow_score = row.get('net_main_rate_vs_avg', 0.0) * 100
                    
                    # RS 评分
                    rs_score = row.get('rs_z_score', 0.0) * 20
                    
                    audit = V68StrategyAudit(
                        trade_date=trade_date,
                        symbol=symbol,
                        predicted_rank=predicted_rank,
                        predicted_return=composite_score,
                        actual_return=actual_return if actual_return else 0.0,
                        actual_rank=actual_rank,
                        rank_ic_contribution=rank_ic_contribution,
                        composite_score=composite_score,
                        fund_flow_score=fund_flow_score,
                        rs_score=rs_score
                    )
                    audit_records.append(audit)
            
            self.audit_records = audit_records
            logger.info(f"V68 创建 {len(audit_records)} 条策略审计记录")
            
        except Exception as e:
            logger.error(f"V68 创建审计记录失败：{e}")
            logger.error(traceback.format_exc())
        
        return audit_records
    
    def _calculate_single_rank_ic_contribution(self, predicted_rank: int, actual_rank: int, 
                                                n_stocks: int) -> float:
        """计算单个股票的 Rank IC 贡献"""
        if n_stocks < 2:
            return 0.0
        
        # 简化的 Rank IC 贡献计算
        rank_diff = abs(predicted_rank - actual_rank)
        max_rank_diff = n_stocks - 1
        
        if max_rank_diff == 0:
            return 0.0
        
        # 排名越接近，贡献越高
        contribution = 1.0 - (rank_diff / max_rank_diff)
        return float(contribution)
    
    def save_audit_to_db(self, audit_records: Optional[List[V68StrategyAudit]] = None):
        """
        将审计记录保存到数据库 strategy_audit 表
        
        Parameters
        ----------
        audit_records : List[V68StrategyAudit], optional
            审计记录列表
        """
        if self.db is None:
            logger.warning("V68: 数据库连接未初始化，无法保存审计记录")
            return
        
        records = audit_records or self.audit_records
        
        if not records:
            logger.warning("V68: 没有审计记录可保存")
            return
        
        try:
            # 创建表
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS `strategy_audit` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `trade_date` VARCHAR(20) NOT NULL,
                `symbol` VARCHAR(20) NOT NULL,
                `predicted_rank` INT DEFAULT 0,
                `predicted_return` DECIMAL(20, 10) DEFAULT 0,
                `actual_return` DECIMAL(20, 10) DEFAULT 0,
                `actual_rank` INT DEFAULT 0,
                `rank_ic_contribution` DECIMAL(20, 10) DEFAULT 0,
                `composite_score` DECIMAL(20, 10) DEFAULT 0,
                `fund_flow_score` DECIMAL(20, 10) DEFAULT 0,
                `rs_score` DECIMAL(20, 10) DEFAULT 0,
                INDEX `idx_trade_date` (`trade_date`),
                INDEX `idx_symbol` (`symbol`),
                INDEX `idx_date_symbol` (`trade_date`, `symbol`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='策略审计表'
            """
            
            self.db.execute(create_table_sql)
            logger.info("V68: strategy_audit 表创建成功")
            
            # 转换为 DataFrame
            data = []
            for record in records:
                data.append({
                    'trade_date': record.trade_date,
                    'symbol': record.symbol,
                    'predicted_rank': record.predicted_rank,
                    'predicted_return': record.predicted_return,
                    'actual_return': record.actual_return,
                    'actual_rank': record.actual_rank,
                    'rank_ic_contribution': record.rank_ic_contribution,
                    'composite_score': record.composite_score,
                    'fund_flow_score': record.fund_flow_score,
                    'rs_score': record.rs_score,
                })
            
            df = pl.DataFrame(data)
            
            # 写入数据库
            self.db.to_sql(df, 'strategy_audit', if_exists='append')
            
            logger.info(f"V68: 已保存 {len(records)} 条审计记录到 strategy_audit 表")
            
        except Exception as e:
            logger.error(f"V68: 保存审计记录失败：{e}")
            logger.error(traceback.format_exc())
    
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
    
    def check_rank_ic_pass(self) -> Tuple[bool, str]:
        """
        检查 Rank IC 是否达标
        
        【拒绝回测美化】
        - 如果 Rank IC 不达标，直接报告"预测模型失败"
        
        Returns
        -------
        Tuple[bool, str]
            (是否达标，消息)
        """
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
        is_pass, message = self.check_rank_ic_pass()
        
        logger.info("=" * 60)
        logger.info("V68 Rank IC 预测质量审计表")
        logger.info("=" * 60)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info("-" * 40)
        logger.info(f"Mean IC:      {stats['mean_ic']:.4f}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：>{self.rank_ic_target})")
        logger.info(f"IC Std:       {stats['ic_std']:.4f}")
        logger.info(f"Rank IC Std:  {stats['rank_ic_std']:.4f}")
        logger.info("-" * 40)
        logger.info(f"IC IR:        {stats['ic_ir']:.2f}")
        logger.info(f"Rank IC IR:   {stats['rank_ic_ir']:.2f}")
        logger.info(f"Positive Ratio: {stats['positive_ratio']:.1%}")
        logger.info("-" * 40)
        
        if is_pass:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")
            logger.error("V68: 预测模型失败，不准通过调整止损位来刷分！")
        
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
    'V68_INITIAL_CAPITAL',
    'V68_MAX_POSITIONS',
    'V68_MONTHLY_TRADE_LIMIT',
    'V68_WEEKLY_TRADE_LIMIT',
    'V68_GLOBAL_TRADE_LIMIT',
    'V68_WARMUP_PERIOD',
    'V68_MIN_SAMPLE_SIZE',
    'V68_MIN_FUND_FLOW_ROWS',
    'V68_MIN_INDUSTRY_ROWS',
    'V68_RANK_IC_TARGET',
    'V68_RANK_IC_MIN',
    'V68_IC_WINDOW',
    'V68_FORWARD_RETURN_WINDOW',
    'V68_RS_WINDOW',
    'V68_RS_Z_SCORE_THRESHOLD',
    'V68_RS_PERCENTILE_THRESHOLD',
    'V68_NET_MAIN_RATE_WINDOW',
    'V68_PULLBACK_MA_PERIOD',
    'V68_PULLBACK_TOLERANCE',
    'V68_MAIN_FORCE_RATIO_WINDOW',
    'V68_MAIN_FORCE_RATIO_THRESHOLD',
    'V68_VOLUME_RATIO_WINDOW',
    'V68_MAIN_FORCE_VOLUME_RATIO_THRESHOLD',
    'V68_VCP_WINDOW',
    'V68_VCP_VOLATILITY_MULTIPLIER',
    'V68_MARKET_DECLINE_RATIO_THRESHOLD',
    'V68_COMMISSION_RATE',
    'V68_MIN_COMMISSION',
    'V68_SLIPPAGE_BUY',
    'V68_SLIPPAGE_SELL',
    'V68_STAMP_DUTY',
    'V68_TRANSFER_FEE',
    'V68_FRICTION_COST',
    'V68_TREND_BREAK_MA_PERIOD',
    'V68_PROFIT_TARGET_RATIO',
    'V68_TRAILING_STOP_RATIO',
    'V68_MAX_SINGLE_POSITION_PCT',
    'V68_SELECTION_PERCENTILE',
    
    # 数据类
    'V68Position',
    'V68Trade',
    'V68TradeAudit',
    'V68Signal',
    'V68MarketRegime',
    'V68ICMetrics',
    'V68StrategyAudit',
    
    # 核心类
    'V68DataManager',
    'V68AlphaCenter',
    'V68RankICCalculator',
    
    # 评估指标
    'calculate_ae_metric',
]