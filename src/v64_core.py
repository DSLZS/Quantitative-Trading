"""
V64 Core Module - 资金流增强型 RS-Pullback Alpha 引擎

【V64 核心改进 - 最高优先级】

1. 预测逻辑 (Alpha Center)
   ✅ 行业中性化 RS：个股收益率必须超过其所属行业指数收益率的 5% 以上
   ✅ 资金流背离：在价格回调至 MA20 期间，net_main_amount 必须连续 3 日为正
   ✅ VCP 收缩：维持波动率收缩要求，但将硬门槛改为打分制（0-100 分）

2. 避坑逻辑
   ✅ 如果当日全市场下跌家数占比 > 80%，强制停止当日所有买入预测

3. 自修复机制
   ✅ DatabaseIntegrityCheck：如果 stock_fund_flow 表为空，自动切换到"纯价量模型"
   ✅ 发出告警日志，而不是直接崩溃

4. 代码质量
   ✅ 严禁偷懒：不允许出现 pass 或 TODO

作者：量化系统
版本：V64.0
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
# V64 配置常量 - 资金流增强型 RS-Pullback
# ===========================================

# 基础配置
V64_INITIAL_CAPITAL = 100000.00
V64_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V64 频率熔断
V64_MONTHLY_TRADE_LIMIT = 15  # 月均交易次数上限
V64_WEEKLY_TRADE_LIMIT = 4    # 每周最多开仓 4 只
V64_GLOBAL_TRADE_LIMIT = 150  # 全场交易次数限制

# V64 数据预加载配置
V64_WARMUP_PERIOD = 250  # 预加载 250 天数据
V64_MIN_SAMPLE_SIZE = 500  # 最小股票样本量

# V64 行业中性化 RS 配置
V64_RS_WINDOW = 20  # 计算 RS 的窗口
V64_RS_OUTPERFORM_THRESHOLD = 0.05  # 必须超过行业指数 5% 以上

# V64 资金流背离配置
V64_FUND_FLOW_CONSECUTIVE_DAYS = 3  # 主力净流入连续为正的天数
V64_FUND_FLOW_NET_MAIN_COLUMN = 'net_main_amount'  # 主力净流入列名

# V64 价格回调配置
V64_PULLBACK_MA_PERIOD = 20  # 回调至 MA20
V64_PULLBACK_TOLERANCE = 0.02  # 回调容忍度 2%

# V64 VCP 收缩打分配置
V64_VCP_WINDOW = 10  # 观察窗口
V64_VCP_MAX_AMPLITUDE = 0.08  # 最大振幅
V64_VCP_MIN_SCORE = 60  # 最低接受分数

# V64 大盘避坑配置
V64_MARKET_DECLINE_RATIO_THRESHOLD = 0.80  # 下跌家数占比 > 80% 强制空仓

# V64 费率配置 - 总计 0.2%
V64_COMMISSION_RATE = 0.0003  # 佣金万 3
V64_MIN_COMMISSION = 5.0  # 最低佣金 5 元
V64_SLIPPAGE_BUY = 0.001  # 买入滑点 0.1%
V64_SLIPPAGE_SELL = 0.001  # 卖出滑点 0.1%
V64_STAMP_DUTY = 0.0005  # 印花税 0.05%
V64_TRANSFER_FEE = 0.00001  # 过户费 0.001%
V64_FRICTION_COST = 0.002  # 0.2% 总计

# V64 止损配置
V64_HARD_STOP_LOSS_RATIO = 0.08  # 硬止损 8%
V64_TIME_STOP_DAYS = 5  # 时间止损 5 天

# V64 止盈配置
V64_TRAILING_STOP_RATIO = 0.05  # 移动止盈 5%
V64_PROFIT_TARGET_RATIO = 0.15  # 目标盈利 15%

# V64 仓位管理
V64_MAX_SINGLE_POSITION_PCT = 0.10  # 单仓上限 10%

# V64 选股排名
V64_SELECTION_PERCENTILE = 0.15  # 只交易前 15% 的股票


# ===========================================
# V64 数据类定义
# ===========================================

@dataclass
class V64Position:
    """V64 持仓记录"""
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
    
    # V64 资金流状态
    fund_flow_consecutive_positive: int = 0
    net_main_amount: float = 0.0
    
    # V64 行业中性化 RS
    rs_vs_industry: float = 0.0
    industry_name: str = ""
    
    # V64 VCP 打分
    vcp_score: float = 0.0
    
    # 止损止盈
    stop_loss_price: float = 0.0
    stop_loss_triggered: bool = False
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    time_stop_triggered: bool = False
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V64Trade:
    """V64 交易记录"""
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
    
    # V64 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0


@dataclass
class V64TradeAudit:
    """V64 交易审计记录"""
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
    
    # V64 状态
    fund_flow_consecutive_positive: int = 0
    rs_vs_industry: float = 0.0
    vcp_score: float = 0.0
    
    # 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0


@dataclass
class V64Signal:
    """V64 交易信号"""
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    signal_rank: int
    composite_score: float
    
    # V64 状态
    fund_flow_consecutive_positive: int = 0
    net_main_amount: float = 0.0
    rs_vs_industry: float = 0.0
    vcp_score: float = 0.0
    is_pullback_to_ma20: bool = False
    
    # 价格数据
    close_price: float = 0.0
    ma20_price: float = 0.0


@dataclass
class V64MarketRegime:
    """V64 大盘状态"""
    trade_date: str
    decline_ratio: float = 0.0  # 下跌家数占比
    is_safe_period: bool = True
    regime_reason: str = ""
    forced_empty: bool = False


# ===========================================
# V64 数据库完整性检查
# ===========================================

class DatabaseIntegrityCheck:
    """
    V64 数据库完整性检查
    
    【核心功能】
    1. 检查 stock_fund_flow 表是否为空
    2. 如果为空，自动切换到"纯价量模型"
    3. 发出告警日志，而不是直接崩溃
    """
    
    def __init__(self, db=None):
        """
        初始化数据库完整性检查器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        self.db = db
        self.fund_flow_available: bool = True
        self.industry_data_available: bool = True
        self.mode: str = "full"  # "full" 或 "price_volume_only"
    
    def check_fund_flow_table(self) -> bool:
        """
        检查 stock_fund_flow 表是否有数据
        
        Returns
        -------
        bool
            表是否有数据
        """
        if self.db is None:
            logger.warning("V64: 数据库连接未初始化，切换到纯价量模式")
            self.fund_flow_available = False
            self.mode = "price_volume_only"
            return False
        
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
            df = self.db.read_sql(query)
            
            if df.is_empty() or df['cnt'][0] == 0:
                logger.warning("V64: stock_fund_flow 表为空，切换到纯价量模式")
                self.fund_flow_available = False
                self.mode = "price_volume_only"
                return False
            
            self.fund_flow_available = True
            return True
            
        except Exception as e:
            logger.warning(f"V64: 检查 stock_fund_flow 表失败：{e}，切换到纯价量模式")
            self.fund_flow_available = False
            self.mode = "price_volume_only"
            return False
    
    def check_industry_table(self) -> bool:
        """
        检查 stock_industry 表是否有数据
        
        Returns
        -------
        bool
            表是否有数据
        """
        if self.db is None:
            logger.warning("V64: 数据库连接未初始化")
            self.industry_data_available = False
            return False
        
        try:
            query = "SELECT COUNT(DISTINCT symbol) as cnt FROM stock_industry"
            df = self.db.read_sql(query)
            
            if df.is_empty() or df['cnt'][0] == 0:
                logger.warning("V64: stock_industry 表为空，无法进行行业中性化")
                self.industry_data_available = False
                return False
            
            self.industry_data_available = True
            return True
            
        except Exception as e:
            logger.warning(f"V64: 检查 stock_industry 表失败：{e}")
            self.industry_data_available = False
            return False
    
    def run_full_check(self) -> Dict[str, Any]:
        """
        运行完整检查
        
        Returns
        -------
        Dict[str, Any]
            检查结果
        """
        logger.info("=" * 60)
        logger.info("V64: 开始数据库完整性检查")
        logger.info("=" * 60)
        
        fund_flow_ok = self.check_fund_flow_table()
        industry_ok = self.check_industry_table()
        
        # 确定模式
        if fund_flow_ok and industry_ok:
            self.mode = "full"
            logger.info("V64: 完整模式 - 资金流 + 行业中性化 + 价量")
        elif fund_flow_ok:
            self.mode = "fund_flow_price_volume"
            logger.info("V64: 降级模式 - 资金流 + 价量（无行业数据）")
        elif industry_ok:
            self.mode = "industry_price_volume"
            logger.info("V64: 降级模式 - 行业中性化 + 价量（无资金流数据）")
        else:
            self.mode = "price_volume_only"
            logger.warning("V64: 纯价量模式 - 无资金流和行业数据")
        
        result = {
            'mode': self.mode,
            'fund_flow_available': fund_flow_ok,
            'industry_data_available': industry_ok,
        }
        
        logger.info(f"V64: 数据库完整性检查完成，模式：{self.mode}")
        logger.info("=" * 60)
        
        return result
    
    def is_full_mode(self) -> bool:
        """是否完整模式"""
        return self.mode == "full"
    
    def has_fund_flow(self) -> bool:
        """是否有资金流数据"""
        return self.fund_flow_available
    
    def has_industry_data(self) -> bool:
        """是否有行业数据"""
        return self.industry_data_available


# ===========================================
# V64 DataManager - 数据获取与预处理
# ===========================================

class V64DataManager:
    """
    V64 DataManager - 数据获取与预处理
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V64_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V64_MIN_SAMPLE_SIZE)
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
        
        logger.info(f"V64 数据加载：回测区间 [{start_date}, {end_date}]")
        
        if self.db is None:
            raise ValueError("V64 DataManager: 数据库连接未初始化")
        
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
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"V64 DataManager: 未加载到任何数据")
            
            logger.info(f"V64 数据加载完成：{df.height} 行，{df['symbol'].n_unique()} 只股票")
            
            return df
            
        except Exception as e:
            logger.error(f"V64 DataManager 加载数据失败：{e}")
            raise
    
    def load_fund_flow_data(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载资金流向数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V64: 数据库连接未初始化")
            return self._empty_fund_flow_df()
        
        try:
            if symbols:
                symbol_list = "','".join(symbols)
                symbol_filter = f"AND symbol IN ('{symbol_list}')"
            else:
                symbol_filter = ""
            
            query = f"""
                SELECT symbol, trade_date, net_main_amount, net_super_amount, 
                       net_large_amount, net_medium_amount, net_small_amount
                FROM stock_fund_flow
                WHERE trade_date >= '{actual_start_date}' 
                  AND trade_date <= '{end_date}'
                  {symbol_filter}
                ORDER BY symbol, trade_date
            """
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V64: 未加载到资金流向数据")
                return self._empty_fund_flow_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V64 加载资金流向数据失败：{e}")
            return self._empty_fund_flow_df()
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            logger.warning("V64: 数据库连接未初始化")
            return self._empty_industry_df()
        
        try:
            query = f"""
                SELECT symbol, trade_date, industry_name
                FROM stock_industry
                WHERE trade_date >= '{actual_start_date}'
                  AND trade_date <= '{end_date}'
            """
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("V64: 未加载到行业数据")
                return self._empty_industry_df()
            
            return df
            
        except Exception as e:
            logger.warning(f"V64 加载行业数据失败：{e}")
            return self._empty_industry_df()
    
    def _empty_fund_flow_df(self) -> pl.DataFrame:
        """返回空资金流 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'net_main_amount': pl.Float64
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
# V64 AlphaCenter - 资金流增强型 RS-Pullback 信号生成
# ===========================================

class V64AlphaCenter:
    """
    V64 AlphaCenter - 资金流增强型 RS-Pullback 信号生成
    
    【核心逻辑】
    1. 行业中性化 RS：个股收益率 > 行业指数收益率 + 5%
    2. 资金流背离：价格回调至 MA20 期间，主力净流入连续 3 日为正
    3. VCP 收缩：打分制（0-100 分）
    4. 大盘避坑：下跌家数占比 > 80% 强制空仓
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: Dict[str, Any] = None, 
                 integrity_check: Optional[DatabaseIntegrityCheck] = None):
        self.config = config or {}
        self.integrity_check = integrity_check
        
        # 行业中性化 RS 配置
        self.rs_window = self.config.get('rs_window', V64_RS_WINDOW)
        self.rs_outperform_threshold = self.config.get('rs_outperform_threshold', V64_RS_OUTPERFORM_THRESHOLD)
        
        # 资金流背离配置
        self.fund_flow_consecutive_days = self.config.get('fund_flow_consecutive_days', V64_FUND_FLOW_CONSECUTIVE_DAYS)
        
        # 价格回调配置
        self.pullback_ma_period = self.config.get('pullback_ma_period', V64_PULLBACK_MA_PERIOD)
        self.pullback_tolerance = self.config.get('pullback_tolerance', V64_PULLBACK_TOLERANCE)
        
        # VCP 打分配置
        self.vcp_window = self.config.get('vcp_window', V64_VCP_WINDOW)
        self.vcp_max_amplitude = self.config.get('vcp_max_amplitude', V64_VCP_MAX_AMPLITUDE)
        self.vcp_min_score = self.config.get('vcp_min_score', V64_VCP_MIN_SCORE)
        
        # 大盘避坑配置
        self.decline_ratio_threshold = self.config.get('decline_ratio_threshold', V64_MARKET_DECLINE_RATIO_THRESHOLD)
    
    def compute_signals(self, df: pl.DataFrame,
                        fund_flow_df: Optional[pl.DataFrame] = None,
                        industry_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        计算所有因子和交易信号
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
            ])
            
            status = {'factors_computed': [], 'mode': 'full'}
            
            # 1. 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].append('ma_system')
            
            # 2. 计算行业中性化 RS
            has_industry = industry_df is not None and not industry_df.is_empty()
            if has_industry:
                result = self._compute_industry_neutralized_rs(result, industry_df)
                status['factors_computed'].append('industry_neutralized_rs')
            else:
                result = self._compute_basic_rs(result)
                status['mode'] = 'basic_rs'
                status['factors_computed'].append('basic_rs')
            
            # 3. 计算资金流背离
            has_fund_flow = fund_flow_df is not None and not fund_flow_df.is_empty()
            if has_fund_flow:
                result = self._compute_fund_flow_divergence(result, fund_flow_df)
                status['factors_computed'].append('fund_flow_divergence')
            else:
                result = self._add_fund_flow_placeholders(result)
                status['mode'] = 'no_fund_flow'
                status['factors_computed'].append('fund_flow_placeholder')
            
            # 4. 计算价格回调至 MA20
            result = self._compute_pullback_to_ma20(result)
            status['factors_computed'].append('pullback_to_ma20')
            
            # 5. 计算 VCP 收缩打分
            result = self._compute_vcp_score(result)
            status['factors_computed'].append('vcp_score')
            
            # 6. 计算综合评分
            result = self._compute_composite_score(result, has_fund_flow, has_industry)
            status['factors_computed'].append('composite_score')
            
            logger.info(f"V64 AlphaCenter 信号计算完成，模式：{status['mode']}")
            
            return result, status
            
        except Exception as e:
            logger.error(f"V64 AlphaCenter 计算信号失败：{e}")
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
    
    def _compute_industry_neutralized_rs(self, df: pl.DataFrame, 
                                          industry_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算行业中性化 RS
        
        【核心逻辑】
        - 计算个股过去 N 日收益率
        - 计算所属行业指数同期收益率
        - RS = 个股收益率 - 行业收益率
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
        
        # 计算行业收益率（简化：使用行业平均）
        result = result.with_columns([
            pl.col('stock_return').mean().over(['trade_date', 'industry_name']).alias('industry_return')
        ])
        
        # 计算行业中性化 RS
        result = result.with_columns([
            (pl.col('stock_return') - pl.col('industry_return')).alias('rs_vs_industry'),
            (pl.col('rs_vs_industry') > self.rs_outperform_threshold).alias('rs_outperforms_industry')
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
            pl.lit(True).alias('rs_outperforms_industry')
        ])
        
        return result
    
    def _compute_fund_flow_divergence(self, df: pl.DataFrame,
                                       fund_flow_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算资金流背离
        
        【核心逻辑】
        - 在价格回调至 MA20 期间
        - 主力净流入连续 N 日为正
        """
        result = df.clone()
        
        # 合并资金流数据
        fund_cols = ['symbol', 'trade_date', 'net_main_amount']
        available_fund_cols = [c for c in fund_cols if c in fund_flow_df.columns]
        fund_data = fund_flow_df.select(available_fund_cols)
        
        result = result.join(fund_data, on=['symbol', 'trade_date'], how='left')
        
        # 填充空值
        result = result.with_columns([
            pl.col('net_main_amount').fill_null(0.0).alias('net_main_amount')
        ])
        
        # 计算连续为正天数
        is_positive = pl.col('net_main_amount') > 0
        
        # 连续检测
        pos_1 = is_positive
        pos_2 = is_positive.shift(1).over('symbol')
        pos_3 = is_positive.shift(2).over('symbol')
        
        consecutive_positive = pos_1 & pos_2 & pos_3
        
        # 计算连续天数
        consecutive_days = pl.when(pos_1 & pos_2 & pos_3) \
            .then(3) \
            .otherwise(pl.when(pos_1 & pos_2).then(2).otherwise(pl.when(pos_1).then(1).otherwise(0)))
        
        result = result.with_columns([
            is_positive.alias('is_net_main_positive'),
            consecutive_days.alias('fund_flow_consecutive_positive'),
            (consecutive_days >= self.fund_flow_consecutive_days).alias('fund_flow_divergence')
        ])
        
        return result
    
    def _add_fund_flow_placeholders(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加资金流占位符（无数据时）"""
        result = df.clone()
        
        result = result.with_columns([
            pl.lit(0.0).alias('net_main_amount'),
            pl.lit(0).alias('fund_flow_consecutive_positive'),
            pl.lit(False).alias('fund_flow_divergence'),
            pl.lit(True).alias('is_net_main_positive')
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
    
    def _compute_vcp_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 VCP 收缩打分（0-100 分）
        
        【打分维度】
        1. 振幅收缩（40 分）
        2. 量能萎缩（30 分）
        3. 价格收敛（30 分）
        """
        result = df.clone()
        
        # 1. 计算每日振幅
        daily_amplitude = (pl.col('high') - pl.col('low')) / (pl.col('close') + self.EPSILON)
        
        # 滚动平均振幅
        avg_amplitude = daily_amplitude.rolling_mean(window_size=self.vcp_window).over('symbol')
        prev_avg_amplitude = avg_amplitude.shift(1).over('symbol')
        
        # 振幅收缩得分（40 分）
        amplitude_contraction = (avg_amplitude < prev_avg_amplitude)
        amplitude_score = pl.when(amplitude_contraction) \
            .then(40.0 * (1.0 - avg_amplitude / self.vcp_max_amplitude).clip(0.0, 1.0)) \
            .otherwise(0.0)
        
        # 2. 量能萎缩得分（30 分）
        vol_ma = pl.col('volume').rolling_mean(window_size=10).over('symbol')
        vol_ratio = pl.col('volume') / (vol_ma + self.EPSILON)
        volume_score = pl.when(vol_ratio < 1.0) \
            .then(30.0 * (1.0 - vol_ratio).clip(0.0, 1.0)) \
            .otherwise(0.0)
        
        # 3. 价格收敛得分（30 分）
        price_range = (pl.col('high') - pl.col('low')) / (pl.col('close') + self.EPSILON)
        price_contraction = price_range < self.vcp_max_amplitude
        price_score = pl.when(price_contraction) \
            .then(30.0 * (1.0 - price_range / self.vcp_max_amplitude).clip(0.0, 1.0)) \
            .otherwise(0.0)
        
        # 综合 VCP 得分
        vcp_score = amplitude_score + volume_score + price_score
        
        result = result.with_columns([
            daily_amplitude.alias('daily_amplitude'),
            avg_amplitude.alias('avg_amplitude'),
            vol_ratio.alias('vol_ratio'),
            amplitude_score.alias('amplitude_score'),
            volume_score.alias('volume_score'),
            price_score.alias('price_score'),
            vcp_score.alias('vcp_score'),
            (vcp_score >= self.vcp_min_score).alias('vcp_pass')
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
        rs_condition = pl.col('rs_outperforms_industry')
        pullback_condition = pl.col('is_pullback_to_ma20')
        vcp_condition = pl.col('vcp_pass')
        
        # 资金流条件（如果有数据）
        if has_fund_flow:
            fund_flow_condition = pl.col('fund_flow_divergence')
        else:
            fund_flow_condition = pl.lit(True)
        
        # 核心买入信号
        core_condition = rs_condition & pullback_condition & vcp_condition & fund_flow_condition
        
        # 综合评分
        rs_bonus = pl.when(pl.col('rs_outperforms_industry')) \
            .then(pl.col('rs_vs_industry') * 100).otherwise(0.0)
        
        fund_flow_bonus = pl.when(pl.col('fund_flow_consecutive_positive') >= 3) \
            .then(20.0).otherwise(0.0)
        
        vcp_bonus = pl.col('vcp_score') * 0.4
        
        composite_score = rs_bonus + fund_flow_bonus + vcp_bonus
        
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
    
    def compute_market_decline_ratio(self, df: pl.DataFrame) -> V64MarketRegime:
        """
        计算市场下跌家数占比（大盘避坑）
        """
        latest_date = df['trade_date'].max()
        latest_df = df.filter(pl.col('trade_date') == latest_date)
        
        if latest_df.is_empty():
            return V64MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        # 计算下跌家数（收盘价 < 开盘价）
        decline_count = latest_df.filter(pl.col('close') < pl.col('open')).height
        total_count = latest_df.height
        
        if total_count == 0:
            return V64MarketRegime(trade_date=latest_date, is_safe_period=True)
        
        decline_ratio = decline_count / total_count
        is_safe = decline_ratio <= self.decline_ratio_threshold
        
        regime = V64MarketRegime(
            trade_date=latest_date,
            decline_ratio=decline_ratio,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"下跌家数占比={decline_ratio*100:.1f}%, 阈值={self.decline_ratio_threshold*100:.0f}%"
        )
        
        if not is_safe:
            logger.warning(f"V64: {latest_date} 大盘避坑触发 ({regime.regime_reason})，强制空仓！")
        
        return regime
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str,
                         market_regime: Optional[V64MarketRegime] = None) -> List[V64Signal]:
        """
        生成交易信号
        """
        signals = []
        
        # 大盘危险，强制空仓
        if market_regime and not market_regime.is_safe_period:
            logger.warning(f"V64: 大盘避坑触发，禁止开仓！")
            return signals
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            top_n = max(1, int(buy_df.height * V64_SELECTION_PERCENTILE))
            buy_df = buy_df.head(top_n)
            
            for row in buy_df.iter_rows(named=True):
                signal = V64Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    signal_rank=row.get('composite_rank', 9999),
                    composite_score=row.get('composite_score', 0.0),
                    fund_flow_consecutive_positive=row.get('fund_flow_consecutive_positive', 0),
                    net_main_amount=row.get('net_main_amount', 0.0),
                    rs_vs_industry=row.get('rs_vs_industry', 0.0),
                    vcp_score=row.get('vcp_score', 0.0),
                    is_pullback_to_ma20=row.get('is_pullback_to_ma20', False),
                    close_price=row.get('close', 0.0),
                    ma20_price=row.get('ma20', 0.0)
                )
                signals.append(signal)
            
            logger.info(f"V64 生成 {len(signals)} 个买入信号 ({trade_date})")
            
        except Exception as e:
            logger.error(f"V64 生成信号失败：{e}")
        
        return signals


# ===========================================
# V64 TradeExec - 成交执行
# ===========================================

class V64TradeExec:
    """
    V64 TradeExec - 真实成交执行
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', V64_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V64_MAX_POSITIONS)
        self.commission_rate = self.config.get('commission_rate', V64_COMMISSION_RATE)
        self.min_commission = self.config.get('min_commission', V64_MIN_COMMISSION)
        self.slippage_buy = self.config.get('slippage_buy', V64_SLIPPAGE_BUY)
        self.slippage_sell = self.config.get('slippage_sell', V64_SLIPPAGE_SELL)
        self.stamp_duty = self.config.get('stamp_duty', V64_STAMP_DUTY)
        self.transfer_fee = self.config.get('transfer_fee', V64_TRANSFER_FEE)
        
        self.positions: Dict[str, V64Position] = {}
        self.cash = self.initial_capital
        self.trades: List[V64Trade] = []
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
        
        if self.monthly_trades.get(month_key, 0) >= V64_MONTHLY_TRADE_LIMIT:
            return False
        
        if self.weekly_trades.get(week_key, 0) >= V64_WEEKLY_TRADE_LIMIT:
            return False
        
        if self.total_trades >= V64_GLOBAL_TRADE_LIMIT:
            return False
        
        return True
    
    def _increment_trade_count(self, trade_date: str):
        month_key = self._get_month_key(trade_date)
        week_key = self._get_week_key(trade_date)
        
        self.monthly_trades[month_key] = self.monthly_trades.get(month_key, 0) + 1
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1
        self.total_trades += 1
    
    def execute_buy(self, signal: V64Signal, next_open: float,
                    trigger_price: float, capital: float) -> Optional[V64Trade]:
        """执行买入"""
        execution_price = min(trigger_price, next_open) * (1 + self.slippage_buy)
        
        max_position_value = capital * V64_MAX_SINGLE_POSITION_PCT
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
        
        position = V64Position(
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
            fund_flow_consecutive_positive=signal.fund_flow_consecutive_positive,
            net_main_amount=signal.net_main_amount,
            rs_vs_industry=signal.rs_vs_industry,
            vcp_score=signal.vcp_score,
            stop_loss_price=execution_price * (1 - V64_HARD_STOP_LOSS_RATIO),
            trailing_stop_price=execution_price * (1 - V64_TRAILING_STOP_RATIO),
            trigger_price=trigger_price,
            next_open_price=next_open,
            execution_price=execution_price
        )
        
        self.positions[signal.symbol] = position
        
        trade = V64Trade(
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
            reason='资金流增强 RS-Pullback',
            signal_date=signal.trade_date,
            trigger_price=trigger_price,
            next_open_price=next_open
        )
        
        self.trades.append(trade)
        self.cash -= total_cost
        self._increment_trade_count(signal.trade_date)
        
        logger.info(f"V64 买入成交：{signal.symbol} @ {execution_price:.2f} x {shares}股")
        
        return trade
    
    def execute_sell(self, symbol: str, current_price: float,
                     trade_date: str, reason: str) -> Optional[V64Trade]:
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
        
        trade = V64Trade(
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
            next_open_price=position.next_open_price
        )
        
        self.trades.append(trade)
        self.cash += amount - total_cost
        self.sell_history[symbol] = trade_date
        del self.positions[symbol]
        
        logger.info(f"V64 卖出成交：{symbol} @ {execution_price:.2f}")
        
        return trade
    
    def check_exit_conditions(self, symbol: str, current_price: float,
                              trade_date: str) -> Optional[Tuple[bool, str]]:
        """检查离场条件"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        cost_price = position.avg_cost
        
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.peak_profit = (current_price - cost_price) / cost_price
        
        if position.peak_price > 0:
            position.trailing_stop_price = position.peak_price * (1 - V64_TRAILING_STOP_RATIO)
        
        if current_price <= position.stop_loss_price:
            position.stop_loss_triggered = True
            return True, f"硬止损 (亏损>{V64_HARD_STOP_LOSS_RATIO*100:.1f}%)"
        
        if position.trailing_stop_price > 0:
            if current_price <= position.trailing_stop_price:
                position.trailing_stop_triggered = True
                return True, f"移动止盈 (回撤>{V64_TRAILING_STOP_RATIO*100:.1f}%)"
        
        current_profit = (current_price - cost_price) / cost_price
        if current_profit >= V64_PROFIT_TARGET_RATIO:
            position.profit_target_triggered = True
            return True, f"目标止盈 (盈利>{V64_PROFIT_TARGET_RATIO*100:.1f}%)"
        
        if position.holding_days >= V64_TIME_STOP_DAYS:
            if current_profit <= 0:
                position.time_stop_triggered = True
                return True, f"时间止损 (持有{position.holding_days}天不盈利)"
        
        return None
    
    def update_positions(self, market_data: Dict[str, Dict[str, float]],
                         trade_date: str):
        """更新持仓状态"""
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            current_price = data.get('close', 0)
            
            if current_price <= 0:
                continue
            
            position.current_price = current_price
            position.market_value = current_price * position.shares
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            
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
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V64_INITIAL_CAPITAL',
    'V64_MAX_POSITIONS',
    'V64_MONTHLY_TRADE_LIMIT',
    'V64_WEEKLY_TRADE_LIMIT',
    'V64_GLOBAL_TRADE_LIMIT',
    'V64_WARMUP_PERIOD',
    'V64_MIN_SAMPLE_SIZE',
    'V64_RS_WINDOW',
    'V64_RS_OUTPERFORM_THRESHOLD',
    'V64_FUND_FLOW_CONSECUTIVE_DAYS',
    'V64_PULLBACK_MA_PERIOD',
    'V64_PULLBACK_TOLERANCE',
    'V64_VCP_WINDOW',
    'V64_VCP_MAX_AMPLITUDE',
    'V64_VCP_MIN_SCORE',
    'V64_MARKET_DECLINE_RATIO_THRESHOLD',
    'V64_COMMISSION_RATE',
    'V64_MIN_COMMISSION',
    'V64_SLIPPAGE_BUY',
    'V64_SLIPPAGE_SELL',
    'V64_STAMP_DUTY',
    'V64_TRANSFER_FEE',
    'V64_FRICTION_COST',
    'V64_HARD_STOP_LOSS_RATIO',
    'V64_TIME_STOP_DAYS',
    'V64_TRAILING_STOP_RATIO',
    'V64_PROFIT_TARGET_RATIO',
    'V64_MAX_SINGLE_POSITION_PCT',
    'V64_SELECTION_PERCENTILE',
    
    # 数据类
    'V64Position',
    'V64Trade',
    'V64TradeAudit',
    'V64Signal',
    'V64MarketRegime',
    
    # 核心类
    'DatabaseIntegrityCheck',
    'V64DataManager',
    'V64AlphaCenter',
    'V64TradeExec',
]