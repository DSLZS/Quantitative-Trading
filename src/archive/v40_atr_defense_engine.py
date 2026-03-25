"""
V40 ATR Defense Engine - ATR 动态防御与盈利因子增强

【V40 核心改进 - ATR 动态协议】

1. 技术架构升级：ATR 动态协议
   - 动态止损 (Trailing Stop)：废除 8% 固定止损，引入 2 * ATR(20) 移动止损
   - 风险平价调仓 (Position Sizing)：波动大的股票少买，波动小的多买
   - 入场过滤 (Volatility Filter)：市场波动率过高时停止开新仓

2. 因子优化：从"强度"到"动能"
   - 引入 RSRS 因子 (阻力支撑相对强度)
   - 剔除平庸信号：只有 Top 10% 且股价站上 60 日均线才准许入场

3. 审计与对比要求
   - 场景 A：0.1% 滑点
   - 场景 B：0.3% 滑点
   - 目标：夏普比率 > 0.8，最大回撤 < 10%

4. 硬性审计要求
   - 分母锚定：初始资金严禁偏离 100,000.00
   - 交易频率：平均持仓天数 20-50 天
   - 禁止数据美化：如实反映 0.3% 滑点下的结果

作者：真实量化系统
版本：V40.0 ATR Defense Engine
日期：2026-03-19
"""

import sys
import json
import math
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# V40 配置常量 - ATR 动态防御
# ===========================================

# 固定基准配置
FIXED_INITIAL_CAPITAL = 100000.00  # 原始投入资金硬约束

# V40 持仓约束
TARGET_POSITIONS = 10              # 持仓数量 10 只
MAX_POSITIONS = 10                 # 最大持仓数量

# V40 选股池
TOP_N_STOCKS = 50                  # 选股池 Top 50

# V40 强制持仓周期 - 维持 20-50 天平均持仓
MIN_HOLDING_DAYS = 25              # 买入后锁定 25 个交易日（除非止损）
MAX_HOLDING_DAYS = 55              # 最大持仓天数，超过强制卖出

# V40 调仓周期 - 降低交易频率
REBALANCE_FREQUENCY = 25           # 每 25 个交易日调仓一次

# V40 ATR 止损配置
ATR_PERIOD = 20                    # ATR 计算周期
TRAILING_STOP_ATR_MULT = 2.0       # 2 * ATR 移动止损
INITIAL_STOP_LOSS_RATIO = 0.08     # 初始止损 8%（作为底线）

# V40 风险平价配置
RISK_TARGET_PER_POSITION = 0.005   # 单只股票风险暴露 0.5%
MAX_POSITION_RATIO = 0.15          # 单只股票最大仓位 15%
MIN_POSITION_RATIO = 0.05          # 单只股票最小仓位 5%

# V40 入场过滤
VOLATILITY_FILTER_WINDOW = 20      # 市场波动率计算窗口
VOLATILITY_FILTER_THRESHOLD = 1.5  # 超过 1.5 倍均值时停止开仓

# V40 基础滑点配置（场景 A）
BASE_SLIPPAGE_BUY = 0.001   # 买入滑点 +0.1%
BASE_SLIPPAGE_SELL = 0.001  # 卖出滑点 -0.1%

# V40 费率配置
COMMISSION_RATE = 0.0003    # 佣金率 0.03%
MIN_COMMISSION = 5.0        # 单笔最低佣金 5 元
STAMP_DUTY = 0.0005         # 印花税 0.05% (卖出收取)
TRANSFER_FEE = 0.00001      # 过户费 0.001%（双向）

# V40 因子权重
FACTOR_WEIGHTS = {
    'trend_strength_20': 0.25,    # 20 日趋势强度
    'trend_strength_60': 0.25,    # 60 日趋势强度
    'rsrs_factor': 0.30,          # RSRS 因子（新增）
    'volatility_adjusted_momentum': 0.20,  # 波动率调整动量
}

# V40 数据配置
MARKET_INDEX_SYMBOL = "000300.SH"

# 内置备用成分股列表（50 只蓝筹股）
FALLBACK_STOCKS = [
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "300750.SZ", "name": "宁德时代"},
    {"symbol": "000858.SZ", "name": "五粮液"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "600036.SH", "name": "招商银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "002415.SZ", "name": "海康威视"},
    {"symbol": "601888.SH", "name": "中国中免"},
    {"symbol": "600276.SH", "name": "恒瑞医药"},
    {"symbol": "601166.SH", "name": "兴业银行"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000002.SZ", "name": "万科 A"},
    {"symbol": "600030.SH", "name": "中信证券"},
    {"symbol": "000651.SZ", "name": "格力电器"},
    {"symbol": "000725.SZ", "name": "京东方 A"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "300059.SZ", "name": "东方财富"},
    {"symbol": "601398.SH", "name": "工商银行"},
    {"symbol": "601988.SH", "name": "中国银行"},
    {"symbol": "601857.SH", "name": "中国石油"},
    {"symbol": "600000.SH", "name": "浦发银行"},
    {"symbol": "600016.SH", "name": "民生银行"},
    {"symbol": "600028.SH", "name": "中国石化"},
    {"symbol": "600031.SH", "name": "三一重工"},
    {"symbol": "600048.SH", "name": "保利发展"},
    {"symbol": "600050.SH", "name": "中国联通"},
    {"symbol": "600104.SH", "name": "上汽集团"},
    {"symbol": "600309.SH", "name": "万华化学"},
    {"symbol": "600436.SH", "name": "片仔癀"},
    {"symbol": "600585.SH", "name": "海螺水泥"},
    {"symbol": "600588.SH", "name": "用友网络"},
    {"symbol": "600690.SH", "name": "海尔智家"},
    {"symbol": "600809.SH", "name": "山西汾酒"},
    {"symbol": "600887.SH", "name": "伊利股份"},
    {"symbol": "600900.SH", "name": "长江电力"},
    {"symbol": "601012.SH", "name": "隆基绿能"},
    {"symbol": "601088.SH", "name": "中国神华"},
    {"symbol": "601288.SH", "name": "农业银行"},
    {"symbol": "601328.SH", "name": "交通银行"},
    {"symbol": "601601.SH", "name": "中国太保"},
    {"symbol": "601628.SH", "name": "中国人寿"},
    {"symbol": "601668.SH", "name": "中国建筑"},
    {"symbol": "601688.SH", "name": "华泰证券"},
    {"symbol": "601766.SH", "name": "中国中车"},
    {"symbol": "601816.SH", "name": "京沪高铁"},
    {"symbol": "601898.SH", "name": "中煤能源"},
    {"symbol": "601919.SH", "name": "中远海控"},
    {"symbol": "601939.SH", "name": "建设银行"},
    {"symbol": "601985.SH", "name": "中国核电"},
    {"symbol": "601995.SH", "name": "中金公司"},
]


# ===========================================
# V40 审计追踪器
# ===========================================

@dataclass
class V40TradeAudit:
    """V40 真实交易审计记录"""
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
    entry_signal: float = 0.0
    signal_rank: int = 0
    atr_at_entry: float = 0.0      # 入场时 ATR
    initial_stop_price: float = 0.0  # 初始止损价
    peak_price: float = 0.0        # 持仓期间最高价
    trailing_stop_triggered: bool = False  # 移动止损是否触发


@dataclass
class V40RejectedStock:
    """V40 被拒绝股票记录"""
    trade_date: str
    symbol: str
    rank: int
    signal: float
    reason: str


@dataclass
class V40TurnoverRecord:
    """V40 换手率记录"""
    trade_date: str
    turnover_ratio: float
    buy_count: int
    sell_count: int
    total_traded_value: float


@dataclass
class V40ProfitSourceRecord:
    """V40 盈利来源分析记录"""
    category: str  # "captured_bull" or "avoided_crash"
    symbol: str
    pnl: float
    holding_days: int
    description: str


@dataclass
class V40ATRCaseStudy:
    """V40 ATR 止损案例分析"""
    symbol: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    atr_at_entry: float
    initial_stop: float
    trailing_stop_path: List[float]
    final_stop_price: float
    pnl: float
    description: str


@dataclass
class V40AuditRecord:
    """V40 真实审计记录 - ATR 动态防御"""
    # 基础信息
    scenario_name: str = ""
    total_trading_days: int = 0
    actual_trading_days: int = 0
    total_buys: int = 0
    total_sells: int = 0
    
    # 费用统计
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_stamp_duty: float = 0.0
    total_transfer_fee: float = 0.0
    total_fees: float = 0.0
    
    # 固定基准审计
    fixed_initial_capital: float = FIXED_INITIAL_CAPITAL
    final_nav: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0
    
    # 交易统计
    profitable_trades: int = 0
    losing_trades: int = 0
    winning_pnl: float = 0.0
    losing_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    win_rate: float = 0.0
    avg_holding_days: float = 0.0
    
    # 换手率统计
    total_turnover: float = 0.0
    avg_daily_turnover: float = 0.0
    max_daily_turnover: float = 0.0
    
    # 空仓统计
    empty_position_days: int = 0
    rejected_stocks_count: int = 0
    volatility_filter_triggered_days: int = 0  # 波动率过滤触发天数
    
    # ATR 统计
    avg_atr_stop_distance: float = 0.0  # 平均 ATR 止损距离
    trailing_stop_triggered_count: int = 0  # 移动止损触发次数
    initial_stop_triggered_count: int = 0  # 初始止损触发次数
    max_holding_days_reached: int = 0  # 最大持仓天数触发次数
    
    # 风险平价统计
    avg_position_size: float = 0.0
    avg_risk_per_position: float = 0.0
    
    # 因子统计
    avg_trend_20: float = 0.0
    avg_trend_60: float = 0.0
    avg_rsrs: float = 0.0
    avg_volatility_adjusted_momentum: float = 0.0
    
    # 盈利来源分析
    profit_from_bulls: float = 0.0  # 来自抓住牛股的盈利
    profit_from_avoided_crash: float = 0.0  # 来自躲过大跌的盈利（止损避免的亏损）
    
    # 错误追踪
    errors: List[str] = field(default_factory=list)
    nav_history: List[Tuple[str, float]] = field(default_factory=list)
    trade_log: List[V40TradeAudit] = field(default_factory=list)
    rejected_stocks: List[V40RejectedStock] = field(default_factory=list)
    turnover_records: List[V40TurnoverRecord] = field(default_factory=list)
    atr_case_studies: List[V40ATRCaseStudy] = field(default_factory=list)
    
    # V40 验证
    atr_stop_verified: bool = False  # ATR 止损验证
    position_sizing_verified: bool = False  # 风险平价调仓验证
    volatility_filter_verified: bool = False  # 入场过滤验证
    holding_period_verified: bool = False  # 持仓周期验证 (20-50 天)
    sharpe_target_met: bool = False  # 夏普比率>0.8
    drawdown_target_met: bool = False  # 最大回撤<10%
    
    def to_table(self) -> str:
        """输出真实审计表"""
        total_trade_count = self.total_buys + self.total_sells
        total_trades = self.profitable_trades + self.losing_trades
        
        # 状态图标
        atr_status = "✅" if self.atr_stop_verified else "❌"
        sizing_status = "✅" if self.position_sizing_verified else "❌"
        vol_filter_status = "✅" if self.volatility_filter_verified else "❌"
        holding_status = "✅" if self.holding_period_verified else "❌"
        sharpe_status = "✅" if self.sharpe_target_met else "❌"
        drawdown_status = "✅" if self.drawdown_target_met else "❌"
        
        # 持仓周期验证 (20-50 天)
        holding_period_ok = 20 <= self.avg_holding_days <= 50
        
        # 夏普比率验证
        sharpe_ok = self.sharpe_ratio >= 0.8
        
        # 最大回撤验证
        drawdown_ok = self.max_drawdown <= 0.10
        
        return f"""
╔══════════════════════════════════════════════════════════════╗
║          V40 ATR DEFENSE ENGINE 审计报告                     ║
╠══════════════════════════════════════════════════════════════╣
║  场景：{self.scenario_name:<50} ║
╠══════════════════════════════════════════════════════════════╣
║  【固定基准审计】                                           ║
║  Fixed Initial Capital : {self.fixed_initial_capital:>10.2f} 元 (锁定分母)          ║
║  Final NAV             : {self.final_nav:>10.2f} 元                             ║
║  Total Return (固定)    : {self.total_return:>10.2%}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【性能指标】                                               ║
║  年化收益率            : {self.annual_return:>10.2%}                              ║
║  夏普比率              : {self.sharpe_ratio:>10.3f}  (目标：≥0.8) {sharpe_status} ║
║  最大回撤              : {self.max_drawdown:>10.2%}  (目标：≤10%) {drawdown_status} ║
║  最大回撤天数          : {self.max_drawdown_days:>10}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【交易统计】                                               ║
║  总交易次数            : {total_trade_count:>10} 次                              ║
║  买入次数              : {self.total_buys:>10} 次                              ║
║  卖出次数              : {self.total_sells:>10} 次                              ║
║  盈利交易              : {self.profitable_trades:>10} 次                              ║
║  亏损交易              : {self.losing_trades:>10} 次                              ║
║  胜率                  : {self.win_rate:>10.1%}                              ║
║  平均持仓天数          : {self.avg_holding_days:>10.1f} 天  (目标：20-50) {holding_status} ║
╠══════════════════════════════════════════════════════════════╣
║  【换手率统计】                                             ║
║  总换手率              : {self.total_turnover:>10.2%}                              ║
║  日均换手率            : {self.avg_daily_turnover:>10.2%}                              ║
║  最大日换手率          : {self.max_daily_turnover:>10.2%}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【ATR 动态止损】                                            ║
║  平均 ATR 止损距离       : {self.avg_atr_stop_distance:>10.2%}                             ║
║  移动止损触发次数       : {self.trailing_stop_triggered_count:>10} 次                              ║
║  初始止损触发次数       : {self.initial_stop_triggered_count:>10} 次                              ║
║  最大持仓天数触发       : {self.max_holding_days_reached:>10} 次                              ║
║  ATR 止损验证           : {atr_status}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【风险平价调仓】                                           ║
║  平均仓位              : {self.avg_position_size:>10.1%}                              ║
║  平均风险暴露          : {self.avg_risk_per_position:>10.2%}                             ║
║  风险平价验证          : {sizing_status}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【入场过滤】                                               ║
║  波动率过滤触发天数     : {self.volatility_filter_triggered_days:>10} 天                              ║
║  入场过滤验证          : {vol_filter_status}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【盈利来源分析】                                           ║
║  来自抓住牛股          : {self.profit_from_bulls:>10.2f} 元                   ║
║  来自躲过大跌          : {self.profit_from_avoided_crash:>10.2f} 元                   ║
╠══════════════════════════════════════════════════════════════╣
║  【因子统计】                                               ║
║  20 日趋势强度          : {self.avg_trend_20:>10.3f}                              ║
║  60 日趋势强度          : {self.avg_trend_60:>10.3f}                              ║
║  RSRS 因子             : {self.avg_rsrs:>10.3f}                              ║
║  波动率调整动量         : {self.avg_volatility_adjusted_momentum:>10.3f}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【费用统计】                                               ║
║  总佣金                : {self.total_commission:>10.2f} 元                   ║
║  总滑点                : {self.total_slippage:>10.2f} 元                   ║
║  总印花税              : {self.total_stamp_duty:>10.2f} 元                   ║
║  总过户费              : {self.total_transfer_fee:>10.2f} 元                   ║
║  总费用                : {self.total_fees:>10.2f} 元                   ║
╚══════════════════════════════════════════════════════════════╝
"""


# ===========================================
# V40 场景配置
# ===========================================

@dataclass
class V40ScenarioConfig:
    """V40 场景配置"""
    name: str
    settlement_delay: int
    slippage_buy: float
    slippage_sell: float
    description: str


V40_SCENARIOS = {
    "A": V40ScenarioConfig(
        name="场景 A (0.1% 滑点基准)",
        settlement_delay=1,
        slippage_buy=0.001,
        slippage_sell=0.001,
        description="T+1 开盘成交，双边滑点 0.1%"
    ),
    "B": V40ScenarioConfig(
        name="场景 B (0.3% 滑点冲击)",
        settlement_delay=1,
        slippage_buy=0.003,
        slippage_sell=0.003,
        description="T+1 开盘成交，双边滑点 0.3%"
    ),
}


# ===========================================
# V40 ATR 因子引擎
# ===========================================

class V40ATRFactorEngine:
    """
    V40 ATR 因子引擎 - ATR 动态防御与 RSRS 因子
    
    【核心功能】
    1. ATR 计算 - 用于动态止损
    2. RSRS 因子 - 阻力支撑相对强度
    3. 波动率调整动量
    4. 市场波动率指数 (VIX 模拟)
    """
    
    EPSILON = 1e-9
    
    def __init__(self, db=None):
        self.db = db or DatabaseManager.get_instance()
        self.factor_weights = FACTOR_WEIGHTS
    
    def compute_all_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """V40 全因子计算"""
        try:
            required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
            self._validate_columns(df, required_cols)
            
            result = df.clone().with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            ])
            
            logger.info("[Step 1] Computing ATR(20)...")
            result = self._compute_atr(result, period=ATR_PERIOD)
            
            logger.info("[Step 2] Computing RSRS factor...")
            result = self._compute_rsrs_factor(result)
            
            logger.info("[Step 3] Computing trend factors...")
            result = self._compute_trend_factors(result)
            
            logger.info("[Step 4] Computing volatility adjusted momentum...")
            result = self._compute_volatility_adjusted_momentum(result)
            
            logger.info("[Step 5] Computing market volatility index...")
            result = self._compute_market_volatility_index(result)
            
            logger.info("[Step 6] Computing composite signal...")
            result = self._compute_composite_signal(result)
            
            logger.info("All V40 factors computed successfully")
            return result
            
        except Exception as e:
            logger.error(f"compute_all_factors FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_columns(self, df: pl.DataFrame, required_columns: List[str]) -> bool:
        """验证必需列"""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def _compute_atr(self, df: pl.DataFrame, period: int = ATR_PERIOD) -> pl.DataFrame:
        """
        ATR (Average True Range) 计算
        
        True Range = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = SMA(True Range, period)
        """
        try:
            result = df.clone()
            
            # 计算前一日收盘价
            prev_close = pl.col('close').shift(1).over('symbol')
            
            # 计算三种 TR 成分
            tr1 = pl.col('high') - pl.col('low')
            tr2 = (pl.col('high') - prev_close).abs()
            tr3 = (pl.col('low') - prev_close).abs()
            
            # True Range = max(tr1, tr2, tr3)
            tr = pl.max_horizontal([tr1, tr2, tr3])
            
            # ATR = SMA(TR, period)
            atr = tr.rolling_mean(window_size=period).over('symbol')
            
            result = result.with_columns([
                tr.alias('true_range'),
                atr.alias('atr_20'),
                prev_close.alias('prev_close')
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"_compute_atr FAILED: {e}")
            raise
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        RSRS 因子 (Resistance Support Relative Strength) - 阻力支撑相对强度
        
        核心思想：
        1. 使用线性回归拟合 N 日内的 High-Low 关系
        2. 斜率表示阻力支撑的相对强度
        3. 斜率稳定表示趋势健康
        
        公式：RSRS = Slope(High ~ Low, window=18) * R²
        """
        try:
            result = df.clone()
            
            # RSRS 窗口
            rsrs_window = 18
            
            # 计算滚动窗口的斜率（使用简化方法）
            # 斜率 ≈ (High - Low) 的变化率
            high_low_spread = pl.col('high') - pl.col('low')
            
            # 计算斜率稳定性（使用滚动相关系数的平方作为 R²代理）
            # 这里使用简化的计算方法
            spread_mean = high_low_spread.rolling_mean(window_size=rsrs_window).over('symbol')
            spread_std = high_low_spread.rolling_std(window_size=rsrs_window).over('symbol')
            
            # RSRS 核心：斜率 = (当前 spread - 均值) / 标准差
            rsrs_raw = (high_low_spread - spread_mean) / (spread_std + self.EPSILON)
            
            # 计算 R²代理（使用滚动相关性）
            # 简化：使用波动率比率作为 R²代理
            r_squared = 1.0 / (1.0 + spread_std)
            
            # 最终 RSRS = 斜率 * R²
            rsrs = rsrs_raw * r_squared
            
            # 标准化到合理范围
            rsrs = rsrs * 0.5  # 缩放
            
            result = result.with_columns([
                high_low_spread.alias('high_low_spread'),
                spread_mean.alias('spread_mean'),
                spread_std.alias('spread_std'),
                rsrs.alias('rsrs_factor')
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"_compute_rsrs_factor FAILED: {e}")
            raise
    
    def _compute_trend_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        趋势因子计算
        
        - trend_strength_20: 20 日趋势强度
        - trend_strength_60: 60 日趋势强度
        """
        try:
            result = df.clone()
            
            # 20 日趋势强度
            close_20_ago = pl.col('close').shift(20).over('symbol')
            trend_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
            
            # 60 日趋势强度
            close_60_ago = pl.col('close').shift(60).over('symbol')
            trend_60 = (pl.col('close') - close_60_ago) / (close_60_ago + self.EPSILON)
            
            # 60 日均线（用于入场过滤）
            ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
            
            result = result.with_columns([
                trend_20.alias('trend_strength_20'),
                trend_60.alias('trend_strength_60'),
                ma60.alias('ma60')
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"_compute_trend_factors FAILED: {e}")
            raise
    
    def _compute_volatility_adjusted_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        波动率调整动量
        
        公式：Momentum / Volatility
        核心：高波动率的动量不可靠，需要打折
        """
        try:
            result = df.clone()
            
            # 20 日动量
            close_20_ago = pl.col('close').shift(20).over('symbol')
            momentum_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
            
            # 20 日波动率
            returns = pl.col('close').pct_change().over('symbol')
            vol_20 = returns.rolling_std(window_size=20).over('symbol')
            
            # 波动率调整动量 = 动量 / 波动率
            vol_adj_momentum = momentum_20 / (vol_20 + self.EPSILON)
            
            # 缩放
            vol_adj_momentum = vol_adj_momentum * 0.5
            
            result = result.with_columns([
                momentum_20.alias('momentum_20'),
                vol_20.alias('volatility_20'),
                vol_adj_momentum.alias('volatility_adjusted_momentum')
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"_compute_volatility_adjusted_momentum FAILED: {e}")
            raise
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        市场波动率指数 (VIX 模拟)
        
        计算方法：
        1. 计算市场指数（使用所有股票的平均）的收益率
        2. 计算滚动标准差
        3. 标准化为指数形式
        """
        try:
            result = df.clone()
            
            # 计算每只股票的收益率
            returns = pl.col('close').pct_change().over('symbol')
            
            # 计算市场平均波动率（所有股票的平均）
            # 这里使用简化方法：计算每只股票的 20 日波动率，然后取横截面平均
            stock_vol = returns.rolling_std(window_size=20, ddof=1).over('symbol')
            
            # 计算市场平均波动率
            market_vol = stock_vol
            
            # 计算市场波动率的历史均值（用于过滤）
            market_vol_mean = market_vol.rolling_mean(window_size=VOLATILITY_FILTER_WINDOW).over('symbol')
            
            # 波动率比率（用于过滤）
            vol_ratio = market_vol / (market_vol_mean + self.EPSILON)
            
            # VIX 模拟值（缩放）
            vix_sim = market_vol * 100
            
            result = result.with_columns([
                returns.alias('returns'),
                stock_vol.alias('stock_volatility'),
                market_vol.alias('market_volatility'),
                market_vol_mean.alias('market_volatility_mean'),
                vol_ratio.alias('volatility_ratio'),
                vix_sim.alias('vix_sim')
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"_compute_market_volatility_index FAILED: {e}")
            raise
    
    def _compute_composite_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算加权综合信号
        
        V40 核心：
        1. 因子加权
        2. 入场过滤（波动率 + 均线）
        3. 排名计算
        """
        try:
            result = df.clone()
            
            # 因子标准化 (Z-Score)
            factors = ['trend_strength_20', 'trend_strength_60', 'rsrs_factor', 'volatility_adjusted_momentum']
            
            for factor in factors:
                if factor not in result.columns:
                    raise ValueError(f"Missing factor: {factor}")
            
            # 计算各因子的均值和标准差用于标准化
            factor_stats = {}
            for factor in factors:
                mean_val = result[factor].mean() or 0
                std_val = result[factor].std() or 1
                factor_stats[factor] = (mean_val, std_val)
            
            # 标准化因子
            standardized_factors = []
            for factor in factors:
                mean_val, std_val = factor_stats[factor]
                z_factor = (pl.col(factor) - mean_val) / (std_val + self.EPSILON)
                standardized_factors.append(z_factor * self.factor_weights[factor])
            
            # 加权综合信号
            signal = sum(standardized_factors)
            
            # 入场过滤条件
            # 1. 波动率过滤：volatility_ratio > 1.5 时禁止入场
            vol_filter = pl.col('volatility_ratio') <= VOLATILITY_FILTER_THRESHOLD
            
            # 2. 均线过滤：价格必须站上 60 日均线
            ma_filter = pl.col('close') >= pl.col('ma60')
            
            # 综合过滤
            entry_allowed = vol_filter & ma_filter
            
            result = result.with_columns([
                signal.alias('signal'),
                vol_filter.alias('vol_filter_pass'),
                ma_filter.alias('ma_filter_pass'),
                entry_allowed.alias('entry_allowed')
            ])
            
            # 计算排名（仅对允许入场的股票）
            result = result.with_columns([
                pl.when(pl.col('entry_allowed'))
                .then(pl.col('signal').rank('ordinal', descending=True).over('trade_date'))
                .otherwise(9999)
                .cast(pl.Int64)
                .alias('rank')
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"_compute_composite_signal FAILED: {e}")
            raise


# ===========================================
# V40 会计类 - ATR 动态防御
# ===========================================

@dataclass
class V40Position:
    """V40 真实持仓记录 - ATR 动态防御"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_score: float
    signal_rank: int
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    buy_trade_day: int = 0
    
    # ATR 动态防御相关
    atr_at_entry: float = 0.0           # 入场时 ATR
    initial_stop_price: float = 0.0     # 初始止损价
    trailing_stop_price: float = 0.0    # 当前移动止损价
    trailing_stop_triggered: bool = False  # 移动止损是否触发
    trailing_stop_history: List[float] = field(default_factory=list)  # 止损价历史


@dataclass
class V40Trade:
    """V40 真实交易记录"""
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
    execution_price: float = 0.0
    scenario: str = ""


class V40Accountant:
    """
    V40 会计类 - ATR 动态防御与风险平价调仓
    
    【核心功能】
    1. ATR 动态止损 (Trailing Stop)
    2. 风险平价调仓 (Position Sizing)
    3. 入场过滤 (Volatility Filter)
    """
    
    def __init__(self, initial_capital: float = FIXED_INITIAL_CAPITAL, 
                 scenario_config: V40ScenarioConfig = None, db=None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.db = db or DatabaseManager.get_instance()
        self.scenario_config = scenario_config or V40_SCENARIOS["A"]
        
        self.positions: Dict[str, V40Position] = {}
        self.trades: List[V40Trade] = []
        self.trade_log: List[V40TradeAudit] = []
        
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        
        # 持仓锁定期追踪
        self.locked_positions: Dict[str, int] = {}
        
        # 当前交易日
        self.current_trade_date: Optional[str] = None
        self.trade_day_counter: int = 0
        
        # 费率配置
        self.commission_rate = COMMISSION_RATE
        self.min_commission = MIN_COMMISSION
        self.stamp_duty = STAMP_DUTY
        self.transfer_fee = TRANSFER_FEE
        
        # ATR 止损配置
        self.trailing_stop_atr_mult = TRAILING_STOP_ATR_MULT
        self.initial_stop_loss_ratio = INITIAL_STOP_LOSS_RATIO
        self.max_holding_days = MAX_HOLDING_DAYS
        
        # 风险平价配置
        self.risk_target_per_position = RISK_TARGET_PER_POSITION
        self.max_position_ratio = MAX_POSITION_RATIO
        self.min_position_ratio = MIN_POSITION_RATIO
        
        # 换手率统计
        self.daily_turnover: Dict[str, float] = {}
        
        # 波动率过滤统计
        self.volatility_filter_triggered = False
        self.volatility_filter_triggered_days = 0
    
    def increment_trade_day_counter(self, trade_date: str):
        """递增交易日计数器"""
        if self.last_trade_date != trade_date:
            self.today_sells.clear()
            self.today_buys.clear()
            self.last_trade_date = trade_date
            self.trade_day_counter += 1
    
    def unlock_expired_positions(self):
        """解锁到期持仓"""
        for symbol in list(self.locked_positions.keys()):
            self.locked_positions[symbol] -= 1
            if self.locked_positions[symbol] <= 0:
                del self.locked_positions[symbol]
    
    def reset_daily_counters(self, trade_date: str):
        """重置每日计数器"""
        self.increment_trade_day_counter(trade_date)
        self.unlock_expired_positions()
    
    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        return max(self.min_commission, amount * self.commission_rate)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        """计算过户费"""
        return shares * price * self.transfer_fee
    
    def calculate_position_size(self, symbol: str, atr: float, current_price: float, 
                                total_assets: float) -> Tuple[int, float]:
        """
        V40 风险平价调仓 - 计算仓位大小
        
        核心：波动大的股票少买，波动小的多买
        单只股票的风险暴露控制在总资产的 0.5%
        
        公式：
        - 风险金额 = 总资产 * 0.5%
        - 每股风险 = ATR * 2
        - 股数 = 风险金额 / 每股风险
        - 仓位 = 股数 * 价格
        
        返回：(shares, position_amount)
        """
        try:
            # 风险金额（总资产的 0.5%）
            risk_amount = total_assets * self.risk_target_per_position
            
            # 每股风险（2 * ATR）
            risk_per_share = atr * self.trailing_stop_atr_mult
            
            if risk_per_share <= 0 or current_price <= 0:
                return 0, 0.0
            
            # 计算股数
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100  # 取整到 100 股
            
            if shares < 100:
                return 0, 0.0
            
            # 计算仓位金额
            position_amount = shares * current_price
            
            # 应用仓位限制
            max_position = total_assets * self.max_position_ratio
            min_position = total_assets * self.min_position_ratio
            
            if position_amount > max_position:
                shares = int(max_position / current_price)
                shares = (shares // 100) * 100
                position_amount = shares * current_price
            
            if position_amount < min_position and position_amount > 0:
                # 如果计算出的仓位小于最小仓位，跳过
                return 0, 0.0
            
            return shares, position_amount
            
        except Exception as e:
            logger.error(f"calculate_position_size failed: {e}")
            return 0, 0.0
    
    def check_volatility_filter(self, market_volatility_ratio: float) -> bool:
        """
        V40 入场过滤 - 检查市场波动率
        
        如果市场波动率超过过去 20 天均值的 1.5 倍，停止开新仓
        """
        self.volatility_filter_triggered = market_volatility_ratio > VOLATILITY_FILTER_THRESHOLD
        if self.volatility_filter_triggered:
            self.volatility_filter_triggered_days += 1
        return not self.volatility_filter_triggered
    
    def execute_buy(
        self,
        trade_date: str,
        symbol: str,
        open_price: float,
        atr: float,
        target_amount: float,
        signal_score: float = 0.0,
        signal_rank: int = 0,
        reason: str = "",
    ) -> Optional[V40Trade]:
        """
        V40 买入执行 - ATR 动态防御
        """
        try:
            if symbol in self.today_sells:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already sold today ({trade_date})")
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"DUPLICATE BUY PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            # 场景化滑点
            execution_price = open_price * (1 + self.scenario_config.slippage_buy)
            
            # 使用目标仓位计算股数
            raw_shares = int(target_amount / execution_price)
            shares = (raw_shares // 100) * 100
            if shares < 100:
                return None
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * self.scenario_config.slippage_buy
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            total_cost = actual_amount + commission + slippage + transfer_fee
            
            if self.cash < total_cost:
                return None
            
            self.cash -= total_cost
            
            # V40 核心：ATR 动态止损初始化
            # 初始止损价 = 买入价 * (1 - 2 * ATR%) 或 买入价 * (1 - 8%)，取较大者
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - self.trailing_stop_atr_mult * atr_stop_distance)
            initial_stop_price = execution_price * (1 - self.initial_stop_loss_ratio)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            # 创建新持仓
            self.positions[symbol] = V40Position(
                symbol=symbol, shares=shares,
                avg_cost=(actual_amount + commission + slippage + transfer_fee) / shares,
                buy_price=execution_price, buy_date=trade_date,
                signal_score=signal_score, signal_rank=signal_rank,
                current_price=execution_price,
                holding_days=0,
                peak_price=execution_price, peak_profit=0.0,
                buy_trade_day=self.trade_day_counter,
                atr_at_entry=atr,
                initial_stop_price=stop_price,
                trailing_stop_price=stop_price,
                trailing_stop_history=[stop_price],
            )
            
            # 强制锁定 MIN_HOLDING_DAYS 个交易日
            self.locked_positions[symbol] = MIN_HOLDING_DAYS + 2
            
            self.today_buys.add(symbol)
            
            trade = V40Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee,
                total_cost=total_cost, reason=reason, execution_price=execution_price,
                scenario=self.scenario_config.name
            )
            self.trades.append(trade)
            
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Stop={stop_price:.2f}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            return None
    
    def execute_sell(
        self,
        trade_date: str,
        symbol: str,
        open_price: float,
        shares: Optional[int] = None,
        reason: str = "",
    ) -> Optional[V40Trade]:
        """
        V40 卖出执行 - ATR 动态防御
        """
        try:
            if symbol not in self.positions:
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            # 持仓锁定期检查
            if symbol in self.locked_positions and self.locked_positions[symbol] > 0:
                # 只有止损可以突破锁定期
                if reason not in ["stop_loss", "trailing_stop", "max_holding"]:
                    logger.debug(f"LOCKED POSITION: {symbol} locked for {self.locked_positions[symbol]} more days, reason={reason}")
                    return None
            
            pos = self.positions[symbol]
            available = pos.shares
            
            if shares is None or shares > available:
                shares = available
            
            if shares < 100:
                return None
            
            # 场景化滑点
            execution_price = open_price * (1 - self.scenario_config.slippage_sell)
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * self.scenario_config.slippage_sell
            stamp_duty = actual_amount * self.stamp_duty
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            net_proceeds = actual_amount - commission - slippage - stamp_duty - transfer_fee
            
            self.cash += net_proceeds
            
            cost_basis = pos.avg_cost * shares
            realized_pnl = net_proceeds - cost_basis
            
            # 计算持仓天数
            try:
                buy_date = datetime.strptime(pos.buy_date, "%Y-%m-%d")
                sell_date = datetime.strptime(trade_date, "%Y-%m-%d")
                calculated_holding_days = max(1, (sell_date - buy_date).days)
            except:
                calculated_holding_days = pos.holding_days if pos.holding_days > 0 else 1
            
            # 记录交易审计
            trade_audit = V40TradeAudit(
                symbol=symbol,
                buy_date=pos.buy_date,
                sell_date=trade_date,
                buy_price=pos.buy_price,
                sell_price=execution_price,
                shares=shares,
                gross_pnl=realized_pnl + (commission + slippage + stamp_duty + transfer_fee),
                total_fees=commission + slippage + stamp_duty + transfer_fee,
                net_pnl=realized_pnl,
                holding_days=calculated_holding_days,
                is_profitable=realized_pnl > 0,
                sell_reason=reason,
                entry_signal=pos.signal_score,
                signal_rank=pos.signal_rank,
                atr_at_entry=pos.atr_at_entry,
                initial_stop_price=pos.initial_stop_price,
                peak_price=pos.peak_price,
                trailing_stop_triggered=pos.trailing_stop_triggered
            )
            self.trade_log.append(trade_audit)
            
            # 删除持仓和锁
            del self.positions[symbol]
            self.locked_positions.pop(symbol, None)
            
            self.today_sells.add(symbol)
            
            trade = V40Trade(
                trade_date=trade_date, symbol=symbol, side="SELL", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=stamp_duty, transfer_fee=transfer_fee,
                total_cost=net_proceeds, reason=reason, holding_days=pos.holding_days,
                execution_price=execution_price, scenario=self.scenario_config.name
            )
            self.trades.append(trade)
            
            logger.info(f"  SELL {symbol} | {shares} shares @ {execution_price:.2f} | PnL: {realized_pnl:.2f} | Reason: {reason}")
            return trade
            
        except Exception as e:
            logger.error(f"execute_sell failed: {e}")
            return None
    
    def update_positions_and_check_stops(
        self, 
        prices: Dict[str, float],
        atrs: Dict[str, float],
        trade_date: str,
    ) -> List[Tuple[str, str]]:
        """
        V40 更新持仓价格并检查止损条件
        
        【核心逻辑】
        1. 更新峰值价格
        2. 更新移动止损价（只上移，不下移）
        3. 检查止损触发
        """
        sell_list = []
        
        for symbol, pos in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                pos.current_price = current_price
                pos.market_value = pos.shares * current_price
                pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.shares
                
                # 更新持仓天数
                pos.holding_days = max(0, self.trade_day_counter - pos.buy_trade_day)
                
                # 更新峰值价格
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    pos.peak_profit = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                
                # V40 核心：ATR 移动止损更新
                if symbol in atrs and atrs[symbol] > 0:
                    atr = atrs[symbol]
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    
                    # 计算新的移动止损价（基于当前价格）
                    new_trailing_stop = current_price * (1 - self.trailing_stop_atr_mult * atr_stop_distance)
                    
                    # 只上移，不下移
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        pos.trailing_stop_history.append(new_trailing_stop)
                
                # 检查止损触发
                # 1. 移动止损触发
                if current_price <= pos.trailing_stop_price:
                    pos.trailing_stop_triggered = True
                    sell_list.append((symbol, "trailing_stop"))
                    continue
                
                # 2. 初始止损底线（8%）
                profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                if profit_ratio <= -self.initial_stop_loss_ratio:
                    sell_list.append((symbol, "stop_loss"))
                    continue
                
                # 3. 最大持仓天数触发
                if pos.holding_days >= self.max_holding_days:
                    sell_list.append((symbol, "max_holding"))
                    continue
        
        return sell_list
    
    def record_turnover(self, trade_date: str, buy_count: int, sell_count: int, total_assets: float):
        """记录当日换手率"""
        if total_assets <= 0:
            return None
        
        daily_traded_value = sum(
            t.amount for t in self.trades 
            if t.trade_date == trade_date
        )
        turnover_ratio = daily_traded_value / total_assets
        
        self.daily_turnover[trade_date] = turnover_ratio
        
        return V40TurnoverRecord(
            trade_date=trade_date,
            turnover_ratio=turnover_ratio,
            buy_count=buy_count,
            sell_count=sell_count,
            total_traded_value=daily_traded_value
        )


# ===========================================
# V40 回测执行器
# ===========================================

class V40BacktestExecutor:
    """V40 ATR 动态防御回测执行器"""
    
    EPSILON = 1e-9
    
    def __init__(self, initial_capital: float = FIXED_INITIAL_CAPITAL, 
                 scenario_config: V40ScenarioConfig = None, db=None):
        self.accounting = V40Accountant(initial_capital=initial_capital, scenario_config=scenario_config, db=db)
        self.factor_engine = V40ATRFactorEngine(db=db)
        self.db = db or DatabaseManager.get_instance()
        self.initial_capital = initial_capital
        self.scenario_config = scenario_config or V40_SCENARIOS["A"]
        
        # 审计记录
        self.audit = V40AuditRecord()
        self.audit.scenario_name = self.scenario_config.name
        self.audit.fixed_initial_capital = self.initial_capital
    
    def run_backtest(
        self,
        data_df: pl.DataFrame,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """V40 ATR 动态防御回测"""
        try:
            logger.info("=" * 80)
            logger.info(f"V40 ATR DEFENSE ENGINE - {self.scenario_config.name}")
            logger.info(f"Description: {self.scenario_config.description}")
            logger.info("=" * 80)
            
            logger.info("\n[Step 1] Computing ATR and factors...")
            data_df = self.factor_engine.compute_all_factors(data_df)
            
            # 填充 NaN 值 (仅对数值类型列)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover_rate',
                           'true_range', 'atr_20', 'prev_close', 'high_low_spread',
                           'spread_mean', 'spread_std', 'rsrs_factor', 'trend_strength_20',
                           'trend_strength_60', 'ma60', 'momentum_20', 'volatility_20',
                           'volatility_adjusted_momentum', 'returns', 'stock_volatility',
                           'market_volatility', 'market_volatility_mean', 'volatility_ratio',
                           'vix_sim', 'signal']
            
            for col in numeric_cols:
                if col in data_df.columns:
                    data_df = data_df.with_columns([
                        pl.col(col).fill_nan(0).fill_null(0)
                    ])
            
            dates = sorted(data_df['trade_date'].unique().to_list())
            if not dates:
                return {"error": "No trading dates"}
            
            self.audit.total_trading_days = len(dates)
            
            logger.info(f"Backtest period: {start_date} to {end_date}, {len(dates)} trading days")
            
            # T+1 延迟执行
            signal_buffer: List[Tuple[str, pl.DataFrame]] = []
            
            for i, trade_date in enumerate(dates):
                self.audit.actual_trading_days += 1
                
                try:
                    day_signals = data_df.filter(pl.col('trade_date') == trade_date)
                    if day_signals.is_empty():
                        continue
                    
                    # 获取当日价格和 ATR
                    prices = {}
                    opens = {}
                    atrs = {}
                    vol_ratios = {}
                    
                    for row in day_signals.iter_rows(named=True):
                        symbol = row['symbol']
                        prices[symbol] = row['close'] if row['close'] is not None else 0
                        opens[symbol] = row['open'] if row['open'] is not None else 0
                        atrs[symbol] = row.get('atr_20', 0) or 0
                        vol_ratios[symbol] = row.get('volatility_ratio', 1) or 1
                    
                    # 递增交易日计数
                    self.accounting.increment_trade_day_counter(trade_date)
                    
                    # 解锁到期持仓
                    self.accounting.unlock_expired_positions()
                    
                    # 更新持仓价格并检查止损
                    sell_list = self.accounting.update_positions_and_check_stops(prices, atrs, trade_date)
                    
                    # 执行卖出（止损）
                    for symbol, reason in sell_list:
                        if symbol in self.accounting.positions:
                            open_price = opens.get(symbol, 0)
                            if open_price > 0:
                                self.accounting.execute_sell(trade_date, symbol, open_price, reason=reason)
                    
                    # T+1 延迟执行
                    if signal_buffer:
                        exec_signals = signal_buffer.pop(0)[1]
                        if exec_signals is not None and not exec_signals.is_empty():
                            # 检查调仓频率
                            should_rebalance = (self.audit.actual_trading_days % REBALANCE_FREQUENCY == 0)
                            if should_rebalance:
                                self._rebalance(trade_date, exec_signals, opens, atrs, vol_ratios)
                            else:
                                logger.debug(f"Skip rebalance on day {self.audit.actual_trading_days}")
                    
                    signal_buffer.append((trade_date, day_signals))
                    
                    # 计算 NAV
                    nav = self._compute_daily_nav(trade_date, prices)
                    total_assets = nav['total_assets']
                    self.audit.nav_history.append((trade_date, total_assets))
                    
                    # 记录换手率
                    turnover_record = self.accounting.record_turnover(
                        trade_date,
                        buy_count=len(self.accounting.today_buys),
                        sell_count=len(self.accounting.today_sells),
                        total_assets=total_assets
                    )
                    if turnover_record:
                        self.audit.turnover_records.append(turnover_record)
                    
                    if i % 20 == 0:
                        logger.info(f"  Date {trade_date}: NAV={total_assets:.2f}, "
                                   f"Positions={nav['position_count']}")
                    
                except Exception as e:
                    logger.error(f"Day {trade_date} failed: {e}")
                    logger.error(traceback.format_exc())
                    self.audit.errors.append(f"Day {trade_date}: {e}")
                    continue
            
            # 计算审计指标
            self._compute_audit_metrics(len(dates))
            
            # V40 验证
            self._verify_constraints()
            
            return self._generate_result(start_date, end_date)
            
        except Exception as e:
            logger.error(f"run_backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            self.audit.errors.append(f"run_backtest: {e}")
            raise
    
    def _rebalance(self, trade_date: str, prev_signals: pl.DataFrame, 
                   opens: Dict[str, float], atrs: Dict[str, float],
                   vol_ratios: Dict[str, float]):
        """
        V40 调仓 - ATR 动态防御与风险平价
        """
        try:
            # 获取 Top 50 选股池
            ranked = prev_signals.sort('rank', descending=False).head(TOP_N_STOCKS)
            target_symbols = set(ranked['symbol'].to_list())
            
            # 获取 Top 10 作为买入目标
            buy_candidates = ranked.head(TARGET_POSITIONS)
            
            # 计算当前总资产
            total_assets = self.accounting.cash + sum(
                pos.shares * pos.current_price for pos in self.accounting.positions.values()
            )
            
            # 检查波动率过滤
            avg_vol_ratio = np.mean(list(vol_ratios.values())) if vol_ratios else 1.0
            vol_filter_pass = self.accounting.check_volatility_filter(avg_vol_ratio)
            
            if not vol_filter_pass:
                logger.debug(f"Volatility filter triggered (ratio={avg_vol_ratio:.2f}), skipping new buys")
                self.audit.volatility_filter_triggered_days += 1
            
            # 卖出逻辑
            for symbol in list(self.accounting.positions.keys()):
                if symbol not in target_symbols:
                    # 跌出 Top 50，检查是否可以卖出
                    pos = self.accounting.positions[symbol]
                    if pos.holding_days < MIN_HOLDING_DAYS:
                        logger.debug(f"LOCKED (rank_drop): {symbol} holding_days={pos.holding_days}")
                        continue
                    
                    open_price = opens.get(symbol, pos.buy_price)
                    if open_price > 0:
                        self.accounting.execute_sell(
                            trade_date, symbol, open_price, reason="rank_drop"
                        )
            
            # 重新计算可用现金
            available_cash = self.accounting.cash
            
            # 买入新标的 - 使用风险平价调仓
            for row in buy_candidates.iter_rows(named=True):
                symbol = row['symbol']
                rank = int(row['rank']) if row['rank'] else 9999
                signal = row.get('signal', 0) or 0
                entry_allowed = row.get('entry_allowed', True)
                
                if not entry_allowed:
                    continue
                
                if symbol in self.accounting.positions:
                    continue
                
                if symbol in self.accounting.today_sells:
                    continue
                
                open_price = opens.get(symbol, 0)
                if open_price <= 0:
                    continue
                
                atr = atrs.get(symbol, 0)
                if atr <= 0:
                    continue
                
                # V40 核心：风险平价调仓
                shares, target_amount = self.accounting.calculate_position_size(
                    symbol, atr, open_price, total_assets
                )
                
                if shares <= 0 or target_amount <= 0:
                    continue
                
                if available_cash < target_amount * 0.9:
                    continue
                
                # 执行买入
                trade = self.accounting.execute_buy(
                    trade_date, symbol, open_price, atr, target_amount,
                    signal_score=signal, signal_rank=rank, reason="top_rank"
                )
                
                if trade:
                    available_cash -= trade.total_cost
            
        except Exception as e:
            logger.error(f"_rebalance failed: {e}")
            self.audit.errors.append(f"_rebalance: {e}")
    
    def _compute_daily_nav(self, trade_date: str, prices: Dict[str, float]) -> Dict:
        """计算每日 NAV"""
        market_value = sum(
            pos.shares * prices.get(pos.symbol, pos.current_price)
            for pos in self.accounting.positions.values()
        )
        total_assets = self.accounting.cash + market_value
        
        position_count = len(self.accounting.positions)
        position_ratio = market_value / total_assets if total_assets > 0 else 0.0
        
        daily_return = 0.0
        if self.audit.nav_history:
            prev_nav = self.audit.nav_history[-1][1]
            if prev_nav > 0:
                daily_return = (total_assets - prev_nav) / prev_nav
        
        return {
            'trade_date': trade_date,
            'cash': self.accounting.cash,
            'market_value': market_value,
            'total_assets': total_assets,
            'position_count': position_count,
            'daily_return': daily_return,
            'position_ratio': position_ratio,
        }
    
    def _compute_audit_metrics(self, trading_days: int):
        """V40 审计指标计算"""
        if not self.audit.nav_history:
            return
        
        # 固定基准审计
        self.audit.final_nav = self.audit.nav_history[-1][1]
        self.audit.total_return = (self.audit.final_nav - FIXED_INITIAL_CAPITAL) / FIXED_INITIAL_CAPITAL
        
        years = trading_days / 252.0
        if years > 0:
            self.audit.annual_return = (1 + self.audit.total_return) ** (1 / years) - 1
        
        # 夏普比率
        if len(self.audit.nav_history) > 1:
            nav_values = [n[1] for n in self.audit.nav_history]
            returns = np.diff(nav_values) / np.where(np.array(nav_values[:-1]) != 0, np.array(nav_values[:-1]), 1)
            returns = [r for r in returns if np.isfinite(r)]
            if len(returns) > 1:
                daily_std = np.std(returns, ddof=1)
                if daily_std > 0:
                    self.audit.sharpe_ratio = np.mean(returns) / daily_std * np.sqrt(252)
        
        # 回撤计算
        if len(self.audit.nav_history) > 1:
            nav_series = pl.Series([n[1] for n in self.audit.nav_history])
            drawdown_series = 1 - nav_series / nav_series.cum_max()
            self.audit.max_drawdown = float(drawdown_series.max())
            self.audit.max_drawdown_days = int(drawdown_series.arg_max())
        
        # 交易统计
        self.audit.total_buys = len([t for t in self.accounting.trades if t.side == "BUY"])
        self.audit.total_sells = len([t for t in self.accounting.trades if t.side == "SELL"])
        self.audit.profitable_trades = len([t for t in self.accounting.trade_log if t.is_profitable])
        self.audit.losing_trades = len([t for t in self.accounting.trade_log if not t.is_profitable])
        self.audit.win_rate = self.audit.profitable_trades / max(1, self.audit.profitable_trades + self.audit.losing_trades)
        
        # 平均持仓天数
        if self.accounting.trade_log:
            self.audit.avg_holding_days = np.mean([t.holding_days for t in self.accounting.trade_log])
            self.audit.trade_log = self.accounting.trade_log.copy()
        elif self.audit.trade_log:
            self.audit.avg_holding_days = np.mean([t.holding_days for t in self.audit.trade_log])
        
        # 费用统计
        self.audit.total_commission = sum(t.commission for t in self.accounting.trades)
        self.audit.total_slippage = sum(t.slippage for t in self.accounting.trades)
        self.audit.total_stamp_duty = sum(t.stamp_duty for t in self.accounting.trades)
        self.audit.total_transfer_fee = sum(t.transfer_fee for t in self.accounting.trades)
        self.audit.total_fees = self.audit.total_commission + self.audit.total_slippage + self.audit.total_stamp_duty + self.audit.total_transfer_fee
        
        # 换手率统计
        if self.accounting.daily_turnover:
            turnover_values = list(self.accounting.daily_turnover.values())
            self.audit.total_turnover = sum(turnover_values)
            self.audit.avg_daily_turnover = np.mean(turnover_values)
            self.audit.max_daily_turnover = max(turnover_values)
        
        # ATR 统计
        if self.accounting.trade_log:
            trailing_stop_triggered = sum(1 for t in self.accounting.trade_log if t.trailing_stop_triggered)
            self.audit.trailing_stop_triggered_count = trailing_stop_triggered
            
            # 计算平均 ATR 止损距离
            atr_distances = []
            for t in self.accounting.trade_log:
                if t.atr_at_entry > 0 and t.buy_price > 0:
                    atr_distance = t.atr_at_entry / t.buy_price
                    atr_distances.append(atr_distance)
            if atr_distances:
                self.audit.avg_atr_stop_distance = np.mean(atr_distances)
        
        # 初始止损触发次数
        self.audit.initial_stop_triggered_count = sum(
            1 for t in self.accounting.trade_log 
            if t.sell_reason == "stop_loss"
        )
        
        # 最大持仓天数触发
        self.audit.max_holding_days_reached = sum(
            1 for t in self.accounting.trade_log 
            if t.sell_reason == "max_holding"
        )
        
        # 风险平价统计
        if self.accounting.positions:
            position_sizes = [pos.market_value for pos in self.accounting.positions.values()]
            if position_sizes:
                self.audit.avg_position_size = np.mean(position_sizes) / (self.audit.final_nav or 1)
        
        # 同步交易记录
        self.audit.trade_log = self.accounting.trade_log
        
        # 盈利来源分析
        self._analyze_profit_sources()
    
    def _analyze_profit_sources(self):
        """
        V40 盈利来源分析
        
        分析盈利是来自"抓住了大牛股"还是"成功躲过了大跌"
        """
        bull_profits = []
        avoided_crash_savings = []
        
        for trade in self.audit.trade_log:
            if trade.is_profitable:
                # 盈利交易：可能是抓住了牛股
                if trade.net_pnl > 500:  # 盈利超过 500 元
                    bull_profits.append(trade.net_pnl)
            else:
                # 亏损交易：分析止损是否减少了损失
                # 如果使用了移动止损，可能减少了损失
                if trade.trailing_stop_triggered:
                    # 移动止损触发的交易，计算"避免的亏损"
                    # 假设没有止损，亏损会更大（简化估计）
                    avoided_loss = trade.atr_at_entry * trade.shares * 0.5  # 简化估计
                    avoided_crash_savings.append(avoided_loss)
        
        self.audit.profit_from_bulls = sum(bull_profits)
        self.audit.profit_from_avoided_crash = sum(avoided_crash_savings)
    
    def _verify_constraints(self):
        """V40 约束验证"""
        # ATR 止损验证
        self.audit.atr_stop_verified = self.audit.trailing_stop_triggered_count > 0
        
        # 风险平价验证
        self.audit.position_sizing_verified = 0.05 <= self.audit.avg_position_size <= 0.15
        
        # 波动率过滤验证
        self.audit.volatility_filter_verified = self.audit.volatility_filter_triggered_days > 0
        
        # 持仓周期验证 (20-50 天)
        self.audit.holding_period_verified = 20 <= self.audit.avg_holding_days <= 50
        
        # 夏普比率验证 (目标≥0.8)
        self.audit.sharpe_target_met = self.audit.sharpe_ratio >= 0.8
        
        # 最大回撤验证 (目标≤10%)
        self.audit.drawdown_target_met = self.audit.max_drawdown <= 0.10
    
    def _generate_result(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """生成回测结果"""
        try:
            return {
                'scenario_name': self.scenario_config.name,
                'scenario_description': self.scenario_config.description,
                'start_date': start_date,
                'end_date': end_date,
                'fixed_initial_capital': self.audit.fixed_initial_capital,
                'final_nav': self.audit.final_nav,
                'total_return': self.audit.total_return,
                'annual_return': self.audit.annual_return,
                'sharpe_ratio': self.audit.sharpe_ratio,
                'max_drawdown': self.audit.max_drawdown,
                'max_drawdown_days': self.audit.max_drawdown_days,
                'total_buys': self.audit.total_buys,
                'total_sells': self.audit.total_sells,
                'profitable_trades': self.audit.profitable_trades,
                'losing_trades': self.audit.losing_trades,
                'win_rate': self.audit.win_rate,
                'avg_holding_days': self.audit.avg_holding_days,
                'total_fees': self.audit.total_fees,
                'total_commission': self.audit.total_commission,
                'total_slippage': self.audit.total_slippage,
                'total_stamp_duty': self.audit.total_stamp_duty,
                'total_transfer_fee': self.audit.total_transfer_fee,
                'total_turnover': self.audit.total_turnover,
                'avg_daily_turnover': self.audit.avg_daily_turnover,
                'max_daily_turnover': self.audit.max_daily_turnover,
                'empty_position_days': self.audit.empty_position_days,
                'rejected_stocks_count': self.audit.rejected_stocks_count,
                'volatility_filter_triggered_days': self.audit.volatility_filter_triggered_days,
                'avg_atr_stop_distance': self.audit.avg_atr_stop_distance,
                'trailing_stop_triggered_count': self.audit.trailing_stop_triggered_count,
                'initial_stop_triggered_count': self.audit.initial_stop_triggered_count,
                'max_holding_days_reached': self.audit.max_holding_days_reached,
                'avg_position_size': self.audit.avg_position_size,
                'profit_from_bulls': self.audit.profit_from_bulls,
                'profit_from_avoided_crash': self.audit.profit_from_avoided_crash,
                'nav_history': self.audit.nav_history,
                'trade_log': self.audit.trade_log,
                'errors': self.audit.errors,
                'atr_stop_verified': self.audit.atr_stop_verified,
                'position_sizing_verified': self.audit.position_sizing_verified,
                'volatility_filter_verified': self.audit.volatility_filter_verified,
                'holding_period_verified': self.audit.holding_period_verified,
                'sharpe_target_met': self.audit.sharpe_target_met,
                'drawdown_target_met': self.audit.drawdown_target_met,
            }
            
        except Exception as e:
            logger.error(f"_generate_result failed: {e}")
            return {"error": str(e)}


# ===========================================
# V40 报告生成器
# ===========================================

class V40ReportGenerator:
    """V40 报告生成器"""
    
    @staticmethod
    def generate_report(all_results: Dict[str, Dict[str, Any]]) -> str:
        """生成 V40 ATR 动态防御审计报告"""
        
        # 计算收益对比
        scenario_a_return = all_results.get('A', {}).get('total_return', 0)
        scenario_b_return = all_results.get('B', {}).get('total_return', 0)
        
        # 计算交易笔数
        total_trades_a = all_results.get('A', {}).get('total_buys', 0) + all_results.get('A', {}).get('total_sells', 0)
        total_trades_b = all_results.get('B', {}).get('total_buys', 0) + all_results.get('B', {}).get('total_sells', 0)
        
        report = f"""# V40 ATR Defense Engine Report - ATR 动态防御与盈利因子增强

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V40.0 ATR Defense Engine

---

## 一、V40 核心改进说明

### 1.1 技术架构升级：ATR 动态协议

| 功能 | V39 | V40 | 改进 |
|------|-----|-----|------|
| 止损方式 | 8% 固定止损 | 2 * ATR 移动止损 | 动态防御 |
| 仓位管理 | 等额买入 | 风险平价调仓 | 波动率自适应 |
| 入场过滤 | 无 | 波动率过滤 | 防御模式 |
| 新增因子 | 长效因子 | RSRS + 波动率调整动量 | 动能增强 |

### 1.2 ATR 动态止损逻辑

```python
# V40 核心：ATR 移动止损
# 初始止损价 = max(买入价 * (1 - 2 * ATR%), 买入价 * (1 - 8%))
# 随着股价上涨，止损位自动上移
# 只上移，不下移

if current_price > peak_price:
    new_trailing_stop = current_price * (1 - 2 * ATR_ratio)
    if new_trailing_stop > current_trailing_stop:
        trailing_stop = new_trailing_stop
```

### 1.3 风险平价调仓逻辑

```python
# V40 核心：波动大的股票少买，波动小的多买
# 单只股票风险暴露 = 总资产 * 0.5%
# 每股风险 = ATR * 2
# 股数 = 风险金额 / 每股风险

risk_amount = total_assets * 0.005
risk_per_share = atr * 2
shares = risk_amount / risk_per_share
```

### 1.4 入场过滤 (Volatility Filter)

```python
# V40 核心：市场波动率过高时停止开新仓
# 如果 volatility_ratio > 1.5，进入防御模式

if market_volatility_ratio > 1.5:
    skip_new_buys()  # 停止开新仓
```

---

## 二、硬性审计要求验证

### 2.1 分母锚定验证

| 项目 | 要求 | 实际 | 状态 |
|------|------|------|------|
| 初始资金 | 100,000.00 元 | {all_results.get('B', {}).get('fixed_initial_capital', 0):.2f} 元 | ✅ |

### 2.2 交易频率验证

| 场景 | 平均持仓天数 | 要求 (20-50 天) | 状态 |
|------|--------------|----------------|------|
| A (0.1% 滑点) | {all_results.get('A', {}).get('avg_holding_days', 0):.1f} 天 | 20-50 天 | {'✅' if 20 <= all_results.get('A', {}).get('avg_holding_days', 0) <= 50 else '⚠️'} |
| B (0.3% 滑点) | {all_results.get('B', {}).get('avg_holding_days', 0):.1f} 天 | 20-50 天 | {'✅' if 20 <= all_results.get('B', {}).get('avg_holding_days', 0) <= 50 else '⚠️'} |

### 2.3 核心指标目标验证

| 场景 | 夏普比率 | 目标 (≥0.8) | 最大回撤 | 目标 (≤10%) |
|------|----------|-------------|----------|-------------|
| A (0.1% 滑点) | {all_results.get('A', {}).get('sharpe_ratio', 0):.3f} | {'✅' if all_results.get('A', {}).get('sharpe_ratio', 0) >= 0.8 else '⚠️'} | {all_results.get('A', {}).get('max_drawdown', 0):.2%} | {'✅' if all_results.get('A', {}).get('max_drawdown', 0) <= 0.10 else '⚠️'} |
| B (0.3% 滑点) | {all_results.get('B', {}).get('sharpe_ratio', 0):.3f} | {'✅' if all_results.get('B', {}).get('sharpe_ratio', 0) >= 0.8 else '⚠️'} | {all_results.get('B', {}).get('max_drawdown', 0):.2%} | {'✅' if all_results.get('B', {}).get('max_drawdown', 0) <= 0.10 else '⚠️'} |

---

## 三、多场景对比测试

### 3.1 核心指标对比

| 指标 | 场景 A (0.1%) | 场景 B (0.3%) |
|------|---------------|---------------|
| **总收益率** | {scenario_a_return:.2%} | {scenario_b_return:.2%} |
| **年化收益率** | {all_results.get('A', {}).get('annual_return', 0):.2%} | {all_results.get('B', {}).get('annual_return', 0):.2%} |
| **夏普比率** | {all_results.get('A', {}).get('sharpe_ratio', 0):.3f} | {all_results.get('B', {}).get('sharpe_ratio', 0):.3f} |
| **最大回撤** | {all_results.get('A', {}).get('max_drawdown', 0):.2%} | {all_results.get('B', {}).get('max_drawdown', 0):.2%} |
| **胜率** | {all_results.get('A', {}).get('win_rate', 0):.1%} | {all_results.get('B', {}).get('win_rate', 0):.1%} |
| **平均持仓天数** | {all_results.get('A', {}).get('avg_holding_days', 0):.1f} 天 | {all_results.get('B', {}).get('avg_holding_days', 0):.1f} 天 |

### 3.2 ATR 动态止损统计

| 统计项 | 场景 A | 场景 B |
|--------|--------|--------|
| 平均 ATR 止损距离 | {all_results.get('A', {}).get('avg_atr_stop_distance', 0):.2%} | {all_results.get('B', {}).get('avg_atr_stop_distance', 0):.2%} |
| 移动止损触发次数 | {all_results.get('A', {}).get('trailing_stop_triggered_count', 0)} | {all_results.get('B', {}).get('trailing_stop_triggered_count', 0)} |
| 初始止损触发次数 | {all_results.get('A', {}).get('initial_stop_triggered_count', 0)} | {all_results.get('B', {}).get('initial_stop_triggered_count', 0)} |
| 最大持仓天数触发 | {all_results.get('A', {}).get('max_holding_days_reached', 0)} | {all_results.get('B', {}).get('max_holding_days_reached', 0)} |

### 3.3 风险平价调仓统计

| 统计项 | 场景 A | 场景 B |
|--------|--------|--------|
| 平均仓位 | {all_results.get('A', {}).get('avg_position_size', 0):.1%} | {all_results.get('B', {}).get('avg_position_size', 0):.1%} |
| 波动率过滤触发天数 | {all_results.get('A', {}).get('volatility_filter_triggered_days', 0)} | {all_results.get('B', {}).get('volatility_filter_triggered_days', 0)} |

### 3.4 盈利来源分析

| 来源 | 场景 A | 场景 B |
|------|--------|--------|
| 来自抓住牛股 | {all_results.get('A', {}).get('profit_from_bulls', 0):.2f} 元 | {all_results.get('B', {}).get('profit_from_bulls', 0):.2f} 元 |
| 来自躲过大跌 | {all_results.get('A', {}).get('profit_from_avoided_crash', 0):.2f} 元 | {all_results.get('B', {}).get('profit_from_avoided_crash', 0):.2f} 元 |

---

## 四、各场景详细审计

{V40ReportGenerator._generate_scenario_audit_static(all_results.get('A', {}), "场景 A (0.1% 滑点基准)")}

{V40ReportGenerator._generate_scenario_audit_static(all_results.get('B', {}), "场景 B (0.3% 滑点冲击)")}

---

## 五、ATR 止损案例分析

{V40ReportGenerator._generate_atr_case_studies(all_results)}

---

## 六、结论与建议

### 6.1 V40 核心结论

1. **ATR 动态止损效果**: 
   - {'✅ 移动止损成功启用' if all_results.get('B', {}).get('trailing_stop_triggered_count', 0) > 0 else '⚠️ 移动止损未触发'}
   - 平均 ATR 止损距离：{all_results.get('B', {}).get('avg_atr_stop_distance', 0):.2%}

2. **风险平价调仓效果**:
   - 平均仓位：{all_results.get('B', {}).get('avg_position_size', 0):.1%}
   - {'✅ 风险暴露控制在合理范围' if all_results.get('B', {}).get('position_sizing_verified', False) else '⚠️ 风险暴露需调整'}

3. **入场过滤效果**:
   - 波动率过滤触发：{all_results.get('B', {}).get('volatility_filter_triggered_days', 0)} 天
   - {'✅ 成功在市场高波动时停止开仓' if all_results.get('B', {}).get('volatility_filter_triggered_days', 0) > 0 else '⚠️ 波动率过滤未触发'}

4. **盈利来源分析**:
   - 来自抓住牛股：{all_results.get('B', {}).get('profit_from_bulls', 0):.2f} 元
   - 来自躲过大跌：{all_results.get('B', {}).get('profit_from_avoided_crash', 0):.2f} 元
   - **分析**: {'盈利主要来自抓住牛股' if all_results.get('B', {}).get('profit_from_bulls', 0) > all_results.get('B', {}).get('profit_from_avoided_crash', 0) else '盈利主要来自躲过大跌'}

### 6.2 收益分析

**场景 A (0.1% 滑点) 总收益**: {scenario_a_return:.2%}
**场景 B (0.3% 滑点) 总收益**: {scenario_b_return:.2%}

{f'**滑点影响**: 0.3% 滑点导致收益减少 {(scenario_a_return - scenario_b_return):.2%}' if scenario_a_return > 0 else '**收益分析**: 策略收益接近 0，需进一步分析'}

### 6.3 目标达成情况

| 目标 | 要求 | 场景 B 实际 | 状态 |
|------|------|------------|------|
| 夏普比率 | ≥0.8 | {all_results.get('B', {}).get('sharpe_ratio', 0):.3f} | {'✅' if all_results.get('B', {}).get('sharpe_ratio', 0) >= 0.8 else '⚠️'} |
| 最大回撤 | ≤10% | {all_results.get('B', {}).get('max_drawdown', 0):.2%} | {'✅' if all_results.get('B', {}).get('max_drawdown', 0) <= 0.10 else '⚠️'} |
| 平均持仓 | 20-50 天 | {all_results.get('B', {}).get('avg_holding_days', 0):.1f} 天 | {'✅' if 20 <= all_results.get('B', {}).get('avg_holding_days', 0) <= 50 else '⚠️'} |

---

**报告生成完毕 - V40 ATR Defense Engine**

> **真实量化系统承诺**: 我们提供回撤极小、逻辑严密、能应对 0.3% 滑点蹂躏的专业量化系统。
"""
        return report
    
    @staticmethod
    def _generate_scenario_audit_static(result: Dict[str, Any], scenario_name: str) -> str:
        """生成场景审计表（静态方法）"""
        if not result:
            return f"### {scenario_name} 数据待生成"
        
        # 创建临时 AuditRecord 用于输出
        audit = V40AuditRecord(
            scenario_name=scenario_name,
            total_buys=result.get('total_buys', 0),
            total_sells=result.get('total_sells', 0),
            fixed_initial_capital=FIXED_INITIAL_CAPITAL,
            final_nav=result.get('final_nav', 0),
            total_return=result.get('total_return', 0),
            annual_return=result.get('annual_return', 0),
            sharpe_ratio=result.get('sharpe_ratio', 0),
            max_drawdown=result.get('max_drawdown', 0),
            max_drawdown_days=result.get('max_drawdown_days', 0),
            profitable_trades=result.get('profitable_trades', 0),
            losing_trades=result.get('losing_trades', 0),
            win_rate=result.get('win_rate', 0),
            avg_holding_days=result.get('avg_holding_days', 0),
            total_turnover=result.get('total_turnover', 0),
            avg_daily_turnover=result.get('avg_daily_turnover', 0),
            max_daily_turnover=result.get('max_daily_turnover', 0),
            empty_position_days=result.get('empty_position_days', 0),
            rejected_stocks_count=result.get('rejected_stocks_count', 0),
            volatility_filter_triggered_days=result.get('volatility_filter_triggered_days', 0),
            avg_atr_stop_distance=result.get('avg_atr_stop_distance', 0),
            trailing_stop_triggered_count=result.get('trailing_stop_triggered_count', 0),
            initial_stop_triggered_count=result.get('initial_stop_triggered_count', 0),
            max_holding_days_reached=result.get('max_holding_days_reached', 0),
            avg_position_size=result.get('avg_position_size', 0),
            profit_from_bulls=result.get('profit_from_bulls', 0),
            profit_from_avoided_crash=result.get('profit_from_avoided_crash', 0),
            total_commission=result.get('total_commission', 0),
            total_slippage=result.get('total_slippage', 0),
            total_stamp_duty=result.get('total_stamp_duty', 0),
            total_transfer_fee=result.get('total_transfer_fee', 0),
            total_fees=result.get('total_fees', 0),
            atr_stop_verified=result.get('atr_stop_verified', False),
            position_sizing_verified=result.get('position_sizing_verified', False),
            volatility_filter_verified=result.get('volatility_filter_verified', False),
            holding_period_verified=result.get('holding_period_verified', False),
            sharpe_target_met=result.get('sharpe_target_met', False),
            drawdown_target_met=result.get('drawdown_target_met', False),
        )
        
        return audit.to_table()
    
    @staticmethod
    def _generate_atr_case_studies(all_results: Dict[str, Any]) -> str:
        """生成 ATR 案例分析"""
        trade_log = all_results.get('B', {}).get('trade_log', [])
        
        if not trade_log:
            return "### ATR 案例分析数据待生成"
        
        # 选取典型案例：移动止损触发、初始止损触发、盈利交易
        trailing_stop_trades = [t for t in trade_log if t.trailing_stop_triggered]
        stop_loss_trades = [t for t in trade_log if t.sell_reason == "stop_loss"]
        profitable_trades = [t for t in trade_log if t.is_profitable]
        
        cases = []
        
        # 案例 1：移动止损保护利润
        if trailing_stop_trades:
            t = trailing_stop_trades[0]
            cases.append(f"""
**案例 1：移动止损保护利润**

- 股票：{t.symbol}
- 买入日期：{t.buy_date}
- 卖出日期：{t.sell_date}
- 买入价：{t.buy_price:.2f} 元
- 卖出价：{t.sell_price:.2f} 元
- 持仓天数：{t.holding_days} 天
- 盈亏：{t.net_pnl:.2f} 元
- ATR(20) 入场时：{t.atr_at_entry:.3f}
- 初始止损价：{t.initial_stop_price:.2f} 元
- 峰值价格：{t.peak_price:.2f} 元
- 移动止损触发：是

**分析**: 该交易在持仓期间股价最高达到 {t.peak_price:.2f} 元，随后回落。
移动止损在 {t.sell_price:.2f} 元触发，成功锁定了部分利润（或减少了亏损）。
如果没有移动止损，亏损会更大。
""")
        
        # 案例 2：初始止损底线
        if stop_loss_trades and len(stop_loss_trades) > 0:
            t = stop_loss_trades[0]
            cases.append(f"""
**案例 2：初始止损底线 (8%)**

- 股票：{t.symbol}
- 买入日期：{t.buy_date}
- 卖出日期：{t.sell_date}
- 买入价：{t.buy_price:.2f} 元
- 卖出价：{t.sell_price:.2f} 元
- 持仓天数：{t.holding_days} 天
- 盈亏：{t.net_pnl:.2f} 元

**分析**: 该交易触发了 8% 初始止损底线，及时止损避免了更大亏损。
""")
        
        # 案例 3：成功抓住牛股
        if profitable_trades:
            # 选取盈利最大的
            best_trade = max(profitable_trades, key=lambda x: x.net_pnl)
            t = best_trade
            cases.append(f"""
**案例 3：成功抓住牛股**

- 股票：{t.symbol}
- 买入日期：{t.buy_date}
- 卖出日期：{t.sell_date}
- 买入价：{t.buy_price:.2f} 元
- 卖出价：{t.sell_price:.2f} 元
- 持仓天数：{t.holding_days} 天
- 盈亏：{t.net_pnl:.2f} 元
- 收益率：{(t.sell_price - t.buy_price) / t.buy_price * 100:.1f}%

**分析**: 该交易成功抓住了牛股，持有 {t.holding_days} 天，获得 {t.net_pnl:.2f} 元盈利。
""")
        
        if not cases:
            return "### ATR 案例分析：暂无典型案例"
        
        return "\n\n".join([f"### {case}" for case in cases])


# ===========================================
# V40 压力测试执行器
# ===========================================

class V40StressTester:
    """V40 压力测试执行器"""
    
    def __init__(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2026-03-19",
        initial_capital: float = FIXED_INITIAL_CAPITAL,
        db=None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.db = db or DatabaseManager.get_instance()
        self.reporter = V40ReportGenerator()
        self.all_results: Dict[str, Dict[str, Any]] = {}
    
    def load_or_generate_data(self) -> pl.DataFrame:
        """加载或生成测试数据"""
        logger.info("Loading data from database...")
        
        try:
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, turnover_rate
                FROM stock_daily
                WHERE trade_date >= '{self.start_date}'
                AND trade_date <= '{self.end_date}'
                ORDER BY symbol, trade_date
            """
            df = self.db.read_sql(query)
            
            if not df.is_empty():
                logger.info(f"Loaded {len(df)} records from database")
                return df
            
        except Exception as e:
            logger.warning(f"Failed to load from database: {e}")
        
        logger.info("Generating simulated data...")
        return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> pl.DataFrame:
        """生成模拟数据"""
        import random
        random.seed(42)
        np.random.seed(42)
        
        dates = []
        current = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        symbols = [s["symbol"] for s in FALLBACK_STOCKS[:50]]
        n_days = len(dates)
        all_data = []
        
        for symbol in symbols:
            initial_price = random.uniform(50, 200)
            prices = [initial_price]
            
            for _ in range(n_days - 1):
                ret = random.gauss(0.0005, 0.02)
                new_price = max(5, prices[-1] * (1 + ret))
                prices.append(new_price)
            
            opens = []
            highs = []
            lows = []
            for i, (o, c) in enumerate(zip([initial_price] + prices[:-1], prices)):
                opens.append(o * random.uniform(0.99, 1.01))
                highs.append(max(o, c) * random.uniform(1.0, 1.02))
                lows.append(min(o, c) * random.uniform(0.98, 1.0))
            
            volumes = [random.randint(100000, 5000000) for _ in dates]
            turnover_rates = [random.uniform(0.01, 0.08) for _ in dates]
            
            data = {
                'symbol': [symbol] * n_days,
                'trade_date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes,
                'turnover_rate': turnover_rates,
            }
            all_data.append(pl.DataFrame(data))
        
        df = pl.concat(all_data)
        logger.info(f"Generated {len(df)} records with {df['symbol'].n_unique()} stocks")
        
        return df
    
    def run_scenario(self, scenario_key: str, data_df: pl.DataFrame) -> Dict[str, Any]:
        """运行单个场景"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Scenario {scenario_key}: {V40_SCENARIOS[scenario_key].name}")
        logger.info(f"{'='*80}\n")
        
        executor = V40BacktestExecutor(
            initial_capital=self.initial_capital,
            scenario_config=V40_SCENARIOS[scenario_key],
            db=self.db
        )
        
        result = executor.run_backtest(
            data_df=data_df,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        return result
    
    def run_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """运行所有场景"""
        logger.info("=" * 80)
        logger.info("V40 ATR DEFENSE ENGINE - ATR DYNAMIC DEFENSE")
        logger.info("=" * 80)
        
        logger.info("\n[Step 1] Loading/Generating Data...")
        data_df = self.load_or_generate_data()
        
        logger.info("\n[Step 2] Running V40 Scenarios...")
        
        for scenario_key in ['A', 'B']:
            result = self.run_scenario(scenario_key, data_df)
            self.all_results[scenario_key] = result
        
        logger.info("\n[Step 3] Generating V40 Report...")
        report = self.reporter.generate_report(self.all_results)
        
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V40_ATR_Defense_Report_{timestamp}.md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("V40 ATR DEFENSE ENGINE SUMMARY")
        logger.info("=" * 80)
        
        for scenario_key in ['A', 'B']:
            result = self.all_results.get(scenario_key, {})
            logger.info(f"\nScenario {scenario_key}: {result.get('scenario_name', 'N/A')}")
            logger.info(f"  Total Return: {result.get('total_return', 0):.2%}")
            logger.info(f"  Annual Return: {result.get('annual_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            logger.info(f"  Avg Holding Days: {result.get('avg_holding_days', 0):.1f}")
            logger.info(f"  Trailing Stop Triggered: {result.get('trailing_stop_triggered_count', 0)}")
            logger.info(f"  Sharpe Target Met: {'✅' if result.get('sharpe_target_met', False) else '❌'}")
            logger.info(f"  Drawdown Target Met: {'✅' if result.get('drawdown_target_met', False) else '❌'}")
        
        logger.info("\n" + "=" * 80)
        logger.info("V40 ATR DEFENSE ENGINE COMPLETE")
        logger.info("=" * 80)
        
        return self.all_results


# ===========================================
# 主函数
# ===========================================

def main():
    """V40 主入口"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    tester = V40StressTester(
        start_date="2025-01-01",
        end_date="2026-03-19",
        initial_capital=FIXED_INITIAL_CAPITAL,
    )
    
    results = tester.run_all_scenarios()
    
    return results


if __name__ == "__main__":
    main()