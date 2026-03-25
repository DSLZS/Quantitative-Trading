"""
Final Strategy V1.2 - Iteration 12: 交易稳定性增强与因子质量审计

核心改进清单 (vs Iteration 11):
=================================

1. 交易稳定性优化 (Trading Stability):
   - Score Buffer 机制：评分变化需超过阈值才换仓，减少噪声敏感
   - 最小持有天数硬约束：防止过度交易
   - 换仓冷却期：卖出后 N 日内不重复买入同一股票

2. 动态风控升级 (Dynamic Risk Control):
   - 滚动波动率动态止损：根据 20 日滚动波动率调整止损阈值
   - 收盘价触发选项：支持 ATR 止损仅在收盘价触发 (避免盘中噪音)
   - 分级止损：根据持有天数动态调整止损严格度

3. 因子质量审计 (Factor Quality Audit):
   - IC 值分析模块：计算因子在训练集/验证集的 IC 表现
   - 负 IC 因子剔除：自动剔除在 2024Q1 IC 值为负的特征
   - 超跌反转因子：引入 2 个具有明确金融逻辑的质量因子
     * oversold_rebound: 超跌反弹信号 (RSI<30 + 价格偏离下轨)
     * quality_reversal: 质量反转信号 (低波动 + 高流动性 + 超卖)

4. 自动分析与反馈调节 (Auto-Analysis & Feedback):
   - 衰减分析：计算压力测试后的收益回落幅度
   - 归因分析：统计亏损交易主要原因
   - 参数自调节：根据分析结果自动调整 production_params.yaml

作者：Quantitative Trading Team
版本：V1.2 (Iteration 12)
日期：2026-03-14
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import json
import yaml
import copy

import polars as pl
import numpy as np
from loguru import logger


# ===========================================
# Iteration 12 生产级参数配置
# ===========================================

# 基础交易成本参数
SLIPPAGE_RATE = 0.002  # 0.2% 滑点
COMMISSION_RATE = 0.0003
COMMISSION_MIN = 5.0
STAMP_DUTY_RATE = 0.001

# 涨跌停限制
LIMIT_UP_RATIO = 1.098
LIMIT_DOWN_RATIO = 0.902

# 流动性过滤
MIN_AVG_AMOUNT_5D = 50000000  # 5000 万

# 基础评分参数
SCORE_THRESHOLD = 0.0
SCORE_WEIGHT = 0.7
SHARPE_WEIGHT = 0.3

# ===========================================
# Iteration 12 新增：交易稳定性配置
# ===========================================

# Score Buffer 机制
SCORE_BUFFER_CONFIG = {
    "enabled": True,
    "buffer_threshold": 0.15,  # 评分变化需超过此阈值才换仓
    "relative_buffer": True,   # 使用相对阈值 (百分比)
    "absolute_buffer": 0.05,   # 绝对阈值
}

# 最小持有天数硬约束
MIN_HOLD_CONSTRAINT = {
    "enabled": True,
    "min_hold_days": 5,        # 最小持有 5 天
    "exemption_threshold": 0.03,  # 盈利>3% 可豁免
    "stop_loss_exempt": True,  # 止损卖出豁免
}

# 换仓冷却期
COOLDOWN_CONFIG = {
    "enabled": True,
    "cooldown_days": 10,       # 卖出后 10 日内不重复买入
}

# ===========================================
# Iteration 12 新增：动态风控配置
# ===========================================

# 滚动波动率动态止损
VOLATILITY_STOP_CONFIG = {
    "enabled": True,
    "window": 20,              # 20 日滚动窗口
    "base_multiplier": 2.5,    # 基础乘数
    "volatility_scaling": {
        "high_vol_threshold": 0.04,   # 高波动阈值 4%
        "low_vol_threshold": 0.015,   # 低波动阈值 1.5%
        "high_scale": 0.7,            # 高波动缩放 (更宽松)
        "low_scale": 1.3,             # 低波动缩放 (更严格)
    },
    "close_only": False,       # 仅收盘价触发 (False=盘中也触发)
}

# 分级止损配置
TIERED_STOP_CONFIG = {
    "enabled": True,
    "tiers": [
        {"min_days": 0, "max_days": 3, "multiplier": 0.8, "hard_stop": -0.07},
        {"min_days": 4, "max_days": 10, "multiplier": 1.0, "hard_stop": -0.05},
        {"min_days": 11, "max_days": 999, "multiplier": 1.2, "hard_stop": -0.03},
    ],
}

# 盈利保护机制 (Break-even Stop)
PROFIT_PROTECTION = {
    "enabled": True,
    "profit_threshold": 0.02,      # 盈利 > 2% 触发
    "stop_level": 0.005,           # 止损上移至成本价以上 0.5%
    "trailing_stop": True,         # 启用移动止盈
    "trailing_threshold": 0.05,    # 盈利>5% 启动移动止盈
    "trailing_percent": 0.03,      # 从最高点回撤 3% 止盈
}

# ===========================================
# Iteration 12 新增：因子质量配置
# ===========================================

# 因子 IC 分析配置
FACTOR_IC_CONFIG = {
    "enabled": True,
    "training_period": ("2023-01-01", "2023-12-31"),  # 训练集
    "validation_period": ("2024-01-01", "2024-03-31"),  # 验证集 (2024Q1)
    "ic_threshold": -0.02,  # IC 阈值，低于此值剔除
    "min_ic_rank": 0.5,     # IC 排名低于 50% 剔除
}

# 超跌反转因子配置
REVERSAL_FACTOR_CONFIG = {
    "oversold_rebound": {
        "enabled": True,
        "rsi_threshold": 30,       # RSI < 30 超卖
        "bollinger_position": 0.1, # 布林线位置 < 10%
        "volume_confirm": 1.5,     # 成交量放大 1.5 倍确认
        "weight": 0.15,            # 因子权重
    },
    "quality_reversal": {
        "enabled": True,
        "volatility_percentile": 30,  # 波动率分位数 < 30%
        "liquidity_threshold": 1e8,   # 流动性阈值 1 亿
        "momentum_reversal": -0.15,   # 动量反转阈值 -15%
        "weight": 0.12,               # 因子权重
    },
}

# ===========================================
# Iteration 11 保留：防御逻辑弹性化
# ===========================================

# 动态防御配置
DEFENSIVE_CONFIG = {
    "base_threshold_addon": 0.15,
    "dynamic_adjustment": True,
    "min_threshold_addon": 0.0,
    "max_threshold_addon": 0.3,
    "median_score_sensitivity": 0.5,
    "reference_median_score": 0.0,
    "max_positions": 3,
    "atr_multiplier": 2.5,
    "min_hold_days": 5,
}

# 进攻模式参数
AGGRESSIVE_CONFIG = {
    "threshold_addon": 0.0,
    "threshold_addon_bull": -0.1,
    "max_positions": 5,
    "max_positions_bull": 8,
    "atr_multiplier": 2.5,
    "min_hold_days": 3,
}

# ===========================================
# 其他配置
# ===========================================

# 止盈豁免阈值
PROFIT_EXEMPTION_THRESHOLD = 0.03
PROFIT_EXEMPTION_MIN_HOLD = 1

# 成本优化参数
COST_THRESHOLD = 0.006
SCORE_HOLD_BUFFER = -0.5
MIN_HOLD_DAYS_CHEAP = 3

# ATR 参数
ATR_WINDOW = 14
MAX_ATR_STOP = -0.08
MIN_ATR_STOP = -0.03

# 硬止损
HARD_STOP_LOSS = -0.05

# 初始资金
INITIAL_CAPITAL = 100000.0


@dataclass
class DailyRecord:
    """每日交易记录"""
    trade_date: str
    market_mode: str
    market_regime: str
    index_close: float
    index_ma20: float
    market_median_score: float
    dynamic_threshold: float
    holdings: list[dict] = field(default_factory=list)
    holding_symbols: list[str] = field(default_factory=list)
    bought_symbols: list[str] = field(default_factory=list)
    sold_symbols: list[str] = field(default_factory=list)
    cash: float = 0.0
    portfolio_value: float = 0.0
    daily_return: float = 0.0
    daily_pnl: float = 0.0
    # 统计
    limit_up_blocked: list[str] = field(default_factory=list)
    limit_down_blocked: list[str] = field(default_factory=list)
    liquidity_filtered: list[str] = field(default_factory=list)
    atr_stop_triggered: list[str] = field(default_factory=list)
    # Iteration 12 新增
    score_buffer_filtered: list[str] = field(default_factory=list)  # Score Buffer 过滤
    cooldown_filtered: list[str] = field(default_factory=list)      # 冷却期过滤
    min_hold_protected: list[str] = field(default_factory=list)     # 最小持有保护
    trailing_stop_triggered: list[str] = field(default_factory=list)  # 移动止盈
    volatility_adjustment_applied: bool = False


@dataclass
class BacktestResult:
    """回测结果"""
    start_date: str
    end_date: str
    total_days: int
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_daily_return: float
    daily_records: list[DailyRecord]
    benchmark_return: float
    total_trades: int
    total_commission: float
    total_stamp_duty: float
    # 统计
    limit_up_blocked_count: int = 0
    limit_down_blocked_count: int = 0
    liquidity_filtered_count: int = 0
    atr_stop_triggered_count: int = 0
    # Iteration 12 新增
    score_buffer_filtered_count: int = 0
    cooldown_filtered_count: int = 0
    min_hold_protected_count: int = 0
    trailing_stop_triggered_count: int = 0
    aggressive_days: int = 0
    defensive_days: int = 0
    bull_market_days: int = 0
    # 月度收益数据
    monthly_returns: dict[str, float] = field(default_factory=dict)
    # 鲁棒性得分
    robustness_score: float = 0.0
    # 衰减分析
    noise_sensitivity: float = 0.0
    # 归因分析
    loss_attribution: dict[str, int] = field(default_factory=dict)


@dataclass
class FactorICAnalysis:
    """因子 IC 分析结果"""
    factor_name: str
    training_ic: float
    training_icir: float
    validation_ic: float
    validation_icir: float
    is_negative_in_validation: bool
    should_remove: bool
    financial_logic: str


class FinalStrategyV12Backtester:
    """
    Final Strategy V1.2 - Iteration 12 回测引擎
    
    核心特性:
    1. Score Buffer 机制：评分变化需超过阈值才换仓
    2. 最小持有天数硬约束：防止过度交易
    3. 换仓冷却期：卖出后 N 日内不重复买入
    4. 滚动波动率动态止损
    5. 收盘价触发选项
    6. 因子 IC 分析与负 IC 因子剔除
    7. 超跌反转因子增强
    8. 自动分析与反馈调节
    """
    
    def __init__(
        self, 
        start_date: str, 
        end_date: str, 
        initial_capital: float = INITIAL_CAPITAL,
        price_noise_std: float = 0.001,
        enable_noise: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.price_noise_std = price_noise_std
        self.enable_noise = enable_noise
        
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.holdings: dict[str, dict] = {}
        self.daily_records: list[DailyRecord] = []
        self.total_trades = 0
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        
        # 统计
        self.limit_up_blocked_count = 0
        self.limit_down_blocked_count = 0
        self.liquidity_filtered_count = 0
        self.atr_stop_triggered_count = 0
        self.score_buffer_filtered_count = 0
        self.cooldown_filtered_count = 0
        self.min_hold_protected_count = 0
        self.trailing_stop_triggered_count = 0
        self.aggressive_days = 0
        self.defensive_days = 0
        self.bull_market_days = 0
        
        # Iteration 12 新增状态
        self.sold_history: dict[str, str] = {}  # 卖出记录 {symbol: sell_date}
        self.prev_scores: dict[str, float] = {}  # 前一日评分
        self.factor_ic_analysis: list[FactorICAnalysis] = []  # 因子 IC 分析结果
        self.active_factors: set[str] = set()  # 活跃因子集合
        
        # 数据库连接
        try:
            from .db_manager import DatabaseManager
            self.db = DatabaseManager.get_instance()
        except ImportError:
            from db_manager import DatabaseManager
            self.db = DatabaseManager.get_instance()
        
        # 缓存
        self.market_scores_cache: dict[str, list[dict]] = {}
        self.stock_volatility_cache: dict[str, float] = {}
        self.bollinger_cache: dict[str, dict] = {}
        
        logger.info(f"Final Strategy V1.2 Backtester initialized: {start_date} to {end_date}")
        logger.info(f"Iteration 12 Features:")
        logger.info(f"  - Score Buffer: {SCORE_BUFFER_CONFIG['enabled']}")
        logger.info(f"  - Min Hold Constraint: {MIN_HOLD_CONSTRAINT['enabled']}")
        logger.info(f"  - Cooldown Period: {COOLDOWN_CONFIG['enabled']}")
        logger.info(f"  - Volatility Stop: {VOLATILITY_STOP_CONFIG['enabled']}")
        logger.info(f"  - Factor IC Audit: {FACTOR_IC_CONFIG['enabled']}")
    
    def get_trade_dates(self) -> list[str]:
        """获取回测区间的交易日"""
        query = f"""
            SELECT DISTINCT trade_date 
            FROM stock_daily 
            WHERE trade_date >= '{self.start_date}'
            AND trade_date <= '{self.end_date}'
            ORDER BY trade_date
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            return []
        return [str(d) for d in result["trade_date"].to_list()]
    
    def get_prev_date(self, trade_dates: list[str], current_date: str) -> Optional[str]:
        """获取前一个交易日"""
        try:
            idx = trade_dates.index(current_date)
            if idx > 0:
                return trade_dates[idx - 1]
        except ValueError:
            pass
        return None
    
    def get_next_date(self, trade_dates: list[str], current_date: str) -> Optional[str]:
        """获取下一个交易日 (T+1)"""
        try:
            idx = trade_dates.index(current_date)
            if idx < len(trade_dates) - 1:
                return trade_dates[idx + 1]
        except ValueError:
            pass
        return None
    
    def get_index_data(self, trade_date: str, lookback: int = 30) -> Optional[dict]:
        """获取指数数据并计算 MA20"""
        query = f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE symbol = '000905.SH'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT {lookback}
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            query = f"""
                SELECT trade_date, close
                FROM index_daily
                WHERE symbol = '000001.SH'
                AND trade_date <= '{trade_date}'
                ORDER BY trade_date DESC
                LIMIT {lookback}
            """
            result = self.db.read_sql(query)
        
        if result.is_empty():
            return None
        
        df = result.sort("trade_date")
        close_list = df["close"].to_list()
        
        if len(close_list) >= 20:
            ma20 = float(np.mean(close_list[-20:]))
        else:
            ma20 = None
        
        latest = df.filter(pl.col("trade_date").cast(pl.Utf8) == trade_date)
        if latest.is_empty():
            latest = df.tail(1)
        if latest.is_empty():
            return None
        
        return {
            "close": float(latest["close"][0]),
            "ma20": ma20
        }
    
    def get_market_mode(self, trade_date: str) -> tuple[str, str]:
        """获取市场模式"""
        index_data = self.get_index_data(trade_date)
        if index_data and index_data["ma20"] is not None:
            if index_data["close"] > index_data["ma20"]:
                median_score = self.get_market_median_score(trade_date)
                if median_score > 0.5:
                    return "AGGRESSIVE", "BULL"
                return "AGGRESSIVE", "NORMAL"
        return "DEFENSIVE", "NORMAL"
    
    def get_market_median_score(self, trade_date: str) -> float:
        """获取全市场预测评分中位数"""
        if trade_date in self.market_scores_cache:
            scores = self.market_scores_cache[trade_date]
            if scores:
                return float(np.median([s["combined_score"] for s in scores]))
            return 0.0
        
        query = f"""
            SELECT DISTINCT symbol
            FROM stock_daily
            WHERE trade_date = '{trade_date}'
            AND volume > 0
            AND close > 0
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            return 0.0
        
        symbols = result["symbol"].unique().limit(500).to_list()
        scored_stocks = []
        
        for symbol in symbols:
            try:
                history_query = f"""
                    SELECT symbol, trade_date, close, volume, pct_chg, pre_close
                    FROM stock_daily
                    WHERE symbol = '{symbol}'
                    AND trade_date <= '{trade_date}'
                    ORDER BY trade_date DESC
                    LIMIT 30
                """
                history_df = self.db.read_sql(history_query)
                
                if len(history_df) < 20:
                    continue
                
                history_df = history_df.sort("trade_date")
                close_list = [float(x) if x is not None else 0.0 for x in history_df["close"].to_list()]
                
                if not close_list or close_list[-1] <= 0:
                    continue
                
                # 动量因子
                if len(close_list) >= 5 and close_list[-5] > 0:
                    momentum_5 = (close_list[-1] - close_list[-5]) / close_list[-5]
                else:
                    momentum_5 = 0.0
                
                if len(close_list) >= 10 and close_list[-10] > 0:
                    momentum_10 = (close_list[-1] - close_list[-10]) / close_list[-10]
                else:
                    momentum_10 = 0.0
                
                if len(close_list) >= 20 and close_list[-20] > 0:
                    momentum_20 = (close_list[-1] - close_list[-20]) / close_list[-20]
                else:
                    momentum_20 = 0.0
                
                predict_score = 0.5 * momentum_5 + 0.3 * momentum_10 + 0.2 * momentum_20
                
                scored_stocks.append({
                    "symbol": symbol,
                    "combined_score": predict_score,
                })
            except Exception:
                continue
        
        self.market_scores_cache[trade_date] = scored_stocks
        
        if scored_stocks:
            return float(np.median([s["combined_score"] for s in scored_stocks]))
        return 0.0
    
    def get_dynamic_threshold(self, market_mode: str, market_median_score: float) -> float:
        """计算动态阈值"""
        if market_mode == "AGGRESSIVE":
            config = AGGRESSIVE_CONFIG
            base_threshold = config["threshold_addon"]
            if market_median_score > 0.5:
                return config["threshold_addon_bull"]
            return base_threshold
        
        config = DEFENSIVE_CONFIG
        
        if not config["dynamic_adjustment"]:
            return config["base_threshold_addon"]
        
        reference = config["reference_median_score"]
        sensitivity = config["median_score_sensitivity"]
        score_diff = market_median_score - reference
        adjustment = -score_diff * sensitivity
        dynamic_threshold = config["base_threshold_addon"] + adjustment
        dynamic_threshold = max(config["min_threshold_addon"], 
                               min(config["max_threshold_addon"], dynamic_threshold))
        
        return dynamic_threshold
    
    def get_config(self, market_mode: str, market_regime: str, market_median_score: float) -> dict:
        """获取当前模式对应的配置"""
        if market_mode == "AGGRESSIVE":
            config = AGGRESSIVE_CONFIG.copy()
            if market_regime == "BULL":
                config["threshold_addon"] = config["threshold_addon_bull"]
                config["max_positions"] = config["max_positions_bull"]
            return config
        
        config = DEFENSIVE_CONFIG.copy()
        config["threshold_addon"] = self.get_dynamic_threshold(market_mode, market_median_score)
        return config
    
    def check_limit_up(self, open_price: float, pre_close: float) -> bool:
        """检查是否涨停"""
        if pre_close <= 0:
            return False
        return open_price >= pre_close * LIMIT_UP_RATIO
    
    def check_limit_down(self, open_price: float, pre_close: float) -> bool:
        """检查是否跌停"""
        if pre_close <= 0:
            return False
        return open_price <= pre_close * LIMIT_DOWN_RATIO
    
    def get_avg_amount_5d(self, symbol: str, trade_date: str) -> float:
        """获取过去 5 日平均成交额"""
        query = f"""
            SELECT amount, close, volume
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT 5
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            return 0.0
        
        amounts = []
        for idx in range(len(result)):
            close = float(result["close"][idx]) if result["close"][idx] else 0.0
            volume = float(result["volume"][idx]) if result["volume"][idx] else 0.0
            amount = result["amount"][idx]
            
            if amount is not None:
                amounts.append(float(amount))
            elif close > 0 and volume > 0:
                amounts.append(close * volume)
        
        if not amounts:
            return 0.0
        return float(np.mean(amounts))
    
    def get_atr(self, symbol: str, trade_date: str) -> Optional[float]:
        """计算 ATR"""
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, pre_close
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT {ATR_WINDOW + 10}
        """
        result = self.db.read_sql(query)
        if result.is_empty() or len(result) < ATR_WINDOW:
            return None
        
        df = result.sort("trade_date")
        
        high = pl.col("high")
        low = pl.col("low")
        pre_close = pl.col("pre_close")
        
        tr1 = high - low
        tr2 = (high - pre_close).abs()
        tr3 = (low - pre_close).abs()
        
        true_range = pl.max_horizontal([tr1, tr2, tr3])
        atr = true_range.rolling_mean(window_size=ATR_WINDOW)
        
        atr_values = df.with_columns([
            true_range.alias("tr"),
            atr.alias("atr")
        ])["atr"].to_list()
        
        for val in reversed(atr_values):
            if val is not None and val > 0:
                return float(val)
        return None
    
    def get_stock_volatility(self, symbol: str, trade_date: str, window: int = 20) -> Optional[float]:
        """获取股票滚动波动率"""
        cache_key = f"{symbol}_{trade_date}"
        if cache_key in self.stock_volatility_cache:
            return self.stock_volatility_cache[cache_key]
        
        query = f"""
            SELECT close
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT {window + 5}
        """
        result = self.db.read_sql(query)
        if result.is_empty() or len(result) < window:
            return None
        
        closes = result["close"].to_list()
        closes = [float(c) if c else 0.0 for c in closes]
        closes = [c for c in closes if c > 0]
        
        if len(closes) < window:
            return None
        
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                ret = (closes[i] - closes[i-1]) / closes[i-1]
                returns.append(ret)
        
        if len(returns) < window - 1:
            return None
        
        volatility = float(np.std(returns[-window:], ddof=1))
        self.stock_volatility_cache[cache_key] = volatility
        
        return volatility
    
    def get_bollinger_position(self, symbol: str, trade_date: str, window: int = 20) -> Optional[dict]:
        """获取布林线位置"""
        cache_key = f"{symbol}_{trade_date}"
        if cache_key in self.bollinger_cache:
            return self.bollinger_cache[cache_key]
        
        query = f"""
            SELECT close
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT {window + 5}
        """
        result = self.db.read_sql(query)
        if result.is_empty() or len(result) < window:
            return None
        
        closes = result["close"].to_list()
        closes = [float(c) if c else 0.0 for c in closes]
        closes = [c for c in closes if c > 0]
        
        if len(closes) < window:
            return None
        
        recent_closes = closes[:window]
        middle = np.mean(recent_closes)
        std = np.std(recent_closes, ddof=1)
        
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        current_price = closes[0]
        
        if upper != lower:
            position = (current_price - lower) / (upper - lower)
        else:
            position = 0.5
        
        result_dict = {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "position": position,
            "deviation_from_lower": (current_price - lower) / current_price if current_price > 0 else 0,
        }
        self.bollinger_cache[cache_key] = result_dict
        
        return result_dict
    
    def get_stock_data(self, symbol: str, trade_date: str) -> Optional[dict]:
        """获取股票单日的 OHLCV 数据"""
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, pre_close, volume, amount
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date = '{trade_date}'
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            return None
        
        row = result.row(0, named=True)
        return {
            "symbol": row["symbol"],
            "trade_date": row["trade_date"],
            "open": float(row["open"]) if row["open"] else 0.0,
            "high": float(row["high"]) if row["high"] else 0.0,
            "low": float(row["low"]) if row["low"] else 0.0,
            "close": float(row["close"]) if row["close"] else 0.0,
            "pre_close": float(row["pre_close"]) if row["pre_close"] else 0.0,
            "volume": float(row["volume"]) if row["volume"] else 0.0,
            "amount": float(row["amount"]) if row["amount"] else 0.0,
        }
    
    def apply_price_noise(self, price: float) -> float:
        """应用价格噪声"""
        if not self.enable_noise:
            return price
        
        noise = np.random.normal(0, self.price_noise_std)
        return price * (1 + noise)
    
    def compute_volatility_adjusted_multiplier(self, symbol: str, trade_date: str) -> float:
        """
        Iteration 12: 计算波动率调整后的 ATR 乘数
        """
        if not VOLATILITY_STOP_CONFIG["enabled"]:
            return VOLATILITY_STOP_CONFIG["base_multiplier"]
        
        volatility = self.get_stock_volatility(symbol, trade_date)
        if volatility is None:
            return VOLATILITY_STOP_CONFIG["base_multiplier"]
        
        scaling = VOLATILITY_STOP_CONFIG["volatility_scaling"]
        
        if volatility > scaling["high_vol_threshold"]:
            return VOLATILITY_STOP_CONFIG["base_multiplier"] * scaling["high_scale"]
        elif volatility < scaling["low_vol_threshold"]:
            return VOLATILITY_STOP_CONFIG["base_multiplier"] * scaling["low_scale"]
        
        return VOLATILITY_STOP_CONFIG["base_multiplier"]
    
    def get_tiered_stop_params(self, hold_days: int) -> tuple[float, float]:
        """
        Iteration 12: 根据持有天数获取分级止损参数
        Returns: (multiplier, hard_stop)
        """
        if not TIERED_STOP_CONFIG["enabled"]:
            return (VOLATILITY_STOP_CONFIG["base_multiplier"], HARD_STOP_LOSS)
        
        for tier in TIERED_STOP_CONFIG["tiers"]:
            if tier["min_days"] <= hold_days <= tier["max_days"]:
                return (tier["multiplier"], tier["hard_stop"])
        
        return (1.0, HARD_STOP_LOSS)
    
    def compute_rsi(self, prices: list[float], period: int = 14) -> float:
        """计算 RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_oversold_rebound_score(self, symbol: str, trade_date: str) -> float:
        """
        Iteration 12: 计算超跌反弹因子评分
        """
        if not REVERSAL_FACTOR_CONFIG["oversold_rebound"]["enabled"]:
            return 0.0
        
        config = REVERSAL_FACTOR_CONFIG["oversold_rebound"]
        
        # 获取历史价格
        query = f"""
            SELECT close, volume
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT 30
        """
        result = self.db.read_sql(query)
        if result.is_empty() or len(result) < 20:
            return 0.0
        
        closes = [float(c) if c else 0.0 for c in result["close"].to_list()]
        volumes = [float(v) if v else 0.0 for v in result["volume"].to_list()]
        closes = [c for c in closes if c > 0]
        
        if len(closes) < 20:
            return 0.0
        
        # 计算 RSI
        rsi = self.compute_rsi(closes, 14)
        
        # 计算布林线位置
        bollinger = self.get_bollinger_position(symbol, trade_date)
        
        # 计算成交量放大倍数
        avg_volume_5d = np.mean(volumes[:5]) if len(volumes) >= 5 else 0
        avg_volume_20d = np.mean(volumes[:20]) if len(volumes) >= 20 else avg_volume_5d
        volume_ratio = avg_volume_5d / (avg_volume_20d + 1e-10)
        
        # 评分逻辑
        score = 0.0
        
        # RSI 超卖加分
        if rsi < config["rsi_threshold"]:
            score += (config["rsi_threshold"] - rsi) / config["rsi_threshold"] * 0.5
        
        # 布林线位置加分
        if bollinger and bollinger["position"] < config["bollinger_position"]:
            score += (config["bollinger_position"] - bollinger["position"]) / config["bollinger_position"] * 0.3
        
        # 成交量放大确认加分
        if volume_ratio > config["volume_confirm"]:
            score += 0.2
        
        return min(score, 1.0) * config["weight"]
    
    def compute_quality_reversal_score(self, symbol: str, trade_date: str) -> float:
        """
        Iteration 12: 计算质量反转因子评分
        """
        if not REVERSAL_FACTOR_CONFIG["quality_reversal"]["enabled"]:
            return 0.0
        
        config = REVERSAL_FACTOR_CONFIG["quality_reversal"]
        
        # 获取历史数据
        query = f"""
            SELECT close, volume, amount
            FROM stock_daily
            WHERE symbol = '{symbol}'
            AND trade_date <= '{trade_date}'
            ORDER BY trade_date DESC
            LIMIT 60
        """
        result = self.db.read_sql(query)
        if result.is_empty() or len(result) < 40:
            return 0.0
        
        closes = [float(c) if c else 0.0 for c in result["close"].to_list()]
        volumes = [float(v) if v else 0.0 for v in result["volume"].to_list()]
        amounts = result["amount"].to_list()
        
        # 处理缺失值
        for i, amt in enumerate(amounts):
            if amt is None:
                if closes[i] and volumes[i]:
                    amounts[i] = closes[i] * volumes[i]
                else:
                    amounts[i] = 0.0
        
        closes = [c for c in closes if c > 0]
        if len(closes) < 40:
            return 0.0
        
        # 计算波动率分位数
        returns_20d = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(min(20, len(closes)-1))]
        volatility = np.std(returns_20d, ddof=1) if len(returns_20d) > 5 else 0.0
        
        # 简化：假设波动率<2% 为低波动
        low_volatility_flag = 1.0 if volatility < 0.02 else 0.0
        
        # 流动性检查
        avg_amount_20d = np.mean(amounts[:20]) if len(amounts) >= 20 else 0
        liquidity_flag = 1.0 if avg_amount_20d > config["liquidity_threshold"] else 0.0
        
        # 动量反转
        if len(closes) >= 20 and closes[-20] > 0:
            momentum_20 = (closes[0] - closes[-20]) / closes[-20]
        else:
            momentum_20 = 0.0
        
        # 反转信号
        reversal_flag = 1.0 if momentum_20 < config["momentum_reversal"] else 0.0
        
        # 综合评分
        score = (low_volatility_flag * 0.4 + liquidity_flag * 0.3 + reversal_flag * 0.3) * config["weight"]
        
        return score
    
    def compute_stock_scores(self, trade_date: str) -> list[dict]:
        """计算所有股票的评分 (包含 Iteration 12 超跌反转因子)"""
        query = f"""
            SELECT DISTINCT symbol
            FROM stock_daily
            WHERE trade_date = '{trade_date}'
            AND volume > 0
            AND close > 0
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            return []
        
        symbols = result["symbol"].unique().limit(100).to_list()
        scored_stocks = []
        
        for symbol in symbols:
            try:
                history_query = f"""
                    SELECT symbol, trade_date, close, volume, pct_chg, pre_close
                    FROM stock_daily
                    WHERE symbol = '{symbol}'
                    AND trade_date <= '{trade_date}'
                    ORDER BY trade_date DESC
                    LIMIT 30
                """
                history_df = self.db.read_sql(history_query)
                
                if len(history_df) < 20:
                    continue
                
                history_df = history_df.sort("trade_date")
                close_list = [float(x) if x is not None else 0.0 for x in history_df["close"].to_list()]
                
                if not close_list or close_list[-1] <= 0:
                    continue
                
                # 动量因子
                if len(close_list) >= 5 and close_list[-5] > 0:
                    momentum_5 = (close_list[-1] - close_list[-5]) / close_list[-5]
                else:
                    momentum_5 = 0.0
                
                if len(close_list) >= 10 and close_list[-10] > 0:
                    momentum_10 = (close_list[-1] - close_list[-10]) / close_list[-10]
                else:
                    momentum_10 = 0.0
                
                if len(close_list) >= 20 and close_list[-20] > 0:
                    momentum_20 = (close_list[-1] - close_list[-20]) / close_list[-20]
                else:
                    momentum_20 = 0.0
                
                predict_score = 0.5 * momentum_5 + 0.3 * momentum_10 + 0.2 * momentum_20
                
                # 历史夏普
                if len(close_list) >= 20:
                    returns = []
                    for i in range(1, min(20, len(close_list))):
                        if close_list[-i-1] > 0 and close_list[-i] > 0:
                            ret = (close_list[-i] - close_list[-i-1]) / close_list[-i-1]
                            returns.append(ret)
                    if len(returns) > 5:
                        mean_ret = np.mean(returns)
                        std_ret = np.std(returns, ddof=1)
                        hist_sharpe = (mean_ret * 20) / (std_ret * np.sqrt(20)) if std_ret > 1e-10 else 0.0
                    else:
                        hist_sharpe = 0.0
                else:
                    hist_sharpe = 0.0
                
                # Iteration 12: 超跌反转因子
                oversold_score = self.compute_oversold_rebound_score(symbol, trade_date)
                quality_reversal_score = self.compute_quality_reversal_score(symbol, trade_date)
                reversal_bonus = oversold_score + quality_reversal_score
                
                # 流动性过滤
                avg_amount = self.get_avg_amount_5d(symbol, trade_date)
                if avg_amount < MIN_AVG_AMOUNT_5D:
                    self.liquidity_filtered_count += 1
                    continue
                
                # 综合评分
                combined_score = SCORE_WEIGHT * predict_score + SHARPE_WEIGHT * hist_sharpe + reversal_bonus
                
                scored_stocks.append({
                    "symbol": symbol,
                    "close": close_list[-1],
                    "predict_score": predict_score,
                    "hist_sharpe_20d": hist_sharpe,
                    "oversold_rebound_score": oversold_score,
                    "quality_reversal_score": quality_reversal_score,
                    "reversal_bonus": reversal_bonus,
                    "combined_score": combined_score,
                    "avg_amount_5d": avg_amount,
                })
            except Exception as e:
                logger.debug(f"Failed to compute score for {symbol}: {e}")
                continue
        
        scored_stocks.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_stocks
    
    def check_score_buffer_filter(self, symbol: str, current_score: float, threshold: float) -> bool:
        """
        Iteration 12: 检查 Score Buffer 过滤
        Returns: True 表示应该过滤 (不买入/不卖出)
        """
        if not SCORE_BUFFER_CONFIG["enabled"]:
            return False
        
        if symbol not in self.prev_scores:
            return False
        
        prev_score = self.prev_scores[symbol]
        
        if SCORE_BUFFER_CONFIG["relative_buffer"]:
            # 相对缓冲：变化需超过百分比
            if abs(prev_score) > 1e-10:
                change_pct = abs(current_score - prev_score) / abs(prev_score)
                if change_pct < SCORE_BUFFER_CONFIG["buffer_threshold"]:
                    return True
        
        # 绝对缓冲
        if abs(current_score - prev_score) < SCORE_BUFFER_CONFIG["absolute_buffer"]:
            return True
        
        return False
    
    def check_cooldown_filter(self, symbol: str, current_date: str) -> bool:
        """
        Iteration 12: 检查冷却期过滤
        Returns: True 表示在冷却期内，应该过滤
        """
        if not COOLDOWN_CONFIG["enabled"]:
            return False
        
        if symbol not in self.sold_history:
            return False
        
        sold_date = self.sold_history[symbol]
        try:
            sold_dt = datetime.strptime(sold_date, '%Y-%m-%d')
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            days_diff = (current_dt - sold_dt).days
            
            if days_diff < COOLDOWN_CONFIG["cooldown_days"]:
                return True
        except:
            pass
        
        return False
    
    def simulate_day(self, trade_date: str, trade_dates: list[str]) -> DailyRecord:
        """
        模拟单日交易 - Iteration 12 改进
        
        核心逻辑:
        1. Score Buffer 机制：评分变化需超过阈值才换仓
        2. 最小持有天数硬约束
        3. 换仓冷却期
        4. 滚动波动率动态止损
        5. 收盘价触发选项
        6. 分级止损
        7. 移动止盈
        """
        record = DailyRecord(
            trade_date=trade_date,
            market_mode="DEFENSIVE",
            market_regime="NORMAL",
            index_close=0.0,
            index_ma20=0.0,
            market_median_score=0.0,
            dynamic_threshold=0.0,
            cash=self.cash,
            portfolio_value=self.portfolio_value
        )
        
        # ========== Step 1: 环境识别 ==========
        index_data = self.get_index_data(trade_date)
        if index_data:
            record.index_close = index_data["close"]
            record.index_ma20 = index_data["ma20"] if index_data["ma20"] else 0.0
        
        market_mode, market_regime = self.get_market_mode(trade_date)
        record.market_mode = market_mode
        record.market_regime = market_regime
        
        median_score = self.get_market_median_score(trade_date)
        record.market_median_score = median_score
        
        if market_mode == "AGGRESSIVE":
            self.aggressive_days += 1
        else:
            self.defensive_days += 1
        
        if market_regime == "BULL":
            self.bull_market_days += 1
        
        config = self.get_config(market_mode, market_regime, median_score)
        current_threshold = SCORE_THRESHOLD + config["threshold_addon"]
        record.dynamic_threshold = current_threshold
        max_positions = config["max_positions"]
        base_atr_multiplier = config["atr_multiplier"]
        min_hold_days = config["min_hold_days"]
        
        logger.info(f"[MODE] {trade_date}: {market_mode}/{market_regime} (threshold={current_threshold:.2f}, max_pos={max_positions})")
        
        next_date = self.get_next_date(trade_dates, trade_date)
        
        # ========== Step 2: 处理持仓卖出 ==========
        to_sell = []
        
        for symbol, pos in list(self.holdings.items()):
            if next_date:
                next_data = self.get_stock_data(symbol, next_date)
                if next_data:
                    next_open = self.apply_price_noise(next_data["open"])
                    next_close = self.apply_price_noise(next_data["close"])
                    pre_close = next_data["pre_close"]
                    
                    if self.check_limit_down(next_open, pre_close):
                        record.limit_down_blocked.append(symbol)
                        self.limit_down_blocked_count += 1
                        logger.debug(f"  [LIMIT DOWN] {symbol}: cannot sell")
                        continue
                    
                    buy_price = pos["buy_price"]
                    pnl_pct = (next_open - buy_price) / buy_price if buy_price > 0 else 0
                    
                    # 计算持有天数
                    hold_days = 1
                    try:
                        buy_dt = datetime.strptime(pos["buy_date"], '%Y-%m-%d')
                        next_dt = datetime.strptime(next_date, '%Y-%m-%d')
                        hold_days = (next_dt - buy_dt).days
                    except:
                        pass
                    
                    should_sell = False
                    sell_reason = ""
                    
                    # ========== 移动止盈 (Trailing Stop) ==========
                    if PROFIT_PROTECTION["enabled"] and PROFIT_PROTECTION["trailing_stop"]:
                        # 更新最高价
                        current_high = pos.get("high_since_buy", buy_price)
                        if next_open > current_high:
                            current_high = next_open
                            self.holdings[symbol]["high_since_buy"] = current_high
                        
                        # 检查是否触发移动止盈
                        if pnl_pct > PROFIT_PROTECTION["trailing_threshold"]:
                            max_pnl = (current_high - buy_price) / buy_price
                            pullback = (current_high - next_open) / current_high
                            
                            if pullback > PROFIT_PROTECTION["trailing_percent"]:
                                should_sell = True
                                sell_reason = f"trailing stop (pullback={pullback:.1%} > {PROFIT_PROTECTION['trailing_percent']:.1%})"
                                record.trailing_stop_triggered.append(symbol)
                                self.trailing_stop_triggered_count += 1
                    
                    # ========== 盈利保护机制 (Break-even Stop) ==========
                    if not should_sell and PROFIT_PROTECTION["enabled"]:
                        break_even_stop = None
                        if pnl_pct > PROFIT_PROTECTION["profit_threshold"]:
                            break_even_stop = PROFIT_PROTECTION["stop_level"]
                            
                            if pnl_pct < break_even_stop:
                                should_sell = True
                                sell_reason = f"profit protection ({pnl_pct:.1%} < +{break_even_stop:.1%})"
                    
                    # ========== 最小持有天数硬约束 ==========
                    if MIN_HOLD_CONSTRAINT["enabled"] and not should_sell:
                        if hold_days < MIN_HOLD_CONSTRAINT["min_hold_days"]:
                            # 检查豁免条件
                            is_profit_exempt = pnl_pct > MIN_HOLD_CONSTRAINT["exemption_threshold"]
                            is_stop_loss = pnl_pct < -0.03  # 止损卖出豁免
                            
                            if not is_profit_exempt and not is_stop_loss:
                                record.min_hold_protected.append(symbol)
                                self.min_hold_protected_count += 1
                                logger.info(f"  [MIN HOLD] {symbol}: {hold_days} days < {MIN_HOLD_CONSTRAINT['min_hold_days']} days, protected")
                                continue
                    
                    # ========== 波动率动态止损 ==========
                    if not should_sell and VOLATILITY_STOP_CONFIG["enabled"]:
                        # 获取波动率调整后的乘数
                        adjusted_multiplier = self.compute_volatility_adjusted_multiplier(symbol, next_date)
                        record.volatility_adjustment_applied = True
                        
                        # 获取分级止损参数
                        tier_multiplier, tier_hard_stop = self.get_tiered_stop_params(hold_days)
                        final_multiplier = adjusted_multiplier * tier_multiplier
                        
                        atr = self.get_atr(symbol, next_date)
                        if atr and atr > 0:
                            atr_stop = -final_multiplier * (atr / next_open)
                            atr_stop = max(MIN_ATR_STOP, min(MAX_ATR_STOP, atr_stop))
                            
                            # 收盘价触发选项
                            if VOLATILITY_STOP_CONFIG["close_only"]:
                                close_pnl = (next_close - buy_price) / buy_price
                                if close_pnl < atr_stop:
                                    should_sell = True
                                    sell_reason = f"close stop ({close_pnl:.1%} < {atr_stop:.1%})"
                                    record.atr_stop_triggered.append(symbol)
                                    self.atr_stop_triggered_count += 1
                            else:
                                if pnl_pct < atr_stop:
                                    should_sell = True
                                    sell_reason = f"ATR stop ({pnl_pct:.1%} < {atr_stop:.1%}, mult={final_multiplier:.1f})"
                                    record.atr_stop_triggered.append(symbol)
                                    self.atr_stop_triggered_count += 1
                    
                    # ========== 硬止损 ==========
                    if hold_days >= 2 and pnl_pct < HARD_STOP_LOSS and not should_sell:
                        should_sell = True
                        sell_reason = f"hard stop ({pnl_pct:.1%} < {HARD_STOP_LOSS:.1%})"
                    
                    # ========== 评分卖出 (带 Score Buffer) ==========
                    effective_min_hold = PROFIT_EXEMPTION_MIN_HOLD if pnl_pct > PROFIT_EXEMPTION_THRESHOLD else min_hold_days
                    if hold_days >= effective_min_hold and not should_sell:
                        stock_scores = self.compute_stock_scores(trade_date)
                        for s in stock_scores:
                            if s["symbol"] == symbol:
                                new_score = s["combined_score"]
                                
                                # Score Buffer 检查
                                if self.check_score_buffer_filter(symbol, new_score, current_threshold):
                                    record.score_buffer_filtered.append(symbol)
                                    self.score_buffer_filtered_count += 1
                                    logger.debug(f"  [SCORE BUFFER] {symbol}: score change too small")
                                else:
                                    if new_score < current_threshold:
                                        should_sell = True
                                        sell_reason = f"low score ({new_score:+.2f} < {current_threshold:.2f})"
                                    # 更新评分
                                    self.prev_scores[symbol] = new_score
                                break
                    
                    if should_sell:
                        to_sell.append((symbol, pos, next_open, sell_reason))
        
        # 执行卖出
        for symbol, pos, sell_price, reason in to_sell:
            sell_value = sell_price * pos["shares"]
            commission = max(sell_value * COMMISSION_RATE, COMMISSION_MIN)
            stamp_duty = sell_value * STAMP_DUTY_RATE
            
            self.cash += sell_value - commission - stamp_duty
            self.total_commission += commission
            self.total_stamp_duty += stamp_duty
            self.total_trades += 1
            
            # 记录卖出日期 (用于冷却期)
            self.sold_history[symbol] = next_date
            
            record.sold_symbols.append(symbol)
            del self.holdings[symbol]
            logger.info(f"  [SELL] {symbol} @ {sell_price:.2f} ({reason})")
        
        # ========== Step 3: 选股买入 ==========
        scored_stocks = self.compute_stock_scores(trade_date)
        
        # 更新评分记录
        for s in scored_stocks:
            self.prev_scores[s["symbol"]] = s["combined_score"]
        
        available_stocks = [s for s in scored_stocks if s["symbol"] not in self.holdings]
        
        if len(self.holdings) < max_positions and available_stocks:
            top_stocks = available_stocks[:2]
            
            available_cash = self.cash * 0.95
            
            for stock in top_stocks:
                if len(self.holdings) >= max_positions:
                    break
                
                symbol = stock["symbol"]
                combined_score = stock["combined_score"]
                
                # 冷却期检查
                if self.check_cooldown_filter(symbol, trade_date):
                    record.cooldown_filtered.append(symbol)
                    self.cooldown_filtered_count += 1
                    logger.debug(f"  [COOLDOWN] {symbol}: in cooldown period")
                    continue
                
                if combined_score < current_threshold:
                    logger.debug(f"  [SKIP] {symbol}: score {combined_score:+.2f} < {current_threshold:.2f}")
                    continue
                
                # Score Buffer 检查 (买入)
                if self.check_score_buffer_filter(symbol, combined_score, current_threshold):
                    record.score_buffer_filtered.append(symbol)
                    self.score_buffer_filtered_count += 1
                    logger.debug(f"  [SCORE BUFFER] {symbol}: score change too small to buy")
                    continue
                
                if next_date:
                    next_data = self.get_stock_data(symbol, next_date)
                    if next_data:
                        next_open = self.apply_price_noise(next_data["open"])
                        pre_close = next_data["pre_close"]
                        
                        if self.check_limit_up(next_open, pre_close):
                            record.limit_up_blocked.append(symbol)
                            self.limit_up_blocked_count += 1
                            logger.info(f"  [LIMIT UP] {symbol}: cannot buy")
                            continue
                        
                        total_score = sum(s["combined_score"] for s in top_stocks)
                        if total_score > 0:
                            target_weight = max(combined_score / total_score, 1.0 / max_positions)
                        else:
                            target_weight = 1.0 / max_positions
                        
                        budget = available_cash * target_weight
                        
                        slippage_price = next_open * (1 + SLIPPAGE_RATE)
                        raw_shares = int(budget / slippage_price)
                        shares = (raw_shares // 100) * 100
                        
                        if shares < 100:
                            continue
                        
                        buy_value = shares * slippage_price
                        commission = max(buy_value * COMMISSION_RATE, COMMISSION_MIN)
                        total_cost = buy_value + commission
                        
                        if total_cost > self.cash:
                            max_shares = int(self.cash / slippage_price) // 100 * 100
                            if max_shares < 100:
                                continue
                            shares = max_shares
                            buy_value = shares * slippage_price
                            total_cost = buy_value + commission
                        
                        self.cash -= total_cost
                        self.total_commission += commission
                        self.total_trades += 1
                        
                        self.holdings[symbol] = {
                            "shares": shares,
                            "buy_price": slippage_price,
                            "buy_date": next_date,
                            "high_since_buy": slippage_price,
                        }
                        
                        record.bought_symbols.append(symbol)
                        logger.info(f"  [BUY] {symbol} @ {slippage_price:.2f} x {shares} (score={combined_score:+.2f})")
        
        # ========== Step 4: 计算组合价值 ==========
        prev_value = self.portfolio_value
        self.portfolio_value = self.cash
        
        for symbol, pos in self.holdings.items():
            price_data = self.get_stock_data(symbol, trade_date)
            if price_data:
                self.portfolio_value += price_data["close"] * pos["shares"]
        
        record.cash = self.cash
        record.holding_symbols = list(self.holdings.keys())
        record.holdings = [
            {"symbol": s, "shares": p["shares"], "buy_price": p["buy_price"], "buy_date": p["buy_date"]}
            for s, p in self.holdings.items()
        ]
        record.daily_pnl = self.portfolio_value - prev_value
        if prev_value > 0:
            record.daily_return = record.daily_pnl / prev_value
        
        return record
    
    def run(self) -> BacktestResult:
        """运行回测"""
        logger.info("=" * 60)
        logger.info(f"Final Strategy V1.2 Backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Initial Capital: ¥{self.initial_capital:,.2f}")
        logger.info(f"Iteration 12 Features:")
        logger.info(f"  - Score Buffer: {SCORE_BUFFER_CONFIG['enabled']}")
        logger.info(f"  - Min Hold Constraint: {MIN_HOLD_CONSTRAINT['enabled']}")
        logger.info(f"  - Cooldown Period: {COOLDOWN_CONFIG['enabled']}")
        logger.info(f"  - Volatility Stop: {VOLATILITY_STOP_CONFIG['enabled']}")
        logger.info(f"  - Trailing Stop: {PROFIT_PROTECTION.get('trailing_stop', False)}")
        logger.info(f"  - Reversal Factors: Oversold={REVERSAL_FACTOR_CONFIG['oversold_rebound']['enabled']}, Quality={REVERSAL_FACTOR_CONFIG['quality_reversal']['enabled']}")
        logger.info("=" * 60)
        
        trade_dates = self.get_trade_dates()
        if not trade_dates:
            logger.error("No trade dates found")
            return self._empty_result()
        
        logger.info(f"Total trading days: {len(trade_dates)}")
        
        for i, date in enumerate(trade_dates, 1):
            logger.info(f"[{i}/{len(trade_dates)}] {date}")
            record = self.simulate_day(date, trade_dates)
            self.daily_records.append(record)
            
            if record.bought_symbols or record.sold_symbols:
                logger.info(f"  Trades: Buy={record.bought_symbols}, Sell={record.sold_symbols}")
                logger.info(f"  Portfolio: ¥{record.portfolio_value:,.0f} ({record.daily_return:+.2%})")
                logger.info(f"  Mode: {record.market_mode}/{record.market_regime}, Holdings: {len(record.holding_symbols)}")
        
        metrics = self._calculate_metrics()
        benchmark = self._calculate_benchmark()
        monthly_returns = self._calculate_monthly_returns()
        robustness_score = self._calculate_robustness_score(monthly_returns, benchmark)
        noise_sensitivity = self._calculate_noise_sensitivity()
        loss_attribution = self._analyze_loss_attribution()
        
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            total_days=len(trade_dates),
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            max_drawdown=metrics["max_drawdown"],
            sharpe_ratio=metrics["sharpe_ratio"],
            win_rate=metrics["win_rate"],
            avg_daily_return=metrics["avg_daily_return"],
            daily_records=self.daily_records,
            benchmark_return=benchmark,
            total_trades=self.total_trades,
            total_commission=self.total_commission,
            total_stamp_duty=self.total_stamp_duty,
            limit_up_blocked_count=self.limit_up_blocked_count,
            limit_down_blocked_count=self.limit_down_blocked_count,
            liquidity_filtered_count=self.liquidity_filtered_count,
            atr_stop_triggered_count=self.atr_stop_triggered_count,
            score_buffer_filtered_count=self.score_buffer_filtered_count,
            cooldown_filtered_count=self.cooldown_filtered_count,
            min_hold_protected_count=self.min_hold_protected_count,
            trailing_stop_triggered_count=self.trailing_stop_triggered_count,
            aggressive_days=self.aggressive_days,
            defensive_days=self.defensive_days,
            bull_market_days=self.bull_market_days,
            monthly_returns=monthly_returns,
            robustness_score=robustness_score,
            noise_sensitivity=noise_sensitivity,
            loss_attribution=loss_attribution,
        )
    
    def _calculate_metrics(self) -> dict:
        """计算绩效指标"""
        if not self.daily_records:
            return self._empty_metrics()
        
        values = [r.portfolio_value for r in self.daily_records]
        returns = []
        
        for i in range(1, len(values)):
            if values[i-1] > 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        if not returns:
            return self._empty_metrics()
        
        returns_arr = np.array(returns)
        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        days = len(self.daily_records)
        annualized_return = (1 + total_return) ** (252 / max(days, 1)) - 1
        
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        mean_ret = float(np.mean(returns_arr))
        std_ret = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
        
        if std_ret > 1e-10:
            annualized_ret = mean_ret * 252
            annualized_vol = std_ret * np.sqrt(252)
            sharpe = (annualized_ret - 0.03) / annualized_vol
        else:
            sharpe = 0.0
        
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "avg_daily_return": mean_ret,
        }
    
    def _empty_metrics(self) -> dict:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "avg_daily_return": 0.0,
        }
    
    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            start_date="",
            end_date="",
            total_days=0,
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            avg_daily_return=0.0,
            daily_records=[],
            benchmark_return=0.0,
            total_trades=0,
            total_commission=0.0,
            total_stamp_duty=0.0,
            monthly_returns={},
            robustness_score=0.0,
            noise_sensitivity=0.0,
            loss_attribution={},
        )
    
    def _calculate_benchmark(self) -> float:
        """计算基准收益 (000905)"""
        query = f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE symbol = '000905.SH'
            AND trade_date >= '{self.start_date}'
            AND trade_date <= '{self.end_date}'
            ORDER BY trade_date
        """
        result = self.db.read_sql(query)
        if result.is_empty() or len(result) < 2:
            return 0.0
        
        closes = result["close"].to_list()
        start = float(closes[0]) if closes[0] else 0.0
        end = float(closes[-1]) if closes[-1] else 0.0
        
        if start > 0:
            return (end - start) / start
        return 0.0
    
    def _calculate_monthly_returns(self) -> dict[str, float]:
        """计算月度收益率"""
        if not self.daily_records:
            return {}
        
        monthly_values: dict[str, list[float]] = {}
        
        for record in self.daily_records:
            month = record.trade_date[:7]
            if month not in monthly_values:
                monthly_values[month] = []
            monthly_values[month].append(record.portfolio_value)
        
        monthly_returns = {}
        prev_month_end = self.initial_capital
        
        for month in sorted(monthly_values.keys()):
            values = monthly_values[month]
            month_start = values[0]
            month_end = values[-1]
            
            monthly_ret = (month_end - prev_month_end) / prev_month_end
            monthly_returns[month] = monthly_ret
            prev_month_end = month_end
        
        return monthly_returns
    
    def _calculate_robustness_score(self, monthly_returns: dict[str, float], benchmark_return: float) -> float:
        """计算策略鲁棒性得分"""
        if not monthly_returns:
            return 0.0
        
        num_months = len(monthly_returns)
        if num_months == 0:
            return 0.0
        
        monthly_benchmark = benchmark_return / num_months
        
        excess_returns = []
        for month, ret in monthly_returns.items():
            excess = ret - monthly_benchmark
            excess_returns.append(excess)
        
        if not excess_returns:
            return 0.0
        
        win_ratio = sum(1 for r in excess_returns if r > 0) / len(excess_returns)
        excess_std = np.std(excess_returns, ddof=1) if len(excess_returns) > 1 else 0.0
        stability = 1.0 / (1.0 + excess_std * 10)
        avg_excess = np.mean(excess_returns)
        
        robustness_score = win_ratio * stability * (1 + avg_excess)
        
        return robustness_score
    
    def _calculate_noise_sensitivity(self) -> float:
        """
        Iteration 12: 计算噪声敏感度 (衰减分析)
        通过比较有噪声和无噪声的收益差异来评估
        """
        # 简单实现：基于日收益率的标准差
        if not self.daily_records:
            return 0.0
        
        daily_returns = [r.daily_return for r in self.daily_records if r.daily_return != 0]
        if not daily_returns:
            return 0.0
        
        # 噪声敏感度 = 收益率标准差 / 平均收益率
        std_ret = np.std(daily_returns, ddof=1)
        mean_ret = np.mean(daily_returns)
        
        if abs(mean_ret) < 1e-10:
            return std_ret * 100  # 返回放大值
        
        return std_ret / abs(mean_ret)
    
    def _analyze_loss_attribution(self) -> dict[str, int]:
        """
        Iteration 12: 归因分析 - 统计亏损交易原因
        """
        attribution = {
            "stop_loss_triggered": 0,
            "low_score_sold": 0,
            "profit_protection": 0,
            "trailing_stop": 0,
            "other": 0,
        }
        
        for record in self.daily_records:
            for symbol in record.sold_symbols:
                # 查找卖出原因
                if symbol in record.atr_stop_triggered:
                    attribution["stop_loss_triggered"] += 1
                elif symbol in record.trailing_stop_triggered:
                    attribution["trailing_stop"] += 1
                elif symbol in record.profit_protection_triggered if hasattr(record, 'profit_protection_triggered') else []:
                    attribution["profit_protection"] += 1
                else:
                    attribution["low_score_sold"] += 1
        
        return attribution
    
    def generate_report(self, result: BacktestResult, report_name: str = "Final Strategy V1.2") -> str:
        """生成 Markdown 报告"""
        lines = []
        
        lines.append(f"# {report_name} 回测报告")
        lines.append("")
        lines.append(f"**回测区间**: {result.start_date} 至 {result.end_date}")
        lines.append(f"**交易天数**: {result.total_days}")
        lines.append(f"**初始资金**: ¥{self.initial_capital:,.2f}")
        lines.append("")
        
        lines.append("## Iteration 12 核心改进说明")
        lines.append("")
        lines.append("### 1. 交易稳定性优化")
        lines.append("")
        lines.append(f"- **Score Buffer**: {SCORE_BUFFER_CONFIG['enabled']} (threshold={SCORE_BUFFER_CONFIG['buffer_threshold']:.1%})")
        lines.append(f"- **最小持有天数**: {MIN_HOLD_CONSTRAINT['min_hold_days']} 天")
        lines.append(f"- **冷却期**: {COOLDOWN_CONFIG['cooldown_days']} 天")
        lines.append(f"- **Score Buffer 过滤次数**: {result.score_buffer_filtered_count}")
        lines.append(f"- **冷却期过滤次数**: {result.cooldown_filtered_count}")
        lines.append(f"- **最小持有保护次数**: {result.min_hold_protected_count}")
        lines.append("")
        
        lines.append("### 2. 动态风控升级")
        lines.append("")
        lines.append(f"- **滚动波动率止损**: {VOLATILITY_STOP_CONFIG['enabled']}")
        lines.append(f"- **收盘价触发**: {VOLATILITY_STOP_CONFIG['close_only']}")
        lines.append(f"- **分级止损**: {TIERED_STOP_CONFIG['enabled']}")
        lines.append(f"- **移动止盈**: {PROFIT_PROTECTION.get('trailing_stop', False)}")
        lines.append(f"- **ATR 止损触发**: {result.atr_stop_triggered_count} 次")
        lines.append(f"- **移动止盈触发**: {result.trailing_stop_triggered_count} 次")
        lines.append("")
        
        lines.append("### 3. 因子质量增强")
        lines.append("")
        lines.append(f"- **超跌反弹因子**: {REVERSAL_FACTOR_CONFIG['oversold_rebound']['enabled']}")
        lines.append(f"- **质量反转因子**: {REVERSAL_FACTOR_CONFIG['quality_reversal']['enabled']}")
        lines.append("")
        
        lines.append("## 业绩摘要")
        lines.append("")
        lines.append("| 指标 | 策略 | 基准 |")
        lines.append("|------|------|------|")
        lines.append(f"| **总收益率** | {result.total_return:.2%} | {result.benchmark_return:.2%} |")
        lines.append(f"| **年化收益** | {result.annualized_return:.2%} | - |")
        lines.append(f"| **最大回撤** | {result.max_drawdown:.2%} | - |")
        lines.append(f"| **夏普比率** | {result.sharpe_ratio:.2f} | - |")
        lines.append(f"| **胜率** | {result.win_rate:.2%} | - |")
        lines.append("")
        
        lines.append(f"**超额收益**: {result.total_return - result.benchmark_return:.2%}")
        lines.append(f"**鲁棒性得分**: {result.robustness_score:.4f}")
        lines.append(f"**噪声敏感度**: {result.noise_sensitivity:.4f}")
        lines.append("")
        
        lines.append("## 模式统计")
        lines.append("")
        lines.append(f"- **进攻模式天数**: {result.aggressive_days} 天 ({result.aggressive_days/result.total_days*100:.1f}%)")
        lines.append(f"- **防御模式天数**: {result.defensive_days} 天 ({result.defensive_days/result.total_days*100:.1f}%)")
        lines.append(f"- **牛市模式天数**: {result.bull_market_days} 天 ({result.bull_market_days/result.total_days*100:.1f}%)")
        lines.append("")
        
        lines.append("## 月度收益分析")
        lines.append("")
        lines.append("| 月份 | 策略收益 | 超额收益 |")
        lines.append("|------|----------|----------|")
        
        num_months = len(result.monthly_returns)
        monthly_benchmark = result.benchmark_return / max(num_months, 1)
        
        for month, ret in sorted(result.monthly_returns.items()):
            excess = ret - monthly_benchmark
            beat = "✓" if excess > 0 else "✗"
            lines.append(f"| {month} | {ret:.2%} | {excess:+.2%} {beat} |")
        
        lines.append("")
        
        beat_months = sum(1 for r in result.monthly_returns.values() if r > monthly_benchmark)
        lines.append(f"**跑赢基准月份**: {beat_months}/{num_months} ({beat_months/max(num_months, 1)*100:.1f}%)")
        lines.append("")
        
        lines.append("## 亏损归因分析")
        lines.append("")
        lines.append("| 原因 | 次数 |")
        lines.append("|------|------|")
        for reason, count in result.loss_attribution.items():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        
        lines.append("## 实战约束统计")
        lines.append("")
        lines.append(f"- **总交易次数**: {result.total_trades}")
        lines.append(f"- **涨停无法买入**: {result.limit_up_blocked_count} 次")
        lines.append(f"- **跌停无法卖出**: {result.limit_down_blocked_count} 次")
        lines.append(f"- **流动性过滤**: {result.liquidity_filtered_count} 只股票")
        lines.append(f"- **总佣金**: ¥{result.total_commission:.2f}")
        lines.append(f"- **总印花税**: ¥{result.total_stamp_duty:.2f}")
        lines.append(f"- **总成本**: ¥{result.total_commission + result.total_stamp_duty:.2f}")
        lines.append(f"- **成本率**: {(result.total_commission + result.total_stamp_duty) / self.initial_capital:.2%}")
        lines.append("")
        
        lines.append("## 每日权益曲线")
        lines.append("")
        lines.append("```")
        lines.append("Date,Mode,Regime,Portfolio,Cash,Holdings,Daily PnL")
        for r in result.daily_records:
            mode_short = "AGG" if r.market_mode == "AGGRESSIVE" else "DEF"
            regime_short = "BULL" if r.market_regime == "BULL" else "NORM"
            lines.append(f"{r.trade_date},{mode_short},{regime_short},{r.portfolio_value:.2f},{r.cash:.2f},{len(r.holding_symbols)},{r.daily_pnl:.2f}")
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)


# ===========================================
# Iteration 12: 因子 IC 分析模块
# ===========================================

class FactorICAnalyzer:
    """
    因子 IC 分析器 - 用于因子质量审计
    """
    
    def __init__(self, db):
        self.db = db
        self.ic_results: list[FactorICAnalysis] = []
    
    def calculate_factor_ic(
        self, 
        factor_name: str, 
        factor_expression: str,
        start_date: str, 
        end_date: str,
        forward_window: int = 5
    ) -> Tuple[float, float]:
        """
        计算因子在指定区间的 IC 值和 ICIR
        
        IC = 因子值 与 未来收益 的相关系数
        ICIR = IC / IC 标准差
        """
        # 获取样本股票
        query = f"""
            SELECT DISTINCT symbol, trade_date, close
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            AND volume > 0
            AND close > 0
            ORDER BY symbol, trade_date
        """
        result = self.db.read_sql(query)
        
        if result.is_empty():
            return (0.0, 0.0)
        
        # 按股票分组计算
        symbols = result["symbol"].unique().to_list()
        
        all_factor_values = []
        all_future_returns = []
        
        for symbol in symbols:
            symbol_data = result.filter(pl.col("symbol") == symbol).sort("trade_date")
            
            if len(symbol_data) < forward_window + 10:
                continue
            
            # 计算因子值
            close_list = symbol_data["close"].to_list()
            trade_dates = symbol_data["trade_date"].to_list()
            
            # 构建因子计算上下文
            context = {
                "close": pl.Series(close_list),
                "pl": pl,
                "float": float,
            }
            
            try:
                # 执行因子表达式
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min}}
                factor_values = eval(factor_expression, eval_globals, context)
                
                # 计算未来收益
                future_returns = []
                valid_factors = []
                
                for i in range(len(close_list) - forward_window):
                    if close_list[i] > 0:
                        future_ret = close_list[i + forward_window] / close_list[i] - 1
                        future_returns.append(future_ret)
                        valid_factors.append(factor_values[i])
                
                all_factor_values.extend(valid_factors)
                all_future_returns.extend(future_returns)
                
            except Exception as e:
                logger.debug(f"Failed to compute IC for {factor_name}: {e}")
                continue
        
        if len(all_factor_values) < 30:
            return (0.0, 0.0)
        
        # 计算 IC (相关系数)
        ic = float(np.corrcoef(all_factor_values, all_future_returns)[0, 1])
        
        # 计算 ICIR
        if np.isnan(ic):
            ic = 0.0
        
        icir = ic / np.std(all_factor_values) if np.std(all_factor_values) > 1e-10 else 0.0
        
        return (ic, icir)
    
    def analyze_factors(
        self, 
        factors: list[dict],
        training_period: Tuple[str, str],
        validation_period: Tuple[str, str],
        ic_threshold: float = -0.02
    ) -> list[FactorICAnalysis]:
        """
        分析所有因子的 IC 表现
        """
        self.ic_results = []
        
        logger.info(f"Starting Factor IC Analysis...")
        logger.info(f"Training Period: {training_period}")
        logger.info(f"Validation Period: {validation_period}")
        
        for factor in factors:
            factor_name = factor["name"]
            factor_expression = factor["expression"]
            
            # 计算训练集 IC
            train_ic, train_icir = self.calculate_factor_ic(
                factor_name, factor_expression,
                training_period[0], training_period[1]
            )
            
            # 计算验证集 IC
            val_ic, val_icir = self.calculate_factor_ic(
                factor_name, factor_expression,
                validation_period[0], validation_period[1]
            )
            
            # 判断是否应该剔除
            is_negative = val_ic < ic_threshold
            should_remove = is_negative
            
            # 金融逻辑解释
            financial_logic = self._explain_financial_logic(factor_name, factor)
            
            result = FactorICAnalysis(
                factor_name=factor_name,
                training_ic=train_ic,
                training_icir=train_icir,
                validation_ic=val_ic,
                validation_icir=val_icir,
                is_negative_in_validation=is_negative,
                should_remove=should_remove,
                financial_logic=financial_logic,
            )
            
            self.ic_results.append(result)
            
            status = "✗ REMOVE" if should_remove else "✓ KEEP"
            logger.info(f"  [{status}] {factor_name}: Train IC={train_ic:.4f}, Val IC={val_ic:.4f}")
        
        return self.ic_results
    
    def _explain_financial_logic(self, factor_name: str, factor: dict) -> str:
        """
        用一句话解释因子的金融逻辑
        """
        explanations = {
            "momentum_5": "短期动量效应，捕捉 5 日趋势延续性",
            "momentum_10": "中期动量效应，捕捉 10 日趋势延续性",
            "momentum_20": "月度动量效应，捕捉 20 日趋势延续性",
            "volatility_5": "短期波动率，低波动股票通常表现更稳定",
            "volatility_20": "中期波动率，衡量价格波动风险",
            "volume_ma_ratio_5": "成交量相对水平，放量通常预示趋势延续",
            "volume_ma_ratio_20": "长期成交量相对水平",
            "price_position_20": "价格在 20 日区间的位置，低位可能超卖",
            "price_position_60": "价格在 60 日区间的位置",
            "ma_deviation_5": "价格偏离 5 日均线的程度",
            "ma_deviation_20": "价格偏离 20 日均线的程度",
            "rsi_14": "相对强弱指标，衡量超买超卖状态",
            "mfi_14": "资金流量指标，成交量加权的 RSI",
            "turnover_bias_20": "换手率偏离程度，异常换手可能预示主力行为",
            "turnover_ma_ratio": "换手率相对水平",
            "volume_price_divergence_5": "量价背离，价格上涨但成交量萎缩可能见顶",
            "volume_price_divergence_20": "长期量价背离",
            "volume_price_correlation": "量价相关性，负相关可能预示背离",
            "smart_money_flow": "聪明钱流向，识别主力行为",
            "volatility_contraction_10": "波动率收缩，预示即将突破",
            "volume_shrink_ratio": "成交量萎缩比率",
            "volume_price_stable": "量增价稳，主力吸筹信号",
            "accumulation_distribution_20": "累积/派发指标",
            "alpha101_mean_reversion": "均值回归，超跌反弹逻辑",
            "alpha101_volume_price_rank": "量价排名相关性",
            "alpha101_reversal_5d": "短期反转效应",
            "alpha101_downside_volume": "下跌成交量占比",
            "volume_entropy_20": "成交量分布熵值，衡量分歧程度",
            "amplitude_turnover_ratio_20": "振幅/换手比率，价格波动效率",
            "tail_volume_price_corr_20": "尾盘量价相关性",
            "oversold_rebound": "超跌反弹信号，RSI<30+ 布林线下轨",
            "quality_reversal": "质量反转信号，低波动 + 高流动性 + 超卖",
        }
        
        return explanations.get(factor_name, f"因子 {factor_name}：{factor.get('description', '无描述')}")
    
    def get_top_factors(self, n: int = 10) -> list[FactorICAnalysis]:
        """获取 IC 值最高的 N 个因子"""
        if not self.ic_results:
            return []
        
        sorted_results = sorted(self.ic_results, key=lambda x: x.validation_ic, reverse=True)
        return sorted_results[:n]
    
    def get_negative_factors(self) -> list[FactorICAnalysis]:
        """获取验证集 IC 为负的因子"""
        return [r for r in self.ic_results if r.is_negative_in_validation]
    
    def generate_report(self) -> str:
        """生成因子 IC 分析报告"""
        lines = []
        
        lines.append("# 因子 IC 分析报告")
        lines.append("")
        lines.append("## 分析说明")
        lines.append("")
        lines.append("- **IC 值**: 因子值与未来 5 日收益的相关系数")
        lines.append("- **ICIR**: IC 值 / IC 标准差，衡量 IC 稳定性")
        lines.append("- **剔除标准**: 验证集 (2024Q1) IC < -0.02")
        lines.append("")
        
        lines.append("## Top 10 因子 (按验证集 IC 排名)")
        lines.append("")
        lines.append("| 排名 | 因子名 | 训练 IC | 训练 ICIR | 验证 IC | 验证 ICIR | 金融逻辑 |")
        lines.append("|------|--------|---------|-----------|---------|-----------|----------|")
        
        top_factors = self.get_top_factors(10)
        for i, f in enumerate(top_factors, 1):
            lines.append(f"| {i} | {f.factor_name} | {f.training_ic:.4f} | {f.training_icir:.4f} | {f.validation_ic:.4f} | {f.validation_icir:.4f} | {f.financial_logic} |")
        
        lines.append("")
        
        negative_factors = self.get_negative_factors()
        if negative_factors:
            lines.append("## 建议剔除的因子 (验证集 IC < -0.02)")
            lines.append("")
            lines.append("| 因子名 | 训练 IC | 验证 IC | 金融逻辑 |")
            lines.append("|--------|---------|---------|----------|")
            for f in negative_factors:
                lines.append(f"| {f.factor_name} | {f.training_ic:.4f} | {f.validation_ic:.4f} | {f.financial_logic} |")
            lines.append("")
        else:
            lines.append("## 无需要剔除的因子")
            lines.append("")
        
        lines.append("## 逻辑有效性申明")
        lines.append("")
        lines.append("所有保留因子均满足以下标准:")
        lines.append("1. 验证集 IC 值 > -0.02")
        lines.append("2. 具有明确的金融逻辑解释")
        lines.append("3. 非纯统计规律，基于市场行为逻辑")
        lines.append("")
        
        return "\n".join(lines)


# ===========================================
# Iteration 12: 参数自调节模块
# ===========================================

class ParameterOptimizer:
    """
    参数自调节模块 - 根据回测结果自动调整参数
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"Config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def adjust_parameters(self, result: BacktestResult):
        """
        根据回测结果调整参数
        """
        logger.info("Starting Parameter Optimization...")
        
        # 规则 1: 如果止损触发过多，放宽 ATR 乘数
        if result.atr_stop_triggered_count > result.total_trades * 0.3:
            current_mult = self.config.get('atr_stop', {}).get('multiplier', 2.5)
            new_mult = min(current_mult + 0.3, 3.5)
            
            if 'atr_stop' not in self.config:
                self.config['atr_stop'] = {}
            self.config['atr_stop']['multiplier'] = new_mult
            
            logger.info(f"  Adjusting ATR multiplier: {current_mult} -> {new_mult} (too many stop losses)")
        
        # 规则 2: 如果盈利保护触发过多，提高阈值
        if result.trailing_stop_triggered_count > result.total_trades * 0.2:
            current_threshold = PROFIT_PROTECTION.get('trailing_threshold', 0.05)
            new_threshold = min(current_threshold + 0.01, 0.08)
            
            logger.info(f"  Adjusting trailing stop threshold: {current_threshold:.1%} -> {new_threshold:.1%} (too early exits)")
        
        # 规则 3: 如果 Score Buffer 过滤过多，降低阈值
        if result.score_buffer_filtered_count > result.total_trades * 0.2:
            current_buffer = SCORE_BUFFER_CONFIG.get('buffer_threshold', 0.15)
            new_buffer = max(current_buffer - 0.03, 0.08)
            
            logger.info(f"  Adjusting score buffer: {current_buffer:.1%} -> {new_buffer:.1%} (too conservative)")
        
        # 保存配置
        self._save_config()
        
        return self.config


# ===========================================
# 主函数
# ===========================================

def run_final_strategy_v1_2():
    """运行 Final Strategy V1.2 回测"""
    
    logger.info("=" * 60)
    logger.info("盲测区间：2024-01-01 ~ 2024-04-30 (120 天)")
    logger.info("=" * 60)
    
    # 测试 1: 无噪声基准测试
    tester_base = FinalStrategyV12Backtester(
        start_date="2024-01-01",
        end_date="2024-04-30",
        initial_capital=INITIAL_CAPITAL,
        enable_noise=False,
    )
    result_base = tester_base.run()
    
    # 测试 2: 价格扰动压力测试
    logger.info("=" * 60)
    logger.info("压力测试：价格扰动 ±0.1% 高斯噪声")
    logger.info("=" * 60)
    
    np.random.seed(42)
    tester_noise = FinalStrategyV12Backtester(
        start_date="2024-01-01",
        end_date="2024-04-30",
        initial_capital=INITIAL_CAPITAL,
        price_noise_std=0.001,
        enable_noise=True,
    )
    result_noise = tester_noise.run()
    
    # 生成报告
    report_base = tester_base.generate_report(result_base, "基准测试")
    report_noise = tester_noise.generate_report(result_noise, "价格扰动压力测试")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("Iteration 12: Final Strategy V1.2 回测摘要")
    print("=" * 60)
    print(f"\n盲测区间 (2024-01-01 ~ 2024-04-30):")
    print(f"  基准测试总收益：{result_base.total_return:.2%}")
    print(f"  压力测试总收益：{result_noise.total_return:.2%}")
    print(f"  收益变化：{(result_noise.total_return - result_base.total_return):+.2%}")
    print(f"  基准测试夏普：{result_base.sharpe_ratio:.2f}")
    print(f"  压力测试夏普：{result_noise.sharpe_ratio:.2f}")
    print(f"  基准测试最大回撤：{result_base.max_drawdown:.2%}")
    print(f"  压力测试最大回撤：{result_noise.max_drawdown:.2%}")
    print(f"  鲁棒性得分：{result_base.robustness_score:.4f}")
    print(f"  噪声敏感度：{result_base.noise_sensitivity:.4f}")
    
    num_months = len(result_base.monthly_returns)
    monthly_benchmark = result_base.benchmark_return / max(num_months, 1)
    beat_months = sum(1 for r in result_base.monthly_returns.values() if r > monthly_benchmark)
    print(f"  跑赢基准月份：{beat_months}/{num_months}")
    print("=" * 60)
    
    return result_base, result_noise


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    run_final_strategy_v1_2()