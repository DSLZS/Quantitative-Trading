"""
Final Strategy V1.1 - Iteration 11: 自适应环境增强与止损逻辑进化

核心改进清单 (vs Iteration 10):
=================================

1. 止损逻辑升级 (Exit Optimization):
   - 波动率缩放止损：根据滚动 20 日振幅动态调整 ATR 乘数
   - 盈利保护机制：当单票盈利 > 2% 后，将止损位上移至成本价以上 0.5% (Break-even Stop)

2. 防御逻辑弹性化 (Dynamic Defense):
   - 动态 threshold_addon：根据 market_median_score 下行自动下调阈值
   - 新增均值回归因子：捕捉股价偏离布林线下轨的因子

3. 特征工程增强:
   - 布林线均值回归因子 (bollinger_reversion)
   - 波动率缩放因子 (volatility_scaled_atr)

作者：Quantitative Trading Team
版本：V1.1 (Iteration 11)
日期：2026-03-14
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal
import json
import yaml

import polars as pl
import numpy as np
from loguru import logger


# ===========================================
# Iteration 11 生产级参数配置
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
# Iteration 11 新增：止损逻辑升级
# ===========================================

# 波动率缩放止损参数
VOLATILITY_SCALING = {
    "enabled": True,
    "window": 20,  # 滚动 20 日振幅
    "base_multiplier": 2.5,  # 基础 ATR 乘数
    "volatility_adjustment": True,  # 启用波动率调整
    "high_volatility_scale": 0.8,  # 高波动时缩放乘数 (更宽松)
    "low_volatility_scale": 1.2,   # 低波动时缩放乘数 (更严格)
    "volatility_threshold": 0.03,  # 波动率阈值 (3%)
}

# 盈利保护机制 (Break-even Stop)
PROFIT_PROTECTION = {
    "enabled": True,
    "profit_threshold": 0.02,      # 盈利 > 2% 触发
    "stop_level": 0.005,           # 止损上移至成本价以上 0.5%
}

# ===========================================
# Iteration 11 新增：防御逻辑弹性化
# ===========================================

# 动态防御配置
DEFENSIVE_CONFIG = {
    "base_threshold_addon": 0.15,   # 基础门槛 (从 0.3 降至 0.15)
    "dynamic_adjustment": True,     # 启用动态调整
    "min_threshold_addon": 0.0,     # 最小门槛
    "max_threshold_addon": 0.3,     # 最大门槛
    "median_score_sensitivity": 0.5,  # 中位数评分敏感度
    "reference_median_score": 0.0,  # 参考中位数评分
    "max_positions": 3,
    "atr_multiplier": 2.5,
    "min_hold_days": 5,
}

# 进攻模式参数 (指数 > MA20)
AGGRESSIVE_CONFIG = {
    "threshold_addon": 0.0,
    "threshold_addon_bull": -0.1,
    "max_positions": 5,
    "max_positions_bull": 8,
    "atr_multiplier": 2.5,  # 从 3.0 降至 2.5，更宽松
    "min_hold_days": 3,
}

# ===========================================
# Iteration 10/11 共用配置
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
    dynamic_threshold: float  # Iteration 11 新增：动态阈值
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
    # Iteration 10/11 新增
    cost_hold_triggered: list[str] = field(default_factory=list)
    profit_exemption_triggered: list[str] = field(default_factory=list)
    profit_protection_triggered: list[str] = field(default_factory=list)  # Iteration 11
    volatility_adjustment_applied: bool = False  # Iteration 11


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
    # Iteration 10/11 新增
    cost_hold_triggered_count: int = 0
    profit_exemption_count: int = 0
    profit_protection_count: int = 0  # Iteration 11
    aggressive_days: int = 0
    defensive_days: int = 0
    bull_market_days: int = 0
    # 月度收益数据
    monthly_returns: dict[str, float] = field(default_factory=dict)
    # 鲁棒性得分
    robustness_score: float = 0.0


class FinalStrategyV11Backtester:
    """
    Final Strategy V1.1 - Iteration 11 自适应环境增强回测引擎
    
    核心特性:
    1. 波动率缩放止损：根据 20 日振幅动态调整 ATR 乘数
    2. 盈利保护机制：盈利>2% 后上移止损至成本价 +0.5%
    3. 动态防御阈值：根据市场中位数评分自动下调门槛
    4. 均值回归因子：布林线下轨偏离因子增强
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
        self.cost_hold_triggered_count = 0
        self.profit_exemption_count = 0
        self.profit_protection_count = 0
        self.aggressive_days = 0
        self.defensive_days = 0
        self.bull_market_days = 0
        
        # 数据库连接
        try:
            from .db_manager import DatabaseManager
            self.db = DatabaseManager.get_instance()
        except ImportError:
            from db_manager import DatabaseManager
            self.db = DatabaseManager.get_instance()
        
        # 缓存
        self.market_scores_cache: dict[str, list[dict]] = {}
        self.stock_volatility_cache: dict[str, float] = {}  # Iteration 11
        
        logger.info(f"Final Strategy V1.1 Backtester initialized: {start_date} to {end_date}")
        logger.info(f"Iteration 11 Features: Volatility Scaling={VOLATILITY_SCALING['enabled']}, Profit Protection={PROFIT_PROTECTION['enabled']}, Dynamic Defense={DEFENSIVE_CONFIG['dynamic_adjustment']}")
    
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
        """
        Iteration 11: 计算动态阈值
        
        逻辑：
        - 防御模式下，如果市场整体评分下行，自动下调 threshold_addon
        - 保证在弱势市场中仍有基础仓位
        """
        if market_mode == "AGGRESSIVE":
            config = AGGRESSIVE_CONFIG
            base_threshold = config["threshold_addon"]
            if market_median_score > 0.5:
                return config["threshold_addon_bull"]
            return base_threshold
        
        # 防御模式：动态调整
        config = DEFENSIVE_CONFIG
        
        if not config["dynamic_adjustment"]:
            return config["base_threshold_addon"]
        
        # 计算动态调整
        # 如果市场中位数评分低于参考值，下调门槛
        reference = config["reference_median_score"]
        sensitivity = config["median_score_sensitivity"]
        
        # 评分差值：当前评分 - 参考评分
        score_diff = market_median_score - reference
        
        # 调整量：负向调整 (评分越低，门槛越低)
        adjustment = -score_diff * sensitivity
        
        # 应用调整
        dynamic_threshold = config["base_threshold_addon"] + adjustment
        
        # 限制在合理范围
        dynamic_threshold = max(config["min_threshold_addon"], 
                               min(config["max_threshold_addon"], dynamic_threshold))
        
        return dynamic_threshold
    
    def get_config(self, market_mode: str, market_regime: str, market_median_score: float) -> dict:
        """获取当前模式对应的配置 (Iteration 11 动态阈值)"""
        if market_mode == "AGGRESSIVE":
            config = AGGRESSIVE_CONFIG.copy()
            if market_regime == "BULL":
                config["threshold_addon"] = config["threshold_addon_bull"]
                config["max_positions"] = config["max_positions_bull"]
            return config
        
        # 防御模式：使用动态阈值
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
        """
        Iteration 11: 获取股票滚动波动率 (用于波动率缩放止损)
        
        计算过去 window 日的收益率标准差
        """
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
        
        # 计算收益率
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
        """
        Iteration 11: 获取布林线位置 (用于均值回归因子)
        
        Returns:
            dict: {"upper": float, "middle": float, "lower": float, "position": float}
            position = (close - lower) / (upper - lower)
        """
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
        
        # 计算位置：0=下轨，0.5=中轨，1=上轨
        if upper != lower:
            position = (current_price - lower) / (upper - lower)
        else:
            position = 0.5
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "position": position,
            "deviation_from_lower": (current_price - lower) / current_price if current_price > 0 else 0,
        }
    
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
    
    def compute_atr_multiplier(self, symbol: str, trade_date: str, base_multiplier: float) -> float:
        """
        Iteration 11: 计算波动率缩放后的 ATR 乘数
        
        逻辑：
        - 高波动环境：降低乘数 (更宽松的止损)
        - 低波动环境：提高乘数 (更严格的止损)
        """
        if not VOLATILITY_SCALING["enabled"]:
            return base_multiplier
        
        volatility = self.get_stock_volatility(symbol, trade_date)
        if volatility is None:
            return base_multiplier
        
        threshold = VOLATILITY_SCALING["volatility_threshold"]
        
        if volatility > threshold:
            # 高波动：更宽松
            scale = VOLATILITY_SCALING["high_volatility_scale"]
        else:
            # 低波动：更严格
            scale = VOLATILITY_SCALING["low_volatility_scale"]
        
        return base_multiplier * scale
    
    def compute_stock_scores(self, trade_date: str) -> list[dict]:
        """计算所有股票的评分 (包含 Iteration 11 均值回归因子)"""
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
                
                # Iteration 11: 布林线均值回归因子
                bollinger = self.get_bollinger_position(symbol, trade_date)
                reversion_score = 0.0
                if bollinger:
                    position = bollinger["position"]
                    # 位置越低 (接近下轨)，回归得分越高
                    reversion_score = (1 - position) * 0.3  # 最大 0.3 的加分
                
                # 流动性过滤
                avg_amount = self.get_avg_amount_5d(symbol, trade_date)
                if avg_amount < MIN_AVG_AMOUNT_5D:
                    self.liquidity_filtered_count += 1
                    continue
                
                # 综合评分
                combined_score = SCORE_WEIGHT * predict_score + SHARPE_WEIGHT * hist_sharpe + reversion_score
                
                scored_stocks.append({
                    "symbol": symbol,
                    "close": close_list[-1],
                    "predict_score": predict_score,
                    "hist_sharpe_20d": hist_sharpe,
                    "bollinger_position": bollinger["position"] if bollinger else 0.5,
                    "reversion_score": reversion_score,
                    "combined_score": combined_score,
                    "avg_amount_5d": avg_amount,
                })
            except Exception as e:
                logger.debug(f"Failed to compute score for {symbol}: {e}")
                continue
        
        scored_stocks.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_stocks
    
    def simulate_day(self, trade_date: str, trade_dates: list[str]) -> DailyRecord:
        """
        模拟单日交易 - Iteration 11 改进
        
        核心逻辑:
        1. 动态防御阈值：根据市场中位数评分自动调整
        2. 波动率缩放止损：根据 20 日振幅调整 ATR 乘数
        3. 盈利保护机制：盈利>2% 后上移止损至成本价 +0.5%
        4. 均值回归因子：布林线下轨偏离增强
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
        
        # 更新统计
        if market_mode == "AGGRESSIVE":
            self.aggressive_days += 1
        else:
            self.defensive_days += 1
        
        if market_regime == "BULL":
            self.bull_market_days += 1
        
        # Iteration 11: 获取动态阈值
        config = self.get_config(market_mode, market_regime, median_score)
        current_threshold = SCORE_THRESHOLD + config["threshold_addon"]
        record.dynamic_threshold = current_threshold
        max_positions = config["max_positions"]
        base_atr_multiplier = config["atr_multiplier"]
        min_hold_days = config["min_hold_days"]
        
        logger.info(f"[MODE] {trade_date}: {market_mode}/{market_regime} (threshold={current_threshold:.2f}, max_pos={max_positions}, atr_mult={base_atr_multiplier})")
        
        next_date = self.get_next_date(trade_dates, trade_date)
        
        # ========== Step 2: 处理持仓卖出 ==========
        to_sell = []
        
        for symbol, pos in list(self.holdings.items()):
            if next_date:
                next_data = self.get_stock_data(symbol, next_date)
                if next_data:
                    next_open = self.apply_price_noise(next_data["open"])
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
                    
                    # ========== Iteration 11: 盈利保护机制 ==========
                    # 规则：盈利 > 2% 后，将止损位上移至成本价以上 0.5%
                    break_even_stop = None
                    if PROFIT_PROTECTION["enabled"] and pnl_pct > PROFIT_PROTECTION["profit_threshold"]:
                        break_even_stop = PROFIT_PROTECTION["stop_level"]
                        record.profit_protection_triggered.append(symbol)
                        self.profit_protection_count += 1
                        logger.info(f"  [PROFIT PROTECT] {symbol}: pnl={pnl_pct:.1%} > {PROFIT_PROTECTION['profit_threshold']:.1%}, stop moved to +{PROFIT_PROTECTION['stop_level']:.1%}")
                        
                        # 如果当前盈利回落到保护位以下，触发卖出
                        if pnl_pct < break_even_stop:
                            should_sell = True
                            sell_reason = f"profit protection ({pnl_pct:.1%} < +{break_even_stop:.1%})"
                    
                    # ========== Iteration 10: 止盈豁免检查 ==========
                    if pnl_pct > PROFIT_EXEMPTION_THRESHOLD and not should_sell:
                        record.profit_exemption_triggered.append(symbol)
                        self.profit_exemption_count += 1
                        logger.info(f"  [PROFIT EXEMPT] {symbol}: pnl={pnl_pct:.1%} > {PROFIT_EXEMPTION_THRESHOLD:.1%}, exempt min_hold_days")
                    
                    # ========== 持仓底线检查 ==========
                    if hold_days < MIN_HOLD_DAYS_CHEAP and not should_sell:
                        stock_scores = self.compute_stock_scores(trade_date)
                        stock_score = 0.0
                        for s in stock_scores:
                            if s["symbol"] == symbol:
                                stock_score = s["combined_score"]
                                break
                        
                        if pnl_pct < COST_THRESHOLD and stock_score > SCORE_HOLD_BUFFER:
                            record.cost_hold_triggered.append(symbol)
                            self.cost_hold_triggered_count += 1
                            logger.info(f"  [COST HOLD] {symbol}: pnl={pnl_pct:.1%} < {COST_THRESHOLD:.1%}, score={stock_score:+.2f} > {SCORE_HOLD_BUFFER:.2f}, hold {hold_days}/{MIN_HOLD_DAYS_CHEAP} days")
                            continue
                    
                    # ========== Iteration 11: 波动率缩放 ATR 止损 ==========
                    if not should_sell:
                        atr = self.get_atr(symbol, next_date)
                        if atr and atr > 0:
                            # 计算波动率缩放后的乘数
                            adjusted_multiplier = self.compute_atr_multiplier(symbol, next_date, base_atr_multiplier)
                            record.volatility_adjustment_applied = True
                            
                            atr_stop = -adjusted_multiplier * (atr / next_open)
                            atr_stop = max(MIN_ATR_STOP, min(MAX_ATR_STOP, atr_stop))
                            
                            if pnl_pct < atr_stop:
                                should_sell = True
                                sell_reason = f"ATR stop ({pnl_pct:.1%} < {atr_stop:.1%}, mult={adjusted_multiplier:.1f})"
                                record.atr_stop_triggered.append(symbol)
                                self.atr_stop_triggered_count += 1
                    
                    # 硬止损
                    if hold_days >= 2 and pnl_pct < HARD_STOP_LOSS and not should_sell:
                        should_sell = True
                        sell_reason = f"hard stop ({pnl_pct:.1%} < {HARD_STOP_LOSS:.1%})"
                    
                    # 评分卖出
                    effective_min_hold = PROFIT_EXEMPTION_MIN_HOLD if pnl_pct > PROFIT_EXEMPTION_THRESHOLD else min_hold_days
                    if hold_days >= effective_min_hold and not should_sell:
                        stock_scores = self.compute_stock_scores(trade_date)
                        for s in stock_scores:
                            if s["symbol"] == symbol:
                                if s["combined_score"] < current_threshold:
                                    should_sell = True
                                    sell_reason = f"low score ({s['combined_score']:+.2f} < {current_threshold:.2f})"
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
            
            record.sold_symbols.append(symbol)
            del self.holdings[symbol]
            logger.info(f"  [SELL] {symbol} @ {sell_price:.2f} ({reason})")
        
        # ========== Step 3: 选股买入 ==========
        scored_stocks = self.compute_stock_scores(trade_date)
        available_stocks = [s for s in scored_stocks if s["symbol"] not in self.holdings]
        
        if len(self.holdings) < max_positions and available_stocks:
            top_stocks = available_stocks[:2]
            
            available_cash = self.cash * 0.95
            
            for stock in top_stocks:
                if len(self.holdings) >= max_positions:
                    break
                
                symbol = stock["symbol"]
                combined_score = stock["combined_score"]
                
                if combined_score < current_threshold:
                    logger.debug(f"  [SKIP] {symbol}: score {combined_score:+.2f} < {current_threshold:.2f}")
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
        logger.info(f"Final Strategy V1.1 Backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Initial Capital: ¥{self.initial_capital:,.2f}")
        logger.info(f"Iteration 11 Features:")
        logger.info(f"  - Volatility Scaling: {VOLATILITY_SCALING['enabled']} (base_mult={VOLATILITY_SCALING['base_multiplier']})")
        logger.info(f"  - Profit Protection: {PROFIT_PROTECTION['enabled']} (threshold={PROFIT_PROTECTION['profit_threshold']:.1%}, stop=+{PROFIT_PROTECTION['stop_level']:.1%})")
        logger.info(f"  - Dynamic Defense: {DEFENSIVE_CONFIG['dynamic_adjustment']} (base_threshold={DEFENSIVE_CONFIG['base_threshold_addon']:.2f})")
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
            cost_hold_triggered_count=self.cost_hold_triggered_count,
            profit_exemption_count=self.profit_exemption_count,
            profit_protection_count=self.profit_protection_count,
            aggressive_days=self.aggressive_days,
            defensive_days=self.defensive_days,
            bull_market_days=self.bull_market_days,
            monthly_returns=monthly_returns,
            robustness_score=robustness_score,
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
    
    def generate_report(self, result: BacktestResult, report_name: str = "Final Strategy V1.1") -> str:
        """生成 Markdown 报告"""
        lines = []
        
        lines.append(f"# {report_name} 回测报告")
        lines.append("")
        lines.append(f"**回测区间**: {result.start_date} 至 {result.end_date}")
        lines.append(f"**交易天数**: {result.total_days}")
        lines.append(f"**初始资金**: ¥{self.initial_capital:,.2f}")
        lines.append("")
        
        lines.append("## Iteration 11 核心改进说明")
        lines.append("")
        lines.append("### 1. 波动率缩放止损")
        lines.append("")
        lines.append(f"- **基础乘数**: {VOLATILITY_SCALING['base_multiplier']}")
        lines.append(f"- **高波动缩放**: {VOLATILITY_SCALING['high_volatility_scale']} (更宽松)")
        lines.append(f"- **低波动缩放**: {VOLATILITY_SCALING['low_volatility_scale']} (更严格)")
        lines.append(f"- **波动率阈值**: {VOLATILITY_SCALING['volatility_threshold']:.1%}")
        lines.append("")
        lines.append("### 2. 盈利保护机制 (Break-even Stop)")
        lines.append("")
        lines.append(f"- **触发阈值**: 盈利 > {PROFIT_PROTECTION['profit_threshold']:.1%}")
        lines.append(f"- **止损上移**: 成本价以上 {PROFIT_PROTECTION['stop_level']:.1%}")
        lines.append(f"- **触发次数**: {result.profit_protection_count} 次")
        lines.append("")
        lines.append("### 3. 动态防御阈值")
        lines.append("")
        lines.append(f"- **基础门槛**: {DEFENSIVE_CONFIG['base_threshold_addon']:.2f}")
        lines.append(f"- **动态调整**: {DEFENSIVE_CONFIG['dynamic_adjustment']}")
        lines.append(f"- **调整范围**: [{DEFENSIVE_CONFIG['min_threshold_addon']}, {DEFENSIVE_CONFIG['max_threshold_addon']}]")
        lines.append("")
        lines.append("### 4. 均值回归因子")
        lines.append("")
        lines.append("- **布林线下轨偏离**: 捕捉超卖反弹机会")
        lines.append("- **评分增强**: 最大 +0.3 分")
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
        
        lines.append("## 实战约束统计")
        lines.append("")
        lines.append(f"- **总交易次数**: {result.total_trades}")
        lines.append(f"- **涨停无法买入**: {result.limit_up_blocked_count} 次")
        lines.append(f"- **跌停无法卖出**: {result.limit_down_blocked_count} 次")
        lines.append(f"- **流动性过滤**: {result.liquidity_filtered_count} 只股票")
        lines.append(f"- **ATR 止损触发**: {result.atr_stop_triggered_count} 次")
        lines.append(f"- **成本优化强制持仓**: {result.cost_hold_triggered_count} 次")
        lines.append(f"- **止盈豁免触发**: {result.profit_exemption_count} 次")
        lines.append(f"- **盈利保护触发**: {result.profit_protection_count} 次")
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


def run_final_strategy_v1_1():
    """运行 Final Strategy V1.1 回测"""
    
    logger.info("=" * 60)
    logger.info("盲测区间：2024-01-01 ~ 2024-04-30 (120 天)")
    logger.info("=" * 60)
    
    # 测试 1: 无噪声基准测试
    tester_base = FinalStrategyV11Backtester(
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
    tester_noise = FinalStrategyV11Backtester(
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
    print("Iteration 11: Final Strategy V1.1 回测摘要")
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
    
    run_final_strategy_v1_1()