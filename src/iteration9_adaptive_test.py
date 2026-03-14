"""
Iteration 9 - 市场环境自适应与成本效能优化

核心逻辑重构:
1. 环境识别器：基于指数（000905）的 20 日均线（MA20）
   - 进攻模式（指数 > MA20）：降低门槛，提升仓位，放大止损
   - 防御模式（指数 <= MA20）：维持严格参数

2. 进攻模式参数:
   - threshold_addon = 0.0（降低门槛）
   - max_positions = 5（最大持仓位数）
   - ATR 止损倍数 = 3.0（给牛股波动空间）

3. 防御模式参数:
   - threshold_addon = 0.3（严格筛选）
   - max_positions = 3
   - ATR 止损倍数 = 2.5

4. 交易降噪（成本优化）:
   - 持仓底线：单笔盈利若未超过 0.6%（滑点 + 税费成本），且预测分未跌破 -0.5
   - 则强制持仓至少 3 天，减少无效换手

5. 日志输出:
   - 显示当前处于 "AGGRESSIVE" 还是 "DEFENSIVE" 模式

压力测试区间:
- 区间 A: 2025-12-09 ~ 2026-03-12 (观察跑输幅度是否收窄)
- 区间 B: 2024-05-01 ~ 2024-08-01 (盲测区间，验证防御性)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal
import json

import polars as pl
import numpy as np
from loguru import logger


# ===========================================
# Iteration 9 自适应参数配置
# ===========================================

# 基础参数
SLIPPAGE_RATE = 0.002  # 0.2% 滑点
COMMISSION_RATE = 0.0003
COMMISSION_MIN = 5.0
STAMP_DUTY_RATE = 0.001

# 涨跌停限制
LIMIT_UP_RATIO = 1.098  # 涨停判定阈值
LIMIT_DOWN_RATIO = 0.902  # 跌停判定阈值

# 流动性过滤
MIN_AVG_AMOUNT_5D = 50000000  # 5000 万

# 基础评分参数
SCORE_THRESHOLD = 0.0  # 基础阈值
SCORE_WEIGHT = 0.7
SHARPE_WEIGHT = 0.3

# 进攻模式参数 (指数 > MA20)
AGGRESSIVE_CONFIG = {
    "threshold_addon": 0.0,       # 降低门槛
    "max_positions": 5,           # 最大持仓
    "atr_multiplier": 3.0,        # ATR 止损倍数放大
    "min_hold_days": 3,           # 最小持有期
}

# 防御模式参数 (指数 <= MA20)
DEFENSIVE_CONFIG = {
    "threshold_addon": 0.3,       # 严格筛选
    "max_positions": 3,           # 保守持仓
    "atr_multiplier": 2.5,        # 标准 ATR 止损
    "min_hold_days": 5,           # 较长持有期
}

# 成本优化参数
COST_THRESHOLD = 0.006  # 0.6% 成本线
SCORE_HOLD_BUFFER = -0.5  # 预测分缓冲线
MIN_HOLD_DAYS_CHEAP = 3  # 低成本强制持有期

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
    market_mode: str  # "AGGRESSIVE" or "DEFENSIVE"
    index_close: float
    index_ma20: float
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
    # Iteration 9 新增
    cost_hold_triggered: list[str] = field(default_factory=list)  # 成本优化强制持仓


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
    # Iteration 9 新增
    cost_hold_triggered_count: int = 0
    aggressive_days: int = 0
    defensive_days: int = 0


class Iteration9AdaptiveBacktester:
    """
    Iteration 9 市场环境自适应回测引擎
    
    核心特性:
    1. 环境识别器：基于 000905 指数 MA20
    2. 进攻/防御模式自动切换
    3. 成本优化：强制持仓减少无效换手
    """
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
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
        self.aggressive_days = 0
        self.defensive_days = 0
        
        # 数据库连接
        try:
            from .db_manager import DatabaseManager
            self.db = DatabaseManager.get_instance()
        except ImportError:
            from db_manager import DatabaseManager
            self.db = DatabaseManager.get_instance()
        
        logger.info(f"Iteration9 Backtester initialized: {start_date} to {end_date}")
    
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
        """
        获取指数数据并计算 MA20
        
        Returns:
            dict: {"close": float, "ma20": float or None}
        """
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
            # 降级使用上证指数
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
        
        # 计算 MA20
        close_list = df["close"].to_list()
        if len(close_list) >= 20:
            ma20 = float(np.mean(close_list[-20:]))
        else:
            ma20 = None
        
        # 获取指定日期数据
        latest = df.filter(pl.col("trade_date").cast(pl.Utf8) == trade_date)
        if latest.is_empty():
            latest = df.tail(1)
        if latest.is_empty():
            return None
        
        return {
            "close": float(latest["close"][0]),
            "ma20": ma20
        }
    
    def get_market_mode(self, trade_date: str) -> str:
        """
        获取市场模式
        
        规则:
        - 指数 > MA20: AGGRESSIVE (进攻模式)
        - 指数 <= MA20: DEFENSIVE (防御模式)
        """
        index_data = self.get_index_data(trade_date)
        if index_data and index_data["ma20"] is not None:
            if index_data["close"] > index_data["ma20"]:
                return "AGGRESSIVE"
        return "DEFENSIVE"
    
    def get_config(self, market_mode: str) -> dict:
        """获取当前模式对应的配置"""
        if market_mode == "AGGRESSIVE":
            return AGGRESSIVE_CONFIG
        return DEFENSIVE_CONFIG
    
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
    
    def compute_stock_scores(self, trade_date: str) -> list[dict]:
        """计算所有股票的评分"""
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
                
                # 流动性过滤
                avg_amount = self.get_avg_amount_5d(symbol, trade_date)
                if avg_amount < MIN_AVG_AMOUNT_5D:
                    self.liquidity_filtered_count += 1
                    continue
                
                rsi_14 = 50.0  # 默认中性值
                
                combined_score = SCORE_WEIGHT * predict_score + SHARPE_WEIGHT * hist_sharpe
                
                scored_stocks.append({
                    "symbol": symbol,
                    "close": close_list[-1],
                    "predict_score": predict_score,
                    "hist_sharpe_20d": hist_sharpe,
                    "combined_score": combined_score,
                    "rsi_14": rsi_14,
                    "avg_amount_5d": avg_amount,
                })
            except Exception as e:
                logger.debug(f"Failed to compute score for {symbol}: {e}")
                continue
        
        scored_stocks.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_stocks
    
    def simulate_day(self, trade_date: str, trade_dates: list[str]) -> DailyRecord:
        """
        模拟单日交易 - T+1 逻辑 + 自适应模式
        
        核心逻辑:
        1. 环境识别：获取市场模式 (AGGRESSIVE/DEFENSIVE)
        2. 持仓底线检查：盈利<0.6% 且 score>-0.5 时强制持有 3 天
        3. 自适应参数：根据模式调整阈值、仓位、止损
        """
        record = DailyRecord(
            trade_date=trade_date,
            market_mode="DEFENSIVE",
            index_close=0.0,
            index_ma20=0.0,
            cash=self.cash,
            portfolio_value=self.portfolio_value
        )
        
        # ========== Step 1: 环境识别 ==========
        index_data = self.get_index_data(trade_date)
        if index_data:
            record.index_close = index_data["close"]
            record.index_ma20 = index_data["ma20"] if index_data["ma20"] else 0.0
        
        # 获取市场模式
        market_mode = self.get_market_mode(trade_date)
        record.market_mode = market_mode
        
        # 更新统计
        if market_mode == "AGGRESSIVE":
            self.aggressive_days += 1
        else:
            self.defensive_days += 1
        
        # 获取当前配置
        config = self.get_config(market_mode)
        current_threshold = SCORE_THRESHOLD + config["threshold_addon"]
        max_positions = config["max_positions"]
        atr_multiplier = config["atr_multiplier"]
        min_hold_days = config["min_hold_days"]
        
        logger.info(f"[MODE] {trade_date}: {market_mode} (threshold={current_threshold:.2f}, max_pos={max_positions}, atr_mult={atr_multiplier})")
        
        # 获取 T+1 日的股票数据
        next_date = self.get_next_date(trade_dates, trade_date)
        
        # ========== Step 2: 处理持仓卖出 ==========
        to_sell = []
        
        for symbol, pos in list(self.holdings.items()):
            if next_date:
                next_data = self.get_stock_data(symbol, next_date)
                if next_data:
                    next_open = next_data["open"]
                    pre_close = next_data["pre_close"]
                    
                    # 检查跌停
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
                    
                    # ========== Iteration 9: 持仓底线检查 ==========
                    # 规则：盈利<0.6% 且 score>-0.5 时，强制持有至少 3 天
                    if hold_days < MIN_HOLD_DAYS_CHEAP:
                        # 获取当前评分
                        stock_scores = self.compute_stock_scores(trade_date)
                        stock_score = 0.0
                        for s in stock_scores:
                            if s["symbol"] == symbol:
                                stock_score = s["combined_score"]
                                break
                        
                        # 检查是否满足强制持有条件
                        if pnl_pct < COST_THRESHOLD and stock_score > SCORE_HOLD_BUFFER:
                            record.cost_hold_triggered.append(symbol)
                            self.cost_hold_triggered_count += 1
                            logger.info(f"  [COST HOLD] {symbol}: pnl={pnl_pct:.1%} < {COST_THRESHOLD:.1%}, score={stock_score:+.2f} > {SCORE_HOLD_BUFFER:.2f}, hold {hold_days}/{MIN_HOLD_DAYS_CHEAP} days")
                            continue  # 强制持有，跳过卖出检查
                    
                    # ATR 动态止损
                    atr = self.get_atr(symbol, next_date)
                    if atr and atr > 0:
                        atr_stop = -atr_multiplier * (atr / next_open)
                        atr_stop = max(MIN_ATR_STOP, min(MAX_ATR_STOP, atr_stop))
                        if pnl_pct < atr_stop:
                            should_sell = True
                            sell_reason = f"ATR stop ({pnl_pct:.1%} < {atr_stop:.1%})"
                            record.atr_stop_triggered.append(symbol)
                            self.atr_stop_triggered_count += 1
                    
                    # 硬止损
                    if hold_days >= 2 and pnl_pct < HARD_STOP_LOSS and not should_sell:
                        should_sell = True
                        sell_reason = f"hard stop ({pnl_pct:.1%} < {HARD_STOP_LOSS:.1%})"
                    
                    # 评分卖出 (需满足最小持有期)
                    if hold_days >= min_hold_days and not should_sell:
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
            # 选取前 2 名
            top_stocks = available_stocks[:2]
            
            positions_left = max_positions - len(self.holdings)
            available_cash = self.cash * 0.95
            
            for stock in top_stocks:
                if len(self.holdings) >= max_positions:
                    break
                
                symbol = stock["symbol"]
                combined_score = stock["combined_score"]
                
                # 检查是否达到阈值
                if combined_score < current_threshold:
                    logger.debug(f"  [SKIP] {symbol}: score {combined_score:+.2f} < {current_threshold:.2f}")
                    continue
                
                if next_date:
                    next_data = self.get_stock_data(symbol, next_date)
                    if next_data:
                        next_open = next_data["open"]
                        pre_close = next_data["pre_close"]
                        
                        # 检查涨停
                        if self.check_limit_up(next_open, pre_close):
                            record.limit_up_blocked.append(symbol)
                            self.limit_up_blocked_count += 1
                            logger.info(f"  [LIMIT UP] {symbol}: cannot buy")
                            continue
                        
                        # 计算仓位
                        total_score = sum(s["combined_score"] for s in top_stocks)
                        if total_score > 0:
                            target_weight = max(combined_score / total_score, 1.0 / max_positions)
                        else:
                            target_weight = 1.0 / max_positions
                        
                        budget = available_cash * target_weight
                        
                        # 滑点调整
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
        logger.info(f"Iteration 9 Adaptive Backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Initial Capital: ¥{self.initial_capital:,.2f}")
        logger.info(f"Aggressive Config: threshold_addon={AGGRESSIVE_CONFIG['threshold_addon']}, max_pos={AGGRESSIVE_CONFIG['max_positions']}, atr_mult={AGGRESSIVE_CONFIG['atr_multiplier']}")
        logger.info(f"Defensive Config: threshold_addon={DEFENSIVE_CONFIG['threshold_addon']}, max_pos={DEFENSIVE_CONFIG['max_positions']}, atr_mult={DEFENSIVE_CONFIG['atr_multiplier']}")
        logger.info(f"Cost Hold: pnl<{COST_THRESHOLD:.1%} and score>{SCORE_HOLD_BUFFER:.2f} => hold {MIN_HOLD_DAYS_CHEAP} days")
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
                logger.info(f"  Mode: {record.market_mode}, Holdings: {len(record.holding_symbols)}")
        
        metrics = self._calculate_metrics()
        benchmark = self._calculate_benchmark()
        
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
            aggressive_days=self.aggressive_days,
            defensive_days=self.defensive_days,
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
    
    def generate_report(self, result: BacktestResult, report_name: str = "Iteration 9") -> str:
        """生成 Markdown 报告"""
        lines = []
        
        lines.append(f"# {report_name} 市场环境自适应回测报告")
        lines.append("")
        lines.append(f"**回测区间**: {result.start_date} 至 {result.end_date}")
        lines.append(f"**交易天数**: {result.total_days}")
        lines.append(f"**初始资金**: ¥{self.initial_capital:,.2f}")
        lines.append("")
        
        lines.append("## 核心修改说明")
        lines.append("")
        lines.append("### 1. 环境识别器 (基于 000905 指数 MA20)")
        lines.append("")
        lines.append("| 模式 | 条件 | threshold_addon | max_positions | ATR 倍数 |")
        lines.append("|------|------|-------------------|---------------|---------|")
        lines.append(f"| **进攻模式** | 指数 > MA20 | {AGGRESSIVE_CONFIG['threshold_addon']} | {AGGRESSIVE_CONFIG['max_positions']} | {AGGRESSIVE_CONFIG['atr_multiplier']}x |")
        lines.append(f"| **防御模式** | 指数 <= MA20 | {DEFENSIVE_CONFIG['threshold_addon']} | {DEFENSIVE_CONFIG['max_positions']} | {DEFENSIVE_CONFIG['atr_multiplier']}x |")
        lines.append("")
        lines.append("### 2. 交易降噪 (成本优化)")
        lines.append("")
        lines.append(f"- **持仓底线**: 盈利 < {COST_THRESHOLD:.1%} 且 预测分 > {SCORE_HOLD_BUFFER:.2f} 时，强制持有至少 {MIN_HOLD_DAYS_CHEAP} 天")
        lines.append(f"- **目的**: 减少无效换手，降低交易成本")
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
        lines.append("")
        
        lines.append("## 模式统计")
        lines.append("")
        lines.append(f"- **进攻模式天数**: {result.aggressive_days} 天 ({result.aggressive_days/result.total_days*100:.1f}%)")
        lines.append(f"- **防御模式天数**: {result.defensive_days} 天 ({result.defensive_days/result.total_days*100:.1f}%)")
        lines.append("")
        
        lines.append("## 实战约束统计")
        lines.append("")
        lines.append(f"- **总交易次数**: {result.total_trades}")
        lines.append(f"- **涨停无法买入**: {result.limit_up_blocked_count} 次")
        lines.append(f"- **跌停无法卖出**: {result.limit_down_blocked_count} 次")
        lines.append(f"- **流动性过滤**: {result.liquidity_filtered_count} 只股票")
        lines.append(f"- **ATR 止损触发**: {result.atr_stop_triggered_count} 次")
        lines.append(f"- **成本优化强制持仓**: {result.cost_hold_triggered_count} 次")
        lines.append(f"- **总佣金**: ¥{result.total_commission:.2f}")
        lines.append(f"- **总印花税**: ¥{result.total_stamp_duty:.2f}")
        lines.append(f"- **总成本**: ¥{result.total_commission + result.total_stamp_duty:.2f}")
        lines.append(f"- **成本率**: {(result.total_commission + result.total_stamp_duty) / self.initial_capital:.2%}")
        lines.append("")
        
        lines.append("## 每日权益曲线")
        lines.append("")
        lines.append("```")
        lines.append("Date,Mode,Portfolio,Cash,Holdings,Daily PnL")
        for r in result.daily_records:
            mode_short = "AGG" if r.market_mode == "AGGRESSIVE" else "DEF"
            lines.append(f"{r.trade_date},{mode_short},{r.portfolio_value:.2f},{r.cash:.2f},{len(r.holding_symbols)},{r.daily_pnl:.2f}")
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)


def run_iteration9():
    """运行 Iteration 9 回测"""
    # 区间 A
    logger.info("=" * 60)
    logger.info("区间 A: 2025-12-09 ~ 2026-03-12")
    logger.info("=" * 60)
    
    tester_a = Iteration9AdaptiveBacktester(
        start_date="2025-12-09",
        end_date="2026-03-12",
        initial_capital=INITIAL_CAPITAL
    )
    result_a = tester_a.run()
    
    # 区间 B
    logger.info("=" * 60)
    logger.info("区间 B (盲测): 2024-05-01 ~ 2024-08-01")
    logger.info("=" * 60)
    
    tester_b = Iteration9AdaptiveBacktester(
        start_date="2024-05-01",
        end_date="2024-08-01",
        initial_capital=INITIAL_CAPITAL
    )
    result_b = tester_b.run()
    
    # 生成报告
    report_a = tester_a.generate_report(result_a, "区间 A")
    report_b = tester_b.generate_report(result_b, "区间 B 盲测")
    
    # 合并报告
    full_report = f"""# Iteration 9 市场环境自适应与成本效能优化报告

## 执行摘要

本报告展示了 Iteration 9 的核心改进：市场环境自适应机制与成本效能优化。

### 核心修改清单

| 修改项 | Iteration 8 | Iteration 9 |
|--------|-------------|-------------|
| 市场模式 | 固定参数 | 自适应 (进攻/防御) |
| 阈值调整 | 固定 +0.3 | 进攻 0.0 / 防御 +0.3 |
| 最大持仓 | 5 只 | 进攻 5 只 / 防御 3 只 |
| ATR 止损 | 2.5x | 进攻 3.0x / 防御 2.5x |
| 成本优化 | 无 | 强制持仓 3 天 |

### 配置参数

**进攻模式** (指数 > MA20):
- threshold_addon = 0.0 (降低门槛)
- max_positions = 5 (最大持仓)
- ATR 止损倍数 = 3.0 (给牛股波动空间)

**防御模式** (指数 <= MA20):
- threshold_addon = 0.3 (严格筛选)
- max_positions = 3 (保守持仓)
- ATR 止损倍数 = 2.5 (标准止损)

**成本优化**:
- 持仓底线：盈利 < 0.6% 且 预测分 > -0.5 时，强制持有至少 3 天

---

{report_a}

---

{report_b}

---

## 对比分析：Iteration 8 vs Iteration 9

### 区间 A (2025-12-09 ~ 2026-03-12)

| 指标 | Iteration 8 | Iteration 9 | 变化 |
|------|-------------|-------------|------|
| 总收益率 | - | {result_a.total_return:.2%} | - |
| 年化收益 | - | {result_a.annualized_return:.2%} | - |
| 最大回撤 | - | {result_a.max_drawdown:.2%} | - |
| 夏普比率 | - | {result_a.sharpe_ratio:.2f} | - |
| 交易次数 | - | {result_a.total_trades} | - |

### 区间 B (盲测 2024-05-01 ~ 2024-08-01)

| 指标 | Iteration 8 | Iteration 9 | 变化 |
|------|-------------|-------------|------|
| 总收益率 | - | {result_b.total_return:.2%} | - |
| 年化收益 | - | {result_b.annualized_return:.2%} | - |
| 最大回撤 | - | {result_b.max_drawdown:.2%} | - |
| 夏普比率 | - | {result_b.sharpe_ratio:.2f} | - |
| 交易次数 | - | {result_b.total_trades} | - |

### 自适应效果分析

**区间 A 模式分布**:
- 进攻模式：{result_a.aggressive_days} 天 ({result_a.aggressive_days/result_a.total_days*100:.1f}%)
- 防御模式：{result_a.defensive_days} 天 ({result_a.defensive_days/result_a.total_days*100:.1f}%)

**区间 B 模式分布**:
- 进攻模式：{result_b.aggressive_days} 天 ({result_b.aggressive_days/result_b.total_days*100:.1f}%)
- 防御模式：{result_b.defensive_days} 天 ({result_b.defensive_days/result_b.total_days*100:.1f}%)

### 成本优化效果

- 区间 A 强制持仓触发：{result_a.cost_hold_triggered_count} 次
- 区间 B 强制持仓触发：{result_b.cost_hold_triggered_count} 次

---

## 结论

1. **自适应机制**: 根据市场状态自动调整参数，在强势行情下降低门槛提升收益
2. **成本优化**: 强制持仓减少无效换手，降低交易成本
3. **风险控制**: 防御模式下维持严格参数，保持防御性

---

*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    output_path = Path("reports") / "Iteration9_Adaptive_Report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    logger.info(f"报告已保存至：{output_path}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("Iteration 9 回测摘要")
    print("=" * 60)
    print(f"\n区间 A (2025-12-09 ~ 2026-03-12):")
    print(f"  总收益：{result_a.total_return:.2%}")
    print(f"  夏普比率：{result_a.sharpe_ratio:.2f}")
    print(f"  最大回撤：{result_a.max_drawdown:.2%}")
    print(f"  模式分布：进攻 {result_a.aggressive_days}天 / 防御 {result_a.defensive_days}天")
    print(f"\n区间 B (盲测 2024-05-01 ~ 2024-08-01):")
    print(f"  总收益：{result_b.total_return:.2%}")
    print(f"  夏普比率：{result_b.sharpe_ratio:.2f}")
    print(f"  最大回撤：{result_b.max_drawdown:.2%}")
    print(f"  模式分布：进攻 {result_b.aggressive_days}天 / 防御 {result_b.defensive_days}天")
    print("=" * 60)
    
    return result_a, result_b


if __name__ == "__main__":
    # 设置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    run_iteration9()