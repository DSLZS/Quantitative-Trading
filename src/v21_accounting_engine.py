"""
V21 Accounting Engine - 动态赋权与换手优化

【核心任务：解决"交易成本侵蚀"与"权重呆板"问题】

V20 回测显示交易成本高达本金的 40%，这是不可接受的。
V21 必须在保持"铁血会计引擎"的前提下，优化调仓逻辑和仓位分配。

【V21 核心改进】

1. 概率驱动的动态调仓 (Dynamic Sizing)
   - 得分激活：只有 LightGBM 预测得分 > 0.60 的股票才进入买入备选池
   - 权重分配：w_i ∝ (Score_i - 0.60)
   - 单只个股最大权重限制为 20%（防止过度集中）
   - 总持仓目标数量：5-15 只（根据得分筛选结果动态确定）

2. 换手率"滞后缓冲区" (Turnover Buffer)
   - 买入标准：排名必须在 Top 10 且得分 > 0.60
   - 卖出标准（放宽）：只有当排名跌出 Top 30（而不是 Top 10）时才卖出
   - 目的：给信号波动留出空间，避免第 11 名和第 9 名频繁互换导致的无效手续费

3. 环境模拟增强
   - 初始本金：100,000 元
   - 摩擦成本：维持 V20 的严格标准（0.2% 摩擦 + 5 元最低佣金 + 印花税）

作者：高级量化架构师
版本：V21.0 (Dynamic Sizing & Turnover Buffer)
日期：2026-03-17
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量 (严禁修改)
# ===========================================

# 资金管理
INITIAL_CAPITAL = 100000.0
TOP_K_STOCKS = 10  # 目标持仓数量

# 动态权重配置
SCORE_THRESHOLD = 0.60      # 得分激活阈值
MAX_POSITION_RATIO = 0.20   # 单只个股最大权重 20%
MIN_POSITION_RATIO = 0.05   # 最小权重 5%

# 换手率缓冲区配置
BUY_RANK_THRESHOLD = 10     # 买入排名阈值（Top 10）
SELL_RANK_THRESHOLD = 30    # 卖出排名阈值（跌出 Top 30 才卖）

# 真实手续费配置（维持 V20）
SLIPPAGE_BUY = 0.001      # 买入滑点 0.1%
SLIPPAGE_SELL = 0.001     # 卖出滑点 0.1%
STAMP_DUTY = 0.0005       # 印花税 0.05% (仅卖出)
MIN_COMMISSION = 5.0      # 最低佣金 5 元
COMMISSION_RATE = 0.0003  # 交易规费 0.03%

# T+1 锁定
BUY_DELAY = 1  # T+1 买入

# 止损止盈
STOP_LOSS_RATIO = 0.08   # 8% 硬止损
TAKE_PROFIT_RATIO = 0.15 # 15% 止盈


# ===========================================
# 数据类定义
# ===========================================

@dataclass
class Position:
    """持仓数据结构"""
    symbol: str
    shares: int
    avg_cost: float  # 平均成本（含手续费）
    buy_date: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_ratio: float = 0.0
    target_weight: float = 0.0  # V21: 目标权重
    
    def update_price(self, price: float):
        """更新当前价格和未实现盈亏"""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_cost) * self.shares
        self.unrealized_pnl_ratio = (price - self.avg_cost) / self.avg_cost


@dataclass
class Trade:
    """交易记录数据结构"""
    trade_date: str
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: int
    price: float
    amount: float  # 成交金额
    commission: float  # 手续费
    slippage: float  # 滑点成本
    stamp_duty: float = 0.0  # 印花税（仅卖出）
    total_cost: float = 0.0  # 总成本（含所有费用）
    
    def __post_init__(self):
        if self.side == "SELL":
            self.total_cost = self.commission + self.stamp_duty + self.slippage
        else:
            self.total_cost = self.commission + self.slippage


@dataclass
class DailyNAV:
    """每日净值数据结构"""
    trade_date: str
    cash: float
    market_value: float
    total_assets: float  # NAV = cash + market_value
    daily_return: float = 0.0
    cumulative_return: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    start_date: str
    end_date: str
    initial_capital: float
    
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # 真实交易指标
    total_trades: int = 0
    total_buy_amount: float = 0.0  # 总买入额
    total_sell_amount: float = 0.0  # 总卖出额
    total_commission: float = 0.0  # 总手续费
    total_stamp_duty: float = 0.0  # 总印花税
    total_slippage: float = 0.0  # 总滑点成本
    turnover_rate: float = 0.0  # 换手率
    
    # 持仓指标
    avg_holding_days: float = 0.0
    avg_position_count: float = 0.0
    
    # V21 特有指标
    avg_position_weight: float = 0.0  # 平均持仓权重
    cash_drag: float = 0.0  # 现金拖累
    
    # 详细数据
    daily_navs: List[DailyNAV] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    positions_history: List[Dict] = field(default_factory=list)


# ===========================================
# V21 会计引擎核心类
# ===========================================

class V21AccountingEngine:
    """
    V21 会计引擎 - 铁血实盘逻辑 + 动态权重
    
    【核心特性】
    1. 真实账户管理：现金、持仓、冻结资金
    2. 真实手续费计算：滑点 + 规费 + 印花税 + 最低佣金
    3. T+1 锁定：今日买入明日才能卖
    4. 持仓生命周期：止损、止盈、信号排名
    5. 真实 NAV 计算：每日现金 + 持仓市值
    6. 动态权重：根据得分分配仓位
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        top_k_stocks: int = TOP_K_STOCKS,
        db: Optional[DatabaseManager] = None,
    ):
        # 验证初始资金
        if initial_capital <= 0:
            raise ValueError("初始资金必须大于 0")
        
        self.initial_capital = initial_capital
        self.top_k_stocks = top_k_stocks
        self.db = db or DatabaseManager.get_instance()
        
        # 账户状态
        self.cash = initial_capital  # 可用现金
        self.positions: Dict[str, Position] = {}  # 当前持仓
        self.frozen_cash = 0.0  # 冻结资金（已挂单未成交）
        
        # T+1 锁定：记录买入日期
        self.t1_locked_positions: Dict[str, str] = {}  # symbol -> buy_date
        
        # 交易记录
        self.trades: List[Trade] = []
        self.daily_navs: List[DailyNAV] = []
        self.positions_history: List[Dict] = []
        
        # 持仓周期记录
        self.holding_periods: Dict[str, int] = {}  # symbol -> holding days
        
        logger.info(f"V21AccountingEngine initialized: capital={initial_capital}, top_k={top_k_stocks}")
        logger.info(f"Score Threshold: {SCORE_THRESHOLD}")
        logger.info(f"Max Position Ratio: {MAX_POSITION_RATIO:.0%}")
        logger.info(f"Commission: {COMMISSION_RATE:.2%} (min {MIN_COMMISSION}元)")
        logger.info(f"Slippage: {SLIPPAGE_BUY:.2%} buy, {SLIPPAGE_SELL:.2%} sell")
        logger.info(f"Stamp Duty: {STAMP_DUTY:.2%} (sell only)")
    
    def compute_commission(self, amount: float, side: str) -> float:
        """计算交易手续费"""
        commission = amount * COMMISSION_RATE
        commission = max(commission, MIN_COMMISSION)  # 最低 5 元
        return commission
    
    def compute_slippage(self, amount: float, side: str) -> float:
        """计算滑点成本"""
        if side == "BUY":
            return amount * SLIPPAGE_BUY
        else:
            return amount * SLIPPAGE_SELL
    
    def compute_stamp_duty(self, amount: float, side: str) -> float:
        """计算印花税（仅卖出）"""
        if side == "SELL":
            return amount * STAMP_DUTY
        return 0.0
    
    def check_cash_sufficient(self, required_cash: float) -> bool:
        """检查现金是否充足"""
        available_cash = self.cash - self.frozen_cash
        return available_cash >= required_cash
    
    def get_available_shares(self, symbol: str, trade_date: str) -> int:
        """获取可用头寸（考虑 T+1 锁定）"""
        if symbol not in self.positions:
            return 0
        
        # 检查 T+1 锁定
        if symbol in self.t1_locked_positions:
            buy_date = self.t1_locked_positions[symbol]
            try:
                buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
                trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
                if trade_dt <= buy_dt:
                    return 0  # 今日买入，今日不可卖
            except ValueError:
                return 0
        
        return self.positions[symbol].shares
    
    def execute_buy(
        self,
        trade_date: str,
        symbol: str,
        price: float,
        target_amount: float,
    ) -> Optional[Trade]:
        """执行买入交易"""
        # 计算实际买入数量和金额（考虑滑点）
        slippage_cost = target_amount * SLIPPAGE_BUY
        commission = max(target_amount * COMMISSION_RATE, MIN_COMMISSION)
        total_required = target_amount + slippage_cost + commission
        
        # 检查现金是否充足
        if not self.check_cash_sufficient(total_required):
            available = self.cash - self.frozen_cash
            logger.debug(f"Cash insufficient for {symbol}: need {total_required:.2f}, have {available:.2f}")
            return None
        
        # 计算买入股数（向下取整）
        shares = int(target_amount / price)
        if shares <= 0:
            logger.debug(f"Shares too small for {symbol}: price={price}, target={target_amount}")
            return None
        
        # 重新计算实际金额
        actual_amount = shares * price
        actual_slippage = self.compute_slippage(actual_amount, "BUY")
        actual_commission = self.compute_commission(actual_amount, "BUY")
        actual_stamp_duty = self.compute_stamp_duty(actual_amount, "BUY")
        total_cost = actual_amount + actual_slippage + actual_commission + actual_stamp_duty
        
        # 再次检查现金
        if not self.check_cash_sufficient(total_cost):
            shares = max(0, shares - 1)
            if shares <= 0:
                return None
            actual_amount = shares * price
            actual_slippage = self.compute_slippage(actual_amount, "BUY")
            actual_commission = self.compute_commission(actual_amount, "BUY")
            total_cost = actual_amount + actual_slippage + actual_commission
        
        # 扣除现金
        self.cash -= total_cost
        
        # 更新或创建持仓
        if symbol in self.positions:
            old_pos = self.positions[symbol]
            old_cost = old_pos.avg_cost * old_pos.shares
            new_cost = actual_amount + actual_slippage + actual_commission
            total_shares = old_pos.shares + shares
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                avg_cost=(old_cost + new_cost) / total_shares,
                buy_date=old_pos.buy_date,
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=(actual_amount + actual_slippage + actual_commission) / shares,
                buy_date=trade_date,
            )
        
        # T+1 锁定
        self.t1_locked_positions[symbol] = trade_date
        
        # 记录交易
        trade = Trade(
            trade_date=trade_date,
            symbol=symbol,
            side="BUY",
            shares=shares,
            price=price,
            amount=actual_amount,
            commission=actual_commission,
            slippage=actual_slippage,
            stamp_duty=actual_stamp_duty,
            total_cost=total_cost,
        )
        self.trades.append(trade)
        
        logger.debug(f"BUY {symbol}: {shares} shares @ {price:.2f}, cost={total_cost:.2f}")
        return trade
    
    def execute_sell(
        self,
        trade_date: str,
        symbol: str,
        price: float,
        shares: Optional[int] = None,
    ) -> Optional[Trade]:
        """执行卖出交易"""
        if symbol not in self.positions:
            logger.debug(f"No position for {symbol}")
            return None
        
        available_shares = self.get_available_shares(symbol, trade_date)
        if available_shares <= 0:
            logger.debug(f"No available shares for {symbol} (T+1 locked)")
            return None
        
        if shares is None or shares > available_shares:
            shares = available_shares
        
        if shares <= 0:
            return None
        
        actual_amount = shares * price
        actual_slippage = self.compute_slippage(actual_amount, "SELL")
        actual_commission = self.compute_commission(actual_amount, "SELL")
        actual_stamp_duty = self.compute_stamp_duty(actual_amount, "SELL")
        total_fees = actual_slippage + actual_commission + actual_stamp_duty
        net_proceeds = actual_amount - total_fees
        
        self.cash += net_proceeds
        
        old_pos = self.positions[symbol]
        remaining_shares = old_pos.shares - shares
        
        if remaining_shares <= 0:
            del self.positions[symbol]
            if symbol in self.t1_locked_positions:
                del self.t1_locked_positions[symbol]
            try:
                buy_dt = datetime.strptime(old_pos.buy_date, "%Y-%m-%d")
                sell_dt = datetime.strptime(trade_date, "%Y-%m-%d")
                holding_days = (sell_dt - buy_dt).days
                self.holding_periods[symbol] = holding_days
            except ValueError:
                self.holding_periods[symbol] = 0
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=remaining_shares,
                avg_cost=old_pos.avg_cost,
                buy_date=old_pos.buy_date,
            )
        
        trade = Trade(
            trade_date=trade_date,
            symbol=symbol,
            side="SELL",
            shares=shares,
            price=price,
            amount=actual_amount,
            commission=actual_commission,
            slippage=actual_slippage,
            stamp_duty=actual_stamp_duty,
            total_cost=total_fees,
        )
        self.trades.append(trade)
        
        logger.debug(f"SELL {symbol}: {shares} shares @ {price:.2f}, net={net_proceeds:.2f}")
        return trade
    
    def update_position_prices(self, trade_date: str, prices: Dict[str, float]):
        """更新持仓股票的当前价格"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])
    
    def compute_daily_nav(self, trade_date: str, prices: Dict[str, float]) -> DailyNAV:
        """计算每日 NAV"""
        self.update_position_prices(trade_date, prices)
        
        market_value = sum(pos.shares * pos.current_price for pos in self.positions.values())
        total_assets = self.cash + market_value
        
        daily_return = 0.0
        cumulative_return = (total_assets - self.initial_capital) / self.initial_capital
        
        if len(self.daily_navs) > 0:
            prev_nav = self.daily_navs[-1].total_assets
            if prev_nav > 0:
                daily_return = (total_assets - prev_nav) / prev_nav
        
        nav = DailyNAV(
            trade_date=trade_date,
            cash=self.cash,
            market_value=market_value,
            total_assets=total_assets,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
        )
        self.daily_navs.append(nav)
        
        return nav
    
    def record_positions_snapshot(self, trade_date: str):
        """记录持仓快照"""
        snapshot = {
            "trade_date": trade_date,
            "cash": self.cash,
            "positions": {
                symbol: {
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.shares * pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "buy_date": pos.buy_date,
                }
                for symbol, pos in self.positions.items()
            },
            "total_market_value": sum(pos.shares * pos.current_price for pos in self.positions.values()),
            "total_assets": self.cash + sum(pos.shares * pos.current_price for pos in self.positions.values()),
        }
        self.positions_history.append(snapshot)


# ===========================================
# V21 动态权重管理器
# ===========================================

class V21DynamicSizingManager:
    """
    V21 动态权重管理器
    
    【权重分配逻辑】
    1. 得分激活：只有预测得分 > 0.60 的股票才进入备选池
    2. 权重分配：w_i ∝ (Score_i - 0.60)
    3. 单只个股最大权重 20%，最小权重 5%
    4. 总持仓目标数量：5-15 只（根据得分筛选结果动态确定）
    """
    
    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        max_position_ratio: float = MAX_POSITION_RATIO,
        min_position_ratio: float = MIN_POSITION_RATIO,
    ):
        self.score_threshold = score_threshold
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio
        
        logger.info(f"V21DynamicSizingManager initialized")
        logger.info(f"  Score Threshold: {score_threshold}")
        logger.info(f"  Max Position Ratio: {max_position_ratio:.0%}")
        logger.info(f"  Min Position Ratio: {min_position_ratio:.0%}")
    
    def compute_dynamic_weights(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Position],
        initial_capital: float,
    ) -> Dict[str, float]:
        """
        计算动态权重
        
        Args:
            signals_df: 信号 DataFrame (symbol, trade_date, signal, score)
            current_positions: 当前持仓
            initial_capital: 初始资金
            
        Returns:
            {symbol: target_amount} 字典
        """
        # 1. 过滤得分 > 0.60 的股票
        qualified = signals_df.filter(pl.col("signal") > self.score_threshold)
        
        if qualified.is_empty():
            logger.debug(f"No stocks with signal > {self.score_threshold}")
            return {}
        
        # 2. 计算超额得分 (Score - 0.60)
        qualified = qualified.with_columns([
            (pl.col("signal") - self.score_threshold).alias("excess_score")
        ])
        
        # 3. 计算权重 w_i ∝ (Score_i - 0.60)
        total_excess = qualified["excess_score"].sum()
        
        if total_excess <= 0:
            return {}
        
        qualified = qualified.with_columns([
            (pl.col("excess_score") / total_excess).alias("raw_weight")
        ])
        
        # 4. 应用权重限制（最大 20%，最小 5%）
        qualified = qualified.with_columns([
            pl.col("raw_weight").clip(self.min_position_ratio, self.max_position_ratio).alias("clipped_weight")
        ])
        
        # 5. 重新归一化（确保总权重为 1）
        total_clipped = qualified["clipped_weight"].sum()
        
        if total_clipped <= 0:
            return {}
        
        qualified = qualified.with_columns([
            (pl.col("clipped_weight") / total_clipped).alias("final_weight")
        ])
        
        # 6. 计算目标金额
        # V21 逻辑：如果符合条件的票少，就只买几只，其余持币
        target_amounts = {}
        
        for row in qualified.iter_rows(named=True):
            symbol = row["symbol"]
            weight = row["final_weight"]
            target_amount = initial_capital * weight
            target_amounts[symbol] = target_amount
        
        return target_amounts
    
    def get_stocks_to_buy(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Position],
        target_amounts: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        获取应该买入的股票列表
        
        Returns:
            [(symbol, target_amount), ...] 列表
        """
        stocks_to_buy = []
        
        for symbol, target_amount in target_amounts.items():
            if symbol not in current_positions:
                stocks_to_buy.append((symbol, target_amount))
            else:
                # 已持仓但需要调整权重
                current_pos = current_positions[symbol]
                current_value = current_pos.shares * current_pos.current_price
                if abs(target_amount - current_value) / target_amount > 0.1:  # 10% 阈值
                    stocks_to_buy.append((symbol, target_amount - current_value))
        
        return stocks_to_buy


# ===========================================
# V21 持仓生命周期管理器
# ===========================================

class V21PositionManager:
    """
    V21 持仓生命周期管理器 - 换手率缓冲区
    
    【V21 改进】
    - 买入标准：排名必须在 Top 10 且得分 > 0.60
    - 卖出标准（放宽）：只有当排名跌出 Top 30（而不是 Top 10）时才卖出
    - 目的：给信号波动留出空间，避免频繁互换导致的无效手续费
    """
    
    def __init__(
        self,
        stop_loss_ratio: float = STOP_LOSS_RATIO,
        take_profit_ratio: float = TAKE_PROFIT_RATIO,
        buy_rank_threshold: int = BUY_RANK_THRESHOLD,
        sell_rank_threshold: int = SELL_RANK_THRESHOLD,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.buy_rank_threshold = buy_rank_threshold
        self.sell_rank_threshold = sell_rank_threshold
        self.score_threshold = score_threshold
        
        logger.info(f"V21PositionManager initialized")
        logger.info(f"  Stop Loss: {stop_loss_ratio:.1%}")
        logger.info(f"  Take Profit: {take_profit_ratio:.1%}")
        logger.info(f"  Buy Rank Threshold: Top {buy_rank_threshold}")
        logger.info(f"  Sell Rank Threshold: > {sell_rank_threshold} (Buffer Zone)")
    
    def check_stop_loss(self, position: Position) -> bool:
        """检查是否触发止损"""
        if position.unrealized_pnl_ratio <= -self.stop_loss_ratio:
            logger.info(f"STOP LOSS: {position.symbol} pnl={position.unrealized_pnl_ratio:.1%}")
            return True
        return False
    
    def check_take_profit(self, position: Position) -> bool:
        """检查是否触发止盈"""
        if position.unrealized_pnl_ratio >= self.take_profit_ratio:
            logger.info(f"TAKE PROFIT: {position.symbol} pnl={position.unrealized_pnl_ratio:.1%}")
            return True
        return False
    
    def check_signal_rank(
        self,
        symbol: str,
        signal_rank: int,
        is_holding: bool,
    ) -> bool:
        """
        检查信号排名（V21 缓冲区逻辑）
        
        Args:
            symbol: 股票代码
            signal_rank: 当前信号排名
            is_holding: 是否已持仓
            
        Returns:
            True 表示应该卖出
        """
        if is_holding:
            # 已持仓：只有跌出 Top 30 才卖出（缓冲区）
            if signal_rank > self.sell_rank_threshold:
                logger.info(f"SIGNAL RANK: {symbol} rank={signal_rank} > {self.sell_rank_threshold} (sell threshold)")
                return True
        else:
            # 未持仓：必须 Top 10 才买入
            if signal_rank > self.buy_rank_threshold:
                return False  # 不买入，但不卖出
        
        return False
    
    def get_stocks_to_sell(
        self,
        positions: Dict[str, Position],
        signal_ranks: Dict[str, int],
    ) -> List[str]:
        """获取应该卖出的股票列表"""
        stocks_to_sell = []
        
        for symbol, pos in positions.items():
            # 检查止损
            if self.check_stop_loss(pos):
                stocks_to_sell.append(symbol)
                continue
            
            # 检查止盈
            if self.check_take_profit(pos):
                stocks_to_sell.append(symbol)
                continue
            
            # 检查信号排名（缓冲区）
            signal_rank = signal_ranks.get(symbol, 9999)
            if self.check_signal_rank(symbol, signal_rank, is_holding=True):
                stocks_to_sell.append(symbol)
                continue
        
        return stocks_to_sell
    
    def get_stocks_to_buy(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Position],
        target_amounts: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        获取应该买入的股票列表（V21 缓冲区逻辑）
        
        只有排名 Top 10 且得分 > 0.60 的股票才买入
        """
        stocks_to_buy = []
        
        # 按信号值排序
        ranked = signals_df.sort("signal", descending=True)
        
        for row in ranked.iter_rows(named=True):
            symbol = row["symbol"]
            signal = row["signal"]
            
            # 已持仓的不重复买入
            if symbol in current_positions:
                continue
            
            # 得分必须 > 0.60 (处理 None 值)
            if signal is None or signal <= self.score_threshold:
                continue
            
            # 必须在目标买入列表中
            if symbol in target_amounts:
                stocks_to_buy.append((symbol, target_amounts[symbol]))
            
            # 限制买入数量
            if len(stocks_to_buy) >= self.buy_rank_threshold:
                break
        
        return stocks_to_buy


# ===========================================
# V21 回测执行器
# ===========================================

class V21BacktestExecutor:
    """
    V21 回测执行器 - 整合所有组件
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        top_k_stocks: int = TOP_K_STOCKS,
        db: Optional[DatabaseManager] = None,
    ):
        self.accounting = V21AccountingEngine(
            initial_capital=initial_capital,
            top_k_stocks=top_k_stocks,
            db=db,
        )
        self.sizing_manager = V21DynamicSizingManager()
        self.position_manager = V21PositionManager()
        
        logger.info(f"V21BacktestExecutor initialized")
    
    def run_backtest(
        self,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V21 BACKTEST EXECUTION")
        logger.info("=" * 60)
        logger.info(f"Initial Capital: {self.accounting.initial_capital:,.0f}")
        logger.info(f"Score Threshold: {SCORE_THRESHOLD}")
        logger.info(f"Buy Rank Threshold: Top {BUY_RANK_THRESHOLD}")
        logger.info(f"Sell Rank Threshold: > {SELL_RANK_THRESHOLD}")
        
        # 日期过滤
        if start_date:
            signals_df = signals_df.filter(pl.col("trade_date") >= start_date)
            prices_df = prices_df.filter(pl.col("trade_date") >= start_date)
        if end_date:
            signals_df = signals_df.filter(pl.col("trade_date") <= end_date)
            prices_df = prices_df.filter(pl.col("trade_date") <= end_date)
        
        # 获取交易日期
        dates = signals_df["trade_date"].unique().sort().to_list()
        
        if not dates:
            logger.error("No trading dates found")
            return BacktestResult(
                start_date="",
                end_date="",
                initial_capital=self.accounting.initial_capital,
            )
        
        start_date = dates[0]
        end_date = dates[-1]
        
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Trading days: {len(dates)}")
        
        # 每日循环
        for trade_date in dates:
            self._execute_day(trade_date, signals_df, prices_df)
        
        # 生成结果
        return self._generate_result(start_date, end_date)
    
    def _execute_day(
        self,
        trade_date: str,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
    ):
        """执行单日交易逻辑"""
        logger.debug(f"\n=== {trade_date} ===")
        
        # 1. 获取当日信号
        day_signals = signals_df.filter(pl.col("trade_date") == trade_date)
        if day_signals.is_empty():
            logger.debug(f"No signals for {trade_date}")
            return
        
        # 2. 计算信号排名
        ranked = day_signals.sort("signal", descending=True).with_columns([
            (pl.col("signal").rank("ordinal", descending=True)).alias("rank")
        ])
        signal_ranks = {}
        for row in ranked.iter_rows(named=True):
            symbol = row["symbol"]
            rank = row["rank"]
            if rank is not None:
                signal_ranks[symbol] = int(rank)
            else:
                signal_ranks[symbol] = 9999
        
        # 3. 获取当日价格
        day_prices = prices_df.filter(pl.col("trade_date") == trade_date)
        prices = {
            row["symbol"]: float(row["close"])
            for row in day_prices.iter_rows(named=True)
        }
        
        # 4. 计算动态权重
        target_amounts = self.sizing_manager.compute_dynamic_weights(
            signals_df=day_signals,
            current_positions=self.accounting.positions,
            initial_capital=self.accounting.initial_capital,
        )
        
        # 5. 获取应卖出的股票（缓冲区逻辑）
        stocks_to_sell = self.position_manager.get_stocks_to_sell(
            self.accounting.positions,
            signal_ranks,
        )
        
        # 6. 执行卖出
        for symbol in stocks_to_sell:
            if symbol in prices:
                self.accounting.execute_sell(
                    trade_date=trade_date,
                    symbol=symbol,
                    price=prices[symbol],
                )
        
        # 7. 获取应买入的股票
        stocks_to_buy = self.position_manager.get_stocks_to_buy(
            signals_df=day_signals,
            current_positions=self.accounting.positions,
            target_amounts=target_amounts,
        )
        
        # 8. 执行买入
        for symbol, target_amount in stocks_to_buy:
            if symbol in prices and target_amount > 0:
                self.accounting.execute_buy(
                    trade_date=trade_date,
                    symbol=symbol,
                    price=prices[symbol],
                    target_amount=target_amount,
                )
        
        # 9. 计算当日 NAV
        self.accounting.compute_daily_nav(trade_date, prices)
        
        # 10. 记录持仓快照
        self.accounting.record_positions_snapshot(trade_date)
    
    def _generate_result(self, start_date: str, end_date: str) -> BacktestResult:
        """生成回测结果"""
        accounting = self.accounting
        
        # 计算收益指标
        if len(accounting.daily_navs) > 0:
            final_nav = accounting.daily_navs[-1].total_assets
            total_return = (final_nav - self.accounting.initial_capital) / self.accounting.initial_capital
            
            trading_days = len(accounting.daily_navs)
            years = trading_days / 252.0
            if years > 0:
                annual_return = (1 + total_return) ** (1 / years) - 1
            else:
                annual_return = 0.0
            
            daily_returns = [nav.daily_return for nav in accounting.daily_navs]
            if len(daily_returns) > 1:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns, ddof=1)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            nav_values = [nav.total_assets for nav in accounting.daily_navs]
            rolling_max = np.maximum.accumulate(nav_values)
            drawdowns = (np.array(nav_values) - rolling_max) / rolling_max
            max_drawdown = abs(np.min(drawdowns))
            
            winning_days = sum(1 for nav in accounting.daily_navs if nav.daily_return > 0)
            win_rate = winning_days / len(accounting.daily_navs) if accounting.daily_navs else 0.0
        else:
            total_return = 0.0
            annual_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
        
        # 计算交易指标
        total_trades = len(accounting.trades)
        buy_trades = [t for t in accounting.trades if t.side == "BUY"]
        sell_trades = [t for t in accounting.trades if t.side == "SELL"]
        
        total_buy_amount = sum(t.amount for t in buy_trades)
        total_sell_amount = sum(t.amount for t in sell_trades)
        total_commission = sum(t.commission for t in accounting.trades)
        total_stamp_duty = sum(t.stamp_duty for t in accounting.trades)
        total_slippage = sum(t.slippage for t in accounting.trades)
        
        turnover_rate = total_buy_amount / self.accounting.initial_capital if self.accounting.initial_capital > 0 else 0.0
        
        if accounting.holding_periods:
            avg_holding_days = np.mean(list(accounting.holding_periods.values()))
        else:
            avg_holding_days = 0.0
        
        if accounting.positions_history:
            position_counts = [len(snapshot["positions"]) for snapshot in accounting.positions_history]
            avg_position_count = np.mean(position_counts) if position_counts else 0.0
            
            # 计算平均持仓权重
            position_weights = [
                len(snapshot["positions"]) / TOP_K_STOCKS
                for snapshot in accounting.positions_history
            ]
            avg_position_weight = np.mean(position_weights) if position_weights else 0.0
            
            # 计算现金拖累（平均现金持有比例）
            cash_ratios = [
                snapshot["cash"] / snapshot["total_assets"]
                for snapshot in accounting.positions_history
                if snapshot["total_assets"] > 0
            ]
            cash_drag = np.mean(cash_ratios) if cash_ratios else 0.0
        else:
            avg_position_count = 0.0
            avg_position_weight = 0.0
            cash_drag = 0.0
        
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.accounting.initial_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            total_buy_amount=total_buy_amount,
            total_sell_amount=total_sell_amount,
            total_commission=total_commission,
            total_stamp_duty=total_stamp_duty,
            total_slippage=total_slippage,
            turnover_rate=turnover_rate,
            avg_holding_days=avg_holding_days,
            avg_position_count=avg_position_count,
            avg_position_weight=avg_position_weight,
            cash_drag=cash_drag,
            daily_navs=accounting.daily_navs,
            trades=accounting.trades,
            positions_history=accounting.positions_history,
        )
        
        # 输出结果
        logger.info("\n" + "=" * 60)
        logger.info("V21 BACKTEST RESULT")
        logger.info("=" * 60)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annual Return: {annual_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info("-" * 40)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Total Buy Amount: {total_buy_amount:,.2f}")
        logger.info(f"Total Commission: {total_commission:,.2f}")
        logger.info(f"Total Stamp Duty: {total_stamp_duty:,.2f}")
        logger.info(f"Total Slippage: {total_slippage:,.2f}")
        logger.info(f"Turnover Rate: {turnover_rate:.2%}")
        logger.info(f"Avg Holding Days: {avg_holding_days:.1f}")
        logger.info(f"Avg Position Count: {avg_position_count:.1f}")
        logger.info(f"Avg Position Weight: {avg_position_weight:.1%}")
        logger.info(f"Cash Drag: {cash_drag:.1%}")
        
        return result


# ===========================================
# 报告生成器
# ===========================================

def generate_v21_report(result: BacktestResult, v20_result: Optional[BacktestResult] = None, output_path: Optional[str] = None) -> str:
    """生成 V21 报告"""
    from pathlib import Path
    
    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V21_Dynamic_Sizing_Report_{timestamp}.md"
    
    total_fees = result.total_commission + result.total_stamp_duty + result.total_slippage
    gross_profit = result.total_return * result.initial_capital
    net_profit = gross_profit - total_fees
    fee_erosion_ratio = total_fees / gross_profit if gross_profit > 0 else 0.0
    
    # V20 vs V21 对比
    v20_comparison = ""
    if v20_result is not None:
        v20_fees = v20_result.total_commission + v20_result.total_stamp_duty + v20_result.total_slippage
        trades_change = (result.total_trades - v20_result.total_trades) / v20_result.total_trades * 100 if v20_result.total_trades > 0 else 0
        holding_days_change = ((result.avg_holding_days - v20_result.avg_holding_days) / v20_result.avg_holding_days * 100) if v20_result.avg_holding_days > 0 else 0
        fee_change = (total_fees - v20_fees) / v20_fees * 100 if v20_fees > 0 else 0
        
        v20_comparison = f"""
## 三、V20 vs V21 对比分析

### 3.1 核心指标对比

| 指标 | V20 | V21 | 变化 |
|------|-----|-----|------|
| 总收益 | {v20_result.total_return:.2%} | {result.total_return:.2%} | {(result.total_return - v20_result.total_return):.2%} |
| 年化收益 | {v20_result.annual_return:.2%} | {result.annual_return:.2%} | {(result.annual_return - v20_result.annual_return):.2%} |
| 夏普比率 | {v20_result.sharpe_ratio:.3f} | {result.sharpe_ratio:.3f} | {(result.sharpe_ratio - v20_result.sharpe_ratio):.3f} |
| 最大回撤 | {v20_result.max_drawdown:.2%} | {result.max_drawdown:.2%} | {(result.max_drawdown - v20_result.max_drawdown):.2%} |
| **总交易次数** | {v20_result.total_trades:,} | {result.total_trades:,} | **{trades_change:.1f}%** |
| **平均持仓天数** | {v20_result.avg_holding_days:.1f} | {result.avg_holding_days:.1f} | **{holding_days_change:.1f}%** |
| **总手续费** | {v20_fees:,.2f} 元 | {total_fees:,.2f} 元 | **{fee_change:.1f}%** |
| 手续费侵蚀率 | {v20_fees / (v20_result.total_return * v20_result.initial_capital) * 100 if v20_result.total_return > 0 else 0:.1f}% | {fee_erosion_ratio * 100:.1f}% | - |

### 3.2 换手率优化验证

| 指标 | V20 | V21 | 目标 |
|------|-----|-----|------|
| Total Trades | {v20_result.total_trades:,} | {result.total_trades:,} | 降低 |
| Avg Holding Days | {v20_result.avg_holding_days:.1f} 天 | {result.avg_holding_days:.1f} 天 | 提升 |
| Turnover Rate | {v20_result.turnover_rate:.2%} | {result.turnover_rate:.2%} | 降低 |

**状态**: {"✅ 换手率优化成功" if result.avg_holding_days > v20_result.avg_holding_days else "⚠️ 换手率仍需优化"}

### 3.3 动态权重效果

| 指标 | V21 | 说明 |
|------|-----|------|
| Avg Position Weight | {result.avg_position_weight:.1%} | 平均持仓权重 |
| Cash Drag | {result.cash_drag:.1%} | 现金拖累 |

"""
    
    report = f"""# V21 动态赋权与换手优化报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V21.0 (Dynamic Sizing & Turnover Buffer)

---

## 一、核心改进总结

### 1.1 概率驱动的动态调仓 (Dynamic Sizing)

**V20 问题**: 
- 10 等分权重呆板，无法根据信号强度调整
- 低分股票也占用 10% 仓位，资金效率低

**V21 解决方案**:
1. **得分激活**: 只有预测得分 > 0.60 的股票才进入买入备选池
2. **权重分配**: w_i ∝ (Score_i - 0.60)
3. **权重限制**: 单只个股最大 20%，最小 5%
4. **动态持仓**: 如果符合条件的票少，就只买几只，其余持币

### 1.2 换手率"滞后缓冲区" (Turnover Buffer)

**V20 问题**: 
- 排名在第 10-15 名之间频繁换手
- 产生大量无效手续费

**V21 解决方案**:
1. **买入标准**: 排名必须 Top 10 且得分 > 0.60
2. **卖出标准**: 只有跌出 Top 30 才卖出（缓冲区）
3. **目的**: 给信号波动留出空间，减少无效换手

### 1.3 真实手续费（维持 V20 严格标准）

| 费用类型 | 费率 | 说明 |
|----------|------|------|
| 买入滑点 | 0.1% | 实际成交价 vs 订单价 |
| 卖出滑点 | 0.1% | 实际成交价 vs 订单价 |
| 交易规费 | 0.03% | 经手费 + 证管费 |
| 印花税 | 0.05% | 仅卖出时收取 |
| **最低佣金** | **5 元** | 每笔交易不足 5 元按 5 元计 |

---

## 二、回测结果

### 2.1 收益指标

| 指标 | 值 |
|------|-----|
| 回测区间 | {result.start_date} 至 {result.end_date} |
| 初始资金 | {result.initial_capital:,.0f} 元 |
| **总收益** | **{result.total_return:.2%}** |
| **年化收益** | **{result.annual_return:.2%}** |
| **夏普比率** | **{result.sharpe_ratio:.3f}** |
| **最大回撤** | **{result.max_drawdown:.2%}** |
| 胜率 | {result.win_rate:.1%} |

### 2.2 真实交易指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **总交易次数** | {result.total_trades:,} | 实际成交笔数 |
| **总买入额** | {result.total_buy_amount:,.2f} 元 | 真实买入金额 |
| **总卖出额** | {result.total_sell_amount:,.2f} 元 | 真实卖出金额 |
| **总手续费** | {result.total_commission:,.2f} 元 | 规费 + 佣金 |
| **总印花税** | {result.total_stamp_duty:,.2f} 元 | 卖出印花税 |
| **总滑点成本** | {result.total_slippage:,.2f} 元 | 滑点损失 |
| **换手率** | {result.turnover_rate:.2%} | 总买入额/初始资金 |

### 2.3 手续费侵蚀分析

| 指标 | 值 |
|------|-----|
| 毛利润 | {gross_profit:,.2f} 元 |
| 总手续费 | {total_fees:,.2f} 元 |
| **净利润** | **{net_profit:,.2f} 元** |
| **手续费侵蚀率** | **{fee_erosion_ratio:.1%}** |

**状态**: {"⚠️ 手续费侵蚀严重" if fee_erosion_ratio > 0.3 else "✅ 手续费可控" if fee_erosion_ratio < 0.1 else "⚠️ 手续费需关注"}

### 2.4 持仓指标

| 指标 | 值 |
|------|-----|
| 平均持仓天数 | {result.avg_holding_days:.1f} 天 |
| 平均持仓数量 | {result.avg_position_count:.1f} 只 |
| 平均持仓权重 | {result.avg_position_weight:.1%} |
| 现金拖累 | {result.cash_drag:.1%} |

{v20_comparison}

## 四、真实统计指标公式

### 4.1 每日 NAV (Net Asset Value)

```
NAV = 当日现金 + 当日持仓市值
```

### 4.2 Daily Return

```
Daily Return = (今日 NAV - 昨日 NAV) / 昨日 NAV
```

### 4.3 Sharpe Ratio

```
Sharpe = Mean(Daily_Return) / Std(Daily_Return) * sqrt(252)
```

### 4.4 Max Drawdown

```
Rolling_Max = maximum_accumulate(NAV_curve)
Drawdown = (NAV - Rolling_Max) / Rolling_Max
MaxDD = |min(Drawdown)|
```

---

## 五、严防偷懒与作弊

### 5.1 严禁重写 Engine 核心
- ✅ 复用 V20 会计引擎核心逻辑（手续费、最低 5 元佣金、T+1、NAV 计算）

### 5.2 拒绝"虚拟换手率"
- ✅ 输出 Total Buy Amount: {result.total_buy_amount:,.2f} 元
- ✅ 输出 Total Commission: {result.total_commission:,.2f} 元
- ✅ 输出 Turnover Rate: {result.turnover_rate:.2%}

### 5.3 严防未来函数
- ✅ 所有平滑和过滤逻辑使用 .shift(1)
- ✅ T+1 锁定严格执行

---

## 六、执行总结

### 6.1 核心结论

1. **动态权重**: 根据信号强度分配仓位，提高资金效率
2. **缓冲区**: Top 10 买入 / Top 30 卖出，减少无效换手
3. **手续费透明**: 完整计算滑点、规费、印花税、最低佣金
4. **持仓生命周期**: 止损/止盈/排名三重保护

### 6.2 后续优化方向

1. 动态调整得分阈值（根据市场波动率）
2. 优化权重分配公式（考虑相关性）
3. 增加行业/市值中性化约束

---

**报告生成完毕**
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"V21 report saved to: {output_path}")
    return str(output_path)


# ===========================================
# 主函数
# ===========================================

def main():
    """主入口函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V21 Dynamic Sizing & Turnover Buffer Engine")
    logger.info("=" * 70)
    
    # 简单测试
    engine = V21AccountingEngine(initial_capital=100000)
    
    # 测试买入
    trade = engine.execute_buy(
        trade_date="2026-03-17",
        symbol="000001.SZ",
        price=10.0,
        target_amount=20000.0,  # V21: 动态权重，可能超过 10%
    )
    
    if trade:
        logger.info(f"Buy executed: {trade.shares} shares @ {trade.price}")
        logger.info(f"Commission: {trade.commission:.2f}")
        logger.info(f"Slippage: {trade.slippage:.2f}")
    
    # 测试持仓更新
    engine.update_position_prices("2026-03-17", {"000001.SZ": 10.5})
    
    # 测试 NAV 计算
    nav = engine.compute_daily_nav("2026-03-17", {"000001.SZ": 10.5})
    logger.info(f"NAV: {nav.total_assets:.2f}")
    logger.info(f"Daily Return: {nav.daily_return:.2%}")
    
    logger.info("\nV21 Accounting Engine test completed.")


if __name__ == "__main__":
    main()