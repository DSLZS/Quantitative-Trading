"""
V41 Risk Manager Module - 风险管理模块

【核心功能】
1. ATR 动态止损（继承 V40）
2. 风险平价调仓（继承 V40）
3. V41 增强：低波动时提升风险暴露（0.5% -> 0.8%）
4. 入场过滤（波动率过滤）
5. 持仓锁定期管理

作者：量化系统
版本：V41.0
日期：2026-03-19
"""

import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V41 风险配置
# ===========================================

# 基础风险配置（继承 V40）
BASE_RISK_TARGET_PER_POSITION = 0.005  # 基础单只股票风险暴露 0.5%
ENHANCED_RISK_TARGET_PER_POSITION = 0.008  # V41 增强：低波动时提升至 0.8%
MAX_POSITION_RATIO = 0.15  # 单只股票最大仓位 15%
MIN_POSITION_RATIO = 0.05  # 单只股票最小仓位 5%

# ATR 止损配置（继承 V40）
ATR_PERIOD = 20
TRAILING_STOP_ATR_MULT = 2.0  # 2 * ATR 移动止损
INITIAL_STOP_LOSS_RATIO = 0.08  # 初始止损 8%（作为底线）

# 持仓周期配置（优化：延长持仓周期，减少交易频率）
MIN_HOLDING_DAYS = 30  # 买入后锁定 30 个交易日（增加）
MAX_HOLDING_DAYS = 60  # 最大持仓天数（增加）

# 调仓周期（继承 V40）
REBALANCE_FREQUENCY = 25  # 每 25 个交易日调仓一次

# 入场过滤配置（继承 V40）
VOLATILITY_FILTER_WINDOW = 20
VOLATILITY_FILTER_THRESHOLD = 1.5  # 超过 1.5 倍均值时停止开仓

# V41 低波动阈值（优化：更严格的低波动定义）
LOW_VOLATILITY_THRESHOLD = 0.7  # 波动率比率低于 0.7 视为低波动环境（更严格）


@dataclass
class RiskManagerConfig:
    """风险管理器配置"""
    # 风险暴露配置
    base_risk_target: float = BASE_RISK_TARGET_PER_POSITION
    enhanced_risk_target: float = ENHANCED_RISK_TARGET_PER_POSITION
    max_position_ratio: float = MAX_POSITION_RATIO
    min_position_ratio: float = MIN_POSITION_RATIO
    max_positions: int = 8  # 最大持仓数量
    
    # ATR 止损配置
    atr_period: int = ATR_PERIOD
    trailing_stop_atr_mult: float = TRAILING_STOP_ATR_MULT
    initial_stop_loss_ratio: float = INITIAL_STOP_LOSS_RATIO
    
    # 持仓周期配置
    min_holding_days: int = MIN_HOLDING_DAYS
    max_holding_days: int = MAX_HOLDING_DAYS
    
    # 调仓周期
    rebalance_frequency: int = REBALANCE_FREQUENCY
    
    # 入场过滤配置
    volatility_filter_threshold: float = VOLATILITY_FILTER_THRESHOLD
    
    # V41 低波动阈值
    low_volatility_threshold: float = LOW_VOLATILITY_THRESHOLD
    
    # 费率配置
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty: float = 0.0005
    transfer_fee: float = 0.00001


@dataclass
class Position:
    """持仓记录"""
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
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_triggered: bool = False
    trailing_stop_history: List[float] = field(default_factory=list)


@dataclass
class Trade:
    """交易记录"""
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


@dataclass
class TradeAudit:
    """交易审计记录"""
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
    atr_at_entry: float = 0.0
    initial_stop_price: float = 0.0
    peak_price: float = 0.0
    trailing_stop_triggered: bool = False


class RiskManager:
    """
    V41 风险管理器 - 模块化设计
    
    【核心功能】
    1. ATR 动态止损（继承 V40）
    2. 风险平价调仓（继承 V40）
    3. V41 增强：低波动时提升风险暴露
    4. 入场过滤（波动率过滤）
    5. 持仓锁定期管理
    """
    
    def __init__(self, config: RiskManagerConfig = None, initial_capital: float = 100000.0):
        self.config = config or RiskManagerConfig()
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.trade_log: List[TradeAudit] = []
        
        self.today_sells: Set[str] = set()
        self.today_buys: Set[str] = set()
        self.last_trade_date: Optional[str] = None
        
        # 持仓锁定期追踪
        self.locked_positions: Dict[str, int] = {}
        
        # 当前交易日
        self.current_trade_date: Optional[str] = None
        self.trade_day_counter: int = 0
        
        # 波动率过滤统计
        self.volatility_filter_triggered = False
        self.volatility_filter_triggered_days = 0
        
        # V41 状态：是否处于低波动环境
        self.is_low_volatility_environment = False
    
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
        return max(self.config.min_commission, amount * self.config.commission_rate)
    
    def _calculate_transfer_fee(self, shares: int, price: float) -> float:
        """计算过户费"""
        return shares * price * self.config.transfer_fee
    
    def check_volatility_environment(self, market_volatility_ratio: float) -> bool:
        """
        V41 增强：检查是否处于低波动环境
        
        低波动环境允许提升风险暴露至 0.8%
        """
        self.is_low_volatility_environment = market_volatility_ratio < self.config.low_volatility_threshold
        return self.is_low_volatility_environment
    
    def check_volatility_filter(self, market_volatility_ratio: float) -> bool:
        """
        入场过滤 - 检查市场波动率
        
        如果市场波动率超过过去 20 天均值的 1.5 倍，停止开新仓
        """
        self.volatility_filter_triggered = market_volatility_ratio > self.config.volatility_filter_threshold
        if self.volatility_filter_triggered:
            self.volatility_filter_triggered_days += 1
        return not self.volatility_filter_triggered
    
    def get_current_risk_target(self, market_volatility_ratio: float) -> float:
        """
        V41 增强：获取当前风险暴露目标
        
        - 低波动环境：0.8%
        - 正常/高波动环境：0.5%
        """
        if self.check_volatility_environment(market_volatility_ratio):
            return self.config.enhanced_risk_target
        return self.config.base_risk_target
    
    def calculate_position_size(
        self, 
        symbol: str, 
        atr: float, 
        current_price: float, 
        total_assets: float,
        market_volatility_ratio: float = 1.0,
    ) -> Tuple[int, float]:
        """
        V41 风险平价调仓 - 计算仓位大小
        
        V41 增强：低波动环境提升风险暴露至 0.8%
        
        核心：波动大的股票少买，波动小的多买
        
        公式：
        - 风险金额 = 总资产 * 风险目标 (0.5% 或 0.8%)
        - 每股风险 = ATR * 2
        - 股数 = 风险金额 / 每股风险
        - 仓位 = 股数 * 价格
        
        返回：(shares, position_amount)
        """
        try:
            # V41 增强：根据波动率环境调整风险目标
            risk_target = self.get_current_risk_target(market_volatility_ratio)
            
            # 风险金额
            risk_amount = total_assets * risk_target
            
            # 每股风险（2 * ATR）
            risk_per_share = atr * self.config.trailing_stop_atr_mult
            
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
            max_position = total_assets * self.config.max_position_ratio
            min_position = total_assets * self.config.min_position_ratio
            
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
        market_volatility_ratio: float = 1.0,
    ) -> Optional[Trade]:
        """
        V41 买入执行 - ATR 动态防御
        """
        try:
            if symbol in self.today_sells:
                logger.warning(f"WASH SALE PREVENTED: {symbol} already sold today ({trade_date})")
                return None
            
            if symbol in self.today_buys:
                logger.warning(f"DUPLICATE BUY PREVENTED: {symbol} already bought today ({trade_date})")
                return None
            
            # 场景化滑点（默认 0.1%）
            slippage_buy = 0.001
            execution_price = open_price * (1 + slippage_buy)
            
            # 使用目标仓位计算股数
            raw_shares = int(target_amount / execution_price)
            shares = (raw_shares // 100) * 100
            if shares < 100:
                return None
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * slippage_buy
            transfer_fee = self._calculate_transfer_fee(shares, execution_price)
            total_cost = actual_amount + commission + slippage + transfer_fee
            
            if self.cash < total_cost:
                return None
            
            self.cash -= total_cost
            
            # V41 核心：ATR 动态止损初始化
            atr_stop_distance = atr / execution_price if execution_price > 0 else 0
            atr_stop_price = execution_price * (1 - self.config.trailing_stop_atr_mult * atr_stop_distance)
            initial_stop_price = execution_price * (1 - self.config.initial_stop_loss_ratio)
            stop_price = max(atr_stop_price, initial_stop_price)
            
            # 创建新持仓
            self.positions[symbol] = Position(
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
            self.locked_positions[symbol] = self.config.min_holding_days + 2
            
            self.today_buys.add(symbol)
            
            trade = Trade(
                trade_date=trade_date, symbol=symbol, side="BUY", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=0.0, transfer_fee=transfer_fee,
                total_cost=total_cost, reason=reason, execution_price=execution_price,
            )
            self.trades.append(trade)
            
            risk_env = "LOW_VOL" if self.is_low_volatility_environment else "NORMAL"
            logger.info(f"  BUY  {symbol} | {shares} shares @ {execution_price:.2f} | ATR={atr:.3f} | Risk={risk_env}")
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
    ) -> Optional[Trade]:
        """
        V41 卖出执行 - ATR 动态防御
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
            
            # 场景化滑点（默认 0.1%）
            slippage_sell = 0.001
            execution_price = open_price * (1 - slippage_sell)
            
            actual_amount = shares * execution_price
            commission = self._calculate_commission(actual_amount)
            slippage = actual_amount * slippage_sell
            stamp_duty = actual_amount * self.config.stamp_duty
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
            trade_audit = TradeAudit(
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
            
            trade = Trade(
                trade_date=trade_date, symbol=symbol, side="SELL", shares=shares,
                price=open_price, amount=actual_amount, commission=commission,
                slippage=slippage, stamp_duty=stamp_duty, transfer_fee=transfer_fee,
                total_cost=net_proceeds, reason=reason, holding_days=pos.holding_days,
                execution_price=execution_price,
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
        V41 更新持仓价格并检查止损条件
        
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
                
                # V41 核心：ATR 移动止损更新
                if symbol in atrs and atrs[symbol] > 0:
                    atr = atrs[symbol]
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    
                    # 计算新的移动止损价（基于当前价格）
                    new_trailing_stop = current_price * (1 - self.config.trailing_stop_atr_mult * atr_stop_distance)
                    
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
                if profit_ratio <= -self.config.initial_stop_loss_ratio:
                    sell_list.append((symbol, "stop_loss"))
                    continue
                
                # 3. 最大持仓天数触发
                if pos.holding_days >= self.config.max_holding_days:
                    sell_list.append((symbol, "max_holding"))
                    continue
        
        return sell_list
    
    def get_position_count(self) -> int:
        """获取当前持仓数量"""
        return len(self.positions)
    
    def get_total_market_value(self, prices: Dict[str, float]) -> float:
        """获取总市值"""
        return sum(
            pos.shares * prices.get(pos.symbol, pos.current_price)
            for pos in self.positions.values()
        )
    
    def get_total_assets(self, prices: Dict[str, float]) -> float:
        """获取总资产"""
        return self.cash + self.get_total_market_value(prices)
    
    def get_position_ratios(self, prices: Dict[str, float]) -> Dict[str, float]:
        """获取各持仓占比"""
        total = self.get_total_assets(prices)
        if total <= 0:
            return {}
        return {
            symbol: pos.shares * prices.get(pos.symbol, pos.current_price) / total
            for symbol, pos in self.positions.items()
        }
    
    def clear_daily_counters(self, trade_date: str):
        """清除每日计数器"""
        self.today_sells.clear()
        self.today_buys.clear()
        self.last_trade_date = trade_date
    
    # ===========================================
    # V41 Engine 接口方法
    # ===========================================
    
    def update_volatility_regime(self, market_vol: float):
        """更新波动率状态"""
        self.current_market_vol = market_vol
        self.check_volatility_environment(market_vol)
    
    def get_risk_per_position(self) -> float:
        """获取当前风险暴露"""
        market_vol = getattr(self, 'current_market_vol', 1.0)
        return self.get_current_risk_target(market_vol)
    
    def get_portfolio_value(self, positions: Dict[str, Position], date_str: str, price_df) -> float:
        """获取组合价值"""
        market_value = 0.0
        for symbol, pos in positions.items():
            try:
                row = price_df.filter((pl.col('symbol') == symbol) & (pl.col('trade_date') == date_str)).select('close').row(0)
                if row:
                    market_value += pos.shares * float(row[0])
            except Exception:
                market_value += pos.shares * pos.current_price
        return self.cash + market_value
    
    def check_stop_loss(self, positions: Dict[str, Position], date_str: str, price_df, factor_df) -> List[str]:
        """检查止损并返回需要卖出的股票列表"""
        sell_list = []
        
        # 获取当日价格和 ATR
        prices = {}
        atrs = {}
        
        for row in price_df.iter_rows(named=True):
            symbol = row['symbol']
            prices[symbol] = row.get('close', 0) or 0
        
        for row in factor_df.iter_rows(named=True):
            symbol = row['symbol']
            atrs[symbol] = row.get('atr_20', 0) or 0
        
        # 更新持仓并检查止损
        for symbol, pos in list(positions.items()):
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
                
                # 更新移动止损
                if symbol in atrs and atrs[symbol] > 0:
                    atr = atrs[symbol]
                    atr_stop_distance = atr / current_price if current_price > 0 else 0
                    new_trailing_stop = current_price * (1 - self.config.trailing_stop_atr_mult * atr_stop_distance)
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        pos.trailing_stop_history.append(new_trailing_stop)
                
                # 检查止损触发
                if current_price <= pos.trailing_stop_price:
                    pos.trailing_stop_triggered = True
                    sell_list.append(symbol)
                    continue
                
                profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
                if profit_ratio <= -self.config.initial_stop_loss_ratio:
                    sell_list.append(symbol)
                    continue
                
                if pos.holding_days >= self.config.max_holding_days:
                    sell_list.append(symbol)
                    continue
        
        return sell_list
    
    def close_position(self, pos: Position, date_str: str, exit_price: float, reason: str):
        """平仓"""
        self.execute_sell(date_str, pos.symbol, exit_price, reason=reason)
    
    def open_position(self, symbol: str, date_str: str, entry_price: float, 
                      portfolio_value: float, risk_per_position: float) -> Optional[Position]:
        """开仓"""
        # 默认 ATR 设为价格的 2%（更合理的估计）
        atr = entry_price * 0.02
        
        # 计算仓位
        risk_amount = portfolio_value * risk_per_position
        risk_per_share = atr * self.config.trailing_stop_atr_mult
        
        if risk_per_share <= 0 or entry_price <= 0:
            return None
        
        shares = int(risk_amount / risk_per_share)
        shares = (shares // 100) * 100
        
        if shares < 100:
            return None
        
        target_amount = shares * entry_price
        
        # 执行买入
        trade = self.execute_buy(
            date_str, symbol, entry_price, atr, target_amount,
            signal_score=0.0, signal_rank=999, reason="signal"
        )
        
        if trade:
            return self.positions.get(symbol)
        return None
    
    def rank_candidates(self, factor_df, positions: Dict[str, Position]) -> List[Dict]:
        """排名候选股票 - 按信号强度排序"""
        try:
            # 过滤已有持仓
            held_symbols = set(positions.keys())
            
            # 使用信号强度排序（而不是 rank）
            if 'signal' in factor_df.columns:
                # 过滤掉持仓，按信号降序排序
                candidates = factor_df.filter(
                    ~pl.col('symbol').is_in(list(held_symbols))
                ).sort('signal', descending=True).head(20)
            else:
                candidates = factor_df.filter(
                    ~pl.col('symbol').is_in(list(held_symbols))
                ).limit(20)
            
            result = []
            for idx, row in enumerate(candidates.iter_rows(named=True)):
                result.append({
                    'symbol': row['symbol'],
                    'rank': idx + 1,
                    'signal': float(row.get('signal', 0)) if row.get('signal') is not None else 0
                })
            
            return result
            
        except Exception as e:
            logger.error(f"rank_candidates failed: {e}")
            return []
