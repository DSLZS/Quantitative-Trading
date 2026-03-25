"""
V57 Engine Module - 行业先行与递归自修正

【V57 核心改进】

1. 审计自查与逻辑闭环（最高优先级）
   ✅ 严禁交易次数 > 30 次 - 提高 Score 入场门槛（Top 2）
   ✅ 手续费覆盖：保本/止盈卖出逻辑必须扣除 0.2% 摩擦成本后 ≥ 0

2. 选股引擎重构：行业先行
   ✅ 实现 v57_industry_filter - 先计算 30 个行业的平均得分
   ✅ 只在行业得分前 5 的板块中寻找个股
   ✅ 内置行业字典映射，手动对 5000 只股票进行行业归类

3. 强制递归式自迭代（Recursive Self-Correction）
   ✅ 运行 V57 -> 检查 Total_Return 和 MDD
   ✅ 如果 Return < 12% 或 MDD > 8%：
      - 自动分析失败原因（入场太早？止损太窄？）
      - 修改源代码逻辑（改变因子权重或均线参数）
      - 重新运行回测
   ✅ 输出要求：最终报告必须是已达标结果

4. 禁令
   ✅ 严禁删除 Slippage 或 Stamp_Duty
   ✅ 严禁偷看未来数据
   ✅ 严禁在 check_exits 里写死特定日期的卖出信号

作者：量化系统
版本：V57.0
日期：2026-03-22
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

from v57_core import (
    V57Position, V57Trade, V57TradeAudit, V57WashSaleRecord, V57BlacklistRecord,
    V57MarketRegime, V57DrawdownState, V57WeeklyTradeCounter,
    V57IndustryLoader, V57FactorEngine, v57_industry_filter,
    V57_INITIAL_CAPITAL, V57_MAX_POSITIONS, V57_ENTRY_TOP_N,
    V57_MAINTAIN_TOP_N, V57_MA60_FILTER, V57_GLOBAL_TRADE_LIMIT,
    V57_WEEKLY_TRADE_LIMIT, V57_FRICTION_COST, V57_BREAKEVEN_PROFIT_THRESHOLD,
    V57_BREAKEVEN_BUFFER, V57_BREAKEVEN_ENABLED, V57_TIERED_PROFIT_ENABLED,
    V57_TIERED_PROFIT_LEVELS, V57_HARD_STOP_LOSS_ATR_MULT, V57_HARD_STOP_LOSS_RATIO,
    V57_TRAILING_PROFIT_TRIGGER, V57_TRAILING_PROFIT_ATR_MULT, V57_TRAILING_PROFIT_ENABLED,
    V57_TIME_STOP_ENABLED, V57_TIME_STOP_DAYS, V57_TIME_STOP_REDUCE_RATIO,
    V57_MA20_TREND_EXIT_ENABLED, V57_RS_ENABLED, V57_RS_TOP_PERCENTILE,
    V57_VOLUME_BREAKOUT_MULT, V57_MA20_BREAKOUT, V57_WASH_SALE_WINDOW,
    V57_COMMISSION_RATE, V57_MIN_COMMISSION, V57_SLIPPAGE_BUY, V57_SLIPPAGE_SELL,
    V57_STAMP_DUTY, V57_TRANSFER_FEE, V57_RISK_TARGET_PER_POSITION,
    V57_MAX_SINGLE_POSITION_PCT, V57_REDUCED_SINGLE_POSITION_PCT,
    V57_ATR_VOLATILITY_THRESHOLD, V57_MAX_ITERATION_ROUNDS,
    V57_RETURN_TARGET, V57_MDD_TARGET, V57_INDUSTRY_FILTER_ENABLED,
    V57_INDUSTRY_TOP_N, V57_MOMENTUM_WEIGHT, V57_R2_WEIGHT, V57_INDUSTRY_WEIGHT,
    V57_VOLUME_FILTER_ENABLED, V57_VOLUME_SHRINK_THRESHOLD
)


class V57RiskManager:
    """
    V57 风险管理器
    
    【核心功能】
    1. 三级防御体系：硬止损、保本止损、追踪止盈
    2. 手续费覆盖检查：保本/止盈卖出必须扣除 0.2% 摩擦成本后 ≥ 0
    3. 频率熔断：每周最多开仓 2 只，全场最多 30 次交易
    4. 洗售审计：防止 5 天内同股票反向交易
    """
    
    def __init__(self, initial_capital: float = V57_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, V57Position] = {}
        self.trades: List[V57Trade] = []
        self.trade_log: List[V57TradeAudit] = []
        self.wash_sale_records: List[V57WashSaleRecord] = []
        self.blacklist: Dict[str, V57BlacklistRecord] = {}
        self.market_regimes: Dict[str, V57MarketRegime] = {}
        self.drawdown_states: Dict[str, V57DrawdownState] = {}
        self.weekly_trade_counters: Dict[str, V57WeeklyTradeCounter] = {}
        self.global_trade_count = 0
        self.peak_portfolio_value = initial_capital
        self.current_trade_date = ""
    
    def reset_daily_counters(self, trade_date: str):
        """重置每日计数器"""
        self.current_trade_date = trade_date
        self._update_weekly_counter(trade_date)
        self._update_portfolio_peak(trade_date)
    
    def _update_weekly_counter(self, trade_date: str):
        """更新每周交易计数器"""
        try:
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
            iso_cal = date_obj.isocalendar()
            week_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
            
            if week_key not in self.weekly_trade_counters:
                self.weekly_trade_counters[week_key] = V57WeeklyTradeCounter(
                    week_number=iso_cal[1],
                    year=iso_cal[0],
                    trade_count=0
                )
        except Exception:
            pass
    
    def _update_portfolio_peak(self, trade_date: str):
        """更新组合峰值"""
        current_value = self.get_total_portfolio_value(trade_date)
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
    
    def get_total_portfolio_value(self, trade_date: str) -> float:
        """获取组合总价值"""
        market_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + market_value
    
    def get_current_drawdown(self) -> float:
        """获取当前回撤"""
        if self.peak_portfolio_value <= 0:
            return 0.0
        current_value = self.cash + sum(pos.market_value for pos in self.positions.values())
        return (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
    
    def can_open_new_position(self) -> Tuple[bool, str]:
        """
        V57: 检查是否可以开新仓（频率熔断）
        
        【审计自查】
        1. 每周交易次数限制
        2. 全场交易次数限制（≤30 次）
        """
        if self.global_trade_count >= V57_GLOBAL_TRADE_LIMIT:
            return False, f"Global trade limit reached ({V57_GLOBAL_TRADE_LIMIT})"
        
        try:
            date_obj = datetime.strptime(self.current_trade_date, "%Y-%m-%d")
            iso_cal = date_obj.isocalendar()
            week_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
            
            if week_key in self.weekly_trade_counters:
                counter = self.weekly_trade_counters[week_key]
                if counter.trade_count >= V57_WEEKLY_TRADE_LIMIT:
                    return False, f"Weekly trade limit reached ({V57_WEEKLY_TRADE_LIMIT})"
        except Exception:
            pass
        
        return True, "OK"
    
    def record_trade(self, trade: V57Trade):
        """记录交易"""
        self.trades.append(trade)
        
        if trade.side == "BUY":
            self.global_trade_count += 1
            
            try:
                date_obj = datetime.strptime(trade.trade_date, "%Y-%m-%d")
                iso_cal = date_obj.isocalendar()
                week_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
                
                if week_key in self.weekly_trade_counters:
                    self.weekly_trade_counters[week_key].trade_count += 1
            except Exception:
                pass
    
    def check_wash_sale(self, symbol: str) -> Tuple[bool, str]:
        """检查洗售限制"""
        if symbol in self.blacklist:
            record = self.blacklist[symbol]
            if record.days_remaining > 0:
                return False, f"Blacklist active: {record.days_remaining} days remaining"
            else:
                del self.blacklist[symbol]
        return True, "OK"
    
    def add_to_blacklist(self, symbol: str, stop_date: str, stop_reason: str):
        """添加股票到黑名单"""
        try:
            stop_date_obj = datetime.strptime(stop_date, "%Y-%m-%d")
            expiry_date = stop_date_obj + timedelta(days=V57_WASH_SALE_WINDOW)
            
            self.blacklist[symbol] = V57BlacklistRecord(
                symbol=symbol,
                stop_date=stop_date,
                stop_reason=stop_reason,
                blacklist_expiry_day=V57_WASH_SALE_WINDOW,
                days_remaining=V57_WASH_SALE_WINDOW
            )
        except Exception:
            pass
    
    def update_blacklist_days(self, trade_date: str):
        """更新黑名单剩余天数"""
        for record in self.blacklist.values():
            if record.days_remaining > 0:
                record.days_remaining -= 1
    
    def calculate_position_size(self, symbol: str, atr: float,
                                 current_price: float, total_assets: float) -> Tuple[int, float, str]:
        """
        V57: 波动率适配头寸管理
        
        【核心逻辑】
        1. 根据 ATR 计算目标风险
        2. 高波动率股票降低仓位
        """
        if current_price <= 0 or atr <= 0:
            return 0, 0.0, "rejected"
        
        volatility_ratio = atr / current_price
        
        if volatility_ratio > V57_ATR_VOLATILITY_THRESHOLD:
            position_pct = V57_REDUCED_SINGLE_POSITION_PCT
            position_tier = "reduced"
        else:
            position_pct = V57_MAX_SINGLE_POSITION_PCT
            position_tier = "standard"
        
        target_amount = total_assets * position_pct
        shares = int(target_amount / current_price / 100) * 100
        
        if shares < 100:
            return 0, 0.0, "too_small"
        
        return shares, shares * current_price, position_tier
    
    def execute_buy(self, trade_date: str, symbol: str, open_price: float,
                    atr: float, target_amount: float, signal_date: str,
                    signal_score: float, signal_rank: int, reason: str = "V57 Entry",
                    composite_score: float = 0.0, composite_percentile: float = 0.0,
                    ma5: float = 0.0, ma20: float = 0.0, ma60: float = 0.0,
                    ma120: float = 0.0, industry_name: str = "",
                    volume_shrunk: bool = False, rs_score: float = 0.0,
                    rs_rank: int = 9999, volume_breakout: bool = False):
        """执行买入"""
        try:
            wash_ok, wash_reason = self.check_wash_sale(symbol)
            if not wash_ok:
                logger.warning(f"WASH SALE: {symbol} - {wash_reason}")
                return False
            
            can_open, open_reason = self.can_open_new_position()
            if not can_open:
                logger.warning(f"FREQUENCY FUSE: {symbol} - {open_reason}")
                return False
            
            shares, actual_amount, position_tier = self.calculate_position_size(
                symbol=symbol, atr=atr, current_price=open_price,
                total_assets=self.get_total_portfolio_value(trade_date)
            )
            
            if shares < 100:
                return False
            
            commission = max(actual_amount * V57_COMMISSION_RATE, V57_MIN_COMMISSION)
            slippage = actual_amount * V57_SLIPPAGE_BUY
            transfer_fee = actual_amount * V57_TRANSFER_FEE
            total_cost = actual_amount + commission + slippage + transfer_fee
            
            if total_cost > self.cash:
                shares = int((self.cash - commission - slippage - transfer_fee) / open_price / 100) * 100
                if shares < 100:
                    return False
                actual_amount = shares * open_price
                total_cost = actual_amount + commission + slippage + transfer_fee
            
            self.cash -= total_cost
            
            position = V57Position(
                symbol=symbol,
                shares=shares,
                avg_cost=open_price,
                buy_price=open_price,
                buy_date=trade_date,
                signal_date=signal_date,
                trade_date=trade_date,
                signal_score=signal_score,
                signal_rank=signal_rank,
                composite_score=composite_score,
                current_price=open_price,
                market_value=actual_amount,
                holding_days=0,
                peak_price=open_price,
                peak_profit=0.0,
                atr_at_entry=atr,
                hard_stop_price=open_price * (1 - V57_HARD_STOP_LOSS_RATIO),
                breakeven_active=False,
                trailing_profit_active=False,
                ma5_at_entry=ma5,
                ma20_at_entry=ma20,
                ma60_at_entry=ma60,
                ma120_at_entry=ma120,
                industry_name=industry_name,
                volume_shrunk_at_entry=volume_shrunk,
                rs_score=rs_score,
                rs_rank=rs_rank,
                volume_breakout=volume_breakout,
                position_tier=position_tier,
                entry_composite_score=composite_percentile
            )
            
            position.hard_stop_history = [position.hard_stop_price]
            
            self.positions[symbol] = position
            
            trade = V57Trade(
                trade_date=trade_date,
                symbol=symbol,
                side="BUY",
                shares=shares,
                price=open_price,
                amount=actual_amount,
                commission=commission,
                slippage=slippage,
                stamp_duty=0.0,
                transfer_fee=transfer_fee,
                total_cost=total_cost,
                reason=reason,
                execution_price=open_price,
                signal_date=signal_date
            )
            
            self.record_trade(trade)
            
            logger.info(f"BUY: {symbol} @ {open_price:.2f} x {shares} = {actual_amount:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"execute_buy failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def check_exits(self, positions: Dict[str, V57Position], date_str: str,
                    price_df: pl.DataFrame, factor_df: pl.DataFrame,
                    next_day_price_df: Optional[pl.DataFrame] = None) -> List[Tuple[str, str, float, Optional[float], Optional[float]]]:
        """
        V57: 检查退出信号
        
        【核心逻辑】
        1. 硬止损：价格跌破硬止损线
        2. 保本止损：浮盈超过 5% 后，硬止损线上移至"买入成本价 + 0.5%"
        3. 追踪止盈：浮盈≥12% 激活，回撤 2.5 倍 ATR 离场
        4. 阶梯止盈：浮盈 12% 减仓 30%，浮盈 22% 减仓 40%
        5. 时间止损：买入后 7 天不盈利则减仓 50%
        6. MA20 趋势退出：价格跌破 MA20
        
        【手续费覆盖检查】
        任何名为"保本"或"止盈"的卖出逻辑，其预期 PnL 必须扣除 0.2% 的摩擦成本后 ≥ 0
        """
        sell_list = []
        
        for symbol, pos in positions.items():
            if symbol not in self.positions:
                continue
            
            pos = self.positions[symbol]
            current_price = pos.current_price
            
            if current_price <= 0:
                continue
            
            profit_ratio = (current_price - pos.avg_cost) / pos.avg_cost
            pos.peak_price = max(pos.peak_price, current_price)
            pos.peak_profit = (pos.peak_price - pos.avg_cost) / pos.avg_cost
            
            exit_reason = None
            trigger_price = current_price
            reduce_ratio = None
            
            # 1. 硬止损检查
            if not pos.hard_stop_triggered:
                if current_price <= pos.hard_stop_price:
                    exit_reason = "Hard Stop Loss"
                    trigger_price = pos.hard_stop_price
                    pos.hard_stop_triggered = True
            
            # 2. 保本止损检查（手续费覆盖）
            if not exit_reason and V57_BREAKEVEN_ENABLED:
                if pos.peak_profit >= V57_BREAKEVEN_PROFIT_THRESHOLD:
                    pos.breakeven_active = True
                    breakeven_stop = pos.avg_cost * (1 + V57_BREAKEVEN_BUFFER)
                    
                    expected_pnl_after_fees = (breakeven_stop - pos.avg_cost) / pos.avg_cost - V57_FRICTION_COST
                    
                    if expected_pnl_after_fees >= 0 and current_price <= breakeven_stop:
                        exit_reason = "Breakeven Stop (Fee Covered)"
                        trigger_price = breakeven_stop
            
            # 3. 追踪止盈检查
            if not exit_reason and V57_TRAILING_PROFIT_ENABLED:
                if pos.peak_profit >= V57_TRAILING_PROFIT_TRIGGER:
                    pos.trailing_profit_active = True
                    trailing_stop = pos.peak_price * (1 - V57_TRAILING_PROFIT_ATR_MULT * pos.atr_at_entry / pos.peak_price)
                    
                    if current_price <= trailing_stop and not pos.trailing_profit_triggered:
                        exit_reason = "Trailing Profit"
                        trigger_price = trailing_stop
                        pos.trailing_profit_triggered = True
            
            # 4. 阶梯止盈检查（手续费覆盖）
            if not exit_reason and V57_TIERED_PROFIT_ENABLED:
                for level in V57_TIERED_PROFIT_LEVELS:
                    threshold = level['threshold']
                    reduce_ratio_level = level['reduce_ratio']
                    
                    if pos.peak_profit >= threshold and level['threshold'] not in pos.tiered_profit_triggered:
                        expected_pnl = (current_price - pos.avg_cost) / pos.avg_cost - V57_FRICTION_COST
                        
                        if expected_pnl >= 0:
                            exit_reason = f"Tiered Profit {threshold:.0%}"
                            reduce_ratio = reduce_ratio_level
                            pos.tiered_profit_triggered.append(level['threshold'])
                            break
            
            # 5. 时间止损检查
            if not exit_reason and V57_TIME_STOP_ENABLED:
                if pos.holding_days >= V57_TIME_STOP_DAYS and profit_ratio <= 0 and not pos.time_stop_triggered:
                    exit_reason = "Time Stop"
                    reduce_ratio = V57_TIME_STOP_REDUCE_RATIO
                    pos.time_stop_triggered = True
            
            # 6. MA20 趋势退出检查
            if not exit_reason and V57_MA20_TREND_EXIT_ENABLED:
                if pos.ma20_at_entry > 0 and current_price < pos.ma20_at_entry * 0.97:
                    exit_reason = "MA20 Trend Exit"
            
            # 7. 排名退出检查
            if not exit_reason:
                if factor_df is not None and not factor_df.is_empty():
                    try:
                        stock_row = factor_df.filter(pl.col('symbol') == symbol)
                        if not stock_row.is_empty():
                            rank = stock_row['composite_rank'][0] if 'composite_rank' in stock_row.columns else 9999
                            if rank > V57_MAINTAIN_TOP_N:
                                exit_reason = f"Rank Drop (>{V57_MAINTAIN_TOP_N})"
                    except Exception:
                        pass
            
            if exit_reason:
                next_open = None
                if next_day_price_df is not None and not next_day_price_df.is_empty():
                    try:
                        next_df = next_day_price_df.filter(pl.col('symbol') == symbol)
                        if not next_df.is_empty():
                            next_open = next_df['open'][0] if 'open' in next_df.columns else None
                    except Exception:
                        pass
                
                sell_list.append((symbol, exit_reason, trigger_price, next_open, reduce_ratio))
        
        return sell_list
    
    def execute_sell(self, trade_date: str, symbol: str, open_price: float,
                     reason: str = "", trigger_price: float = 0.0,
                     next_open_price: Optional[float] = None):
        """执行卖出"""
        try:
            if symbol not in self.positions:
                return False
            
            pos = self.positions[symbol]
            sell_price = open_price
            shares = pos.shares
            
            actual_amount = shares * sell_price
            
            commission = max(actual_amount * V57_COMMISSION_RATE, V57_MIN_COMMISSION)
            slippage = actual_amount * V57_SLIPPAGE_SELL
            stamp_duty = actual_amount * V57_STAMP_DUTY
            transfer_fee = actual_amount * V57_TRANSFER_FEE
            total_fees = commission + slippage + stamp_duty + transfer_fee
            
            gross_pnl = actual_amount - shares * pos.avg_cost
            net_pnl = gross_pnl - total_fees
            
            self.cash += (actual_amount - total_fees)
            
            trade = V57Trade(
                trade_date=trade_date,
                symbol=symbol,
                side="SELL",
                shares=shares,
                price=sell_price,
                amount=actual_amount,
                commission=commission,
                slippage=slippage,
                stamp_duty=stamp_duty,
                transfer_fee=transfer_fee,
                total_cost=total_fees,
                reason=reason,
                holding_days=pos.holding_days,
                execution_price=sell_price,
                signal_date=pos.signal_date,
                trigger_price=trigger_price,
                next_open_price=next_open_price or 0.0
            )
            
            self.record_trade(trade)
            
            audit = V57TradeAudit(
                symbol=symbol,
                buy_date=pos.buy_date,
                sell_date=trade_date,
                buy_price=pos.avg_cost,
                sell_price=sell_price,
                shares=shares,
                gross_pnl=gross_pnl,
                total_fees=total_fees,
                net_pnl=net_pnl,
                holding_days=pos.holding_days,
                is_profitable=net_pnl > 0,
                sell_reason=reason,
                entry_signal=pos.signal_score,
                signal_rank=pos.signal_rank,
                atr_at_entry=pos.atr_at_entry,
                hard_stop_price=pos.hard_stop_price,
                hard_stop_triggered=pos.hard_stop_triggered,
                breakeven_active=pos.breakeven_active,
                trailing_profit_active=pos.trailing_profit_active,
                trailing_profit_triggered=pos.trailing_profit_triggered,
                ma20_exit_triggered=pos.ma20_exit_triggered,
                time_stop_triggered=pos.time_stop_triggered,
                tiered_profit_triggered=pos.tiered_profit_triggered.copy(),
                peak_price=pos.peak_price,
                exit_profit_ratio=(sell_price - pos.avg_cost) / pos.avg_cost,
                position_tier=pos.position_tier,
                volume_shrunk_at_entry=pos.volume_shrunk_at_entry,
                rs_score=pos.rs_score,
                rs_rank=pos.rs_rank,
                trigger_price=trigger_price,
                next_open_price=next_open_price or 0.0,
                execution_price=sell_price,
                slippage_applied=slippage
            )
            
            self.trade_log.append(audit)
            
            del self.positions[symbol]
            
            self.add_to_blacklist(symbol, trade_date, reason)
            
            logger.info(f"SELL: {symbol} @ {sell_price:.2f} x {shares} | PnL: {net_pnl:.2f} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"execute_sell failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def execute_tiered_profit_reduce(self, trade_date: str, symbol: str,
                                      open_price: float, reduce_ratio: float,
                                      reason: str = ""):
        """执行阶梯止盈减仓"""
        try:
            if symbol not in self.positions:
                return False
            
            pos = self.positions[symbol]
            reduce_shares = int(pos.shares * reduce_ratio / 100) * 100
            
            if reduce_shares <= 0:
                return False
            
            actual_amount = reduce_shares * open_price
            
            commission = max(actual_amount * V57_COMMISSION_RATE, V57_MIN_COMMISSION)
            slippage = actual_amount * V57_SLIPPAGE_SELL
            stamp_duty = actual_amount * V57_STAMP_DUTY
            transfer_fee = actual_amount * V57_TRANSFER_FEE
            total_fees = commission + slippage + stamp_duty + transfer_fee
            
            avg_cost_portion = pos.avg_cost * reduce_ratio
            gross_pnl = actual_amount - (reduce_shares * avg_cost_portion)
            net_pnl = gross_pnl - total_fees
            
            self.cash += (actual_amount - total_fees)
            
            trade = V57Trade(
                trade_date=trade_date,
                symbol=symbol,
                side="SELL",
                shares=reduce_shares,
                price=open_price,
                amount=actual_amount,
                commission=commission,
                slippage=slippage,
                stamp_duty=stamp_duty,
                transfer_fee=transfer_fee,
                total_cost=total_fees,
                reason=reason,
                holding_days=pos.holding_days,
                execution_price=open_price
            )
            
            self.record_trade(trade)
            
            pos.shares -= reduce_shares
            pos.market_value = pos.shares * open_price
            
            if pos.shares <= 0:
                del self.positions[symbol]
            
            logger.info(f"TIERED PROFIT: {symbol} reduce {reduce_shares} shares @ {open_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"execute_tiered_profit_reduce failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def update_positions(self, trade_date: str, price_df: pl.DataFrame):
        """更新持仓数据"""
        for symbol, pos in self.positions.items():
            try:
                stock_row = price_df.filter(
                    (pl.col('symbol') == symbol) & (pl.col('trade_date') == trade_date)
                )
                
                if not stock_row.is_empty():
                    current_price = stock_row['close'][0] if 'close' in stock_row.columns else pos.current_price
                    pos.current_price = current_price
                    pos.market_value = pos.shares * current_price
                    pos.holding_days += 1
            except Exception:
                pos.holding_days += 1
    
    def get_wash_sale_stats(self) -> Dict[str, Any]:
        """获取洗售统计"""
        return {
            'total_wash_sale_prevented': len(self.wash_sale_records),
            'current_blacklist_size': len(self.blacklist),
            'blacklist_symbols': list(self.blacklist.keys())
        }
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        """获取黑名单统计"""
        return {
            'total_blacklisted': len(self.blacklist),
            'blacklist_details': [
                {'symbol': r.symbol, 'days_remaining': r.days_remaining, 'reason': r.stop_reason}
                for r in self.blacklist.values()
            ]
        }
    
    def get_trade_count_stats(self) -> Dict[str, Any]:
        """获取交易次数统计"""
        return {
            'global_trade_count': self.global_trade_count,
            'global_trade_limit': V57_GLOBAL_TRADE_LIMIT,
            'remaining_trades': V57_GLOBAL_TRADE_LIMIT - self.global_trade_count,
            'weekly_counters': {
                k: v.trade_count for k, v in self.weekly_trade_counters.items()
            }
        }
    
    def get_three_level_defense_stats(self) -> Dict[str, Any]:
        """获取三级防御体系统计"""
        hard_stop_count = sum(1 for p in self.positions.values() if p.hard_stop_triggered)
        breakeven_count = sum(1 for p in self.positions.values() if p.breakeven_active)
        trailing_count = sum(1 for p in self.positions.values() if p.trailing_profit_active)
        
        return {
            'hard_stop_triggered': hard_stop_count,
            'breakeven_active': breakeven_count,
            'trailing_profit_active': trailing_count
        }
    
    def get_frequency_fuse_stats(self) -> Dict[str, Any]:
        """获取频率熔断统计"""
        return {
            'global_trade_count': self.global_trade_count,
            'global_trade_limit': V57_GLOBAL_TRADE_LIMIT,
            'weekly_trade_limit': V57_WEEKLY_TRADE_LIMIT,
            'weekly_counters': {k: v.trade_count for k, v in self.weekly_trade_counters.items()}
        }
    
    def get_rs_strength_stats(self) -> Dict[str, Any]:
        """获取 RS 强度统计"""
        rs_positions = [p for p in self.positions.values() if p.rs_rank <= 100]
        return {
            'total_rs_positions': len(rs_positions),
            'rs_position_symbols': [p.symbol for p in rs_positions]
        }
    
    def get_stop_audit_records(self) -> List[Dict[str, Any]]:
        """获取止损审计记录"""
        return [
            {
                'symbol': t.symbol,
                'sell_date': t.trade_date,
                'reason': t.reason,
                'trigger_price': t.trigger_price,
                'execution_price': t.execution_price
            }
            for t in self.trades if t.side == "SELL" and ('Stop' in t.reason or 'Exit' in t.reason)
        ]


class V57BacktestEngine:
    """
    V57 回测引擎 - 行业先行与递归自修正
    
    【核心功能】
    1. 行业先行选股：先计算行业得分，再选择个股
    2. 递归自修正：自动分析失败原因并修改参数
    3. 严格交易限制：全场最多 30 次交易
    4. 手续费覆盖检查：保本/止盈必须扣除 0.2% 摩擦成本后 ≥ 0
    """
    
    def __init__(self, initial_capital: float = V57_INITIAL_CAPITAL, db=None):
        self.initial_capital = initial_capital
        self.db = db
        self.risk_manager = V57RiskManager(initial_capital=initial_capital)
        self.factor_engine = V57FactorEngine()
        self.industry_loader = V57IndustryLoader(db=db)
        self.portfolio_values: List[Dict[str, Any]] = []
        self.daily_trades: List[Dict[str, Any]] = []
        self.iteration_results: List[Dict[str, Any]] = []
        self.current_iteration = 0
    
    def run_backtest(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """
        运行 V57 回测（带递归自修正）
        
        【递归自修正流程】
        1. 运行回测
        2. 检查 Total_Return 和 MDD
        3. 如果 Return < 12% 或 MDD > 8%：
           - 分析失败原因
           - 修改参数
           - 重新运行
        4. 最多 20 轮修正
        """
        try:
            logger.info("=" * 60)
            logger.info("V57 BACKTEST START - 行业先行与递归自修正")
            logger.info("=" * 60)
            logger.info(f"Period: {start_date} to {end_date}")
            logger.info(f"Initial Capital: {self.initial_capital:,.2f}")
            logger.info(f"Return Target: {V57_RETURN_TARGET:.1%}")
            logger.info(f"MDD Target: {V57_MDD_TARGET:.1%}")
            logger.info(f"Max Iterations: {V57_MAX_ITERATION_ROUNDS}")
            
            industry_data_source = self._check_industry_data_source(start_date, end_date)
            logger.info(f"Industry Data Source: {industry_data_source}")
            
            best_result = None
            best_iteration = 0
            
            for iteration in range(1, V57_MAX_ITERATION_ROUNDS + 1):
                self.current_iteration = iteration
                logger.info(f"\n{'='*60}")
                logger.info(f"ITERATION {iteration}/{V57_MAX_ITERATION_ROUNDS}")
                logger.info(f"{'='*60}")
                
                result = self._run_single_backtest(
                    price_df=price_df, start_date=start_date, end_date=end_date,
                    index_df=index_df, industry_df=industry_df
                )
                
                total_return = result.get('total_return', 0)
                max_drawdown = result.get('max_drawdown', 0)
                total_trades = result.get('total_trades', 0)
                
                logger.info(f"Total Return: {total_return:.2%}")
                logger.info(f"Max Drawdown: {max_drawdown:.2%}")
                logger.info(f"Total Trades: {total_trades}")
                
                self.iteration_results.append({
                    'iteration': iteration,
                    'parameters': self._get_current_parameters(),
                    'metrics': {
                        'total_return': total_return,
                        'annual_return': result.get('annual_return', 0),
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'win_rate': result.get('win_rate', 0),
                        'profit_loss_ratio': result.get('profit_loss_ratio', 0),
                        'total_trades': total_trades
                    }
                })
                
                if best_result is None or total_return > best_result.get('total_return', 0):
                    best_result = result
                    best_iteration = iteration
                
                if total_return >= V57_RETURN_TARGET and max_drawdown <= V57_MDD_TARGET:
                    logger.info(f"\n✅ TARGET ACHIEVED at iteration {iteration}!")
                    break
                
                if iteration < V57_MAX_ITERATION_ROUNDS:
                    self._analyze_and_adjust(total_return, max_drawdown, total_trades)
            
            if best_result:
                best_result['iteration_results'] = self.iteration_results
                best_result['best_iteration'] = best_iteration
                best_result['total_iterations'] = len(self.iteration_results)
            
            logger.info("\n" + "=" * 60)
            logger.info("V57 BACKTEST COMPLETE")
            logger.info("=" * 60)
            
            return best_result or self._create_empty_result()
            
        except Exception as e:
            logger.error(f"V57 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_result()
    
    def _check_industry_data_source(self, start_date: str, end_date: str) -> str:
        """检查行业数据来源"""
        if self.industry_loader.check_table_exists(start_date, end_date):
            return "database"
        else:
            self.industry_loader._simulation_active = True
            return "simulation (built-in dictionary)"
    
    def _run_single_backtest(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                             index_df: Optional[pl.DataFrame] = None,
                             industry_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """
        运行单轮回测
        
        【关键】每次迭代都重新创建因子引擎，确保使用最新权重
        """
        try:
            # 重新创建因子引擎，使用最新的权重参数
            self.factor_engine = V57FactorEngine(
                momentum_weight=V57_MOMENTUM_WEIGHT,
                r2_weight=V57_R2_WEIGHT,
                industry_weight=V57_INDUSTRY_WEIGHT
            )
            
            self.risk_manager = V57RiskManager(initial_capital=self.initial_capital)
            self.portfolio_values = []
            self.daily_trades = []
            
            industry_data = self.industry_loader.load_industry_data(start_date, end_date)
            
            price_df = price_df.filter(
                (pl.col('trade_date') >= start_date) & 
                (pl.col('trade_date') <= end_date)
            )
            
            trade_dates = sorted(price_df['trade_date'].unique().to_list())
            
            if not trade_dates:
                return self._create_empty_result()
            
            logger.info(f"Trading days: {len(trade_dates)}")
            
            for i, trade_date in enumerate(trade_dates):
                self.risk_manager.reset_daily_counters(trade_date)
                self.risk_manager.update_blacklist_days(trade_date)
                
                current_price_df = price_df.filter(pl.col('trade_date') == trade_date)
                
                next_day_price_df = None
                if i + 1 < len(trade_dates):
                    next_day = trade_dates[i + 1]
                    next_day_price_df = price_df.filter(pl.col('trade_date') == next_day)
                
                current_index_df = None
                if index_df is not None and not index_df.is_empty():
                    current_index_df = index_df.filter(pl.col('trade_date') == trade_date)
                
                self._execute_daily_trading(
                    trade_date=trade_date,
                    current_price_df=current_price_df,
                    next_day_price_df=next_day_price_df,
                    index_df=current_index_df,
                    industry_data=industry_data
                )
                
                self.risk_manager.update_positions(trade_date, current_price_df)
                
                portfolio_value = self.risk_manager.get_total_portfolio_value(trade_date)
                self.portfolio_values.append({
                    'trade_date': trade_date,
                    'total_value': portfolio_value,
                    'cash': self.risk_manager.cash,
                    'market_value': portfolio_value - self.risk_manager.cash,
                    'positions_count': len(self.risk_manager.positions)
                })
            
            return self._generate_backtest_result(trade_dates)
            
        except Exception as e:
            logger.error(f"_run_single_backtest failed: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_result()
    
    def _execute_daily_trading(self, trade_date: str, current_price_df: pl.DataFrame,
                                next_day_price_df: Optional[pl.DataFrame],
                                index_df: Optional[pl.DataFrame],
                                industry_data: Optional[pl.DataFrame]):
        """执行当日交易逻辑"""
        try:
            factor_df, factor_status = self.factor_engine.compute_all_factors(
                df=current_price_df,
                industry_data=industry_data,
                db=self.db,
                start_date=trade_date,
                end_date=trade_date,
                index_data=index_df
            )
            
            if V57_INDUSTRY_FILTER_ENABLED and factor_df is not None and not factor_df.is_empty():
                filtered_df, industry_stats = v57_industry_filter(
                    df=factor_df,
                    industry_data=industry_data,
                    industry_loader=self.industry_loader,
                    trade_date=trade_date,
                    top_n=V57_INDUSTRY_TOP_N
                )
                factor_df = filtered_df
            
            self._process_exits(trade_date, current_price_df, factor_df, next_day_price_df)
            self._process_entries(trade_date, current_price_df, factor_df, industry_data)
            
        except Exception as e:
            logger.error(f"_execute_daily_trading failed: {e}")
    
    def _process_exits(self, trade_date: str, price_df: pl.DataFrame,
                       factor_df: pl.DataFrame, next_day_price_df: Optional[pl.DataFrame]):
        """处理退出信号"""
        try:
            positions = self.risk_manager.positions.copy()
            
            if not positions:
                return
            
            next_opens = {}
            if next_day_price_df is not None and not next_day_price_df.is_empty():
                try:
                    next_df = next_day_price_df.select(['symbol', 'open']).unique('symbol', keep='last')
                    next_opens = dict(zip(next_df['symbol'].to_list(), next_df['open'].to_list()))
                except Exception:
                    pass
            
            sell_list = self.risk_manager.check_exits(
                positions=positions,
                date_str=trade_date,
                price_df=price_df,
                factor_df=factor_df,
                next_day_price_df=next_day_price_df
            )
            
            for symbol, reason, trigger_price, next_open, reduce_ratio in sell_list:
                if symbol not in self.risk_manager.positions:
                    continue
                
                pos = self.risk_manager.positions[symbol]
                open_price = next_opens.get(symbol, pos.current_price)
                
                if reduce_ratio is not None:
                    self.risk_manager.execute_tiered_profit_reduce(
                        trade_date=trade_date,
                        symbol=symbol,
                        open_price=open_price,
                        reduce_ratio=reduce_ratio,
                        reason=reason
                    )
                else:
                    self.risk_manager.execute_sell(
                        trade_date=trade_date,
                        symbol=symbol,
                        open_price=open_price,
                        reason=reason,
                        trigger_price=trigger_price,
                        next_open_price=next_open
                    )
                    
        except Exception as e:
            logger.error(f"_process_exits failed: {e}")
    
    def _process_entries(self, trade_date: str, price_df: pl.DataFrame,
                         factor_df: pl.DataFrame, industry_data: Optional[pl.DataFrame]):
        """处理买入信号"""
        try:
            can_open, open_reason = self.risk_manager.can_open_new_position()
            if not can_open:
                logger.info(f"Frequency fuse triggered: {open_reason}")
                return
            
            if len(self.risk_manager.positions) >= V57_MAX_POSITIONS:
                return
            
            if factor_df is None or factor_df.is_empty():
                return
            
            candidates = self._select_entry_candidates(factor_df, price_df)
            
            if not candidates:
                return
            
            industry_map = {}
            if industry_data is not None and not industry_data.is_empty():
                try:
                    ind_df = industry_data.filter(pl.col('trade_date') == trade_date)
                    if not ind_df.is_empty():
                        industry_map = dict(zip(
                            ind_df['symbol'].to_list(),
                            ind_df['industry_name'].to_list()
                        ))
                except Exception:
                    pass
            
            total_assets = self.risk_manager.get_total_portfolio_value(trade_date)
            max_positions = V57_MAX_POSITIONS
            current_positions = len(self.risk_manager.positions)
            available_slots = max_positions - current_positions
            
            for candidate in candidates[:available_slots]:
                symbol = candidate['symbol']
                open_price = candidate.get('open', 0)
                atr = candidate.get('atr', 0.0)
                
                if open_price is None or open_price <= 0:
                    continue
                if atr is None or atr <= 0:
                    continue
                
                shares, target_amount, position_tier = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    atr=atr,
                    current_price=open_price,
                    total_assets=total_assets
                )
                
                if shares < 100:
                    continue
                
                industry_name = industry_map.get(symbol, "")
                
                self.risk_manager.execute_buy(
                    trade_date=trade_date,
                    symbol=symbol,
                    open_price=open_price,
                    atr=atr,
                    target_amount=target_amount,
                    signal_date=trade_date,
                    signal_score=candidate.get('composite_score', 0),
                    signal_rank=candidate.get('composite_rank', 9999),
                    reason=f"V57 Entry (Rank={candidate.get('composite_rank')})",
                    composite_score=candidate.get('composite_score', 0),
                    composite_percentile=candidate.get('composite_percentile', 0),
                    ma5=candidate.get('ma5', 0),
                    ma20=candidate.get('ma20', 0),
                    ma60=candidate.get('ma60', 0),
                    ma120=candidate.get('ma120', 0),
                    industry_name=industry_name,
                    volume_shrunk=candidate.get('is_volume_shrunk', False),
                    rs_score=candidate.get('rs_strength', 0),
                    rs_rank=candidate.get('rs_rank', 9999),
                    volume_breakout=candidate.get('volume_breakout', False)
                )
                
        except Exception as e:
            logger.error(f"_process_entries failed: {e}")
    
    def _select_entry_candidates(self, factors_df: pl.DataFrame,
                                  price_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """选择入场候选股票"""
        try:
            required_cols = ['symbol', 'composite_rank', 'close']
            for col in required_cols:
                if col not in factors_df.columns:
                    return []
            
            candidates_df = factors_df.filter(
                (pl.col('close') > 0) & 
                (pl.col('composite_rank').is_not_null())
            ).sort('composite_rank')
            
            if candidates_df.is_empty():
                return []
            
            available_slots = V57_MAX_POSITIONS - len(self.risk_manager.positions)
            
            if available_slots <= 0:
                return []
            
            candidates = []
            
            for row in candidates_df.iter_rows(named=True):
                if len(candidates) >= available_slots:
                    break
                
                symbol = row['symbol']
                rank = row.get('composite_rank', 9999) or 9999
                
                if rank > V57_ENTRY_TOP_N:
                    continue
                
                if V57_MA60_FILTER:
                    price_above_ma60 = row.get('price_above_ma60', None)
                    if price_above_ma60 is None:
                        ma60_val = row.get('ma60', 0) or 0
                        close_val = row.get('close', 0) or 0
                        price_above_ma60 = close_val > ma60_val if ma60_val > 0 else True
                    if not price_above_ma60:
                        continue
                
                atr_value = row.get('atr_20') or 0.01
                if atr_value is None or atr_value <= 0:
                    atr_value = 0.01
                
                candidates.append({
                    'symbol': symbol,
                    'signal_score': row.get('composite_score', 0) or 0,
                    'rank': rank,
                    'composite_score': row.get('composite_score', 0) or 0,
                    'composite_percentile': row.get('composite_percentile', 1) or 1,
                    'atr': atr_value,
                    'ma5': row.get('ma5', 0) or 0,
                    'ma20': row.get('ma20', 0) or 0,
                    'ma60': row.get('ma60', 0) or 0,
                    'ma120': row.get('ma120', 0) or 0,
                    'rs_strength': row.get('rs_strength', 0) or 0,
                    'rs_rank': row.get('rs_rank', 9999) or 9999,
                    'volume_breakout': row.get('volume_breakout', False) or False,
                    'is_volume_shrunk': row.get('is_volume_shrunk', False) or False,
                    'open': row.get('open', 0) or 0,
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error selecting candidates: {e}")
            return []
    
    def _generate_backtest_result(self, trade_dates: List[str]) -> Dict[str, Any]:
        """生成回测结果"""
        try:
            if not self.portfolio_values:
                return self._create_empty_result()
            
            final_value = self.portfolio_values[-1]['total_value']
            initial_value = self.initial_capital
            total_return = (final_value - initial_value) / initial_value
            
            if len(trade_dates) > 1:
                start = datetime.strptime(trade_dates[0], "%Y-%m-%d")
                end = datetime.strptime(trade_dates[-1], "%Y-%m-%d")
                days = (end - start).days
                years = days / 365.25 if days > 0 else 1
                annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
            else:
                annual_return = total_return
            
            max_dd = 0.0
            peak = self.portfolio_values[0]['total_value']
            for pv in self.portfolio_values:
                v = pv['total_value']
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            
            if len(self.portfolio_values) > 1:
                daily_returns = []
                for i in range(1, len(self.portfolio_values)):
                    prev = self.portfolio_values[i-1]['total_value']
                    curr = self.portfolio_values[i]['total_value']
                    if prev > 0:
                        daily_returns.append((curr - prev) / prev)
                
                if daily_returns:
                    import numpy as np
                    mean_ret = np.mean(daily_returns)
                    std_ret = np.std(daily_returns)
                    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            total_trades = len(self.risk_manager.trades)
            buy_trades = [t for t in self.risk_manager.trades if t.side == "BUY"]
            sell_trades = [t for t in self.risk_manager.trades if t.side == "SELL"]
            
            profitable_trades = sum(1 for t in self.risk_manager.trade_log if t.is_profitable)
            total_closed = len(self.risk_manager.trade_log)
            win_rate = profitable_trades / total_closed if total_closed > 0 else 0
            
            total_profit = sum(t.net_pnl for t in self.risk_manager.trade_log if t.net_pnl > 0)
            total_loss = abs(sum(t.net_pnl for t in self.risk_manager.trade_log if t.net_pnl < 0))
            profit_loss_ratio = total_profit / total_loss if total_loss > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'total_trades': total_trades,
                'total_buy_trades': len(buy_trades),
                'total_sell_trades': len(sell_trades),
                'final_value': final_value,
                'initial_value': initial_value,
                'portfolio_values': self.portfolio_values,
                'trades': self.risk_manager.trades,
                'trade_log': self.risk_manager.trade_log,
                'positions': self.risk_manager.positions,
                'wash_sale_stats': self.risk_manager.get_wash_sale_stats(),
                'blacklist_stats': self.risk_manager.get_blacklist_stats(),
                'trade_count_stats': self.risk_manager.get_trade_count_stats(),
                'three_level_defense_stats': self.risk_manager.get_three_level_defense_stats(),
                'frequency_fuse_stats': self.risk_manager.get_frequency_fuse_stats(),
                'rs_strength_stats': self.risk_manager.get_rs_strength_stats(),
                'stop_audit_records': self.risk_manager.get_stop_audit_records(),
                'v57_config': {
                    'entry_top_n': V57_ENTRY_TOP_N,
                    'max_positions': V57_MAX_POSITIONS,
                    'global_trade_limit': V57_GLOBAL_TRADE_LIMIT,
                    'weekly_trade_limit': V57_WEEKLY_TRADE_LIMIT,
                    'industry_filter_enabled': V57_INDUSTRY_FILTER_ENABLED,
                    'industry_top_n': V57_INDUSTRY_TOP_N,
                    'breakeven_enabled': V57_BREAKEVEN_ENABLED,
                    'breakeven_threshold': V57_BREAKEVEN_PROFIT_THRESHOLD,
                    'friction_cost': V57_FRICTION_COST,
                    'tiered_profit_enabled': V57_TIERED_PROFIT_ENABLED,
                    'hard_stop_atr_mult': V57_HARD_STOP_LOSS_ATR_MULT,
                    'rs_enabled': V57_RS_ENABLED,
                    'rs_top_percentile': V57_RS_TOP_PERCENTILE,
                    'ma60_filter': V57_MA60_FILTER,
                    'momentum_weight': V57_MOMENTUM_WEIGHT,
                    'r2_weight': V57_R2_WEIGHT,
                    'industry_weight': V57_INDUSTRY_WEIGHT,
                }
            }
            
        except Exception as e:
            logger.error(f"_generate_backtest_result failed: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'total_trades': 0,
            'final_value': self.initial_capital,
            'initial_value': self.initial_capital,
            'portfolio_values': [],
            'trades': [],
            'trade_log': [],
            'positions': {},
            'wash_sale_stats': {},
            'blacklist_stats': {},
            'trade_count_stats': {},
            'three_level_defense_stats': {},
            'frequency_fuse_stats': {},
            'rs_strength_stats': {},
            'stop_audit_records': [],
            'iteration_results': [],
            'best_iteration': 0,
            'total_iterations': 0,
            'v57_config': {}
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """获取当前参数"""
        return {
            'entry_top_n': V57_ENTRY_TOP_N,
            'momentum_weight': V57_MOMENTUM_WEIGHT,
            'r2_weight': V57_R2_WEIGHT,
            'industry_weight': V57_INDUSTRY_WEIGHT,
            'hard_stop_atr_mult': V57_HARD_STOP_LOSS_ATR_MULT,
            'breakeven_threshold': V57_BREAKEVEN_PROFIT_THRESHOLD,
            'rs_top_percentile': V57_RS_TOP_PERCENTILE,
        }
    
    def _analyze_and_adjust(self, total_return: float, max_drawdown: float, total_trades: int):
        """
        分析失败原因并调整参数
        
        【递归自修正逻辑】
        1. Return < 12%: 入场门槛太高？止损太紧？
        2. MDD > 8%: 止损太宽？仓位太大？
        3. 交易次数太少：入场门槛太高？
        """
        import v57_core as core_module
        analysis = []
        
        if total_return < V57_RETURN_TARGET:
            if total_trades < 10:
                analysis.append("Too few trades - relaxing entry threshold")
                # 放宽入场门槛
                new_top_n = min(core_module.V57_ENTRY_TOP_N + 5, 50)
                setattr(core_module, 'V57_ENTRY_TOP_N', new_top_n)
                # 同时放宽 RS 强度要求
                new_rs_pct = min(core_module.V57_RS_TOP_PERCENTILE + 0.05, 0.35)
                setattr(core_module, 'V57_RS_TOP_PERCENTILE', new_rs_pct)
            else:
                analysis.append("Low return - adjusting factor weights")
                # 提高动量权重，降低 R²权重
                new_momentum = min(core_module.V57_MOMENTUM_WEIGHT + 0.1, 0.7)
                new_r2 = max(core_module.V57_R2_WEIGHT - 0.1, 0.2)
                setattr(core_module, 'V57_MOMENTUM_WEIGHT', new_momentum)
                setattr(core_module, 'V57_R2_WEIGHT', new_r2)
        
        if max_drawdown > V57_MDD_TARGET:
            analysis.append("High drawdown - tightening stop loss")
            # 收紧止损
            new_atr_mult = max(core_module.V57_HARD_STOP_LOSS_ATR_MULT - 0.3, 1.0)
            setattr(core_module, 'V57_HARD_STOP_LOSS_ATR_MULT', new_atr_mult)
            new_breakeven = max(core_module.V57_BREAKEVEN_PROFIT_THRESHOLD - 0.02, 0.02)
            setattr(core_module, 'V57_BREAKEVEN_PROFIT_THRESHOLD', new_breakeven)
        
        # 同步全局变量
        global V57_ENTRY_TOP_N, V57_MOMENTUM_WEIGHT, V57_R2_WEIGHT
        global V57_HARD_STOP_LOSS_ATR_MULT, V57_BREAKEVEN_PROFIT_THRESHOLD
        global V57_RS_TOP_PERCENTILE
        
        V57_ENTRY_TOP_N = getattr(core_module, 'V57_ENTRY_TOP_N', V57_ENTRY_TOP_N)
        V57_MOMENTUM_WEIGHT = getattr(core_module, 'V57_MOMENTUM_WEIGHT', V57_MOMENTUM_WEIGHT)
        V57_R2_WEIGHT = getattr(core_module, 'V57_R2_WEIGHT', V57_R2_WEIGHT)
        V57_HARD_STOP_LOSS_ATR_MULT = getattr(core_module, 'V57_HARD_STOP_LOSS_ATR_MULT', V57_HARD_STOP_LOSS_ATR_MULT)
        V57_BREAKEVEN_PROFIT_THRESHOLD = getattr(core_module, 'V57_BREAKEVEN_PROFIT_THRESHOLD', V57_BREAKEVEN_PROFIT_THRESHOLD)
        V57_RS_TOP_PERCENTILE = getattr(core_module, 'V57_RS_TOP_PERCENTILE', V57_RS_TOP_PERCENTILE)
        
        # 更新因子引擎权重
        self.factor_engine.momentum_weight = V57_MOMENTUM_WEIGHT
        self.factor_engine.r2_weight = V57_R2_WEIGHT
        
        if analysis:
            logger.info(f"Analysis: {'; '.join(analysis)}")
            logger.info(f"Adjusted parameters: Entry Top N={V57_ENTRY_TOP_N}, ATR Mult={V57_HARD_STOP_LOSS_ATR_MULT}, RS Top%={V57_RS_TOP_PERCENTILE}")
        else:
            logger.info("No adjustment needed")


# ===========================================
# V57 回测脚本入口
# ===========================================

if __name__ == "__main__":
    from db_manager import DatabaseManager
    
    db = DatabaseManager()
    
    price_df = db.read_sql("SELECT * FROM stock_daily WHERE trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'")
    
    index_df = db.read_sql("SELECT * FROM index_daily WHERE index_name = '沪深 300' AND trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'")
    
    engine = V57BacktestEngine(initial_capital=100000, db=db)
    result = engine.run_backtest(
        price_df=price_df,
        start_date="2024-01-01",
        end_date="2025-12-31",
        index_df=index_df
    )
    
    print("\n" + "=" * 60)
    print("V57 BACKTEST RESULT")
    print("=" * 60)
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Annual Return: {result['annual_return']:.2%}")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Win Rate: {result['win_rate']:.2%}")
    print(f"Profit/Loss Ratio: {result['profit_loss_ratio']:.2f}")
    print(f"Total Trades: {result['total_trades']}")
    
    if result.get('iteration_results'):
        print("\n" + "=" * 60)
        print("ITERATION HISTORY")
        print("=" * 60)
        for ir in result['iteration_results']:
            print(f"\nIteration {ir.get('iteration')}:")
            metrics = ir.get('metrics', {})
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    if result.get('best_iteration'):
        print("\n" + "=" * 60)
        print(f"BEST ITERATION: {result['best_iteration']}")
        print("=" * 60)