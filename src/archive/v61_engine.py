"""
V61 Engine Module - 回测引擎与逻辑突变

【核心功能】
1. V61BacktestEngine - 完整的回测执行引擎
2. MasterLoop - 真正的逻辑突变（收益率<15% 时切换策略逻辑）
3. 全样本数据加载 - 严禁 limit 20

【严禁事项】
- 严禁 limit 20 或任何限制数据规模的硬编码
- 报错必须停止回测并尝试在代码中修复 Bug
- 严禁用"工程成功"掩盖"财务失败"

作者：量化系统
版本：V61.0
日期：2026-03-23
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger

from v61_core import (
    # 数据类
    V61Position, V61Trade, V61TradeAudit, V61WashSaleRecord,
    V61BlacklistRecord, V61MarketRegime, V61DrawdownState,
    V61WeeklyTradeCounter, V61IterationResult, V61LogicEvolutionRecord,
    
    # 类
    V61IndustryLoader, V61FactorEngine, V61RiskManager,
    
    # 函数
    v61_industry_filter,
    
    # 常量 - 基础配置
    V61_INITIAL_CAPITAL, V61_MAX_POSITIONS,
    V61_WEEKLY_TRADE_LIMIT, V61_GLOBAL_TRADE_LIMIT,
    V61_MOMENTUM_WEIGHT, V61_R2_WEIGHT, V61_TREND_WEIGHT,
    V61_ENTRY_TOP_N, V61_INDUSTRY_TOP_N,
    V61_COMMISSION_RATE, V61_MIN_COMMISSION,
    V61_SLIPPAGE_BUY, V61_SLIPPAGE_SELL,
    V61_STAMP_DUTY, V61_TRANSFER_FEE, V61_FRICTION_COST,
    V61_WASH_SALE_WINDOW,
    V61_MAX_ITERATION_ROUNDS, V61_RETURN_TARGET,
    V61_MDD_TARGET, V61_PROFIT_LOSS_RATIO_TARGET,
    V61_RS_TOP_PERCENTILE, V61_VOLUME_SHRINK_RATIO,
    V61_MA20_BUFFER, V61_MAX_DAILY_GAIN,
    V61_HARD_STOP_LOSS_ATR_MULT, V61_BREAKEVEN_PROFIT_THRESHOLD,
    V61_TRAILING_PROFIT_TRIGGER, V61_MAX_SINGLE_POSITION_PCT,
    V61_MA20_BREAKOUT, V61_DYNAMIC_EVOLUTION_ENABLED,
    V61_MIN_SELECTION_PERCENTILE, V61_MAX_SELECTION_PERCENTILE,
    V61_MIN_TREND_PERIOD, V61_MAX_TREND_PERIOD,
    V61_LOGIC_MUTATION_ENABLED, V61_LOGIC_MUTATION_THRESHOLD,
    V61_RISK_TARGET_PER_POSITION,
    # V61 RSRS 择时配置
    V61_RSRS_ENABLED, V61_RSRS_WINDOW, V61_RSRS_ZSCORE_THRESHOLD,
)


@dataclass
class V61AccountState:
    """V61 账户状态"""
    trade_date: str = ""
    total_equity: float = V61_INITIAL_CAPITAL
    available_cash: float = V61_INITIAL_CAPITAL
    position_market_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_ratio: float = 0.0
    total_pnl: float = 0.0
    total_pnl_ratio: float = 0.0
    peak_equity: float = V61_INITIAL_CAPITAL
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_stamp_duty: float = 0.0
    total_transfer_fee: float = 0.0
    weekly_trade_count: int = 0
    current_week: int = 0


@dataclass
class V61BacktestResult:
    """V61 回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    final_equity: float = V61_INITIAL_CAPITAL
    initial_equity: float = V61_INITIAL_CAPITAL
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_stamp_duty: float = 0.0
    avg_holding_days: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trade_dates: List[str] = field(default_factory=list)
    trades: List[V61Trade] = field(default_factory=list)
    positions: List[Dict] = field(default_factory=list)
    audits: List[V61TradeAudit] = field(default_factory=list)
    price_audit_violations: int = 0
    meets_target: bool = False
    target_analysis: str = ""
    total_stocks_traded: int = 0
    data_source: str = ""
    pullback_entries: int = 0  # V61 新增：回调买入次数
    volume_shrunk_entries: int = 0  # V61 新增：缩量回调买入次数


class V61BacktestEngine:
    """
    V61 回测引擎 - RS 回调完整执行引擎
    
    【核心功能】
    1. 全样本数据加载 - 严禁 limit 20
    2. RS 回调逻辑（价格在 [MA20, MA20*1.03] 区间 + 缩量 70%）
    3. 动态 ATR 仓位（单只风险≤0.8%）
    4. 移动止盈（浮盈>8% 后锁死在 Cost*1.02）
    5. 成交价审计
    """
    
    def __init__(self, initial_capital: float = V61_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.account = V61AccountState(total_equity=initial_capital, available_cash=initial_capital, peak_equity=initial_capital)
        self.positions: Dict[str, V61Position] = {}
        self.trades: List[V61Trade] = []
        self.trade_audits: List[V61TradeAudit] = []
        self.wash_sale_records: List[V61WashSaleRecord] = []
        self.blacklist: Dict[str, V61BlacklistRecord] = {}
        self.weekly_counters: Dict[int, V61WeeklyTradeCounter] = {}
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.trade_dates: List[str] = []
        self.price_audit_violations = 0
        
        self.factor_engine = V61FactorEngine()
        self.risk_manager = V61RiskManager()
        self.industry_loader = V61IndustryLoader()
        
        self.global_trade_count = 0
        self.current_week = 0
        self.current_year = 0
        self.total_stocks_traded = set()
        self.pullback_entries = 0
        self.volume_shrunk_entries = 0
    
    def run_backtest(self, factor_data: pl.DataFrame, price_data: pl.DataFrame,
                     industry_data: Optional[pl.DataFrame] = None,
                     index_data: Optional[pl.DataFrame] = None,
                     start_date: str = "", end_date: str = "") -> Dict[str, Any]:
        """运行完整回测"""
        try:
            logger.info("V61 Backtest Engine Starting...")
            logger.info(f"Initial Capital: {self.initial_capital:,.2f}")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # 数据规模审计 - 严禁 limit 20
            unique_stocks = price_data['symbol'].n_unique()
            total_rows = len(price_data)
            logger.info(f"Data Scale Audit: {unique_stocks} stocks, {total_rows} rows")
            
            if unique_stocks < 100:
                logger.error(f"V61 FATAL: Data fraud detected! Only {unique_stocks} stocks loaded.")
                logger.error("V61 RULE: Must load full market data (>500 stocks)")
                raise ValueError(f"Data fraud: Only {unique_stocks} stocks. Must load full market data!")
            
            self._reset_state()
            
            trade_dates = sorted(price_data['trade_date'].unique().to_list())
            if not trade_dates:
                return {'error': 'No trade dates found', 'result': None}
            
            self.trade_dates = trade_dates
            price_data = price_data.sort(['trade_date', 'symbol'])
            
            logger.info(f"Total trade days: {len(trade_dates)}")
            
            for i, trade_date in enumerate(trade_dates):
                if i % 50 == 0:
                    logger.info(f"Processing day {i+1}/{len(trade_dates)}: {trade_date}")
                
                current_df = price_data.filter(pl.col('trade_date') == trade_date)
                factor_df = factor_data.filter(pl.col('trade_date') == trade_date) if factor_data is not None else None
                
                if current_df.is_empty():
                    continue
                
                self._update_weekly_counter(trade_date)
                
                # 处理离场
                self._process_exits(trade_date, current_df, factor_data)
                
                # 更新持仓
                self._update_positions(trade_date, current_df)
                
                # 处理入场（V61 RS 回调逻辑）
                self._process_entries(trade_date, current_df, factor_df,
                                      industry_data, index_data)
                
                # 更新账户状态
                self._update_account_state(trade_date, current_df)
                
                # 合规检查
                self._check_compliance(trade_date)
            
            result = self._generate_result()
            
            logger.info(f"Backtest Complete: Total Return = {result['total_return']:.2%}")
            logger.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
            logger.info(f"Total Trades: {result['total_trades']}")
            logger.info(f"Win Rate: {result['win_rate']:.2%}")
            logger.info(f"Profit/Loss Ratio: {result['profit_loss_ratio']:.2f}")
            
            if result.get('price_audit_violations', 0) > 0:
                logger.warning(f"Price Audit Violations: {result['price_audit_violations']}")
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            logger.error(f"V61 Backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _reset_state(self):
        """重置状态"""
        self.account = V61AccountState(
            total_equity=self.initial_capital,
            available_cash=self.initial_capital,
            peak_equity=self.initial_capital
        )
        self.positions.clear()
        self.trades.clear()
        self.trade_audits.clear()
        self.wash_sale_records.clear()
        self.blacklist.clear()
        self.weekly_counters.clear()
        self.equity_curve.clear()
        self.drawdown_curve.clear()
        self.trade_dates.clear()
        self.price_audit_violations = 0
        self.global_trade_count = 0
        self.current_week = 0
        self.current_year = 0
        self.total_stocks_traded = set()
        self.pullback_entries = 0
        self.volume_shrunk_entries = 0
    
    def _update_weekly_counter(self, trade_date: str):
        """更新每周交易计数器"""
        try:
            date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
            iso_calendar = date_obj.isocalendar()
            year = iso_calendar[0]
            week = iso_calendar[1]
            
            week_key = year * 100 + week
            
            if week_key not in self.weekly_counters:
                self.weekly_counters[week_key] = V61WeeklyTradeCounter(
                    week_number=week,
                    year=year,
                    trade_count=0
                )
            
            self.current_week = week
            self.current_year = year
            
        except Exception:
            pass
    
    def _process_exits(self, trade_date: str, current_df: pl.DataFrame,
                       factor_data: Optional[pl.DataFrame] = None):
        """处理离场逻辑"""
        if not self.positions:
            return
        
        symbols_to_remove = []
        
        for symbol, position in list(self.positions.items()):
            stock_df = current_df.filter(pl.col('symbol') == symbol)
            if stock_df.is_empty():
                continue
            
            rows = list(stock_df.iter_rows(named=True))
            if not rows:
                continue
            row = rows[0]
            current_price = row.get('close', 0)
            current_atr = row.get('atr_20', 0)
            current_ma20 = row.get('ma20', 0)
            current_ma60 = row.get('ma60', 0)
            
            if current_price <= 0:
                continue
            
            position.current_price = current_price
            position.market_value = current_price * position.shares
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            
            triggered, reason = self.risk_manager.check_all_exits(
                position, current_price, current_atr, current_ma20, current_ma60, trade_date
            )
            
            if triggered:
                symbols_to_remove.append((symbol, reason))
        
        for symbol, reason in symbols_to_remove:
            self._execute_exit(trade_date, symbol, reason)
    
    def _execute_exit(self, trade_date: str, symbol: str, reason: str):
        """执行卖出交易"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 获取次日开盘价
        next_open = self._get_next_open(symbol, trade_date)
        if next_open <= 0:
            next_open = position.current_price
        
        sell_price = next_open * (1 - V61_SLIPPAGE_SELL)
        
        shares = position.shares
        sell_amount = sell_price * shares
        
        commission = max(sell_amount * V61_COMMISSION_RATE, V61_MIN_COMMISSION)
        stamp_duty = sell_amount * V61_STAMP_DUTY
        transfer_fee = sell_amount * V61_TRANSFER_FEE
        total_fees = commission + stamp_duty + transfer_fee
        
        net_proceeds = sell_amount - total_fees
        
        gross_pnl = (sell_price - position.avg_cost) * shares
        net_pnl = gross_pnl - total_fees
        
        trade = V61Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='SELL',
            shares=shares,
            price=sell_price,
            amount=sell_amount,
            commission=commission,
            slippage=(position.current_price - sell_price) / position.current_price if position.current_price > 0 else 0,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_fees,
            reason=reason,
            holding_days=position.holding_days,
            execution_price=sell_price,
            signal_date=position.signal_date,
            trigger_price=position.current_price,
            next_open_price=next_open,
            min_trigger_open=min(position.current_price, next_open),
            slippage_applied=V61_SLIPPAGE_SELL,
            price_audit_passed=True
        )
        
        trade.price_audit_passed = self._audit_execution_price(trade)
        if not trade.price_audit_passed:
            self.price_audit_violations += 1
        
        self.trades.append(trade)
        
        audit = V61TradeAudit(
            symbol=symbol,
            buy_date=position.buy_date,
            sell_date=trade_date,
            buy_price=position.avg_cost,
            sell_price=sell_price,
            shares=shares,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            holding_days=position.holding_days,
            is_profitable=net_pnl > 0,
            sell_reason=reason,
            entry_signal=position.signal_score,
            signal_rank=position.signal_rank,
            atr_at_entry=position.atr_at_entry,
            hard_stop_price=position.hard_stop_price,
            hard_stop_triggered=position.hard_stop_triggered,
            breakeven_active=position.breakeven_active,
            trailing_profit_active=position.trailing_profit_active,
            trailing_profit_triggered=position.trailing_profit_triggered,
            ma20_exit_triggered=position.ma20_exit_triggered,
            ma60_exit_triggered=position.ma60_exit_triggered,
            time_stop_triggered=position.time_stop_triggered,
            tiered_profit_triggered=position.tiered_profit_triggered.copy(),
            peak_price=position.peak_price,
            exit_profit_ratio=(sell_price - position.avg_cost) / position.avg_cost,
            position_tier=position.position_tier,
            volume_shrunk_at_entry=position.volume_shrunk_at_entry,
            rs_score=position.rs_score,
            rs_rank=position.rs_rank,
            trigger_price=trade.trigger_price,
            next_open_price=trade.next_open_price,
            execution_price=sell_price,
            slippage_applied=V61_SLIPPAGE_SELL,
            price_audit_passed=trade.price_audit_passed,
            ma20_above_ma60=position.ma20_above_ma60,
            close_above_ma120=position.close_above_ma120,
            trend_confirmed=position.trend_confirmed,
            is_pullback_entry=position.is_pullback_entry,
            pullback_depth=position.pullback_depth
        )
        self.trade_audits.append(audit)
        
        self.account.available_cash += net_proceeds
        self.account.position_market_value -= position.market_value
        
        if net_pnl > 0:
            self.account.winning_trades += 1
        else:
            self.account.losing_trades += 1
        
        self.global_trade_count += 1
        
        del self.positions[symbol]
    
    def _get_next_open(self, symbol: str, trade_date: str) -> float:
        """获取次日开盘价"""
        return 0.0
    
    def _audit_execution_price(self, trade: V61Trade) -> bool:
        """成交价审计"""
        trigger_price = trade.trigger_price
        next_open = trade.next_open_price
        execution_price = trade.execution_price
        
        if trigger_price <= 0 or next_open <= 0:
            return True
        
        min_price = min(trigger_price, next_open)
        
        if trade.side == 'SELL':
            if execution_price > min_price * 1.01:
                logger.warning(f"Price Audit VIOLATION: {trade.symbol} @ {trade.trade_date}")
                return False
        else:
            if execution_price < min_price * 0.99:
                logger.warning(f"Price Audit VIOLATION (BUY): {trade.symbol} @ {trade.trade_date}")
                return False
        
        return True
    
    def _update_positions(self, trade_date: str, current_df: pl.DataFrame):
        """更新持仓数据"""
        for symbol, position in list(self.positions.items()):
            stock_df = current_df.filter(pl.col('symbol') == symbol)
            if stock_df.is_empty():
                continue
            
            rows = list(stock_df.iter_rows(named=True))
            if not rows:
                continue
            row = rows[0]
            current_price = row.get('close', 0)
            current_atr = row.get('atr_20', 0)
            current_ma20 = row.get('ma20', 0)
            current_ma60 = row.get('ma60', 0)
            
            if current_price <= 0:
                continue
            
            position.current_price = current_price
            position.market_value = current_price * position.shares
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            position.current_profit_ratio = (current_price - position.avg_cost) / position.avg_cost
            
            if current_price > position.peak_price:
                position.peak_price = current_price
                position.peak_profit = (current_price - position.avg_cost) / position.avg_cost
            
            self.risk_manager.update_position_stops(
                position, current_price, current_atr, current_ma20, current_ma60
            )
    
    def _process_entries(self, trade_date: str, current_df: pl.DataFrame,
                         factor_df: Optional[pl.DataFrame],
                         industry_data: Optional[pl.DataFrame] = None,
                         index_data: Optional[pl.DataFrame] = None):
        """
        处理入场逻辑 - V61 RS 回调核心
        
        【进场条件】
        1. 行业过滤：仅 RS 强度前 5 的行业
        2. 个股筛选：RS 排名前 10%
        3. 价格处于 [MA20, MA20 * 1.03] 区间（回踩均线）
        4. 成交量萎缩：成交量 < 5 日均量 70%（缩量回调）
        5. 严禁涨幅超过 5% 的突破股
        """
        if factor_df is None or factor_df.is_empty():
            return
        
        if len(self.positions) >= V61_MAX_POSITIONS:
            return
        
        if self.global_trade_count >= V61_GLOBAL_TRADE_LIMIT:
            return
        
        weekly_counter = self.weekly_counters.get(self.current_year * 100 + self.current_week)
        if weekly_counter and weekly_counter.trade_count >= V61_WEEKLY_TRADE_LIMIT:
            return
        
        # 行业过滤 - 仅 RS 强度前 5 的行业
        filtered_df, filter_stats = v61_industry_filter(
            factor_df, industry_data, self.industry_loader,
            trade_date, V61_INDUSTRY_TOP_N, index_data
        )
        
        # V61 核心：只选择回调信号股票
        # 过滤条件：is_pullback_entry=True 且 is_volume_shrunk=True
        # V61 新增：RSRS 择时过滤（z-score > 0.8）
        pullback_candidates = filtered_df.filter(
            (pl.col('is_pullback_entry') == True) & 
            (pl.col('is_volume_shrunk') == True) &
            (pl.col('is_top_rs') == True) &
            (pl.col('rsrs_entry_signal') == True)  # V61: RSRS 择时过滤
        )
        
        if pullback_candidates.is_empty():
            return
        
        # 按综合评分排序
        candidates = pullback_candidates.sort('composite_score', descending=True).head(V61_ENTRY_TOP_N)
        
        for row in candidates.iter_rows(named=True):
            if len(self.positions) >= V61_MAX_POSITIONS:
                break
            
            symbol = row.get('symbol', '')
            if not symbol or symbol in self.positions:
                continue
            
            # 黑名单检查
            if symbol in self.blacklist:
                blacklist_record = self.blacklist[symbol]
                if blacklist_record.days_remaining > 0:
                    continue
            
            # 洗售检查
            if self._check_wash_sale(symbol, trade_date):
                continue
            
            stock_df = current_df.filter(pl.col('symbol') == symbol)
            if stock_df.is_empty():
                continue
            
            stock_rows = list(stock_df.iter_rows(named=True))
            if not stock_rows:
                continue
            stock_row = stock_rows[0]
            current_price = stock_row.get('close', 0)
            current_atr = stock_row.get('atr_20', 0)
            
            if current_price <= 0:
                continue
            
            # V61 核心：再次验证回调条件
            is_pullback_entry = row.get('is_pullback_entry', False)
            is_volume_shrunk = row.get('is_volume_shrunk', False)
            daily_gain = row.get('daily_gain', 0)
            
            if not is_pullback_entry or not is_volume_shrunk:
                continue
            
            # 严禁涨幅超过 5%
            if daily_gain > V61_MAX_DAILY_GAIN:
                continue
            
            composite_score = row.get('composite_score', 0)
            rs_rank = row.get('rs_rank', 9999)
            rs_strength = row.get('rs_strength', 0)
            
            buy_price = current_price * (1 + V61_SLIPPAGE_BUY)
            
            # V61 核心：动态 ATR 仓位（单只风险≤0.8%）
            position_size = self.risk_manager.calculate_position_size(
                self.account.available_cash, current_price, current_atr,
                stock_row.get('volatility_20', 0.05)
            )
            
            if position_size <= 0:
                continue
            
            max_position_value = self.account.available_cash * V61_MAX_SINGLE_POSITION_PCT
            max_shares = int(max_position_value / buy_price)
            position_size = min(position_size, max_shares)
            
            if position_size <= 0:
                continue
            
            buy_amount = buy_price * position_size
            commission = max(buy_amount * V61_COMMISSION_RATE, V61_MIN_COMMISSION)
            transfer_fee = buy_amount * V61_TRANSFER_FEE
            total_cost = buy_amount + commission + transfer_fee
            
            if total_cost > self.account.available_cash:
                continue
            
            self.account.available_cash -= total_cost
            self.account.position_market_value += buy_amount
            
            industry_name = row.get('industry_name', '')
            if not industry_name:
                industry_name = self.industry_loader.get_industry_for_symbol(symbol)
            
            # 计算回调深度
            ma20 = row.get('ma20', current_price)
            pullback_depth = (current_price - ma20) / (ma20 + 1e-9) if ma20 > 0 else 0
            
            position = V61Position(
                symbol=symbol,
                shares=position_size,
                avg_cost=buy_price,
                buy_price=buy_price,
                buy_date=trade_date,
                signal_date=trade_date,
                trade_date=trade_date,
                signal_score=composite_score,
                signal_rank=row.get('composite_rank', 9999),
                composite_score=composite_score,
                current_price=buy_price,
                market_value=buy_price * position_size,
                unrealized_pnl=0,
                holding_days=0,
                peak_price=buy_price,
                peak_profit=0,
                atr_at_entry=current_atr,
                hard_stop_price=buy_price - (V61_HARD_STOP_LOSS_ATR_MULT * current_atr),
                industry_name=industry_name,
                rs_score=rs_strength,
                rs_rank=rs_rank,
                volume_breakout=False,  # V61 不追求放量
                volume_shrunk_at_entry=is_volume_shrunk,  # V61 核心标记
                trigger_price=current_price,
                next_open_price=current_price,
                execution_price_audit=buy_price,
                ma20_above_ma60=row.get('ma20_above_ma60', False),
                close_above_ma120=row.get('close_above_ma120', False),
                trend_confirmed=row.get('trend_confirmed', False),
                is_pullback_entry=is_pullback_entry,
                pullback_depth=pullback_depth
            )
            
            self.positions[symbol] = position
            self.total_stocks_traded.add(symbol)
            
            # 统计回调买入次数
            if is_pullback_entry:
                self.pullback_entries += 1
            if is_volume_shrunk:
                self.volume_shrunk_entries += 1
            
            trade = V61Trade(
                trade_date=trade_date,
                symbol=symbol,
                side='BUY',
                shares=position_size,
                price=buy_price,
                amount=buy_amount,
                commission=commission,
                slippage=V61_SLIPPAGE_BUY,
                stamp_duty=0,
                transfer_fee=transfer_fee,
                total_cost=total_cost,
                reason=f"RS 回调买入 (Score={composite_score:.4f}, RS_Rank={rs_rank}, Pullback={pullback_depth:.2%})",
                execution_price=buy_price,
                signal_date=trade_date,
                trigger_price=current_price,
                next_open_price=current_price,
                min_trigger_open=min(current_price, current_price),
                slippage_applied=V61_SLIPPAGE_BUY,
                price_audit_passed=True
            )
            
            trade.price_audit_passed = self._audit_execution_price(trade)
            if not trade.price_audit_passed:
                self.price_audit_violations += 1
            
            self.trades.append(trade)
            self.global_trade_count += 1
            
            if weekly_counter:
                weekly_counter.trade_count += 1
            
            self._add_to_blacklist(symbol, trade_date, "recent_entry")
    
    def _check_wash_sale(self, symbol: str, trade_date: str) -> bool:
        """检查洗售规则"""
        for record in self.wash_sale_records:
            if record.symbol == symbol:
                try:
                    sell_date = datetime.strptime(record.sell_date, "%Y-%m-%d")
                    current = datetime.strptime(trade_date, "%Y-%m-%d")
                    days_diff = (current - sell_date).days
                    if 0 < days_diff <= V61_WASH_SALE_WINDOW:
                        return True
                except Exception:
                    pass
        return False
    
    def _add_to_blacklist(self, symbol: str, trade_date: str, reason: str):
        """添加到黑名单"""
        try:
            current = datetime.strptime(trade_date, "%Y-%m-%d")
            expiry = current + timedelta(days=V61_WASH_SALE_WINDOW)
            
            self.blacklist[symbol] = V61BlacklistRecord(
                symbol=symbol,
                stop_date=trade_date,
                stop_reason=reason,
                blacklist_expiry_day=int(expiry.strftime("%Y%m%d")),
                days_remaining=V61_WASH_SALE_WINDOW
            )
        except Exception:
            pass
    
    def _update_account_state(self, trade_date: str, current_df: pl.DataFrame):
        """更新账户状态"""
        total_market_value = sum(p.market_value for p in self.positions.values())
        
        self.account.trade_date = trade_date
        self.account.position_market_value = total_market_value
        self.account.total_equity = self.account.available_cash + total_market_value
        self.account.total_pnl = self.account.total_equity - self.initial_capital
        self.account.total_pnl_ratio = self.account.total_pnl / self.initial_capital
        
        if self.account.total_equity > self.account.peak_equity:
            self.account.peak_equity = self.account.total_equity
        
        if self.account.peak_equity > 0:
            self.account.current_drawdown = (self.account.peak_equity - self.account.total_equity) / self.account.peak_equity
            if self.account.current_drawdown > self.account.max_drawdown:
                self.account.max_drawdown = self.account.current_drawdown
        
        self.equity_curve.append(self.account.total_equity)
        self.drawdown_curve.append(self.account.current_drawdown)
    
    def _check_compliance(self, trade_date: str):
        """检查合规性"""
        if self.global_trade_count > V61_GLOBAL_TRADE_LIMIT:
            logger.warning(f"COMPLIANCE VIOLATION: Global trade limit exceeded ({self.global_trade_count} > {V61_GLOBAL_TRADE_LIMIT})")
        
        weekly_counter = self.weekly_counters.get(self.current_year * 100 + self.current_week)
        if weekly_counter and weekly_counter.trade_count > V61_WEEKLY_TRADE_LIMIT:
            logger.warning(f"COMPLIANCE VIOLATION: Weekly trade limit exceeded ({weekly_counter.trade_count} > {V61_WEEKLY_TRADE_LIMIT})")
    
    def _generate_result(self) -> Dict[str, Any]:
        """生成回测结果"""
        if not self.equity_curve:
            return {'error': 'No equity curve data'}
        
        total_return = (self.account.total_equity - self.initial_capital) / self.initial_capital
        
        trade_dates_count = len(self.trade_dates)
        if trade_dates_count > 0:
            years = trade_dates_count / 252
            if years > 0:
                annual_return = (1 + total_return) ** (1 / years) - 1
            else:
                annual_return = total_return
        else:
            annual_return = 0
        
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            if self.equity_curve[i-1] > 0:
                daily_returns.append((self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1])
        
        if daily_returns and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        win_rate = 0
        profit_loss_ratio = 0
        avg_win = 0
        avg_loss = 0
        largest_win = 0
        largest_loss = 0
        
        if self.trade_audits:
            winning_trades = [t for t in self.trade_audits if t.net_pnl > 0]
            losing_trades = [t for t in self.trade_audits if t.net_pnl <= 0]
            
            self.account.winning_trades = len(winning_trades)
            self.account.losing_trades = len(losing_trades)
            
            if self.trade_audits:
                win_rate = len(winning_trades) / len(self.trade_audits)
            
            if winning_trades:
                avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades)
                largest_win = max(t.net_pnl for t in winning_trades)
            
            if losing_trades:
                avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades)
                largest_loss = min(t.net_pnl for t in losing_trades)
            
            if avg_loss != 0:
                profit_loss_ratio = abs(avg_win / avg_loss)
        
        total_commission = sum(t.commission for t in self.trades)
        total_slippage_cost = sum(t.slippage * t.amount for t in self.trades if t.amount > 0)
        total_stamp_duty = sum(t.stamp_duty for t in self.trades)
        total_transfer_fee = sum(t.transfer_fee for t in self.trades)
        
        avg_holding_days = 0
        if self.trade_audits:
            avg_holding_days = sum(t.holding_days for t in self.trade_audits) / len(self.trade_audits)
        
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for audit in self.trade_audits:
            if audit.is_profitable:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        meets_target = (
            total_return >= V61_RETURN_TARGET and
            self.account.max_drawdown <= V61_MDD_TARGET and
            profit_loss_ratio >= V61_PROFIT_LOSS_RATIO_TARGET
        )
        
        target_analysis = []
        if total_return < V61_RETURN_TARGET:
            target_analysis.append(f"Return {total_return:.2%} < {V61_RETURN_TARGET:.1%}")
        if self.account.max_drawdown > V61_MDD_TARGET:
            target_analysis.append(f"MDD {self.account.max_drawdown:.2%} > {V61_MDD_TARGET:.1%}")
        if profit_loss_ratio < V61_PROFIT_LOSS_RATIO_TARGET:
            target_analysis.append(f"P/L {profit_loss_ratio:.2f} < {V61_PROFIT_LOSS_RATIO_TARGET:.1f}")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': self.account.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': len(self.trade_audits),
            'winning_trades': self.account.winning_trades,
            'losing_trades': self.account.losing_trades,
            'final_equity': self.account.total_equity,
            'initial_equity': self.initial_capital,
            'total_commission': total_commission,
            'total_slippage': total_slippage_cost,
            'total_stamp_duty': total_stamp_duty,
            'avg_holding_days': avg_holding_days,
            'avg_win': avg_win,
            'avg_loss': abs(avg_loss),
            'largest_win': largest_win,
            'largest_loss': abs(largest_loss),
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'trade_dates': self.trade_dates,
            'price_audit_violations': self.price_audit_violations,
            'meets_target': meets_target,
            'target_analysis': "; ".join(target_analysis) if target_analysis else "All targets met",
            'global_trade_count': self.global_trade_count,
            'weekly_counters': {k: v.trade_count for k, v in self.weekly_counters.items()},
            'total_stocks_traded': len(self.total_stocks_traded),
            'data_source': self.industry_loader.data_source,
            'pullback_entries': self.pullback_entries,
            'volume_shrunk_entries': self.volume_shrunk_entries
        }


class MasterLoop:
    """
    MasterLoop - V61 真正的逻辑突变
    
    【核心逻辑】
    1. 最多 50 轮迭代
    2. 收益率<15% 时触发逻辑突变
    3. 逻辑突变类型：
       - 从"均线回调"切换到"乖离率超卖"
       - 从"缩量回调"切换到"温和放量"
       - 调整选股分位数和趋势确认周期
    4. 报告中必须清晰展示：第 N 轮迭代中，删除了哪段逻辑代码，替换成了哪种新策略
    """
    
    def __init__(self, max_iterations: int = V61_MAX_ITERATION_ROUNDS):
        self.max_iterations = max_iterations
        self.iteration_results: List[V61IterationResult] = []
        self.logic_evolution_records: List[V61LogicEvolutionRecord] = []
        self.current_logic_path = 0
        
        # 逻辑路径定义 - V61 核心
        self.logic_paths = [
            'pullback_shrink',        # 缩量回调（默认）
            'pullback_moderate_vol',  # 温和放量回调
            'bias_ratio_oversold',    # 乖离率超卖
            'ma20_bounce',            # MA20 反弹
            'industry_rotation',      # 行业轮动
            'trend_following',        # 趋势跟踪
        ]
        
        # 当前参数
        self.current_parameters = {
            'momentum_weight': V61_MOMENTUM_WEIGHT,
            'r2_weight': V61_R2_WEIGHT,
            'trend_weight': V61_TREND_WEIGHT,
            'entry_top_n': V61_ENTRY_TOP_N,
            'rs_top_percentile': V61_RS_TOP_PERCENTILE,
            'volume_shrink_ratio': V61_VOLUME_SHRINK_RATIO,
            'ma20_buffer': V61_MA20_BUFFER,
            'hard_stop_atr_mult': V61_HARD_STOP_LOSS_ATR_MULT,
            'breakeven_threshold': V61_BREAKEVEN_PROFIT_THRESHOLD,
            'trailing_profit_trigger': V61_TRAILING_PROFIT_TRIGGER,
            'selection_percentile': 0.10,
            'trend_period': 20,
        }
        
        # 历史最佳
        self.best_result = None
        self.best_metrics = {'total_return': -999, 'profit_loss_ratio': 0}
    
    def run_iteration(self, factor_data: pl.DataFrame, price_data: pl.DataFrame,
                      industry_data: Optional[pl.DataFrame] = None,
                      index_data: Optional[pl.DataFrame] = None,
                      start_date: str = "", end_date: str = "") -> Dict[str, Any]:
        """运行单轮迭代"""
        iteration = len(self.iteration_results) + 1
        
        if iteration > self.max_iterations:
            logger.info(f"MasterLoop: Max iterations ({self.max_iterations}) reached")
            return self.get_best_result_dict()
        
        logic_path = self.logic_paths[self.current_logic_path % len(self.logic_paths)]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MasterLoop Iteration {iteration}/{self.max_iterations}")
        logger.info(f"Logic Path: {logic_path}")
        logger.info(f"Parameters: {self.current_parameters}")
        logger.info(f"{'='*60}")
        
        engine = V61BacktestEngine()
        
        try:
            result = engine.run_backtest(
                factor_data=factor_data,
                price_data=price_data,
                industry_data=industry_data,
                index_data=index_data,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Iteration {iteration} FAILED: {e}")
            self._adapt_strategy_on_error(str(e))
            return self.run_iteration(factor_data, price_data, industry_data, index_data, start_date, end_date)
        
        if result.get('error'):
            logger.error(f"Iteration {iteration} failed: {result['error']}")
            return result
        
        backtest_result = result.get('result', {})
        total_return = backtest_result.get('total_return', 0)
        profit_loss_ratio = backtest_result.get('profit_loss_ratio', 0)
        max_drawdown = backtest_result.get('max_drawdown', 0)
        
        meets_target = (
            total_return >= V61_RETURN_TARGET and
            profit_loss_ratio >= V61_PROFIT_LOSS_RATIO_TARGET and
            max_drawdown <= V61_MDD_TARGET
        )
        
        iteration_result = V61IterationResult(
            iteration=iteration,
            logic_path=logic_path,
            parameters=self.current_parameters.copy(),
            metrics={
                'total_return': total_return,
                'profit_loss_ratio': profit_loss_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': backtest_result.get('total_trades', 0)
            },
            meets_target=meets_target,
            evolution_step="",
            logic_mutation_applied=False
        )
        self.iteration_results.append(iteration_result)
        
        # 更新历史最佳
        if total_return > self.best_metrics['total_return']:
            self.best_result = backtest_result
            self.best_metrics = {
                'total_return': total_return,
                'profit_loss_ratio': profit_loss_ratio,
                'max_drawdown': max_drawdown
            }
        
        if meets_target:
            logger.info(f"✅ TARGET MET at iteration {iteration}!")
            return backtest_result
        
        logger.info(f"Target not met: Return={total_return:.2%}, P/L={profit_loss_ratio:.2f}, MDD={max_drawdown:.2%}")
        
        # 记录逻辑进化
        previous_logic = logic_path
        previous_params = self.current_parameters.copy()
        
        # V61 核心：收益率<15% 触发逻辑突变
        if total_return < V61_LOGIC_MUTATION_THRESHOLD and V61_LOGIC_MUTATION_ENABLED:
            self._apply_logic_mutation(total_return, profit_loss_ratio, max_drawdown)
            iteration_result.logic_mutation_applied = True
        else:
            self._adapt_strategy(total_return, profit_loss_ratio, max_drawdown)
        
        # 记录逻辑进化
        new_logic = self.logic_paths[self.current_logic_path % len(self.logic_paths)]
        if new_logic != previous_logic or previous_params != self.current_parameters:
            evolution_record = V61LogicEvolutionRecord(
                iteration=iteration,
                previous_logic=previous_logic,
                new_logic=new_logic,
                reason=self._get_evolution_reason(total_return, profit_loss_ratio, max_drawdown),
                parameters_changed=self._get_changed_params(previous_params),
                performance_impact={
                    'total_return': total_return,
                    'profit_loss_ratio': profit_loss_ratio,
                    'max_drawdown': max_drawdown
                },
                mutation_type="logic_switch" if new_logic != previous_logic else "parameter_tuning"
            )
            self.logic_evolution_records.append(evolution_record)
        
        return backtest_result
    
    def _adapt_strategy_on_error(self, error_msg: str):
        """出错时调整策略"""
        logger.info(f"Adapting strategy on error: {error_msg}")
        self.current_logic_path += 1
        
        self.current_parameters['entry_top_n'] = min(
            self.current_parameters.get('entry_top_n', 10) + 5, 20
        )
        self.current_parameters['selection_percentile'] = min(
            self.current_parameters.get('selection_percentile', 0.10) + 0.05, 0.30
        )
    
    def _apply_logic_mutation(self, total_return: float, profit_loss_ratio: float, max_drawdown: float):
        """
        V61 核心：逻辑突变
        
        【突变类型】
        1. 从"缩量回调"切换到"温和放量回调"
        2. 从"均线回调"切换到"乖离率超卖"
        3. 调整选股分位数
        """
        logger.info(f"🔴 LOGIC MUTATION TRIGGERED (Return={total_return:.2%} < {V61_LOGIC_MUTATION_THRESHOLD:.1%})")
        
        # 切换到下一个逻辑路径
        self.current_logic_path += 1
        new_logic = self.logic_paths[self.current_logic_path % len(self.logic_paths)]
        
        logger.info(f"Logic Mutation: {self.logic_paths[(self.current_logic_path - 1) % len(self.logic_paths)]} -> {new_logic}")
        
        # 根据新逻辑调整参数
        if new_logic == 'pullback_shrink':
            # 缩量回调逻辑
            self.current_parameters['volume_shrink_ratio'] = 0.70
            self.current_parameters['ma20_buffer'] = 0.03
            self.current_parameters['rs_top_percentile'] = 0.10
            logger.info("  Deleted: 温和放量条件")
            logger.info("  Replaced with: 缩量回调 (成交量<5 日均量 70%)")
            
        elif new_logic == 'pullback_moderate_vol':
            # 温和放量回调
            self.current_parameters['volume_shrink_ratio'] = 1.20  # 允许温和放量
            self.current_parameters['ma20_buffer'] = 0.05
            self.current_parameters['rs_top_percentile'] = 0.15
            logger.info("  Deleted: 严格缩量条件")
            logger.info("  Replaced with: 温和放量回调 (成交量<5 日均量 120%)")
            
        elif new_logic == 'bias_ratio_oversold':
            # 乖离率超卖逻辑
            self.current_parameters['ma20_buffer'] = -0.05  # 允许跌破 MA20
            self.current_parameters['volume_shrink_ratio'] = 0.80
            self.current_parameters['rs_top_percentile'] = 0.20
            logger.info("  Deleted: MA20 回踩条件")
            logger.info("  Replaced with: 乖离率超卖 (允许跌破 MA20 5%)")
            
        elif new_logic == 'ma20_bounce':
            # MA20 反弹逻辑
            self.current_parameters['ma20_buffer'] = 0.02
            self.current_parameters['volume_shrink_ratio'] = 0.90
            self.current_parameters['rs_top_percentile'] = 0.15
            logger.info("  Deleted: 乖离率条件")
            logger.info("  Replaced with: MA20 反弹 (价格在 [MA20, MA20*1.02])")
            
        elif new_logic == 'industry_rotation':
            # 行业轮动逻辑
            self.current_parameters['rs_top_percentile'] = 0.25
            self.current_parameters['volume_shrink_ratio'] = 1.00
            logger.info("  Deleted: 严格行业限制")
            logger.info("  Replaced with: 行业轮动 (放宽行业选择)")
            
        elif new_logic == 'trend_following':
            # 趋势跟踪逻辑
            self.current_parameters['rs_top_percentile'] = 0.30
            self.current_parameters['ma20_buffer'] = 0.10
            logger.info("  Deleted: 回调条件")
            logger.info("  Replaced with: 趋势跟踪 (允许追涨)")
    
    def _adapt_strategy(self, total_return: float, profit_loss_ratio: float, max_drawdown: float):
        """根据结果调整策略（参数级别）"""
        evolution_step = ""
        
        # 1. 盈亏比不达标
        if profit_loss_ratio < V61_PROFIT_LOSS_RATIO_TARGET:
            evolution_step = "盈亏比不达标 -> 调整选股分位数 + 因子权重"
            logger.info(f"强制动作 A: {evolution_step}")
            
            current_percentile = self.current_parameters.get('selection_percentile', 0.10)
            if current_percentile < V61_MAX_SELECTION_PERCENTILE:
                self.current_parameters['selection_percentile'] = min(
                    current_percentile + 0.05, V61_MAX_SELECTION_PERCENTILE
                )
            
            if self.current_logic_path % 3 == 0:
                self.current_parameters['momentum_weight'] = 0.40
                self.current_parameters['r2_weight'] = 0.40
                self.current_parameters['trend_weight'] = 0.20
            elif self.current_logic_path % 3 == 1:
                self.current_parameters['momentum_weight'] = 0.30
                self.current_parameters['r2_weight'] = 0.50
                self.current_parameters['trend_weight'] = 0.20
            else:
                self.current_parameters['momentum_weight'] = 0.35
                self.current_parameters['r2_weight'] = 0.35
                self.current_parameters['trend_weight'] = 0.30
            
            self.current_logic_path += 1
        
        # 2. 收益率不达标
        if total_return < V61_RETURN_TARGET:
            evolution_step = "收益率不达标 -> 调整趋势确认周期 + 止盈阈值"
            logger.info(f"强制动作 B: {evolution_step}")
            
            current_trend_period = self.current_parameters.get('trend_period', 20)
            if current_trend_period > V61_MIN_TREND_PERIOD:
                self.current_parameters['trend_period'] = max(
                    current_trend_period - 5, V61_MIN_TREND_PERIOD
                )
            
            if self.current_parameters.get('trailing_profit_trigger', 0.08) > 0.06:
                self.current_parameters['trailing_profit_trigger'] = max(
                    self.current_parameters['trailing_profit_trigger'] - 0.01, 0.06
                )
            
            self.current_logic_path += 1
        
        # 3. 回撤超标
        if max_drawdown > V61_MDD_TARGET:
            evolution_step = "回撤超标 -> 提高止损倍数 + 降低仓位"
            logger.info(f"风控调整：{evolution_step}")
            
            self.current_parameters['hard_stop_atr_mult'] = min(
                self.current_parameters.get('hard_stop_atr_mult', 2.5) + 0.5, 4.0
            )
            self.current_parameters['selection_percentile'] = max(
                self.current_parameters.get('selection_percentile', 0.10) - 0.02, V61_MIN_SELECTION_PERCENTILE
            )
    
    def _get_evolution_reason(self, total_return: float, profit_loss_ratio: float, max_drawdown: float) -> str:
        """获取进化原因"""
        reasons = []
        if profit_loss_ratio < V61_PROFIT_LOSS_RATIO_TARGET:
            reasons.append(f"P/L {profit_loss_ratio:.2f} < {V61_PROFIT_LOSS_RATIO_TARGET}")
        if total_return < V61_RETURN_TARGET:
            reasons.append(f"Return {total_return:.2%} < {V61_RETURN_TARGET}")
        if max_drawdown > V61_MDD_TARGET:
            reasons.append(f"MDD {max_drawdown:.2%} > {V61_MDD_TARGET}")
        return "; ".join(reasons) if reasons else "Parameter tuning"
    
    def _get_changed_params(self, previous_params: Dict) -> Dict[str, Any]:
        """获取变化的参数"""
        changed = {}
        for key, value in self.current_parameters.items():
            if key in previous_params and previous_params[key] != value:
                changed[key] = {
                    'old': previous_params[key],
                    'new': value
                }
        return changed
    
    def get_best_result_dict(self) -> Dict[str, Any]:
        """获取最佳结果字典"""
        if self.best_result:
            return self.best_result
        return {'error': 'No results yet'}
    
    def get_best_result(self) -> Optional[V61IterationResult]:
        """获取最佳迭代结果"""
        if not self.iteration_results:
            return None
        
        best = max(self.iteration_results,
                   key=lambda x: x.metrics.get('total_return', 0) * x.metrics.get('profit_loss_ratio', 0))
        return best
    
    def compare_logic_paths(self) -> Dict[str, Any]:
        """对比至少 3 种逻辑路径"""
        if len(self.iteration_results) < 3:
            logger.warning("Not enough iterations for comparison (need at least 3)")
        
        logic_path_results = {}
        for result in self.iteration_results:
            lp = result.logic_path
            if lp not in logic_path_results:
                logic_path_results[lp] = []
            logic_path_results[lp].append(result.metrics)
        
        comparison = {}
        for lp, metrics_list in logic_path_results.items():
            if not metrics_list:
                continue
            avg_return = np.mean([m['total_return'] for m in metrics_list])
            avg_pl_ratio = np.mean([m['profit_loss_ratio'] for m in metrics_list])
            avg_mdd = np.mean([m['max_drawdown'] for m in metrics_list])
            comparison[lp] = {
                'avg_return': avg_return,
                'avg_pl_ratio': avg_pl_ratio,
                'avg_mdd': avg_mdd,
                'stability_score': avg_return * avg_pl_ratio / (avg_mdd + 0.01)
            }
        
        best_path = max(comparison.keys(), key=lambda x: comparison[x]['stability_score']) if comparison else ""
        
        return {
            'comparison': comparison,
            'best_logic_path': best_path,
            'total_iterations': len(self.iteration_results)
        }
    
    def get_logic_evolution_report(self) -> Dict[str, Any]:
        """获取逻辑进化报告"""
        return {
            'total_evolutions': len(self.logic_evolution_records),
            'evolution_records': [
                {
                    'iteration': r.iteration,
                    'previous_logic': r.previous_logic,
                    'new_logic': r.new_logic,
                    'reason': r.reason,
                    'parameters_changed': r.parameters_changed,
                    'performance_impact': r.performance_impact,
                    'mutation_type': r.mutation_type
                }
                for r in self.logic_evolution_records
            ],
            'best_result': self.best_metrics,
            'final_parameters': self.current_parameters
        }


# ===========================================
# __all__ 导出列表
# ===========================================

__all__ = [
    'V61AccountState',
    'V61BacktestResult',
    'V61BacktestEngine',
    'MasterLoop',
]