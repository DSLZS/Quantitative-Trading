"""
V77 Backtest Engine - 量价交互特征挖掘与 Rank IC 爆发回测引擎

【核心功能】
1. 基于 V77 核心逻辑执行回测
2. 严格的仓位管理和止损止盈
3. 行业分散性控制（单行业不超过 20%）
4. Rank IC 实时监控与月度统计
5. 评分分布直方图生成
6. 数据完整性检查与重试机制
7. Rank IC 与收益率散点图分析

【V77 新增特性】
- MAD 去极值化处理
- Money Flow Persistence（资金流持续性自相关系数）
- RS Breakout Quality（突破质量）
- 时序动量审计

作者：量化系统
版本：V77.0
日期：2026-03-26
"""

import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger

from src.db_manager import DatabaseManager
from src.core.v77_logic import (
    V77DataManager,
    V77AlphaCenter,
    V77RankICCalculator,
    V77Signal,
    V77Position,
    V77Trade,
    V77MonthlyICStats,
    V77_INITIAL_CAPITAL,
    V77_MAX_POSITIONS,
    V77_MAX_SINGLE_POSITION_PCT,
    V77_MAX_SECTOR_WEIGHT,
    V77_COMMISSION_RATE,
    V77_MIN_COMMISSION,
    V77_SLIPPAGE_BUY,
    V77_SLIPPAGE_SELL,
    V77_STAMP_DUTY,
    V77_TRANSFER_FEE,
    V77_STOP_LOSS_RATIO,
    V77_PROFIT_TARGET_RATIO,
    V77_TRAILING_STOP_RATIO,
    V77_MIN_FUND_FLOW_ROWS,
    V77_RANK_IC_TARGET,
    V77_MAX_DRAWDOWN_TARGET,
    V77_CALMAR_RATIO_TARGET,
    generate_score_histogram_data,
    generate_monthly_ic_ascii_chart,
)


# ===========================================
# V77 回测结果数据类
# ===========================================

@dataclass
class V77BacktestResult:
    """V77 回测结果"""
    # 基本信息
    start_date: str
    end_date: str
    initial_capital: float
    
    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易指标
    total_trades: int = 0
    win_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 持仓指标
    avg_holding_days: float = 0.0
    max_positions: int = 0
    
    # Rank IC 指标
    mean_rank_ic: float = 0.0
    monthly_rank_ic: float = 0.0
    rank_ic_pass: bool = False
    negative_months: int = 0
    monthly_ic_stats: List[V77MonthlyICStats] = field(default_factory=list)
    
    # 评分分布
    score_mean: float = 0.0
    score_std: float = 0.0
    score_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # V77 因子统计
    avg_flow_persistence: float = 0.0
    avg_rs_breakout: float = 0.0
    avg_temporal_momentum: float = 0.0
    temporal_penalty_count: int = 0
    
    # 行业拥挤度统计
    crowded_sector_trades: int = 0
    sector_penalty_applied: int = 0
    
    # 详细数据
    daily_values: Optional[pl.DataFrame] = None
    trades: List[V77Trade] = field(default_factory=list)
    
    # 行业分布统计
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Rank IC 散点图数据
    rank_ic_scatter_data: List[Dict[str, float]] = field(default_factory=list)


# ===========================================
# V77 回测引擎
# ===========================================

class V77BacktestEngine:
    """
    V77 回测引擎 - 量价交互特征挖掘
    
    【核心特性】
    1. 初始资金严格锁定 100,000 元
    2. 费率、止盈止损规则严禁修改
    3. 全市场股票连续评分
    4. 行业分散性控制（单行业不超过 20%）
    5. Rank IC 实时监控
    6. 月度 Rank IC 统计输出
    7. 数据完整性检查与重试机制
    8. Rank IC 与收益率散点图分析
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None,
                 config: Dict[str, Any] = None):
        """
        初始化 V77 回测引擎
        
        Args:
            db: 数据库管理器
            config: 配置字典
        """
        self.db = db or DatabaseManager()
        self.config = config or {}
        
        # 初始资金（严禁修改）
        self.initial_capital = self.config.get('initial_capital', V77_INITIAL_CAPITAL)
        
        # 仓位管理
        self.max_positions = self.config.get('max_positions', V77_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get(
            'max_single_position_pct', V77_MAX_SINGLE_POSITION_PCT
        )
        self.max_sector_weight = self.config.get(
            'max_sector_weight', V77_MAX_SECTOR_WEIGHT
        )
        
        # 费率（严禁修改）
        self.commission_rate = V77_COMMISSION_RATE
        self.min_commission = V77_MIN_COMMISSION
        self.slippage_buy = V77_SLIPPAGE_BUY
        self.slippage_sell = V77_SLIPPAGE_SELL
        self.stamp_duty = V77_STAMP_DUTY
        self.transfer_fee = V77_TRANSFER_FEE
        
        # 止损止盈（严禁修改）
        self.stop_loss_ratio = V77_STOP_LOSS_RATIO
        self.profit_target_ratio = V77_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V77_TRAILING_STOP_RATIO
        
        # 初始化组件
        self.data_manager = V77DataManager(db=self.db, config=self.config)
        self.alpha_center = V77AlphaCenter(config=self.config)
        self.rank_ic_calculator = V77RankICCalculator(db=self.db, config=self.config)
        
        # 回测状态
        self.positions: Dict[str, V77Position] = {}
        self.trades: List[V77Trade] = []
        self.daily_values: List[Dict[str, Any]] = []
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        # V77 因子统计
        self.flow_persistence_values: List[float] = []
        self.rs_breakout_values: List[float] = []
        self.temporal_momentum_values: List[float] = []
        self.temporal_penalty_count = 0
        
        # 行业拥挤度统计
        self.crowded_sector_trades = 0
        self.sector_penalty_applied = 0
        
        # 行业持仓统计
        self.sector_holdings: Dict[str, float] = {}
        
        # Rank IC 散点图数据
        self.rank_ic_scatter_data: List[Dict[str, float]] = []
        
        logger.info(f"V77BacktestEngine 初始化完成：初始资金={self.initial_capital:,.0f}")
    
    def _check_data_integrity(self) -> bool:
        """
        检查数据完整性
        
        【核心逻辑】
        1. 检查 2024 全年资金流数据
        2. 若数据不足，记录警告但继续（由用户决定是否补抓）
        3. 严禁报错跳过，必须明确报告
        """
        logger.info("=" * 60)
        logger.info("V77: 数据完整性检查")
        logger.info("=" * 60)
        
        is_valid, message = self.data_manager.check_2024_data_integrity()
        
        if not is_valid:
            logger.warning(f"V77: 数据完整性检查未通过 - {message}")
            logger.warning("V77: 建议先运行数据同步脚本补充数据")
        else:
            logger.info(f"V77: 数据完整性检查通过 - {message}")
        
        return is_valid
    
    def run_backtest(self, start_date: str, end_date: str) -> V77BacktestResult:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            V77BacktestResult: 回测结果
        """
        logger.info("=" * 60)
        logger.info("V77 回测开始：量价交互特征挖掘与 Rank IC 爆发计划")
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.0f}")
        logger.info(f"Rank IC 目标：>={V77_RANK_IC_TARGET}")
        logger.info(f"Calmar Ratio 目标：>={V77_CALMAR_RATIO_TARGET}")
        logger.info("=" * 60)
        
        try:
            # 0. 数据完整性检查
            logger.info("Step 0: 数据完整性检查...")
            self._check_data_integrity()
            
            # 1. 加载数据
            logger.info("Step 1: 加载数据...")
            stock_df = self.data_manager.load_stock_data(start_date, end_date)
            fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date)
            industry_df = self.data_manager.load_industry_data(start_date, end_date)
            index_df = self.data_manager.load_index_data(start_date, end_date)
            
            # 2. 计算信号
            logger.info("Step 2: 计算信号...")
            signals_df, status = self.alpha_center.compute_signals(
                stock_df, fund_flow_df, industry_df, index_df
            )
            
            # 3. 计算 Rank IC
            logger.info("Step 3: 计算 Rank IC...")
            self.rank_ic_calculator.calculate_ic_series(signals_df)
            ic_stats = self.rank_ic_calculator.get_ic_statistics()
            monthly_ic_stats = self.rank_ic_calculator.get_monthly_rank_ic_statistics()
            
            # 4. 获取评分分布
            logger.info("Step 4: 分析评分分布...")
            score_hist_data = generate_score_histogram_data(signals_df)
            
            # 5. 生成月度 IC 柱状图
            logger.info("Step 5: 生成月度 Rank IC 柱状图...")
            monthly_ic_chart = self.rank_ic_calculator.generate_monthly_ic_chart()
            logger.info("\n" + monthly_ic_chart)
            
            # 6. 执行回测交易
            logger.info("Step 6: 执行交易...")
            self._execute_trades(signals_df, start_date, end_date)
            
            # 7. 计算回测结果
            logger.info("Step 7: 计算回测指标...")
            result = self._compute_backtest_result(
                start_date, end_date,
                ic_stats, monthly_ic_stats,
                score_hist_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"V77 回测执行失败：{e}")
            logger.error(f"【错误分析】{traceback.format_exc()}")
            logger.error("【修复方案】检查数据完整性，确保所有必要数据已加载")
            raise
    
    def _execute_trades(self, signals_df: pl.DataFrame,
                        start_date: str, end_date: str) -> None:
        """执行交易逻辑（带行业分散性控制）"""
        unique_dates = sorted(signals_df['trade_date'].unique().to_list())
        
        for trade_date in unique_dates:
            if trade_date < start_date:
                continue
            
            # 1. 更新持仓价格
            self._update_positions(signals_df, trade_date)
            
            # 2. 检查止损止盈
            self._check_stop_loss_profit(trade_date)
            
            # 3. 生成买入信号
            signals = self.alpha_center.generate_signals(signals_df, trade_date)
            
            # 4. 执行买入（带行业分散性控制）
            self._execute_buy_with_sector_control(signals, trade_date)
            
            # 5. 记录每日净值
            self._record_daily_value(trade_date)
    
    def _update_positions(self, signals_df: pl.DataFrame, trade_date: str) -> None:
        """更新持仓价格"""
        for symbol, position in list(self.positions.items()):
            price_data = signals_df.filter(
                (pl.col('symbol') == symbol) & 
                (pl.col('trade_date') == trade_date)
            )
            
            if not price_data.is_empty():
                close_price = price_data['close'][0]
                position.current_price = close_price
                position.market_value = close_price * position.shares
                position.holding_days += 1
                
                # 更新峰值价格
                if close_price > position.peak_price:
                    position.peak_price = close_price
                    position.peak_profit = (close_price - position.avg_cost) / position.avg_cost
    
    def _check_stop_loss_profit(self, trade_date: str) -> None:
        """检查止损止盈条件"""
        for symbol, position in list(self.positions.items()):
            current_price = position.current_price
            avg_cost = position.avg_cost
            peak_price = position.peak_price
            
            return_pct = (current_price - avg_cost) / avg_cost
            
            # 止损检查
            if return_pct <= -self.stop_loss_ratio:
                self._execute_sell(symbol, position, trade_date, "止损")
                continue
            
            # 止盈检查
            if return_pct >= self.profit_target_ratio:
                self._execute_sell(symbol, position, trade_date, "止盈")
                continue
            
            # 移动止盈检查
            if peak_price > avg_cost * (1 + self.trailing_stop_ratio):
                trailing_stop_price = peak_price * (1 - self.trailing_stop_ratio)
                if current_price <= trailing_stop_price:
                    self._execute_sell(symbol, position, trade_date, "移动止盈")
    
    def _get_sector_exposure(self) -> Dict[str, float]:
        """获取当前行业敞口（市值占比）"""
        sector_values: Dict[str, float] = {}
        total_value = sum(p.market_value for p in self.positions.values())
        
        if total_value < 1:
            return sector_values
        
        for position in self.positions.values():
            industry = position.industry_name or 'UNKNOWN'
            if industry not in sector_values:
                sector_values[industry] = 0.0
            sector_values[industry] += position.market_value
        
        # 转换为占比
        for industry in sector_values:
            sector_values[industry] /= total_value
        
        return sector_values
    
    def _can_buy_sector(self, industry: str, target_value: float) -> bool:
        """
        检查是否可以买入某行业（行业分散性控制）
        
        【核心逻辑】
        1. 计算买入后该行业的市值占比
        2. 若超过 V77_MAX_SECTOR_WEIGHT (20%)，禁止买入
        """
        current_industry_value = sum(
            p.market_value for p in self.positions.values() 
            if p.industry_name == industry
        )
        
        # 计算买入后的行业价值
        new_industry_value = current_industry_value + target_value
        
        # 计算组合总资产（持仓 + 现金）
        current_portfolio_value = self.portfolio_value
        
        # 计算买入后的行业占比
        new_sector_weight = new_industry_value / current_portfolio_value if current_portfolio_value > 0 else 1.0
        
        if new_sector_weight > self.max_sector_weight:
            logger.debug(f"行业 {industry} 占比 {new_sector_weight:.2%} 超过上限 {self.max_sector_weight:.0%}")
            return False
        
        return True
    
    def _execute_buy_with_sector_control(self, signals: List[V77Signal], 
                                          trade_date: str) -> None:
        """
        执行买入（带行业分散性控制）
        
        【核心逻辑】
        1. 按评分排序
        2. 检查每个信号的行业是否超过 20% 上限
        3. 只买入未超限的行业
        """
        if not signals:
            return
        
        current_positions_count = len(self.positions)
        available_slots = self.max_positions - current_positions_count
        
        if available_slots <= 0:
            return
        
        target_amount_per_stock = self.portfolio_value * self.max_single_position_pct
        
        # 按评分排序
        sorted_signals = sorted(signals, key=lambda x: x.composite_score, reverse=True)
        
        bought = 0
        for signal in sorted_signals:
            if bought >= available_slots:
                break
            
            if signal.symbol in self.positions:
                continue
            
            # 行业分散性检查
            industry = signal.industry_name or 'UNKNOWN'
            if not self._can_buy_sector(industry, target_amount_per_stock):
                logger.debug(f"{trade_date}: 行业 {industry} 已达上限，跳过 {signal.symbol}")
                continue
            
            buy_price = signal.close_price * (1 + self.slippage_buy)
            shares = int(target_amount_per_stock / buy_price / 100) * 100
            
            if shares < 100:
                continue
            
            buy_amount = buy_price * shares
            commission = max(buy_amount * self.commission_rate, self.min_commission)
            slippage_cost = buy_amount * self.slippage_buy
            transfer_fee = shares * self.transfer_fee
            total_cost = buy_amount + commission + slippage_cost + transfer_fee
            
            if total_cost > self.cash:
                continue
            
            self.cash -= total_cost
            
            position = V77Position(
                symbol=signal.symbol,
                shares=shares,
                avg_cost=buy_price,
                buy_price=buy_price,
                buy_date=trade_date,
                signal_date=signal.trade_date,
                trade_date=trade_date,
                signal_score=signal.composite_score,
                composite_score=signal.composite_score,
                
                # V77 新增因子
                rs_breakout_score=signal.rs_breakout_score,
                flow_persistence_score=signal.flow_persistence,
                temporal_momentum_score=signal.temporal_momentum_score,
                ma_slope=signal.ma_slope,
                range_position=signal.range_position,
                
                sector_crowding=signal.sector_crowding,
                volatility=signal.volatility,
                current_price=buy_price,
                market_value=buy_price * shares,
                peak_price=buy_price,
                stop_loss_price=buy_price * (1 - self.stop_loss_ratio),
                trailing_stop_price=buy_price * (1 + self.trailing_stop_ratio),
                industry_name=industry,
            )
            
            self.positions[signal.symbol] = position
            
            # 统计因子值
            self.flow_persistence_values.append(signal.flow_persistence)
            self.rs_breakout_values.append(signal.rs_breakout_score)
            self.temporal_momentum_values.append(signal.temporal_momentum_score)
            
            # 统计时序惩罚
            if signal.temporal_penalty < 1.0:
                self.temporal_penalty_count += 1
            
            # 统计拥挤度交易
            if signal.sector_penalty < 1.0:
                self.crowded_sector_trades += 1
                self.sector_penalty_applied += 1
            
            trade = V77Trade(
                trade_date=trade_date,
                symbol=signal.symbol,
                side='buy',
                shares=shares,
                price=buy_price,
                amount=buy_amount,
                commission=commission,
                slippage=slippage_cost,
                stamp_duty=0.0,
                transfer_fee=transfer_fee,
                total_cost=total_cost,
                reason=f"买入信号 (评分={signal.composite_score:.2f}, 行业={industry})",
                signal_date=signal.trade_date,
            )
            self.trades.append(trade)
            bought += 1
        
        if bought > 0:
            logger.info(f"{trade_date} 买入 {bought} 只股票")
    
    def _execute_sell(self, symbol: str, position: V77Position,
                      trade_date: str, reason: str) -> None:
        """执行卖出"""
        current_price = position.current_price
        shares = position.shares
        
        sell_amount = current_price * shares
        commission = max(sell_amount * self.commission_rate, self.min_commission)
        slippage_cost = sell_amount * self.slippage_sell
        stamp_duty = sell_amount * self.stamp_duty
        transfer_fee = shares * self.transfer_fee
        total_cost = commission + slippage_cost + stamp_duty + transfer_fee
        net_proceeds = sell_amount - total_cost
        
        self.cash += net_proceeds
        
        trade = V77Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='sell',
            shares=shares,
            price=current_price,
            amount=sell_amount,
            commission=commission,
            slippage=slippage_cost,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason=reason,
            holding_days=position.holding_days,
            signal_date=position.signal_date,
        )
        self.trades.append(trade)
        
        del self.positions[symbol]
        
        logger.debug(f"{trade_date} 卖出 {symbol} ({reason})")
    
    def _record_daily_value(self, trade_date: str) -> None:
        """记录每日净值"""
        portfolio_market_value = sum(p.market_value for p in self.positions.values())
        self.portfolio_value = self.cash + portfolio_market_value
        
        self.daily_values.append({
            'trade_date': trade_date,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'positions_count': len(self.positions),
        })
    
    def _compute_backtest_result(self, start_date: str, end_date: str,
                                  ic_stats: Dict[str, float],
                                  monthly_ic_stats: Dict[str, float],
                                  score_hist_data: Dict[str, Any]) -> V77BacktestResult:
        """计算回测结果"""
        # 1. 收益指标
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        days = (datetime.strptime(end_date, '%Y-%m-%d') - 
                datetime.strptime(start_date, '%Y-%m-%d')).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # 2. 风险指标
        daily_values_df = pl.DataFrame(self.daily_values)
        if not daily_values_df.is_empty():
            daily_returns = daily_values_df['portfolio_value'].pct_change().drop_nulls()
            volatility = float(daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
            
            if volatility > 0:
                sharpe_ratio = (annualized_return - 0.02) / volatility
            else:
                sharpe_ratio = 0.0
            
            nav = daily_values_df['portfolio_value'].to_numpy()
            peak = np.maximum.accumulate(nav)
            drawdown = (nav - peak) / peak
            max_drawdown = abs(float(np.min(drawdown)))
            
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            calmar_ratio = 0.0
        
        # 3. 交易指标
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        win_trades = []
        loss_trades = []
        
        buy_trade_map: Dict[str, List[V77Trade]] = {}
        for bt in buy_trades:
            if bt.symbol not in buy_trade_map:
                buy_trade_map[bt.symbol] = []
            buy_trade_map[bt.symbol].append(bt)
        
        for sell_trade in sell_trades:
            symbol = sell_trade.symbol
            if symbol in buy_trade_map:
                for buy_trade in reversed(buy_trade_map[symbol]):
                    buy_amount = buy_trade.price * buy_trade.shares
                    sell_amount = sell_trade.price * sell_trade.shares
                    total_cost = buy_trade.total_cost + sell_trade.total_cost
                    pnl = sell_amount - buy_amount - total_cost
                    
                    if pnl > 0:
                        win_trades.append(sell_trade)
                    else:
                        loss_trades.append(sell_trade)
                    break
        
        total_sell_trades = len(sell_trades)
        win_count = len(win_trades)
        win_rate = win_count / total_sell_trades if total_sell_trades > 0 else 0.0
        
        avg_win = np.mean([t.amount for t in win_trades]) if win_trades else 0.0
        avg_loss = np.mean([t.amount for t in loss_trades]) if loss_trades else 0.0
        
        if loss_trades and avg_loss > 0 and win_trades:
            profit_factor = abs(avg_win * win_count / (avg_loss * len(loss_trades)))
        else:
            profit_factor = 0.0
        
        holding_days = [t.holding_days for t in sell_trades if t.holding_days > 0]
        avg_holding_days = np.mean(holding_days) if holding_days else 0.0
        
        # 4. Rank IC 指标
        mean_rank_ic = ic_stats.get('mean_rank_ic', 0.0)
        monthly_rank_ic = monthly_ic_stats.get('monthly_mean_rank_ic', 0.0)
        rank_ic_pass = monthly_ic_stats.get('monthly_pass', False)
        negative_months = monthly_ic_stats.get('negative_months', 0)
        
        # 5. 评分分布
        stats = score_hist_data.get('statistics', {})
        score_mean = stats.get('mean', 0.0)
        score_std = stats.get('std', 0.0)
        
        # 6. V77 因子统计
        avg_flow_persistence = np.mean(self.flow_persistence_values) if self.flow_persistence_values else 0.0
        avg_rs_breakout = np.mean(self.rs_breakout_values) if self.rs_breakout_values else 0.0
        avg_temporal_momentum = np.mean(self.temporal_momentum_values) if self.temporal_momentum_values else 0.0
        
        # 7. 行业分布统计
        sector_allocation = self._get_sector_exposure()
        
        # 8. 生成 Rank IC 散点图数据
        self._generate_rank_ic_scatter_data()
        
        # 构建结果
        result = V77BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=len(buy_trades),
            win_trades=win_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding_days,
            max_positions=max(len(self.positions) for _ in [1]) if self.positions else 0,
            mean_rank_ic=mean_rank_ic,
            monthly_rank_ic=monthly_rank_ic,
            rank_ic_pass=rank_ic_pass,
            negative_months=negative_months,
            monthly_ic_stats=self.rank_ic_calculator.monthly_stats,
            score_mean=score_mean,
            score_std=score_std,
            score_distribution=score_hist_data,
            
            # V77 因子统计
            avg_flow_persistence=avg_flow_persistence,
            avg_rs_breakout=avg_rs_breakout,
            avg_temporal_momentum=avg_temporal_momentum,
            temporal_penalty_count=self.temporal_penalty_count,
            
            crowded_sector_trades=self.crowded_sector_trades,
            sector_penalty_applied=self.sector_penalty_applied,
            daily_values=daily_values_df,
            trades=self.trades,
            sector_allocation=sector_allocation,
            rank_ic_scatter_data=self.rank_ic_scatter_data,
        )
        
        return result
    
    def _generate_rank_ic_scatter_data(self) -> None:
        """生成 Rank IC 与收益率散点图数据"""
        for ic_metric in self.rank_ic_calculator.ic_results:
            self.rank_ic_scatter_data.append({
                'trade_date': ic_metric.trade_date,
                'rank_ic': ic_metric.rank_ic,
                'ic': ic_metric.ic,
            })
    
    def print_backtest_result(self, result: V77BacktestResult) -> None:
        """打印回测结果"""
        logger.info("=" * 60)
        logger.info("V77 回测结果 - 量价交互特征挖掘与 Rank IC 爆发计划")
        logger.info("=" * 60)
        logger.info(f"回测区间：{result.start_date} 至 {result.end_date}")
        logger.info("-" * 40)
        logger.info("【收益指标】")
        logger.info(f"  总收益：     {result.total_return:.4f} ({result.total_return*100:.2f}%)")
        logger.info(f"  年化收益：   {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%)")
        logger.info("-" * 40)
        logger.info("【风险指标】")
        logger.info(f"  最大回撤：   {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)")
        logger.info(f"  波动率：     {result.volatility:.4f}")
        logger.info(f"  夏普比率：   {result.sharpe_ratio:.3f}")
        logger.info(f"  Calmar 比率：{result.calmar_ratio:.3f} (目标：>{V77_CALMAR_RATIO_TARGET})")
        logger.info("-" * 40)
        logger.info("【交易指标】")
        logger.info(f"  总交易数：   {result.total_trades}")
        logger.info(f"  胜率：       {result.win_rate:.2%}")
        logger.info(f"  平均盈利：   {result.avg_win:,.2f}")
        logger.info(f"  平均亏损：   {result.avg_loss:,.2f}")
        logger.info(f"  盈亏比：     {result.profit_factor:.2f}")
        logger.info(f"  平均持仓天数：{result.avg_holding_days:.1f}")
        logger.info("-" * 40)
        logger.info("【V77 因子统计】")
        logger.info(f"  平均资金流持续性：{result.avg_flow_persistence:.4f}")
        logger.info(f"  平均 RS Breakout:  {result.avg_rs_breakout:.2f}")
        logger.info(f"  平均时序动量：     {result.avg_temporal_momentum:.2f}")
        logger.info(f"  时序惩罚次数：     {result.temporal_penalty_count}")
        logger.info("-" * 40)
        logger.info("【Rank IC 指标】（核心验收标准）")
        logger.info(f"  Mean Rank IC:      {result.mean_rank_ic:.4f} (目标：>{V77_RANK_IC_TARGET})")
        logger.info(f"  月度 Rank IC 均值：  {result.monthly_rank_ic:.4f} (目标：>{V77_RANK_IC_TARGET})")
        logger.info(f"  负值月份数量：     {result.negative_months} (上限：2)")
        logger.info(f"  Rank IC 达标：     {result.rank_ic_pass}")
        logger.info("-" * 40)
        logger.info("【评分分布】")
        logger.info(f"  评分均值：   {result.score_mean:.2f}")
        logger.info(f"  评分标准差：{result.score_std:.2f}")
        logger.info("-" * 40)
        logger.info("【行业拥挤度控制】")
        logger.info(f"  拥挤行业交易数：{result.crowded_sector_trades}")
        logger.info(f"  惩罚应用次数：{result.sector_penalty_applied}")
        logger.info("-" * 40)
        logger.info("【行业分布】")
        for sector, weight in sorted(result.sector_allocation.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {sector}: {weight*100:.1f}%")
        logger.info("=" * 60)
    
    def generate_report(self, result: V77BacktestResult, 
                        output_path: Optional[str] = None) -> str:
        """生成回测报告"""
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"v77_backtest_report_{timestamp}.md"
        
        # 评分分布直方图数据
        hist_data = result.score_distribution.get('histogram', {})
        hist_counts = hist_data.get('counts', [])
        bin_edges = hist_data.get('bin_edges', [])
        
        # 构建直方图 ASCII
        histogram_ascii = self._generate_ascii_histogram(hist_counts, bin_edges)
        
        # 月度 IC 柱状图
        monthly_ic_chart = generate_monthly_ic_ascii_chart(
            result.monthly_ic_stats, 
            V77_RANK_IC_TARGET
        )
        
        # 检查回撤是否达标
        drawdown_pass = result.max_drawdown <= V77_MAX_DRAWDOWN_TARGET
        
        # 检查 Calmar Ratio 是否达标
        calmar_pass = result.calmar_ratio >= V77_CALMAR_RATIO_TARGET
        
        # 检查负值月份是否达标
        negative_months_pass = result.negative_months <= 2
        
        report_lines = [
            "# V77 回测报告 - 量价交互特征挖掘与 Rank IC 爆发计划",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 核心算法",
            "",
            "1. **MAD 去极值化**: Median Absolute Deviation 处理，3.5σ截断",
            "2. **Money Flow Persistence**: 资金流持续性自相关系数（5 日窗口）",
            "3. **RS Breakout Quality**: 突破质量 = RS × Range Position × Volume Confirm",
            "4. **时序动量审计**: 均线斜率检查，前 5% 且掉头打 5 折",
            "5. **线性融合**: Score = 0.5*RS_Breakout + 0.3*Flow_Persistence + 0.2*Temporal_Momentum",
            "",
            "## 基本信息",
            "",
            "| 项目 | 值 |",
            "|------|-----|",
            f"| 回测区间 | {result.start_date} 至 {result.end_date} |",
            f"| 初始资金 | {result.initial_capital:,.0f} |",
            f"| 最大持仓数 | {self.max_positions} |",
            f"| 单仓上限 | {self.max_single_position_pct*100:.1f}% |",
            f"| 单行业上限 | {self.max_sector_weight*100:.1f}% |",
            "",
            "## 收益指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总收益 | {result.total_return:.4f} ({result.total_return*100:.2f}%) |",
            f"| 年化收益 | {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%) |",
            f"| 最大回撤 | {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%) |",
            f"| 夏普比率 | {result.sharpe_ratio:.3f} |",
            f"| Calmar 比率 | {result.calmar_ratio:.3f} (目标：>{V77_CALMAR_RATIO_TARGET}) |",
            "",
            "## 交易指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总交易数 | {result.total_trades} |",
            f"| 胜率 | {result.win_rate:.2%} |",
            f"| 平均盈利 | {result.avg_win:,.2f} |",
            f"| 平均亏损 | {result.avg_loss:,.2f} |",
            f"| 盈亏比 | {result.profit_factor:.2f} |",
            f"| 平均持仓天数 | {result.avg_holding_days:.1f} |",
            "",
            "## V77 因子统计",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 平均资金流持续性 | {result.avg_flow_persistence:.4f} |",
            f"| 平均 RS Breakout | {result.avg_rs_breakout:.2f} |",
            f"| 平均时序动量 | {result.avg_temporal_momentum:.2f} |",
            f"| 时序惩罚次数 | {result.temporal_penalty_count} |",
            "",
            "## Rank IC 指标（验收标准 A）",
            "",
            "| 指标 | 值 | 目标 | 达标 |",
            "|------|-----|------|------|",
            f"| Mean Rank IC | {result.mean_rank_ic:.4f} | >{V77_RANK_IC_TARGET} | {'✓' if result.mean_rank_ic >= V77_RANK_IC_TARGET else '✗'} |",
            f"| 月度 Rank IC 均值 | {result.monthly_rank_ic:.4f} | >{V77_RANK_IC_TARGET} | {'✓' if result.rank_ic_pass else '✗'} |",
            f"| 负值月份数量 | {result.negative_months} | <=2 | {'✓' if negative_months_pass else '✗'} |",
            "",
            monthly_ic_chart,
            "",
            "## 评分分布直方图",
            "",
            "```",
            histogram_ascii,
            "```",
            "",
            "| 统计量 | 值 |",
            "|------|-----|",
            f"| 评分均值 | {result.score_mean:.2f} |",
            f"| 评分标准差 | {result.score_std:.2f} |",
            f"| 评分最小值 | {result.score_distribution.get('statistics', {}).get('min', 0.0):.2f} |",
            f"| 评分最大值 | {result.score_distribution.get('statistics', {}).get('max', 0.0):.2f} |",
            "",
            "## 行业拥挤度控制",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 拥挤行业交易数 | {result.crowded_sector_trades} |",
            f"| 惩罚应用次数 | {result.sector_penalty_applied} |",
            "",
            "## 行业分布（前 10 大）",
            "",
            "| 行业 | 占比 |",
            "|------|-----|",
        ]
        
        for sector, weight in sorted(result.sector_allocation.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
            report_lines.append(f"| {sector} | {weight*100:.1f}% |")
        
        report_lines.extend([
            "",
            "## 回撤控制（验收标准 B）",
            "",
            f"- 最大回撤：{result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)",
            f"- 目标上限：{V77_MAX_DRAWDOWN_TARGET*100:.1f}%",
            f"- 达标状态：{'✓' if drawdown_pass else '✗'}",
            "",
            "## Calmar Ratio（验收标准 B）",
            "",
            f"- Calmar Ratio: {result.calmar_ratio:.3f}",
            f"- 目标下限：{V77_CALMAR_RATIO_TARGET}",
            f"- 达标状态：{'✓' if calmar_pass else '✗'}",
            "",
            "## 负值月份控制（验收标准 C）",
            "",
            f"- 负值月份数量：{result.negative_months}",
            f"- 目标上限：2",
            f"- 达标状态：{'✓' if negative_months_pass else '✗'}",
            "",
            "## 综合验收结论",
            "",
            f"1. **Rank IC 达标**: {'✓ 是' if result.rank_ic_pass else '✗ 否'} (Mean Rank IC = {result.mean_rank_ic:.4f})",
            f"2. **Calmar Ratio 达标**: {'✓ 是' if calmar_pass else '✗ 否'} (Calmar = {result.calmar_ratio:.3f})",
            f"3. **最大回撤达标**: {'✓ 是' if drawdown_pass else '✗ 否'} (Max DD = {result.max_drawdown*100:.2f}%)",
            f"4. **负值月份达标**: {'✓ 是' if negative_months_pass else '✗ 否'} (Negative Months = {result.negative_months})",
            "",
            f"**总收益**: {result.total_return*100:.2f}%",
            "",
            "---",
            "*V77 回测报告完成*",
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"报告已保存至：{output_path}")
        
        return report_content
    
    def _generate_ascii_histogram(self, counts: List[int], 
                                   bin_edges: List[float],
                                   width: int = 50) -> str:
        """生成 ASCII 直方图"""
        if not counts or not bin_edges:
            return "无数据"
        
        max_count = max(counts) if counts else 1
        lines = []
        
        for i, count in enumerate(counts):
            bar_len = int(count / max_count * width) if max_count > 0 else 0
            bar = "█" * bar_len
            label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            lines.append(f"{label:>8} |{bar} ({count})")
        
        return "\n".join(lines)


# ===========================================
# 主函数
# ===========================================

def run_v77_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     output_path: Optional[str] = None) -> V77BacktestResult:
    """
    运行 V77 回测
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        output_path: 报告输出路径
        
    Returns:
        V77BacktestResult: 回测结果
    """
    # 初始化数据库
    db = DatabaseManager()
    
    # 初始化引擎
    engine = V77BacktestEngine(db=db)
    
    # 运行回测
    result = engine.run_backtest(start_date, end_date)
    
    # 打印结果
    engine.print_backtest_result(result)
    
    # 生成报告
    engine.generate_report(result, output_path)
    
    # 打印 Rank IC 报告
    engine.rank_ic_calculator.print_rank_ic_report()
    
    return result


if __name__ == "__main__":
    # 运行回测
    run_v77_backtest()