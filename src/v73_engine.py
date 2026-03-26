"""
V73 Backtest Engine - 多因子融合回测引擎

【核心功能】
1. 基于 V73 核心逻辑执行回测
2. 严格的仓位管理和止损止盈
3. Rank IC 实时监控与报告
4. 评分分布直方图生成

作者：量化系统
版本：V73.0
日期：2026-03-25
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger

from src.db_manager import DatabaseManager
from src.core.v73_logic import (
    V73DataManager,
    V73AlphaCenter,
    V73RankICCalculator,
    V73Signal,
    V73Position,
    V73Trade,
    V73_INITIAL_CAPITAL,
    V73_MAX_POSITIONS,
    V73_MAX_SINGLE_POSITION_PCT,
    V73_COMMISSION_RATE,
    V73_MIN_COMMISSION,
    V73_SLIPPAGE_BUY,
    V73_SLIPPAGE_SELL,
    V73_STAMP_DUTY,
    V73_TRANSFER_FEE,
    V73_STOP_LOSS_RATIO,
    V73_PROFIT_TARGET_RATIO,
    V73_TRAILING_STOP_RATIO,
    generate_score_histogram_data,
)


# ===========================================
# V73 回测结果数据类
# ===========================================

@dataclass
class V73BacktestResult:
    """V73 回测结果"""
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
    
    # 评分分布
    score_mean: float = 0.0
    score_std: float = 0.0
    score_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # 详细数据
    daily_values: Optional[pl.DataFrame] = None
    trades: List[V73Trade] = field(default_factory=list)


# ===========================================
# V73 回测引擎
# ===========================================

class V73BacktestEngine:
    """
    V73 回测引擎 - 多因子融合回测
    
    【核心特性】
    1. 初始资金严格锁定 100,000 元
    2. 费率、止盈止损规则严禁修改
    3. 全市场股票连续评分
    4. Rank IC 实时监控
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None,
                 config: Dict[str, Any] = None):
        """
        初始化 V73 回测引擎
        
        Args:
            db: 数据库管理器
            config: 配置字典
        """
        self.db = db or DatabaseManager()
        self.config = config or {}
        
        # 初始资金（严禁修改）
        self.initial_capital = self.config.get('initial_capital', V73_INITIAL_CAPITAL)
        
        # 仓位管理
        self.max_positions = self.config.get('max_positions', V73_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get(
            'max_single_position_pct', V73_MAX_SINGLE_POSITION_PCT
        )
        
        # 费率（严禁修改）
        self.commission_rate = V73_COMMISSION_RATE
        self.min_commission = V73_MIN_COMMISSION
        self.slippage_buy = V73_SLIPPAGE_BUY
        self.slippage_sell = V73_SLIPPAGE_SELL
        self.stamp_duty = V73_STAMP_DUTY
        self.transfer_fee = V73_TRANSFER_FEE
        
        # 止损止盈（严禁修改）
        self.stop_loss_ratio = V73_STOP_LOSS_RATIO
        self.profit_target_ratio = V73_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V73_TRAILING_STOP_RATIO
        
        # 初始化组件
        self.data_manager = V73DataManager(db=self.db, config=self.config)
        self.alpha_center = V73AlphaCenter(config=self.config)
        self.rank_ic_calculator = V73RankICCalculator(db=self.db, config=self.config)
        
        # 回测状态
        self.positions: Dict[str, V73Position] = {}
        self.trades: List[V73Trade] = []
        self.daily_values: List[Dict[str, Any]] = []
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        logger.info(f"V73BacktestEngine 初始化完成：初始资金={self.initial_capital:,.0f}")
    
    def run_backtest(self, start_date: str, end_date: str) -> V73BacktestResult:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            V73BacktestResult: 回测结果
        """
        logger.info("=" * 60)
        logger.info(f"V73 回测开始：[{start_date}, {end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.0f}")
        logger.info("=" * 60)
        
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
        
        # 5. 执行回测交易
        logger.info("Step 5: 执行交易...")
        self._execute_trades(signals_df, start_date, end_date)
        
        # 6. 计算回测结果
        logger.info("Step 6: 计算回测指标...")
        result = self._compute_backtest_result(
            start_date, end_date,
            ic_stats, monthly_ic_stats,
            score_hist_data
        )
        
        return result
    
    def _execute_trades(self, signals_df: pl.DataFrame,
                        start_date: str, end_date: str) -> None:
        """
        执行交易逻辑
        
        【核心逻辑】
        1. 每日生成买入信号
        2. 检查持仓止损止盈条件
        3. 仓位管理：单仓上限 10%，最多 10 只
        """
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
            
            # 4. 执行买入
            self._execute_buy(signals, trade_date)
            
            # 5. 记录每日净值
            self._record_daily_value(trade_date)
    
    def _update_positions(self, signals_df: pl.DataFrame, trade_date: str) -> None:
        """更新持仓价格"""
        for symbol, position in list(self.positions.items()):
            # 获取当日收盘价
            price_data = signals_df.filter(
                (pl.col('symbol') == symbol) & 
                (pl.col('trade_date') == trade_date)
            )
            
            if not price_data.is_empty():
                close_price = price_data['close'][0]
                position.current_price = close_price
                position.market_value = close_price * position.shares
                
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
            
            # 计算收益率
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
    
    def _execute_buy(self, signals: List[V73Signal], trade_date: str) -> None:
        """执行买入"""
        if not signals:
            return
        
        # 计算可用仓位
        current_positions_count = len(self.positions)
        available_slots = self.max_positions - current_positions_count
        
        if available_slots <= 0:
            return
        
        # 计算每只股票的目标金额
        target_amount_per_stock = self.portfolio_value * self.max_single_position_pct
        
        # 按评分排序，买入前 N 只
        sorted_signals = sorted(signals, key=lambda x: x.composite_score, reverse=True)
        
        bought = 0
        for signal in sorted_signals:
            if bought >= available_slots:
                break
            
            # 检查是否已持仓
            if signal.symbol in self.positions:
                continue
            
            # 计算买入数量
            buy_price = signal.close_price * (1 + self.slippage_buy)
            shares = int(target_amount_per_stock / buy_price / 100) * 100
            
            if shares < 100:
                continue
            
            # 计算交易成本
            buy_amount = buy_price * shares
            commission = max(buy_amount * self.commission_rate, self.min_commission)
            slippage_cost = buy_amount * self.slippage_buy
            transfer_fee = shares * self.transfer_fee
            total_cost = buy_amount + commission + slippage_cost + transfer_fee
            
            # 检查现金是否足够
            if total_cost > self.cash:
                continue
            
            # 执行买入
            self.cash -= total_cost
            
            position = V73Position(
                symbol=signal.symbol,
                shares=shares,
                avg_cost=buy_price,
                buy_price=buy_price,
                buy_date=trade_date,
                signal_date=signal.trade_date,
                trade_date=trade_date,
                signal_score=signal.signal_score,
                composite_score=signal.composite_score,
                rs_score=signal.rs_score,
                snr_weight=signal.snr_weight,
                current_price=buy_price,
                market_value=buy_price * shares,
                peak_price=buy_price,
                stop_loss_price=buy_price * (1 - self.stop_loss_ratio),
                trailing_stop_price=buy_price * (1 + self.trailing_stop_ratio),
            )
            
            self.positions[signal.symbol] = position
            
            # 记录交易
            trade = V73Trade(
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
                reason=f"买入信号 (评分={signal.composite_score:.2f})",
                signal_date=signal.trade_date,
            )
            self.trades.append(trade)
            bought += 1
        
        if bought > 0:
            logger.info(f"{trade_date} 买入 {bought} 只股票")
    
    def _execute_sell(self, symbol: str, position: V73Position,
                      trade_date: str, reason: str) -> None:
        """执行卖出"""
        current_price = position.current_price
        shares = position.shares
        
        # 计算交易金额和成本
        sell_amount = current_price * shares
        commission = max(sell_amount * self.commission_rate, self.min_commission)
        slippage_cost = sell_amount * self.slippage_sell
        stamp_duty = sell_amount * self.stamp_duty
        transfer_fee = shares * self.transfer_fee
        total_cost = commission + slippage_cost + stamp_duty + transfer_fee
        net_proceeds = sell_amount - total_cost
        
        # 更新现金
        self.cash += net_proceeds
        
        # 记录交易
        trade = V73Trade(
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
        
        # 移除持仓
        del self.positions[symbol]
        
        logger.debug(f"{trade_date} 卖出 {symbol} ({reason})")
    
    def _record_daily_value(self, trade_date: str) -> None:
        """记录每日净值"""
        # 计算持仓总市值
        portfolio_market_value = sum(p.market_value for p in self.positions.values())
        
        # 总资产
        self.portfolio_value = self.cash + portfolio_market_value
        
        # 记录
        self.daily_values.append({
            'trade_date': trade_date,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'positions_count': len(self.positions),
        })
    
    def _compute_backtest_result(self, start_date: str, end_date: str,
                                  ic_stats: Dict[str, float],
                                  monthly_ic_stats: Dict[str, float],
                                  score_hist_data: Dict[str, Any]) -> V73BacktestResult:
        """计算回测结果"""
        # 1. 收益指标
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # 年化收益
        days = (datetime.strptime(end_date, '%Y-%m-%d') - 
                datetime.strptime(start_date, '%Y-%m-%d')).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # 2. 风险指标
        daily_values_df = pl.DataFrame(self.daily_values)
        if not daily_values_df.is_empty():
            daily_returns = daily_values_df['portfolio_value'].pct_change().drop_nulls()
            volatility = float(daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
            
            # 夏普比率
            if volatility > 0:
                sharpe_ratio = (annualized_return - 0.02) / volatility
            else:
                sharpe_ratio = 0.0
            
            # 最大回撤
            nav = daily_values_df['portfolio_value'].to_numpy()
            peak = np.maximum.accumulate(nav)
            drawdown = (nav - peak) / peak
            max_drawdown = abs(float(np.min(drawdown)))
            
            # Calmar 比率
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            calmar_ratio = 0.0
        
        # 3. 交易指标
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        # 计算胜率 - 修复 PNL 计算
        win_trades = []
        loss_trades = []
        pnl_list = []
        
        # 构建买入交易索引
        buy_trade_map: Dict[str, List[V73Trade]] = {}
        for bt in buy_trades:
            if bt.symbol not in buy_trade_map:
                buy_trade_map[bt.symbol] = []
            buy_trade_map[bt.symbol].append(bt)
        
        for sell_trade in sell_trades:
            symbol = sell_trade.symbol
            if symbol in buy_trade_map:
                # 找到最近的未配对买入交易
                for buy_trade in reversed(buy_trade_map[symbol]):
                    # 简单 PNL 计算：卖出金额 - 买入金额 - 交易成本
                    buy_amount = buy_trade.price * buy_trade.shares
                    sell_amount = sell_trade.price * sell_trade.shares
                    total_cost = buy_trade.total_cost + sell_trade.total_cost
                    pnl = sell_amount - buy_amount - total_cost
                    pnl_list.append(pnl)
                    
                    if pnl > 0:
                        win_trades.append(sell_trade)
                    else:
                        loss_trades.append(sell_trade)
                    break
        
        total_sell_trades = len(sell_trades)
        win_count = len(win_trades)
        win_rate = win_count / total_sell_trades if total_sell_trades > 0 else 0.0
        
        # 平均盈亏
        avg_win = np.mean([t.amount for t in win_trades]) if win_trades else 0.0
        avg_loss = np.mean([t.amount for t in loss_trades]) if loss_trades else 0.0
        
        # 盈亏比
        if loss_trades and avg_loss > 0 and win_trades:
            profit_factor = abs(avg_win * win_count / (avg_loss * len(loss_trades)))
        else:
            profit_factor = 0.0
        
        # 平均持仓天数
        holding_days = [t.holding_days for t in sell_trades if t.holding_days > 0]
        avg_holding_days = np.mean(holding_days) if holding_days else 0.0
        
        # 4. Rank IC 指标 - 使用月度统计
        mean_rank_ic = ic_stats.get('mean_rank_ic', 0.0)
        monthly_rank_ic = monthly_ic_stats.get('monthly_mean_rank_ic', 0.0)
        rank_ic_pass = monthly_ic_stats.get('monthly_pass', False)
        
        # 5. 评分分布 - 修复 nan 问题（直接从直方图数据计算）
        stats = score_hist_data.get('statistics', {})
        score_mean = stats.get('mean', 0.0)
        score_std = stats.get('std', 0.0)
        score_min = stats.get('min', 0.0)
        score_max = stats.get('max', 0.0)
        score_q1 = stats.get('q1', 0.0)
        score_q3 = stats.get('q3', 0.0)
        
        # 如果统计信息无效（nan 或 0），从直方图估算
        import math
        if score_mean is None or (isinstance(score_mean, float) and (math.isnan(score_mean) or score_mean == 0.0)):
            hist_data = score_hist_data.get('histogram', {})
            counts = hist_data.get('counts', [])
            bin_edges = hist_data.get('bin_edges', [])
            if counts and bin_edges and len(counts) == len(bin_edges) - 1:
                total = sum(counts)
                if total > 0:
                    score_mean = sum(counts[i] * (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))) / total
                    score_std = (sum(counts[i] * ((bin_edges[i] + bin_edges[i+1]) / 2 - score_mean) ** 2 for i in range(len(counts))) / total) ** 0.5
                    score_min = bin_edges[0]
                    score_max = bin_edges[-1]
                    # 估算分位数
                    cumsum = 0
                    q1_idx = int(total * 0.25)
                    q3_idx = int(total * 0.75)
                    q1_found = False
                    q3_found = False
                    for i, count in enumerate(counts):
                        cumsum += count
                        if not q1_found and cumsum >= q1_idx:
                            score_q1 = (bin_edges[i] + bin_edges[i+1]) / 2
                            q1_found = True
                        if not q3_found and cumsum >= q3_idx:
                            score_q3 = (bin_edges[i] + bin_edges[i+1]) / 2
                            q3_found = True
                            break
        
        # 更新 score_hist_data 中的统计信息
        score_hist_data['statistics'] = {
            'mean': score_mean,
            'std': score_std,
            'min': score_min,
            'max': score_max,
            'q1': score_q1,
            'q3': score_q3,
        }
        
        # 构建结果
        result = V73BacktestResult(
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
            score_mean=score_mean,
            score_std=score_std,
            score_distribution=score_hist_data,
            daily_values=daily_values_df,
            trades=self.trades,
        )
        
        return result
    
    def print_backtest_result(self, result: V73BacktestResult) -> None:
        """打印回测结果"""
        logger.info("=" * 60)
        logger.info("V73 回测结果")
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
        logger.info(f"  Calmar 比率：{result.calmar_ratio:.3f}")
        logger.info("-" * 40)
        logger.info("【交易指标】")
        logger.info(f"  总交易数：   {result.total_trades}")
        logger.info(f"  胜率：       {result.win_rate:.2%}")
        logger.info(f"  平均盈利：   {result.avg_win:,.2f}")
        logger.info(f"  平均亏损：   {result.avg_loss:,.2f}")
        logger.info(f"  盈亏比：     {result.profit_factor:.2f}")
        logger.info(f"  平均持仓天数：{result.avg_holding_days:.1f}")
        logger.info("-" * 40)
        logger.info("【Rank IC 指标】")
        logger.info(f"  Mean Rank IC:    {result.mean_rank_ic:.4f}")
        logger.info(f"  月度 Rank IC 均值：{result.monthly_rank_ic:.4f}")
        logger.info(f"  Rank IC 达标：   {result.rank_ic_pass}")
        logger.info("-" * 40)
        logger.info("【评分分布】")
        logger.info(f"  评分均值：   {result.score_mean:.2f}")
        logger.info(f"  评分标准差：{result.score_std:.2f}")
        logger.info("=" * 60)
    
    def generate_report(self, result: V73BacktestResult, 
                        output_path: Optional[str] = None) -> str:
        """生成回测报告"""
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"v73_backtest_report_{timestamp}.md"
        
        # 评分分布直方图数据
        hist_data = result.score_distribution.get('histogram', {})
        hist_counts = hist_data.get('counts', [])
        bin_edges = hist_data.get('bin_edges', [])
        
        # 构建直方图 ASCII
        histogram_ascii = self._generate_ascii_histogram(hist_counts, bin_edges)
        
        report_lines = [
            "# V73 回测报告 - 多因子融合与评分分布优化",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 基本信息",
            "",
            "| 项目 | 值 |",
            "|------|-----|",
            f"| 回测区间 | {result.start_date} 至 {result.end_date} |",
            f"| 初始资金 | {result.initial_capital:,.0f} |",
            f"| 最大持仓数 | {self.max_positions} |",
            f"| 单仓上限 | {self.max_single_position_pct*100:.1f}% |",
            "",
            "## 收益指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总收益 | {result.total_return:.4f} ({result.total_return*100:.2f}%) |",
            f"| 年化收益 | {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%) |",
            f"| 最大回撤 | {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%) |",
            f"| 夏普比率 | {result.sharpe_ratio:.3f} |",
            f"| Calmar 比率 | {result.calmar_ratio:.3f} |",
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
            "## Rank IC 指标（验收标准）",
            "",
            "| 指标 | 值 | 目标 | 达标 |",
            "|------|-----|------|------|",
            f"| Mean Rank IC | {result.mean_rank_ic:.4f} | >0.02 | {'✓' if result.mean_rank_ic >= 0.02 else '✗'} |",
            f"| 月度 Rank IC 均值 | {result.monthly_rank_ic:.4f} | >0.02 | {'✓' if result.rank_ic_pass else '✗'} |",
            "",
            "## 评分分布直方图（验证连续性）",
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
            f"| Q1 (25 分位) | {result.score_distribution.get('statistics', {}).get('q1', 0.0):.2f} |",
            f"| Q3 (75 分位) | {result.score_distribution.get('statistics', {}).get('q3', 0.0):.2f} |",
            "",
            "## 结论",
            "",
            f"1. **Rank IC 达标**: {'是' if result.rank_ic_pass else '否'}",
            f"2. **评分分布连续**: 标准差={result.score_std:.2f}, 覆盖范围=[{result.score_distribution.get('statistics', {}).get('min', 0.0):.2f}, {result.score_distribution.get('statistics', {}).get('max', 0.0):.2f}]",
            f"3. **总收益**: {result.total_return*100:.2f}%",
            f"4. **最大回撤**: {result.max_drawdown*100:.2f}%",
            "",
            "---",
            "*V73 回测报告完成*",
        ]
        
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

def run_v73_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     output_path: Optional[str] = None) -> V73BacktestResult:
    """
    运行 V73 回测
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        output_path: 报告输出路径
        
    Returns:
        V73BacktestResult: 回测结果
    """
    # 初始化数据库
    db = DatabaseManager()
    
    # 初始化引擎
    engine = V73BacktestEngine(db=db)
    
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
    run_v73_backtest()