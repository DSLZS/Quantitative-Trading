"""
V74 Backtest Engine - 特征工程进阶与非线性因子融合回测引擎

【核心功能】
1. 基于 V74 核心逻辑执行回测
2. 严格的仓位管理和止损止盈
3. Rank IC 实时监控与月度柱状图
4. 评分分布直方图生成
5. 数据完整性检查与重试机制

【V74 新增特性】
- VPD 量价背离因子
- 熵权法 SNR + 偏度审计
- Tanh 非线性融合
- 月度 Rank IC 柱状图输出

作者：量化系统
版本：V74.0
日期：2026-03-26
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
from src.core.v74_logic import (
    V74DataManager,
    V74AlphaCenter,
    V74RankICCalculator,
    V74Signal,
    V74Position,
    V74Trade,
    V74MonthlyICStats,
    V74_INITIAL_CAPITAL,
    V74_MAX_POSITIONS,
    V74_MAX_SINGLE_POSITION_PCT,
    V74_COMMISSION_RATE,
    V74_MIN_COMMISSION,
    V74_SLIPPAGE_BUY,
    V74_SLIPPAGE_SELL,
    V74_STAMP_DUTY,
    V74_TRANSFER_FEE,
    V74_STOP_LOSS_RATIO,
    V74_PROFIT_TARGET_RATIO,
    V74_TRAILING_STOP_RATIO,
    V74_MIN_FUND_FLOW_ROWS,
    V74_RANK_IC_TARGET,
    V74_SCORE_STD_TARGET_MIN,
    V74_SCORE_STD_TARGET_MAX,
    generate_score_histogram_data,
    generate_monthly_ic_ascii_chart,
)


# ===========================================
# V74 回测结果数据类
# ===========================================

@dataclass
class V74BacktestResult:
    """V74 回测结果"""
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
    monthly_ic_stats: List[V74MonthlyICStats] = field(default_factory=list)
    
    # 评分分布
    score_mean: float = 0.0
    score_std: float = 0.0
    score_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # 详细数据
    daily_values: Optional[pl.DataFrame] = None
    trades: List[V74Trade] = field(default_factory=list)


# ===========================================
# V74 回测引擎
# ===========================================

class V74BacktestEngine:
    """
    V74 回测引擎 - 特征工程进阶与非线性因子融合
    
    【核心特性】
    1. 初始资金严格锁定 100,000 元
    2. 费率、止盈止损规则严禁修改
    3. 全市场股票连续评分
    4. Rank IC 实时监控
    5. 月度 Rank IC 柱状图输出
    6. 数据完整性检查与重试机制
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None,
                 config: Dict[str, Any] = None):
        """
        初始化 V74 回测引擎
        
        Args:
            db: 数据库管理器
            config: 配置字典
        """
        self.db = db or DatabaseManager()
        self.config = config or {}
        
        # 初始资金（严禁修改）
        self.initial_capital = self.config.get('initial_capital', V74_INITIAL_CAPITAL)
        
        # 仓位管理
        self.max_positions = self.config.get('max_positions', V74_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get(
            'max_single_position_pct', V74_MAX_SINGLE_POSITION_PCT
        )
        
        # 费率（严禁修改）
        self.commission_rate = V74_COMMISSION_RATE
        self.min_commission = V74_MIN_COMMISSION
        self.slippage_buy = V74_SLIPPAGE_BUY
        self.slippage_sell = V74_SLIPPAGE_SELL
        self.stamp_duty = V74_STAMP_DUTY
        self.transfer_fee = V74_TRANSFER_FEE
        
        # 止损止盈（严禁修改）
        self.stop_loss_ratio = V74_STOP_LOSS_RATIO
        self.profit_target_ratio = V74_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V74_TRAILING_STOP_RATIO
        
        # 初始化组件
        self.data_manager = V74DataManager(db=self.db, config=self.config)
        self.alpha_center = V74AlphaCenter(config=self.config)
        self.rank_ic_calculator = V74RankICCalculator(db=self.db, config=self.config)
        
        # 回测状态
        self.positions: Dict[str, V74Position] = {}
        self.trades: List[V74Trade] = []
        self.daily_values: List[Dict[str, Any]] = []
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        logger.info(f"V74BacktestEngine 初始化完成：初始资金={self.initial_capital:,.0f}")
    
    def _check_data_integrity(self) -> bool:
        """
        检查数据完整性
        
        【核心逻辑】
        1. 检查 2024 全年资金流数据
        2. 若数据不足，记录警告但继续（由用户决定是否补抓）
        3. 严禁报错跳过，必须明确报告
        """
        logger.info("=" * 60)
        logger.info("V74: 数据完整性检查")
        logger.info("=" * 60)
        
        is_valid, message = self.data_manager.check_2024_data_integrity()
        
        if not is_valid:
            logger.warning(f"V74: 数据完整性检查未通过 - {message}")
            logger.warning("V74: 建议先运行数据同步脚本补充数据")
            # 不中断回测，继续执行
        else:
            logger.info(f"V74: 数据完整性检查通过 - {message}")
        
        return is_valid
    
    def run_backtest(self, start_date: str, end_date: str) -> V74BacktestResult:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            V74BacktestResult: 回测结果
        """
        logger.info("=" * 60)
        logger.info("V74 回测开始：特征工程进阶与非线性因子融合")
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.0f}")
        logger.info("=" * 60)
        
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
    
    def _execute_trades(self, signals_df: pl.DataFrame,
                        start_date: str, end_date: str) -> None:
        """执行交易逻辑"""
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
    
    def _execute_buy(self, signals: List[V74Signal], trade_date: str) -> None:
        """执行买入"""
        if not signals:
            return
        
        current_positions_count = len(self.positions)
        available_slots = self.max_positions - current_positions_count
        
        if available_slots <= 0:
            return
        
        target_amount_per_stock = self.portfolio_value * self.max_single_position_pct
        
        sorted_signals = sorted(signals, key=lambda x: x.composite_score, reverse=True)
        
        bought = 0
        for signal in sorted_signals:
            if bought >= available_slots:
                break
            
            if signal.symbol in self.positions:
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
            
            position = V74Position(
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
                vpd_score=signal.vpd_score,
                snr_weight=signal.snr_weight,
                current_price=buy_price,
                market_value=buy_price * shares,
                peak_price=buy_price,
                stop_loss_price=buy_price * (1 - self.stop_loss_ratio),
                trailing_stop_price=buy_price * (1 + self.trailing_stop_ratio),
            )
            
            self.positions[signal.symbol] = position
            
            trade = V74Trade(
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
    
    def _execute_sell(self, symbol: str, position: V74Position,
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
        
        trade = V74Trade(
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
                                  score_hist_data: Dict[str, Any]) -> V74BacktestResult:
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
        
        buy_trade_map: Dict[str, List[V74Trade]] = {}
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
        
        # 5. 评分分布
        stats = score_hist_data.get('statistics', {})
        score_mean = stats.get('mean', 0.0)
        score_std = stats.get('std', 0.0)
        
        # 构建结果
        result = V74BacktestResult(
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
            monthly_ic_stats=self.rank_ic_calculator.monthly_stats,
            score_mean=score_mean,
            score_std=score_std,
            score_distribution=score_hist_data,
            daily_values=daily_values_df,
            trades=self.trades,
        )
        
        return result
    
    def print_backtest_result(self, result: V74BacktestResult) -> None:
        """打印回测结果"""
        logger.info("=" * 60)
        logger.info("V74 回测结果 - 特征工程进阶与非线性因子融合")
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
        logger.info("【Rank IC 指标】（核心验收标准）")
        logger.info(f"  Mean Rank IC:      {result.mean_rank_ic:.4f}")
        logger.info(f"  月度 Rank IC 均值：  {result.monthly_rank_ic:.4f} (目标：>{V74_RANK_IC_TARGET})")
        logger.info(f"  Rank IC 达标：     {result.rank_ic_pass}")
        logger.info("-" * 40)
        logger.info("【评分分布】（验证连续性）")
        logger.info(f"  评分均值：   {result.score_mean:.2f}")
        logger.info(f"  评分标准差：{result.score_std:.2f} (目标：{V74_SCORE_STD_TARGET_MIN}-{V74_SCORE_STD_TARGET_MAX})")
        
        # 检查标准差是否达标
        std_pass = V74_SCORE_STD_TARGET_MIN <= result.score_std <= V74_SCORE_STD_TARGET_MAX
        logger.info(f"  标准差达标：{std_pass}")
        logger.info("=" * 60)
    
    def generate_report(self, result: V74BacktestResult, 
                        output_path: Optional[str] = None) -> str:
        """生成回测报告"""
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"v74_backtest_report_{timestamp}.md"
        
        # 评分分布直方图数据
        hist_data = result.score_distribution.get('histogram', {})
        hist_counts = hist_data.get('counts', [])
        bin_edges = hist_data.get('bin_edges', [])
        
        # 构建直方图 ASCII
        histogram_ascii = self._generate_ascii_histogram(hist_counts, bin_edges)
        
        # 月度 IC 柱状图
        monthly_ic_chart = generate_monthly_ic_ascii_chart(
            result.monthly_ic_stats, 
            V74_RANK_IC_TARGET
        )
        
        # 检查标准差是否达标
        std_pass = V74_SCORE_STD_TARGET_MIN <= result.score_std <= V74_SCORE_STD_TARGET_MAX
        
        report_lines = [
            "# V74 回测报告 - 特征工程进阶与非线性因子融合",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 核心算法",
            "",
            "1. **VPD 量价背离因子**: 成交量加权的一阶导数审计",
            "2. **熵权 SNR**: 信息熵 + 偏度审计资金流突发性",
            "3. **非线性融合**: Tanh 平滑函数处理极端分值",
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
            f"| Mean Rank IC | {result.mean_rank_ic:.4f} | >0.035 | {'✓' if result.mean_rank_ic >= V74_RANK_IC_TARGET else '✗'} |",
            f"| 月度 Rank IC 均值 | {result.monthly_rank_ic:.4f} | >0.035 | {'✓' if result.rank_ic_pass else '✗'} |",
            "",
            monthly_ic_chart,
            "",
            "## 评分分布直方图（验证连续性）",
            "",
            "```",
            histogram_ascii,
            "```",
            "",
            "| 统计量 | 值 | 目标 | 达标 |",
            "|------|-----|------|------|",
            f"| 评分均值 | {result.score_mean:.2f} | - | - |",
            f"| 评分标准差 | {result.score_std:.2f} | {V74_SCORE_STD_TARGET_MIN}-{V74_SCORE_STD_TARGET_MAX} | {'✓' if std_pass else '✗'} |",
            f"| 评分最小值 | {result.score_distribution.get('statistics', {}).get('min', 0.0):.2f} | - | - |",
            f"| 评分最大值 | {result.score_distribution.get('statistics', {}).get('max', 0.0):.2f} | - | - |",
            f"| Q1 (25 分位) | {result.score_distribution.get('statistics', {}).get('q1', 0.0):.2f} | - | - |",
            f"| Q3 (75 分位) | {result.score_distribution.get('statistics', {}).get('q3', 0.0):.2f} | - | - |",
            "",
            "## 结论",
            "",
            f"1. **Rank IC 达标**: {'✓ 是' if result.rank_ic_pass else '✗ 否'}",
            f"2. **评分分布连续**: {'✓ 是' if std_pass else '✗ 否'} (标准差={result.score_std:.2f})",
            f"3. **总收益**: {result.total_return*100:.2f}%",
            f"4. **最大回撤**: {result.max_drawdown*100:.2f}%",
            "",
            "---",
            "*V74 回测报告完成*",
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

def run_v74_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     output_path: Optional[str] = None) -> V74BacktestResult:
    """
    运行 V74 回测
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        output_path: 报告输出路径
        
    Returns:
        V74BacktestResult: 回测结果
    """
    # 初始化数据库
    db = DatabaseManager()
    
    # 初始化引擎
    engine = V74BacktestEngine(db=db)
    
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
    run_v74_backtest()