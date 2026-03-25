"""
Visualizer Module - Performance visualization for backtest results.

This module provides a Visualizer class for:
- Calculating performance metrics
- Generating equity curve plots
- Saving visualization to files

核心功能:
    - 计算核心绩效指标（年化收益、最大回撤、夏普比率）
    - 生成资金曲线图
    - 生成收益分布图
    - 保存可视化结果

注意:
    - 使用 matplotlib.use('Agg') 确保无 GUI 环境也能保存图片
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无 GUI 环境

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from loguru import logger

# 设置中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    回测结果可视化器。
    
    功能特性:
        - 计算核心绩效指标
        - 生成资金曲线图
        - 生成收益分布图
        - 生成月度热力图
        - 保存可视化结果到文件
    
    使用示例:
        >>> viz = Visualizer()
        >>> viz.plot_equity_curve(equity_curve, save_path="data/plots/backtest_result.png")
    """
    
    def __init__(
        self,
        figsize: tuple[int, int] = (14, 10),
        dpi: int = 100,
        style: str = "seaborn-v0_8",
    ) -> None:
        """
        初始化可视化器。
        
        Args:
            figsize (tuple[int, int]): 图形尺寸，默认 (14, 10)
            dpi (int): 图形分辨率，默认 100
            style (str): matplotlib 样式，默认 "seaborn-v0_8"
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # 尝试设置样式
        try:
            plt.style.use(style)
        except OSError:
            # 如果样式不可用，使用默认样式
            pass
        
        logger.info(f"Visualizer initialized with figsize={figsize}, dpi={dpi}")
    
    def calculate_metrics(
        self,
        equity_curve: pl.DataFrame,
        initial_capital: float = 1_000_000.0,
        risk_free_rate: float = 0.03,
        benchmark_returns: Optional[list[float]] = None,
        trade_records: Optional[pl.DataFrame] = None,
    ) -> dict[str, float]:
        """
        计算核心绩效指标。
        
        Args:
            equity_curve (pl.DataFrame): 资金曲线数据
                必须包含列：trade_date, portfolio_value
            initial_capital (float): 初始资金，默认 100 万
            risk_free_rate (float): 无风险利率，默认 3%
            benchmark_returns (list[float], optional): 基准收益率序列（如沪深 300）
            trade_records (pl.DataFrame, optional): 交易记录（用于计算胜率、盈亏比）
            
        Returns:
            dict[str, float]: 绩效指标字典
                - annualized_return: 年化收益率
                - max_drawdown: 最大回撤
                - sharpe_ratio: 夏普比率
                - total_return: 总收益率
                - win_rate: 胜率
                - profit_factor: 盈亏比
                - calmar_ratio: 卡玛比率
                - volatility: 年化波动率
                - alpha: 相对基准的超额收益
                - information_ratio: 信息比率
                - avg_daily_turnover: 日均换手率
        """
        if equity_curve.is_empty():
            return self._empty_metrics()
        
        # 按日期排序
        equity_curve = equity_curve.sort("trade_date")
        
        # 获取组合价值序列
        portfolio_values = equity_curve["portfolio_value"].to_list()
        dates = equity_curve["trade_date"].to_list()
        
        if len(portfolio_values) < 2:
            return self._empty_metrics()
        
        # 计算收益率序列
        returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i - 1] > 0:
                daily_return = portfolio_values[i] / portfolio_values[i - 1] - 1
                returns.append(daily_return)
        
        if not returns:
            return self._empty_metrics()
        
        returns_array = np.array(returns)
        num_days = len(dates)
        
        # 总收益率
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        
        # 年化收益率
        annualized_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1
        
        # 最大回撤
        peak = portfolio_values[0]
        max_drawdown = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 年化波动率
        volatility = np.std(returns_array, ddof=1) * np.sqrt(252)
        
        # 夏普比率
        mean_return = np.mean(returns_array)
        if volatility > 0:
            sharpe_ratio = (mean_return * 252 - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # 卡玛比率 (年化收益 / 最大回撤)
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = float('inf') if annualized_return > 0 else 0.0
        
        # 基准对比指标（如果提供了基准收益）
        alpha = 0.0
        information_ratio = 0.0
        benchmark_total_return = 0.0
        
        if benchmark_returns and len(benchmark_returns) == len(returns):
            benchmark_array = np.array(benchmark_returns)
            # 超额收益
            excess_returns = returns_array - benchmark_array
            alpha = np.mean(excess_returns) * 252  # 年化超额收益
            
            # 信息比率
            tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
            if tracking_error > 0:
                information_ratio = (np.mean(excess_returns) * 252) / tracking_error
            
            # 基准总收益
            benchmark_total_return = 1.0
            for r in benchmark_returns:
                benchmark_total_return *= (1 + r)
            benchmark_total_return -= 1
        
        # 胜率和盈亏比（从交易记录计算）
        win_rate = 0.0
        profit_factor = 0.0
        avg_daily_turnover = 0.0
        total_transaction_cost = 0.0
        
        if trade_records is not None and not trade_records.is_empty():
            trade_records = trade_records.filter(pl.col("action") == "SELL")
            if not trade_records.is_empty():
                profits = trade_records["profit"].to_list()
                costs = trade_records["cost"].to_list()
                total_transaction_cost = sum(costs)
                
                winning_trades = sum(1 for p in profits if p > 0)
                win_rate = winning_trades / len(profits) if profits else 0.0
                
                gross_profit = sum(p for p in profits if p > 0)
                gross_loss = abs(sum(p for p in profits if p < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float('inf')
                
                # 日均换手率 = 总交易成本 / 交易天数
                avg_daily_turnover = total_transaction_cost / max(num_days, 1)
        
        return {
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "volatility": volatility,
            "calmar_ratio": calmar_ratio,
            "final_value": portfolio_values[-1],
            "num_trading_days": num_days,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_daily_turnover": avg_daily_turnover,
            "alpha": alpha,
            "information_ratio": information_ratio,
            "benchmark_total_return": benchmark_total_return,
            "total_transaction_cost": total_transaction_cost,
        }
    
    def _empty_metrics(self) -> dict[str, float]:
        """返回空指标字典。"""
        return {
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
            "calmar_ratio": 0.0,
            "final_value": 0.0,
            "num_trading_days": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_daily_turnover": 0.0,
            "alpha": 0.0,
            "information_ratio": 0.0,
            "benchmark_total_return": 0.0,
            "total_transaction_cost": 0.0,
        }
    
    def plot_equity_curve(
        self,
        equity_curve: pl.DataFrame,
        initial_capital: float = 1_000_000.0,
        benchmark_values: Optional[list[float]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        绘制资金曲线图（支持基准对比）。
        
        Args:
            equity_curve (pl.DataFrame): 资金曲线数据
                必须包含列：trade_date, portfolio_value
            initial_capital (float): 初始资金，默认 100 万
            benchmark_values (list[float], optional): 基准价值序列（如沪深 300）
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形，默认 False
            
        Returns:
            plt.Figure: 生成的图形对象
        """
        if equity_curve.is_empty():
            logger.warning("Empty equity curve data")
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=20)
            ax.set_title("Equity Curve - No Data Available")
            return fig
        
        # 按日期排序
        equity_curve = equity_curve.sort("trade_date")
        
        # 转换日期为字符串（用于绘图）
        dates = [str(d) for d in equity_curve["trade_date"].to_list()]
        portfolio_values = equity_curve["portfolio_value"].to_list()
        
        # 计算累计收益率
        cumulative_returns = [(v - initial_capital) / initial_capital * 100 
                              for v in portfolio_values]
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle("Backtest Performance - Equity Curve", fontsize=16)
        
        # 上图：资金曲线（带基准对比）
        ax1 = axes[0]
        ax1.plot(dates, portfolio_values, linewidth=1.5, color='#2E86AB', label='Strategy')
        ax1.fill_between(range(len(dates)), initial_capital, portfolio_values, 
                        alpha=0.3, color='#2E86AB')
        
        # 绘制基准线（如果提供）
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            ax1.plot(dates, benchmark_values, linewidth=1.5, color='#FF6B6B', linestyle='--', label='Benchmark')
            ax1.legend(loc='upper left')
        
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.set_title(f"Portfolio Value Over Time (Initial: ${initial_capital:,.0f})", fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 旋转 x 轴标签
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 下图：累计收益率
        ax2 = axes[1]
        ax2.plot(dates, cumulative_returns, linewidth=1.5, color='#A23B72')
        ax2.fill_between(range(len(dates)), 0, cumulative_returns, 
                        alpha=0.3, color='#A23B72')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_title("Cumulative Return Over Time", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 旋转 x 轴标签
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加绩效指标文本框
        metrics = self.calculate_metrics(equity_curve, initial_capital)
        textstr = (
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Final Value: ${metrics['final_value']:,.0f}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_drawdown_curve(
        self,
        equity_curve: pl.DataFrame,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        绘制回撤曲线图。
        
        Args:
            equity_curve (pl.DataFrame): 资金曲线数据
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            plt.Figure: 生成的图形对象
        """
        if equity_curve.is_empty():
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=20)
            return fig
        
        equity_curve = equity_curve.sort("trade_date")
        
        portfolio_values = equity_curve["portfolio_value"].to_list()
        dates = [str(d) for d in equity_curve["trade_date"].to_list()]
        
        # 计算回撤序列
        drawdowns = []
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)
        
        ax.fill_between(range(len(dates)), 0, drawdowns, alpha=0.5, color='#F18F01')
        ax.plot(dates, drawdowns, linewidth=1.5, color='#F18F01')
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title("Drawdown Over Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加最大回撤标注
        max_dd = max(drawdowns)
        max_dd_idx = drawdowns.index(max_dd)
        ax.annotate(
            f'Max DD: {max_dd:.2f}%',
            xy=(max_dd_idx, max_dd),
            xytext=(max_dd_idx - 20, max_dd + 2),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10,
            color='red',
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Drawdown curve saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_returns_distribution(
        self,
        trade_records: pl.DataFrame,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        绘制收益分布图。
        
        Args:
            trade_records (pl.DataFrame): 交易记录
                必须包含列：profit
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            plt.Figure: 生成的图形对象
        """
        if trade_records.is_empty():
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, "No Trade Data", ha='center', va='center', fontsize=20)
            return fig
        
        # 过滤卖出交易
        sells = trade_records.filter(pl.col("action") == "SELL")
        if sells.is_empty():
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, "No Sell Trades", ha='center', va='center', fontsize=20)
            return fig
        
        profits = sells["profit"].to_list()
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle("Trade Returns Distribution", fontsize=16)
        
        # 左图：收益直方图
        ax1 = axes[0]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p <= 0]
        
        ax1.hist(winning, bins=30, alpha=0.7, color='green', label=f'Winning ({len(winning)})')
        ax1.hist(losing, bins=30, alpha=0.7, color='red', label=f'Losing ({len(losing)})')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel("Profit/Loss ($)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Profit/Loss Distribution", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：累计收益曲线
        ax2 = axes[1]
        cumulative = []
        running_sum = 0
        for p in profits:
            running_sum += p
            cumulative.append(running_sum)
        
        ax2.plot(range(len(cumulative)), cumulative, linewidth=1.5, color='#2E86AB')
        ax2.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3, color='#2E86AB')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax2.set_xlabel("Trade Number", fontsize=12)
        ax2.set_ylabel("Cumulative Profit ($)", fontsize=12)
        ax2.set_title("Cumulative Profit Over Trades", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Returns distribution saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_report(
        self,
        equity_curve: pl.DataFrame,
        trade_records: Optional[pl.DataFrame] = None,
        initial_capital: float = 1_000_000.0,
        save_dir: str = "data/plots",
    ) -> dict[str, Any]:
        """
        生成完整的回测报告，包括所有图表和指标。
        
        Args:
            equity_curve (pl.DataFrame): 资金曲线数据
            trade_records (pl.DataFrame, optional): 交易记录
            initial_capital (float): 初始资金
            save_dir (str): 保存目录
            
        Returns:
            dict[str, Any]: 报告数据，包含:
                - metrics: 绩效指标
                - plot_paths: 保存的图表路径
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("Generating Backtest Report")
        logger.info("=" * 50)
        
        # 计算指标（传入交易记录以计算胜率、盈亏比、换手率）
        metrics = self.calculate_metrics(equity_curve, initial_capital, trade_records=trade_records)
        
        # 打印指标
        logger.info("Performance Metrics:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        logger.info(f"  Volatility: {metrics['volatility']:.2%}")
        logger.info(f"  Final Value: ${metrics['final_value']:,.0f}")
        logger.info(f"  Trading Days: {metrics['num_trading_days']}")
        
        plot_paths = {}
        
        # 生成资金曲线图
        equity_path = str(save_dir / "equity_curve.png")
        self.plot_equity_curve(equity_curve, initial_capital, save_path=equity_path)
        plot_paths["equity_curve"] = equity_path
        
        # 生成回撤曲线图
        drawdown_path = str(save_dir / "drawdown_curve.png")
        self.plot_drawdown_curve(equity_curve, save_path=drawdown_path)
        plot_paths["drawdown_curve"] = drawdown_path
        
        # 生成收益分布图（如果有交易记录）
        if trade_records is not None and not trade_records.is_empty():
            returns_path = str(save_dir / "returns_distribution.png")
            self.plot_returns_distribution(trade_records, save_path=returns_path)
            plot_paths["returns_distribution"] = returns_path
        
        # 生成综合报告图
        report_path = str(save_dir / "backtest_result.png")
        self._generate_summary_plot(equity_curve, metrics, save_path=report_path)
        plot_paths["summary"] = report_path
        
        logger.info(f"Report saved to {save_dir}")
        
        return {
            "metrics": metrics,
            "plot_paths": plot_paths,
        }
    
    def _generate_summary_plot(
        self,
        equity_curve: pl.DataFrame,
        metrics: dict[str, float],
        save_path: str,
    ) -> plt.Figure:
        """
        生成综合报告图。
        
        Args:
            equity_curve (pl.DataFrame): 资金曲线数据
            metrics (dict[str, float]): 绩效指标
            save_path (str): 保存路径
            
        Returns:
            plt.Figure: 生成的图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle("Backtest Performance Summary Report", fontsize=18, fontweight='bold')
        
        # 左上：资金曲线
        ax1 = axes[0, 0]
        if not equity_curve.is_empty():
            equity_curve = equity_curve.sort("trade_date")
            dates = [str(d) for d in equity_curve["trade_date"].to_list()]
            values = equity_curve["portfolio_value"].to_list()
            ax1.plot(dates, values, linewidth=1.5, color='#2E86AB')
            ax1.fill_between(range(len(dates)), values[0], values, alpha=0.3)
            ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
            ax1.set_title("Equity Curve", fontsize=13)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax1.grid(True, alpha=0.3)
        
        # 右上：绩效指标
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        # 构建绩效指标文本，包含胜率和盈亏比
        metrics_text = (
            f"Performance Metrics\n"
            f"{'='*40}\n"
            f"Total Return:      {metrics['total_return']:>12.2%}\n"
            f"Annualized Return: {metrics['annualized_return']:>12.2%}\n"
            f"Max Drawdown:      {metrics['max_drawdown']:>12.2%}\n"
            f"Sharpe Ratio:      {metrics['sharpe_ratio']:>12.2f}\n"
            f"Calmar Ratio:      {metrics['calmar_ratio']:>12.2f}\n"
            f"Volatility:        {metrics['volatility']:>12.2%}\n"
            f"Final Value:       ${metrics['final_value']:>11,.0f}\n"
            f"Trading Days:      {metrics['num_trading_days']:>12d}\n"
        )
        
        # 添加胜率和盈亏比（如果可用）
        if metrics.get('win_rate', 0) > 0 or metrics.get('profit_factor', 0) > 0:
            metrics_text += (
                f"\n"
                f"Win Rate:          {metrics.get('win_rate', 0):>12.1%}\n"
                f"Profit Factor:     {metrics.get('profit_factor', 0):>12.2f}\n"
                f"Avg Daily Turnover: ${metrics.get('avg_daily_turnover', 0):>10,.0f}\n"
            )
        ax2.text(0.1, 0.7, metrics_text, fontsize=12, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # 左下：回撤曲线
        ax3 = axes[1, 0]
        if not equity_curve.is_empty():
            equity_curve_sorted = equity_curve.sort("trade_date")
            values = equity_curve_sorted["portfolio_value"].to_list()
            dates = [str(d) for d in equity_curve_sorted["trade_date"].to_list()]
            
            drawdowns = []
            peak = values[0]
            for v in values:
                if v > peak:
                    peak = v
                drawdowns.append((peak - v) / peak * 100)
            
            ax3.fill_between(range(len(dates)), 0, drawdowns, alpha=0.5, color='#F18F01')
            ax3.plot(dates, drawdowns, linewidth=1, color='#F18F01')
            ax3.set_ylabel("Drawdown (%)", fontsize=11)
            ax3.set_xlabel("Date", fontsize=11)
            ax3.set_title("Drawdown Curve", fontsize=13)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 右下：收益分布（需要交易数据，这里简化显示）
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 计算一些统计信息
        if metrics['num_trading_days'] > 0:
            daily_returns = []
            if not equity_curve.is_empty():
                values = equity_curve.sort("trade_date")["portfolio_value"].to_list()
                for i in range(1, len(values)):
                    if values[i-1] > 0:
                        daily_returns.append((values[i] - values[i-1]) / values[i-1] * 100)
            
            if daily_returns:
                avg_daily = np.mean(daily_returns)
                best_day = max(daily_returns)
                worst_day = min(daily_returns)
                positive_days = sum(1 for r in daily_returns if r > 0)
                win_rate = positive_days / len(daily_returns) * 100
                
                stats_text = (
                    f"Daily Return Statistics\n"
                    f"{'='*40}\n"
                    f"Average Daily Return: {avg_daily:>10.3f}%\n"
                    f"Best Day:             {best_day:>10.3f}%\n"
                    f"Worst Day:            {worst_day:>10.3f}%\n"
                    f"Positive Days:        {positive_days:>10d}\n"
                    f"Daily Win Rate:       {win_rate:>10.1f}%\n"
                )
            else:
                stats_text = "No daily return data available"
        else:
            stats_text = "No trading data available"
        
        ax4.text(0.1, 0.7, stats_text, fontsize=12, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 保存
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Summary plot saved to {save_path}")
        
        return fig


def generate_backtest_report(
    equity_curve: pl.DataFrame,
    trade_records: Optional[pl.DataFrame] = None,
    save_dir: str = "data/plots",
) -> dict[str, Any]:
    """
    便捷函数：生成回测报告。
    
    Args:
        equity_curve (pl.DataFrame): 资金曲线数据
        trade_records (pl.DataFrame, optional): 交易记录
        save_dir (str): 保存目录
        
    Returns:
        dict[str, Any]: 报告数据
    """
    viz = Visualizer()
    return viz.generate_report(equity_curve, trade_records, save_dir=save_dir)


if __name__ == "__main__":
    # 测试可视化器
    print("Visualizer module loaded successfully")
    print("Use generate_backtest_report() to generate performance charts")