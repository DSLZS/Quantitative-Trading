"""
Backtest Engine Module - 统一回测框架。

核心功能:
    - 标准化的多空分析（Q1-Q5 分组）
    - 最大回撤、夏普比率计算
    - 换手率计算
    - 收益曲线生成
    - 错误分析（亏损股票特征分析）

使用示例:
    >>> from backtest_engine import BacktestEngine
    >>> engine = BacktestEngine(initial_capital=100000)
    >>> results = engine.run_backtest(signals_df, returns_df)
"""

import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


@dataclass
class BacktestResult:
    """回测结果数据结构"""
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
    
    # 多空分析
    q1_return: float = 0.0  # 第一组（最低分）收益
    q2_return: float = 0.0
    q3_return: float = 0.0
    q4_return: float = 0.0
    q5_return: float = 0.0  # 第五组（最高分）收益
    q1_q5_spread: float = 0.0  # Q5 - Q1 多空收益
    
    # 交易指标
    turnover_rate: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    
    # 错误分析
    worst_stocks: List[Dict[str, Any]] = field(default_factory=list)
    loss_attribution: Dict[str, Any] = field(default_factory=dict)
    
    # 详细数据
    daily_returns: Optional[pl.DataFrame] = None
    cumulative_returns: Optional[pl.DataFrame] = None
    q_group_returns: Optional[pl.DataFrame] = None


class BacktestEngine:
    """
    统一回测引擎 - 支持标准化的策略评估。
    
    核心特性:
        1. Q1-Q5 分组分析 - 验证因子/信号单调性
        2. 最大回撤计算 - 评估极端风险
        3. 夏普比率 - 风险调整后收益
        4. 换手率 - 交易成本估算
        5. 错误分析 - 亏损股票特征诊断
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "initial_capital": 100000,  # 初始资金 10 万（严禁私自修改）
        "risk_free_rate": 0.02,  # 无风险利率（年化 2%）
        "transaction_cost": 0.001,  # 交易成本（0.1%）
        "slippage": 0.0005,  # 滑点（0.05%）
        "holding_period": 5,  # 持仓周期（T+5）
        "top_k_stocks": 10,  # 持仓股票数量
        "quantile_groups": 5,  # Q 分组数量
    }
    
    def __init__(
        self,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.02,
        transaction_cost: float = 0.001,
        holding_period: int = 5,
        top_k_stocks: int = 10,
        db: Optional[DatabaseManager] = None,
    ):
        """
        初始化回测引擎。
        
        Args:
            initial_capital: 初始资金（默认 10 万）
            risk_free_rate: 无风险利率（年化）
            transaction_cost: 单边交易成本
            holding_period: 持仓周期
            top_k_stocks: 持仓股票数量
            db: 数据库管理器
        """
        # 验证初始资金
        if initial_capital <= 0:
            raise ValueError("初始资金必须大于 0")
        
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.slippage = transaction_cost * 0.5  # 滑点为交易成本的一半
        self.holding_period = holding_period
        self.top_k_stocks = top_k_stocks
        self.db = db or DatabaseManager()
        self.quantile_groups = 5  # Q 分组数量
        
        logger.info(f"BacktestEngine initialized: capital={initial_capital}, top_k={top_k_stocks}")
    
    def compute_q_group_returns(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
        n_groups: int = 5,
    ) -> pl.DataFrame:
        """
        计算 Q 分组收益 - 验证信号单调性。
        
        【金融逻辑】
        - 将股票按信号值分为 N 组（默认 5 组）
        - 计算每组的平均收益
        - 有效的信号应该呈现单调性：Q5 > Q4 > Q3 > Q2 > Q1
        
        Args:
            signals: 信号 DataFrame (symbol, trade_date, signal)
            returns: 收益 DataFrame (symbol, trade_date, future_return)
            n_groups: 分组数量
            
        Returns:
            包含每组收益的 DataFrame
        """
        # 合并信号和收益
        merged = signals.join(
            returns,
            on=["symbol", "trade_date"],
            how="inner"
        )
        
        if merged.is_empty():
            logger.warning("Signals and returns have no overlap")
            return pl.DataFrame()
        
        # 按日期分组，计算每天的 Q 分组
        result = merged.sort("trade_date").group_by("trade_date", maintain_order=True).agg([
            # 计算信号的分位数
            pl.col("signal").quantile(0.2).alias("q1_threshold"),
            pl.col("signal").quantile(0.4).alias("q2_threshold"),
            pl.col("signal").quantile(0.6).alias("q3_threshold"),
            pl.col("signal").quantile(0.8).alias("q4_threshold"),
        ])
        
        # 为每个股票分配 Q 组
        merged = merged.join(result, on="trade_date", how="left")
        
        # 分配 Q 组标签
        merged = merged.with_columns([
            pl.when(pl.col("signal") <= pl.col("q1_threshold"))
            .then(1)
            .when(pl.col("signal") <= pl.col("q2_threshold"))
            .then(2)
            .when(pl.col("signal") <= pl.col("q3_threshold"))
            .then(3)
            .when(pl.col("signal") <= pl.col("q4_threshold"))
            .then(4)
            .otherwise(5)
            .alias("q_group")
        ])
        
        # 计算每组的平均收益
        q_returns = merged.group_by("q_group").agg([
            pl.col("future_return").mean().alias("avg_return"),
            pl.col("future_return").std().alias("std_return"),
            pl.col("symbol").count().alias("count"),
        ]).sort("q_group")
        
        return q_returns
    
    def compute_max_drawdown(self, cumulative_returns: pl.Series) -> Tuple[float, int, int]:
        """
        计算最大回撤。
        
        Args:
            cumulative_returns: 累计收益序列
            
        Returns:
            (max_drawdown, peak_idx, trough_idx)
        """
        if len(cumulative_returns) == 0:
            return 0.0, 0, 0
        
        # 计算累积净值曲线
        nav = (1 + cumulative_returns).cum_prod()
        
        # 计算滚动最大值
        rolling_max = nav.cum_max()
        
        # 计算回撤
        drawdown = (nav - rolling_max) / rolling_max
        
        # 找到最大回撤
        max_dd = float(drawdown.min())
        trough_idx = int(drawdown.arg_min())
        
        # 找到峰值位置
        peak_idx = int(nav[:trough_idx + 1].arg_max())
        
        return abs(max_dd), peak_idx, trough_idx
    
    def compute_sharpe_ratio(
        self,
        returns: pl.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """
        计算夏普比率。
        
        Args:
            returns: 日收益序列
            risk_free_rate: 年化无风险利率
            periods_per_year: 每年周期数
            
        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0.0
        
        # 计算超额收益
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        
        # 计算均值和标准差
        mean_return = float(excess_returns.mean())
        std_return = float(excess_returns.std())
        
        if std_return < 1e-10:
            return 0.0
        
        # 年化夏普比率
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        
        return sharpe
    
    def compute_turnover_rate(
        self,
        positions: pl.DataFrame,
        prev_positions: Optional[pl.DataFrame] = None,
    ) -> float:
        """
        计算换手率。
        
        Args:
            positions: 当前持仓 DataFrame
            prev_positions: 上期持仓 DataFrame
            
        Returns:
            换手率
        """
        if prev_positions is None or prev_positions.is_empty():
            return 0.0
        
        if positions.is_empty():
            return 0.0
        
        # 计算持仓变化
        merged = positions.join(
            prev_positions,
            on="symbol",
            how="outer",
            suffix="_prev"
        ).fill_null(0)
        
        # 计算买卖金额
        buy_amount = (pl.col("weight") - pl.col("weight_prev")).clip(0, None).sum()
        sell_amount = (pl.col("weight_prev") - pl.col("weight")).clip(0, None).sum()
        
        turnover = float((buy_amount + sell_amount) / 2)
        
        return turnover
    
    def analyze_worst_stocks(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
        metadata: Optional[pl.DataFrame] = None,
        n_worst: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        错误分析 - 分析亏损最多的股票特征。
        
        【金融逻辑】
        - 找出亏损最多的 N 只股票
        - 分析其特征：行业、市值、ST 状态等
        - 判断亏损原因：行业暴跌、因子失效、个股黑天鹅
        
        Args:
            signals: 信号 DataFrame
            returns: 收益 DataFrame
            metadata: 股票元数据（行业、市值、ST 状态）
            n_worst: 分析最差的股票数量
            
        Returns:
            亏损股票特征列表
        """
        # 合并数据
        merged = signals.join(
            returns,
            on=["symbol", "trade_date"],
            how="inner"
        )
        
        if merged.is_empty():
            return []
        
        # 按股票分组，计算总收益
        stock_returns = merged.group_by("symbol").agg([
            pl.col("future_return").sum().alias("total_return"),
            pl.col("future_return").mean().alias("avg_return"),
            pl.col("future_return").std().alias("volatility"),
            pl.col("signal").mean().alias("avg_signal"),
            pl.col("trade_date").count().alias("trade_count"),
        ])
        
        # 找出最差的股票
        worst_stocks = stock_returns.sort("total_return").limit(n_worst)
        
        # 如果元数据存在，合并分析
        if metadata is not None and not metadata.is_empty():
            worst_stocks = worst_stocks.join(
                metadata,
                on="symbol",
                how="left"
            )
        
        # 转换为字典列表
        result = []
        for row in worst_stocks.iter_rows(named=True):
            stock_info = {
                "symbol": row.get("symbol", "N/A"),
                "total_return": float(row.get("total_return", 0)),
                "avg_return": float(row.get("avg_return", 0)),
                "volatility": float(row.get("volatility", 0)),
                "avg_signal": float(row.get("avg_signal", 0)),
                "trade_count": int(row.get("trade_count", 0)),
                "industry": row.get("industry_code", "N/A"),
                "market_cap": row.get("total_mv", None),
                "is_st": int(row.get("is_st", 0)),
            }
            result.append(stock_info)
        
        return result
    
    def compute_loss_attribution(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
        metadata: Optional[pl.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        亏损归因分析。
        
        Args:
            signals: 信号 DataFrame
            returns: 收益 DataFrame
            metadata: 股票元数据
            
        Returns:
            归因分析结果
        """
        merged = signals.join(returns, on=["symbol", "trade_date"], how="inner")
        
        if merged.is_empty():
            return {"error": "No data"}
        
        attribution = {}
        
        # 按行业归因
        if metadata is not None and "industry_code" in metadata.columns:
            merged = merged.join(metadata, on="symbol", how="left")
            
            industry_returns = merged.group_by("industry_code").agg([
                pl.col("future_return").sum().alias("total_return"),
                pl.col("symbol").count().alias("count"),
            ]).sort("total_return")
            
            # 找出亏损最多的行业
            worst_industries = industry_returns.filter(pl.col("total_return") < 0).sort("total_return").limit(5)
            attribution["worst_industries"] = worst_industries.to_dicts() if not worst_industries.is_empty() else []
        
        # 按 ST 状态归因
        if metadata is not None and "is_st" in metadata.columns:
            st_returns = merged.group_by("is_st").agg([
                pl.col("future_return").mean().alias("avg_return"),
                pl.col("symbol").count().alias("count"),
            ])
            attribution["st_performance"] = st_returns.to_dicts() if not st_returns.is_empty() else []
        
        # 按信号分位数归因
        merged = merged.with_columns([
            pl.col("signal").quantile(0.2).over("trade_date").alias("q1_threshold"),
            pl.col("signal").quantile(0.8).over("trade_date").alias("q5_threshold"),
        ])
        
        q1_returns = merged.filter(pl.col("signal") <= pl.col("q1_threshold"))["future_return"].mean()
        q5_returns = merged.filter(pl.col("signal") >= pl.col("q5_threshold"))["future_return"].mean()
        
        attribution["q1_avg_return"] = float(q1_returns) if q1_returns is not None else 0
        attribution["q5_avg_return"] = float(q5_returns) if q5_returns is not None else 0
        attribution["q1_q5_spread"] = attribution["q5_avg_return"] - attribution["q1_avg_return"]
        
        return attribution
    
    def run_backtest(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
        metadata: Optional[pl.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        运行完整回测。
        
        Args:
            signals: 信号 DataFrame (symbol, trade_date, signal)
            returns: 收益 DataFrame (symbol, trade_date, future_return)
            metadata: 股票元数据（可选）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            BacktestResult: 回测结果
        """
        logger.info("=" * 60)
        logger.info(f"BACKTEST START - 回测开始")
        logger.info(f"Initial Capital: {self.initial_capital:,.0f}")
        logger.info(f"Top K Stocks: {self.top_k_stocks}")
        logger.info(f"Holding Period: {self.holding_period} days")
        logger.info("=" * 60)
        
        # 日期过滤
        if start_date:
            signals = signals.filter(pl.col("trade_date") >= start_date)
            returns = returns.filter(pl.col("trade_date") >= start_date)
        if end_date:
            signals = signals.filter(pl.col("trade_date") <= end_date)
            returns = returns.filter(pl.col("trade_date") <= end_date)
        
        # 获取日期范围
        dates = signals["trade_date"].unique().sort()
        start_date = str(dates[0]) if len(dates) > 0 else "N/A"
        end_date = str(dates[-1]) if len(dates) > 0 else "N/A"
        
        # 1. Q 分组分析
        logger.info("Step 1: Computing Q-group returns...")
        q_returns = self.compute_q_group_returns(signals, returns, n_groups=self.quantile_groups)
        
        # 提取 Q 组收益
        q_stats = {}
        if not q_returns.is_empty():
            for row in q_returns.iter_rows(named=True):
                q_stats[f"q{int(row['q_group'])}_return"] = float(row["avg_return"])
        
        q1_return = q_stats.get("q1_return", 0)
        q5_return = q_stats.get("q5_return", 0)
        q1_q5_spread = q5_return - q1_return
        
        # 2. 计算总体收益
        logger.info("Step 2: Computing overall returns...")
        total_return = float(returns["future_return"].mean()) if not returns.is_empty() else 0
        
        # 3. 计算夏普比率
        logger.info("Step 3: Computing risk metrics...")
        daily_returns = returns.group_by("trade_date").agg(
            pl.col("future_return").mean().alias("daily_return")
        ).sort("trade_date")
        
        sharpe = self.compute_sharpe_ratio(daily_returns["daily_return"]) if not daily_returns.is_empty() else 0
        
        # 4. 计算最大回撤
        if not daily_returns.is_empty():
            cumulative = daily_returns["daily_return"].cum_sum()
            max_dd, _, _ = self.compute_max_drawdown(cumulative)
        else:
            max_dd = 0
        
        # 5. 错误分析
        logger.info("Step 4: Running error analysis...")
        worst_stocks = self.analyze_worst_stocks(signals, returns, metadata, n_worst=5)
        loss_attr = self.compute_loss_attribution(signals, returns, metadata)
        
        # 构建结果
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            q1_return=q1_return,
            q2_return=q_stats.get("q2_return", 0),
            q3_return=q_stats.get("q3_return", 0),
            q4_return=q_stats.get("q4_return", 0),
            q5_return=q5_return,
            q1_q5_spread=q1_q5_spread,
            worst_stocks=worst_stocks,
            loss_attribution=loss_attr,
            daily_returns=daily_returns,
        )
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("BACKTEST RESULT - 回测结果")
        logger.info("=" * 60)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Total Return: {total_return:.4f}")
        logger.info(f"Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"Max Drawdown: {max_dd:.4f}")
        logger.info("-" * 40)
        logger.info("Q-Group Analysis (单调性检验):")
        logger.info(f"  Q1 (Low Signal):  {q1_return:.4f}")
        logger.info(f"  Q2:               {result.q2_return:.4f}")
        logger.info(f"  Q3:               {result.q3_return:.4f}")
        logger.info(f"  Q4:               {result.q4_return:.4f}")
        logger.info(f"  Q5 (High Signal): {q5_return:.4f}")
        logger.info(f"  Q5-Q1 Spread:     {q1_q5_spread:.4f}")
        logger.info("-" * 40)
        logger.info("Worst 5 Stocks (错误分析):")
        for i, stock in enumerate(worst_stocks, 1):
            logger.info(f"  {i}. {stock['symbol']}: {stock['total_return']:.4f} (Industry: {stock['industry']})")
        logger.info("=" * 60)
        
        return result
    
    def generate_report(self, result: BacktestResult, output_path: Optional[str] = None) -> str:
        """
        生成回测报告。
        
        Args:
            result: 回测结果
            output_path: 输出路径
            
        Returns:
            报告内容
        """
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"backtest_report_{timestamp}.md"
        
        report_lines = [
            "# Backtest Report",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 基本信息",
            "",
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| 回测区间 | {result.start_date} 至 {result.end_date} |",
            f"| 初始资金 | {result.initial_capital:,.0f} |",
            f"| 持仓周期 | {self.holding_period} 天 |",
            f"| 持仓股票数 | {self.top_k_stocks} |",
            "",
            "## 收益指标",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总收益 | {result.total_return:.4f} |",
            f"| 夏普比率 | {result.sharpe_ratio:.3f} |",
            f"| 最大回撤 | {result.max_drawdown:.4f} |",
            "",
            "## Q 分组分析 (单调性检验)",
            "",
            "有效的信号应该呈现单调性：Q5 > Q4 > Q3 > Q2 > Q1",
            "",
            f"| 分组 | 平均收益 |",
            f"|------|----------|",
            f"| Q1 (Low) | {result.q1_return:.4f} |",
            f"| Q2 | {result.q2_return:.4f} |",
            f"| Q3 | {result.q3_return:.4f} |",
            f"| Q4 | {result.q4_return:.4f} |",
            f"| Q5 (High) | {result.q5_return:.4f} |",
            f"| **Q5-Q1 Spread** | **{result.q1_q5_spread:.4f}** |",
            "",
            "## 错误分析 (亏损股票特征)",
            "",
        ]
        
        if result.worst_stocks:
            report_lines.append("| 排名 | 股票代码 | 总收益 | 行业 | ST 状态 |")
            report_lines.append("|------|----------|--------|------|---------|")
            for i, stock in enumerate(result.worst_stocks, 1):
                st_status = "是" if stock.get("is_st", 0) else "否"
                report_lines.append(
                    f"| {i} | {stock['symbol']} | {stock['total_return']:.4f} | "
                    f"{stock.get('industry', 'N/A')} | {st_status} |"
                )
        else:
            report_lines.append("无数据")
        
        report_lines.extend([
            "",
            "## 亏损归因分析",
            "",
            "```json",
            str(result.loss_attribution),
            "```",
            "",
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {output_path}")
        
        return report_content