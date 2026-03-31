"""
Backtest Accounting Module - V101 Backtesting and Reporting.

负责回测、计算扣费、生成报告。
核心功能:
    - 基于 Alpha 预测评分进行回测
    - 交易成本计算（佣金、印花税、滑点）
    - 组合收益计算
    - 风险指标计算
    - 生成审计报告
"""

from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import json

import polars as pl
import numpy as np
from loguru import logger

# 内存优化
pl.Config.set_streaming_chunk_size(10000)


class BacktestAccounting:
    """
    V101 回测会计引擎。
    
    【核心功能】
    1. 基于 predict_score 生成交易信号
    2. 计算交易成本（佣金、印花税、滑点）
    3. 计算组合收益和风险指标
    4. 生成 IC 为核心的审计报告
    
    【参数说明】
    - 不在回测引擎中修改调仓频率、初始资金
    - 所有优化发生在 alpha_research.py 内部
    """
    
    # 默认交易成本参数
    DEFAULT_COMMISSION = 0.0003  # 佣金率（万分之三）
    DEFAULT_STAMP_DUTY = 0.001   # 印花税（千分之一，卖出收取）
    DEFAULT_SLIPPAGE = 0.001     # 滑点（千分之一）
    
    # 默认持仓参数
    DEFAULT_TOP_N = 50           # 默认持有股票数量
    DEFAULT_POSITION_PER_STOCK = 0.02  # 单只股票仓位（2% = 100%/50）
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = DEFAULT_COMMISSION,
        stamp_duty_rate: float = DEFAULT_STAMP_DUTY,
        slippage_rate: float = DEFAULT_SLIPPAGE,
        top_n: int = DEFAULT_TOP_N,
        output_dir: str = "reports",
    ) -> None:
        """
        初始化回测会计引擎。
        
        Args:
            initial_capital: 初始资金（默认 100 万）
            commission_rate: 佣金率
            stamp_duty_rate: 印花税率
            slippage_rate: 滑点率
            top_n: 持有股票数量
            output_dir: 报告输出目录
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_rate = slippage_rate
        self.top_n = top_n
        self.position_per_stock = 1.0 / top_n if top_n > 0 else 0.02
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("BacktestAccounting initialized")
        logger.info(f"  Initial capital: {initial_capital:,.0f}")
        logger.info(f"  Top N stocks: {top_n}")
        logger.info(f"  Commission: {commission_rate:.2%}, Stamp duty: {stamp_duty_rate:.2%}, Slippage: {slippage_rate:.2%}")
    
    def generate_signals(self, df: pl.DataFrame, score_column: str = "predict_score") -> pl.DataFrame:
        """
        生成交易信号。
        
        【信号生成逻辑】
        - 每日按 predict_score 排名，选择前 top_n 只股票
        - 信号值：1 = 买入/持有，0 = 不持有
        
        Args:
            df: 包含 predict_score 的数据
            score_column: 评分列名
            
        Returns:
            包含 signal 列的 DataFrame
        """
        if score_column not in df.columns:
            logger.error(f"Score column '{score_column}' not found")
            return df
        
        if "trade_date" not in df.columns:
            logger.error("trade_date column not found")
            return df
        
        result = df.clone()
        
        # 按日期分组，计算排名
        if "trade_date" in result.columns:
            # 计算每日排名（降序，分数越高排名越前）
            rank = pl.col(score_column).rank("dense", descending=True).over("trade_date")
            
            # 生成信号：前 top_n 名为 1，其余为 0
            signal = pl.when(rank <= self.top_n).then(1).otherwise(0)
            
            result = result.with_columns([
                signal.alias("signal"),
                rank.alias("score_rank"),
            ])
        else:
            # 无日期列，全局排名
            rank = pl.col(score_column).rank("dense", descending=True)
            signal = pl.when(rank <= self.top_n).then(1).otherwise(0)
            result = result.with_columns([
                signal.alias("signal"),
                rank.alias("score_rank"),
            ])
        
        logger.debug(f"[Generate Signals] Generated signals for {result['signal'].sum()} positions")
        return result
    
    def calculate_transaction_cost(
        self,
        buy_amount: float,
        sell_amount: float,
    ) -> dict[str, float]:
        """
        计算交易成本。
        
        Args:
            buy_amount: 买入金额
            sell_amount: 卖出金额
            
        Returns:
            交易成本明细
        """
        # 买入成本：佣金 + 滑点
        buy_cost = buy_amount * (self.commission_rate + self.slippage_rate)
        
        # 卖出成本：佣金 + 印花税 + 滑点
        sell_cost = sell_amount * (self.commission_rate + self.stamp_duty_rate + self.slippage_rate)
        
        return {
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "total_cost": buy_cost + sell_cost,
        }
    
    def run_backtest(self, df: pl.DataFrame, score_column: str = "predict_score") -> dict[str, Any]:
        """
        运行回测。
        
        【回测流程】
        1. 生成交易信号
        2. 计算每日持仓
        3. 计算调仓交易
        4. 计算交易成本
        5. 计算组合收益
        
        Args:
            df: 包含 predict_score 的数据
            score_column: 评分列名
            
        Returns:
            回测结果字典
        """
        logger.info("=" * 60)
        logger.info("V101 Backtest - Running...")
        logger.info("=" * 60)
        
        # 1. 生成交易信号
        signal_df = self.generate_signals(df, score_column)
        
        # 2. 按日期排序
        if "trade_date" not in signal_df.columns:
            logger.error("trade_date column not found")
            return {"error": "Missing trade_date column"}
        
        signal_df = signal_df.sort(["trade_date", "symbol"])
        
        # 3. 计算每日持仓收益
        unique_dates = sorted(signal_df["trade_date"].unique())
        
        portfolio_values = []
        daily_returns = []
        total_costs = []
        
        current_capital = self.initial_capital
        prev_positions = {}  # {symbol: position_value}
        
        for i, date in enumerate(unique_dates):
            day_data = signal_df.filter(pl.col("trade_date") == date)
            
            if len(day_data) < self.top_n:
                continue
            
            # 获取当日信号
            positions = day_data.filter(pl.col("signal") == 1)
            
            if len(positions) == 0:
                continue
            
            # 计算调仓
            current_positions = set(positions["symbol"].to_list())
            prev_position_set = set(prev_positions.keys())
            
            # 需要卖出的：之前持有但今日不持有
            to_sell = prev_position_set - current_positions
            # 需要买入的：今日持有但之前不持有
            to_buy = current_positions - prev_position_set
            
            # 计算交易金额
            sell_amount = sum(prev_positions.get(sym, 0) for sym in to_sell)
            buy_amount = current_capital * self.position_per_stock * len(to_buy) if to_buy else 0
            
            # 计算交易成本
            cost = self.calculate_transaction_cost(buy_amount, sell_amount)
            total_costs.append(cost)
            
            # 更新资金
            current_capital = current_capital + sell_amount - buy_amount - cost["total_cost"]
            
            # 计算当日持仓价值（使用 T+1 收益）
            if "t1_return" in day_data.columns:
                # 计算持仓加权收益
                position_returns = positions.join(
                    day_data.select(["symbol", "t1_return"]),
                    on="symbol",
                    how="left",
                )
                
                if len(position_returns) > 0:
                    weighted_return = (position_returns["t1_return"].fill_null(0) * self.position_per_stock).sum()
                    daily_profit = current_capital * weighted_return
                else:
                    daily_profit = 0
            else:
                daily_profit = 0
            
            # 更新持仓
            prev_positions = {
                sym: current_capital * self.position_per_stock
                for sym in current_positions
            }
            
            # 记录当日数据
            portfolio_value = current_capital + daily_profit
            portfolio_values.append({
                "trade_date": date,
                "portfolio_value": portfolio_value,
                "daily_return": daily_profit / current_capital if current_capital > 0 else 0,
                "num_positions": len(current_positions),
                "transaction_cost": cost["total_cost"],
            })
            
            daily_returns.append(daily_profit / current_capital if current_capital > 0 else 0)
        
        # 4. 计算回测统计
        if not portfolio_values:
            return {"error": "No portfolio values calculated"}
        
        portfolio_df = pl.DataFrame(portfolio_values)
        
        # 计算累计收益
        cumulative_returns = (1 + portfolio_df["daily_return"]).cum_prod() - 1
        
        # 计算年化收益
        num_days = len(unique_dates)
        if num_days > 0:
            total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
            annual_return = (1 + total_return) ** (252 / num_days) - 1
        else:
            annual_return = 0
            total_return = 0
        
        # 计算波动率
        daily_returns_np = np.array(daily_returns)
        volatility = np.std(daily_returns_np, ddof=1) * np.sqrt(252) if len(daily_returns_np) > 1 else 0
        
        # 计算夏普比率
        mean_daily_return = np.mean(daily_returns_np) if len(daily_returns_np) > 0 else 0
        sharpe = (mean_daily_return * 252) / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cum_values = (1 + portfolio_df["daily_return"]).cum_prod()
        running_max = cum_values.cum_max()
        drawdown = (cum_values - running_max) / running_max
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0
        
        # 计算总交易成本
        total_transaction_cost = sum(c["total_cost"] for c in total_costs)
        
        result = {
            "portfolio_df": portfolio_df,
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "total_transaction_cost": float(total_transaction_cost),
            "num_trading_days": num_days,
            "final_value": float(portfolio_df["portfolio_value"][-1]) if len(portfolio_df) > 0 else self.initial_capital,
        }
        
        logger.info(f"[Backtest] Total Return: {total_return:.2%}")
        logger.info(f"[Backtest] Annual Return: {annual_return:.2%}")
        logger.info(f"[Backtest] Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"[Backtest] Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"[Backtest] Total Transaction Cost: {total_transaction_cost:,.2f}")
        
        return result
    
    def generate_report(
        self,
        backtest_result: dict[str, Any],
        alpha_result: dict[str, Any],
        report_name: str = "v101_backtest_report",
    ) -> str:
        """
        生成回测报告。
        
        Args:
            backtest_result: 回测结果
            alpha_result: Alpha 分析结果
            report_name: 报告名称
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{report_name}_{timestamp}.md"
        
        # 提取数据
        t1_ic = alpha_result.get("t1_ic", {})
        top_factor = alpha_result.get("top_factor", {})
        passed = alpha_result.get("passed", False)
        
        total_return = backtest_result.get("total_return", 0)
        annual_return = backtest_result.get("annual_return", 0)
        sharpe = backtest_result.get("sharpe_ratio", 0)
        max_dd = backtest_result.get("max_drawdown", 0)
        total_cost = backtest_result.get("total_transaction_cost", 0)
        
        # 生成 Markdown 报告
        report_content = f"""# V101 Backtest Audit Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Alpha Prediction Metrics (核心指标)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| T+1 Rank IC (Mean) | {t1_ic.get('mean_ic', 0):.4f} | > 0.05 | {'✓' if t1_ic.get('mean_ic', 0) > 0.05 else '✗'} |
| IC IR (Stability) | {t1_ic.get('ic_ir', 0):.2f} | > 0.6 | {'✓' if t1_ic.get('ic_ir', 0) > 0.6 else '✗'} |
| Top Factor IC | {top_factor.get('ic', 0):.4f} | > 0.04 | {'✓' if abs(top_factor.get('ic', 0)) > 0.04 else '✗'} |
| Top Factor Name | {top_factor.get('factor', 'N/A')} | - | - |

**Overall Assessment**: {'PASSED ✓' if passed else 'FAILED ✗'}

---

## 2. Backtest Performance (回测表现)

| Metric | Value |
|--------|-------|
| Initial Capital | {self.initial_capital:,.0f} |
| Final Value | {backtest_result.get('final_value', 0):,.2f} |
| Total Return | {total_return:.2%} |
| Annual Return | {annual_return:.2%} |
| Sharpe Ratio | {sharpe:.2f} |
| Max Drawdown | {max_dd:.2%} |
| Volatility (Ann.) | {backtest_result.get('volatility', 0):.2%} |

---

## 3. Transaction Cost Analysis (交易成本)

| Cost Type | Rate | Total Cost |
|-----------|------|------------|
| Commission | {self.commission_rate:.2%} | - |
| Stamp Duty | {self.stamp_duty_rate:.2%} | - |
| Slippage | {self.slippage_rate:.2%} | - |
| **Total** | - | {total_cost:,.2f} |

---

## 4. Configuration (配置参数)

| Parameter | Value |
|-----------|-------|
| Top N Stocks | {self.top_n} |
| Position per Stock | {self.position_per_stock:.1%} |
| Trading Days | {backtest_result.get('num_trading_days', 0)} |

---

## 5. Factor Weights (因子权重)

| Factor | Weight |
|--------|--------|
| momentum_5 | 0.15 |
| momentum_10 | 0.10 |
| momentum_20 | 0.05 |
| volatility_5 | -0.05 |
| volatility_20 | -0.05 |
| volume_ma_ratio_5 | 0.10 |
| volume_ma_ratio_20 | 0.05 |
| volume_price_divergence_5 | 0.12 |
| volume_price_health | 0.10 |
| vcp_score | 0.12 |
| volume_entropy_20 | 0.08 |
| turnover_stable | 0.08 |
| rsi_14 | 0.05 |
| macd | 0.08 |
| macd_signal | 0.06 |

---

## 6. Conclusion (结论)

{'The V101 Alpha prediction system has PASSED all acceptance criteria. The model demonstrates strong predictive power with T+1 IC above 0.05 and stable IC IR above 0.6.' if passed else 'The V101 Alpha prediction system has NOT MET all acceptance criteria. Further optimization of the alpha research module is recommended.'}

---

*Report generated by V101 Backtest Accounting Module*
"""
        
        # 保存报告
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # 同时保存 JSON 结果
        json_result = {
            "alpha_metrics": {
                "t1_ic": t1_ic,
                "top_factor": top_factor,
                "passed": passed,
            },
            "backtest_metrics": {
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_transaction_cost": total_cost,
            },
            "config": {
                "initial_capital": self.initial_capital,
                "top_n": self.top_n,
                "commission_rate": self.commission_rate,
                "stamp_duty_rate": self.stamp_duty_rate,
                "slippage_rate": self.slippage_rate,
            },
        }
        
        json_path = self.output_dir / f"{report_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2)
        
        logger.info(f"JSON result saved to: {json_path}")
        
        return str(report_path)
    
    def run_full_audit(
        self,
        df: pl.DataFrame,
        alpha_result: dict[str, Any],
        report_name: str = "v101_full_audit",
    ) -> dict[str, Any]:
        """
        运行完整审计流程。
        
        Args:
            df: 包含 predict_score 和 t1_return 的数据
            alpha_result: Alpha 分析结果
            report_name: 报告名称
            
        Returns:
            完整审计结果
        """
        # 运行回测
        backtest_result = self.run_backtest(df, score_column="predict_score")
        
        # 生成报告
        report_path = self.generate_report(backtest_result, alpha_result, report_name)
        
        return {
            "backtest_result": backtest_result,
            "alpha_result": alpha_result,
            "report_path": report_path,
        }


def get_backtest_accounting(
    initial_capital: float = 1_000_000.0,
    top_n: int = 50,
    output_dir: str = "reports",
) -> BacktestAccounting:
    """获取 BacktestAccounting 实例。"""
    return BacktestAccounting(
        initial_capital=initial_capital,
        top_n=top_n,
        output_dir=output_dir,
    )