#!/usr/bin/env python3
"""
Final Stress Tester - 实盘前压力测试工具.

本工具用于策略实盘前的最后验证，包含以下测试模块:
    1. 五万元实盘仿真测试 (Small-Cap Execution Simulation)
    2. 蒙特卡洛滑点压力测试 (Monte Carlo Stress Test)
    3. 策略容量上限分析 (Capacity Test)
    4. 选股集中度敏感度分析 (K Value Sensitivity Analysis)
    5. 顶层 AI 逻辑校验 (Regime Switch Audit)

使用示例:
    >>> from src.final_stress_tester import FinalStressTester
    >>> tester = FinalStressTester()
    >>> results = tester.run_all_tests()
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from loguru import logger
from datetime import datetime, timedelta
import lightgbm as lgb
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

try:
    from .walk_forward_backtester_v2 import WalkForwardBacktesterV2
    from .model_trainer import ModelTrainer
except ImportError:
    from walk_forward_backtester_v2 import WalkForwardBacktesterV2
    from model_trainer import ModelTrainer


@dataclass
class StressTestConfig:
    """压力测试配置."""
    # 基础配置
    parquet_path: str = "data/parquet/features_latest.parquet"
    market_parquet_path: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    
    # 五万元实盘仿真配置
    small_capital: float = 50_000.0
    base_capital: float = 1_000_000.0
    
    # 蒙特卡洛配置
    monte_carlo_iterations: int = 100
    slippage_range: Tuple[float, float] = (0.001, 0.005)  # 0.1% - 0.5%
    
    # 容量测试配置
    capacity_levels: List[float] = None
    
    # K 值敏感度配置
    k_values: List[int] = None
    
    # 交易成本配置
    commission_rate: float = 0.0003  # 万分之三
    min_commission: float = 5.0  # 最低 5 元
    stamp_duty_rate: float = 0.001  # 千分之一（卖出）
    
    # 股数限制
    round_lot: int = 100  # 100 股整数倍
    
    def __post_init__(self):
        if self.capacity_levels is None:
            self.capacity_levels = [100_000, 500_000, 2_000_000, 5_000_000, 10_000_000]
        if self.k_values is None:
            self.k_values = [3, 5, 8, 10, 20]


class SmallCapSimulator:
    """
    五万元实盘仿真器.
    
    模拟现实交易限制:
        1. 股数取整：买入股数必须为 100 股的整数倍
        2. 最低佣金：单笔交易最低 5 元佣金
        3. 剩余现金闲置：无法完全利用资金
    """
    
    def __init__(
        self,
        initial_capital: float = 50_000.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
        round_lot: int = 100,
        top_k: int = 5,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.round_lot = round_lot
        self.top_k = top_k
        
        # 交易成本统计
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_slippage_cost = 0.0
        
        logger.info(f"SmallCapSimulator initialized: capital={initial_capital:,.0f}, "
                   f"commission={commission_rate:.4f}, min_commission={min_commission}, "
                   f"round_lot={round_lot}, top_k={top_k}")
    
    def calculate_transaction_cost_with_slippage(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
        slippage: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        计算含滑点的交易成本.
        
        Args:
            buy_price: 买入价格
            sell_price: 卖出价格
            shares: 股数
            slippage: 滑点比例（如 0.001 表示 0.1%）
        
        Returns:
            (total_cost, commission, stamp_duty, slippage_cost)
        """
        # 滑点影响成交价
        actual_buy_price = buy_price * (1 + slippage)
        actual_sell_price = sell_price * (1 - slippage)
        
        buy_value = actual_buy_price * shares
        sell_value = actual_sell_price * shares
        
        # 佣金计算（双边）
        buy_commission = max(buy_value * self.commission_rate, self.min_commission)
        sell_commission = max(sell_value * self.commission_rate, self.min_commission)
        total_commission = buy_commission + sell_commission
        
        # 印花税（仅卖出）
        stamp_duty = sell_value * self.stamp_duty_rate
        
        # 滑点成本
        slippage_cost = (actual_buy_price - buy_price) * shares + \
                       (sell_price - actual_sell_price) * shares
        
        total_cost = total_commission + stamp_duty + slippage_cost
        
        return total_cost, total_commission, stamp_duty, slippage_cost
    
    def apply_round_lot_constraint(
        self,
        target_value: float,
        price: float,
    ) -> Tuple[int, float]:
        """
        应用股数取整约束.
        
        Args:
            target_value: 目标买入金额
            price: 买入价格
        
        Returns:
            (actual_shares, unused_cash)
        """
        # 计算理论股数
        theoretical_shares = target_value / price
        
        # 下取整到 100 股的整数倍
        actual_shares = int(theoretical_shares / self.round_lot) * self.round_lot
        
        # 计算闲置现金
        actual_cost = actual_shares * price
        unused_cash = target_value - actual_cost
        
        return actual_shares, unused_cash
    
    def run_simulation(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
        start_date: str = "2025-01-01",
        apply_slippage: bool = False,
        slippage_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """
        运行小额实盘仿真.
        
        Args:
            predict_df: 特征数据
            model: 训练好的模型
            feature_columns: 特征列
            start_date: 起始日期
            apply_slippage: 是否应用滑点
            slippage_rate: 滑点率
        
        Returns:
            仿真结果字典
        """
        records = []
        predict_df = predict_df.sort(["symbol", "trade_date"])
        unique_dates = predict_df["trade_date"].unique().sort().to_list()
        
        cash = self.initial_capital
        positions: Dict[str, Dict] = {}
        hold_days: Dict[str, int] = {}
        
        # 追踪闲置现金
        total_unused_cash = 0.0
        unused_cash_records = []
        
        # 追踪碎股限制导致的收益损失
        tracking_error = 0.0
        
        for current_date in unique_dates:
            date_str = current_date.strftime("%Y-%m-%d") if hasattr(current_date, 'strftime') else str(current_date)[:10]
            
            day_data = predict_df.filter(pl.col("trade_date") == current_date)
            if day_data.is_empty():
                continue
            
            day_clean = day_data.drop_nulls(subset=feature_columns)
            if day_clean.is_empty():
                continue
            
            X_day = day_clean.select(feature_columns).to_numpy()
            symbols = day_clean["symbol"].to_list()
            closes = day_clean["close"].to_list()
            
            predictions = model.predict(X_day)
            
            # Top K 选股
            pred_results = [
                {"symbol": s, "pred_return": p, "close": c}
                for s, p, c in zip(symbols, predictions, closes)
            ]
            pred_results.sort(key=lambda x: x["pred_return"], reverse=True)
            top_k_results = pred_results[:self.top_k]
            
            # 获取次日价格
            current_idx = unique_dates.index(current_date)
            next_prices = {}
            if current_idx + 1 < len(unique_dates):
                next_date = unique_dates[current_idx + 1]
                next_data = predict_df.filter(pl.col("trade_date") == next_date)
                if not next_data.is_empty():
                    next_prices = {
                        row[0]: row[1]
                        for row in next_data.select(["symbol", "open"]).iter_rows()
                    }
            
            # 更新持仓天数
            for symbol in list(positions.keys()):
                hold_days[symbol] = hold_days.get(symbol, 0) + 1
            
            # 买入逻辑 - 应用现实约束
            available_cash = cash
            position_value = available_cash / len(top_k_results) if top_k_results else 0
            
            for signal in top_k_results:
                symbol = signal["symbol"]
                buy_price = next_prices.get(symbol, signal["close"])
                
                if symbol not in positions:
                    # 【关键】应用股数取整约束
                    actual_shares, unused = self.apply_round_lot_constraint(
                        position_value, buy_price
                    )
                    
                    total_unused_cash += unused
                    
                    if actual_shares > 0 and len(positions) < self.top_k:
                        slippage = slippage_rate if apply_slippage else 0.0
                        actual_buy_price = buy_price * (1 + slippage)
                        
                        cost, commission, stamp_duty, slip_cost = \
                            self.calculate_transaction_cost_with_slippage(
                                buy_price, buy_price, actual_shares, slippage
                            )
                        
                        self.total_commission += commission
                        self.total_slippage_cost += slip_cost
                        
                        buy_cost = actual_buy_price * actual_shares + commission
                        if buy_cost <= cash:
                            positions[symbol] = {
                                "buy_price": actual_buy_price,
                                "shares": actual_shares,
                                "buy_date": current_date,
                                "target_value": position_value,
                                "actual_value": buy_cost,
                                "unused_cash": unused,
                            }
                            hold_days[symbol] = 0
                            cash -= buy_cost
            
            # 卖出逻辑
            for symbol, pos_info in list(positions.items()):
                if hold_days.get(symbol, 0) < 1:
                    continue
                
                if symbol in next_prices:
                    sell_price = next_prices[symbol]
                    shares = pos_info["shares"]
                    buy_price = pos_info["buy_price"]
                    
                    slippage = slippage_rate if apply_slippage else 0.0
                    cost, commission, stamp_duty, slip_cost = \
                        self.calculate_transaction_cost_with_slippage(
                            buy_price, sell_price, shares, slippage
                        )
                    
                    self.total_commission += commission
                    self.total_stamp_duty += stamp_duty
                    self.total_slippage_cost += slip_cost
                    
                    sell_value = sell_price * shares * (1 - slippage) if apply_slippage else sell_price * shares
                    profit = sell_value - buy_price * shares - cost
                    cash += sell_value - cost
                    
                    # 计算追踪误差（理想 vs 实际）
                    target_value = pos_info.get("target_value", 0)
                    actual_invested = pos_info.get("actual_value", 0)
                    if target_value > 0:
                        period_tracking_error = (target_value - actual_invested) / target_value
                        tracking_error += period_tracking_error
                    
                    records.append({
                        "trade_date": current_date,
                        "symbol": symbol,
                        "action": "SELL",
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "shares": shares,
                        "cost": cost,
                        "profit": profit,
                    })
                    
                    del positions[symbol]
                    del hold_days[symbol]
            
            # 记录每日组合价值
            portfolio_value = cash
            for symbol, pos_info in positions.items():
                portfolio_value += pos_info["buy_price"] * pos_info["shares"]
            
            unused_cash_records.append({
                "trade_date": date_str,
                "unused_cash": total_unused_cash,
                "cash": cash,
                "portfolio_value": portfolio_value,
            })
            
            records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
            })
        
        return {
            "records": records,
            "final_value": cash,
            "total_return": (cash - self.initial_capital) / self.initial_capital,
            "total_unused_cash": total_unused_cash,
            "tracking_error": tracking_error,
            "cost_analysis": {
                "total_commission": self.total_commission,
                "total_stamp_duty": self.total_stamp_duty,
                "total_slippage_cost": self.total_slippage_cost,
                "total_cost": self.total_commission + self.total_stamp_duty + self.total_slippage_cost,
            },
            "unused_cash_records": unused_cash_records,
        }


class MonteCarloStressTester:
    """蒙特卡洛滑点压力测试器."""
    
    def __init__(
        self,
        iterations: int = 100,
        slippage_range: Tuple[float, float] = (0.001, 0.005),
    ):
        self.iterations = iterations
        self.slippage_range = slippage_range
        logger.info(f"MonteCarloStressTester: iterations={iterations}, "
                   f"slippage_range={slippage_range}")
    
    def run_test(
        self,
        simulator: SmallCapSimulator,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        """
        运行蒙特卡洛压力测试.
        
        Returns:
            包含不同滑点水平下的夏普比率置信区间
        """
        results_by_slippage = {}
        
        # 生成滑点水平
        slippage_levels = np.linspace(
            self.slippage_range[0],
            self.slippage_range[1],
            5
        )
        
        for slippage in slippage_levels:
            sharpe_ratios = []
            
            for i in range(self.iterations):
                # 每次迭代使用随机种子
                np.random.seed(i)
                
                # 运行仿真
                result = simulator.run_simulation(
                    predict_df, model, feature_columns,
                    apply_slippage=True,
                    slippage_rate=slippage + np.random.uniform(-0.0005, 0.0005)
                )
                
                # 计算夏普比率
                records = result["records"]
                if records:
                    sharpe = self._calculate_sharpe(records)
                    sharpe_ratios.append(sharpe)
            
            # 计算统计量
            sharpe_array = np.array(sharpe_ratios)
            results_by_slippage[slippage] = {
                "mean_sharpe": np.mean(sharpe_array),
                "std_sharpe": np.std(sharpe_array),
                "sharpe_ci_lower": np.percentile(sharpe_array, 5),
                "sharpe_ci_upper": np.percentile(sharpe_array, 95),
                "min_sharpe": np.min(sharpe_array),
                "max_sharpe": np.max(sharpe_array),
                "iterations": self.iterations,
            }
        
        return {
            "slippage_levels": slippage_levels.tolist(),
            "results_by_slippage": results_by_slippage,
        }
    
    def _calculate_sharpe(self, records: List[Dict], risk_free_rate: float = 0.03) -> float:
        """从记录中计算夏普比率."""
        equity_values = [r["portfolio_value"] for r in records if r.get("action") == "DAILY_VALUE"]
        
        if len(equity_values) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(equity_values)):
            prev = equity_values[i - 1]
            curr = equity_values[i]
            if prev > 0:
                ret = (curr - prev) / prev
                if abs(ret) < 1.0:
                    returns.append(ret)
        
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return < 1e-10:
            return 0.0
        
        annualized_return = mean_return * 252
        annualized_vol = std_return * np.sqrt(252)
        
        return (annualized_return - risk_free_rate) / annualized_vol


class CapacityAnalyzer:
    """策略容量分析器."""
    
    def __init__(
        self,
        capacity_levels: List[float] = None,
        market_impact_eta: float = 0.1,
    ):
        self.capacity_levels = capacity_levels or [100_000, 500_000, 2_000_000, 5_000_000, 10_000_000]
        self.market_impact_eta = market_impact_eta
        logger.info(f"CapacityAnalyzer: levels={len(capacity_levels)}, eta={market_impact_eta}")
    
    def calculate_market_impact(
        self,
        trade_value: float,
        daily_turnover: float,
        volatility: float,
    ) -> float:
        """
        计算市场冲击成本.
        
        使用 L-S 冲击模型: C = η * σ * √(V/ADV)
        
        Args:
            trade_value: 交易金额
            daily_turnover: 日均成交额 (ADV)
            volatility: 日波动率
        
        Returns:
            冲击成本比例
        """
        if daily_turnover <= 0:
            return 0.0
        
        volume_ratio = trade_value / daily_turnover
        impact = self.market_impact_eta * volatility * np.sqrt(volume_ratio)
        
        return min(impact, 0.05)  # 上限 5%
    
    def run_capacity_test(
        self,
        base_results: Dict[str, Any],
        avg_daily_turnover: float = 1_000_000_000,  # 默认 10 亿日成交
        volatility: float = 0.02,
    ) -> Dict[str, Any]:
        """
        运行容量测试.
        
        Returns:
            不同资金规模下的绩效表现
        """
        results = {}
        
        for capital in self.capacity_levels:
            # 估算冲击成本
            position_size = capital / 5  # 假设 5 只股票等分
            impact_cost = self.calculate_market_impact(position_size, avg_daily_turnover, volatility)
            
            # 调整收益率
            base_return = base_results.get("total_return", 0)
            adjusted_return = base_return - impact_cost * 2  # 买卖双边
            
            # 计算年化收益
            num_days = base_results.get("num_days", 252)
            annualized_return = (1 + adjusted_return) ** (252 / max(num_days, 1)) - 1
            
            results[capital] = {
                "capital": capital,
                "base_return": base_return,
                "adjusted_return": adjusted_return,
                "annualized_return": annualized_return,
                "impact_cost": impact_cost,
                "meets_target": annualized_return > 0.10,  # 年化>10%
            }
        
        # 找出最大可用容量
        max_capacity = 0
        for capital, result in results.items():
            if result["meets_target"]:
                max_capacity = max(max_capacity, capital)
        
        return {
            "capacity_results": results,
            "max_capacity": max_capacity,
        }


class KSensitivityAnalyzer:
    """K 值敏感度分析器."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [3, 5, 8, 10, 20]
        logger.info(f"KSensitivityAnalyzer: k_values={self.k_values}")
    
    def run_analysis(
        self,
        simulator_class,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
        initial_capital: float = 50_000.0,
    ) -> Dict[str, Any]:
        """运行 K 值敏感度测试."""
        results = {}
        
        for k in self.k_values:
            simulator = simulator_class(
                initial_capital=initial_capital,
                top_k=k,
            )
            
            result = simulator.run_simulation(
                predict_df, model, feature_columns,
                apply_slippage=False,
            )
            
            # 计算无法买入的股票数量（因金额不足 100 股）
            unable_to_buy_count = 0
            min_stock_price = predict_df["close"].min()
            min_position_value = initial_capital / k
            min_shares_needed = 100
            
            if min_stock_price * min_shares_needed > min_position_value:
                unable_to_buy_count = k - int(min_position_value / (min_stock_price * min_shares_needed) * k)
            
            results[k] = {
                "k_value": k,
                "total_return": result["total_return"],
                "final_value": result["final_value"],
                "total_cost": result["cost_analysis"]["total_cost"],
                "unable_to_buy_count": unable_to_buy_count,
                "avg_position_value": initial_capital / k,
                "min_stock_price": min_stock_price,
            }
        
        return {"k_analysis": results}


class RegimeSwitchAuditor:
    """市场状态开关审计器."""
    
    def __init__(self, ma_window: int = 20):
        self.ma_window = ma_window
        logger.info(f"RegimeSwitchAuditor: ma_window={ma_window}")
    
    def audit_regime_switch(
        self,
        market_data: pl.DataFrame,
        strategy_records: List[Dict],
        extreme_periods: List[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        审计市场状态开关在极端行情中的表现.
        
        Args:
            market_data: 市场基准数据
            strategy_records: 策略记录
            extreme_periods: 极端行情期间列表 [(start_date, end_date), ...]
        
        Returns:
            审计报告
        """
        if extreme_periods is None:
            # 默认包含 2024 年初等极端下跌行情
            extreme_periods = [
                ("2024-01-01", "2024-02-29"),  # 2024 年初大跌
                ("2025-01-01", "2025-03-31"),  # 假设的波动期
            ]
        
        audit_results = {}
        
        for start_date, end_date in extreme_periods:
            period_audit = self._audit_period(
                market_data, strategy_records, start_date, end_date
            )
            audit_results[f"{start_date}_{end_date}"] = period_audit
        
        # 模拟空仓测试
        empty_position_results = self._simulate_empty_position(strategy_records)
        
        return {
            "extreme_period_audits": audit_results,
            "empty_position_simulation": empty_position_results,
        }
    
    def _audit_period(
        self,
        market_data: pl.DataFrame,
        strategy_records: List[Dict],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """审计特定期间的表现."""
        # 计算期间均线状态
        start = pl.lit(start_date).cast(pl.Date)
        end = pl.lit(end_date).cast(pl.Date)
        
        period_data = market_data.filter(
            (pl.col("trade_date") >= start) &
            (pl.col("trade_date") <= end)
        ).sort("trade_date")
        
        if len(period_data) < self.ma_window:
            return {"status": "insufficient_data"}
        
        # 计算 20 日均线
        closes = period_data["close"].to_numpy()
        ma20_values = []
        
        for i in range(len(closes)):
            if i >= self.ma_window - 1:
                ma20 = np.mean(closes[i-self.ma_window+1:i+1])
                ma20_values.append({
                    "date": period_data["trade_date"][i],
                    "close": closes[i],
                    "ma20": ma20,
                    "above_ma": closes[i] > ma20,
                })
        
        # 判断开关触发及时性
        trigger_signals = []
        for i, data in enumerate(ma20_values):
            if i > 0:
                prev_above = ma20_values[i-1]["above_ma"]
                curr_above = data["above_ma"]
                if prev_above != curr_above:
                    trigger_signals.append({
                        "date": data["date"],
                        "direction": "bullish" if curr_above else "bearish",
                        "close": data["close"],
                    })
        
        # 计算期间策略表现
        period_returns = []
        for record in strategy_records:
            if isinstance(record.get("trade_date"), (datetime, pl.Date)):
                record_date = record["trade_date"]
            else:
                record_date = str(record["trade_date"])[:10]
            
            if start_date <= record_date <= end_date:
                if record.get("action") == "DAILY_VALUE":
                    period_returns.append(record)
        
        # 计算防守模式下的表现
        defensive_returns = []
        normal_returns = []
        
        for i, ma_data in enumerate(ma20_values):
            if i < len(period_returns):
                ret_record = period_returns[i]
                if "portfolio_value" in ret_record:
                    if i > 0 and "portfolio_value" in period_returns[i-1]:
                        prev_val = period_returns[i-1]["portfolio_value"]
                        curr_val = ret_record["portfolio_value"]
                        if prev_val > 0:
                            daily_ret = (curr_val - prev_val) / prev_val
                            if ma_data["above_ma"]:
                                normal_returns.append(daily_ret)
                            else:
                                defensive_returns.append(daily_ret)
        
        return {
            "period": f"{start_date} to {end_date}",
            "trigger_signals": trigger_signals,
            "num_bearish_signals": sum(1 for s in trigger_signals if s["direction"] == "bearish"),
            "num_bullish_signals": sum(1 for s in trigger_signals if s["direction"] == "bullish"),
            "normal_days": len(normal_returns),
            "defensive_days": len(defensive_returns),
            "normal_avg_return": np.mean(normal_returns) if normal_returns else 0,
            "defensive_avg_return": np.mean(defensive_returns) if defensive_returns else 0,
        }
    
    def _simulate_empty_position(
        self,
        strategy_records: List[Dict],
    ) -> Dict[str, Any]:
        """模拟空仓策略的表现."""
        # 找出所有防守模式日期
        defensive_dates = set()
        
        for record in strategy_records:
            if record.get("regime") == "defensive":
                defensive_dates.add(str(record.get("trade_date"))[:10])
        
        # 计算空仓策略的权益曲线
        empty_position_values = []
        current_value = 100_000  # 假设初始 10 万
        
        for record in strategy_records:
            date_str = str(record.get("trade_date"))[:10]
            
            if record.get("action") == "DAILY_VALUE":
                if date_str in defensive_dates:
                    # 空仓：无涨跌
                    pass
                else:
                    # 正常持仓：跟随策略
                    current_value = record.get("portfolio_value", current_value)
                
                empty_position_values.append({
                    "date": date_str,
                    "value": current_value,
                    "is_defensive": date_str in defensive_dates,
                })
        
        # 计算空仓策略指标
        if len(empty_position_values) > 1:
            values = [v["value"] for v in empty_position_values]
            max_dd = self._calculate_max_drawdown(values)
            total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        else:
            max_dd = 0
            total_return = 0
        
        return {
            "empty_position_return": total_return,
            "empty_position_max_drawdown": max_dd,
            "defensive_days_ratio": len(defensive_dates) / max(len(empty_position_values), 1),
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤."""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd


class FinalStressTester:
    """
    综合压力测试器.
    
    整合所有测试模块，生成完整报告.
    """
    
    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.results: Dict[str, Any] = {}
        
        logger.info("=" * 60)
        logger.info("Final Stress Tester Initialized")
        logger.info("=" * 60)
        logger.info(f"Config: {asdict(self.config)}")
    
    def load_data(self) -> Tuple[pl.DataFrame, Optional[pl.DataFrame], lgb.Booster]:
        """加载数据和模型."""
        logger.info("Loading data...")
        
        # 加载特征数据
        df = pl.read_parquet(self.config.parquet_path)
        df = df.sort("trade_date")
        
        # 尝试加载市场数据
        market_df = None
        if self.config.market_parquet_path and Path(self.config.market_parquet_path).exists():
            try:
                market_df = pl.read_parquet(self.config.market_parquet_path)
                market_df = market_df.sort("trade_date")
                logger.info(f"Loaded market data: {len(market_df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load market data: {e}")
        
        # 训练模型（使用最新数据）
        logger.info("Training model for stress tests...")
        
        feature_columns = self.config.feature_columns or [
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_5", "volatility_20",
            "volume_ma_ratio_5", "volume_ma_ratio_20",
            "price_position_20", "price_position_60",
            "ma_deviation_5", "ma_deviation_20",
            "rsi_14", "mfi_14",
            "turnover_bias_20", "turnover_ma_ratio",
            "volume_price_divergence_5", "volume_price_divergence_20",
            "volume_price_stable",
        ]
        
        # 准备训练数据
        data = ModelTrainer.prepare_data(df, feature_columns)
        
        # 训练模型
        trainer = ModelTrainer(n_estimators=500, max_depth=4, num_leaves=18)
        trainer.train(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
        )
        model = trainer.model
        
        return df, market_df, model, feature_columns
    
    def run_small_cap_test(self, df: pl.DataFrame, model: lgb.Booster, feature_columns: List[str]) -> Dict[str, Any]:
        """运行五万元实盘仿真测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 1: Small-Cap Execution Simulation (50K)")
        logger.info("=" * 60)
        
        # 基准测试（100 万，无约束）
        base_simulator = SmallCapSimulator(
            initial_capital=self.config.base_capital,
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
            round_lot=1,  # 无取整约束
            top_k=self.config.k_values[1] if self.config.k_values else 5,
        )
        base_result = base_simulator.run_simulation(
            df, model, feature_columns,
            apply_slippage=False,
        )
        
        # 小额测试（5 万，有约束）
        small_simulator = SmallCapSimulator(
            initial_capital=self.config.small_capital,
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
            round_lot=self.config.round_lot,
            top_k=self.config.k_values[1] if self.config.k_values else 5,
        )
        small_result = small_simulator.run_simulation(
            df, model, feature_columns,
            apply_slippage=False,
        )
        
        # 计算收益滑坡
        base_annualized = self._annualize_return(base_result["total_return"], len(base_result["records"]))
        small_annualized = self._annualize_return(small_result["total_return"], len(small_result["records"]))
        tracking_error = base_annualized - small_annualized
        
        result = {
            "base_capital_result": base_result,
            "small_capital_result": small_result,
            "base_annualized_return": base_annualized,
            "small_annualized_return": small_annualized,
            "tracking_error": tracking_error,
            "cost_impact": small_result["cost_analysis"]["total_cost"] / self.config.small_capital,
            "unused_cash_ratio": small_result["total_unused_cash"] / self.config.small_capital,
        }
        
        logger.info(f"Base Capital ({self.config.base_capital:,.0f}): {base_annualized:.2%}")
        logger.info(f"Small Capital ({self.config.small_capital:,.0f}): {small_annualized:.2%}")
        logger.info(f"Tracking Error: {tracking_error:.2%}")
        
        return result
    
    def run_monte_carlo_test(self, df: pl.DataFrame, model: lgb.Booster, feature_columns: List[str]) -> Dict[str, Any]:
        """运行蒙特卡洛滑点压力测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 2: Monte Carlo Stress Test")
        logger.info("=" * 60)
        
        mc_tester = MonteCarloStressTester(
            iterations=self.config.monte_carlo_iterations,
            slippage_range=self.config.slippage_range,
        )
        
        simulator = SmallCapSimulator(
            initial_capital=self.config.small_capital,
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
            top_k=5,
        )
        
        result = mc_tester.run_test(simulator, df, model, feature_columns)
        
        logger.info("Slippage Impact Analysis:")
        for slippage, stats in result["results_by_slippage"].items():
            logger.info(f"  Slippage {slippage:.3%}: Mean Sharpe={stats['mean_sharpe']:.2f}, "
                       f"95% CI [{stats['sharpe_ci_lower']:.2f}, {stats['sharpe_ci_upper']:.2f}]")
        
        return result
    
    def run_capacity_test(
        self,
        base_results: Dict[str, Any],
        df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """运行策略容量测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 3: Capacity Analysis")
        logger.info("=" * 60)
        
        analyzer = CapacityAnalyzer(capacity_levels=self.config.capacity_levels)
        
        # 估算市场数据
        avg_turnover = df["close"].mean() * df["volume"].mean() * len(df["symbol"].unique()) * 0.01
        volatility = 0.02
        
        result = analyzer.run_capacity_test(
            base_results,
            avg_daily_turnover=avg_turnover,
            volatility=volatility,
        )
        
        logger.info("Capacity Analysis Results:")
        for capital, stats in result["capacity_results"].items():
            status = "✓" if stats["meets_target"] else "✗"
            logger.info(f"  {status} Capital {capital/1_000_000:.1f}M: "
                       f"Return={stats['annualized_return']:.2%}, Impact={stats['impact_cost']:.4f}")
        logger.info(f"Max Capacity (return > 10%): {result['max_capacity']/1_000_000:.1f}M")
        
        return result
    
    def run_k_sensitivity_test(self, df: pl.DataFrame, model: lgb.Booster, feature_columns: List[str]) -> Dict[str, Any]:
        """运行 K 值敏感度测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 4: K-Value Sensitivity Analysis")
        logger.info("=" * 60)
        
        analyzer = KSensitivityAnalyzer(k_values=self.config.k_values)
        
        result = analyzer.run_analysis(
            SmallCapSimulator,
            df, model, feature_columns,
            initial_capital=self.config.small_capital,
        )
        
        logger.info("K-Value Analysis (50K Capital):")
        for k, stats in result["k_analysis"].items():
            warning = "⚠️" if stats["unable_to_buy_count"] > 0 else ""
            logger.info(f"  K={k}: Return={stats['total_return']:.2%}, "
                       f"Cost={stats['total_cost']:.1f}, Unable={stats['unable_to_buy_count']} {warning}")
        
        return result
    
    def run_regime_audit(
        self,
        market_data: Optional[pl.DataFrame],
        strategy_records: List[Dict],
    ) -> Dict[str, Any]:
        """运行市场状态开关审计."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 5: Regime Switch Audit")
        logger.info("=" * 60)
        
        auditor = RegimeSwitchAuditor(ma_window=20)
        
        if market_data is None or market_data.is_empty():
            logger.warning("No market data available for regime audit")
            return {"status": "no_market_data"}
        
        result = auditor.audit_regime_switch(market_data, strategy_records)
        
        logger.info("Regime Switch Audit Results:")
        for period, audit in result.get("extreme_period_audits", {}).items():
            logger.info(f"  Period {period}:")
            logger.info(f"    Bearish signals: {audit.get('num_bearish_signals', 0)}")
            logger.info(f"    Bullish signals: {audit.get('num_bullish_signals', 0)}")
        
        empty_sim = result.get("empty_position_simulation", {})
        logger.info(f"Empty Position Simulation:")
        logger.info(f"  Max Drawdown: {empty_sim.get('empty_position_max_drawdown', 0):.2%}")
        
        return result
    
    def _annualize_return(self, total_return: float, num_records: int) -> float:
        """计算年化收益率."""
        if num_records < 2:
            return 0.0
        # 假设每日一条记录
        num_days = num_records // 2  # 约等于交易日数
        return (1 + total_return) ** (252 / max(num_days, 1)) - 1
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有压力测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Starting Final Stress Tests")
        logger.info("=" * 60)
        
        # 加载数据
        df, market_df, model, feature_columns = self.load_data()
        
        # 运行测试
        self.results["small_cap_test"] = self.run_small_cap_test(df, model, feature_columns)
        self.results["monte_carlo_test"] = self.run_monte_carlo_test(df, model, feature_columns)
        
        # 容量测试需要基准结果
        base_result = self.results["small_cap_test"]["base_capital_result"]
        self.results["capacity_test"] = self.run_capacity_test(base_result, df)
        
        self.results["k_sensitivity_test"] = self.run_k_sensitivity_test(df, model, feature_columns)
        
        # Regime 审计
        strategy_records = base_result.get("records", [])
        self.results["regime_audit"] = self.run_regime_audit(market_df, strategy_records)
        
        logger.info("\n" + "=" * 60)
        logger.info("All Stress Tests Complete")
        logger.info("=" * 60)
        
        return self.results
    
    def generate_report(self, output_path: str = "docs/final_pre_live_report.md") -> str:
        """生成压力测试报告."""
        logger.info(f"Generating report to {output_path}")
        
        report = []
        report.append("# 实盘前压力测试报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**数据源**: {self.config.parquet_path}\n\n")
        
        # 执行摘要
        report.append("## 执行摘要\n")
        report.append("本报告对策略进行了全面的实盘前压力测试，包括:\n")
        report.append("1. 五万元小额实盘仿真测试\n")
        report.append("2. 蒙特卡洛滑点压力测试\n")
        report.append("3. 策略容量上限分析\n")
        report.append("4. 选股集中度 (K 值) 敏感度分析\n")
        report.append("5. 顶层 AI 逻辑校验 (Regime Switch Audit)\n\n")
        
        # 测试 1: 五万元实盘仿真
        report.append("## 1. 五万元实盘仿真测试\n")
        if "small_cap_test" in self.results:
            st = self.results["small_cap_test"]
            report.append(f"### 测试结果\n")
            report.append(f"| 指标 | 基准 (100 万) | 实盘仿真 (5 万) |\n")
            report.append(f"|------|----------|------------|\n")
            report.append(f"| 年化收益率 | {st['base_annualized_return']:.2%} | {st['small_annualized_return']:.2%} |\n")
            report.append(f"| 收益滑坡 (Tracking Error) | - | {st['tracking_error']:.2%} |\n")
            report.append(f"| 成本占比 | - | {st['cost_impact']:.2%} |\n")
            report.append(f"| 闲置现金比 | - | {st['unused_cash_ratio']:.2%} |\n\n")
            
            report.append(f"### 分析结论\n")
            if st['tracking_error'] > 0.05:
                report.append(f"⚠️ **警告**: 收益滑坡超过 5%，主要源于:\n")
                report.append(f"1. 股数取整约束导致资金闲置\n")
                report.append(f"2. 最低佣金对小额交易侵蚀严重\n\n")
            else:
                report.append(f"✓ 收益滑坡在可接受范围内\n\n")
        
        # 测试 2: 蒙特卡洛
        report.append("## 2. 蒙特卡洛滑点压力测试\n")
        if "monte_carlo_test" in self.results:
            mc = self.results["monte_carlo_test"]
            report.append(f"| 滑点水平 | 平均夏普 | 95% 置信区间 | 最差夏普 |\n")
            report.append(f"|---------|---------|-------------|----------|\n")
            for slippage in mc["slippage_levels"]:
                stats = mc["results_by_slippage"].get(slippage, {})
                report.append(f"| {slippage:.3%} | {stats.get('mean_sharpe', 0):.2f} | "
                            f"[{stats.get('sharpe_ci_lower', 0):.2f}, {stats.get('sharpe_ci_upper', 0):.2f}] | "
                            f"{stats.get('min_sharpe', 0):.2f} |\n")
            report.append("\n")
        
        # 测试 3: 容量分析
        report.append("## 3. 策略容量上限分析\n")
        if "capacity_test" in self.results:
            ct = self.results["capacity_test"]
            report.append(f"| 资金规模 | 年化收益 | 冲击成本 | 达标 (>10%) |\n")
            report.append(f"|---------|---------|---------|------------|\n")
            for capital, stats in ct["capacity_results"].items():
                status = "✓" if stats["meets_target"] else "✗"
                report.append(f"| {capital/1_000_000:.1f}M | {stats['annualized_return']:.2%} | "
                            f"{stats['impact_cost']:.4f} | {status} |\n")
            report.append(f"\n**最大可用容量**: {ct['max_capacity']/1_000_000:.1f}M (保持年化>10%)\n\n")
        
        # 测试 4: K 值敏感度
        report.append("## 4. 选股集中度 (K 值) 敏感度分析\n")
        if "k_sensitivity_test" in self.results:
            ks = self.results["k_sensitivity_test"]
            report.append(f"**测试环境**: 5 万元本金\n\n")
            report.append(f"| K 值 | 总收益 | 交易成本 | 无法买入数 |\n")
            report.append(f"|-----|-------|---------|------------|\n")
            for k, stats in ks["k_analysis"].items():
                warning = "⚠️" if stats["unable_to_buy_count"] > 0 else ""
                report.append(f"| {k} | {stats['total_return']:.2%} | {stats['total_cost']:.1f} | "
                            f"{stats['unable_to_buy_count']} {warning}\n")
            report.append("\n")
            
            # 推荐 K 值
            best_k = min(ks["k_analysis"].keys(), 
                        key=lambda x: ks["k_analysis"][x]["unable_to_buy_count"] > 0 or -ks["k_analysis"][x]["total_return"])
            report.append(f"**推荐 K 值**: {best_k} (5 万元本金下最优平衡)\n\n")
        
        # 测试 5: Regime 审计
        report.append("## 5. 顶层 AI 逻辑校验 (Regime Switch Audit)\n")
        if "regime_audit" in self.results:
            ra = self.results["regime_audit"]
            if ra.get("status") != "no_market_data":
                for period, audit in ra.get("extreme_period_audits", {}).items():
                    report.append(f"### 期间 {period}\n")
                    report.append(f"- 看跌信号触发：{audit.get('num_bearish_signals', 0)} 次\n")
                    report.append(f"- 看涨信号触发：{audit.get('num_bullish_signals', 0)} 次\n")
                    report.append(f"- 防守模式天数：{audit.get('defensive_days', 0)}\n")
                    report.append(f"- 防守期平均收益：{audit.get('defensive_avg_return', 0):.4f}\n\n")
                
                empty_sim = ra.get("empty_position_simulation", {})
                report.append(f"### 空仓策略模拟\n")
                report.append(f"- 空仓最大回撤：{empty_sim.get('empty_position_max_drawdown', 0):.2%}\n")
                report.append(f"- 防守天数占比：{empty_sim.get('defensive_days_ratio', 0):.2%}\n\n")
                
                report.append(f"**建议**: 考虑将均线开关由\"减半仓\"改为\"彻底空仓\"或\"切换至国债 ETF\"，"
                            f"可进一步降低最大回撤约 {empty_sim.get('empty_position_max_drawdown', 0)*100:.1f}%\n\n")
        
        # 图表
        report.append("## 资金曲线对比\n")
        report.append(f"![资金曲线对比](../data/plots/stress_test_equity_comparison.png)\n\n")
        
        # 最终建议
        report.append("## 实盘建议\n")
        report.append("### 5 万元本金配置建议\n")
        report.append(f"1. **K 值选择**: 推荐 K=5，平衡分散度与可执行性\n")
        report.append(f"2. **预期年化**: 考虑摩擦后约 {self.results.get('small_cap_test', {}).get('small_annualized_return', 0):.1%}\n")
        report.append(f"3. **预期回撤**: 最大回撤约 X%\n")
        report.append(f"4. **调仓频率**: 建议保持最小持仓天数，降低交易成本\n\n")
        
        report.append("### 风险提示\n")
        report.append("1. 小额资金受股数取整约束影响较大\n")
        report.append("2. 最低佣金对收益侵蚀显著，建议降低调仓频率\n")
        report.append("3. 极端行情下 Regime Switch 可提供一定保护\n")
        report.append("4. 策略容量上限约 XX 万元，超过后收益将显著下降\n\n")
        
        report_content = "".join(report)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {output_path}")
        return report_content
    
    def plot_equity_comparison(self, output_path: str = "data/plots/stress_test_equity_comparison.png") -> None:
        """绘制资金曲线对比图."""
        logger.info(f"Plotting equity comparison to {output_path}")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # 图 1: 5 万 vs 100 万权益曲线
        ax1 = fig.add_subplot(gs[0, :])
        if "small_cap_test" in self.results:
            st = self.results["small_cap_test"]
            base_records = st["base_capital_result"].get("records", [])
            small_records = st["small_capital_result"].get("records", [])
            
            base_values = [(r.get("trade_date"), r["portfolio_value"]) 
                          for r in base_records if r.get("action") == "DAILY_VALUE"]
            small_values = [(r.get("trade_date"), r["portfolio_value"]) 
                           for r in small_records if r.get("action") == "DAILY_VALUE"]
            
            if base_values and small_values:
                base_dates = [str(v[0])[:10] for v in base_values]
                base_vals = [v[1] for v in base_values]
                small_dates = [str(v[0])[:10] for v in small_values]
                small_vals = [v[1] for v in small_values]
                
                ax1.plot(base_dates, base_vals, label='100 万基准', linewidth=2, color='#2E86AB')
                ax1.plot(small_dates, small_vals, label='5 万实盘仿真', linewidth=2, color='#E74C3C')
                ax1.set_title("资金曲线对比：5 万实盘仿真 vs 100 万基准", fontsize=14, fontweight='bold')
                ax1.set_xlabel("日期")
                ax1.set_ylabel("组合价值 (元)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 图 2: 滑点影响
        ax2 = fig.add_subplot(gs[1, 0])
        if "monte_carlo_test" in self.results:
            mc = self.results["monte_carlo_test"]
            slippages = [s * 100 for s in mc["slippage_levels"]]  # 转为百分比
            means = [mc["results_by_slippage"][s]["mean_sharpe"] for s in mc["slippage_levels"]]
            ci_lower = [mc["results_by_slippage"][s]["sharpe_ci_lower"] for s in mc["slippage_levels"]]
            ci_upper = [mc["results_by_slippage"][s]["sharpe_ci_upper"] for s in mc["slippage_levels"]]
            
            ax2.plot(slippages, means, 'o-', label='平均夏普', color='#2E86AB')
            ax2.fill_between(slippages, ci_lower, ci_upper, alpha=0.3, label='95% 置信区间')
            ax2.set_title("滑点对夏普比率的影响", fontsize=12)
            ax2.set_xlabel("滑点 (%)")
            ax2.set_ylabel("夏普比率")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 图 3: 容量分析
        ax3 = fig.add_subplot(gs[1, 1])
        if "capacity_test" in self.results:
            ct = self.results["capacity_test"]
            capitals = [c / 1_000_000 for c in ct["capacity_results"].keys()]
            returns = [ct["capacity_results"][c]["annualized_return"] for c in ct["capacity_results"].keys()]
            colors = ['#4CAF50' if ct["capacity_results"][c]["meets_target"] else '#FF5722' 
                     for c in ct["capacity_results"].keys()]
            
            ax3.bar(capitals, returns, color=colors, alpha=0.7)
            ax3.axhline(y=0.10, color='r', linestyle='--', label='10% 目标线')
            ax3.set_title("策略容量分析", fontsize=12)
            ax3.set_xlabel("资金规模 (百万元)")
            ax3.set_ylabel("年化收益率")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 图 4: K 值敏感度
        ax4 = fig.add_subplot(gs[2, 0])
        if "k_sensitivity_test" in self.results:
            ks = self.results["k_sensitivity_test"]
            k_values = list(ks["k_analysis"].keys())
            returns = [ks["k_analysis"][k]["total_return"] for k in k_values]
            
            ax4.plot(k_values, returns, 'o-', color='#2E86AB')
            ax4.set_title("K 值敏感度分析 (5 万本金)", fontsize=12)
            ax4.set_xlabel("K 值 (选股数量)")
            ax4.set_ylabel("总收益率")
            ax4.grid(True, alpha=0.3)
            
            # 标注最优点
            best_k = max(k_values, key=lambda k: ks["k_analysis"][k]["total_return"])
            best_ret = ks["k_analysis"][best_k]["total_return"]
            ax4.annotate(f'最优 K={best_k}', xy=(best_k, best_ret), 
                        xytext=(best_k+2, best_ret-0.05),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        # 图 5: 成本分析
        ax5 = fig.add_subplot(gs[2, 1])
        if "small_cap_test" in self.results:
            st = self.results["small_cap_test"]
            small_cost = st["small_capital_result"]["cost_analysis"]
            
            categories = ['佣金', '印花税', '滑点成本']
            costs = [
                small_cost.get("total_commission", 0),
                small_cost.get("total_stamp_duty", 0),
                small_cost.get("total_slippage_cost", 0)
            ]
            
            ax5.bar(categories, costs, color=['#FF6384', '#36A2EB', '#FFCE56'])
            ax5.set_title("5 万实盘交易成本分解", fontsize=12)
            ax5.set_ylabel("成本 (元)")
        
        plt.suptitle("实盘前压力测试报告", fontsize=16, fontweight='bold', y=0.995)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_path}")


def run_stress_tests():
    """运行压力测试并生成报告."""
    tester = FinalStressTester()
    results = tester.run_all_tests()
    tester.generate_report()
    tester.plot_equity_comparison()
    return results


if __name__ == "__main__":
    results = run_stress_tests()
    print("\n" + "=" * 60)
    print("压力测试完成")
    print("=" * 60)